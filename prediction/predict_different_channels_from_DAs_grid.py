"""
cnn_importance.py

Predicts per-DA importance (patient or therapist) from dialogue act sequences
using a 1D CNN.  Each transcript is processed as one full sequence in a single
forward pass, producing one binary prediction per DA position.

Task
----
Sequence-to-sequence classification: given a full transcript of N DA tokens,
predict for each position whether that DA is important (1) or not (0).

Model architecture
------------------
  Embedding        — learnable dense vector per (da_label × speaker) token
  Optional TS      — normalised timestamp appended as extra channel
  1D Conv stack    — num_layers conv layers with ReLU + Dropout, each with
                     kernel_size and same-padding so output length = input length
  Linear → logit   — per-position binary output (one logit per DA)

The key parameter controlling how much context each prediction uses is:

    Effective receptive field = num_layers × (kernel_size - 1) + 1

So kernel_size=5 + num_layers=3 gives a receptive field of 13 DAs (±6 on each
side of each predicted position).  Increase kernel_size for wider local context;
increase num_layers for deeper context at the same per-layer cost.

Class imbalance
---------------
BCEWithLogitsLoss(pos_weight=n_neg/n_pos) up-weights false negatives on
important DAs proportionally to their rarity.
Optional --downsample_neg_rate randomly excludes a fraction of non-important
positions from the loss each epoch, further reducing negative gradient dominance.

Evaluation
----------
LOOCV: one fold per transcript.  Test always uses the full unmodified transcript.
Primary metric: F1 for the important=1 class.

Grid Search
-----------
Use --grid_search to sweep all combinations of the provided comma-separated
parameter lists.  Each combination runs the full LOOCV; results are ranked by
mean per-fold F1(important) and saved to a CSV.  The best configuration then
trains a final model on all data.

Usage
-----
# Single run (original behaviour):
python cnn_importance.py \\
    --dir /path/to/csv_dir \\
    --granularity groups \\
    --target patient \\
    --kernel_size 5 \\
    --num_layers 3 \\
    --outdir cnn_output/

# Grid search:
python cnn_importance.py \\
    --dir /path/to/csv_dir \\
    --granularity groups \\
    --target patient \\
    --grid_search \\
    --gs_kernel_sizes 3,5,7 \\
    --gs_num_layers 2,3,4 \\
    --gs_hidden_dims 64,128 \\
    --gs_dropouts 0.2,0.3 \\
    --gs_lrs 1e-3,5e-4 \\
    --gs_thresholds 0.3,0.5 \\
    --outdir cnn_output/

Requires: torch  (pip install torch)
          scikit-learn
Drop alongside analyze_da_patterns.py.
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from pathlib import Path
from dataclasses import dataclass, asdict, fields
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

from plotting.common_patterns import (
    DA_COLUMN,
    load_da_level,
    get_label,
)

# ── reproducibility ───────────────────────────────────────────────────────────

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


# ── hyperparameter config ─────────────────────────────────────────────────────

@dataclass
class HParams:
    """All tuneable hyperparameters in one place."""
    kernel_size:         int   = 5
    num_layers:          int   = 3
    embed_dim:           int   = 64
    hidden_dim:          int   = 128
    dropout:             float = 0.3
    lr:                  float = 1e-3
    epochs:              int   = 30
    threshold:           float = 0.5
    downsample_neg_rate: float = 0.2
    pos_weight_scale:    float = 1.0
    context_before:      int   = 0
    context_after:       int   = 0

    def label(self) -> str:
        return (
            f"k{self.kernel_size}_l{self.num_layers}"
            f"_h{self.hidden_dim}_d{int(self.dropout*100)}"
            f"_lr{self.lr:.0e}_e{self.epochs}"
            f"_t{int(self.threshold*100)}"
            f"_neg{int(self.downsample_neg_rate*100)}"
            f"_pw{self.pos_weight_scale:.2f}"
            f"_cb{self.context_before}_ca{self.context_after}"
        )


# ── token normalisation ───────────────────────────────────────────────────────

def _normalise_speaker(raw: str) -> str:
    if isinstance(raw, str) and raw.strip().lower() == "therapist":
        return "therapist"
    return "patient"


def _normalise_timestamp(df: pd.DataFrame) -> list[float] | None:
    if "timestamp" not in df.columns:
        return None
    ts = pd.to_numeric(df["timestamp"], errors="coerce")
    if ts.isna().all():
        return None
    ts_min, ts_max = ts.min(), ts.max()
    if ts_max == ts_min:
        return [0.5] * len(ts)
    return ((ts - ts_min) / (ts_max - ts_min)).fillna(0.5).tolist()


# ── vocabulary ────────────────────────────────────────────────────────────────

PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class Vocabulary:
    """
    Maps da_label tokens to integer indices.

    Speaker is handled as a separate explicit 0/1 input channel to the CNN
    (0 = therapist, 1 = any other speaker) rather than being folded into
    the token identity.
    """

    def __init__(self):
        self.token2idx: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2token: list[str]      = [PAD_TOKEN, UNK_TOKEN]

    def add(self, da_label: str):
        if da_label not in self.token2idx:
            self.token2idx[da_label] = len(self.idx2token)
            self.idx2token.append(da_label)

    def encode(self, da_label: str) -> int:
        return self.token2idx.get(da_label, self.token2idx[UNK_TOKEN])

    def __len__(self) -> int:
        return len(self.idx2token)


def build_vocabulary(
    transcripts: dict[str, pd.DataFrame],
    granularity: str,
) -> Vocabulary:
    vocab = Vocabulary()
    for df in transcripts.values():
        for _, row in df.iterrows():
            da = get_label(row[DA_COLUMN], row["da_group"], granularity)
            vocab.add(da)
    return vocab


# ── data preparation ──────────────────────────────────────────────────────────

def df_to_tensors(
    df:          pd.DataFrame,
    granularity: str,
    target_col:  str,
    vocab:       Vocabulary,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert one transcript DataFrame to four tensors of shape (seq_len,):
      token_ids  : LongTensor   — vocab index per DA (label only, no speaker)
      spkr_vals  : FloatTensor  — speaker channel: 0.0=therapist, 1.0=other
      ts_vals    : FloatTensor  — normalised timestamp per DA (zeros if absent)
      labels     : FloatTensor  — per-DA importance 0.0 / 1.0
    """
    ts_norm = _normalise_timestamp(df)
    has_ts  = ts_norm is not None

    token_ids, spkr_vals, ts_vals, labels = [], [], [], []

    for i, (_, row) in enumerate(df.iterrows()):
        da   = get_label(row[DA_COLUMN], row["da_group"], granularity)
        spkr = _normalise_speaker(str(row.get("speaker", "patient")))
        token_ids.append(vocab.encode(da))
        spkr_vals.append(0.0 if spkr == "therapist" else 1.0)
        ts_vals.append(float(ts_norm[i]) if has_ts else 0.0)
        labels.append(float(int(row[target_col])))

    return (
        torch.tensor(token_ids, dtype=torch.long),
        torch.tensor(spkr_vals, dtype=torch.float),
        torch.tensor(ts_vals,   dtype=torch.float),
        torch.tensor(labels,    dtype=torch.float),
    )


def make_loss_mask(
    labels:              torch.Tensor,
    downsample_neg_rate: float,
    rng:                 np.random.Generator,
) -> torch.Tensor:
    """
    Boolean mask (seq_len,) — True = include this position in the loss.

    All positive positions are always included.
    Negative positions are randomly kept at rate downsample_neg_rate.
    Rate 1.0 keeps all positions (default, no downsampling).
    """
    if downsample_neg_rate >= 1.0:
        return torch.ones(len(labels), dtype=torch.bool)

    neg_idx = (labels == 0).nonzero(as_tuple=True)[0].numpy()
    n_keep  = max(1, int(len(neg_idx) * downsample_neg_rate))
    if len(neg_idx) > n_keep:
        keep    = rng.choice(len(neg_idx), size=n_keep, replace=False)
        neg_idx = neg_idx[keep]

    mask = (labels == 1).clone()
    for idx in neg_idx:
        mask[idx] = True
    return mask


# ── 1D CNN model ──────────────────────────────────────────────────────────────

class CNNImportanceClassifier(nn.Module):
    """
    Per-position 1D CNN classifier.

    Stacks num_layers conv-ReLU-dropout blocks with same-padding so the
    output sequence length equals the input.  A final per-position linear
    layer maps each hidden vector to one logit.

    Effective receptive field per position:
        num_layers × (kernel_size - 1) + 1

    Increase kernel_size to widen local context.
    Increase num_layers to deepen it without widening per-layer.
    """

    def __init__(
        self,
        vocab_size:  int,
        embed_dim:   int   = 64,
        hidden_dim:  int   = 128,
        num_layers:  int   = 3,
        kernel_size: int   = 5,
        dropout:     float = 0.3,
        use_ts:      bool  = False,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.use_ts    = use_ts
        # +1 for explicit speaker channel (always present), +1 for timestamp if available
        in_channels    = embed_dim + 1 + (1 if use_ts else 0)
        pad            = kernel_size // 2   # preserves sequence length

        layers: list[nn.Module] = []
        for i in range(num_layers):
            layers += [
                nn.Conv1d(
                    in_channels  if i == 0 else hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=pad,
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]

        self.conv_stack = nn.Sequential(*layers)
        self.output     = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        token_ids: torch.Tensor,   # (seq_len,) or (batch, seq_len)
        spkr_vals: torch.Tensor,   # (seq_len,) or (batch, seq_len)  0=therapist 1=other
        ts_vals:   torch.Tensor,   # (seq_len,) or (batch, seq_len)
    ) -> torch.Tensor:             # (seq_len,) or (batch, seq_len)  logits

        unbatched = token_ids.dim() == 1
        if unbatched:
            token_ids = token_ids.unsqueeze(0)
            spkr_vals = spkr_vals.unsqueeze(0)
            ts_vals   = ts_vals.unsqueeze(0)

        x = self.embedding(token_ids)           # (B, L, embed_dim)
        x = torch.cat([x, spkr_vals.unsqueeze(-1)], dim=-1)  # (B, L, embed_dim+1)

        if self.use_ts:
            x = torch.cat([x, ts_vals.unsqueeze(-1)], dim=-1)

        x = x.permute(0, 2, 1)                 # (B, C, L)
        x = self.conv_stack(x)                  # (B, hidden_dim, L)
        x = x.permute(0, 2, 1)                  # (B, L, hidden_dim)
        logits = self.output(x).squeeze(-1)     # (B, L)

        return logits.squeeze(0) if unbatched else logits


# ── training helpers ──────────────────────────────────────────────────────────

def train_epoch(
    model:               CNNImportanceClassifier,
    optimizer:           torch.optim.Optimizer,
    criterion:           nn.BCEWithLogitsLoss,
    sequences:           list[tuple],
    device:              torch.device,
    rng:                 np.random.Generator,
    downsample_neg_rate: float,
    context_before:      int = 0,
    context_after:       int = 0,
) -> float:
    """
    One epoch: one forward pass per transcript (no cross-transcript batching
    since transcripts have different lengths).  Returns mean loss.

    Labels are expanded by context_before/context_after before computing the
    loss, so the model learns that the lead-up to important DAs is also
    important.  Evaluation always uses the original unmodified labels.
    """
    model.train()
    total_loss = 0.0
    order      = rng.permutation(len(sequences)).tolist()

    for idx in order:
        tok, spkr, ts, lbl, _ = sequences[idx]
        tok  = tok.to(device)
        spkr = spkr.to(device)
        ts   = ts.to(device)

        # Expand labels for training targets; original lbl is never mutated
        lbl_expanded = torch.tensor(
            expand_context(lbl.long().tolist(), context_before, context_after),
            dtype=torch.float,
        ).to(device)

        mask   = make_loss_mask(lbl_expanded.cpu(), downsample_neg_rate, rng).to(device)
        logits = model(tok, spkr, ts)
        loss   = criterion(logits[mask], lbl_expanded[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(sequences), 1)


@torch.no_grad()
def predict_transcript(
    model:     CNNImportanceClassifier,
    tok:       torch.Tensor,
    spkr:      torch.Tensor,
    ts:        torch.Tensor,
    device:    torch.device,
    threshold: float = 0.5,
) -> list[int]:
    model.eval()
    logits = model(tok.to(device), spkr.to(device), ts.to(device))
    probs  = torch.sigmoid(logits).cpu()
    return (probs >= threshold).long().tolist()


def expand_context(
    labels:         list[int],
    context_before: int,
    context_after:  int,
) -> list[int]:
    """
    Expand true-important label positions by context_before steps to the left
    and context_after steps to the right.

    This softens the evaluation so that predicted-important DAs in the lead-up
    to a truly important DA (e.g. unlabelled setup questions) are not penalised
    as false positives.  Any gap within the expansion window is filled naturally.

    Applied to training labels only — evaluation always uses the original
    unmodified annotations so results remain honest.

    Example (context_before=2, context_after=0):
      original labels: [0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
      expanded labels: [0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
                          ^^^           ^^^  window before each true-important run
    """
    if context_before == 0 and context_after == 0:
        return labels

    n      = len(labels)
    result = list(labels)

    for i, lbl in enumerate(labels):
        if lbl == 1:
            for j in range(max(0, i - context_before), i):
                result[j] = 1
            for j in range(i + 1, min(n, i + context_after + 1)):
                result[j] = 1

    return result


# ── LOOCV ─────────────────────────────────────────────────────────────────────

def run_loocv(
    transcripts:  dict[str, pd.DataFrame],
    vocab:        Vocabulary,
    granularity:  str,
    target_col:   str,
    hp:           HParams,
    device:       torch.device,
    verbose:      bool = True,
    train_final:  bool = True,
) -> tuple[dict, CNNImportanceClassifier | None]:
    """
    Full LOOCV for a given HParams configuration.

    Parameters
    ----------
    verbose     : print per-fold progress (suppress during grid search)
    train_final : train a final model on all data after LOOCV
    """
    rng     = np.random.default_rng(SEED)
    fnames  = list(transcripts.keys())
    n_folds = len(fnames)
    use_ts  = any("timestamp" in df.columns for df in transcripts.values())
    # Speaker is always an explicit input channel; use_ts covers timestamp only
    eff     = hp.num_layers * (hp.kernel_size - 1) + 1

    if verbose:
        print(f"\n{'='*60}")
        print(f"  CNN LOOCV  |  {n_folds} folds  |  target: {target_col}")
        print(f"  kernel={hp.kernel_size}  layers={hp.num_layers}  "
              f"receptive field={eff} DAs (±{eff//2})")
        print(f"  embed={hp.embed_dim}  hidden={hp.hidden_dim}  "
              f"dropout={hp.dropout}")
        print(f"  lr={hp.lr}  epochs={hp.epochs}  threshold={hp.threshold}  "
              f"neg_downsample={hp.downsample_neg_rate:.0%}")
        print(f"  device={device}")
        print(f"{'='*60}")

    # Pre-convert all transcripts to tensors
    # Each entry: (token_ids, spkr_vals, ts_vals, labels, fname)
    all_tensors: dict[str, tuple] = {
        fname: (*df_to_tensors(df, granularity, target_col, vocab), fname)
        for fname, df in transcripts.items()
    }

    all_true:    list[int] = []
    all_pred:    list[int] = []
    fold_results = []

    for fold_idx, test_fname in enumerate(fnames):
        if verbose:
            print(f"\n  Fold {fold_idx+1}/{n_folds}  —  test: {test_fname}")

        train_seqs = [v for k, v in all_tensors.items() if k != test_fname]

        n_pos = sum(int(t[3].sum()) for t in train_seqs)
        n_tot = sum(len(t[3]) for t in train_seqs)
        n_neg = n_tot - n_pos
        if verbose:
            print(f"    train: {n_tot} DAs  {n_pos} positive "
                  f"({100*n_pos/max(n_tot,1):.1f}%)")

        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1) * hp.pos_weight_scale], dtype=torch.float
        ).to(device)

        model = CNNImportanceClassifier(
            vocab_size=len(vocab),
            embed_dim=hp.embed_dim,
            hidden_dim=hp.hidden_dim,
            num_layers=hp.num_layers,
            kernel_size=hp.kernel_size,
            dropout=hp.dropout,
            use_ts=use_ts,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(hp.epochs):
            loss = train_epoch(
                model, optimizer, criterion,
                train_seqs, device, rng, hp.downsample_neg_rate,
                context_before=hp.context_before,
                context_after=hp.context_after,
            )
            if verbose and (epoch + 1) % max(1, hp.epochs // 5) == 0:
                print(f"      epoch {epoch+1}/{hp.epochs}  loss={loss:.4f}")

        tok, spkr, ts, lbl, _ = all_tensors[test_fname]
        y_pred = predict_transcript(model, tok, spkr, ts, device, hp.threshold)
        y_true = lbl.long().tolist()  # original annotations, never expanded

        all_true.extend(y_true)
        all_pred.extend(y_pred)

        n_pos_test = sum(y_true)
        n_pred_pos = sum(y_pred)
        f1_imp = f1_score(y_true, y_pred, pos_label=1,
                          average="binary", zero_division=0)
        f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_bal = f1_score(y_true, y_pred, average="macro", zero_division=0)
        prec   = precision_score(y_true, y_pred, pos_label=1,
                                 average="binary", zero_division=0)
        rec    = recall_score(y_true, y_pred, pos_label=1,
                              average="binary", zero_division=0)
        if verbose:
            print(f"    test: {len(y_true)} DAs  {n_pos_test} true pos  "
                  f"{n_pred_pos} predicted  "
                  f"F1(imp)={f1_imp:.3f}  prec={prec:.3f}  rec={rec:.3f}")

        fold_results.append({
            "fold":             fold_idx + 1,
            "transcript":       test_fname,
            "n_das":            len(y_true),
            "n_important":      n_pos_test,
            "pct_important":    round(100 * n_pos_test / max(len(y_true), 1), 1),
            "n_pred_important": n_pred_pos,
            "f1_important":     round(f1_imp, 4),
            "f1_macro":         round(f1_mac, 4),
            "f1_balanced":      round(f1_bal, 4),
            "precision_imp":    round(prec,   4),
            "recall_imp":       round(rec,    4),
        })

    if not all_true:
        return {"fold_results": fold_results}, None

    if verbose:
        print(f"\n{'─'*60}")
        print("  Aggregate classification report (all folds):")
        print(classification_report(
            all_true, all_pred,
            labels=[0, 1],
            target_names=["not_important", "important"],
            zero_division=0,
        ))

    agg_f1_imp   = f1_score(all_true, all_pred, pos_label=1,
                            average="binary", zero_division=0)
    agg_f1_mac   = f1_score(all_true, all_pred,
                            average="weighted", zero_division=0)
    # Balanced / macro F1 across both classes — comparable to Stephen's metric
    agg_f1_bal   = f1_score(all_true, all_pred,
                            average="macro", zero_division=0)
    mean_fold_f1 = float(np.mean([r["f1_important"] for r in fold_results]))
    std_fold_f1  = float(np.std( [r["f1_important"] for r in fold_results]))
    mean_fold_bal= float(np.mean([r["f1_balanced"]  for r in fold_results]))

    if verbose:
        print(f"  Pooled F1(important):   {agg_f1_imp:.4f}")
        print(f"  Pooled F1(balanced):    {agg_f1_bal:.4f}")
        print(f"  Pooled F1(weighted):    {agg_f1_mac:.4f}")
        print(f"  Per-fold F1(imp):  mean={mean_fold_f1:.4f}  std={std_fold_f1:.4f}")

    results = {
        "fold_results":              fold_results,
        "pooled_f1_imp":             round(agg_f1_imp,   4),
        "pooled_f1_balanced":        round(agg_f1_bal,   4),
        "pooled_f1_weighted":        round(agg_f1_mac,   4),
        "mean_fold_f1_imp":          round(mean_fold_f1, 4),
        "std_fold_f1_imp":           round(std_fold_f1,  4),
        "mean_fold_f1_balanced":     round(mean_fold_bal,4),
        "effective_receptive_field": eff,
    }

    final_model = None
    if train_final:
        if verbose:
            print("\n  Training final model on all transcripts …")
        n_pos_all = sum(int(t[3].sum()) for t in all_tensors.values())
        n_tot_all = sum(len(t[3]) for t in all_tensors.values())
        pos_weight_all = torch.tensor(
            [(n_tot_all - n_pos_all) / max(n_pos_all, 1) * hp.pos_weight_scale],
            dtype=torch.float,
        ).to(device)

        final_model = CNNImportanceClassifier(
            vocab_size=len(vocab), embed_dim=hp.embed_dim,
            hidden_dim=hp.hidden_dim, num_layers=hp.num_layers,
            kernel_size=hp.kernel_size, dropout=hp.dropout, use_ts=use_ts,
        ).to(device)
        final_opt       = torch.optim.Adam(final_model.parameters(), lr=hp.lr)
        final_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_all)

        for epoch in range(hp.epochs):
            loss = train_epoch(
                final_model, final_opt, final_criterion,
                list(all_tensors.values()), device, rng, hp.downsample_neg_rate,
                context_before=hp.context_before,
                context_after=hp.context_after,
            )
            if verbose and (epoch + 1) % max(1, hp.epochs // 5) == 0:
                print(f"    epoch {epoch+1}/{hp.epochs}  loss={loss:.4f}",
                      flush=True)

    return results, final_model


# ── grid search ───────────────────────────────────────────────────────────────

def _parse_list(s: str, cast) -> list:
    """Parse a comma-separated string into a typed list."""
    return [cast(v.strip()) for v in s.split(",") if v.strip()]


def build_grid(args) -> list[HParams]:
    """
    Build all HParams combinations from the --gs_* arguments.
    Falls back to the single-value --<param> arguments for any dimension
    not explicitly swept, so you only need to specify what you want to vary.
    """
    kernel_sizes         = _parse_list(args.gs_kernel_sizes,  int)
    num_layers_list      = _parse_list(args.gs_num_layers,     int)
    embed_dims           = _parse_list(args.gs_embed_dims,     int)
    hidden_dims          = _parse_list(args.gs_hidden_dims,    int)
    dropouts             = _parse_list(args.gs_dropouts,       float)
    lrs                  = _parse_list(args.gs_lrs,            float)
    epochs_list          = _parse_list(args.gs_epochs,         int)
    thresholds           = _parse_list(args.gs_thresholds,     float)
    neg_rates            = _parse_list(args.gs_neg_rates,      float)
    pw_scales            = _parse_list(args.gs_pw_scales,      float)
    ctx_befores          = _parse_list(args.gs_context_before,  int)
    ctx_afters           = _parse_list(args.gs_context_after,   int)

    combos = list(itertools.product(
        kernel_sizes, num_layers_list, embed_dims, hidden_dims,
        dropouts, lrs, epochs_list, thresholds, neg_rates, pw_scales,
        ctx_befores, ctx_afters,
    ))

    grid = [
        HParams(
            kernel_size=k, num_layers=nl, embed_dim=ed, hidden_dim=hd,
            dropout=dr, lr=lr, epochs=ep, threshold=th,
            downsample_neg_rate=neg, pos_weight_scale=pw,
            context_before=cb, context_after=ca,
        )
        for k, nl, ed, hd, dr, lr, ep, th, neg, pw, cb, ca in combos
    ]
    return grid


def run_grid_search(
    transcripts: dict[str, pd.DataFrame],
    vocab:       Vocabulary,
    granularity: str,
    target_col:  str,
    grid:        list[HParams],
    device:      torch.device,
    outdir:      str,
    label_prefix: str,
) -> HParams:
    """
    Evaluate every HParams in grid via LOOCV (no final model per combo).
    Saves a ranked CSV of all results and returns the best HParams.

    Ranking criterion: mean_fold_f1_imp  (per-fold mean is more robust
    than pooled F1 for selecting hyperparameters).
    """
    n = len(grid)
    print(f"\n{'#'*60}")
    print(f"  GRID SEARCH  —  {n} configurations")
    print(f"  Ranking by: mean per-fold F1(important)")
    print(f"{'#'*60}")

    rows: list[dict[str, Any]] = []

    for i, hp in enumerate(grid):
        eff = hp.num_layers * (hp.kernel_size - 1) + 1
        print(f"\n[{i+1}/{n}]  {hp.label()}  (receptive field={eff})")
        try:
            results, _ = run_loocv(
                transcripts=transcripts,
                vocab=vocab,
                granularity=granularity,
                target_col=target_col,
                hp=hp,
                device=device,
                verbose=False,       # suppress per-fold noise during sweep
                train_final=False,   # skip final model until best is known
            )
            row = {**asdict(hp), **{
                k: v for k, v in results.items()
                if k != "fold_results"
            }}
            print(
                f"    mean_fold_f1_imp={results['mean_fold_f1_imp']:.4f}  "
                f"±{results['std_fold_f1_imp']:.4f}  "
                f"pooled_f1_bal={results['pooled_f1_balanced']:.4f}"
            )
        except Exception as exc:
            print(f"    ERROR: {exc}")
            row = {**asdict(hp), "error": str(exc)}

        rows.append(row)

    # Sort by mean_fold_f1_imp descending, std ascending as tiebreak
    df_gs = pd.DataFrame(rows)
    if "mean_fold_f1_imp" in df_gs.columns:
        df_gs = df_gs.sort_values(
            ["mean_fold_f1_imp", "std_fold_f1_imp"],
            ascending=[False, True],
        ).reset_index(drop=True)

    gs_path = os.path.join(outdir, f"grid_search_{label_prefix}.csv")
    df_gs.to_csv(gs_path, index=False)
    print(f"\n  Grid search results saved: {gs_path}")

    # Best config
    best_row = df_gs.iloc[0]
    best_hp  = HParams(
        kernel_size=         int(best_row["kernel_size"]),
        num_layers=          int(best_row["num_layers"]),
        embed_dim=           int(best_row["embed_dim"]),
        hidden_dim=          int(best_row["hidden_dim"]),
        dropout=             float(best_row["dropout"]),
        lr=                  float(best_row["lr"]),
        epochs=              int(best_row["epochs"]),
        threshold=           float(best_row["threshold"]),
        downsample_neg_rate= float(best_row["downsample_neg_rate"]),
        pos_weight_scale=    float(best_row["pos_weight_scale"]),
        context_before=      int(best_row["context_before"]),
        context_after=       int(best_row["context_after"]),
    )

    eff_best = best_hp.num_layers * (best_hp.kernel_size - 1) + 1
    print(f"\n  Best config  (mean_fold_f1_imp="
          f"{best_row.get('mean_fold_f1_imp', '?'):.4f}):")
    print(f"    {best_hp.label()}")
    print(f"    receptive field = {eff_best} DAs")

    # Print top-5 summary
    top5 = df_gs.head(5)[
        ["kernel_size", "num_layers", "hidden_dim", "dropout",
         "lr", "threshold", "downsample_neg_rate", "pos_weight_scale",
         "mean_fold_f1_imp", "std_fold_f1_imp", "pooled_f1_balanced"]
    ].to_string(index=False)
    print(f"\n  Top 5 configurations:\n{top5}")

    return best_hp


# ── save helpers ──────────────────────────────────────────────────────────────

def save_results(
    results: dict,
    model:   CNNImportanceClassifier | None,
    outdir:  str,
    label:   str,
):
    os.makedirs(outdir, exist_ok=True)

    fold_path = os.path.join(outdir, f"cnn_{label}_fold_results.csv")
    pd.DataFrame(results["fold_results"]).to_csv(fold_path, index=False)
    print(f"\n  Saved: {fold_path}")

    agg_path = os.path.join(outdir, f"cnn_{label}_aggregate.json")
    with open(agg_path, "w", encoding="utf-8") as f:
        json.dump(
            {k: v for k, v in results.items() if k != "fold_results"},
            f, indent=2,
        )
    print(f"  Saved: {agg_path}")

    if model is not None:
        model_path = os.path.join(outdir, f"cnn_{label}_model.pt")
        torch.save(model.state_dict(), model_path)
        print(f"  Saved: {model_path}")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "1D CNN per-DA importance classifier.\n"
            "Use --grid_search to sweep hyperparameters via LOOCV."
        )
    )

    # ── data / task ──────────────────────────────────────────────────────────
    parser.add_argument("--dir",         required=True,
                        help="Directory containing transcript CSV/TSV/XLSX files.")
    parser.add_argument("--granularity", default="groups",
                        choices=["groups", "raw"])
    parser.add_argument("--target",      default="patient",
                        choices=["patient", "therapist"])

    # ── single-run hyperparameters ───────────────────────────────────────────
    parser.add_argument("--kernel_size",         type=int,   default=5)
    parser.add_argument("--num_layers",          type=int,   default=3)
    parser.add_argument("--embed_dim",           type=int,   default=64)
    parser.add_argument("--hidden_dim",          type=int,   default=128)
    parser.add_argument("--dropout",             type=float, default=0.3)
    parser.add_argument("--lr",                  type=float, default=1e-3)
    parser.add_argument("--epochs",              type=int,   default=30)
    parser.add_argument("--downsample_neg_rate", type=float, default=0.2)
    parser.add_argument("--threshold",           type=float, default=0.5)
    parser.add_argument("--pos_weight_scale",    type=float, default=1.0,
                        help="Multiplier on the n_neg/n_pos pos_weight. "
                             "<1.0 reduces false positives; >1.0 boosts recall. "
                             "(default: 1.0)")
    parser.add_argument("--context_before",      type=int,   default=0,
                        help="Expand each predicted-important position N steps "
                             "backwards at inference time. Fills any gap within "
                             "the window. (default: 0)")
    parser.add_argument("--context_after",       type=int,   default=0,
                        help="Expand each predicted-important position N steps "
                             "forwards at inference time. (default: 0)")
    
    # ── grid search ──────────────────────────────────────────────────────────
    parser.add_argument("--grid_search", action="store_true",
                        help="Run a grid search over all --gs_* parameter lists.")

    parser.add_argument("--gs_kernel_sizes",  default="11",
                        help="Comma-separated kernel sizes to sweep. (default: 5,11,15)")
    parser.add_argument("--gs_num_layers",    default="11",
                        help="Comma-separated layer counts to sweep. (default: 5,11,15)")
    parser.add_argument("--gs_embed_dims",    default="64",
                        help="Comma-separated embed dims to sweep. (default: 64)")
    parser.add_argument("--gs_hidden_dims",   default="32",
                        help="Comma-separated hidden dims to sweep. (default: 32)")
    parser.add_argument("--gs_dropouts",      default="0.3",
                        help="Comma-separated dropout rates to sweep. (default: 0.3)")
    parser.add_argument("--gs_lrs",           default="1e-4",
                        help="Comma-separated learning rates to sweep.")
    parser.add_argument("--gs_epochs",        default="30",
                        help="Comma-separated epoch counts to sweep. (default: 20,30)")
    parser.add_argument("--gs_thresholds",    default="0.4",
                        help="Comma-separated decision thresholds to sweep. (default: 0.4)")
    parser.add_argument("--gs_neg_rates",     default="1.0",
                        help="Comma-separated neg downsample rates to sweep. (default: 1.0)")
    parser.add_argument("--gs_pw_scales",     default="1.0",
                        help="Comma-separated pos_weight_scale values to sweep. "
                             "<1.0 reduces false positives. (default: 0.5,0.75,1.0)")
    parser.add_argument("--gs_context_before", default="4",
                        help="Comma-separated context_before values to sweep. "
                             "(default: 0,2,4)")
    parser.add_argument("--gs_context_after",  default="0",
                        help="Comma-separated context_after values to sweep. "
                             "(default: 0)")

    # ── output ───────────────────────────────────────────────────────────────
    parser.add_argument("--outdir", default="cnn_output/")

    args   = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dir_path = Path(args.dir)
    if not dir_path.exists():
        raise ValueError(f"Directory not found: {args.dir}")
    os.makedirs(args.outdir, exist_ok=True)

    target_col  = f"{args.target}_important"
    allowed_ext = {".csv", ".tsv", ".xlsx"}

    # ── load transcripts ──────────────────────────────────────────────────────
    transcripts: dict[str, pd.DataFrame] = {}
    for fp in sorted(dir_path.iterdir()):
        if fp.suffix.lower() in allowed_ext:
            print(f"Loading {fp.name} …")
            df = load_da_level(fp)
            df = df[df[DA_COLUMN] != "I-"].reset_index(drop=True)
            if target_col not in df.columns:
                print(f"  Warning: '{target_col}' not found — skipping.")
                continue
            df[target_col] = df[target_col].fillna(0).astype(int)
            n_pos = int(df[target_col].sum())
            n_tot = len(df)
            print(f"  {n_tot} DAs  {n_pos} important "
                  f"({100*n_pos/max(n_tot,1):.1f}%)")
            transcripts[fp.name] = df

    if len(transcripts) < 2:
        raise RuntimeError(
            f"Need ≥2 transcripts for LOOCV, found {len(transcripts)}."
        )

    vocab = build_vocabulary(transcripts, args.granularity)
    print(f"\nLoaded {len(transcripts)} transcripts.")
    print(f"Vocabulary: {len(vocab)} (DA × speaker) tokens")
    print(f"Granularity: {args.granularity}  |  Target: {target_col}")
    print(f"Device: {device}")

    label_prefix = f"{args.target}_{args.granularity}"

    # ── grid search path ──────────────────────────────────────────────────────
    if args.grid_search:
        grid = build_grid(args)
        print(f"\nGrid size: {len(grid)} combinations")

        best_hp = run_grid_search(
            transcripts=transcripts,
            vocab=vocab,
            granularity=args.granularity,
            target_col=target_col,
            grid=grid,
            device=device,
            outdir=args.outdir,
            label_prefix=label_prefix,
        )

        # Full verbose LOOCV + final model with best config
        print(f"\n{'#'*60}")
        print("  Re-running best config (verbose) + training final model …")
        print(f"{'#'*60}")
        results, final_model = run_loocv(
            transcripts=transcripts,
            vocab=vocab,
            granularity=args.granularity,
            target_col=target_col,
            hp=best_hp,
            device=device,
            verbose=True,
            train_final=True,
        )
        run_label = f"{label_prefix}_best_{best_hp.label()}"

    # ── single-run path ───────────────────────────────────────────────────────
    else:
        hp = HParams(
            kernel_size=         args.kernel_size,
            num_layers=          args.num_layers,
            embed_dim=           args.embed_dim,
            hidden_dim=          args.hidden_dim,
            dropout=             args.dropout,
            lr=                  args.lr,
            epochs=              args.epochs,
            threshold=           args.threshold,
            downsample_neg_rate= args.downsample_neg_rate,
            pos_weight_scale=    args.pos_weight_scale,
            context_before=      args.context_before,
            context_after=       args.context_after,
        )
        eff = hp.num_layers * (hp.kernel_size - 1) + 1
        print(f"Receptive field: {eff} DAs  "
              f"(kernel={hp.kernel_size} × layers={hp.num_layers}, ±{eff//2})")

        results, final_model = run_loocv(
            transcripts=transcripts,
            vocab=vocab,
            granularity=args.granularity,
            target_col=target_col,
            hp=hp,
            device=device,
            verbose=True,
            train_final=True,
        )
        run_label = (
            f"{label_prefix}"
            f"_k{hp.kernel_size}_l{hp.num_layers}"
            f"_e{hp.epochs}_t{int(hp.threshold*100)}"
        )

    save_results(results, final_model, args.outdir, run_label)
    print(f"\nDone. Outputs in: {args.outdir}")


if __name__ == "__main__":
    main()

