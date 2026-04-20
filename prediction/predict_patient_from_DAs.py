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

Usage
-----
python cnn_importance.py \\
    --dir /path/to/csv_dir \\
    --granularity groups \\
    --target patient \\
    --kernel_size 5 \\
    --num_layers 3 \\
    --outdir cnn_output/

Requires: torch  (pip install torch)
          scikit-learn
Drop alongside analyze_da_patterns.py.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

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
    Maps (da_label × speaker) pairs to integer indices.

    Combining both into one token lets the embedding learn joint
    representations — therapist::canonical_questions gets its own
    vector distinct from patient::canonical_questions.
    """

    def __init__(self):
        self.token2idx: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2token: list[str]      = [PAD_TOKEN, UNK_TOKEN]

    def add(self, da_label: str, speaker: str):
        key = f"{speaker}::{da_label}"
        if key not in self.token2idx:
            self.token2idx[key] = len(self.idx2token)
            self.idx2token.append(key)

    def encode(self, da_label: str, speaker: str) -> int:
        return self.token2idx.get(
            f"{speaker}::{da_label}",
            self.token2idx[UNK_TOKEN],
        )

    def __len__(self) -> int:
        return len(self.idx2token)


def build_vocabulary(
    transcripts: dict[str, pd.DataFrame],
    granularity: str,
) -> Vocabulary:
    vocab = Vocabulary()
    for df in transcripts.values():
        for _, row in df.iterrows():
            da   = get_label(row[DA_COLUMN], row["da_group"], granularity)
            spkr = _normalise_speaker(str(row.get("speaker", "patient")))
            vocab.add(da, spkr)
    return vocab


# ── data preparation ──────────────────────────────────────────────────────────

def df_to_tensors(
    df:          pd.DataFrame,
    granularity: str,
    target_col:  str,
    vocab:       Vocabulary,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Convert one transcript DataFrame to three tensors of shape (seq_len,):
      token_ids  : LongTensor   — vocab index per DA
      ts_vals    : FloatTensor  — normalised timestamp per DA (zeros if absent)
      labels     : FloatTensor  — per-DA importance 0.0 / 1.0
    """
    ts_norm = _normalise_timestamp(df)
    has_ts  = ts_norm is not None

    token_ids, ts_vals, labels = [], [], []

    for i, (_, row) in enumerate(df.iterrows()):
        da   = get_label(row[DA_COLUMN], row["da_group"], granularity)
        spkr = _normalise_speaker(str(row.get("speaker", "patient")))
        token_ids.append(vocab.encode(da, spkr))
        ts_vals.append(float(ts_norm[i]) if has_ts else 0.0)
        labels.append(float(int(row[target_col])))

    return (
        torch.tensor(token_ids, dtype=torch.long),
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
        in_channels    = embed_dim + (1 if use_ts else 0)
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
        ts_vals:   torch.Tensor,   # (seq_len,) or (batch, seq_len)
    ) -> torch.Tensor:             # (seq_len,) or (batch, seq_len)  logits

        unbatched = token_ids.dim() == 1
        if unbatched:
            token_ids = token_ids.unsqueeze(0)
            ts_vals   = ts_vals.unsqueeze(0)

        x = self.embedding(token_ids)           # (B, L, embed_dim)

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
) -> float:
    """
    One epoch: one forward pass per transcript (no cross-transcript batching
    since transcripts have different lengths).  Returns mean loss.
    """
    model.train()
    total_loss = 0.0
    order      = rng.permutation(len(sequences)).tolist()

    for idx in order:
        tok, ts, lbl, _ = sequences[idx]
        tok = tok.to(device)
        ts  = ts.to(device)
        lbl = lbl.to(device)

        mask   = make_loss_mask(lbl.cpu(), downsample_neg_rate, rng).to(device)
        logits = model(tok, ts)
        loss   = criterion(logits[mask], lbl[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(sequences), 1)


@torch.no_grad()
def predict_transcript(
    model:     CNNImportanceClassifier,
    tok:       torch.Tensor,
    ts:        torch.Tensor,
    device:    torch.device,
    threshold: float = 0.5,
) -> list[int]:
    model.eval()
    logits = model(tok.to(device), ts.to(device))
    probs  = torch.sigmoid(logits).cpu()
    return (probs >= threshold).long().tolist()


# ── LOOCV ─────────────────────────────────────────────────────────────────────

def run_loocv(
    transcripts:         dict[str, pd.DataFrame],
    vocab:               Vocabulary,
    granularity:         str,
    target_col:          str,
    embed_dim:           int,
    hidden_dim:          int,
    num_layers:          int,
    kernel_size:         int,
    dropout:             float,
    lr:                  float,
    epochs:              int,
    threshold:           float,
    downsample_neg_rate: float,
    device:              torch.device,
) -> tuple[dict, CNNImportanceClassifier | None]:

    rng     = np.random.default_rng(SEED)
    fnames  = list(transcripts.keys())
    n_folds = len(fnames)
    use_ts  = any("timestamp" in df.columns for df in transcripts.values())
    eff     = num_layers * (kernel_size - 1) + 1

    print(f"\n{'='*60}")
    print(f"  CNN LOOCV  |  {n_folds} folds  |  target: {target_col}")
    print(f"  kernel={kernel_size}  layers={num_layers}  "
          f"receptive field={eff} DAs (±{eff//2})")
    print(f"  embed={embed_dim}  hidden={hidden_dim}  dropout={dropout}")
    print(f"  lr={lr}  epochs={epochs}  threshold={threshold}  "
          f"neg_downsample={downsample_neg_rate:.0%}")
    print(f"  device={device}")
    print(f"{'='*60}")

    # Pre-convert all transcripts to tensors
    all_tensors: dict[str, tuple] = {
        fname: (*df_to_tensors(df, granularity, target_col, vocab), fname)
        for fname, df in transcripts.items()
    }

    all_true:    list[int] = []
    all_pred:    list[int] = []
    fold_results = []

    for fold_idx, test_fname in enumerate(fnames):
        print(f"\n  Fold {fold_idx+1}/{n_folds}  —  test: {test_fname}")

        train_seqs = [v for k, v in all_tensors.items() if k != test_fname]

        n_pos = sum(int(t[2].sum()) for t in train_seqs)
        n_tot = sum(len(t[2]) for t in train_seqs)
        n_neg = n_tot - n_pos
        print(f"    train: {n_tot} DAs  {n_pos} positive "
              f"({100*n_pos/max(n_tot,1):.1f}%)")

        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1)], dtype=torch.float
        ).to(device)

        model = CNNImportanceClassifier(
            vocab_size=len(vocab),
            embed_dim=embed_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            kernel_size=kernel_size,
            dropout=dropout,
            use_ts=use_ts,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(epochs):
            loss = train_epoch(
                model, optimizer, criterion,
                train_seqs, device, rng, downsample_neg_rate,
            )
            if (epoch + 1) % max(1, epochs // 5) == 0:
                print(f"      epoch {epoch+1}/{epochs}  loss={loss:.4f}")

        tok, ts, lbl, _ = all_tensors[test_fname]
        y_pred = predict_transcript(model, tok, ts, device, threshold)
        y_true = lbl.long().tolist()

        all_true.extend(y_true)
        all_pred.extend(y_pred)

        n_pos_test = sum(y_true)
        n_pred_pos = sum(y_pred)
        f1_imp = f1_score(y_true, y_pred, pos_label=1,
                          average="binary", zero_division=0)
        f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
        prec   = precision_score(y_true, y_pred, pos_label=1,
                                 average="binary", zero_division=0)
        rec    = recall_score(y_true, y_pred, pos_label=1,
                              average="binary", zero_division=0)
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
            "precision_imp":    round(prec,   4),
            "recall_imp":       round(rec,    4),
        })

    if not all_true:
        return {"fold_results": fold_results}, None

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
                            average="macro", zero_division=0)
    mean_fold_f1 = float(np.mean([r["f1_important"] for r in fold_results]))
    std_fold_f1  = float(np.std( [r["f1_important"] for r in fold_results]))

    print(f"  Pooled F1(important): {agg_f1_imp:.4f}")
    print(f"  Pooled F1(macro):     {agg_f1_mac:.4f}")
    print(f"  Per-fold F1(imp):  mean={mean_fold_f1:.4f}  std={std_fold_f1:.4f}")

    results = {
        "fold_results":            fold_results,
        "pooled_f1_imp":           round(agg_f1_imp,  4),
        "pooled_f1_macro":         round(agg_f1_mac,  4),
        "mean_fold_f1_imp":        round(mean_fold_f1, 4),
        "std_fold_f1_imp":         round(std_fold_f1,  4),
        "effective_receptive_field": eff,
    }

    # Final model on all transcripts
    print("\n  Training final model on all transcripts …")
    n_pos_all = sum(int(t[2].sum()) for t in all_tensors.values())
    n_tot_all = sum(len(t[2]) for t in all_tensors.values())
    pos_weight_all = torch.tensor(
        [(n_tot_all - n_pos_all) / max(n_pos_all, 1)], dtype=torch.float
    ).to(device)

    final_model = CNNImportanceClassifier(
        vocab_size=len(vocab), embed_dim=embed_dim, hidden_dim=hidden_dim,
        num_layers=num_layers, kernel_size=kernel_size,
        dropout=dropout, use_ts=use_ts,
    ).to(device)
    final_opt       = torch.optim.Adam(final_model.parameters(), lr=lr)
    final_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_all)

    for epoch in range(epochs):
        loss = train_epoch(
            final_model, final_opt, final_criterion,
            list(all_tensors.values()), device, rng, downsample_neg_rate,
        )
        if (epoch + 1) % max(1, epochs // 5) == 0:
            print(f"    epoch {epoch+1}/{epochs}  loss={loss:.4f}", flush=True)

    return results, final_model


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


def main():
    parser = argparse.ArgumentParser(
        description="1D CNN per-DA importance classifier (full-sequence forward pass)."
    )
    parser.add_argument("--dir",                 required=True)
    parser.add_argument("--granularity",         default="groups",
                        choices=["groups", "raw"])
    parser.add_argument("--target",              default="patient",
                        choices=["patient", "therapist"])

    parser.add_argument("--kernel_size",  type=int, default=5,
                        help="Conv kernel width.  Each position sees "
                             "(kernel_size-1)/2 neighbours on each side per layer. "
                             "kernel=5 → ±2 per layer.  (default: 5)")
    parser.add_argument("--num_layers",   type=int, default=3,
                        help="Stacked conv layers.  "
                             "Receptive field = num_layers×(kernel_size-1)+1. "
                             "3 layers of kernel=5 → field=13 (±6).  (default: 3)")

    parser.add_argument("--embed_dim",    type=int,   default=64)
    parser.add_argument("--hidden_dim",   type=int,   default=128)
    parser.add_argument("--dropout",      type=float, default=0.3)

    parser.add_argument("--lr",                  type=float, default=1e-3)
    parser.add_argument("--epochs",              type=int,   default=30)
    parser.add_argument("--downsample_neg_rate", type=float, default=0.2,
                        help="Fraction of non-important DAs included in the loss "
                             "each epoch (default 1.0 = all, no downsampling). "
                             "0.3 keeps 30%% of negatives in each gradient step, "
                             "complementing pos_weight.")
    parser.add_argument("--threshold",           type=float, default=0.5,
                        help="Sigmoid decision threshold (default 0.5). "
                             "Lower to increase recall.")
    parser.add_argument("--outdir",              default="cnn_output/")
    args = parser.parse_args()

    device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dir_path = Path(args.dir)
    if not dir_path.exists():
        raise ValueError(f"Directory not found: {args.dir}")
    os.makedirs(args.outdir, exist_ok=True)

    target_col  = f"{args.target}_important"
    allowed_ext = {".csv", ".tsv", ".xlsx"}

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
    eff   = args.num_layers * (args.kernel_size - 1) + 1

    print(f"\nLoaded {len(transcripts)} transcripts.")
    print(f"Vocabulary: {len(vocab)} (DA × speaker) tokens")
    print(f"Granularity: {args.granularity}  |  Target: {target_col}")
    print(f"Receptive field: {eff} DAs  "
          f"(kernel={args.kernel_size} × layers={args.num_layers}, ±{eff//2})")
    print(f"Device: {device}")

    results, final_model = run_loocv(
        transcripts=transcripts, vocab=vocab,
        granularity=args.granularity, target_col=target_col,
        embed_dim=args.embed_dim, hidden_dim=args.hidden_dim,
        num_layers=args.num_layers, kernel_size=args.kernel_size,
        dropout=args.dropout, lr=args.lr, epochs=args.epochs,
        threshold=args.threshold,
        downsample_neg_rate=args.downsample_neg_rate,
        device=device,
    )

    label = (
        f"{args.target}_{args.granularity}"
        f"_k{args.kernel_size}_l{args.num_layers}"
        f"_e{args.epochs}_t{int(args.threshold*100)}"
    )
    save_results(results, final_model, args.outdir, label)
    print(f"\nDone. Outputs in: {args.outdir}")


if __name__ == "__main__":
    main()