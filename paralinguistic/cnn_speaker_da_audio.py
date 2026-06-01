"""
cnn_importance.py

Predicts per-DA importance (patient or therapist) from dialogue act sequences
using a 1D CNN, with optional acoustic feature channels.

Per-DA acoustic features are loaded from per-transcript CSVs produced by
acoustic_analysis.py. Since features are extracted at turn level, all DAs
within the same turn share the same feature vector. DAs whose turn has no
acoustic features are masked out of the loss (not penalised for missing data).
Acoustic features are z-score normalised per transcript.

Task
----
Sequence-to-sequence classification: given a full transcript of N DA tokens,
predict for each position whether that DA is important (1) or not (0).

Model architecture
------------------
  Embedding        — learnable dense vector per da_label token
  Speaker channel  — explicit 0/1 float (0=therapist, 1=other)
  Optional TS      — normalised timestamp appended as extra channel
  Optional acoustic — n_acoustic features per DA (turn-level, z-score normalised)
  1D Conv stack    — num_layers conv layers with ReLU + Dropout, same-padding
  Linear → logit   — per-position binary output

Effective receptive field = num_layers x (kernel_size - 1) + 1

Class imbalance
---------------
BCEWithLogitsLoss(pos_weight = n_neg/n_pos * pos_weight_scale).
Optional --downsample_neg_rate randomly excludes negatives from the loss.

Context expansion
-----------------
--context_before / --context_after expand the positive label region during
training only. Evaluation always uses original unmodified annotations.

Grid Search
-----------
Use --grid_search to sweep all --gs_* parameter lists via LOOCV.

Usage
-----
# Single run (no acoustics):
python cnn_importance.py --dir /path/to/csvs --target therapist

# Single run with acoustics:
python cnn_importance.py --dir /path/to/csvs --target therapist \\
    --features_dir acoustic_features/

# Grid search:
python cnn_importance.py --dir /path/to/csvs --target therapist \\
    --grid_search \\
    --gs_kernel_sizes 5,11 --gs_num_layers 3,5 --gs_hidden_dims 64,128
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
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

SEED = 42
torch.manual_seed(SEED)
np.random.seed(SEED)


@dataclass
class HParams:
    kernel_size: int = 11
    num_layers: int = 11
    embed_dim: int = 64
    hidden_dim: int = 32
    dropout: float = 0.3
    lr: float = 1e-4
    epochs: int = 30
    threshold: float = 0.4
    downsample_neg_rate: float = 1.0
    pos_weight_scale: float = 0.75
    context_before: int = 4
    context_after: int = 0

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


def expand_context(
    labels: list[int],
    context_before: int,
    context_after: int,
) -> list[int]:
    """
    Expand true-positive label positions by context_before steps left
    and context_after steps right. Applied to training labels only —
    evaluation always uses original annotations.

    Example (context_before=2, context_after=0):
      original: [0, 0, 0, 1, 1, 0, 0, 0, 1, 0]
      expanded: [0, 1, 1, 1, 1, 0, 0, 1, 1, 0]
    """
    if context_before == 0 and context_after == 0:
        return labels
    n = len(labels)
    result = list(labels)
    for i, lbl in enumerate(labels):
        if lbl == 1:
            for j in range(max(0, i - context_before), i):
                result[j] = 1
            for j in range(i + 1, min(n, i + context_after + 1)):
                result[j] = 1
    return result


PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class Vocabulary:
    """
    Maps da_label tokens to integer indices.
    Speaker is handled as a separate explicit 0/1 channel in the model,
    not folded into the token identity.
    """

    def __init__(self):
        self.token2idx: dict[str, int] = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2token: list[str] = [PAD_TOKEN, UNK_TOKEN]

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


def _find_acoustic_csv(transcript_name: str, features_dir: Path) -> Path | None:
    stem = Path(transcript_name).stem
    candidate = features_dir / f"{stem}_acoustic.csv"
    if candidate.exists():
        return candidate
    if features_dir.exists():
        for f in features_dir.iterdir():
            if f.name.lower() == f"{stem.lower()}_acoustic.csv":
                return f
    return None


def load_acoustic_features(
    transcript_name: str,
    features_dir: Path,
) -> tuple[dict[int, np.ndarray] | None, int]:
    csv_path = _find_acoustic_csv(transcript_name, features_dir)
    if csv_path is None:
        return None, 0
    df = pd.read_csv(csv_path)
    meta_cols = {"filename", "therapist_id", "patient_id", "turn_id",
                 "start_s", "end_s", "speaker", "pat_label", "ther_label",
                 "has_audio"}
    feat_cols = [c for c in df.columns if c not in meta_cols]
    if not feat_cols:
        return None, 0
    turn_to_feat: dict[int, np.ndarray] = {}
    for _, row in df.iterrows():
        turn_id = int(row["turn_id"])
        feat = row[feat_cols].values.astype(np.float32)
        feat = np.where(np.isfinite(feat), feat, 0.0)
        turn_to_feat[turn_id] = feat
    return turn_to_feat, len(feat_cols)


def normalise_acoustic_per_transcript(
    feat_matrix: np.ndarray,
    valid_mask: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(feat_matrix, dtype=np.float32)
    if not valid_mask.any():
        return result
    valid_feats = feat_matrix[valid_mask]
    mean = valid_feats.mean(axis=0)
    std = valid_feats.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    result[valid_mask] = (valid_feats - mean) / std
    return result


def _assign_turn_ids(df: pd.DataFrame) -> list[int]:
    turn_ids, turn_id, prev_key = [], 0, None
    for _, row in df.iterrows():
        ts = row.get("timestamp", 0)
        spkr = _normalise_speaker(str(row.get("speaker", "patient")))
        key = (ts, spkr)
        if key != prev_key:
            if prev_key is not None:
                turn_id += 1
            prev_key = key
        turn_ids.append(turn_id)
    return turn_ids


def df_to_tensors(
    df: pd.DataFrame,
    granularity: str,
    target_col: str,
    vocab: Vocabulary,
    turn_to_feat: dict[int, np.ndarray] | None,
    n_acoustic: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor,
           torch.Tensor, torch.Tensor, torch.Tensor, str]:
    """
    Returns:
      token_ids : LongTensor  (seq_len,)
      spkr_vals : FloatTensor (seq_len,)   0=therapist 1=other
      ts_vals : FloatTensor (seq_len,)
      labels : FloatTensor (seq_len,)
      acoustic : FloatTensor (seq_len, n_acoustic)
      acou_mask : BoolTensor  (seq_len,)
      fname : str
    """
    ts_norm = _normalise_timestamp(df)
    has_ts = ts_norm is not None

    token_ids, spkr_vals, ts_vals, labels = [], [], [], []
    acou_rows, acou_valid = [], []

    turn_id_per_da = _assign_turn_ids(df)

    for i, (_, row) in enumerate(df.iterrows()):
        da = get_label(row[DA_COLUMN], row["da_group"], granularity)
        spkr = _normalise_speaker(str(row.get("speaker", "patient")))
        token_ids.append(vocab.encode(da))
        spkr_vals.append(0.0 if spkr == "therapist" else 1.0)
        ts_vals.append(float(ts_norm[i]) if has_ts else 0.0)
        labels.append(float(int(row[target_col])))

        turn_id = turn_id_per_da[i]
        if turn_to_feat is not None and turn_id in turn_to_feat:
            acou_rows.append(turn_to_feat[turn_id].copy())
            acou_valid.append(True)
        else:
            acou_rows.append(np.zeros(n_acoustic, dtype=np.float32))
            acou_valid.append(False)

    feat_matrix = np.stack(acou_rows) if acou_rows else np.zeros((0, n_acoustic))
    valid_mask = np.array(acou_valid, dtype=bool)
    feat_norm = normalise_acoustic_per_transcript(feat_matrix, valid_mask)

    return (
        torch.tensor(token_ids, dtype=torch.long),
        torch.tensor(spkr_vals, dtype=torch.float),
        torch.tensor(ts_vals, dtype=torch.float),
        torch.tensor(labels, dtype=torch.float),
        torch.tensor(feat_norm, dtype=torch.float),
        torch.tensor(valid_mask, dtype=torch.bool),
        df.attrs.get("fname", ""),
    )


def make_loss_mask(
    labels: torch.Tensor,
    acou_mask: torch.Tensor,
    downsample_neg_rate: float,
    rng: np.random.Generator,
    require_acoustic: bool = False,
) -> torch.Tensor:
    base_mask = acou_mask.clone() if require_acoustic else torch.ones(len(labels), dtype=torch.bool)

    if downsample_neg_rate >= 1.0:
        return base_mask

    neg_positions = ((labels == 0) & base_mask).nonzero(as_tuple=True)[0].numpy()
    n_keep = max(1, int(len(neg_positions) * downsample_neg_rate))
    if len(neg_positions) > n_keep:
        keep = rng.choice(len(neg_positions), size=n_keep, replace=False)
        neg_positions = neg_positions[keep]

    mask = (labels == 1) & base_mask
    for idx in neg_positions:
        mask[idx] = True
    return mask


class CNNImportanceClassifier(nn.Module):
    """
    Per-position 1D CNN classifier.

    Input channels per position:
      embed_dim (DA embedding) + 1 (speaker) + 1 (timestamp, optional)
      + n_acoustic (acoustic features, optional)

    All concatenated before the conv stack. Conv filters learn to combine
    sequential DA patterns, speaker identity, and prosodic features jointly.
    """

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 64,
        hidden_dim: int = 32,
        num_layers: int = 11,
        kernel_size: int = 11,
        dropout: float = 0.3,
        use_ts: bool = False,
        n_acoustic: int = 0,
    ):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.use_ts = use_ts
        self.n_acoustic = n_acoustic

        # +1 for explicit speaker channel (always present)
        in_channels = embed_dim + 1
        if use_ts:
            in_channels += 1
        if n_acoustic > 0:
            in_channels += n_acoustic

        pad = kernel_size // 2

        layers: list[nn.Module] = []
        for i in range(num_layers):
            layers += [
                nn.Conv1d(
                    in_channels if i == 0 else hidden_dim,
                    hidden_dim,
                    kernel_size=kernel_size,
                    padding=pad,
                ),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]

        self.conv_stack = nn.Sequential(*layers)
        self.output = nn.Linear(hidden_dim, 1)

    def forward(
        self,
        token_ids: torch.Tensor,
        spkr_vals: torch.Tensor,
        ts_vals: torch.Tensor,
        acoustic: torch.Tensor,
    ) -> torch.Tensor:
        unbatched = token_ids.dim() == 1
        if unbatched:
            token_ids = token_ids.unsqueeze(0)
            spkr_vals = spkr_vals.unsqueeze(0)
            ts_vals = ts_vals.unsqueeze(0)
            acoustic = acoustic.unsqueeze(0)

        x = self.embedding(token_ids)
        x = torch.cat([x, spkr_vals.unsqueeze(-1)], dim=-1)

        if self.use_ts:
            x = torch.cat([x, ts_vals.unsqueeze(-1)], dim=-1)
        if self.n_acoustic > 0:
            x = torch.cat([x, acoustic], dim=-1)

        x = x.permute(0, 2, 1)
        x = self.conv_stack(x)
        x = x.permute(0, 2, 1)
        logits = self.output(x).squeeze(-1)

        return logits.squeeze(0) if unbatched else logits


def train_epoch(
    model: CNNImportanceClassifier,
    optimizer: torch.optim.Optimizer,
    criterion: nn.BCEWithLogitsLoss,
    sequences: list[tuple],
    device: torch.device,
    rng: np.random.Generator,
    hp: HParams,
    require_acoustic: bool,
) -> float:
    model.train()
    total_loss = 0.0
    order = rng.permutation(len(sequences)).tolist()

    for idx in order:
        tok, spkr, ts, lbl, acou, acou_mask, _ = sequences[idx]
        tok = tok.to(device)
        spkr = spkr.to(device)
        ts = ts.to(device)
        acou = acou.to(device)
        acou_mask = acou_mask.to(device)

        lbl_expanded = torch.tensor(
            expand_context(lbl.long().tolist(),
                           hp.context_before, hp.context_after),
            dtype=torch.float,
        ).to(device)

        mask = make_loss_mask(
            lbl_expanded.cpu(), acou_mask.cpu(),
            hp.downsample_neg_rate, rng, require_acoustic,
        ).to(device)

        if mask.sum() == 0:
            continue

        logits = model(tok, spkr, ts, acou)
        loss = criterion(logits[mask], lbl_expanded[mask])

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    return total_loss / max(len(sequences), 1)


@torch.no_grad()
def predict_transcript(
    model: CNNImportanceClassifier,
    tok: torch.Tensor,
    spkr: torch.Tensor,
    ts: torch.Tensor,
    acou: torch.Tensor,
    device: torch.device,
    threshold: float = 0.5,
) -> list[int]:
    model.eval()
    logits = model(tok.to(device), spkr.to(device),
                   ts.to(device), acou.to(device))
    probs = torch.sigmoid(logits).cpu()
    return (probs >= threshold).long().tolist()


def run_loocv(
    transcripts: dict[str, pd.DataFrame],
    vocab: Vocabulary,
    granularity: str,
    target_col: str,
    hp: HParams,
    features_dir: Path | None,
    n_acoustic: int,
    require_acoustic: bool,
    device: torch.device,
    verbose: bool = True,
    train_final: bool = True,
) -> tuple[dict, CNNImportanceClassifier | None]:

    rng = np.random.default_rng(SEED)
    fnames = list(transcripts.keys())
    n_folds = len(fnames)
    use_ts = any("timestamp" in df.columns for df in transcripts.values())
    eff = hp.num_layers * (hp.kernel_size - 1) + 1

    if verbose:
        print(f"\n{'='*60}")
        print(f"  CNN LOOCV  |  {n_folds} folds  |  target: {target_col}")
        print(f"  kernel={hp.kernel_size}  layers={hp.num_layers}  "
              f"receptive field={eff} DAs (+-{eff//2})")
        print(f"  embed={hp.embed_dim}  hidden={hp.hidden_dim}  "
              f"dropout={hp.dropout}")
        print(f"  lr={hp.lr}  epochs={hp.epochs}  threshold={hp.threshold}  "
              f"neg_downsample={hp.downsample_neg_rate:.0%}")
        print(f"  pos_weight_scale={hp.pos_weight_scale}  "
              f"context_before={hp.context_before}  "
              f"context_after={hp.context_after}")
        print(f"  n_acoustic={n_acoustic}  require_acoustic={require_acoustic}")
        print(f"  device={device}")
        print(f"{'='*60}")

    all_tensors: dict[str, tuple] = {}
    for fname, df in transcripts.items():
        turn_to_feat = None
        if features_dir is not None:
            turn_to_feat, _ = load_acoustic_features(fname, features_dir)
            if turn_to_feat is None and verbose:
                print(f"  Warning: no acoustic features for {fname}", flush=True)
        df.attrs["fname"] = fname
        all_tensors[fname] = df_to_tensors(
            df, granularity, target_col, vocab, turn_to_feat, n_acoustic,
        )

    if verbose:
        n_with = sum(1 for v in all_tensors.values() if v[5].any())
        print(f"  Acoustic features loaded for {n_with}/{n_folds} transcripts",
              flush=True)

    all_true, all_pred, fold_results = [], [], []

    for fold_idx, test_fname in enumerate(fnames):
        if verbose:
            print(f"\n  Fold {fold_idx+1}/{n_folds}  -  test: {test_fname}")

        train_seqs = [v for k, v in all_tensors.items() if k != test_fname]
        n_pos = sum(int(t[3].sum()) for t in train_seqs)
        n_tot = sum(len(t[3])       for t in train_seqs)
        n_neg = n_tot - n_pos
        if verbose:
            print(f"    train: {n_tot} DAs  {n_pos} positive "
                  f"({100*n_pos/max(n_tot,1):.1f}%)")

        pos_weight = torch.tensor(
            [n_neg / max(n_pos, 1) * hp.pos_weight_scale], dtype=torch.float
        ).to(device)

        model = CNNImportanceClassifier(
            vocab_size=len(vocab), embed_dim=hp.embed_dim,
            hidden_dim=hp.hidden_dim, num_layers=hp.num_layers,
            kernel_size=hp.kernel_size, dropout=hp.dropout,
            use_ts=use_ts, n_acoustic=n_acoustic,
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=hp.lr)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        for epoch in range(hp.epochs):
            loss = train_epoch(
                model, optimizer, criterion,
                train_seqs, device, rng, hp, require_acoustic,
            )
            if verbose and (epoch + 1) % max(1, hp.epochs // 5) == 0:
                print(f"      epoch {epoch+1}/{hp.epochs}  loss={loss:.4f}")

        tok, spkr, ts, lbl, acou, acou_mask, _ = all_tensors[test_fname]
        y_pred = predict_transcript(model, tok, spkr, ts, acou, device, hp.threshold)
        y_true = lbl.long().tolist()

        all_true.extend(y_true)
        all_pred.extend(y_pred)

        n_pos_test = sum(y_true)
        n_pred_pos = sum(y_pred)
        n_acou_valid = int(acou_mask.sum())
        f1_imp = f1_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
        f1_mac = f1_score(y_true, y_pred, average="macro", zero_division=0)
        f1_bal = f1_score(y_true, y_pred, average="macro", zero_division=0)
        prec = precision_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)
        rec = recall_score(y_true, y_pred, pos_label=1, average="binary", zero_division=0)

        if verbose:
            print(f"    test: {len(y_true)} DAs  {n_pos_test} true pos  "
                  f"{n_pred_pos} predicted  {n_acou_valid} with acoustic  "
                  f"F1(imp)={f1_imp:.3f}  prec={prec:.3f}  rec={rec:.3f}")

        fold_results.append({
            "fold": fold_idx + 1,
            "transcript": test_fname,
            "n_das": len(y_true),
            "n_important": n_pos_test,
            "pct_important": round(100 * n_pos_test / max(len(y_true), 1), 1),
            "n_pred_important": n_pred_pos,
            "n_das_with_acoustic": n_acou_valid,
            "f1_important": round(f1_imp, 4),
            "f1_macro": round(f1_mac, 4),
            "f1_balanced": round(f1_bal, 4),
            "precision_imp": round(prec, 4),
            "recall_imp": round(rec, 4),
        })

    if not all_true:
        return {"fold_results": fold_results}, None

    if verbose:
        print(f"\n{'─'*60}")
        print("  Aggregate classification report (all folds):")
        print(classification_report(
            all_true, all_pred, labels=[0, 1],
            target_names=["not_important", "important"], zero_division=0,
        ))

    agg_f1_imp = f1_score(all_true, all_pred, pos_label=1, average="binary", zero_division=0)
    agg_f1_bal = f1_score(all_true, all_pred, average="macro", zero_division=0)
    agg_f1_mac = f1_score(all_true, all_pred, average="weighted", zero_division=0)
    mean_fold_f1 = float(np.mean([r["f1_important"] for r in fold_results]))
    std_fold_f1 = float(np.std( [r["f1_important"] for r in fold_results]))
    mean_fold_bal= float(np.mean([r["f1_balanced"]  for r in fold_results]))

    if verbose:
        print(f"  Pooled F1(important): {agg_f1_imp:.4f}")
        print(f"  Pooled F1(balanced): {agg_f1_bal:.4f}")
        print(f"  Pooled F1(weighted): {agg_f1_mac:.4f}")
        print(f"  Per-fold F1(imp): mean={mean_fold_f1:.4f}  std={std_fold_f1:.4f}")

    results = {
        "fold_results": fold_results,
        "pooled_f1_imp": round(agg_f1_imp, 4),
        "pooled_f1_balanced": round(agg_f1_bal, 4),
        "pooled_f1_weighted": round(agg_f1_mac, 4),
        "mean_fold_f1_imp": round(mean_fold_f1, 4),
        "std_fold_f1_imp": round(std_fold_f1, 4),
        "mean_fold_f1_balanced": round(mean_fold_bal, 4),
        "effective_receptive_field": eff,
        "n_acoustic_features": n_acoustic,
        "require_acoustic": require_acoustic,
    }

    final_model = None
    if train_final:
        if verbose:
            print("\n  Training final model on all transcripts ...")
        n_pos_all = sum(int(t[3].sum()) for t in all_tensors.values())
        n_tot_all = sum(len(t[3])       for t in all_tensors.values())
        pos_weight_all = torch.tensor(
            [(n_tot_all - n_pos_all) / max(n_pos_all, 1) * hp.pos_weight_scale],
            dtype=torch.float,
        ).to(device)

        final_model = CNNImportanceClassifier(
            vocab_size=len(vocab), embed_dim=hp.embed_dim,
            hidden_dim=hp.hidden_dim, num_layers=hp.num_layers,
            kernel_size=hp.kernel_size, dropout=hp.dropout,
            use_ts=use_ts, n_acoustic=n_acoustic,
        ).to(device)
        final_opt = torch.optim.Adam(final_model.parameters(), lr=hp.lr)
        final_criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight_all)

        for epoch in range(hp.epochs):
            loss = train_epoch(
                final_model, final_opt, final_criterion,
                list(all_tensors.values()), device, rng, hp, require_acoustic,
            )
            if verbose and (epoch + 1) % max(1, hp.epochs // 5) == 0:
                print(f"    epoch {epoch+1}/{hp.epochs}  loss={loss:.4f}", flush=True)

    return results, final_model



def _parse_list(s: str, cast) -> list:
    return [cast(v.strip()) for v in s.split(",") if v.strip()]


def build_grid(args) -> list[HParams]:
    combos = list(itertools.product(
        _parse_list(args.gs_kernel_sizes, int),
        _parse_list(args.gs_num_layers, int),
        _parse_list(args.gs_embed_dims, int),
        _parse_list(args.gs_hidden_dims, int),
        _parse_list(args.gs_dropouts, float),
        _parse_list(args.gs_lrs, float),
        _parse_list(args.gs_epochs, int),
        _parse_list(args.gs_thresholds, float),
        _parse_list(args.gs_neg_rates, float),
        _parse_list(args.gs_pw_scales, float),
        _parse_list(args.gs_context_before, int),
        _parse_list(args.gs_context_after, int),
    ))
    return [
        HParams(kernel_size=k, num_layers=nl, embed_dim=ed, hidden_dim=hd,
                dropout=dr, lr=lr, epochs=ep, threshold=th,
                downsample_neg_rate=neg, pos_weight_scale=pw,
                context_before=cb, context_after=ca)
        for k, nl, ed, hd, dr, lr, ep, th, neg, pw, cb, ca in combos
    ]


def run_grid_search(
    transcripts: dict[str, pd.DataFrame],
    vocab: Vocabulary,
    granularity: str,
    target_col: str,
    grid: list[HParams],
    features_dir: Path | None,
    n_acoustic: int,
    require_acoustic: bool,
    device: torch.device,
    outdir: str,
    label_prefix: str,
) -> HParams:
    n = len(grid)
    print(f"\n{'#'*60}")
    print(f"  GRID SEARCH  -  {n} configurations")
    print(f"  Ranking by: mean per-fold F1(important)")
    print(f"{'#'*60}")

    rows: list[dict[str, Any]] = []

    for i, hp in enumerate(grid):
        eff = hp.num_layers * (hp.kernel_size - 1) + 1
        print(f"\n[{i+1}/{n}]  {hp.label()}  (receptive field={eff})")
        try:
            results, _ = run_loocv(
                transcripts=transcripts, vocab=vocab,
                granularity=granularity, target_col=target_col,
                hp=hp, features_dir=features_dir,
                n_acoustic=n_acoustic, require_acoustic=require_acoustic,
                device=device, verbose=False, train_final=False,
            )
            row = {**asdict(hp), **{
                k: v for k, v in results.items() if k != "fold_results"
            }}
            print(
                f"    mean_fold_f1_imp={results['mean_fold_f1_imp']:.4f}  "
                f"+-{results['std_fold_f1_imp']:.4f}  "
                f"pooled_f1_bal={results['pooled_f1_balanced']:.4f}"
            )
        except Exception as exc:
            print(f"    ERROR: {exc}")
            row = {**asdict(hp), "error": str(exc)}
        rows.append(row)

    df_gs = pd.DataFrame(rows)
    if "mean_fold_f1_imp" in df_gs.columns:
        df_gs = df_gs.sort_values(
            ["mean_fold_f1_imp", "std_fold_f1_imp"],
            ascending=[False, True],
        ).reset_index(drop=True)

    gs_path = os.path.join(outdir, f"grid_search_{label_prefix}.csv")
    df_gs.to_csv(gs_path, index=False)
    print(f"\n  Grid search results saved: {gs_path}")

    best_row = df_gs.iloc[0]
    best_hp = HParams(
        kernel_size= int(best_row["kernel_size"]),
        num_layers= int(best_row["num_layers"]),
        embed_dim= int(best_row["embed_dim"]),
        hidden_dim= int(best_row["hidden_dim"]),
        dropout= float(best_row["dropout"]),
        lr= float(best_row["lr"]),
        epochs= int(best_row["epochs"]),
        threshold= float(best_row["threshold"]),
        downsample_neg_rate= float(best_row["downsample_neg_rate"]),
        pos_weight_scale= float(best_row["pos_weight_scale"]),
        context_before= int(best_row["context_before"]),
        context_after= int(best_row["context_after"]),
    )

    eff_best = best_hp.num_layers * (best_hp.kernel_size - 1) + 1
    print(f"\n  Best config (mean_fold_f1_imp="
          f"{best_row.get('mean_fold_f1_imp', '?'):.4f}):")
    print(f"    {best_hp.label()}")
    print(f"    receptive field = {eff_best} DAs")

    top5 = df_gs.head(5)[
        ["kernel_size", "num_layers", "hidden_dim", "dropout", "lr",
         "threshold", "downsample_neg_rate", "pos_weight_scale",
         "context_before", "mean_fold_f1_imp", "std_fold_f1_imp",
         "pooled_f1_balanced"]
    ].to_string(index=False)
    print(f"\n  Top 5 configurations:\n{top5}")

    return best_hp


def save_results(
    results: dict,
    model: CNNImportanceClassifier | None,
    outdir: str,
    label: str,
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
        description="1D CNN per-DA importance classifier with optional acoustic channels."
    )

    parser.add_argument("--dir", required=True)
    parser.add_argument("--granularity", default="groups", choices=["groups", "raw"])
    parser.add_argument("--target", default="patient", choices=["patient", "therapist"])

    parser.add_argument("--features_dir", default=None,
                        help="Directory of per-transcript acoustic CSVs from "
                             "acoustic_analysis.py. None = text-only.")
    parser.add_argument("--require_acoustic", action="store_true",
                        help="Exclude DAs with no acoustic features from the loss.")

    # single-run hyperparameters (defaults = best tuned values)
    parser.add_argument("--kernel_size", type=int, default=11)
    parser.add_argument("--num_layers", type=int, default=11)
    parser.add_argument("--embed_dim", type=int, default=64)
    parser.add_argument("--hidden_dim", type=int, default=32)
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--downsample_neg_rate", type=float, default=1.0)
    parser.add_argument("--threshold", type=float, default=0.4)
    parser.add_argument("--pos_weight_scale", type=float, default=0.75)
    parser.add_argument("--context_before", type=int, default=4)
    parser.add_argument("--context_after", type=int, default=0)

    # grid search
    parser.add_argument("--grid_search", action="store_true")
    parser.add_argument("--gs_kernel_sizes", default="11")
    parser.add_argument("--gs_num_layers", default="11")
    parser.add_argument("--gs_embed_dims", default="64")
    parser.add_argument("--gs_hidden_dims", default="32")
    parser.add_argument("--gs_dropouts", default="0.3")
    parser.add_argument("--gs_lrs", default="1e-4")
    parser.add_argument("--gs_epochs", default="30")
    parser.add_argument("--gs_thresholds", default="0.4")
    parser.add_argument("--gs_neg_rates", default="1.0")
    parser.add_argument("--gs_pw_scales", default="0.75")
    parser.add_argument("--gs_context_before", default="4")
    parser.add_argument("--gs_context_after", default="0")

    parser.add_argument("--outdir", default="cnn_output/")

    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dir_path = Path(args.dir)
    features_dir = Path(args.features_dir) if args.features_dir else None

    if not dir_path.exists():
        raise ValueError(f"Directory not found: {args.dir}")
    if features_dir and not features_dir.exists():
        print(f"Warning: features_dir not found: {args.features_dir} — "
              f"running without acoustic features.", flush=True)
        features_dir = None

    os.makedirs(args.outdir, exist_ok=True)

    target_col = f"{args.target}_important"
    allowed_ext = {".csv", ".tsv", ".xlsx"}

    transcripts: dict[str, pd.DataFrame] = {}
    for fp in sorted(dir_path.iterdir()):
        if fp.suffix.lower() in allowed_ext:
            print(f"Loading {fp.name} ...")
            df = load_da_level(fp)
            df = df[df[DA_COLUMN] != "I-"].reset_index(drop=True)
            if target_col not in df.columns:
                print(f"  Warning: '{target_col}' not found - skipping.")
                continue
            df[target_col] = df[target_col].fillna(0).astype(int)
            n_pos = int(df[target_col].sum())
            n_tot = len(df)
            print(f"  {n_tot} DAs  {n_pos} important "
                  f"({100*n_pos/max(n_tot,1):.1f}%)")
            transcripts[fp.name] = df

    if len(transcripts) < 2:
        raise RuntimeError(f"Need >=2 transcripts for LOOCV, found {len(transcripts)}.")

    # Determine n_acoustic
    n_acoustic = 0
    if features_dir is not None:
        for fname in transcripts:
            _, n = load_acoustic_features(fname, features_dir)
            if n > 0:
                n_acoustic = n
                break
        if n_acoustic == 0:
            print("Warning: no acoustic features could be loaded — running text-only.",
                  flush=True)
            features_dir = None

    vocab = build_vocabulary(transcripts, args.granularity)
    label_prefix = f"{args.target}_{args.granularity}"
    acou_tag = f"_acou{n_acoustic}" if n_acoustic > 0 else "_textonly"

    print(f"\nLoaded {len(transcripts)} transcripts.")
    print(f"Vocabulary: {len(vocab)} DA tokens")
    print(f"Granularity: {args.granularity}  |  Target: {target_col}")
    print(f"Acoustic features: {n_acoustic} "
          f"({str(features_dir) if features_dir else 'none'})")
    print(f"Device: {device}")

    if args.grid_search:
        grid = build_grid(args)
        print(f"\nGrid size: {len(grid)} combinations")
        best_hp = run_grid_search(
            transcripts=transcripts, vocab=vocab,
            granularity=args.granularity, target_col=target_col,
            grid=grid, features_dir=features_dir,
            n_acoustic=n_acoustic, require_acoustic=args.require_acoustic,
            device=device, outdir=args.outdir,
            label_prefix=label_prefix + acou_tag,
        )
        print(f"\n{'#'*60}")
        print("  Re-running best config (verbose) + training final model ...")
        print(f"{'#'*60}")
        results, final_model = run_loocv(
            transcripts=transcripts, vocab=vocab,
            granularity=args.granularity, target_col=target_col,
            hp=best_hp, features_dir=features_dir,
            n_acoustic=n_acoustic, require_acoustic=args.require_acoustic,
            device=device, verbose=True, train_final=True,
        )
        run_label = f"{label_prefix}{acou_tag}_best_{best_hp.label()}"

    else:
        hp = HParams(
            kernel_size= args.kernel_size,
            num_layers= args.num_layers,
            embed_dim= args.embed_dim,
            hidden_dim= args.hidden_dim,
            dropout= args.dropout,
            lr= args.lr,
            epochs= args.epochs,
            threshold= args.threshold,
            downsample_neg_rate= args.downsample_neg_rate,
            pos_weight_scale= args.pos_weight_scale,
            context_before= args.context_before,
            context_after= args.context_after,
        )
        eff = hp.num_layers * (hp.kernel_size - 1) + 1
        print(f"Receptive field: {eff} DAs  "
              f"(kernel={hp.kernel_size} x layers={hp.num_layers}, +-{eff//2})")

        results, final_model = run_loocv(
            transcripts=transcripts, vocab=vocab,
            granularity=args.granularity, target_col=target_col,
            hp=hp, features_dir=features_dir,
            n_acoustic=n_acoustic, require_acoustic=args.require_acoustic,
            device=device, verbose=True, train_final=True,
        )
        run_label = (
            f"{label_prefix}{acou_tag}"
            f"_k{hp.kernel_size}_l{hp.num_layers}"
            f"_e{hp.epochs}_t{int(hp.threshold*100)}"
        )

    save_results(results, final_model, args.outdir, run_label)
    print(f"\nDone. Outputs in: {args.outdir}")


if __name__ == "__main__":
    main()


# Useful cmds:
# grep pooled_f1_imp ./*