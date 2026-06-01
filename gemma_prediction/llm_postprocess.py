"""
llm_postprocess.py

Post-processing for sliding window importance predictions.

After the sliding window produces per-DA vote fractions, this module
applies a base-rate constrained re-ranking step that enforces the
empirical base rate from the training set.

APPROACH
--------
The sliding window classifier assigns each DA a vote fraction
(proportion of windows that voted important for that DA). Rather than
applying a fixed threshold, base-rate re-ranking:

  1. Ranks all DAs by vote fraction descending.
  2. Marks the top ceil(base_rate * n_das) DAs as important.
  3. Applies the standard min-run filter for contiguity.

This directly enforces the empirical base rate from the training set,
preventing the model from predicting 40% of a transcript as important
when the true rate is ~10%.

Both the original threshold predictions and the re-ranked predictions
are returned so they can be evaluated and compared side by side.

USAGE
-----
Called from llm_importance_sliding.py after run_predictions:

    from llm_postprocess import apply_base_rate_reranking, evaluate_reranked

    reranked_rows, reranked_metrics = apply_base_rate_reranking(
        pred_rows=pred_rows,
        base_rate=base_rate,
        min_important_run=args.min_important_run,
        min_unimportant_run=args.min_unimportant_run,
        filter_order=args.filter_order,
    )
"""

from __future__ import annotations

import math
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)


# ── helpers ───────────────────────────────────────────────────────────────────

def _apply_min_run_filter(
    preds:           list[int],
    min_important:   int,
    min_unimportant: int,
    filter_order:    str = "unimportant_first",
) -> list[int]:
    """
    Same logic as apply_min_run_filter in llm_importance_sliding.py.
    Duplicated here so this module has no imports from the sliding script.
    """
    result = list(preds)

    passes = (
        [(0, min_unimportant), (1, min_important)]
        if filter_order == "unimportant_first"
        else [(1, min_important), (0, min_unimportant)]
    )

    for target, min_len in passes:
        i = 0
        while i < len(result):
            if result[i] == target:
                j = i
                while j < len(result) and result[j] == target:
                    j += 1
                if j - i < min_len:
                    for k in range(i, j):
                        result[k] = 1 - target
                i = j
            else:
                i += 1

    return result


def _evaluate_silent(y_true: list[int], y_pred: list[int]) -> dict:
    """Compute metrics without printing."""
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TP  = int(cm[1, 1]);  TN = int(cm[0, 0])
    FP  = int(cm[0, 1]);  FN = int(cm[1, 0])

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    precision   = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1_imp      = (2 * precision * sensitivity / (precision + sensitivity)
                   if (precision + sensitivity) > 0 else 0.0)
    f1_bal      = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_wt       = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "sensitivity":  round(sensitivity, 4),
        "specificity":  round(specificity, 4),
        "precision":    round(precision,   4),
        "f1_important": round(f1_imp,      4),
        "f1_balanced":  round(f1_bal,      4),
        "f1_weighted":  round(f1_wt,       4),
    }


# ── main post-processing function ─────────────────────────────────────────────

def apply_base_rate_reranking(
    pred_rows:           list[dict],
    base_rate:           float,
    min_important_run:   int   = 10,
    min_unimportant_run: int   = 1,
    filter_order:        str   = "unimportant_first",
) -> tuple[list[dict], dict]:
    """
    Apply base-rate constrained re-ranking to sliding window predictions.

    For each transcript:
      1. Rank DAs by vote_frac descending.
      2. Mark the top ceil(base_rate * n_das) DAs as important.
      3. Apply the min-run filter for contiguity.
      4. Add 'pred_reranked' field to each row.

    Parameters
    ----------
    pred_rows : list of DA-level prediction dicts from run_predictions.
        Each dict must have: filename, position, label, vote_frac.
    base_rate : empirical fraction of important DAs from training set.
    min_important_run : passed to min-run filter.
    min_unimportant_run : passed to min-run filter.
    filter_order : passed to min-run filter.

    Returns
    -------
    updated_rows : pred_rows with 'pred_reranked' column added.
    metrics : aggregate evaluation metrics for the re-ranked predictions,
              prefixed with 'reranked_' for easy comparison in metrics JSON.
    """
    # Group rows by transcript
    by_transcript: dict[str, list[dict]] = defaultdict(list)
    for row in pred_rows:
        by_transcript[row["filename"]].append(row)

    y_true_all: list[int] = []
    y_pred_all: list[int] = []

    for fname, rows in by_transcript.items():
        # Sort by position to ensure order
        rows.sort(key=lambda r: r["position"])
        n = len(rows)

        # Compute how many DAs to mark as important
        n_important = math.ceil(base_rate * n)
        n_important = max(0, min(n_important, n))

        # Rank by vote_frac descending — ties broken by position (earlier wins)
        ranked = sorted(
            range(n),
            key=lambda i: (-rows[i]["vote_frac"], rows[i]["position"]),
        )
        important_positions = set(ranked[:n_important])

        # Raw re-ranked predictions
        raw_reranked = [
            1 if i in important_positions else 0
            for i in range(n)
        ]

        # Apply min-run filter
        final_reranked = _apply_min_run_filter(
            raw_reranked, min_important_run, min_unimportant_run, filter_order
        )

        n_raw  = sum(raw_reranked)
        n_post = sum(final_reranked)
        n_true = sum(r["label"] for r in rows)
        print(f"  [{fname}] base-rate cap: {n_important}/{n} DAs  "
              f"raw={n_raw}  after_filter={n_post}  true={n_true}",
              flush=True)

        for i, row in enumerate(rows):
            row["pred_reranked"] = final_reranked[i]
            y_true_all.append(row["label"])
            y_pred_all.append(final_reranked[i])

    # Aggregate metrics
    if not y_true_all:
        return pred_rows, {}

    metrics_raw = _evaluate_silent(y_true_all, y_pred_all)
    metrics     = {f"reranked_{k}": v for k, v in metrics_raw.items()}

    print(f"\n  Base-rate re-ranking results:", flush=True)
    print(f"    base_rate={base_rate:.4f}  "
          f"filter_order={filter_order}", flush=True)
    print(f"    F1(important): {metrics_raw['f1_important']:.4f}  "
          f"precision: {metrics_raw['precision']:.4f}  "
          f"recall: {metrics_raw['sensitivity']:.4f}", flush=True)
    print(f"    F1(balanced):  {metrics_raw['f1_balanced']:.4f}", flush=True)
    print(f"\n{classification_report(y_true_all, y_pred_all, labels=[0, 1], target_names=['not_important', 'important'], zero_division=0)}",
          flush=True)

    return pred_rows, metrics


# ── convenience: evaluate and print comparison ────────────────────────────────

def print_comparison(
    original_metrics:  dict,
    reranked_metrics:  dict,
    base_rate:         float,
) -> None:
    """
    Print a side-by-side comparison of original vs re-ranked metrics.
    """
    def _get(d: dict, key: str, prefix: str = "") -> str:
        v = d.get(f"{prefix}{key}", d.get(key))
        return f"{v:.4f}" if isinstance(v, float) else str(v)

    print(f"\n{'='*60}", flush=True)
    print(f"  COMPARISON: original threshold vs base-rate re-ranking",
          flush=True)
    print(f"  training base rate: {base_rate*100:.1f}%", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"  {'Metric':<22} {'Original':>10} {'Re-ranked':>10}", flush=True)
    print(f"  {'─'*44}", flush=True)

    for key in ("f1_important", "f1_balanced", "precision",
                "sensitivity", "specificity"):
        orig = _get(original_metrics, key)
        rerank = _get(reranked_metrics, key, prefix="reranked_")
        print(f"  {key:<22} {orig:>10} {rerank:>10}", flush=True)

    for key in ("TP", "TN", "FP", "FN"):
        orig   = str(original_metrics.get(key, "?"))
        rerank = str(reranked_metrics.get(f"reranked_{key}", "?"))
        print(f"  {key:<22} {orig:>10} {rerank:>10}", flush=True)

    print(f"{'='*60}", flush=True)