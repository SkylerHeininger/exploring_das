import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import csv
import datetime
import itertools
import json
import os
import random
import re
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, confusion_matrix, classification_report

from plotting.common_patterns import DA_COLUMN, load_da_level, get_label

# Import all helpers from the sliding window script
# (assumed to be in the same directory)
from gemma_prediction.slide_gemma4 import (
    parse_patient_id,
    _normalise_speaker,
    load_transcripts,
    split_by_patient,
    build_examples_from_transcripts,
    load_codebook,
    compute_base_rate,
    build_system_prompt,
    construct_prompt,
    apply_min_run_filter,
    load_model,
    generate_prediction,
    parse_prediction,
    build_window,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# ── fixed params shared across all runs ──────────────────────────────────────

FIXED = dict(
    window_stride      = 6,
    vote_threshold     = 0.75,
    min_important_run  = 10,
    min_unimportant_run= 1,
    max_context_das    = 5,
    temperature        = 0.0,
    max_tokens         = 10,
    max_input_tokens   = 12000,
    max_retries        = 3,
)


# ── grid ─────────────────────────────────────────────────────────────────────

GRID = {
    "window_size":    [15, 20, 30],
    "n_few_shot":     [8, 16],
    "pos_proportion": [0.5, 0.75],
}


# ── summary CSV helpers ───────────────────────────────────────────────────────

SUMMARY_COLS = [
    "run_id", "run_type",
    "window_size", "n_few_shot", "pos_proportion",
    "use_codebook", "use_examples",
    "f1_important", "f1_balanced", "f1_weighted",
    "TP", "TN", "FP", "FN",
    "sensitivity", "specificity", "precision",
    "n_test_das", "n_test_pos",
    "timestamp",
]


def _ensure_summary(summary_path: str) -> None:
    """Create the summary CSV with headers if it doesn't exist yet."""
    if not os.path.exists(summary_path):
        with open(summary_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=SUMMARY_COLS)
            writer.writeheader()
        print(f"  Created summary: {summary_path}", flush=True)


def _append_summary(summary_path: str, row: dict) -> None:
    """Append one result row to the summary CSV immediately."""
    with open(summary_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=SUMMARY_COLS, extrasaction="ignore")
        writer.writerow(row)
    print(f"  Summary updated: {summary_path}", flush=True)


def _rank_summary(summary_path: str) -> None:
    """Re-sort the summary CSV by f1_important descending and overwrite."""
    df = pd.read_csv(summary_path)
    df = df.sort_values("f1_important", ascending=False).reset_index(drop=True)
    df.to_csv(summary_path, index=False)
    print(f"\n  Summary ranked by f1_important: {summary_path}", flush=True)
    print(df[["run_id", "run_type", "window_size", "n_few_shot",
              "pos_proportion", "use_codebook", "use_examples",
              "f1_important", "f1_balanced"]].to_string(index=False),
          flush=True)


# ── single-run prediction ─────────────────────────────────────────────────────

def run_single(
    test_transcripts:    dict[str, dict],
    pos_examples:        list[str],
    neg_examples:        list[str],
    model_and_processor: tuple,
    system_prompt:       str,
    window_size:         int,
    run_id:              str,
    outdir:              str,
) -> dict:
    """
    Run predictions for one configuration on the fixed test set.
    Returns a metrics dict.
    """
    window_stride       = FIXED["window_stride"]
    vote_threshold      = FIXED["vote_threshold"]
    min_important_run   = FIXED["min_important_run"]
    min_unimportant_run = FIXED["min_unimportant_run"]
    temperature         = FIXED["temperature"]
    max_tokens          = FIXED["max_tokens"]
    max_retries         = FIXED["max_retries"]
    max_input_tokens    = FIXED["max_input_tokens"]

    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    pred_rows:  list[dict] = []

    for fname, rec in test_transcripts.items():
        n = len(rec["das"])
        print(f"    Transcript: {fname}  ({n} DAs)", flush=True)

        vote_counts: list[int] = [0] * n
        vote_totals: list[int] = [0] * n

        starts = list(range(0, n, window_stride))
        if starts and starts[-1] + window_size < n:
            starts.append(n - window_size)

        n_windows = len(starts)
        print(f"      {n_windows} windows", flush=True)

        for w_idx, start in enumerate(starts):
            window_text = build_window(rec, start, window_size)
            prompt      = construct_prompt(
                pos_examples, neg_examples, window_text,
                processor=model_and_processor[1],
                max_input_tokens=max_input_tokens,
            )

            pred     = None
            response = ""
            attempt  = 0
            for attempt in range(max_retries + 1):
                retry_note = (
                    ""
                    if attempt == 0
                    else (
                        "\n\nIMPORTANT: Your previous response could not be "
                        "parsed. You must answer with exactly one of: "
                        "'important' or 'not important'. Nothing else."
                    )
                )
                response = generate_prediction(
                    prompt, system_prompt, model_and_processor,
                    temperature, max_tokens,
                    retry_note=retry_note,
                    max_input_tokens=max_input_tokens,
                )
                pred = parse_prediction(response)
                if pred is not None:
                    break
                print(f"      [retry {attempt+1}/{max_retries}] "
                      f"window {w_idx} unparseable: '{response.strip()}'",
                      flush=True)

            if pred is None:
                print(f"      [FAILED] window {w_idx} defaulting to 0", flush=True)
                pred = 0

            actual_end = min(start + window_size, n)
            for da_idx in range(start, actual_end):
                vote_counts[da_idx] += pred
                vote_totals[da_idx] += 1

        raw_preds = [
            1 if (vote_totals[i] > 0 and
                  vote_counts[i] / vote_totals[i] >= vote_threshold)
            else 0
            for i in range(n)
        ]
        final_preds = apply_min_run_filter(
            raw_preds, min_important_run, min_unimportant_run
        )

        transcript_true_pos = sum(rec["labels"])
        transcript_pred_pos = sum(final_preds)
        print(f"      true_pos={transcript_true_pos}  "
              f"pred_pos={transcript_pred_pos}", flush=True)

        for i in range(n):
            y_true_all.append(rec["labels"][i])
            y_pred_all.append(final_preds[i])
            pred_rows.append({
                "run_id":     run_id,
                "filename":   fname,
                "patient_id": rec["patient_id"],
                "position":   i,
                "da":         rec["das"][i],
                "speaker":    rec["speakers"][i],
                "text":       rec["texts"][i],
                "label":      rec["labels"][i],
                "pred":       final_preds[i],
                "pred_raw":   raw_preds[i],
                "vote_count": vote_counts[i],
                "vote_total": vote_totals[i],
                "vote_frac":  round(vote_counts[i] / max(vote_totals[i], 1), 4),
            })

    # Save per-run predictions
    pred_path = os.path.join(outdir, f"{run_id}_predictions.csv")
    pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
    print(f"      Saved: {pred_path}", flush=True)

    # Compute metrics
    cm  = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1])
    TP  = int(cm[1, 1]);  TN = int(cm[0, 0])
    FP  = int(cm[0, 1]);  FN = int(cm[1, 0])

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    precision   = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1_imp      = f1_score(y_true_all, y_pred_all, pos_label=1,
                           average="binary", zero_division=0)
    f1_bal      = f1_score(y_true_all, y_pred_all,
                           average="macro", zero_division=0)
    f1_wt       = f1_score(y_true_all, y_pred_all,
                           average="weighted", zero_division=0)

    print(f"    F1(imp)={f1_imp:.4f}  F1(bal)={f1_bal:.4f}  "
          f"TP={TP}  FP={FP}  FN={FN}", flush=True)

    return {
        "f1_important": round(f1_imp, 4),
        "f1_balanced":  round(f1_bal, 4),
        "f1_weighted":  round(f1_wt,  4),
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "sensitivity":  round(sensitivity, 4),
        "specificity":  round(specificity, 4),
        "precision":    round(precision,   4),
        "n_test_das":   len(y_true_all),
        "n_test_pos":   sum(y_true_all),
    }


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Grid search + ablation for the sliding-window LLM importance classifier.\n"
            "Sweeps window_size, n_few_shot, pos_proportion.\n"
            "After the sweep, runs ablations on the best config."
        )
    )

    parser.add_argument("--dir",           required=True)
    parser.add_argument("--granularity",   default="groups",
                        choices=["groups", "raw"])
    parser.add_argument("--target",        default="therapist",
                        choices=["patient", "therapist"])
    parser.add_argument("--text_col",      required=True)

    parser.add_argument("--codebook",      default="codebook.xlsx")
    parser.add_argument("--n_train_patients", type=int, default=5)
    parser.add_argument("--n_test_transcripts", type=int, default=5,
                        help="Number of test transcripts to use. Randomly "
                             "sampled once with SEED. (default: 5)")

    parser.add_argument("--model_id",    default="google/gemma-4-E4B-it")
    parser.add_argument("--hf_cache_dir", default=None)
    parser.add_argument("--outdir",      default="gridsearch_output/")

    args = parser.parse_args()

    dir_path = Path(args.dir)
    if not dir_path.exists():
        raise ValueError(f"Directory not found: {args.dir}")
    os.makedirs(args.outdir, exist_ok=True)

    summary_path = os.path.join(args.outdir, "grid_summary.csv")
    _ensure_summary(summary_path)

    print(f"Grid Search + Ablation", flush=True)
    print(f"target={args.target}  granularity={args.granularity}", flush=True)
    print(f"n_train_patients={args.n_train_patients}  "
          f"n_test_transcripts={args.n_test_transcripts}", flush=True)
    print(f"model_id={args.model_id}", flush=True)
    print(f"Grid: {GRID}", flush=True)

    # ── load all transcripts ──────────────────────────────────────────────────
    transcripts = load_transcripts(
        dir_path, args.target, args.granularity, args.text_col
    )
    print(f"\nLoaded {len(transcripts)} transcripts.", flush=True)

    # ── patient split ─────────────────────────────────────────────────────────
    print(f"\nSplitting by patient …", flush=True)
    train_transcripts, test_pool = split_by_patient(
        transcripts, args.n_train_patients
    )

    # ── select fixed test subset ──────────────────────────────────────────────
    test_fnames = sorted(test_pool.keys())
    rng_test    = random.Random(SEED)
    rng_test.shuffle(test_fnames)
    test_fnames = test_fnames[:args.n_test_transcripts]
    test_transcripts = {f: test_pool[f] for f in test_fnames}

    print(f"\nFixed test set ({len(test_transcripts)} transcripts):", flush=True)
    for f in test_fnames:
        n_pos = sum(test_transcripts[f]["labels"])
        n_tot = len(test_transcripts[f]["labels"])
        print(f"  {f}  ({n_tot} DAs  {n_pos} important)", flush=True)

    # ── codebook + base rate ──────────────────────────────────────────────────
    print(f"\nLoading codebook …", flush=True)
    codebook  = load_codebook(args.codebook) if args.codebook else {}
    base_rate = compute_base_rate(train_transcripts)
    print(f"  Base rate: {base_rate*100:.1f}%  "
          f"Codes loaded: {len(codebook)}", flush=True)

    # ── load model once ───────────────────────────────────────────────────────
    print(f"\nLoading model {args.model_id} …", flush=True)
    model_and_processor = load_model(args.model_id, args.hf_cache_dir)
    print(f"Model ready.", flush=True)

    # ── PHASE 1: grid search ──────────────────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"  PHASE 1: Grid Search", flush=True)
    print(f"{'='*60}", flush=True)

    grid_keys   = list(GRID.keys())
    grid_values = list(GRID.values())
    combos      = list(itertools.product(*grid_values))
    n_combos    = len(combos)
    print(f"  {n_combos} configurations to evaluate", flush=True)

    # Build system prompt with codebook (used for all grid configs)
    system_prompt_full = build_system_prompt(codebook, base_rate)

    best_f1     = -1.0
    best_config = {}

    for combo_idx, combo in enumerate(combos):
        config = dict(zip(grid_keys, combo))
        run_id = (f"grid_{combo_idx+1:03d}"
                  f"_ws{config['window_size']}"
                  f"_fs{config['n_few_shot']}"
                  f"_pp{int(config['pos_proportion']*100)}")

        print(f"\n{'─'*60}", flush=True)
        print(f"  [{combo_idx+1}/{n_combos}]  {run_id}", flush=True)
        print(f"  {config}", flush=True)

        # Build examples for this config
        rng_ex = random.Random(SEED)
        pos_ex, neg_ex = build_examples_from_transcripts(
            train_transcripts,
            window_size=config["window_size"],
            n_few_shot=config["n_few_shot"],
            rng=rng_ex,
            pos_proportion=config["pos_proportion"],
            max_context_das=FIXED["max_context_das"],
        )

        metrics = run_single(
            test_transcripts=test_transcripts,
            pos_examples=pos_ex,
            neg_examples=neg_ex,
            model_and_processor=model_and_processor,
            system_prompt=system_prompt_full,
            window_size=config["window_size"],
            run_id=run_id,
            outdir=args.outdir,
        )

        row = {
            "run_id":          run_id,
            "run_type":        "grid",
            "window_size":     config["window_size"],
            "n_few_shot":      config["n_few_shot"],
            "pos_proportion":  config["pos_proportion"],
            "use_codebook":    bool(codebook),
            "use_examples":    True,
            "timestamp":       datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            **metrics,
        }
        _append_summary(summary_path, row)

        if metrics["f1_important"] > best_f1:
            best_f1     = metrics["f1_important"]
            best_config = config.copy()
            print(f"  *** New best F1(imp)={best_f1:.4f} ***", flush=True)

    print(f"\n  Grid search complete.  Best config: {best_config}  "
          f"F1={best_f1:.4f}", flush=True)

    # ── PHASE 2: ablations on best config ────────────────────────────────────
    print(f"\n{'='*60}", flush=True)
    print(f"  PHASE 2: Ablations (best config: {best_config})", flush=True)
    print(f"{'='*60}", flush=True)

    ablations = [
        # (label, use_codebook, use_examples)
        ("no_codebook",          False, True),
        ("no_examples",          True,  False),
        ("no_codebook_no_examples", False, False),
    ]

    for abl_label, use_codebook, use_examples in ablations:
        run_id = (f"ablation_{abl_label}"
                  f"_ws{best_config['window_size']}"
                  f"_fs{best_config['n_few_shot']}"
                  f"_pp{int(best_config['pos_proportion']*100)}")

        print(f"\n{'─'*60}", flush=True)
        print(f"  Ablation: {abl_label}", flush=True)
        print(f"  use_codebook={use_codebook}  use_examples={use_examples}",
              flush=True)

        abl_codebook = codebook if use_codebook else {}
        abl_system   = build_system_prompt(abl_codebook, base_rate)

        if use_examples:
            rng_ex = random.Random(SEED)
            pos_ex, neg_ex = build_examples_from_transcripts(
                train_transcripts,
                window_size=best_config["window_size"],
                n_few_shot=best_config["n_few_shot"],
                rng=rng_ex,
                pos_proportion=best_config["pos_proportion"],
                max_context_das=FIXED["max_context_das"],
            )
        else:
            pos_ex, neg_ex = [], []

        metrics = run_single(
            test_transcripts=test_transcripts,
            pos_examples=pos_ex,
            neg_examples=neg_ex,
            model_and_processor=model_and_processor,
            system_prompt=abl_system,
            window_size=best_config["window_size"],
            run_id=run_id,
            outdir=args.outdir,
        )

        row = {
            "run_id":         run_id,
            "run_type":       f"ablation_{abl_label}",
            "window_size":    best_config["window_size"],
            "n_few_shot":     best_config["n_few_shot"] if use_examples else 0,
            "pos_proportion": best_config["pos_proportion"] if use_examples else 0,
            "use_codebook":   use_codebook,
            "use_examples":   use_examples,
            "timestamp":      datetime.datetime.now().strftime("%Y%m%d-%H%M%S"),
            **metrics,
        }
        _append_summary(summary_path, row)

    # ── final ranked summary ──────────────────────────────────────────────────
    _rank_summary(summary_path)

    # Save run metadata
    meta_path = os.path.join(args.outdir, "run_metadata.json")
    with open(meta_path, "w") as f:
        json.dump({
            "target":               args.target,
            "granularity":          args.granularity,
            "model_id":             args.model_id,
            "codebook":             args.codebook,
            "n_train_patients":     args.n_train_patients,
            "n_test_transcripts":   args.n_test_transcripts,
            "test_transcripts":     test_fnames,
            "base_rate_train":      round(base_rate, 4),
            "fixed_params":         FIXED,
            "grid":                 GRID,
            "best_grid_config":     best_config,
            "best_grid_f1":         round(best_f1, 4),
        }, f, indent=2)
    print(f"\n  Saved metadata: {meta_path}", flush=True)
    print(f"\nDone. Outputs in: {args.outdir}", flush=True)


if __name__ == "__main__":
    main()