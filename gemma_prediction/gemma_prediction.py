"""
llm_importance.py

Predicts per-DA importance (patient or therapist) using a local LLM via an
OpenAI-compatible API (LM Studio, Ollama, vLLM, llama.cpp server, etc.).

Each DA is classified by presenting the LLM with a context window of
preceding and following DAs alongside the spoken text, asking it to classify
the centre DA as important or not important.

Positive examples from the training split are included in the prompt to
guide the LLM toward the correct label format and style.

Patient split
-------------
Patient ID is parsed from the filename as the two digits following "AC"
(e.g. "randomAC01_session.csv"-> patient "01").
A fixed train/test split is performed: 5 patients for training (positive
example pool), 15 for testing.  Split is seeded for reproducibility.

Context window
--------------
For each target DA at position i, the prompt includes:
  - context_window DAs before (or [START OF TRANSCRIPT] padding)
  - the target DA (marked clearly)
  - context_window DAs after (or [END OF TRANSCRIPT] padding)

Each DA in the window is shown as:
  Speaker: DA_type - "spoken text"

Usage
-----
python llm_importance.py \\
    --dir /path/to/csv_dir \\
    --target patient \\
    --text_col text \\
    --server_url http://localhost:1234/v1 \\
    --model_path /models/lmstudio-community/gemma-3-12b-it \\
    --outdir llm_output/

Requires: openai, pandas, scikit-learn
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
import random
import re
import datetime
import logging
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
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
random.seed(SEED)
np.random.seed(SEED)

logger = logging.getLogger(__name__)

START_TOKEN = "[START OF TRANSCRIPT]"
END_TOKEN = "[END OF TRANSCRIPT]"


def parse_patient_id(filename: str) -> str | None:
    """
    Extract patient ID from filename.
    Finds the two digits immediately following 'AC'.
    E.g. 'randomAC01_session.csv' -> '01', 'AC12_data.csv' -> '12'.
    Returns None if no match.
    """
    match = re.search(r"AC(\d{2})", Path(filename).stem, re.IGNORECASE)
    return match.group(1) if match else None


def _normalise_speaker(raw: str) -> str:
    if isinstance(raw, str) and raw.strip().lower() == "therapist":
        return "therapist"
    return "patient"

def load_transcripts(
    dir_path: Path,
    target: str,
    granularity: str,
    text_col: str,
) -> dict[str, dict]:
    """
    Load all transcripts.  Returns:
    {
      filename: {
        patient_id: str,
        df: pd.DataFrame,
        target_col: str,
        das: list[str],
        texts: list[str],
        speakers: list[str],
        labels: list[int],
      }
    }
    """
    target_col = f"{target}_important"
    allowed_ext = {".csv", ".tsv", ".xlsx"}
    transcripts = {}

    for fp in sorted(dir_path.iterdir()):
        if fp.suffix.lower() not in allowed_ext:
            continue

        patient_id = parse_patient_id(fp.name)
        if patient_id is None:
            print(f"Warning: could not parse patient ID from {fp.name} — skipping.",
                  flush=True)
            continue

        print(f"Loading {fp.name}  (patient={patient_id}) …", flush=True)
        df = load_da_level(fp)
        df = df[df[DA_COLUMN] != "I-"].reset_index(drop=True)

        if target_col not in df.columns:
            print(f"Warning: '{target_col}' not found — skipping.", flush=True)
            continue

        if text_col not in df.columns:
            print(f"Warning: text column '{text_col}' not found — skipping.",
                  flush=True)
            continue

        df[target_col] = df[target_col].fillna(0).astype(int)
        n_pos = int(df[target_col].sum())
        n_tot = len(df)
        print(f"{n_tot} DAs  {n_pos} important "
              f"({100*n_pos/max(n_tot,1):.1f}%)", flush=True)

        das = [get_label(row[DA_COLUMN], row["da_group"], granularity)
                    for _, row in df.iterrows()]
        texts = [str(row.get(text_col, "")).strip() for _, row in df.iterrows()]
        speakers = [_normalise_speaker(str(row.get("speaker", "patient")))
                    for _, row in df.iterrows()]
        labels = df[target_col].tolist()

        transcripts[fp.name] = {
            "patient_id": patient_id,
            "df": df,
            "target_col": target_col,
            "das": das,
            "texts": texts,
            "speakers": speakers,
            "labels": labels,
        }

    return transcripts


def split_by_patient(
    transcripts: dict[str, dict],
    n_train_patients: int = 5,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    Split transcripts into train and test by patient ID.
    n_train_patients patients are randomly selected for training;
    all others go to test.  Seeded for reproducibility.
    """
    patient_to_files: dict[str, list[str]] = defaultdict(list)
    for fname, rec in transcripts.items():
        patient_to_files[rec["patient_id"]].append(fname)

    all_patients = sorted(patient_to_files.keys())
    rng = random.Random(SEED)
    train_patients = set(rng.sample(all_patients, min(n_train_patients,
                                                       len(all_patients))))
    test_patients = set(all_patients) - train_patients

    train = {f: transcripts[f] for p in train_patients
             for f in patient_to_files[p]}
    test = {f: transcripts[f] for p in test_patients
             for f in patient_to_files[p]}

    print(f"\n  Train patients: {sorted(train_patients)}  "
          f"({len(train)} transcripts)", flush=True)
    print(f"Test  patients: {sorted(test_patients)}  "
          f"({len(test)} transcripts)", flush=True)

    return train, test


def format_da_line(
    speaker: str,
    da: str,
    text: str,
    marker: str = "",
) -> str:
    """Format a single DA line for the prompt."""
    prefix = f"{marker} "if marker else ""
    return f'{prefix}{speaker.capitalize()}: [{da}] "{text}"'


def build_context_window(
    rec: dict,
    position: int,
    context_window: int,
) -> str:
    """
    Build the context window string for a target DA at `position`.

    Format:
      [Context]
        Speaker: [DA] "text"
        ...
      [TARGET - classify this DA]
        Speaker: [DA] "text"
      [Context]
        Speaker: [DA] "text"
        ...
    """
    das = rec["das"]
    texts = rec["texts"]
    speakers = rec["speakers"]
    n = len(das)

    lines = []

    # Preceding context
    lines.append("[Preceding context]")
    for offset in range(-context_window, 0):
        idx = position + offset
        if idx < 0:
            lines.append(f"{START_TOKEN}")
        else:
            lines.append(format_da_line(speakers[idx], das[idx], texts[idx]))

    # Target DA
    lines.append("[TARGET DA — classify this one]")
    lines.append(format_da_line(
        speakers[position], das[position], texts[position], marker=">>>"
    ))

    # Following context
    lines.append("[Following context]")
    for offset in range(1, context_window + 1):
        idx = position + offset
        if idx >= n:
            lines.append(f"{END_TOKEN}")
        else:
            lines.append(format_da_line(speakers[idx], das[idx], texts[idx]))

    return "\n".join(lines)


def build_positive_examples(
    train_transcripts: dict[str, dict],
    context_window: int,
    n_examples: int = 16,
) -> list[dict]:
    """
    Collect positive (important=1) examples from training transcripts.
    Returns a list of {context_str, label} dicts.
    """
    positives = []

    for rec in train_transcripts.values():
        for i, lbl in enumerate(rec["labels"]):
            if lbl == 1:
                ctx = build_context_window(rec, i, context_window)
                positives.append({"context": ctx, "label": 1})

    rng = random.Random(SEED)
    rng.shuffle(positives)
    return positives[:n_examples]


def construct_prompt(
    context_str: str,
    positive_examples: list[dict],
    n_few_shot: int = 8,
) -> str:
    """
    Construct the full few-shot prompt for one target DA.

    Shows n_few_shot positive examples followed by the target DA window.
    The model is asked to classify only the TARGET DA.
    """
    examples = positive_examples[:n_few_shot]

    prompt_lines = []

    if examples:
        prompt_lines.append(
            "Below are examples of dialogue act sequences where the TARGET DA "
            "is important:\n"
        )
        for i, ex in enumerate(examples):
            prompt_lines.append(f"Example {i+1}:")
            prompt_lines.append(ex["context"])
            prompt_lines.append("Classification: important\n")

    prompt_lines.append(
        "Now classify the TARGET DA in the following sequence as either "
        "'important' or 'not important'.\n"
        "Only classify the TARGET DA and text marked with >>>. "
        "Answer with exactly one of: 'important' or 'not important'.\n"
    )
    prompt_lines.append(context_str)
    prompt_lines.append("\nClassification:")

    return "\n".join(prompt_lines)


def generate_prediction(
    prompt: str,
    system: str,
    server_url: str,
    model_path: str,
    temperature: float = 0.0,
    max_tokens: int = 5,
    retry_note: str = "",
) -> str:
    """
    Call the local LLM server and return the raw response string.
    retry_note is appended to the prompt on retry attempts to remind
    the model of the required output format.
    """
    from openai import OpenAI
    client = OpenAI(base_url=server_url, api_key="lm-studio")
    full_prompt = prompt if not retry_note else f"{prompt}{retry_note}"
    completion = client.chat.completions.create(
        model=model_path,
        messages=[
            {"role": "system",  "content": system},
            {"role": "user",    "content": full_prompt},
        ],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=0.95,
    )
    response = completion.choices[0].message.content
    logger.debug(f"Response: {response}")
    return response



def parse_prediction(response: str) -> int:
    """Parse LLM response to binary label. 0=not important, 1=important."""
    r = response.lower().strip()
    if "not important"in r:
        return 0
    if "important"in r:
        return 1
    # Fallback: default to not important if unparseable
    logger.warning(f"Unparseable response: '{response}' — defaulting to 0")
    return 0


SYSTEM_PROMPT = (
    "You are an expert behavioral psychologist analysing therapy session "
    "transcripts. You are given a sequence of dialogue acts (DAs) from a "
    "therapy session, each with the speaker, DA type, and spoken text. "
    "Your task is to classify whether the TARGET DA and accompanying text (marked with >>>) is "
    "important or not important in the context of the therapy session. "
    "Answer with exactly one of: 'important' or 'not important'. "
    "Do not explain your answer."
)

def evaluate(y_true: list[int], y_pred: list[int]) -> dict:
    """Compute and print evaluation metrics."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TP = int(cm[1, 1])
    TN = int(cm[0, 0])
    FP = int(cm[0, 1])
    FN = int(cm[1, 0])

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    precision = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1_imp = 2 * precision * sensitivity / (precision + sensitivity) \
                  if (precision + sensitivity) > 0 else 0.0
    f1_bal = f1_score(y_true, y_pred, average="macro", zero_division=0)
    f1_wt = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*60}", flush=True)
    print("EVALUATION RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"TP={TP}  TN={TN}  FP={FP}  FN={FN}", flush=True)
    print(f"Sensitivity (recall): {sensitivity:.4f}", flush=True)
    print(f"Specificity: {specificity:.4f}", flush=True)
    print(f"Precision: {precision:.4f}", flush=True)
    print(f"F1(important): {f1_imp:.4f}", flush=True)
    print(f"F1(balanced/macro): {f1_bal:.4f}", flush=True)
    print(f"F1(weighted): {f1_wt:.4f}", flush=True)
    print(f"\n{classification_report(y_true, y_pred, labels=[0,1], target_names=['not_important','important'], zero_division=0)}",
          flush=True)

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "sensitivity": round(sensitivity, 4),
        "specificity": round(specificity, 4),
        "precision": round(precision,   4),
        "f1_important": round(f1_imp,      4),
        "f1_balanced": round(f1_bal,      4),
        "f1_weighted": round(f1_wt,       4),
    }


def run_predictions(
    test_transcripts: dict[str, dict],
    positive_examples: list[dict],
    server_url: str,
    model_path: str,
    context_window: int,
    n_few_shot: int,
    temperature: float,
    max_tokens: int,
    max_retries: int,
    verbose: bool,
) -> tuple[list[int], list[int], list[dict]]:
    """
    Run LLM predictions over all test transcripts.
    Returns (y_true, y_pred, prediction_rows).
    If a response cannot be parsed it is retried up to max_retries times,
    with a format reminder appended to the prompt on each retry.
    Defaults to 0 (not important) if all retries are exhausted.
    """
    y_true_all: list[int] = []
    y_pred_all: list[int] = []
    pred_rows: list[dict] = []

    total_das = sum(len(rec["labels"]) for rec in test_transcripts.values())
    done = 0

    for fname, rec in test_transcripts.items():
        n = len(rec["das"])
        print(f"\n  Transcript: {fname}  ({n} DAs)", flush=True)

        for i in range(n):
            context_str = build_context_window(rec, i, context_window)
            prompt = construct_prompt(context_str, positive_examples,
                                           n_few_shot)

            pred = None
            response = ""
            attempt = 0
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
                    prompt, SYSTEM_PROMPT, server_url, model_path,
                    temperature, max_tokens,
                    retry_note=retry_note,
                )
                pred = parse_prediction(response)
                if pred is not None:
                    break
                print(f"[retry {attempt+1}/{max_retries}] "
                      f"unparseable: '{response.strip()}'", flush=True)
                logger.warning(f"{fname}[{i}] retry {attempt+1}: "
                               f"'{response.strip()}'")

            if pred is None:
                print(f"[FAILED] max retries exhausted at position {i}"
                      f"— defaulting to 0", flush=True)
                logger.error(f"{fname}[{i}] max retries exhausted, "
                             f"defaulting to 0")
                pred = 0

            label = rec["labels"][i]

            y_true_all.append(label)
            y_pred_all.append(pred)

            pred_rows.append({
                "filename": fname,
                "patient_id": rec["patient_id"],
                "position": i,
                "da": rec["das"][i],
                "speaker": rec["speakers"][i],
                "text": rec["texts"][i],
                "label": label,
                "pred": pred,
                "response": response.strip(),
                "n_retries": attempt,
            })

            done += 1
            if verbose and (done % 10 == 0 or done == total_das):
                n_pos_so_far = sum(y_true_all)
                n_pred_so_far = sum(y_pred_all)
                print(f"[{done}/{total_das}]  "
                      f"true_pos={n_pos_so_far}  "
                      f"pred_pos={n_pred_so_far}", flush=True)

            logger.debug(f"{fname}[{i}]  label={label}  pred={pred}  "
                         f"response='{response.strip()}'")

    return y_true_all, y_pred_all, pred_rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "LLM-based per-DA importance classifier with context window.\n"
            "Uses an OpenAI-compatible local server (LM Studio, Ollama, etc.)."
        )
    )

    parser.add_argument("--dir",           required=True,
                        help="Directory containing transcript CSV/TSV/XLSX files.")
    parser.add_argument("--granularity",   default="groups",
                        choices=["groups", "raw"])
    parser.add_argument("--target",        default="patient",
                        choices=["patient", "therapist"])
    parser.add_argument("--text_col",      required=True,
                        help="Column name containing spoken text for each DA.")

    parser.add_argument("--n_train_patients", type=int, default=5,
                        help="Number of patients to use for training (positive "
                             "example pool). Remaining go to test. (default: 5)")

    parser.add_argument("--context_window", type=int, default=5,
                        help="Number of DAs before and after the target DA to "
                             "include in the prompt. (default: 5)")
    parser.add_argument("--n_few_shot",     type=int, default=8,
                        help="Number of positive few-shot examples to include "
                             "in the prompt. (default: 8)")

    parser.add_argument("--server_url",  default="http://localhost:1234/v1",
                        help="OpenAI-compatible server URL. (default: "
                             "http://localhost:1234/v1)")
    parser.add_argument("--model_path",  required=True,
                        help="Model identifier as expected by the server "
                             "(e.g. /models/lmstudio-community/gemma-3-12b-it).")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM sampling temperature. 0.0 = deterministic. "
                             "(default: 0.0)")
    parser.add_argument("--max_tokens",  type=int, default=5,
                        help="Max tokens for LLM response. (default: 5)")

    parser.add_argument("--outdir",  default="llm_output/")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress every 10 DAs.")
    parser.add_argument("--log",     action="store_true",
                        help="Enable logging to a timestamped log file.")

    args = parser.parse_args()

    if args.log:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = os.path.join(args.outdir, f"run-{ts}.log")
        os.makedirs(args.outdir, exist_ok=True)
        logging.basicConfig(filename=log_path, level=logging.DEBUG)
        print(f"Logging to: {log_path}", flush=True)

    dir_path = Path(args.dir)
    if not dir_path.exists():
        raise ValueError(f"Directory not found: {args.dir}")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"\n{'='*60}", flush=True)
    print(f"LLM Importance Classifier", flush=True)
    print(f"target={args.target}  granularity={args.granularity}", flush=True)
    print(f"text_col={args.text_col}  context_window={args.context_window}",
          flush=True)
    print(f"n_few_shot={args.n_few_shot}  "
          f"n_train_patients={args.n_train_patients}  "
          f"max_retries={args.max_retries}", flush=True)
    print(f"server_url={args.server_url}", flush=True)
    print(f"model_path={args.model_path}", flush=True)
    print(f"{'='*60}", flush=True)

    transcripts = load_transcripts(
        dir_path, args.target, args.granularity, args.text_col
    )
    if len(transcripts) < 2:
        raise RuntimeError(
            f"Need ≥2 transcripts, found {len(transcripts)}."
        )
    print(f"\nLoaded {len(transcripts)} transcripts.", flush=True)

    print(f"\nSplitting by patient (train={args.n_train_patients} patients) …",
          flush=True)
    train_transcripts, test_transcripts = split_by_patient(
        transcripts, args.n_train_patients
    )
    print(f"Train: {len(train_transcripts)} transcripts  "
          f"Test: {len(test_transcripts)} transcripts", flush=True)

    print(f"\nBuilding positive example pool from training transcripts …",
          flush=True)
    positive_examples = build_positive_examples(
        train_transcripts, args.context_window, n_examples=args.n_few_shot * 4
    )
    print(f"Found {len(positive_examples)} positive examples  "
          f"(using {min(args.n_few_shot, len(positive_examples))} in prompt)",
          flush=True)

    if len(positive_examples) == 0:
        print("Warning: no positive examples found in training set. "
              "Proceeding without few-shot examples.", flush=True)

    total_test_das = sum(len(r["labels"]) for r in test_transcripts.values())
    print(f"\nRunning predictions on {len(test_transcripts)} test transcripts "
          f"({total_test_das} DAs) …", flush=True)

    y_true, y_pred, pred_rows = run_predictions(
        test_transcripts=test_transcripts,
        positive_examples=positive_examples,
        server_url=args.server_url,
        model_path=args.model_path,
        context_window=args.context_window,
        n_few_shot=args.n_few_shot,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        verbose=args.verbose,
    )

    metrics = evaluate(y_true, y_pred)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    label = (
        f"{args.target}_{args.granularity}"
        f"_ctx{args.context_window}"
        f"_fs{args.n_few_shot}"
        f"_{ts}"
    )

    pred_path = os.path.join(args.outdir, f"llm_{label}_predictions.csv")
    pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
    print(f"\n  Saved: {pred_path}", flush=True)

    metrics_path = os.path.join(args.outdir, f"llm_{label}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "label": label,
            "target": args.target,
            "granularity": args.granularity,
            "context_window": args.context_window,
            "n_few_shot": args.n_few_shot,
            "n_train_patients": args.n_train_patients,
            "model_path": args.model_path,
            "server_url": args.server_url,
            "temperature": args.temperature,
            "n_test_das": len(y_true),
            "n_test_pos": sum(y_true),
            "max_retries": args.max_retries,
            "n_retried": int(sum(1 for r in pred_rows if r["n_retries"] > 0)),
            **metrics,
        }, f, indent=2)
    print(f"Saved: {metrics_path}", flush=True)

    print(f"\nDone. Outputs in: {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
