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


def build_negative_examples(
    train_transcripts: dict[str, dict],
    context_window: int,
    n_examples: int = 16,
    max_neg_length: int = 50,
) -> list[dict]:
    """
    Collect negative (important=0) examples from training transcripts.
    Samples individual non-important DA positions, capping the context
    window text at max_neg_length DAs to avoid very long non-important
    runs dominating the prompt.
    Returns a list of {context_str, label} dicts.
    """
    negatives = []

    for rec in train_transcripts.values():
        labels = rec["labels"]
        n      = len(labels)
        for i, lbl in enumerate(labels):
            if lbl == 0:
                # Cap context window to avoid including too many DAs
                effective_window = min(context_window, max_neg_length // 2)
                ctx = build_context_window(rec, i, effective_window)
                negatives.append({"context": ctx, "label": 0})

    rng = random.Random(SEED + 1)
    rng.shuffle(negatives)
    return negatives[:n_examples]


def construct_prompt(
    context_str: str,
    positive_examples: list[dict],
    negative_examples: list[dict],
    n_few_shot: int = 8,
) -> str:
    """
    Construct the balanced few-shot prompt for one target DA.

    Shows n_few_shot//2 positive and n_few_shot//2 negative examples
    interleaved, then asks the model to classify the target DA.
    Balanced examples help prevent the model defaulting to "important".
    """
    n_pos = n_few_shot // 2
    n_neg = n_few_shot - n_pos
    pos   = positive_examples[:n_pos]
    neg   = negative_examples[:n_neg]

    # Interleave positive and negative examples
    all_examples = []
    for p, n in zip(pos, neg):
        all_examples.append(p)
        all_examples.append(n)
    # Append any remainder
    for ex in pos[len(neg):]:
        all_examples.append(ex)
    for ex in neg[len(pos):]:
        all_examples.append(ex)

    prompt_lines = []

    if all_examples:
        prompt_lines.append(
            "Below are examples of dialogue act sequences with their "
            "classifications. Study both important and not important examples "
            "carefully before classifying the target.\n"
        )
        for i, ex in enumerate(all_examples):
            label_str = "important" if ex["label"] == 1 else "not important"
            prompt_lines.append(f"Example {i+1}:")
            prompt_lines.append(ex["context"])
            prompt_lines.append(f"Classification: {label_str}\n")

    prompt_lines.append(
        "Now classify the TARGET DA in the following sequence as either "
        "'important' or 'not important'.\n"
        "Only classify the TARGET DA and text marked with >>>. "
        "Most DAs are NOT important — only classify as important if the "
        "content is clinically or therapeutically significant.\n"
        "Answer with exactly one of: 'important' or 'not important'.\n"
    )
    prompt_lines.append(context_str)
    prompt_lines.append("\nClassification:")

    return "\n".join(prompt_lines)


def load_model(model_id: str, hf_cache_dir: str | None = None):
    """
    Load Gemma3ForConditionalGeneration + AutoProcessor from HuggingFace.
    This is the correct class for google/gemma-3-4b-it and all Gemma 3
    models 4B and above, which are multimodal (vision-language) checkpoints.
    Gemma3ForCausalLM is only correct for the 1B text-only variant.

    Returns (model, processor) tuple.
    """
    import torch
    from transformers import AutoProcessor, Gemma3ForConditionalGeneration

    if hf_cache_dir:
        os.environ["HF_HOME"]            = hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
        os.environ["HF_DATASETS_CACHE"]  = hf_cache_dir
        print(f"HuggingFace cache dir: {hf_cache_dir}", flush=True)

    print(f"Loading processor: {model_id} …", flush=True)
    processor = AutoProcessor.from_pretrained(model_id)

    print(f"Loading model: {model_id} …", flush=True)
    model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id,
        device_map="auto",
    ).eval()

    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}", flush=True)

    return model, processor


def generate_prediction(
    prompt: str,
    system: str,
    model_and_tokenizer: tuple,
    temperature: float = 0.0,
    max_tokens: int = 10,
    retry_note: str = "",
) -> str:
    """
    Run inference using Gemma3ForConditionalGeneration + AutoProcessor.
    retry_note is appended to the prompt on retry attempts.
    """
    import torch

    model, processor = model_and_tokenizer
    full_prompt = prompt if not retry_note else f"{prompt}{retry_note}"

    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": system}],
        },
        {
            "role": "user",
            "content": [{"type": "text", "text": full_prompt}],
        },
    ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]

    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=temperature > 0.0,
    )
    if temperature > 0.0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"] = 0.95

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    # Decode only the newly generated tokens
    new_tokens = output_ids[0][input_len:]
    response   = processor.decode(new_tokens, skip_special_tokens=True).strip()

    logger.debug(f"Response: {response}")
    return response

def parse_prediction(response: str) -> int | None:
    """
    Parse LLM response to binary label. 0=not important, 1=important.
    Returns None if unparseable so caller can retry.

    Checks for "not important" before "important" to avoid the substring
    match trap, and anchors to the start of the response since the model
    should answer with just the label.
    """
    r = response.lower().strip()
    # Check "not important" first — must come before "important" check
    # since "important" is a substring of "not important"
    if re.search(r"\bnot important\b", r):
        return 0
    if re.search(r"\bimportant\b", r):
        return 1
    logger.warning(f"Unparseable response: '{response}'")
    return None


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
    negative_examples: list[dict],
    model_and_tokenizer: tuple,
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
                                           negative_examples, n_few_shot)

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
                    prompt, SYSTEM_PROMPT, model_and_tokenizer,
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
            "Loads the model in-process via HuggingFace transformers.\n"
            "Model is downloaded automatically on first run and cached locally."
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
                        help="Number of few-shot examples to include in the "
                             "prompt total, split 50/50 positive/negative. "
                             "(default: 8)")
    parser.add_argument("--max_neg_length", type=int, default=50,
                        help="Maximum number of DAs to include in a negative "
                             "example context window. Caps long non-important "
                             "runs. (default: 50)")

    parser.add_argument("--model_id",    default="google/gemma-3-4b-it",
                        help="HuggingFace model ID. Downloaded automatically "
                             "on first run and cached locally. "
                             "(default: google/gemma-3-4b-it)")
    parser.add_argument("--hf_cache_dir", default=None,
                        help="Path to HuggingFace cache directory. Useful on "
                             "HPC to avoid filling home quota. If not set, "
                             "uses the default HuggingFace cache (~/.cache/huggingface).")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM sampling temperature. 0.0 = deterministic. "
                             "(default: 0.0)")
    parser.add_argument("--max_tokens",  type=int, default=5,
                        help="Max tokens for LLM response. (default: 5)")
    parser.add_argument("--max_retries",  type=int, default=3,
                        help="Max retries for LLM response fails. (default: 3)")

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

    print(f"LLM Importance Classifier", flush=True)
    print(f"target={args.target}  granularity={args.granularity}", flush=True)
    print(f"text_col={args.text_col}  context_window={args.context_window}",
          flush=True)
    print(f"n_few_shot={args.n_few_shot}  "
          f"n_train_patients={args.n_train_patients}  "
          f"max_retries={args.max_retries}", flush=True)
    print(f"model_id={args.model_id}", flush=True)
    print(f"hf_cache_dir={args.hf_cache_dir}", flush=True)

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
    n_each = args.n_few_shot * 4
    positive_examples = build_positive_examples(
        train_transcripts, args.context_window, n_examples=n_each
    )
    negative_examples = build_negative_examples(
        train_transcripts, args.context_window,
        n_examples=n_each, max_neg_length=args.max_neg_length
    )
    n_pos_used = min(args.n_few_shot // 2, len(positive_examples))
    n_neg_used = min(args.n_few_shot - n_pos_used, len(negative_examples))
    print(f"Found {len(positive_examples)} positive examples  "
          f"(using {n_pos_used} in prompt)", flush=True)
    print(f"Found {len(negative_examples)} negative examples  "
          f"(using {n_neg_used} in prompt)", flush=True)

    if len(positive_examples) == 0:
        print("Warning: no positive examples found in training set. "
              "Proceeding without few-shot examples.", flush=True)
    if len(negative_examples) == 0:
        print("Warning: no negative examples found in training set.", flush=True)

    print(f"\nLoading model {args.model_id} …", flush=True)
    model_and_tokenizer = load_model(args.model_id, args.hf_cache_dir)
    print(f"Model ready.", flush=True)


    total_test_das = sum(len(r["labels"]) for r in test_transcripts.values())
    print(f"\nRunning predictions on {len(test_transcripts)} test transcripts "
          f"({total_test_das} DAs) …", flush=True)

    y_true, y_pred, pred_rows = run_predictions(
        test_transcripts=test_transcripts,
        positive_examples=positive_examples,
        negative_examples=negative_examples,
        model_and_tokenizer=model_and_tokenizer,
        context_window=args.context_window,
        n_few_shot=args.n_few_shot,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
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
            "model_id": args.model_id,
            "temperature": args.temperature,
            "max_neg_length": args.max_neg_length,
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
