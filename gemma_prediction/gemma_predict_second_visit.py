import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import json
import os
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
)

from plotting.common_patterns import (
    DA_COLUMN,
    load_da_level,
    get_label,
)

SEED = 42
np.random.seed(SEED)

logger = logging.getLogger(__name__)

START_TOKEN = "[START OF TRANSCRIPT]"
END_TOKEN   = "[END OF TRANSCRIPT]"


# ── filename parsing ──────────────────────────────────────────────────────────

def parse_patient_id(filename: str) -> str | None:
    """
    Extract patient ID: two digits immediately following 'AC'.
    E.g. 'randomAC01V1_x.csv' -> '01'.
    """
    match = re.search(r"AC(\d{2})", Path(filename).stem, re.IGNORECASE)
    return match.group(1) if match else None


def parse_session_number(filename: str) -> str | None:
    """
    Extract session number: single digit immediately following 'V'.
    E.g. 'randomAC01V1_x.csv' -> '1', 'sessionAC03V2.csv' -> '2'.
    Returns None if no match.
    """
    match = re.search(r"V(\d)", Path(filename).stem, re.IGNORECASE)
    return match.group(1) if match else None


def _normalise_speaker(raw: str) -> str:
    if isinstance(raw, str) and raw.strip().lower() == "therapist":
        return "therapist"
    return "patient"


# ── data loading ──────────────────────────────────────────────────────────────

def load_transcripts(
    dir_path:    Path,
    target:      str,
    granularity: str,
    text_col:    str,
) -> dict[str, dict]:
    """
    Load all transcripts.  Returns:
    {
      filename: {
        patient_id:  str,
        session:     str,   # "1" or "2"
        target_col:  str,
        das:         list[str],
        texts:       list[str],
        speakers:    list[str],
        labels:      list[int],
      }
    }
    Skips files missing the target column, text column, or session number.
    """
    target_col  = f"{target}_important"
    allowed_ext = {".csv", ".tsv", ".xlsx"}
    transcripts = {}

    for fp in sorted(dir_path.iterdir()):
        if fp.suffix.lower() not in allowed_ext:
            continue

        patient_id = parse_patient_id(fp.name)
        session    = parse_session_number(fp.name)

        if patient_id is None:
            print(f"Warning: could not parse patient ID from {fp.name} — skipping.",
                  flush=True)
            continue

        if session is None:
            print(f"Warning: could not parse session number from {fp.name} — skipping.",
                  flush=True)
            continue

        print(f"Loading {fp.name}  (patient={patient_id}  session={session}) …",
              flush=True)
        df = load_da_level(fp)
        df = df[df[DA_COLUMN] != "I-"].reset_index(drop=True)

        if target_col not in df.columns:
            print(f"  Warning: '{target_col}' not found — skipping.", flush=True)
            continue

        if text_col not in df.columns:
            print(f"  Warning: text column '{text_col}' not found — skipping.",
                  flush=True)
            continue

        df[target_col] = df[target_col].fillna(0).astype(int)
        n_pos = int(df[target_col].sum())
        n_tot = len(df)
        print(f"  {n_tot} DAs  {n_pos} important "
              f"({100*n_pos/max(n_tot,1):.1f}%)", flush=True)

        das      = [get_label(row[DA_COLUMN], row["da_group"], granularity)
                    for _, row in df.iterrows()]
        texts    = [str(row.get(text_col, "")).strip() for _, row in df.iterrows()]
        speakers = [_normalise_speaker(str(row.get("speaker", "patient")))
                    for _, row in df.iterrows()]
        labels   = df[target_col].tolist()

        transcripts[fp.name] = {
            "patient_id": patient_id,
            "session":    session,
            "target_col": target_col,
            "das":        das,
            "texts":      texts,
            "speakers":   speakers,
            "labels":     labels,
        }

    return transcripts


# ── session pairing ───────────────────────────────────────────────────────────

def pair_sessions(
    transcripts: dict[str, dict],
) -> list[tuple[str, str]]:
    """
    Pair V1 and V2 transcripts by patient ID.
    Returns list of (v1_filename, v2_filename) tuples.
    Patients with only one session are skipped with a warning.
    """
    by_patient: dict[str, dict[str, str]] = defaultdict(dict)
    for fname, rec in transcripts.items():
        by_patient[rec["patient_id"]][rec["session"]] = fname

    pairs = []
    for patient_id, sessions in sorted(by_patient.items()):
        if "1" not in sessions:
            print(f"  Warning: patient {patient_id} has no V1 session — skipping.",
                  flush=True)
            continue
        if "2" not in sessions:
            print(f"  Warning: patient {patient_id} has no V2 session — skipping.",
                  flush=True)
            continue
        pairs.append((sessions["1"], sessions["2"]))
        print(f"  Paired patient {patient_id}: "
              f"V1={sessions['1']}  V2={sessions['2']}", flush=True)

    return pairs


# ── context window formatting ─────────────────────────────────────────────────

def format_da_line(
    speaker: str,
    da:      str,
    text:    str,
    marker:  str = "",
) -> str:
    """Format a single DA line for the prompt."""
    prefix = f"{marker} " if marker else ""
    return f'{prefix}{speaker.capitalize()}: [{da}] "{text}"'


def build_context_window(
    rec:            dict,
    position:       int,
    context_window: int,
) -> str:
    """
    Build the context window string for a target DA at `position`.
    Edge positions are padded with START_TOKEN / END_TOKEN.
    """
    das      = rec["das"]
    texts    = rec["texts"]
    speakers = rec["speakers"]
    n        = len(das)
    lines    = []

    lines.append("[Preceding context]")
    for offset in range(-context_window, 0):
        idx = position + offset
        if idx < 0:
            lines.append(START_TOKEN)
        else:
            lines.append(format_da_line(speakers[idx], das[idx], texts[idx]))

    lines.append("[TARGET DA — classify this one]")
    lines.append(format_da_line(
        speakers[position], das[position], texts[position], marker=">>>"
    ))

    lines.append("[Following context]")
    for offset in range(1, context_window + 1):
        idx = position + offset
        if idx >= n:
            lines.append(END_TOKEN)
        else:
            lines.append(format_da_line(speakers[idx], das[idx], texts[idx]))

    return "\n".join(lines)


# ── session 1 context builder ─────────────────────────────────────────────────

def build_session_context(
    rec:        dict,
    max_tokens: int,
    tokenizer,
) -> str:
    """
    Build the full session 1 transcript as a labelled sequence for the prompt,
    truncated to fit within max_tokens.

    Each DA is shown as:
      Speaker: [DA] "text"  ->  important / not important

    The transcript is presented in order with no resampling or rebalancing.
    Truncation removes from the end once the token budget is reached.
    """
    das      = rec["das"]
    texts    = rec["texts"]
    speakers = rec["speakers"]
    labels   = rec["labels"]

    header = (
        "Below is a complete therapy session transcript with each dialogue act "
        "labeled as important or not important. Use this as context to understand "
        "what kinds of exchanges are considered important in this therapy setting.\n"
    )

    lines = [header]
    for i in range(len(das)):
        label_str = "important" if labels[i] == 1 else "not important"
        line      = (f"{format_da_line(speakers[i], das[i], texts[i])}"
                     f"  ->  {label_str}")
        lines.append(line)

    full_text = "\n".join(lines)

    # Truncate to max_tokens if a tokenizer and limit are provided
    if max_tokens > 0 and tokenizer is not None:
        # Gemma3Processor wraps the tokenizer — access it directly for encoding
        tok = tokenizer.tokenizer if hasattr(tokenizer, "tokenizer") else tokenizer
        tokens = tok.encode(full_text)
        if len(tokens) > max_tokens:
            # Re-build line by line until we hit the budget
            kept  = [header]
            token_count = len(tok.encode(header))
            for i in range(len(das)):
                label_str = "important" if labels[i] == 1 else "not important"
                line      = (f"{format_da_line(speakers[i], das[i], texts[i])}"
                             f"  ->  {label_str}")
                line_tokens = len(tok.encode(line))
                if token_count + line_tokens > max_tokens:
                    kept.append(
                        f"[TRUNCATED — {len(das) - i} further DAs not shown]"
                    )
                    break
                kept.append(line)
                token_count += line_tokens
            full_text = "\n".join(kept)
            print(f"    Session context truncated at {token_count} tokens "
                  f"({i}/{len(das)} DAs shown)", flush=True)

    return full_text


# ── prompt construction ───────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert behavioral psychologist analysing therapy session "
    "transcripts. You are given a complete first therapy session with each "
    "dialogue act labeled as important or not important, followed by a second "
    "session where you must classify a single TARGET DA. "
    "Use the first session to understand what kinds of exchanges are considered "
    "important for this specific patient and therapist. "
    "Answer with exactly one of: 'important' or 'not important'. "
    "Do not explain your answer."
)


def construct_prompt(
    session_context: str,
    target_context:  str,
    n_few_shot:      int = -1,
) -> str:
    """
    Construct the prompt for one target DA.

    session_context: the full (possibly truncated) V1 transcript
    target_context:  the context window around the target DA from V2
    n_few_shot:      if > 0, session_context is pre-truncated to this many
                     examples by the caller; if -1, use the full context
    """
    prompt_lines = [
        session_context,
        "\nNow classify the TARGET DA in the following sequence from the "
        "second session as either 'important' or 'not important'.\n"
        "Only classify the TARGET DA and text marked with >>>. "
        "Most DAs are NOT important — only classify as important if the "
        "content is clinically or therapeutically significant.\n"
        "Answer with exactly one of: 'important' or 'not important'.\n",
        target_context,
        "\nClassification:",
    ]
    return "\n".join(prompt_lines)


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(model_id: str, hf_cache_dir: str | None = None):
    """
    Load Gemma3ForConditionalGeneration + AutoProcessor from HuggingFace.
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


# ── inference ─────────────────────────────────────────────────────────────────

def generate_prediction(
    prompt:              str,
    system:              str,
    model_and_tokenizer: tuple,
    temperature:         float = 0.0,
    max_tokens:          int   = 10,
    retry_note:          str   = "",
) -> str:
    """
    Run inference using Gemma3ForConditionalGeneration + AutoProcessor.
    retry_note is appended on retry attempts.
    """
    import torch

    model, processor = model_and_tokenizer
    full_prompt      = prompt if not retry_note else f"{prompt}{retry_note}"

    messages = [
        {
            "role":    "system",
            "content": [{"type": "text", "text": system}],
        },
        {
            "role":    "user",
            "content": [{"type": "text", "text": full_prompt}],
        },
    ]

    inputs    = processor.apply_chat_template(
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
        gen_kwargs["top_p"]       = 0.95

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_tokens = output_ids[0][input_len:]
    response   = processor.decode(new_tokens, skip_special_tokens=True).strip()

    logger.debug(f"Response: {response}")
    return response


def parse_prediction(response: str) -> int | None:
    """
    Parse LLM response to binary label. 0=not important, 1=important.
    Returns None if unparseable so caller can retry.
    Checks 'not important' before 'important' to avoid substring trap.
    """
    r = response.lower().strip()
    if re.search(r"\bnot important\b", r):
        return 0
    if re.search(r"\bimportant\b", r):
        return 1
    logger.warning(f"Unparseable response: '{response}'")
    return None


# ── evaluation ────────────────────────────────────────────────────────────────

def evaluate(y_true: list[int], y_pred: list[int]) -> dict:
    """Compute and print evaluation metrics."""
    cm  = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TP  = int(cm[1, 1])
    TN  = int(cm[0, 0])
    FP  = int(cm[0, 1])
    FN  = int(cm[1, 0])

    sensitivity = TP / (TP + FN) if (TP + FN) > 0 else 0.0
    specificity = TN / (TN + FP) if (TN + FP) > 0 else 0.0
    precision   = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    f1_imp      = (2 * precision * sensitivity / (precision + sensitivity)
                   if (precision + sensitivity) > 0 else 0.0)
    f1_bal      = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_wt       = f1_score(y_true, y_pred, average="weighted", zero_division=0)

    print(f"\n{'='*60}", flush=True)
    print("EVALUATION RESULTS", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"TP={TP}  TN={TN}  FP={FP}  FN={FN}", flush=True)
    print(f"Sensitivity (recall): {sensitivity:.4f}", flush=True)
    print(f"Specificity:          {specificity:.4f}", flush=True)
    print(f"Precision:            {precision:.4f}", flush=True)
    print(f"F1(important):        {f1_imp:.4f}", flush=True)
    print(f"F1(balanced/macro):   {f1_bal:.4f}", flush=True)
    print(f"F1(weighted):         {f1_wt:.4f}", flush=True)
    print(f"\n{classification_report(y_true, y_pred, labels=[0,1], target_names=['not_important','important'], zero_division=0)}",
          flush=True)

    return {
        "TP": TP, "TN": TN, "FP": FP, "FN": FN,
        "sensitivity":  round(sensitivity, 4),
        "specificity":  round(specificity, 4),
        "precision":    round(precision,   4),
        "f1_important": round(f1_imp,      4),
        "f1_balanced":  round(f1_bal,      4),
        "f1_weighted":  round(f1_wt,       4),
    }


# ── prediction loop ───────────────────────────────────────────────────────────

def run_predictions(
    pairs:               list[tuple[str, str]],
    transcripts:         dict[str, dict],
    model_and_tokenizer: tuple,
    context_window:      int,
    max_context_tokens:  int,
    n_few_shot:          int,
    temperature:         float,
    max_tokens:          int,
    max_retries:         int,
    verbose:             bool,
) -> tuple[list[int], list[int], list[dict]]:
    """
    Run predictions over all V1/V2 patient pairs.

    For each pair:
      - Build session context from V1 (truncated to max_context_tokens)
      - For each DA in V2, build a context window and classify it
    """
    _, processor = model_and_tokenizer

    y_true_all: list[int]  = []
    y_pred_all: list[int]  = []
    pred_rows:  list[dict] = []

    for v1_fname, v2_fname in pairs:
        v1_rec = transcripts[v1_fname]
        v2_rec = transcripts[v2_fname]
        patient_id = v2_rec["patient_id"]

        print(f"\n{'─'*60}", flush=True)
        print(f"  Patient {patient_id}  |  "
              f"context: {v1_fname}  |  "
              f"predict: {v2_fname}", flush=True)

        # Build V1 session context (truncated to token budget)
        tokenizer_for_trunc = processor if max_context_tokens > 0 else None
        session_context = build_session_context(
            v1_rec, max_context_tokens, tokenizer_for_trunc
        )

        n_v2 = len(v2_rec["das"])
        print(f"  V2 DAs to predict: {n_v2}", flush=True)

        transcript_true_pos = 0
        transcript_pred_pos = 0

        for i in range(n_v2):
            target_context = build_context_window(v2_rec, i, context_window)
            prompt         = construct_prompt(
                session_context, target_context, n_few_shot
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
                    prompt, SYSTEM_PROMPT, model_and_tokenizer,
                    temperature, max_tokens,
                    retry_note=retry_note,
                )
                pred = parse_prediction(response)
                if pred is not None:
                    break
                print(f"    [retry {attempt+1}/{max_retries}] "
                      f"unparseable: '{response.strip()}'", flush=True)
                logger.warning(f"{v2_fname}[{i}] retry {attempt+1}: "
                               f"'{response.strip()}'")

            if pred is None:
                print(f"    [FAILED] max retries exhausted at position {i} "
                      f"— defaulting to 0", flush=True)
                logger.error(f"{v2_fname}[{i}] max retries exhausted, "
                             f"defaulting to 0")
                pred = 0

            label = v2_rec["labels"][i]

            y_true_all.append(label)
            y_pred_all.append(pred)

            transcript_true_pos += label
            transcript_pred_pos += pred

            pred_rows.append({
                "patient_id":  patient_id,
                "v1_filename": v1_fname,
                "v2_filename": v2_fname,
                "position":    i,
                "da":          v2_rec["das"][i],
                "speaker":     v2_rec["speakers"][i],
                "text":        v2_rec["texts"][i],
                "label":       label,
                "pred":        pred,
                "response":    response.strip(),
                "n_retries":   attempt,
            })

            if verbose and (i + 1) % 10 == 0:
                print(f"    [{i+1}/{n_v2}]  "
                      f"transcript true_pos={transcript_true_pos}  "
                      f"transcript pred_pos={transcript_pred_pos}", flush=True)

        print(f"  Patient {patient_id} done  —  "
              f"true_pos={transcript_true_pos}  "
              f"pred_pos={transcript_pred_pos}  "
              f"({n_v2} DAs)", flush=True)

        logger.debug(f"{v2_fname} complete: "
                     f"true_pos={transcript_true_pos} "
                     f"pred_pos={transcript_pred_pos}")

    return y_true_all, y_pred_all, pred_rows


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "LLM-based per-DA importance classifier.\n"
            "Uses session V1 as labelled context and predicts on session V2.\n"
            "Model loaded in-process via HuggingFace transformers."
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

    parser.add_argument("--context_window",     type=int, default=5,
                        help="DAs before/after the target DA in the prompt. "
                             "(default: 5)")
    parser.add_argument("--max_context_tokens", type=int, default=4096,
                        help="Maximum tokens the V1 session context may occupy. "
                             "0 = no limit (use full session). (default: 4096)")
    parser.add_argument("--n_few_shot",         type=int, default=-1,
                        help="If > 0, limit the session context to this many "
                             "DA examples from V1.  -1 = use full session up to "
                             "max_context_tokens. (default: -1)")

    parser.add_argument("--model_id",    default="google/gemma-3-4b-it",
                        help="HuggingFace model ID. (default: google/gemma-3-4b-it)")
    parser.add_argument("--hf_cache_dir", default=None,
                        help="HuggingFace cache directory. Useful on HPC.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="LLM sampling temperature. (default: 0.0)")
    parser.add_argument("--max_tokens",  type=int,   default=10,
                        help="Max new tokens for LLM response. (default: 10)")
    parser.add_argument("--max_retries", type=int,   default=3,
                        help="Max retries for unparseable responses. (default: 3)")

    parser.add_argument("--outdir",  default="llm_output/")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress every 10 DAs per transcript.")
    parser.add_argument("--log",     action="store_true",
                        help="Enable logging to a timestamped log file.")

    args = parser.parse_args()

    if args.log:
        ts       = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        log_path = os.path.join(args.outdir, f"run-{ts}.log")
        os.makedirs(args.outdir, exist_ok=True)
        logging.basicConfig(filename=log_path, level=logging.DEBUG)
        print(f"Logging to: {log_path}", flush=True)

    dir_path = Path(args.dir)
    if not dir_path.exists():
        raise ValueError(f"Directory not found: {args.dir}")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"LLM Importance Classifier (V1 context → V2 prediction)", flush=True)
    print(f"target={args.target}  granularity={args.granularity}", flush=True)
    print(f"text_col={args.text_col}  context_window={args.context_window}",
          flush=True)
    print(f"max_context_tokens={args.max_context_tokens}  "
          f"n_few_shot={args.n_few_shot}", flush=True)
    print(f"model_id={args.model_id}", flush=True)
    print(f"hf_cache_dir={args.hf_cache_dir}", flush=True)
    print(f"max_retries={args.max_retries}", flush=True)

    # ── load ──────────────────────────────────────────────────────────────────
    transcripts = load_transcripts(
        dir_path, args.target, args.granularity, args.text_col
    )
    print(f"\nLoaded {len(transcripts)} transcripts.", flush=True)

    # ── pair sessions ─────────────────────────────────────────────────────────
    print(f"\nPairing V1/V2 sessions …", flush=True)
    pairs = pair_sessions(transcripts)
    if not pairs:
        raise RuntimeError("No valid V1/V2 pairs found.")
    print(f"{len(pairs)} patient pair(s) ready.", flush=True)

    # ── load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model {args.model_id} …", flush=True)
    model_and_tokenizer = load_model(args.model_id, args.hf_cache_dir)
    print(f"Model ready.", flush=True)

    # ── predict ───────────────────────────────────────────────────────────────
    total_das = sum(len(transcripts[v2]["labels"]) for _, v2 in pairs)
    print(f"\nRunning predictions on {len(pairs)} patient(s) "
          f"({total_das} DAs total) …", flush=True)

    y_true, y_pred, pred_rows = run_predictions(
        pairs=pairs,
        transcripts=transcripts,
        model_and_tokenizer=model_and_tokenizer,
        context_window=args.context_window,
        max_context_tokens=args.max_context_tokens,
        n_few_shot=args.n_few_shot,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        verbose=args.verbose,
    )

    # ── evaluate ──────────────────────────────────────────────────────────────
    metrics = evaluate(y_true, y_pred)

    # ── save ──────────────────────────────────────────────────────────────────
    ts    = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    label = (
        f"{args.target}_{args.granularity}"
        f"_ctx{args.context_window}"
        f"_mct{args.max_context_tokens}"
        f"_{ts}"
    )

    pred_path = os.path.join(args.outdir, f"llm_{label}_predictions.csv")
    pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
    print(f"\n  Saved: {pred_path}", flush=True)

    metrics_path = os.path.join(args.outdir, f"llm_{label}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "label":              label,
            "target":             args.target,
            "granularity":        args.granularity,
            "context_window":     args.context_window,
            "max_context_tokens": args.max_context_tokens,
            "n_few_shot":         args.n_few_shot,
            "model_id":           args.model_id,
            "temperature":        args.temperature,
            "n_pairs":            len(pairs),
            "n_test_das":         len(y_true),
            "n_test_pos":         sum(y_true),
            "max_retries":        args.max_retries,
            "n_retried":          int(sum(1 for r in pred_rows
                                         if r["n_retries"] > 0)),
            **metrics,
        }, f, indent=2)
    print(f"  Saved: {metrics_path}", flush=True)

    print(f"\nDone. Outputs in: {args.outdir}", flush=True)


if __name__ == "__main__":
    main()