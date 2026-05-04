"""
This is similar to slide_gemma4.py, but adds the entire transcript as context. I think
in practice this could work well with RAG, but not alone.
"""
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
END_TOKEN   = "[END OF TRANSCRIPT]"

# DA group abbreviation -> human-readable full name (spaces, no underscores)
DA_FULL_NAME: dict[str, str] = {
    "canonical_questions":     "canonical question",
    "non_canonical_questions": "non canonical question",
    "canonical_answers":       "canonical answer",
    "non_canonical_answers":   "non canonical answer",
    "backchannel":             "backchannel",
    "statements":              "statement",
    "hedge":                   "hedge",
    "social_ritual":           "social ritual",
    "acknowledgement":         "acknowledgement",
    "elaboration":             "elaboration",
    "action":                  "action",
    "noise":                   "noise",
    "other":                   "other",
}


def _da_full(da: str) -> str:
    """Convert DA label to human-readable form. Falls back to original if unknown."""
    return DA_FULL_NAME.get(da, da.replace("_", " "))


# ── filename parsing ──────────────────────────────────────────────────────────

def parse_patient_id(filename: str) -> str | None:
    """Two digits immediately following 'AC'. E.g. 'randomAC01_x.csv' -> '01'."""
    match = re.search(r"AC(\d{2})", Path(filename).stem, re.IGNORECASE)
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
    Load all transcripts. Returns:
    {
      filename: {
        patient_id: str,
        das:        list[str],   # human-readable DA names
        texts:      list[str],
        speakers:   list[str],
        labels:     list[int],
      }
    }
    """
    target_col  = f"{target}_important"
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

        # Use human-readable DA names
        das      = [_da_full(get_label(row[DA_COLUMN], row["da_group"], granularity))
                    for _, row in df.iterrows()]
        texts    = [str(row.get(text_col, "")).strip() for _, row in df.iterrows()]
        speakers = [_normalise_speaker(str(row.get("speaker", "patient")))
                    for _, row in df.iterrows()]
        labels   = df[target_col].tolist()

        transcripts[fp.name] = {
            "patient_id": patient_id,
            "das":        das,
            "texts":      texts,
            "speakers":   speakers,
            "labels":     labels,
        }

    return transcripts


# ── patient split ─────────────────────────────────────────────────────────────

def split_by_patient(
    transcripts:      dict[str, dict],
    n_train_patients: int = 5,
) -> tuple[dict[str, dict], dict[str, dict]]:
    """Split by patient ID. n_train_patients for example pool, rest for test."""
    patient_to_files: dict[str, list[str]] = defaultdict(list)
    for fname, rec in transcripts.items():
        patient_to_files[rec["patient_id"]].append(fname)

    all_patients   = sorted(patient_to_files.keys())
    rng            = random.Random(SEED)
    train_patients = set(rng.sample(all_patients,
                                    min(n_train_patients, len(all_patients))))
    test_patients  = set(all_patients) - train_patients

    train = {f: transcripts[f] for p in train_patients
             for f in patient_to_files[p]}
    test  = {f: transcripts[f] for p in test_patients
             for f in patient_to_files[p]}

    print(f"\n  Train patients: {sorted(train_patients)}  "
          f"({len(train)} transcripts)", flush=True)
    print(f"  Test  patients: {sorted(test_patients)}  "
          f"({len(test)} transcripts)", flush=True)

    return train, test


# ── formatting ────────────────────────────────────────────────────────────────

def format_da_line(
    speaker: str,
    da:      str,
    text:    str,
    label:   int | None = None,
) -> str:
    """Format a single DA line, optionally with a label suffix."""
    label_str = ""
    if label is not None:
        label_str = f"  ->  {'important' if label == 1 else 'not important'}"
    return f'{speaker.capitalize()}: [{da}] "{text}"{label_str}'


# ── window and context builders ───────────────────────────────────────────────

def build_window(
    rec:         dict,
    start:       int,
    window_size: int,
) -> str:
    """
    Build the target window text (speaker, DA, text — no labels).
    Edge positions use START_TOKEN / END_TOKEN.
    """
    das      = rec["das"]
    texts    = rec["texts"]
    speakers = rec["speakers"]
    n        = len(das)
    lines    = []

    for offset in range(window_size):
        idx = start + offset
        if idx < 0:
            lines.append(START_TOKEN)
        elif idx >= n:
            lines.append(END_TOKEN)
        else:
            lines.append(format_da_line(speakers[idx], das[idx], texts[idx]))

    return "\n".join(lines)


def build_transcript_context(
    rec:            dict,
    window_start:   int,
    window_end:     int,
    context_before: int,
    context_after:  int,
    tok,
    token_budget:   int,
) -> str:
    """
    Build the transcript context surrounding the target window.

    Takes up to context_before DAs before the window and up to context_after
    DAs after the window. No labels shown — raw speaker/DA/text only.

    If the combined context exceeds token_budget, it is trimmed:
      - After-context is trimmed first (from the far end inward)
      - Before-context is trimmed second (from the far end inward)
    This preserves the closest context to the window.

    Returns (context_text, n_before_used, n_after_used).
    """
    das      = rec["das"]
    texts    = rec["texts"]
    speakers = rec["speakers"]
    n        = len(das)

    before_start = max(0, window_start - context_before)
    after_end    = min(n, window_end + context_after)

    before_lines = [
        format_da_line(speakers[i], das[i], texts[i])
        for i in range(before_start, window_start)
    ]
    after_lines = [
        format_da_line(speakers[i], das[i], texts[i])
        for i in range(window_end, after_end)
    ]

    if token_budget <= 0 or tok is None:
        sections = []
        if before_lines:
            sections.append("[Preceding conversation]\n" + "\n".join(before_lines))
        if after_lines:
            sections.append("[Following conversation]\n" + "\n".join(after_lines))
        return "\n\n".join(sections)

    # Token-aware trimming: trim after first, then before, from far end inward
    before_hdr_tokens = len(tok.encode("[Preceding conversation]"))
    after_hdr_tokens  = len(tok.encode("[Following conversation]"))
    used_tokens       = before_hdr_tokens + after_hdr_tokens

    # Fit after-context from nearest to furthest
    kept_after = []
    for line in after_lines:
        t = len(tok.encode(line))
        if used_tokens + t > token_budget:
            break
        kept_after.append(line)
        used_tokens += t

    # Fit before-context from nearest to furthest (reverse order, then re-reverse)
    kept_before = []
    for line in reversed(before_lines):
        t = len(tok.encode(line))
        if used_tokens + t > token_budget:
            break
        kept_before.append(line)
        used_tokens += t
    kept_before = list(reversed(kept_before))

    if len(kept_before) < len(before_lines) or len(kept_after) < len(after_lines):
        print(f"    Context trimmed: {len(kept_before)}/{len(before_lines)} before  "
              f"{len(kept_after)}/{len(after_lines)} after  "
              f"({used_tokens} tokens)", flush=True)

    sections = []
    if kept_before:
        sections.append("[Preceding conversation]\n" + "\n".join(kept_before))
    if kept_after:
        sections.append("[Following conversation]\n" + "\n".join(kept_after))
    return "\n\n".join(sections)


# ── few-shot example builders ─────────────────────────────────────────────────

def build_examples_from_transcripts(
    train_transcripts: dict[str, dict],
    window_size:       int,
    n_few_shot:        int,
    rng:               random.Random,
    pos_proportion:    float = 2 / 3,
    max_context_das:   int   = 5,
) -> tuple[list[str], list[str]]:
    """
    Build positive and negative few-shot examples from training transcripts.

    Positive: important block + up to max_context_das surrounding non-important
      DAs (randomly shifted). Majority of DAs are important.
      Window-level label shown in example header — no per-DA labels.

    Negative: window from a fully non-important stretch.
      Window-level label shown in example header — no per-DA labels.
    """
    pos_examples: list[str] = []
    neg_examples: list[str] = []

    for rec in train_transcripts.values():
        das      = rec["das"]
        texts    = rec["texts"]
        speakers = rec["speakers"]
        labels   = rec["labels"]
        n        = len(labels)

        # ── positive ──────────────────────────────────────────────────────────
        i = 0
        while i < n:
            if labels[i] == 1:
                j = i
                while j < n and labels[j] == 1:
                    j += 1

                max_before = min(max_context_das, i)
                max_after  = min(max_context_das, n - j)
                total_ctx  = max_before + max_after

                if total_ctx > 0:
                    before_take = rng.randint(
                        max(0, total_ctx - max_after),
                        min(total_ctx, max_before)
                    )
                    after_take = total_ctx - before_take
                else:
                    before_take = 0
                    after_take  = 0

                start = i - before_take
                end   = j + after_take
                lines = [
                    format_da_line(speakers[k], das[k], texts[k])
                    for k in range(start, end)
                ]
                pos_examples.append("\n".join(lines))
                i = j
            else:
                i += 1

        # ── negative ──────────────────────────────────────────────────────────
        i = 0
        while i < n:
            if labels[i] == 0:
                j = i
                while j < n and labels[j] == 0:
                    j += 1
                run_len = j - i
                if run_len >= 2:
                    win       = min(window_size, run_len)
                    max_start = run_len - win
                    offset    = rng.randint(0, max_start)
                    start     = i + offset
                    end       = start + win
                    lines     = [
                        format_da_line(speakers[k], das[k], texts[k])
                        for k in range(start, end)
                    ]
                    neg_examples.append("\n".join(lines))
                i = j
            else:
                i += 1

    rng.shuffle(pos_examples)
    rng.shuffle(neg_examples)

    if n_few_shot == -1:
        n_pos = len(pos_examples)
        n_neg = max(0, round(n_pos * (1 - pos_proportion) / pos_proportion))
        n_neg = min(n_neg, len(neg_examples))
    else:
        n_pos = round(n_few_shot * pos_proportion)
        n_neg = n_few_shot - n_pos
        n_pos = min(n_pos, len(pos_examples))
        n_neg = min(n_neg, len(neg_examples))

    print(f"    Examples: {n_pos} positive + {n_neg} negative = "
          f"{n_pos + n_neg} total", flush=True)

    return pos_examples[:n_pos], neg_examples[:n_neg]


# ── codebook / system prompt ──────────────────────────────────────────────────

def load_codebook(codebook_path: str) -> dict[str, str]:
    """
    Load coding codebook. Expects 'Abbreviation' and 'Comment' columns.
    Returns {abbreviation: comment} dict.
    """
    if not codebook_path or not Path(codebook_path).exists():
        return {}
    df = (pd.read_excel(codebook_path)
          if str(codebook_path).endswith(('.xlsx', '.xls'))
          else pd.read_csv(codebook_path))
    codebook = {}
    for _, row in df.iterrows():
        abbrev  = str(row.get("Abbreviation", "")).strip()
        comment = str(row.get("Comment", row.get("comment", ""))).strip()
        if abbrev and abbrev.lower() != "nan" and comment and comment.lower() != "nan":
            codebook[abbrev] = comment
    print(f"  Loaded {len(codebook)} codes: {list(codebook.keys())}", flush=True)
    return codebook


def compute_base_rate(train_transcripts: dict[str, dict]) -> float:
    """Pooled base rate of important DAs across training transcripts."""
    n_total = sum(len(r["labels"]) for r in train_transcripts.values())
    n_imp   = sum(sum(r["labels"]) for r in train_transcripts.values())
    return n_imp / max(n_total, 1)


def build_system_prompt(codebook: dict[str, str], base_rate: float) -> str:
    """Build dynamic system prompt with codebook descriptions and base rate."""
    lines = [
        "You are an expert behavioral psychologist analysing therapy session "
        "transcripts. You will be shown labeled examples of dialogue act windows, "
        "followed by transcript context and a target window to classify.",
        "",
        "IMPORTANT MOMENTS are defined as windows containing dialogue that is "
        "clinically or therapeutically significant. They are characterised by "
        "one or more of the following codes:",
        "",
    ]
    if codebook:
        for abbrev, comment in codebook.items():
            lines.append(f"  {abbrev}: {comment}")
    else:
        lines.append("  (No codebook provided — use clinical judgment.)")

    lines += [
        "",
        f"BASE RATE: In a typical therapy session, approximately "
        f"{base_rate*100:.0f}% of dialogue act windows are important. "
        f"The vast majority of windows are NOT important.",
        "",
        "CLASSIFICATION RULES:",
        "  - Classify as 'important' ONLY if the window clearly contains "
        "content matching one of the codes above.",
        "  - If you are unsure, classify as 'not important'.",
        "  - Routine conversation, logistics, and filler content are "
        "almost always 'not important'.",
        "  - When in doubt, answer 'not important'.",
        "  - The transcript context shows what came before and after the "
        "target window — use it to understand the flow of the session, "
        "but classify only the TARGET WINDOW.",
        "",
        "Answer with exactly one of: 'important' or 'not important'.",
        "Do not explain your answer.",
    ]
    return "\n".join(lines)


# ── prompt construction ───────────────────────────────────────────────────────

def construct_prompt(
    pos_examples:       list[str],
    neg_examples:       list[str],
    transcript_context: str,
    window_text:        str,
    tok=None,
    max_input_tokens:   int = 0,
) -> str:
    """
    Construct the prompt for one window classification.

    Token budget priority (highest to lowest):
      1. Target window + classification instructions (never trimmed)
      2. Transcript context (trimmed from far end by build_transcript_context)
      3. Few-shot examples (dropped last to first to fit remaining budget)

    Structure:
      [Few-shot examples]
      [Transcript context — preceding / following conversation]
      [Target window]
      Classification:
    """
    tail = "\n".join([
        "Now classify the following TARGET WINDOW as either 'important' or "
        "'not important'.\n"
        "Use the transcript context above to understand the conversation flow, "
        "but classify only the TARGET WINDOW.\n"
        "Answer with exactly one of: 'important' or 'not important'.\n",
        "[TARGET WINDOW]",
        window_text,
        "\nClassification:",
    ])

    ctx_section = ""
    if transcript_context.strip():
        ctx_section = (
            "[TRANSCRIPT CONTEXT — for reference only, do not classify]\n"
            + transcript_context
        )

    example_header = (
        "Below are examples of therapy session windows with their "
        "window-level classification. Each example shows a sequence of "
        "dialogue acts — classify the window as a whole, not individual lines.\n"
    )

    # Interleave pos and neg examples
    all_examples = []
    for p, ng in zip(pos_examples, neg_examples):
        all_examples.append((p, "important"))
        all_examples.append((ng, "not important"))
    for p in pos_examples[len(neg_examples):]:
        all_examples.append((p, "important"))
    for ng in neg_examples[len(pos_examples):]:
        all_examples.append((ng, "not important"))

    if not all_examples and not ctx_section:
        return tail

    if max_input_tokens <= 0 or tok is None:
        lines = []
        if all_examples:
            lines.append(example_header)
            for i, (ex, lbl) in enumerate(all_examples):
                lines.append(f"--- Example {i+1} (classification: {lbl}) ---")
                lines.append(ex)
                lines.append("")
        if ctx_section:
            lines.append(ctx_section)
            lines.append("")
        lines.append(tail)
        return "\n".join(lines)

    # Token-aware: tail + context are already sized; fit examples in remainder
    tail_tokens    = len(tok.encode(tail))
    ctx_tokens     = len(tok.encode(ctx_section)) if ctx_section else 0
    hdr_tokens     = len(tok.encode(example_header)) if all_examples else 0
    fixed_tokens   = tail_tokens + ctx_tokens + hdr_tokens
    example_budget = max_input_tokens - fixed_tokens

    kept        = []
    used_tokens = 0
    for ex, lbl in all_examples:
        block     = f"--- Example {len(kept)+1} (classification: {lbl}) ---\n{ex}\n"
        ex_tokens = len(tok.encode(block))
        if example_budget > 0 and used_tokens + ex_tokens > example_budget:
            break
        kept.append((ex, lbl))
        used_tokens += ex_tokens

    if len(kept) < len(all_examples):
        print(f"    Prompt budget: kept {len(kept)}/{len(all_examples)} examples  "
              f"total ~{fixed_tokens + used_tokens} tokens", flush=True)

    lines = []
    if kept:
        lines.append(example_header)
        for i, (ex, lbl) in enumerate(kept):
            lines.append(f"--- Example {i+1} (classification: {lbl}) ---")
            lines.append(ex)
            lines.append("")
    if ctx_section:
        lines.append(ctx_section)
        lines.append("")
    lines.append(tail)
    return "\n".join(lines)


# ── post-processing ───────────────────────────────────────────────────────────

def apply_min_run_filter(
    preds:           list[int],
    min_important:   int,
    min_unimportant: int,
) -> list[int]:
    """
    Flip runs shorter than their minimum length to the opposite label.
    important runs < min_important -> 0
    not-important runs < min_unimportant -> 1
    Applied in that order.
    """
    result = list(preds)
    for target, min_len in [(1, min_important), (0, min_unimportant)]:
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


# ── model loading ─────────────────────────────────────────────────────────────

def load_model(model_id: str, hf_cache_dir: str | None = None):
    """Load AutoModelForImageTextToText + AutoProcessor. Returns (model, processor)."""
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if hf_cache_dir:
        os.environ["HF_HOME"]            = hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
        os.environ["HF_DATASETS_CACHE"]  = hf_cache_dir
        print(f"HuggingFace cache dir: {hf_cache_dir}", flush=True)

    print(f"Loading processor: {model_id} …", flush=True)
    processor = AutoProcessor.from_pretrained(model_id)

    print(f"Loading model: {model_id} in float16 …", flush=True)
    model = AutoModelForImageTextToText.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map="auto",
    ).eval()

    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}", flush=True)

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved  = torch.cuda.memory_reserved()  / 1024**3
        print(f"GPU memory: {allocated:.1f}GB allocated  "
              f"{reserved:.1f}GB reserved", flush=True)

    return model, processor


# ── inference ─────────────────────────────────────────────────────────────────

def generate_prediction(
    prompt:              str,
    system:              str,
    model_and_processor: tuple,
    temperature:         float = 0.0,
    max_tokens:          int   = 10,
    retry_note:          str   = "",
    max_input_tokens:    int   = 0,
) -> str:
    """Run inference via AutoModelForImageTextToText."""
    import torch

    model, processor = model_and_processor
    full_prompt      = prompt if not retry_note else f"{prompt}{retry_note}"

    if max_input_tokens > 0:
        tok    = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        tokens = tok.encode(full_prompt)
        if len(tokens) > max_input_tokens:
            logger.warning(f"Prompt {len(tokens)} tokens exceeds budget "
                           f"{max_input_tokens} after trimming")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system}]},
        {"role": "user",   "content": [{"type": "text", "text": full_prompt}]},
    ]

    inputs    = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len = inputs["input_ids"].shape[-1]
    gen_kwargs = dict(max_new_tokens=max_tokens, do_sample=temperature > 0.0)
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
    """Parse response to 0/1. Returns None if unparseable."""
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
    TP  = int(cm[1, 1]);  TN = int(cm[0, 0])
    FP  = int(cm[0, 1]);  FN = int(cm[1, 0])

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


# ── sliding window prediction ─────────────────────────────────────────────────

def run_predictions(
    test_transcripts:    dict[str, dict],
    pos_examples:        list[str],
    neg_examples:        list[str],
    model_and_processor: tuple,
    system_prompt:       str,
    window_size:         int,
    window_stride:       int,
    context_before:      int,
    context_after:       int,
    vote_threshold:      float,
    min_important_run:   int,
    min_unimportant_run: int,
    temperature:         float,
    max_tokens:          int,
    max_retries:         int,
    max_input_tokens:    int,
    verbose:             bool,
) -> tuple[list[int], list[int], list[dict]]:
    """
    Sliding window ensemble prediction with transcript context.

    For each window:
      1. Build transcript context: up to context_before DAs before the window
         and context_after DAs after (raw speaker/DA/text, no labels).
      2. Build the few-shot prompt with context included, respecting the
         token budget (target window > context > examples).
      3. Collect one binary vote per window.
      4. Aggregate votes per DA, threshold, apply min-run filter.
    """
    _, processor = model_and_processor
    tok = (processor.tokenizer if hasattr(processor, "tokenizer")
           else processor)

    y_true_all: list[int]  = []
    y_pred_all: list[int]  = []
    pred_rows:  list[dict] = []

    for fname, rec in test_transcripts.items():
        n = len(rec["das"])
        print(f"\n  Transcript: {fname}  ({n} DAs)", flush=True)

        vote_counts: list[int] = [0] * n
        vote_totals: list[int] = [0] * n

        starts = list(range(0, n, window_stride))
        if starts and starts[-1] + window_size < n:
            starts.append(n - window_size)

        n_windows = len(starts)
        print(f"    {n_windows} windows  "
              f"(size={window_size}  stride={window_stride}  "
              f"ctx_before={context_before}  ctx_after={context_after})",
              flush=True)

        for w_idx, start in enumerate(starts):
            window_end  = min(start + window_size, n)
            window_text = build_window(rec, start, window_size)

            # Token budget for context: total - (estimated tail + examples)
            # We pre-reserve space for tail and a minimum of 1 example pair
            # Context gets what's left up to max_input_tokens
            ctx_token_budget = max(0, max_input_tokens - 1024) \
                               if max_input_tokens > 0 else 0

            transcript_context = build_transcript_context(
                rec, start, window_end,
                context_before, context_after,
                tok if ctx_token_budget > 0 else None,
                ctx_token_budget,
            )

            prompt = construct_prompt(
                pos_examples, neg_examples,
                transcript_context, window_text,
                tok=tok,
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
                print(f"    [retry {attempt+1}/{max_retries}] "
                      f"window {w_idx} unparseable: '{response.strip()}'",
                      flush=True)
                logger.warning(f"{fname} window {w_idx} retry {attempt+1}: "
                               f"'{response.strip()}'")

            if pred is None:
                print(f"    [FAILED] window {w_idx} defaulting to 0", flush=True)
                logger.error(f"{fname} window {w_idx} max retries exhausted")
                pred = 0

            actual_end = min(start + window_size, n)
            for da_idx in range(start, actual_end):
                vote_counts[da_idx] += pred
                vote_totals[da_idx] += 1

            if verbose and (w_idx + 1) % 10 == 0:
                print(f"    [{w_idx+1}/{n_windows} windows]", flush=True)

            logger.debug(f"{fname} window {w_idx} start={start} pred={pred} "
                         f"response='{response.strip()}'")

        # ── threshold + post-process ──────────────────────────────────────────
        raw_preds = [
            1 if (vote_totals[i] > 0 and
                  vote_counts[i] / vote_totals[i] >= vote_threshold)
            else 0
            for i in range(n)
        ]
        final_preds = apply_min_run_filter(
            raw_preds, min_important_run, min_unimportant_run
        )

        n_raw_pos   = sum(raw_preds)
        n_final_pos = sum(final_preds)
        print(f"  Raw predictions:      {n_raw_pos}/{n} important", flush=True)
        print(f"  After min-run filter: {n_final_pos}/{n} important", flush=True)

        # ── collect results ───────────────────────────────────────────────────
        transcript_true_pos = 0
        transcript_pred_pos = 0

        for i in range(n):
            label = rec["labels"][i]
            pred  = final_preds[i]
            y_true_all.append(label)
            y_pred_all.append(pred)
            transcript_true_pos += label
            transcript_pred_pos += pred

            pred_rows.append({
                "filename":   fname,
                "patient_id": rec["patient_id"],
                "position":   i,
                "da":         rec["das"][i],
                "speaker":    rec["speakers"][i],
                "text":       rec["texts"][i],
                "label":      label,
                "pred":       pred,
                "pred_raw":   raw_preds[i],
                "vote_count": vote_counts[i],
                "vote_total": vote_totals[i],
                "vote_frac":  round(vote_counts[i] / max(vote_totals[i], 1), 4),
            })

        print(f"  Done: {fname}  —  "
              f"true_pos={transcript_true_pos}  "
              f"pred_pos={transcript_pred_pos}", flush=True)

    return y_true_all, y_pred_all, pred_rows


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sliding-window ensemble LLM importance classifier with\n"
            "transcript context. Each window is classified with up to\n"
            "context_before DAs of preceding and context_after DAs of\n"
            "following conversation shown as unlabeled context."
        )
    )

    parser.add_argument("--dir",           required=True)
    parser.add_argument("--granularity",   default="groups",
                        choices=["groups", "raw"])
    parser.add_argument("--target",        default="patient",
                        choices=["patient", "therapist"])
    parser.add_argument("--text_col",      required=True)

    parser.add_argument("--n_train_patients",  type=int,   default=5)
    parser.add_argument("--n_few_shot",        type=int,   default=12,
                        help="-1 = all available. (default: 16)")
    parser.add_argument("--pos_proportion",    type=float, default=2/3)
    parser.add_argument("--max_context_das",   type=int,   default=5,
                        help="Max surrounding non-important DAs in positive "
                             "examples. (default: 5)")
    parser.add_argument("--codebook",          default=None)

    parser.add_argument("--window_size",       type=int,   default=20)
    parser.add_argument("--window_stride",     type=int,   default=5)
    parser.add_argument("--context_before",    type=int,   default=100,
                        help="Max DAs of preceding transcript context shown "
                             "per window. (default: 200)")
    parser.add_argument("--context_after",     type=int,   default=50,
                        help="Max DAs of following transcript context shown "
                             "per window. (default: 100)")
    parser.add_argument("--vote_threshold",    type=float, default=0.5,
                        help=">= this fraction of votes -> important. (default: 0.5)")
    parser.add_argument("--min_important_run",   type=int, default=10)
    parser.add_argument("--min_unimportant_run", type=int, default=1)

    parser.add_argument("--model_id",         default="google/gemma-4-E4B-it")
    parser.add_argument("--hf_cache_dir",     default=None)
    parser.add_argument("--temperature",      type=float, default=0.0)
    parser.add_argument("--max_tokens",       type=int,   default=10)
    parser.add_argument("--max_input_tokens", type=int,   default=12000,
                        help="Max prompt tokens. 0 = no limit. (default: 8192)")
    parser.add_argument("--max_retries",      type=int,   default=3)

    parser.add_argument("--outdir",  default="llm_output/")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log",     action="store_true")

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

    print(f"Sliding-Window + Transcript Context LLM Classifier", flush=True)
    print(f"target={args.target}  granularity={args.granularity}", flush=True)
    print(f"text_col={args.text_col}", flush=True)
    print(f"window_size={args.window_size}  window_stride={args.window_stride}  "
          f"vote_threshold={args.vote_threshold}", flush=True)
    print(f"context_before={args.context_before}  "
          f"context_after={args.context_after}", flush=True)
    print(f"min_important_run={args.min_important_run}  "
          f"min_unimportant_run={args.min_unimportant_run}", flush=True)
    print(f"n_train_patients={args.n_train_patients}  "
          f"n_few_shot={args.n_few_shot}  "
          f"pos_proportion={args.pos_proportion:.2f}", flush=True)
    print(f"model_id={args.model_id}", flush=True)
    print(f"codebook={args.codebook}", flush=True)
    print(f"max_input_tokens={args.max_input_tokens}  "
          f"max_retries={args.max_retries}", flush=True)

    # ── load ──────────────────────────────────────────────────────────────────
    transcripts = load_transcripts(
        dir_path, args.target, args.granularity, args.text_col
    )
    print(f"\nLoaded {len(transcripts)} transcripts.", flush=True)

    # ── split ─────────────────────────────────────────────────────────────────
    print(f"\nSplitting by patient (train={args.n_train_patients}) …",
          flush=True)
    train_transcripts, test_transcripts = split_by_patient(
        transcripts, args.n_train_patients
    )

    # ── codebook + base rate + system prompt ──────────────────────────────────
    print(f"\nLoading codebook …", flush=True)
    codebook = load_codebook(args.codebook) if args.codebook else {}
    if not codebook:
        print("  No codebook — system prompt will omit code descriptions.",
              flush=True)

    base_rate = compute_base_rate(train_transcripts)
    print(f"  Training base rate: {base_rate*100:.1f}% important DAs",
          flush=True)

    system_prompt = build_system_prompt(codebook, base_rate)
    print(f"  System prompt: {len(system_prompt)} chars", flush=True)

    # ── examples ──────────────────────────────────────────────────────────────
    print(f"\nBuilding few-shot examples …", flush=True)
    rng = random.Random(SEED)
    pos_examples, neg_examples = build_examples_from_transcripts(
        train_transcripts, args.window_size, args.n_few_shot, rng,
        pos_proportion=args.pos_proportion,
        max_context_das=args.max_context_das,
    )

    expected_votes = args.window_size // args.window_stride
    print(f"\nExpected votes per DA (mid-transcript): ~{expected_votes}",
          flush=True)

    # ── model ─────────────────────────────────────────────────────────────────
    print(f"\nLoading model {args.model_id} …", flush=True)
    model_and_processor = load_model(args.model_id, args.hf_cache_dir)
    print(f"Model ready.", flush=True)

    # ── predict ───────────────────────────────────────────────────────────────
    total_das = sum(len(r["labels"]) for r in test_transcripts.values())
    print(f"\nRunning predictions on {len(test_transcripts)} test transcripts "
          f"({total_das} DAs) …", flush=True)

    y_true, y_pred, pred_rows = run_predictions(
        test_transcripts=test_transcripts,
        pos_examples=pos_examples,
        neg_examples=neg_examples,
        model_and_processor=model_and_processor,
        system_prompt=system_prompt,
        window_size=args.window_size,
        window_stride=args.window_stride,
        context_before=args.context_before,
        context_after=args.context_after,
        vote_threshold=args.vote_threshold,
        min_important_run=args.min_important_run,
        min_unimportant_run=args.min_unimportant_run,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        max_input_tokens=args.max_input_tokens,
        verbose=args.verbose,
    )

    # ── evaluate ──────────────────────────────────────────────────────────────
    metrics = evaluate(y_true, y_pred)

    # ── save ──────────────────────────────────────────────────────────────────
    ts    = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    label = (
        f"{args.target}_{args.granularity}"
        f"_ws{args.window_size}_st{args.window_stride}"
        f"_cb{args.context_before}_ca{args.context_after}"
        f"_thr{int(args.vote_threshold*100)}"
        f"_{ts}"
    )

    pred_path = os.path.join(args.outdir, f"llm_{label}_predictions.csv")
    pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
    print(f"\n  Saved: {pred_path}", flush=True)

    metrics_path = os.path.join(args.outdir, f"llm_{label}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({
            "label":               label,
            "target":              args.target,
            "granularity":         args.granularity,
            "window_size":         args.window_size,
            "window_stride":       args.window_stride,
            "context_before":      args.context_before,
            "context_after":       args.context_after,
            "vote_threshold":      args.vote_threshold,
            "min_important_run":   args.min_important_run,
            "min_unimportant_run": args.min_unimportant_run,
            "n_few_shot":          args.n_few_shot,
            "pos_proportion":      args.pos_proportion,
            "max_context_das":     args.max_context_das,
            "n_train_patients":    args.n_train_patients,
            "model_id":            args.model_id,
            "codebook":            args.codebook,
            "base_rate_train":     round(base_rate, 4),
            "temperature":         args.temperature,
            "max_input_tokens":    args.max_input_tokens,
            "n_test_das":          len(y_true),
            "n_test_pos":          sum(y_true),
            "n_examples_used":     len(pos_examples) + len(neg_examples),
            "max_retries":         args.max_retries,
            **metrics,
        }, f, indent=2)
    print(f"  Saved: {metrics_path}", flush=True)

    print(f"\nDone. Outputs in: {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
