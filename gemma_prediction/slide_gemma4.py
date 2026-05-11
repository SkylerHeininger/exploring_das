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
        das:        list[str],
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

        das      = [get_label(row[DA_COLUMN], row["da_group"], granularity)
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
    """
    Split by patient ID. n_train_patients randomly selected for the few-shot
    example pool; remainder used for prediction.
    """
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


# ── window builder ────────────────────────────────────────────────────────────

def build_window(
    rec:         dict,
    start:       int,
    window_size: int,
) -> str:
    """
    Build a text representation of a window of DAs starting at `start`
    with length `window_size`. Labels are NOT shown — the model classifies
    the whole window as important or not important.
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


# ── few-shot example builders ─────────────────────────────────────────────────

def build_examples_from_transcripts(
    train_transcripts:  dict[str, dict],
    window_size:        int,
    n_few_shot:         int,
    rng:                random.Random,
    pos_proportion:     float = 2 / 3,
    max_context_das:    int   = 5,
) -> tuple[list[str], list[str]]:
    """
    Build positive and negative few-shot examples from training transcripts.

    Positive examples: the important block plus up to max_context_das
      non-important DAs of surrounding context (randomly shifted before/after).
      The majority of DAs in the window are important.
      No per-DA labels shown — only the window-level label in the header.

    Negative examples: a window sampled entirely from a non-important stretch,
      capped at window_size DAs.
      No per-DA labels shown.

    This matches the window-level binary prediction task: the model sees a
    window of unlabeled DAs and must give one binary answer for the whole window.
    """
    pos_examples: list[str] = []
    neg_examples: list[str] = []

    for rec in train_transcripts.values():
        das      = rec["das"]
        texts    = rec["texts"]
        speakers = rec["speakers"]
        labels   = rec["labels"]
        n        = len(labels)

        # ── positive: important block + up to max_context_das surrounding ─────
        i = 0
        while i < n:
            if labels[i] == 1:
                j = i
                while j < n and labels[j] == 1:
                    j += 1

                # Available non-important context before and after
                max_before = min(max_context_das, i)
                max_after  = min(max_context_das, n - j)

                # Randomly distribute context budget before/after
                total_ctx    = max_before + max_after
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

                # No per-DA labels — window-level label in example header
                lines = [
                    format_da_line(speakers[k], das[k], texts[k])
                    for k in range(start, end)
                ]
                pos_examples.append("\n".join(lines))
                i = j
            else:
                i += 1

        # ── negative: window from a non-important stretch ──────────────────────
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


# ── prompt construction ───────────────────────────────────────────────────────

def load_codebook(codebook_path: str) -> dict[str, str]:
    """
    Load the coding codebook from a CSV file.
    Expects columns 'Abbreviation' and 'comment'.
    Returns {abbreviation: comment} dict, skipping rows with empty values.
    """
    if not codebook_path or not Path(codebook_path).exists():
        return {}
    df = pd.read_excel(codebook_path) if str(codebook_path).endswith(('.xlsx', '.xls')) else pd.read_csv(codebook_path)
    codebook = {}
    for _, row in df.iterrows():
        abbrev  = str(row.get("Abbreviation", "")).strip()
        comment = str(row.get("Comment", row.get("comment", ""))).strip()
        if abbrev and abbrev.lower() != "nan" and comment and comment.lower() != "nan":
            codebook[abbrev] = comment
    print(f"  Loaded {len(codebook)} codes from codebook: "
          f"{list(codebook.keys())}", flush=True)
    return codebook


def compute_base_rate(train_transcripts: dict[str, dict]) -> float:
    """
    Compute the pooled base rate of important DAs across all training transcripts.
    Returns the fraction of DAs labeled important.
    """
    n_total = sum(len(r["labels"]) for r in train_transcripts.values())
    n_imp   = sum(sum(r["labels"]) for r in train_transcripts.values())
    return n_imp / max(n_total, 1)


def build_system_prompt(
    codebook:  dict[str, str],
    base_rate: float,
) -> str:
    """
    Build the system prompt dynamically, incorporating:
      - The coding scheme descriptions from the codebook
      - The empirical base rate of important DAs from training data
      - Explicit bias toward not-important to reduce false positives
    """
    lines = [
        "You are an expert behavioral psychologist analysing therapy session "
        "transcripts. You will be shown labeled examples of dialogue act windows, "
        "then a new unlabeled window to classify.",
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
        "Important moments are specific, intentional therapeutic acts — a "
        "deliberate intervention, a focused exchange, or a distinct shift in "
        "the conversation. The mere presence of a relevant topic does NOT make "
        "a window important. Most discussion of relevant topics is background "
        "conversation and NOT important. Only windows where something specific "
        "and purposeful is happening should be classified as important.",
        "",
        f"BASE RATE: In a typical therapy session, approximately "
        f"{base_rate*100:.0f}% of dialogue act windows are important. "
        f"The vast majority of windows are NOT important.",
        "",
        "CLASSIFICATION RULES:",
        "  - Classify as 'important' ONLY if the window contains a specific, "
        "purposeful act matching one of the codes above — not merely a topic "
        "that relates to one of the codes.",
        "  - If you are unsure, classify as 'not important'.",
        "  - Routine conversation, logistics, and filler content are "
        "almost always 'not important'.",
        "  - When in doubt, answer 'not important'.",
        "",
        "Answer with exactly one of: 'important' or 'not important'.",
        "Do not explain your answer.",
    ]

    return "\n".join(lines)


SUMMARY_SYSTEM_PROMPT = (
    "You are a neutral observer summarising a therapy session transcript. "
    "Your task is to describe what happened in the session factually and "
    "objectively. Do NOT indicate what was or was not clinically important. "
    "Do NOT use language that suggests significance, breakthroughs, or key "
    "moments. Simply describe the session as it unfolded."
)

SUMMARY_USER_PROMPT = (
    "Please summarise the following therapy session transcript in two short "
    "paragraphs:\n"
    "TOPICS: The main subjects and themes discussed during the session "
    "(2-3 sentences).\n"
    "ARC: How the conversation developed and shifted over time "
    "(2-3 sentences).\n"
    "Be neutral and factual. Do not comment on clinical significance. "
    "Do not say anything is important or meaningful."
)


def build_transcript_text(rec: dict, max_das: int = 500) -> str:
    """
    Render a transcript record as plain text for the summary LLM.
    Truncated to max_das DAs to keep the summary call within token limits.
    Shows speaker, DA type, and text — no labels.
    """
    das      = rec["das"]
    texts    = rec["texts"]
    speakers = rec["speakers"]
    n        = min(len(das), max_das)

    lines = []
    for i in range(n):
        lines.append(f"{speakers[i].capitalize()}: [{das[i]}] \"{texts[i]}\"")

    if len(rec["das"]) > max_das:
        lines.append(f"... [{len(rec['das']) - max_das} further DAs not shown]")

    return "\n".join(lines)


def generate_transcript_summary(
    rec:                 dict,
    fname:               str,
    model_and_processor: tuple,
    summaries_dir:       str,
    max_das:             int = 500,
    max_tokens:          int = 400,
) -> str:
    """
    Generate a neutral factual summary of a transcript using the same model
    as the importance classifier.

    The summary is saved to {summaries_dir}/{fname_stem}_summary.txt so it
    can be inspected and is not recomputed on reruns if the file exists.

    Returns the summary string.
    """
    os.makedirs(summaries_dir, exist_ok=True)
    stem         = Path(fname).stem
    summary_path = os.path.join(summaries_dir, f"{stem}_summary.txt")

    # Return cached summary if it exists
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = f.read().strip()
        print(f"    Summary loaded from cache: {summary_path}", flush=True)
        return summary

    transcript_text = build_transcript_text(rec, max_das=max_das)
    user_content    = f"{SUMMARY_USER_PROMPT}\n\n{transcript_text}"

    print(f"    Generating summary for {fname} …", flush=True)
    summary = generate_prediction(
        prompt               = user_content,
        system               = SUMMARY_SYSTEM_PROMPT,
        model_and_processor  = model_and_processor,
        temperature          = 0.0,
        max_tokens           = max_tokens,
    )

    # Save to disk
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Transcript: {fname}\n")
        f.write("=" * 60 + "\n\n")
        f.write(summary)
    print(f"    Summary saved: {summary_path}", flush=True)

    return summary


def build_system_prompt_with_summary(
    codebook:  dict[str, str],
    base_rate: float,
    summary:   str,
) -> str:
    """
    Build the per-transcript system prompt, incorporating the neutral
    transcript summary before the base rate and classification rules.
    The summary gives the model context about the session arc and topics
    without priming it to label specific content as important.
    """
    lines = [
        "You are an expert behavioral psychologist analysing therapy session "
        "transcripts. You will be shown labeled examples of dialogue act windows, "
        "then a new unlabeled window to classify.",
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
        "SESSION OVERVIEW:",
        "The following is a neutral factual summary of the session you are "
        "classifying. Use it to understand the overall context, but do NOT "
        "use it to decide what is important — most of the session is not "
        "important regardless of the topics discussed.",
        "",
    ]
    for line in summary.splitlines():
        lines.append(f"  {line}")

    lines += [
        "",
        "Important moments are specific, intentional therapeutic acts — a "
        "deliberate intervention, a focused exchange, or a distinct shift in "
        "the conversation. The mere presence of a relevant topic does NOT make "
        "a window important. Most discussion of relevant topics is background "
        "conversation and NOT important. Only windows where something specific "
        "and purposeful is happening should be classified as important.",
        "",
        f"BASE RATE: In a typical therapy session, approximately "
        f"{base_rate*100:.0f}% of dialogue act windows are important. "
        f"The vast majority of windows are NOT important.",
        "",
        "CLASSIFICATION RULES:",
        "  - Classify as 'important' ONLY if the window contains a specific, "
        "purposeful act matching one of the codes above — not merely a topic "
        "that relates to one of the codes.",
        "  - If you are unsure, classify as 'not important'.",
        "  - Routine conversation, logistics, and filler content are "
        "almost always 'not important'.",
        "  - When in doubt, answer 'not important'.",
        "",
        "Answer with exactly one of: 'important' or 'not important'.",
        "Do not explain your answer.",
    ]

    return "\n".join(lines)


def construct_prompt(
    pos_examples:     list[str],
    neg_examples:     list[str],
    window_text:      str,
    processor=None,
    max_input_tokens: int = 0,
) -> str:
    """
    Construct the prompt for one window classification.

    Interleaves positive and negative examples. If processor and
    max_input_tokens are provided, examples are dropped from the end
    (never the window_text or instructions) to fit within the token budget.
    """
    # Fixed tail that must always be present
    tail = "\n".join([
        "Now classify the following window as either 'important' or "
        "'not important'.\n"
        "Answer with exactly one of: 'important' or 'not important'.\n",
        window_text,
        "\nClassification:",
    ])

    header = (
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

    if not all_examples:
        return f"{tail}"

    if max_input_tokens <= 0 or processor is None:
        lines = [header]
        for i, (ex, lbl) in enumerate(all_examples):
            lines.append(f"--- Example {i+1} (classification: {lbl}) ---")
            lines.append(ex)
            lines.append("")
        lines.append(tail)
        return "\n".join(lines)

    # Token-aware: fit as many examples as possible
    tok         = _get_tokenizer(processor)
    tail_tokens = len(tok.encode(tail))
    hdr_tokens  = len(tok.encode(header))
    budget      = max_input_tokens - tail_tokens - hdr_tokens

    kept        = []
    used_tokens = 0
    for ex, lbl in all_examples:
        block       = f"--- Example {len(kept)+1} (classification: {lbl}) ---\n{ex}\n"
        ex_tokens   = len(tok.encode(block))
        if budget > 0 and used_tokens + ex_tokens > budget:
            break
        kept.append((ex, lbl))
        used_tokens += ex_tokens

    if len(kept) < len(all_examples):
        print(f"    Prompt budget: kept {len(kept)}/{len(all_examples)} examples "
              f"({used_tokens + tail_tokens + hdr_tokens} tokens)", flush=True)

    lines = [header]
    for i, (ex, lbl) in enumerate(kept):
        lines.append(f"--- Example {i+1} (classification: {lbl}) ---")
        lines.append(ex)
        lines.append("")
    lines.append(tail)
    return "\n".join(lines)


# ── post-processing ───────────────────────────────────────────────────────────

def apply_min_run_filter(
    preds:           list[int],
    min_important:   int,
    min_unimportant: int,
    filter_order:    str = "unimportant_first",
) -> list[int]:
    """
    Post-processing: remove short runs by flipping them to their neighbours.

    Any contiguous run of 1s shorter than min_important is set to 0.
    Any contiguous run of 0s shorter than min_unimportant is set to 1.

    filter_order controls which pass runs first:
      "unimportant_first" (default): fill short non-important gaps first,
        then remove short isolated important runs. Best when the main issue
        is small holes within contiguous important regions (e.g. segmentation
        model leaving single-DA gaps inside a run).
      "important_first": remove short important bursts first, then fill gaps.
        Better when the main issue is isolated false-positive spikes.
    """
    result = list(preds)

    if filter_order == "unimportant_first":
        passes = [(0, min_unimportant), (1, min_important)]
    else:
        passes = [(1, min_important), (0, min_unimportant)]

    for target, min_len in passes:
        i = 0
        while i < len(result):
            if result[i] == target:
                j = i
                while j < len(result) and result[j] == target:
                    j += 1
                run_len = j - i
                if run_len < min_len:
                    flip = 1 - target
                    for k in range(i, j):
                        result[k] = flip
                i = j
            else:
                i += 1

    return result


# ── model loading ─────────────────────────────────────────────────────────────

def _is_multimodal(model_id: str) -> bool:
    """
    Returns True if the model should be loaded with AutoModelForImageTextToText
    + AutoProcessor (multimodal). Returns False for text-only models which use
    AutoModelForCausalLM + AutoTokenizer.

    Gemma 4 E2B and E4B are multimodal (image+audio).
    Gemma 4 26B-A4B and 31B are text-only.
    All other models default to text-only (CausalLM).
    """
    mid = model_id.lower()
    # Gemma 4 small multimodal variants
    if re.search(r"gemma.4.e[24]b", mid):
        return True
    # Everything else: text-only
    return False


def _get_tokenizer(processor):
    """
    Return a tokenizer-like object from either an AutoProcessor
    (multimodal — has .tokenizer attribute) or an AutoTokenizer
    (text-only — is already a tokenizer).
    """
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def load_model(model_id: str, hf_cache_dir: str | None = None):
    """
    Load model + tokenizer/processor. Returns (model, processor) tuple.

    Multimodal models (Gemma 4 E2B/E4B):
        AutoModelForImageTextToText + AutoProcessor

    Text-only models (Gemma 4 26B-A4B, 31B, Qwen, Llama, etc.):
        AutoModelForCausalLM + AutoTokenizer

    Both return the same (model, processor) interface — downstream code
    uses _get_tokenizer(processor) to get the tokenizer in either case,
    and apply_chat_template is available on both AutoProcessor and
    AutoTokenizer.
    """
    import torch

    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if hf_cache_dir:
        os.environ["HF_HOME"]            = hf_cache_dir
        os.environ["TRANSFORMERS_CACHE"] = hf_cache_dir
        os.environ["HF_DATASETS_CACHE"]  = hf_cache_dir
        print(f"HuggingFace cache dir: {hf_cache_dir}", flush=True)

    multimodal = _is_multimodal(model_id)
    print(f"Model type: {'multimodal (ImageTextToText)' if multimodal else 'text-only (CausalLM)'}",
          flush=True)

    if multimodal:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        print(f"Loading processor: {model_id} …", flush=True)
        processor = AutoProcessor.from_pretrained(model_id)
        print(f"Loading model: {model_id} in float16 …", flush=True)
        model = AutoModelForImageTextToText.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        ).eval()
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        print(f"Loading tokenizer: {model_id} …", flush=True)
        processor = AutoTokenizer.from_pretrained(model_id)
        print(f"Loading model: {model_id} in float16 …", flush=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            torch_dtype=torch.float16,
            device_map="auto",
        ).eval()

    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}", flush=True)

    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved  = torch.cuda.memory_reserved()  / 1024**3
        print(f"GPU memory after load: {allocated:.1f}GB allocated  "
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
    """
    Run inference. Works with both multimodal (AutoProcessor) and
    text-only (AutoTokenizer) models.

    For multimodal models, content is wrapped as typed dicts
    ({type: text, text: ...}) as required by AutoProcessor.
    For text-only models, content is passed as plain strings
    as required by AutoTokenizer.apply_chat_template.
    """
    import torch

    model, processor  = model_and_processor
    tok               = _get_tokenizer(processor)
    full_prompt       = prompt if not retry_note else f"{prompt}{retry_note}"
    is_multimodal_proc = hasattr(processor, "tokenizer")

    if max_input_tokens > 0:
        tokens = tok.encode(full_prompt)
        if len(tokens) > max_input_tokens:
            logger.warning(f"Prompt still {len(tokens)} tokens after example "
                           f"trimming (budget={max_input_tokens})")

    if is_multimodal_proc:
        # AutoProcessor expects typed content dicts
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]},
            {"role": "user",   "content": [{"type": "text", "text": full_prompt}]},
        ]
    else:
        # AutoTokenizer expects plain string content
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": full_prompt},
        ]

    inputs = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len  = inputs["input_ids"].shape[-1]
    gen_kwargs = dict(max_new_tokens=max_tokens, do_sample=temperature > 0.0)
    if temperature > 0.0:
        gen_kwargs["temperature"] = temperature
        gen_kwargs["top_p"]       = 0.95

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_tokens = output_ids[0][input_len:]
    response   = tok.decode(new_tokens, skip_special_tokens=True).strip()
    logger.debug(f"Response: {response}")
    return response


def parse_prediction(response: str) -> int | None:
    """
    Parse response to 0/1. Returns None if unparseable.
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


def _extract_ngrams(seq: list[str], n: int) -> list[tuple]:
    """Extract all n-grams from a DA sequence."""
    return [tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)]


def _ngram_js(seqs_a: list[list[str]], seqs_b: list[list[str]], n: int) -> float:
    """JS divergence between n-gram distributions of two groups."""
    from collections import defaultdict
    all_ng = sorted({
        ng for seqs in (seqs_a, seqs_b)
        for seq in seqs for ng in _extract_ngrams(seq, n)
    })
    if not all_ng:
        return float("nan")

    def _vec(seqs, vocab, _n=n):
        counts = defaultdict(int)
        for seq in seqs:
            for ng in _extract_ngrams(seq, _n):
                counts[ng] += 1
        return np.array([counts.get(ng, 0) for ng in vocab], dtype=float)

    va = _vec(seqs_a, all_ng) + 1e-10
    vb = _vec(seqs_b, all_ng) + 1e-10
    va /= va.sum();  vb /= vb.sum()
    m   = 0.5 * (va + vb)
    kl  = lambda p, q: float(np.sum(p * np.log(p / q)))
    return 0.5 * kl(va, m) + 0.5 * kl(vb, m)


def _combined_ngram_js(seqs_a: list[list[str]],
                       seqs_b: list[list[str]],
                       ngram_ns: list[int]) -> float:
    """
    JS divergence over a single concatenated vector of all ngram orders.
    Counts are NOT normalised per order before concatenation so higher-order
    ngrams contribute in proportion to how often they appear.
    """
    from collections import defaultdict
    vecs_a, vecs_b = [], []
    for n in ngram_ns:
        all_ng = sorted({
            ng for seqs in (seqs_a, seqs_b)
            for seq in seqs for ng in _extract_ngrams(seq, n)
        })
        if not all_ng:
            continue

        def _vec(seqs, vocab, _n=n):
            counts = defaultdict(int)
            for seq in seqs:
                for ng in _extract_ngrams(seq, _n):
                    counts[ng] += 1
            return np.array([counts.get(ng, 0) for ng in vocab], dtype=float)

        vecs_a.append(_vec(seqs_a, all_ng))
        vecs_b.append(_vec(seqs_b, all_ng))

    if not vecs_a:
        return float("nan")

    va = np.concatenate(vecs_a) + 1e-10
    vb = np.concatenate(vecs_b) + 1e-10
    va /= va.sum();  vb /= vb.sum()
    m   = 0.5 * (va + vb)
    kl  = lambda p, q: float(np.sum(p * np.log(p / q)))
    return 0.5 * kl(va, m) + 0.5 * kl(vb, m)


def compute_ngram_ceiling(
    train_transcripts: dict[str, dict],
    test_transcripts:  dict[str, dict],
    ngram_ns:          list[int] = list(range(4, 14)),
) -> dict:
    """
    Naive Bayes ngram ceiling classifier.

    Uses the proper train/test split — no leakage.

    TRAIN:
      Extract all contiguous important and non-important DA sequences.
      Build smoothed ngram frequency distributions for each class
      (Laplace smoothing with alpha=1 to handle unseen ngrams).

    TEST:
      For each DA position, extract ngrams from a centred window of size
      max(ngram_ns) — just large enough to extract the largest ngram.
      Centring: ceil((max_n - 1) / 2) before, floor((max_n - 1) / 2) after.
      Edge positions are padded with a <PAD> token.

      Score each position against both class distributions:
        score = sum of log P(ngram | class) over all ngrams in window
      Predict the class with the higher log-likelihood score.

    EVALUATE:
      Per-DA F1(important) against true test labels.
      This gives an empirical upper bound on what ngram-based features
      can achieve with this train/test split.
    """
    import math
    from collections import defaultdict

    max_n      = max(ngram_ns)
    ctx_before = math.ceil((max_n - 1) / 2)
    ctx_after  = math.floor((max_n - 1) / 2)
    PAD        = "<PAD>"

    # ── build class distributions from train set ──────────────────────────────
    imp_counts:   dict[int, dict[tuple, int]] = {n: defaultdict(int) for n in ngram_ns}
    nonim_counts: dict[int, dict[tuple, int]] = {n: defaultdict(int) for n in ngram_ns}
    imp_total:    dict[int, int]              = {n: 0 for n in ngram_ns}
    nonim_total:  dict[int, int]              = {n: 0 for n in ngram_ns}

    for rec in train_transcripts.values():
        das    = rec["das"]
        labels = rec["labels"]
        nt     = len(labels)

        i = 0
        while i < nt:
            if labels[i] == 1:
                j = i
                while j < nt and labels[j] == 1:
                    j += 1
                seq = das[i:j]
                for n in ngram_ns:
                    for ng in _extract_ngrams(seq, n):
                        imp_counts[n][ng] += 1
                        imp_total[n]      += 1
                i = j
            else:
                j = i
                while j < nt and labels[j] == 0:
                    j += 1
                seq = das[i:j]
                for n in ngram_ns:
                    for ng in _extract_ngrams(seq, n):
                        nonim_counts[n][ng] += 1
                        nonim_total[n]      += 1
                i = j

    # Collect all vocab per order for Laplace smoothing
    vocab: dict[int, set] = {
        n: set(imp_counts[n]) | set(nonim_counts[n])
        for n in ngram_ns
    }

    n_train_imp   = sum(1 for rec in train_transcripts.values()
                        for l in rec["labels"] if l == 1)
    n_train_nonim = sum(1 for rec in train_transcripts.values()
                        for l in rec["labels"] if l == 0)
    n_train_total = n_train_imp + n_train_nonim

    # Class log-priors
    log_prior_imp   = np.log(n_train_imp   / max(n_train_total, 1) + 1e-10)
    log_prior_nonim = np.log(n_train_nonim / max(n_train_total, 1) + 1e-10)

    print(f"  Train: {n_train_imp} important DAs  "
          f"{n_train_nonim} non-important DAs", flush=True)
    for n in ngram_ns:
        print(f"    {n}-gram vocab: {len(vocab[n])}  "
              f"imp_total={imp_total[n]}  nonim_total={nonim_total[n]}",
              flush=True)

    def _log_likelihood(ng: tuple, n: int, counts: dict, total: int,
                        vsize: int) -> float:
        """Laplace-smoothed log P(ngram | class)."""
        return np.log((counts.get(ng, 0) + 1) / (total + vsize + 1e-10))

    # ── predict on test set ───────────────────────────────────────────────────
    y_true: list[int] = []
    y_pred: list[int] = []

    for fname, rec in test_transcripts.items():
        das    = rec["das"]
        labels = rec["labels"]
        nt     = len(das)

        for i in range(nt):
            # Build centred window with padding
            window = []
            for offset in range(-ctx_before, ctx_after + 1):
                idx = i + offset
                window.append(das[idx] if 0 <= idx < nt else PAD)

            # Score against each class
            score_imp   = log_prior_imp
            score_nonim = log_prior_nonim

            for n in ngram_ns:
                vsize = len(vocab[n])
                for ng in _extract_ngrams(window, n):
                    score_imp   += _log_likelihood(
                        ng, n, imp_counts[n],   imp_total[n],   vsize)
                    score_nonim += _log_likelihood(
                        ng, n, nonim_counts[n], nonim_total[n], vsize)

            y_pred.append(1 if score_imp >= score_nonim else 0)
            y_true.append(labels[i])

    if not y_true:
        print("  Warning: no test predictions made.", flush=True)
        return {}

    # ── evaluate ──────────────────────────────────────────────────────────────
    from sklearn.metrics import f1_score, confusion_matrix
    cm    = confusion_matrix(y_true, y_pred, labels=[0, 1])
    TP    = int(cm[1, 1]);  TN = int(cm[0, 0])
    FP    = int(cm[0, 1]);  FN = int(cm[1, 0])
    f1_imp  = f1_score(y_true, y_pred, pos_label=1,
                       average="binary", zero_division=0)
    f1_bal  = f1_score(y_true, y_pred, average="macro", zero_division=0)
    prec    = TP / (TP + FP) if (TP + FP) > 0 else 0.0
    rec     = TP / (TP + FN) if (TP + FN) > 0 else 0.0

    print(f"\n  Ngram NB ceiling (train→test):", flush=True)
    print(f"    window={max_n} (ctx_before={ctx_before} ctx_after={ctx_after})",
          flush=True)
    print(f"    TP={TP}  TN={TN}  FP={FP}  FN={FN}", flush=True)
    print(f"    F1(imp)={f1_imp:.4f}  F1(bal)={f1_bal:.4f}  "
          f"prec={prec:.4f}  rec={rec:.4f}", flush=True)

    return {
        "ceiling_nb_f1_important":  round(f1_imp, 4),
        "ceiling_nb_f1_balanced":   round(f1_bal, 4),
        "ceiling_nb_precision":     round(prec,   4),
        "ceiling_nb_recall":        round(rec,    4),
        "ceiling_nb_TP":            TP,
        "ceiling_nb_TN":            TN,
        "ceiling_nb_FP":            FP,
        "ceiling_nb_FN":            FN,
        "ceiling_nb_window":        max_n,
        "ceiling_nb_ctx_before":    ctx_before,
        "ceiling_nb_ctx_after":     ctx_after,
    }


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
    vote_threshold:      float,
    min_important_run:   int,
    min_unimportant_run: int,
    filter_order:        str,
    temperature:         float,
    max_tokens:          int,
    max_retries:         int,
    max_input_tokens:    int,
    verbose:             bool,
    summaries_dir:       str | None = None,
    summary_max_das:     int        = 200,
    codebook:            dict       | None = None,
    base_rate:           float      = 0.0,
) -> tuple[list[int], list[int], list[dict]]:
    """
    Sliding window ensemble prediction.

    For each transcript:
      1. If summaries_dir is provided, generate (or load cached) a neutral
         factual summary of the transcript and build a per-transcript system
         prompt that includes it. Otherwise use the shared system_prompt.
      2. Slide a window of window_size DAs across the transcript at
         window_stride intervals, collecting one binary vote per window.
      3. Each DA accumulates votes from every window that covers it.
      4. Vote fraction >= vote_threshold -> DA predicted important.
      5. Apply min-run-length post-processing to enforce contiguity.

    Windows at the edges of the transcript will receive fewer votes —
    this is expected and acceptable.
    """
    y_true_all: list[int]  = []
    y_pred_all: list[int]  = []
    pred_rows:  list[dict] = []

    for fname, rec in test_transcripts.items():
        n = len(rec["das"])
        print(f"\n  Transcript: {fname}  ({n} DAs)", flush=True)

        # ── per-transcript system prompt with summary ──────────────────────
        if summaries_dir is not None:
            summary = generate_transcript_summary(
                rec, fname, model_and_processor,
                summaries_dir=summaries_dir,
                max_das=summary_max_das,
            )
            active_system_prompt = build_system_prompt_with_summary(
                codebook   or {},
                base_rate,
                summary,
            )
            print(f"    System prompt with summary: "
                  f"{len(active_system_prompt)} chars", flush=True)
        else:
            active_system_prompt = system_prompt

        # vote_counts[i] = number of windows that voted important for DA i
        # vote_totals[i] = number of windows that covered DA i
        vote_counts: list[int] = [0] * n
        vote_totals: list[int] = [0] * n

        # Generate all window start positions
        starts = list(range(0, n, window_stride))
        # Ensure the last window covers the end of the transcript
        if starts[-1] + window_size < n:
            starts.append(n - window_size)

        n_windows = len(starts)
        print(f"    {n_windows} windows  "
              f"(size={window_size}  stride={window_stride})", flush=True)

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
                    prompt, active_system_prompt, model_and_processor,
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

            # Accumulate votes for all DAs in this window
            actual_end = min(start + window_size, n)
            for da_idx in range(start, actual_end):
                vote_counts[da_idx] += pred
                vote_totals[da_idx] += 1

            if verbose and (w_idx + 1) % 10 == 0:
                print(f"    [{w_idx+1}/{n_windows} windows]", flush=True)

            logger.debug(f"{fname} window {w_idx} start={start} pred={pred} "
                         f"response='{response.strip()}'")

        # ── threshold votes to get raw per-DA predictions ─────────────────────
        raw_preds = [
            1 if (vote_totals[i] > 0 and
                  vote_counts[i] / vote_totals[i] >= vote_threshold)
            else 0
            for i in range(n)
        ]

        # ── apply min-run-length post-processing ──────────────────────────────
        final_preds = apply_min_run_filter(
            raw_preds, min_important_run, min_unimportant_run,
            filter_order=filter_order,
        )

        n_raw_pos   = sum(raw_preds)
        n_final_pos = sum(final_preds)
        print(f"  Raw predictions:   {n_raw_pos}/{n} important", flush=True)
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
                "filename":    fname,
                "patient_id":  rec["patient_id"],
                "position":    i,
                "da":          rec["das"][i],
                "speaker":     rec["speakers"][i],
                "text":        rec["texts"][i],
                "label":       label,
                "pred":        pred,
                "pred_raw":    raw_preds[i],
                "vote_count":  vote_counts[i],
                "vote_total":  vote_totals[i],
                "vote_frac":   round(vote_counts[i] / max(vote_totals[i], 1), 4),
            })

        print(f"  Done: {fname}  —  "
              f"true_pos={transcript_true_pos}  "
              f"pred_pos={transcript_pred_pos}", flush=True)

    return y_true_all, y_pred_all, pred_rows


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Sliding-window ensemble LLM importance classifier.\n"
            "Each window is classified as important/not important, votes are\n"
            "accumulated per DA, thresholded, then post-processed with a\n"
            "minimum run-length filter to enforce contiguous regions."
        )
    )

    parser.add_argument("--dir",           required=True,
                        help="Directory containing transcript CSV/TSV/XLSX files.")
    parser.add_argument("--granularity",   default="groups",
                        choices=["groups", "raw"])
    parser.add_argument("--target",        default="patient",
                        choices=["patient", "therapist"])
    parser.add_argument("--text_col",      required=True,
                        help="Column containing spoken text for each DA.")

    parser.add_argument("--codebook",      default=None,
                        help="Path to codebook CSV with 'Abbreviation' and "
                             "'comment' columns. Used to describe importance "
                             "codes in the system prompt. (default: None)")
    parser.add_argument("--n_train_patients", type=int, default=5,
                        help="Patients used for few-shot example pool. "
                             "(default: 5)")
    parser.add_argument("--n_few_shot",    type=int, default=16,
                        help="Total few-shot examples. -1 = all available. "
                             "(default: 16)")
    parser.add_argument("--pos_proportion", type=float, default=2/3,
                        help="Proportion of few-shot examples that are positive. "
                             "(default: 0.667)")

    parser.add_argument("--max_context_das", type=int, default=5,
                        help="Max non-important DAs of surrounding context to "
                             "include around the important block in positive "
                             "examples. Keeps the majority of the example "
                             "window important. (default: 5)")
    parser.add_argument("--window_size",   type=int, default=20,
                        help="Number of DAs per sliding window. (default: 20)")
    parser.add_argument("--window_stride", type=int, default=5,
                        help="Stride between window start positions. A stride of "
                             "window_size//4 gives ~4-5 votes per DA. (default: 5)")
    parser.add_argument("--vote_threshold", type=float, default=0.5,
                        help="Vote fraction >= this -> DA predicted important. "
                             "(default: 0.5, above-inclusive)")

    parser.add_argument("--min_important_run",   type=int, default=10,
                        help="Minimum contiguous run of important DAs. Shorter "
                             "runs are flipped to not important. (default: 10)")
    parser.add_argument("--min_unimportant_run", type=int, default=1,
                        help="Minimum contiguous run of not-important DAs. "
                             "Shorter runs are flipped to important. (default: 1)")
    parser.add_argument("--filter_order", default="unimportant_first",
                        choices=["unimportant_first", "important_first"],
                        help="Order of min-run filter passes. "
                             "'unimportant_first' fills short gaps first then "
                             "removes short important bursts — best for "
                             "segmentation where small holes are common. "
                             "'important_first' removes isolated spikes first. "
                             "(default: unimportant_first)")

    parser.add_argument("--model_id",    default="google/gemma-4-E4B-it",
                        help="HuggingFace model ID. (default: google/gemma-4-E4B-it)")
    parser.add_argument("--hf_cache_dir", default=None,
                        help="HuggingFace cache directory.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature. (default: 0.0)")
    parser.add_argument("--max_tokens",  type=int,   default=10,
                        help="Max new tokens for response. (default: 10)")
    parser.add_argument("--max_input_tokens", type=int, default=4096,
                        help="Max prompt tokens. 0 = no limit. (default: 4096)")
    parser.add_argument("--max_retries", type=int,   default=3,
                        help="Max retries for unparseable responses. (default: 3)")
    parser.add_argument("--ngram_ns",    type=str,   default="4,5,6,7,8,9,10,11,12,13",
                        help="Comma-separated ngram sizes for ceiling estimate. "
                             "(default: 4 through 13)")

    parser.add_argument("--use_summary", action="store_true",
                        help="Generate a neutral per-transcript summary and include "
                             "it in the system prompt for all windows of that "
                             "transcript. Summaries are cached to "
                             "{outdir}/summaries/ and reused on reruns.")
    parser.add_argument("--summary_max_das", type=int, default=200,
                        help="Max DAs to include in transcript summary prompt. "
                             "Lower values reduce memory usage. (default: 200)")
    parser.add_argument("--outdir",  default="llm_output/")
    parser.add_argument("--verbose", action="store_true",
                        help="Print progress every 10 windows.")
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

    print(f"Sliding-Window LLM Importance Classifier", flush=True)
    print(f"target={args.target}  granularity={args.granularity}", flush=True)
    print(f"text_col={args.text_col}", flush=True)
    print(f"window_size={args.window_size}  window_stride={args.window_stride}  "
          f"vote_threshold={args.vote_threshold}", flush=True)
    print(f"min_important_run={args.min_important_run}  "
          f"min_unimportant_run={args.min_unimportant_run}", flush=True)
    print(f"n_train_patients={args.n_train_patients}  "
          f"n_few_shot={args.n_few_shot}  "
          f"pos_proportion={args.pos_proportion:.2f}", flush=True)
    print(f"model_id={args.model_id}", flush=True)
    print(f"codebook={args.codebook}", flush=True)
    print(f"hf_cache_dir={args.hf_cache_dir}", flush=True)
    print(f"max_retries={args.max_retries}  max_input_tokens={args.max_input_tokens}",
          flush=True)

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

    # ── build examples ────────────────────────────────────────────────────────
    print(f"\nBuilding few-shot examples …", flush=True)
    rng = random.Random(SEED)
    pos_examples, neg_examples = build_examples_from_transcripts(
        train_transcripts, args.window_size, args.n_few_shot, rng,
        pos_proportion=args.pos_proportion,
        max_context_das=args.max_context_das,
    )

    # ── load codebook and compute base rate ──────────────────────────────────
    print(f"\nLoading codebook …", flush=True)
    codebook  = load_codebook(args.codebook) if args.codebook else {}
    if not codebook:
        print("  No codebook provided — system prompt will not include code descriptions.",
              flush=True)

    base_rate = compute_base_rate(train_transcripts)
    print(f"  Training base rate: {base_rate*100:.1f}% important DAs", flush=True)

    system_prompt = build_system_prompt(codebook, base_rate)
    print(f"  System prompt built ({len(system_prompt)} chars)", flush=True)

    # Sanity check: expected votes per DA
    expected_votes = args.window_size // args.window_stride
    print(f"\nExpected votes per DA (mid-transcript): ~{expected_votes}", flush=True)

    # ── load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model {args.model_id} …", flush=True)
    model_and_processor = load_model(args.model_id, args.hf_cache_dir)
    print(f"Model ready.", flush=True)

    # ── predict ───────────────────────────────────────────────────────────────
    total_das = sum(len(r["labels"]) for r in test_transcripts.values())
    print(f"\nRunning predictions on {len(test_transcripts)} test transcripts "
          f"({total_das} DAs) …", flush=True)

    summaries_dir = (
        os.path.join(args.outdir, "summaries") if args.use_summary else None
    )
    if summaries_dir:
        print(f"\nTranscript summaries will be saved to: {summaries_dir}",
              flush=True)

    y_true, y_pred, pred_rows = run_predictions(
        test_transcripts=test_transcripts,
        pos_examples=pos_examples,
        neg_examples=neg_examples,
        model_and_processor=model_and_processor,
        system_prompt=system_prompt,
        window_size=args.window_size,
        window_stride=args.window_stride,
        vote_threshold=args.vote_threshold,
        min_important_run=args.min_important_run,
        min_unimportant_run=args.min_unimportant_run,
        filter_order=args.filter_order,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        max_input_tokens=args.max_input_tokens,
        verbose=args.verbose,
        summaries_dir=summaries_dir,
        summary_max_das=args.summary_max_das,
        codebook=codebook,
        base_rate=base_rate,
    )

    # ── evaluate ──────────────────────────────────────────────────────────────
    metrics = evaluate(y_true, y_pred)

    # ── ngram ceiling estimate on test set ────────────────────────────────────
    print(f"\nComputing ngram ceiling on test set …", flush=True)
    ngram_ns  = [int(s.strip()) for s in args.ngram_ns.split(",") if s.strip()]
    ceiling   = compute_ngram_ceiling(train_transcripts, test_transcripts, ngram_ns)

    # ── save ──────────────────────────────────────────────────────────────────
    ts    = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    label = (
        f"{args.target}_{args.granularity}"
        f"_ws{args.window_size}_st{args.window_stride}"
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
            "vote_threshold":      args.vote_threshold,
            "min_important_run":   args.min_important_run,
            "min_unimportant_run": args.min_unimportant_run,
            "n_few_shot":          args.n_few_shot,
            "pos_proportion":      args.pos_proportion,
            "n_train_patients":    args.n_train_patients,
            "model_id":            args.model_id,
            "temperature":         args.temperature,
            "n_test_das":          len(y_true),
            "n_test_pos":          sum(y_true),
            "n_examples_used":     len(pos_examples) + len(neg_examples),
            "max_context_das":     args.max_context_das,
            "codebook":            args.codebook,
            "base_rate_train":     round(base_rate, 4),
            "n_codes_in_prompt":   len(codebook),
            "max_retries":         args.max_retries,
            "use_summary":         args.use_summary,
            "n_retried":           0,  # window-level retries not tracked per DA
            **metrics,
            **ceiling,
        }, f, indent=2)
    print(f"  Saved: {metrics_path}", flush=True)

    print(f"\nDone. Outputs in: {args.outdir}", flush=True)


if __name__ == "__main__":
    main()