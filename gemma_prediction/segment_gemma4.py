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
import math
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

# Re-use helpers from the sliding window script
from gemma_prediction.slide_gemma4 import (
    parse_patient_id,
    _normalise_speaker,
    _is_multimodal,
    _get_tokenizer,
    load_transcripts,
    split_by_patient,
    apply_min_run_filter,
    load_model,
    generate_prediction,
    evaluate,
    compute_ngram_ceiling,
    _extract_ngrams,
    _ngram_js,
    _combined_ngram_js,
    load_codebook,
    compute_base_rate,
    generate_transcript_summary,
    build_system_prompt_with_summary,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_PROMPT,
)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

logger = logging.getLogger(__name__)

START_TOKEN = "[START OF TRANSCRIPT]"
END_TOKEN   = "[END OF TRANSCRIPT]"


# ── stats from training data ──────────────────────────────────────────────────

def compute_avg_importance_length(train_transcripts: dict[str, dict]) -> float:
    """
    Compute the average length (in DAs) of contiguous important runs
    across all training transcripts.
    """
    lengths = []
    for rec in train_transcripts.values():
        labels = rec["labels"]
        n      = len(labels)
        i      = 0
        while i < n:
            if labels[i] == 1:
                j = i
                while j < n and labels[j] == 1:
                    j += 1
                lengths.append(j - i)
                i = j
            else:
                i += 1
    return float(np.mean(lengths)) if lengths else 0.0


# ── formatting ────────────────────────────────────────────────────────────────

def format_da_line(
    speaker: str,
    da:      str,
    text:    str,
    idx:     int | None = None,
) -> str:
    """
    Format a single DA line. If idx is provided, prefix with the
    window-relative index so the model can refer to it in its answer.
    """
    prefix = f"[{idx}] " if idx is not None else ""
    return f'{prefix}{speaker.capitalize()}: [{da}] "{text}"'


# ── window builder ────────────────────────────────────────────────────────────

def build_window(
    rec:         dict,
    start:       int,
    window_size: int,
) -> str:
    """
    Build a window of DAs with window-relative indices shown.
    Labels NOT shown. Edge positions use START/END tokens.
    """
    das      = rec["das"]
    texts    = rec["texts"]
    speakers = rec["speakers"]
    n        = len(das)
    lines    = []

    for offset in range(window_size):
        idx = start + offset
        if idx < 0:
            lines.append(f"[{offset}] {START_TOKEN}")
        elif idx >= n:
            lines.append(f"[{offset}] {END_TOKEN}")
        else:
            lines.append(format_da_line(
                speakers[idx], das[idx], texts[idx], idx=offset
            ))

    return "\n".join(lines)


# ── few-shot example builders ─────────────────────────────────────────────────

def build_examples_from_transcripts(
    train_transcripts: dict[str, dict],
    window_size:       int,
    n_few_shot:        int,
    rng:               random.Random,
    pos_proportion:    float = 2 / 3,
    max_context_das:   int   = 5,
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """
    Build positive and negative few-shot examples.

    Returns lists of (window_text, answer_str) tuples where:
      - Positive: answer_str = "A-B" (window-relative indices of important block)
      - Negative: answer_str = "not important"

    The important block is randomly shifted within the context budget so it
    appears at different positions within the window — the model must learn
    to identify where importance starts and ends, not just whether the
    window is important.
    """
    pos_examples: list[tuple[str, str]] = []
    neg_examples: list[tuple[str, str]] = []

    for rec in train_transcripts.values():
        das      = rec["das"]
        texts    = rec["texts"]
        speakers = rec["speakers"]
        labels   = rec["labels"]
        n        = len(labels)

        # ── positive: important block + random context shift ──────────────────
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

                # Window-relative indices of the important block
                block_start_rel = before_take          # = i - start
                block_end_rel   = before_take + (j - i) - 1  # inclusive

                lines = [
                    format_da_line(
                        speakers[k], das[k], texts[k], idx=k - start
                    )
                    for k in range(start, end)
                ]
                answer = f"{block_start_rel}-{block_end_rel}"
                pos_examples.append(("\n".join(lines), answer))
                i = j
            else:
                i += 1

        # ── negative: fully non-important window ──────────────────────────────
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
                        format_da_line(
                            speakers[k], das[k], texts[k], idx=k - start
                        )
                        for k in range(start, end)
                    ]
                    neg_examples.append(("\n".join(lines), "not important"))
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


# ── system prompt ─────────────────────────────────────────────────────────────

def build_system_prompt(
    codebook:        dict[str, str],
    base_rate:       float,
    avg_imp_length:  float,
) -> str:
    """
    Build the system prompt for the segmentation task.
    Includes codebook, intentionality framing, base rate, and
    average important segment length.
    """
    lines = [
        "You are an expert behavioral psychologist analysing therapy session "
        "transcripts. You will be shown labeled examples of dialogue act windows, "
        "then a new unlabeled window to segment.",
        "",
        "IMPORTANT MOMENTS are defined as sequences of dialogue acts that contain "
        "a specific, intentional therapeutic act. They are characterised by one or "
        "more of the following codes:",
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
        "a segment important. Most discussion of relevant topics is background "
        "conversation and NOT important. Only segments where something specific "
        "and purposeful is happening should be marked as important.",
        "",
        f"BASE RATE: Approximately {base_rate*100:.0f}% of dialogue acts in a "
        f"typical session are important. The vast majority are NOT important.",
        "",
        f"SEGMENT LENGTH: Important segments typically span around "
        f"{avg_imp_length:.0f} dialogue acts. Segments much shorter than this "
        f"are unlikely to be important.",
        "",
        "TASK: Each dialogue act in the window is labeled with its index [N]. "
        "Identify the single most important contiguous segment in the window, "
        "if one exists.",
        "",
        "OUTPUT FORMAT:",
        "  - If an important segment exists: respond with the start and end "
        "indices (inclusive) in the format 'A-B', e.g. '3-9'.",
        "  - If no important segment exists: respond with 'not important'.",
        "  - Output only the range or 'not important'. Nothing else.",
        "  - The range must be within the window indices shown.",
        "  - If importance extends to the edge of the window, stop at the "
        "last visible index.",
    ]

    return "\n".join(lines)


def build_system_prompt_segmentation_with_summary(
    codebook:       dict[str, str],
    base_rate:      float,
    avg_imp_length: float,
    summary:        str,
) -> str:
    """Per-transcript system prompt with neutral summary included."""
    lines = [
        "You are an expert behavioral psychologist analysing therapy session "
        "transcripts. You will be shown labeled examples of dialogue act windows, "
        "then a new unlabeled window to segment.",
        "",
        "IMPORTANT MOMENTS are defined as sequences of dialogue acts that contain "
        "a specific, intentional therapeutic act. They are characterised by one or "
        "more of the following codes:",
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
        "a segment important.",
        "",
        "SESSION OVERVIEW:",
        "The following is a neutral factual summary of the session. Use it to "
        "understand the overall context only — do NOT use it to decide what is "
        "important.",
        "",
    ]
    for line in summary.splitlines():
        lines.append(f"  {line}")

    lines += [
        "",
        f"BASE RATE: Approximately {base_rate*100:.0f}% of dialogue acts in a "
        f"typical session are important. The vast majority are NOT important.",
        "",
        f"SEGMENT LENGTH: Important segments typically span around "
        f"{avg_imp_length:.0f} dialogue acts. Segments much shorter than this "
        f"are unlikely to be important.",
        "",
        "TASK: Each dialogue act in the window is labeled with its index [N]. "
        "Identify the single most important contiguous segment in the window, "
        "if one exists.",
        "",
        "OUTPUT FORMAT:",
        "  - If an important segment exists: respond with the start and end "
        "indices (inclusive) in the format 'A-B', e.g. '3-9'.",
        "  - If no important segment exists: respond with 'not important'.",
        "  - Output only the range or 'not important'. Nothing else.",
        "  - The range must be within the window indices shown.",
        "  - If importance extends to the edge of the window, stop at the "
        "last visible index.",
    ]

    return "\n".join(lines)


# ── prompt construction ───────────────────────────────────────────────────────

def construct_prompt(
    pos_examples:     list[tuple[str, str]],
    neg_examples:     list[tuple[str, str]],
    window_text:      str,
    window_size:      int,
    processor=None,
    max_input_tokens: int = 0,
) -> str:
    """
    Construct the segmentation prompt.

    Examples show window text with indexed DAs and the ground-truth answer
    (either "A-B" or "not important"). Target window follows with no answer.
    Examples are dropped from the end if the token budget is exceeded.
    """
    tail = "\n".join([
        "Now identify the important segment in the following window.",
        f"Window indices run from 0 to {window_size - 1}.",
        "If an important segment exists respond with its start-end indices "
        "as 'A-B'. If not, respond with 'not important'. Nothing else.\n",
        window_text,
        "\nAnswer:",
    ])

    header = (
        "Below are examples of therapy session windows. Each DA is prefixed "
        "with its index [N]. The answer shows the important segment as a range "
        "'A-B' (inclusive window indices) or 'not important'.\n"
    )

    # Interleave pos and neg
    all_examples: list[tuple[str, str]] = []
    for p, ng in zip(pos_examples, neg_examples):
        all_examples.append(p)
        all_examples.append(ng)
    for p in pos_examples[len(neg_examples):]:
        all_examples.append(p)
    for ng in neg_examples[len(pos_examples):]:
        all_examples.append(ng)

    if not all_examples:
        return tail

    if max_input_tokens <= 0 or processor is None:
        lines = [header]
        for i, (ex, ans) in enumerate(all_examples):
            lines.append(f"--- Example {i+1} (answer: {ans}) ---")
            lines.append(ex)
            lines.append("")
        lines.append(tail)
        return "\n".join(lines)

    tok         = _get_tokenizer(processor)
    tail_tokens = len(tok.encode(tail))
    hdr_tokens  = len(tok.encode(header))
    budget      = max_input_tokens - tail_tokens - hdr_tokens

    kept:       list[tuple[str, str]] = []
    used_tokens = 0
    for ex, ans in all_examples:
        block     = f"--- Example {len(kept)+1} (answer: {ans}) ---\n{ex}\n"
        ex_tokens = len(tok.encode(block))
        if budget > 0 and used_tokens + ex_tokens > budget:
            break
        kept.append((ex, ans))
        used_tokens += ex_tokens

    if len(kept) < len(all_examples):
        print(f"    Prompt budget: kept {len(kept)}/{len(all_examples)} examples "
              f"({used_tokens + tail_tokens + hdr_tokens} tokens)", flush=True)

    lines = [header]
    for i, (ex, ans) in enumerate(kept):
        lines.append(f"--- Example {i+1} (answer: {ans}) ---")
        lines.append(ex)
        lines.append("")
    lines.append(tail)
    return "\n".join(lines)


# ── response parsing ──────────────────────────────────────────────────────────

def parse_segmentation(
    response:    str,
    window_size: int,
) -> tuple[int, int] | None:
    """
    Parse model response into (start, end) window-relative indices.

    Accepts:
      "not important" -> None
      "A-B"           -> (A, B)  clamped to [0, window_size-1]

    Returns None if unparseable or not important.
    """
    r = response.lower().strip()

    if re.search(r"\bnot important\b", r):
        return None

    # Match A-B pattern — tolerant of spaces around dash
    m = re.search(r"(\d+)\s*[-–]\s*(\d+)", r)
    if m:
        a = int(m.group(1))
        b = int(m.group(2))
        # Clamp to valid window range
        a = max(0, min(a, window_size - 1))
        b = max(0, min(b, window_size - 1))
        if a > b:
            a, b = b, a   # swap if reversed
        return (a, b)

    # Single index — treat as a single-DA segment
    m2 = re.search(r"\b(\d+)\b", r)
    if m2:
        idx = int(m2.group(1))
        idx = max(0, min(idx, window_size - 1))
        return (idx, idx)

    logger.warning(f"Unparseable segmentation response: '{response}'")
    return None


# ── sliding window prediction ─────────────────────────────────────────────────

def run_predictions(
    test_transcripts:    dict[str, dict],
    pos_examples:        list[tuple[str, str]],
    neg_examples:        list[tuple[str, str]],
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
    summary_max_das:     int   = 200,
    codebook:            dict | None = None,
    base_rate:           float = 0.0,
    avg_imp_length:      float = 0.0,
) -> tuple[list[int], list[int], list[dict]]:
    """
    Sliding window segmentation prediction.

    For each window the model outputs either 'not important' or 'A-B'
    (window-relative indices of the important segment). These are mapped
    back to transcript positions and each DA accumulates a vote of 1 if
    it falls within any predicted segment, 0 otherwise. Vote fraction
    >= vote_threshold -> DA predicted important. Min-run filter applied.
    """
    y_true_all: list[int]  = []
    y_pred_all: list[int]  = []
    pred_rows:  list[dict] = []

    for fname, rec in test_transcripts.items():
        n = len(rec["das"])
        print(f"\n  Transcript: {fname}  ({n} DAs)", flush=True)

        # ── per-transcript system prompt ──────────────────────────────────────
        if summaries_dir is not None:
            summary = generate_transcript_summary(
                rec, fname, model_and_processor,
                summaries_dir=summaries_dir,
                max_das=summary_max_das,
            )
            active_system_prompt = build_system_prompt_segmentation_with_summary(
                codebook or {}, base_rate, avg_imp_length, summary,
            )
        else:
            active_system_prompt = system_prompt

        vote_counts: list[int] = [0] * n
        vote_totals: list[int] = [0] * n

        starts = list(range(0, n, window_stride))
        if starts and starts[-1] + window_size < n:
            starts.append(n - window_size)

        n_windows = len(starts)
        print(f"    {n_windows} windows  "
              f"(size={window_size}  stride={window_stride})", flush=True)

        for w_idx, start in enumerate(starts):
            window_text = build_window(rec, start, window_size)
            prompt      = construct_prompt(
                pos_examples, neg_examples, window_text, window_size,
                processor=model_and_processor[1],
                max_input_tokens=max_input_tokens,
            )

            segment  = None
            response = ""
            attempt  = 0
            for attempt in range(max_retries + 1):
                retry_note = (
                    ""
                    if attempt == 0
                    else (
                        "\n\nIMPORTANT: Your previous response could not be "
                        "parsed. Respond with ONLY 'A-B' (e.g. '3-9') if an "
                        "important segment exists, or 'not important'. "
                        "Nothing else."
                    )
                )
                response = generate_prediction(
                    prompt, active_system_prompt, model_and_processor,
                    temperature, max_tokens,
                    retry_note=retry_note,
                    max_input_tokens=max_input_tokens,
                )
                segment = parse_segmentation(response, window_size)
                # None means either "not important" (valid) or unparseable
                # Distinguish by checking for "not important" in response
                if re.search(r"\bnot important\b", response.lower()):
                    break   # explicit not-important, valid
                if segment is not None:
                    break   # valid range
                if attempt < max_retries:
                    print(f"    [retry {attempt+1}/{max_retries}] "
                          f"window {w_idx} unparseable: '{response.strip()}'",
                          flush=True)
                    logger.warning(f"{fname} window {w_idx} retry {attempt+1}: "
                                   f"'{response.strip()}'")

            if segment is None and not re.search(r"\bnot important\b",
                                                  response.lower()):
                print(f"    [FAILED] window {w_idx} defaulting to not important",
                      flush=True)
                logger.error(f"{fname} window {w_idx} max retries exhausted")

            # Accumulate votes: DAs within the predicted segment get a vote of 1
            actual_end = min(start + window_size, n)
            for da_idx in range(start, actual_end):
                offset = da_idx - start
                vote_totals[da_idx] += 1
                if segment is not None and segment[0] <= offset <= segment[1]:
                    vote_counts[da_idx] += 1

            if verbose and (w_idx + 1) % 10 == 0:
                print(f"    [{w_idx+1}/{n_windows} windows]", flush=True)

            logger.debug(f"{fname} window {w_idx} start={start} "
                         f"segment={segment} response='{response.strip()}'")

        # ── threshold + post-process ──────────────────────────────────────────
        raw_preds = [
            1 if (vote_totals[i] > 0 and
                  vote_counts[i] / vote_totals[i] >= vote_threshold)
            else 0
            for i in range(n)
        ]
        final_preds = apply_min_run_filter(
            raw_preds, min_important_run, min_unimportant_run,
            filter_order=filter_order,
        )

        n_raw_pos   = sum(raw_preds)
        n_final_pos = sum(final_preds)
        print(f"  Raw predictions:      {n_raw_pos}/{n} important", flush=True)
        print(f"  After min-run filter: {n_final_pos}/{n} important", flush=True)

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
            "Sliding-window LLM importance segmentation classifier.\n"
            "The model identifies the start and end indices of important\n"
            "segments within each window (or 'not important'). Votes are\n"
            "accumulated per DA and thresholded as in the sliding window script."
        )
    )

    parser.add_argument("--dir",           required=True)
    parser.add_argument("--granularity",   default="groups",
                        choices=["groups", "raw"])
    parser.add_argument("--target",        default="patient",
                        choices=["patient", "therapist"])
    parser.add_argument("--text_col",      required=True)

    parser.add_argument("--codebook",         default=None)
    parser.add_argument("--n_train_patients", type=int, default=5)
    parser.add_argument("--n_few_shot",       type=int, default=16,
                        help="-1 = all available. (default: 16)")
    parser.add_argument("--pos_proportion",   type=float, default=2/3)
    parser.add_argument("--max_context_das",  type=int,   default=5)

    parser.add_argument("--window_size",      type=int,   default=30,
                        help="Larger windows give the model more segmentation "
                             "context. (default: 30)")
    parser.add_argument("--window_stride",    type=int,   default=5)
    parser.add_argument("--vote_threshold",   type=float, default=0.5,
                        help=">= this fraction of votes -> important. (default: 0.5)")
    parser.add_argument("--min_important_run",   type=int, default=10)
    parser.add_argument("--min_unimportant_run", type=int, default=1)
    parser.add_argument("--filter_order", default="unimportant_first",
                        choices=["unimportant_first", "important_first"],
                        help="Order of min-run filter passes. "
                             "'unimportant_first' fills short gaps first — "
                             "recommended for segmentation. (default: unimportant_first)")

    parser.add_argument("--model_id",         default="google/gemma-4-E4B-it")
    parser.add_argument("--hf_cache_dir",     default=None)
    parser.add_argument("--temperature",      type=float, default=0.0)
    parser.add_argument("--max_tokens",       type=int,   default=15,
                        help="Max new tokens — slightly higher than binary "
                             "to accommodate 'A-B' format. (default: 15)")
    parser.add_argument("--max_input_tokens", type=int,   default=4096)
    parser.add_argument("--max_retries",      type=int,   default=3)
    parser.add_argument("--ngram_ns",         type=str,
                        default="4,5,6,7,8,9,10,11,12,13")

    parser.add_argument("--use_summary",  action="store_true")
    parser.add_argument("--summary_max_das", type=int, default=200,
                        help="Max DAs to include in transcript summary prompt. "
                             "Lower values reduce memory usage. (default: 200)")
    parser.add_argument("--outdir",       default="llm_seg_output/")
    parser.add_argument("--verbose",      action="store_true")
    parser.add_argument("--log",          action="store_true")

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

    print(f"Sliding-Window Segmentation LLM Classifier", flush=True)
    print(f"target={args.target}  granularity={args.granularity}", flush=True)
    print(f"window_size={args.window_size}  window_stride={args.window_stride}  "
          f"vote_threshold={args.vote_threshold}", flush=True)
    print(f"n_train_patients={args.n_train_patients}  "
          f"n_few_shot={args.n_few_shot}  "
          f"pos_proportion={args.pos_proportion:.2f}", flush=True)
    print(f"model_id={args.model_id}  use_summary={args.use_summary}", flush=True)

    # ── load ──────────────────────────────────────────────────────────────────
    transcripts = load_transcripts(
        dir_path, args.target, args.granularity, args.text_col
    )
    print(f"\nLoaded {len(transcripts)} transcripts.", flush=True)

    # ── split ─────────────────────────────────────────────────────────────────
    train_transcripts, test_transcripts = split_by_patient(
        transcripts, args.n_train_patients
    )

    # ── codebook + stats ──────────────────────────────────────────────────────
    codebook = load_codebook(args.codebook) if args.codebook else {}
    if not codebook:
        print("  No codebook — system prompt will omit code descriptions.",
              flush=True)

    base_rate      = compute_base_rate(train_transcripts)
    avg_imp_length = compute_avg_importance_length(train_transcripts)
    print(f"  Base rate: {base_rate*100:.1f}%  "
          f"Avg importance length: {avg_imp_length:.1f} DAs", flush=True)

    system_prompt = build_system_prompt(codebook, base_rate, avg_imp_length)
    print(f"  System prompt: {len(system_prompt)} chars", flush=True)

    # ── examples ──────────────────────────────────────────────────────────────
    print(f"\nBuilding few-shot examples …", flush=True)
    rng = random.Random(SEED)
    pos_examples, neg_examples = build_examples_from_transcripts(
        train_transcripts, args.window_size, args.n_few_shot, rng,
        pos_proportion=args.pos_proportion,
        max_context_das=args.max_context_das,
    )

    # ── model ─────────────────────────────────────────────────────────────────
    print(f"\nLoading model {args.model_id} …", flush=True)
    model_and_processor = load_model(args.model_id, args.hf_cache_dir)
    print(f"Model ready.", flush=True)

    # ── summaries dir ─────────────────────────────────────────────────────────
    summaries_dir = (
        os.path.join(args.outdir, "summaries") if args.use_summary else None
    )

    # ── predict ───────────────────────────────────────────────────────────────
    total_das = sum(len(r["labels"]) for r in test_transcripts.values())
    print(f"\nRunning segmentation predictions on "
          f"{len(test_transcripts)} transcripts ({total_das} DAs) …", flush=True)

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
        avg_imp_length=avg_imp_length,
    )

    # ── evaluate ──────────────────────────────────────────────────────────────
    metrics = evaluate(y_true, y_pred)

    # ── ngram ceiling ─────────────────────────────────────────────────────────
    print(f"\nComputing ngram ceiling …", flush=True)
    ngram_ns = [int(s.strip()) for s in args.ngram_ns.split(",") if s.strip()]
    ceiling  = compute_ngram_ceiling(train_transcripts, test_transcripts, ngram_ns)

    # ── save ──────────────────────────────────────────────────────────────────
    ts    = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    label = (
        f"{args.target}_{args.granularity}"
        f"_seg_ws{args.window_size}_st{args.window_stride}"
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
            "filter_order":          args.filter_order,
            "n_few_shot":          args.n_few_shot,
            "pos_proportion":      args.pos_proportion,
            "n_train_patients":    args.n_train_patients,
            "model_id":            args.model_id,
            "temperature":         args.temperature,
            "max_tokens":          args.max_tokens,
            "n_test_das":          len(y_true),
            "n_test_pos":          sum(y_true),
            "n_examples_used":     len(pos_examples) + len(neg_examples),
            "codebook":            args.codebook,
            "base_rate_train":     round(base_rate, 4),
            "avg_imp_length_train": round(avg_imp_length, 2),
            "use_summary":         args.use_summary,
            "max_retries":         args.max_retries,
            **metrics,
            **ceiling,
        }, f, indent=2)
    print(f"  Saved: {metrics_path}", flush=True)
    print(f"\nDone. Outputs in: {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
