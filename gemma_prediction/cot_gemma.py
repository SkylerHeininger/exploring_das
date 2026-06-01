"""
llm_importance_cot2.py

Two-call zero-shot chain-of-thought (CoT) sliding window importance classifier.

For each window, two sequential LLM calls are made:

  CALL 1 — REASONING
    The model receives the window (speaker + DA type + text) and is asked
    to reason through two questions:
      - CODE SCAN: which importance codes are present and why?
      - INTENTIONALITY CHECK: is this a deliberate therapeutic act or
        background discussion?
    Output is free-text reasoning with no label. The model commits to
    this assessment before seeing the classification prompt.

  CALL 2 — IMPACT + LABEL
    The model receives the original window, the call 1 reasoning as
    prior context, and (if --use_summary) the session summary. It is
    asked to:
      - Assess whether this exchange is specifically notable for
        therapist/patient growth (impact reasoning).
      - Output CLASSIFICATION: important or CLASSIFICATION: not important.
    Having already committed to the code scan and intentionality check,
    the model cannot backfill reasoning to justify a predetermined label.

Zero-shot: no few-shot examples. The 26B model is large enough that
structured zero-shot CoT works well without examples, and avoids the
complexity of generating multi-call example chains.

All data loading, model loading, splitting, evaluation, and post-processing
are imported from llm_importance_sliding.py.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import datetime
import json
import logging
import os
import random
import re

import numpy as np
import pandas as pd

from gemma_prediction.slide_gemma4 import (
    SEED,
    START_TOKEN,
    END_TOKEN,
    load_transcripts,
    split_by_patient,
    format_da_line,
    load_codebook,
    compute_base_rate,
    apply_min_run_filter,
    _is_multimodal,
    _get_tokenizer,
    load_model,
    evaluate,
    generate_prediction,
    build_transcript_text,
    compute_ngram_ceiling,
    SUMMARY_SYSTEM_PROMPT,
    SUMMARY_USER_PROMPT,
)

random.seed(SEED)
np.random.seed(SEED)

logger = logging.getLogger(__name__)


# ── avg importance length ─────────────────────────────────────────────────────


COMBINE_SUMMARY_PROMPT = (
    "Below are summaries of the first and second halves of a therapy session. "
    "Write a single combined summary in two short paragraphs:\n"
    "TOPICS: The main subjects and themes discussed across the whole session "
    "(2-3 sentences).\n"
    "ARC: How the conversation developed and shifted over the full session "
    "(2-3 sentences).\n"
    "Be neutral and factual. Do not comment on clinical significance."
)


def _count_tokens(text: str, processor) -> int:
    """Estimate token count for a string using the model's tokenizer."""
    tok = _get_tokenizer(processor)
    return len(tok.encode(text))


def _build_half_text(rec: dict, start_da: int, end_da: int) -> str:
    """Render a slice of transcript DAs as plain text."""
    das      = rec["das"]
    texts    = rec["texts"]
    speakers = rec["speakers"]
    lines    = []
    for i in range(start_da, min(end_da, len(das))):
        lines.append(
            f"{speakers[i].capitalize()}: [{das[i]}] \"{texts[i]}\""
        )
    return "\n".join(lines)


def generate_transcript_summary(
    rec:                 dict,
    fname:               str,
    model_and_processor: tuple,
    summaries_dir:       str,
    max_das:             int = 200,
    max_tokens:          int = 400,
    max_input_tokens:    int = 0,
) -> str:
    """
    Generate a neutral summary of a transcript, with automatic splitting
    if the transcript text would exceed max_input_tokens.

    Strategy:
      1. Build transcript text (up to max_das DAs).
      2. Estimate token count. If within budget, summarise in one call.
      3. If over budget, split DAs in half:
           - Summarise first half (up to 1000 tokens output)
           - Summarise second half (up to 1000 tokens output)
           - Combine both summaries in a final call (max_tokens output)
      4. Cache to {summaries_dir}/{stem}_summary.txt and reuse on reruns.
    """
    import torch

    os.makedirs(summaries_dir, exist_ok=True)
    stem         = Path(fname).stem
    summary_path = os.path.join(summaries_dir, f"{stem}_summary.txt")

    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = f.read().strip()
        print(f"    Summary loaded from cache: {summary_path}", flush=True)
        return summary

    _, processor = model_and_processor
    das          = rec["das"]
    n_das        = len(das)
    n_use        = min(n_das, max_das)

    transcript_text = _build_half_text(rec, 0, n_use)
    user_content    = f"{SUMMARY_USER_PROMPT}\n\n{transcript_text}"

    # Check if within token budget
    needs_split = False
    if max_input_tokens > 0:
        n_tokens = _count_tokens(user_content, processor)
        if n_tokens > max_input_tokens:
            needs_split = True
            print(f"    Transcript too long ({n_tokens} tokens > "
                  f"{max_input_tokens} limit) — splitting into halves",
                  flush=True)

    if not needs_split:
        # Single call — wrapped in OOM guard, falls back to split if it fails
        print(f"    Generating summary for {fname} …", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            summary = generate_prediction(
                prompt              = user_content,
                system              = SUMMARY_SYSTEM_PROMPT,
                model_and_processor = model_and_processor,
                temperature         = 0.0,
                max_tokens          = max_tokens,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"    OOM on single-call summary — falling back to split",
                  flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            needs_split = True

    if needs_split:
        # Split in half and summarise each, then combine.
        # Each call is wrapped in an OOM guard — if a half still OOMs,
        # fall back to an empty string for that half so the run can continue.
        mid = n_use // 2

        # First half
        half1_text    = _build_half_text(rec, 0, mid)
        half1_content = f"{SUMMARY_USER_PROMPT}\n\n{half1_text}"
        print(f"    Summarising first half (DAs 0-{mid}) …", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            summary_half1 = generate_prediction(
                prompt              = half1_content,
                system              = SUMMARY_SYSTEM_PROMPT,
                model_and_processor = model_and_processor,
                temperature         = 0.0,
                max_tokens          = 1000,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"    OOM on first-half summary — skipping half",
                  flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            summary_half1 = "(first half summary unavailable due to memory)"

        # Second half
        half2_text    = _build_half_text(rec, mid, n_use)
        half2_content = f"{SUMMARY_USER_PROMPT}\n\n{half2_text}"
        print(f"    Summarising second half (DAs {mid}-{n_use}) …", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            summary_half2 = generate_prediction(
                prompt              = half2_content,
                system              = SUMMARY_SYSTEM_PROMPT,
                model_and_processor = model_and_processor,
                temperature         = 0.0,
                max_tokens          = 1000,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"    OOM on second-half summary — skipping half",
                  flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            summary_half2 = "(second half summary unavailable due to memory)"

        # Combine
        combine_content = (
            f"{COMBINE_SUMMARY_PROMPT}\n\n"
            f"FIRST HALF SUMMARY:\n{summary_half1}\n\n"
            f"SECOND HALF SUMMARY:\n{summary_half2}"
        )
        print(f"    Combining half summaries …", flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        try:
            summary = generate_prediction(
                prompt              = combine_content,
                system              = SUMMARY_SYSTEM_PROMPT,
                model_and_processor = model_and_processor,
                temperature         = 0.0,
                max_tokens          = max_tokens,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"    OOM on combine step — using concatenated half summaries",
                  flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            summary = f"{summary_half1}\n\n{summary_half2}"

    # Save
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write(f"Transcript: {fname}\n")
        f.write("=" * 60 + "\n\n")
        f.write(summary)
    print(f"    Summary saved: {summary_path}", flush=True)

    return summary


def compute_avg_importance_length(train_transcripts: dict[str, dict]) -> float:
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


# ── system prompts ────────────────────────────────────────────────────────────

def build_call1_system_prompt(
    codebook:       dict[str, str],
    base_rate:      float,
    avg_imp_length: float,
) -> str:
    """
    System prompt for call 1: code scan + intentionality check.
    No label is requested — the model reasons only.
    """
    lines = [
        "You are an expert behavioral psychologist analysing therapy session "
        "transcripts. You will be shown a window of dialogue acts and asked "
        "to reason about it. Do NOT give a classification yet — only reason.",
        "",
        "IMPORTANCE CODES — these define what makes a moment clinically "
        "significant:",
        "",
    ]

    if codebook:
        for abbrev, comment in codebook.items():
            lines.append(f"  {abbrev}: {comment}")
    else:
        lines.append("  (No codebook provided — use clinical judgment.)")

    lines += [
        "",
        f"BASE RATE: Approximately {base_rate*100:.0f}% of windows are "
        f"important. The vast majority are NOT important.",
        "",
        f"SEGMENT LENGTH: Important segments typically span around "
        f"{avg_imp_length:.0f} dialogue acts.",
        "",
        "For the window you are shown, answer these two questions:",
        "",
        "CODE SCAN:",
        "Which of the above codes, if any, are present in this window? "
        "For each code you identify, give one sentence explaining what "
        "specific content in the window supports that code. "
        "If no codes are present, write: None identified.",
        "",
        "INTENTIONALITY CHECK:",
        "Is this a deliberate, purposeful therapeutic act — or is it "
        "background discussion of a topic that relates to a code? "
        "One sentence. Be specific about what in the window makes it "
        "deliberate or not.",
        "",
        "Respond with only these two sections. Do not give a classification.",
    ]

    return "\n".join(lines)


def build_call2_system_prompt(
    target:         str,
    summary:        str | None = None,
) -> str:
    """
    System prompt for call 2: impact reasoning + final label.
    Receives the reasoning from call 1 as prior context.
    Optionally includes the session summary for impact anchoring.
    """
    target_desc = (
        "the therapist's deliberate therapeutic interventions"
        if target == "therapist"
        else "moments significant for the patient's growth or insight"
    )

    lines = [
        "You are an expert behavioral psychologist. You have already analysed "
        "a therapy session window and produced a reasoning assessment. "
        "Now make your final classification.",
        "",
    ]

    if summary:
        lines += [
            "SESSION OVERVIEW:",
            "The following is a neutral factual summary of this session. "
            "Use it only to judge whether the specific exchange is notable "
            "relative to the rest of the session — do NOT use it to identify "
            "topics as important.",
            "",
        ]
        for line in summary.splitlines():
            lines.append(f"  {line}")
        lines.append("")

    lines += [
        "IMPACT REASONING:",
        f"Given your reasoning above, would a reviewer specifically notice "
        f"this exchange as notable for {target_desc}? Or is it representative "
        f"of typical session content? One sentence — anchor to the specific "
        f"exchange, not the topic.",
        "",
        "FINAL CLASSIFICATION:",
        "Based on your code scan, intentionality check, and impact reasoning, "
        "classify this window.",
        "",
        "Rules:",
        "  - If no codes were identified or the exchange is not a deliberate "
        "act, classify as 'not important'.",
        "  - When in doubt, classify as 'not important'.",
        "  - End your response with exactly one of these two lines:",
        "    CLASSIFICATION: important",
        "    CLASSIFICATION: not important",
        "  - Do not add any text after the CLASSIFICATION line.",
    ]

    return "\n".join(lines)


# ── window builder ────────────────────────────────────────────────────────────

def build_window(rec: dict, start: int, window_size: int) -> str:
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


# ── two-call inference ────────────────────────────────────────────────────────

def generate_multiturn(
    system:              str,
    turns:               list[dict],
    model_and_processor: tuple,
    temperature:         float,
    max_tokens:          int,
) -> str:
    """
    Run inference with a multi-turn conversation history.

    turns is a list of {"role": "user"/"assistant", "content": str} dicts.
    The system prompt is prepended. The last turn must be from the user —
    the model generates the next assistant turn.

    Handles both multimodal (AutoProcessor) and text-only (AutoTokenizer).
    """
    import torch

    model, processor   = model_and_processor
    is_multimodal_proc = hasattr(processor, "tokenizer")
    tok                = _get_tokenizer(processor)

    if is_multimodal_proc:
        messages = [
            {"role": "system", "content": [{"type": "text", "text": system}]}
        ]
        for t in turns:
            messages.append({
                "role":    t["role"],
                "content": [{"type": "text", "text": t["content"]}],
            })
    else:
        messages = [{"role": "system", "content": system}]
        for t in turns:
            messages.append({
                "role":    t["role"],
                "content": t["content"],
            })

    inputs    = processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    ).to(model.device)

    input_len  = inputs["input_ids"].shape[-1]
    gen_kwargs = dict(
        max_new_tokens=max_tokens,
        do_sample=temperature > 0.0,
        top_p=0.95,
    )
    if temperature > 0.0:
        gen_kwargs["temperature"] = temperature

    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)

    new_tokens = output_ids[0][input_len:]
    response   = tok.decode(new_tokens, skip_special_tokens=True).strip()
    logger.debug(f"Multiturn response: {response[:120]}")
    return response


def run_two_call_window(
    window_text:          str,
    system_call1:         str,
    system_call2:         str,
    model_and_processor:  tuple,
    temperature:          float,
    max_tokens_call1:     int,
    max_tokens_call2:     int,
    max_retries:          int,
    max_reasoning_chars:  int = 800,
) -> tuple[str, str, int | None]:
    """
    Run both calls for one window.

    Call 1: window → reasoning (code scan + intentionality)
    Call 2: window + reasoning → impact + CLASSIFICATION label

    max_reasoning_chars: reasoning from call 1 is truncated to this many
      characters before being passed to call 2, capping the call 2 input
      size and reducing the risk of OOM on long call 1 outputs.

    torch.cuda.empty_cache() is called between call 1 and call 2 to free
    fragmented reserved memory before the larger call 2 allocation.

    Returns (reasoning, call2_response, pred) where pred is 0/1/None.
    """
    import torch

    # ── call 1: reasoning ────────────────────────────────────────────────────
    call1_prompt = (
        "Analyse the following therapy session window.\n\n"
        f"{window_text}\n\n"
        "Answer the CODE SCAN and INTENTIONALITY CHECK questions. "
        "Do not give a classification."
    )

    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    try:
        reasoning = generate_multiturn(
            system=system_call1,
            turns=[{"role": "user", "content": call1_prompt}],
            model_and_processor=model_and_processor,
            temperature=temperature,
            max_tokens=max_tokens_call1,
        )
    except torch.cuda.OutOfMemoryError:
        print(f"      [OOM] call 1 failed — skipping window, defaulting to 0",
              flush=True)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return "", "", 0

    # Truncate reasoning to cap call 2 input size
    if len(reasoning) > max_reasoning_chars:
        reasoning_for_call2 = reasoning[:max_reasoning_chars] + " [truncated]"
        logger.debug(f"Reasoning truncated {len(reasoning)} → "
                     f"{max_reasoning_chars} chars")
    else:
        reasoning_for_call2 = reasoning

    # Free fragmented GPU memory before the larger call 2 allocation
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # ── call 2: impact + label ───────────────────────────────────────────────
    call2_prompt = (
        "Here is the therapy session window you analysed:\n\n"
        f"{window_text}\n\n"
        "Your previous reasoning:\n"
        f"{reasoning_for_call2}\n\n"
        "Now complete the IMPACT REASONING and give your FINAL CLASSIFICATION."
    )

    pred       = None
    call2_resp = ""

    for attempt in range(max_retries + 1):
        retry_suffix = (
            "" if attempt == 0 else
            "\n\nIMPORTANT: End your response with exactly one of:\n"
            "CLASSIFICATION: important\n"
            "CLASSIFICATION: not important"
        )

        try:
            call2_resp = generate_multiturn(
                system=system_call2,
                turns=[
                    {"role": "user",      "content": call1_prompt},
                    {"role": "assistant", "content": reasoning_for_call2},
                    {"role": "user",      "content": call2_prompt + retry_suffix},
                ],
                model_and_processor=model_and_processor,
                temperature=temperature,
                max_tokens=max_tokens_call2,
            )
        except torch.cuda.OutOfMemoryError:
            print(f"      [OOM] call 2 attempt {attempt+1} failed — "
                  f"defaulting to 0", flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            pred = 0
            break

        pred = parse_cot_prediction(call2_resp)
        if pred is not None:
            break

        if attempt < max_retries:
            print(f"      [retry {attempt+1}/{max_retries}] "
                  f"unparseable call2: '{call2_resp[:60].strip()}'",
                  flush=True)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if pred is None:
        print(f"      [FAILED] defaulting to 0", flush=True)
        pred = 0

    return reasoning, call2_resp, pred


# ── response parsing ──────────────────────────────────────────────────────────

def parse_cot_prediction(response: str) -> int | None:
    """
    Parse call 2 response. Searches from end of response for
    'CLASSIFICATION:' prefix, falls back to scanning whole response.
    """
    for line in reversed(response.strip().splitlines()):
        line = line.strip()
        if line.lower().startswith("classification:"):
            rest = line[len("classification:"):].strip().lower()
            if "not important" in rest:
                return 0
            if "important" in rest:
                return 1

    # Fallback
    r = response.lower()
    if re.search(r"\bnot important\b", r):
        return 0
    if re.search(r"\bimportant\b", r):
        return 1

    return None


# ── sliding window prediction ─────────────────────────────────────────────────

def run_predictions(
    test_transcripts:    dict[str, dict],
    model_and_processor: tuple,
    system_call1:        str,
    target:              str,
    window_size:         int,
    window_stride:       int,
    vote_threshold:      float,
    min_important_run:   int,
    min_unimportant_run: int,
    filter_order:        str,
    temperature:         float,
    max_tokens_call1:    int,
    max_tokens_call2:    int,
    max_retries:         int,
    max_reasoning_chars: int,
    max_input_tokens:    int,
    verbose:             bool,
    summaries_dir:       str | None,
    base_rate:           float,
    avg_imp_length:      float,
    summary_max_das:     int,
) -> tuple[list[int], list[int], list[dict]]:

    y_true_all: list[int]  = []
    y_pred_all: list[int]  = []
    pred_rows:  list[dict] = []

    for fname, rec in test_transcripts.items():
        n = len(rec["das"])
        print(f"\n  Transcript: {fname}  ({n} DAs)", flush=True)

        # Per-transcript call 2 system prompt (includes summary if enabled)
        summary = None
        if summaries_dir is not None:
            summary = generate_transcript_summary(
                rec, fname, model_and_processor,
                summaries_dir=summaries_dir,
                max_das=summary_max_das,
                max_input_tokens=max_input_tokens,
            )
            print(f"    Summary loaded ({len(summary)} chars)", flush=True)

        system_call2 = build_call2_system_prompt(target, summary)

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

            reasoning, call2_resp, pred = run_two_call_window(
                window_text=window_text,
                system_call1=system_call1,
                system_call2=system_call2,
                model_and_processor=model_and_processor,
                temperature=temperature,
                max_tokens_call1=max_tokens_call1,
                max_tokens_call2=max_tokens_call2,
                max_retries=max_retries,
                max_reasoning_chars=max_reasoning_chars,
            )

            # Log reasoning for inspection
            logger.debug(
                f"{fname} w{w_idx}  pred={pred}\n"
                f"  REASONING: {reasoning[:200]}\n"
                f"  CALL2: {call2_resp[:100]}"
            )

            # Accumulate votes
            actual_end = min(start + window_size, n)
            for da_idx in range(start, actual_end):
                vote_counts[da_idx] += pred
                vote_totals[da_idx] += 1

            if verbose and (w_idx + 1) % 10 == 0:
                print(f"    [{w_idx+1}/{n_windows} windows]", flush=True)

        # Threshold + post-process
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
        print(f"  Raw:          {n_raw_pos}/{n} important", flush=True)
        print(f"  After filter: {n_final_pos}/{n} important", flush=True)

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
            "Two-call zero-shot CoT sliding window importance classifier.\n"
            "Call 1: code scan + intentionality check (no label).\n"
            "Call 2: impact reasoning + CLASSIFICATION label."
        )
    )

    parser.add_argument("--dir",          required=True)
    parser.add_argument("--granularity",  default="groups",
                        choices=["groups", "raw"])
    parser.add_argument("--target",       default="therapist",
                        choices=["patient", "therapist"])
    parser.add_argument("--text_col",     required=True)

    parser.add_argument("--codebook",         required=True,
                        help="Path to codebook XLSX/CSV. Required — codes are "
                             "included in the call 1 system prompt.")
    parser.add_argument("--n_train_patients", type=int, default=5)

    parser.add_argument("--window_size",      type=int,   default=25)
    parser.add_argument("--window_stride",    type=int,   default=5)
    parser.add_argument("--vote_threshold",   type=float, default=0.6)
    parser.add_argument("--min_important_run",   type=int, default=5)
    parser.add_argument("--min_unimportant_run", type=int, default=5)
    parser.add_argument("--filter_order",     default="unimportant_first",
                        choices=["unimportant_first", "important_first"])

    parser.add_argument("--model_id",          default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--hf_cache_dir",      default=None)
    parser.add_argument("--temperature",       type=float, default=0.2)
    parser.add_argument("--max_tokens_call1",  type=int,   default=150,
                        help="Max tokens for call 1 (reasoning). "
                             "Needs room for code scan + intentionality. "
                             "(default: 150)")
    parser.add_argument("--max_tokens_call2",  type=int,   default=100,
                        help="Max tokens for call 2 (impact + label). "
                             "Must reach CLASSIFICATION line. (default: 100)")
    parser.add_argument("--max_input_tokens",  type=int, default=6000,
                        help="Max input tokens per LLM call. Used to detect "
                             "when transcript summary needs splitting. "
                             "(default: 6000)")
    parser.add_argument("--max_retries",       type=int,   default=2,
                        help="Retries for unparseable call 2 responses. "
                             "(default: 2)")
    parser.add_argument("--max_reasoning_chars", type=int, default=800,
                        help="Maximum characters of call 1 reasoning passed "
                             "to call 2. Caps call 2 input size to reduce "
                             "OOM risk. (default: 800)")
    parser.add_argument("--ngram_ns",          type=str,
                        default="4,5,6,7,8,9,10,11,12,13")

    parser.add_argument("--use_summary",     action="store_true")
    parser.add_argument("--summary_max_das", type=int, default=200)
    parser.add_argument("--outdir",          default="llm_cot2_output/")
    parser.add_argument("--verbose",         action="store_true")
    parser.add_argument("--log",             action="store_true")

    args = parser.parse_args()

    if args.log:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(args.outdir, exist_ok=True)
        logging.basicConfig(
            filename=os.path.join(args.outdir, f"run-{ts}.log"),
            level=logging.DEBUG,
        )

    dir_path = Path(args.dir)
    if not dir_path.exists():
        raise ValueError(f"Directory not found: {args.dir}")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Two-Call Zero-Shot CoT Classifier", flush=True)
    print(f"target={args.target}  window_size={args.window_size}  "
          f"window_stride={args.window_stride}", flush=True)
    print(f"temperature={args.temperature}  "
          f"max_tokens_call1={args.max_tokens_call1}  "
          f"max_tokens_call2={args.max_tokens_call2}", flush=True)
    print(f"model_id={args.model_id}", flush=True)

    # ── load ──────────────────────────────────────────────────────────────────
    transcripts = load_transcripts(
        dir_path, args.target, args.granularity, args.text_col
    )
    print(f"\nLoaded {len(transcripts)} transcripts.", flush=True)

    train_transcripts, test_transcripts = split_by_patient(
        transcripts, args.n_train_patients
    )

    codebook = load_codebook(args.codebook)
    if not codebook:
        print("WARNING: codebook empty — call 1 prompt will lack code "
              "descriptions.", flush=True)

    base_rate      = compute_base_rate(train_transcripts)
    avg_imp_length = compute_avg_importance_length(train_transcripts)
    print(f"  Base rate: {base_rate*100:.1f}%  "
          f"Avg importance length: {avg_imp_length:.1f} DAs", flush=True)

    # ── system prompts ────────────────────────────────────────────────────────
    system_call1 = build_call1_system_prompt(
        codebook, base_rate, avg_imp_length
    )
    print(f"  Call 1 system prompt: {len(system_call1)} chars", flush=True)
    # Call 2 system prompt is built per-transcript (includes summary if used)
    print(f"  Call 2 system prompt: built per-transcript "
          f"(use_summary={args.use_summary})", flush=True)

    # ── model ─────────────────────────────────────────────────────────────────
    print(f"\nLoading model {args.model_id} …", flush=True)
    model_and_processor = load_model(args.model_id, args.hf_cache_dir)
    print(f"Model ready.", flush=True)

    summaries_dir = (
        os.path.join(args.outdir, "summaries") if args.use_summary else None
    )

    # ── predict ───────────────────────────────────────────────────────────────
    total_das = sum(len(r["labels"]) for r in test_transcripts.values())
    print(f"\nRunning two-call CoT predictions on "
          f"{len(test_transcripts)} transcripts ({total_das} DAs) …",
          flush=True)
    print(f"  ~{2 * (total_das // args.window_stride)} LLM calls total",
          flush=True)

    y_true, y_pred, pred_rows = run_predictions(
        test_transcripts=test_transcripts,
        model_and_processor=model_and_processor,
        system_call1=system_call1,
        target=args.target,
        window_size=args.window_size,
        window_stride=args.window_stride,
        vote_threshold=args.vote_threshold,
        min_important_run=args.min_important_run,
        min_unimportant_run=args.min_unimportant_run,
        filter_order=args.filter_order,
        temperature=args.temperature,
        max_tokens_call1=args.max_tokens_call1,
        max_tokens_call2=args.max_tokens_call2,
        max_retries=args.max_retries,
        max_reasoning_chars=args.max_reasoning_chars,
        max_input_tokens=args.max_input_tokens,
        verbose=args.verbose,
        summaries_dir=summaries_dir,
        base_rate=base_rate,
        avg_imp_length=avg_imp_length,
        summary_max_das=args.summary_max_das,
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
        f"_cot2_ws{args.window_size}_st{args.window_stride}"
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
            "filter_order":        args.filter_order,
            "n_train_patients":    args.n_train_patients,
            "model_id":            args.model_id,
            "temperature":         args.temperature,
            "max_tokens_call1":    args.max_tokens_call1,
            "max_tokens_call2":    args.max_tokens_call2,
            "codebook":            args.codebook,
            "base_rate_train":     round(base_rate, 4),
            "avg_imp_length_train": round(avg_imp_length, 2),
            "use_summary":         args.use_summary,
            "max_retries":         args.max_retries,
            "max_reasoning_chars": args.max_reasoning_chars,
            "n_test_das":          len(y_true),
            "n_test_pos":          sum(y_true),
            **metrics,
            **ceiling,
        }, f, indent=2)
    print(f"  Saved: {metrics_path}", flush=True)
    print(f"\nDone. Outputs in: {args.outdir}", flush=True)


if __name__ == "__main__":
    main()