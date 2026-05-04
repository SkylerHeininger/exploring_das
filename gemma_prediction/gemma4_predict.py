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
    Split transcripts by patient ID. n_train_patients randomly selected
    for the few-shot example pool; remainder used for prediction.
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


# ── context window and example formatting ────────────────────────────────────

def format_da_line(
    speaker:        str,
    da:             str,
    text:           str,
    label:          int | None = None,
    marker:         str = "",
    predicted:      bool = False,
    use_pred_label: bool = False,
) -> str:
    """
    Format a single DA line.
    If label is provided, appends a label suffix.
    If predicted=True and use_pred_label=True, uses "predicted important" /
    "predicted not important" to distinguish model predictions from ground
    truth. If predicted=True and use_pred_label=False, uses the plain
    "important" / "not important" format.
    If marker is provided (e.g. '>>>'), prepends it for the target DA.
    """
    prefix = f"{marker} " if marker else ""
    label_str = ""
    if label is not None:
        if predicted and use_pred_label:
            label_str = (f"  ->  predicted "
                         f"{'important' if label == 1 else 'not important'}")
        else:
            label_str = f"  ->  {'important' if label == 1 else 'not important'}"
    return f'{prefix}{speaker.capitalize()}: [{da}] "{text}"{label_str}'



def build_context_window(
    rec:            dict,
    position:       int,
    context_window: int,
    prior_preds:    dict[int, int] | None = None,
    use_pred_label: bool = False,
) -> str:
    """
    Build the context window string for a target DA at `position`.

    If prior_preds is provided (a dict of {position: pred}), any DA in
    the context window that has a prior prediction is shown with that
    prediction as its label, using "predicted important/not important"
    format if use_pred_label=True.

    Only model-predicted positions are shown labeled — interpolated
    positions are NOT included in prior_preds so they never appear
    as labeled context.

    Labels are never shown for the target position itself.
    """
    das      = rec["das"]
    texts    = rec["texts"]
    speakers = rec["speakers"]
    n        = len(das)
    lines    = []
    pp       = prior_preds or {}

    lines.append("[Preceding context]")
    for offset in range(-context_window, 0):
        idx = position + offset
        if idx < 0:
            lines.append(START_TOKEN)
        elif idx in pp:
            lines.append(format_da_line(
                speakers[idx], das[idx], texts[idx],
                label=pp[idx], predicted=True, use_pred_label=use_pred_label,
            ))
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
        elif idx in pp:
            lines.append(format_da_line(
                speakers[idx], das[idx], texts[idx],
                label=pp[idx], predicted=True, use_pred_label=use_pred_label,
            ))
        else:
            lines.append(format_da_line(speakers[idx], das[idx], texts[idx]))

    return "\n".join(lines)


def build_examples_from_transcripts(
    train_transcripts:   dict[str, dict],
    context_window:      int,
    n_few_shot:          int,
    rng:                 random.Random,
    pos_proportion:      float = 2/3,
) -> list[str]:
    """
    Build few-shot examples from training transcripts.

    Positive examples (containing an important block):
      - context_window non-important DAs before the block (randomly shifted)
      - the important block itself
      - context_window non-important DAs after the block (randomly shifted)

    Negative examples (fully non-important):
      - A window of context_window DAs sampled from a contiguous non-important
        stretch, capped at context_window length, all labeled not important.
        Randomly positioned within the stretch.

    pos_proportion controls the ratio of positive to total examples.
    Default 2/3 positive, 1/3 negative.

    If n_few_shot == -1, all available examples are returned.
    """
    pos_examples: list[str] = []
    neg_examples: list[str] = []

    for rec in train_transcripts.values():
        das      = rec["das"]
        texts    = rec["texts"]
        speakers = rec["speakers"]
        labels   = rec["labels"]
        n        = len(labels)

        # ── Positive: windows around important blocks ─────────────────────────
        i = 0
        while i < n:
            if labels[i] == 1:
                j = i
                while j < n and labels[j] == 1:
                    j += 1

                max_before    = i
                max_after     = n - j
                before_budget = min(context_window, max_before)
                after_budget  = min(context_window, max_after)
                total_budget  = before_budget + after_budget

                if total_budget > 0:
                    before_take = rng.randint(
                        max(0, total_budget - after_budget),
                        min(total_budget, before_budget)
                    )
                    after_take = total_budget - before_take
                else:
                    before_take = 0
                    after_take  = 0

                start = i - before_take
                end   = j + after_take
                lines = [
                    format_da_line(speakers[k], das[k], texts[k], label=labels[k])
                    for k in range(start, end)
                ]
                pos_examples.append("\n".join(lines))
                i = j
            else:
                i += 1

        # ── Negative: windows from fully non-important stretches ──────────────
        i = 0
        while i < n:
            if labels[i] == 0:
                # Find end of non-important run
                j = i
                while j < n and labels[j] == 0:
                    j += 1
                run_len = j - i

                if run_len >= 2:
                    # Sample a window of up to context_window DAs
                    win = min(context_window, run_len)
                    # Random start within the run
                    max_start = run_len - win
                    offset    = rng.randint(0, max_start)
                    start     = i + offset
                    end       = start + win
                    lines     = [
                        format_da_line(speakers[k], das[k], texts[k], label=0)
                        for k in range(start, end)
                    ]
                    neg_examples.append("\n".join(lines))
                i = j
            else:
                i += 1

    rng.shuffle(pos_examples)
    rng.shuffle(neg_examples)

    if n_few_shot == -1:
        # Use all positives, fill remainder with negatives at the right ratio
        n_pos = len(pos_examples)
        n_neg = max(0, round(n_pos * (1 - pos_proportion) / pos_proportion))
        n_neg = min(n_neg, len(neg_examples))
    else:
        n_pos = round(n_few_shot * pos_proportion)
        n_neg = n_few_shot - n_pos
        n_pos = min(n_pos, len(pos_examples))
        n_neg = min(n_neg, len(neg_examples))

    combined = pos_examples[:n_pos] + neg_examples[:n_neg]
    rng.shuffle(combined)

    print(f"    Examples: {n_pos} positive + {n_neg} negative = {len(combined)} total",
          flush=True)
    return combined


# ── prompt construction ───────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    "You are an expert behavioral psychologist analysing therapy session "
    "transcripts. You will be shown examples of dialogue act sequences with "
    "each DA labeled as important or not important. Then you must classify a "
    "single TARGET DA marked with >>>. "
    "Most DAs are NOT important, only classify as important if the content "
    "is clinically or therapeutically significant based on the examples shown. "
    "Most regions of important and not important are contiguous and long. "
    "Answer with exactly one of: 'important' or 'not important'. "
    "Do not explain your answer."
)


def construct_prompt(
    examples:         list[str],
    target_context:   str,
    processor=None,
    max_input_tokens: int = 0,
) -> str:
    """
    Construct the few-shot prompt.

    If processor and max_input_tokens are provided, examples are dropped
    from the end until the full prompt fits within the token budget.
    The target context and classification instructions are NEVER truncated —
    only examples are dropped, from last to first.
    """
    tail = "\n".join([
        "Now classify the TARGET DA marked with >>> in the following sequence.\n"
        "Only classify the TARGET DA. "
        "Answer with exactly one of: 'important' or 'not important'.\n",
        target_context,
        "\nClassification:",
    ])

    header = (
        "Below are examples from therapy sessions showing sequences of "
        "dialogue acts, each labeled as important or not important. "
        "Use these to understand what makes a DA important.\n"
    )

    if max_input_tokens <= 0 or processor is None:
        prompt_lines = []
        if examples:
            prompt_lines.append(header)
            for i, ex in enumerate(examples):
                prompt_lines.append(f"--- Example {i+1} ---")
                prompt_lines.append(ex)
                prompt_lines.append("")
        prompt_lines.append(tail)
        return "\n".join(prompt_lines)

    # Token-aware: fit as many examples as possible before the target
    tok         = processor.tokenizer if hasattr(processor, "tokenizer") else processor
    tail_tokens = len(tok.encode(tail))
    hdr_tokens  = len(tok.encode(header))
    budget      = max_input_tokens - tail_tokens - hdr_tokens

    kept        = []
    used_tokens = 0
    for ex in examples:
        ex_tokens = len(tok.encode(f"--- Example {len(kept)+1} ---\n{ex}\n"))
        if budget > 0 and used_tokens + ex_tokens > budget:
            break
        kept.append(ex)
        used_tokens += ex_tokens

    if len(kept) < len(examples):
        print(f"    Prompt budget: kept {len(kept)}/{len(examples)} examples "
              f"({used_tokens + tail_tokens + hdr_tokens} tokens)", flush=True)

    prompt_lines = []
    if kept:
        prompt_lines.append(header)
        for i, ex in enumerate(kept):
            prompt_lines.append(f"--- Example {i+1} ---")
            prompt_lines.append(ex)
            prompt_lines.append("")
    prompt_lines.append(tail)
    return "\n".join(prompt_lines)




# ── model loading ─────────────────────────────────────────────────────────────

def load_model(model_id: str, hf_cache_dir: str | None = None):
    """
    Load AutoModelForImageTextToText + AutoProcessor from HuggingFace.
    Correct class for Gemma 4 E4B and similar multimodal models.
    Returns (model, processor) tuple.

    Loads in float16 to halve memory vs float32 default.
    Sets PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce
    memory fragmentation during long-context inference.
    """
    import torch
    from transformers import AutoProcessor, AutoModelForImageTextToText

    # Reduce CUDA memory fragmentation — recommended by PyTorch for OOM issues
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
        print(f"GPU memory after load: {allocated:.1f}GB allocated  "
              f"{reserved:.1f}GB reserved", flush=True)

    return model, processor


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
    Run inference using AutoModelForImageTextToText + AutoProcessor.
    retry_note is appended on retry attempts.
    max_input_tokens: if > 0, truncate the prompt to this many tokens
    before passing to the model to prevent OOM on long inputs.
    """
    import torch

    model, processor = model_and_processor
    full_prompt      = prompt if not retry_note else f"{prompt}{retry_note}"

    # Warn if prompt still exceeds budget after example trimming (rare)
    if max_input_tokens > 0:
        tok    = processor.tokenizer if hasattr(processor, "tokenizer") else processor
        tokens = tok.encode(full_prompt)
        if len(tokens) > max_input_tokens:
            logger.warning(f"Prompt still {len(tokens)} tokens after example "
                           f"trimming (budget={max_input_tokens})")

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

def _predict_single(
    rec:            dict,
    position:       int,
    examples:       list[str],
    model_and_processor: tuple,
    context_window: int,
    temperature:    float,
    max_tokens:     int,
    max_retries:    int,
    max_input_tokens: int,
    fname:          str,
    prior_preds:    dict[int, int] | None = None,
    use_pred_label: bool = False,
) -> tuple[int, str, int]:
    """
    Predict importance for a single DA position.
    prior_preds: dict of {position: pred} for already-predicted positions
                 that fall within the context window. Only model-predicted
                 positions should be in this dict — NOT interpolated ones.
    Returns (pred, response, n_retries).
    """
    target_context = build_context_window(
        rec, position, context_window,
        prior_preds=prior_preds,
        use_pred_label=use_pred_label,
    )
    prompt = construct_prompt(
        examples, target_context,
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
            prompt, SYSTEM_PROMPT, model_and_processor,
            temperature, max_tokens,
            retry_note=retry_note,
            max_input_tokens=max_input_tokens,
        )
        pred = parse_prediction(response)
        if pred is not None:
            break
        print(f"    [retry {attempt+1}/{max_retries}] pos={position} "
              f"unparseable: '{response.strip()}'", flush=True)
        logger.warning(f"{fname}[{position}] retry {attempt+1}: "
                       f"'{response.strip()}'")

    if pred is None:
        print(f"    [FAILED] max retries exhausted at position {position} "
              f"— defaulting to 0", flush=True)
        logger.error(f"{fname}[{position}] max retries exhausted, defaulting to 0")
        pred = 0

    return pred, response, attempt


def run_predictions(
    test_transcripts:    dict[str, dict],
    examples:            list[str],
    model_and_processor: tuple,
    context_window:      int,
    temperature:         float,
    max_tokens:          int,
    max_retries:         int,
    max_input_tokens:    int,
    stride:              int,
    use_pred_label:      bool,
    verbose:             bool,
) -> tuple[list[int], list[int], list[dict]]:
    """
    Run predictions over all test transcripts.

    stride == 0: predict every DA.
    stride >  0: predict every stride DAs, fill gaps:
      - Both sides agree (0->0 or 1->1): interpolate all DAs between.
        Interpolated positions are NOT added to prior_preds.
      - Sides disagree (0->1 or 1->0): predict each DA individually
        (no interpolation). These ARE added to prior_preds.
      - Edge DAs: predicted individually.

    Prior predictions are passed into each subsequent _predict_single call
    so the model can see what it has already decided for DAs in the context
    window. Only model-predicted positions (not interpolated) are passed.
    """
    y_true_all: list[int]  = []
    y_pred_all: list[int]  = []
    pred_rows:  list[dict] = []

    for fname, rec in test_transcripts.items():
        n = len(rec["das"])
        print(f"\n  Transcript: {fname}  ({n} DAs  stride={stride})", flush=True)

        transcript_true_pos = 0
        transcript_pred_pos = 0

        # preds/responses/retries for final output (includes interpolated)
        preds:     list[int | None] = [None] * n
        responses: list[str]        = [""]   * n
        retries:   list[int]        = [0]    * n
        sampled:   list[bool]       = [False] * n

        # prior_preds: only model-predicted positions, used for context
        prior_preds: dict[int, int] = {}

        def predict_pos(pos):
            """Predict one position and register in prior_preds."""
            pred, resp, att = _predict_single(
                rec, pos, examples, model_and_processor,
                context_window, temperature, max_tokens,
                max_retries, max_input_tokens, fname,
                prior_preds=prior_preds,
                use_pred_label=use_pred_label,
            )
            preds[pos]     = pred
            responses[pos] = resp
            retries[pos]   = att
            sampled[pos]   = True
            prior_preds[pos] = pred   # register for future context

        if stride == 0:
            sample_positions = list(range(n))
        else:
            sample_positions = list(range(0, n, stride))

        # ── predict sampled positions ─────────────────────────────────────────
        n_sampled = len(sample_positions)
        print(f"    Predicting {n_sampled} sampled positions …", flush=True)
        for idx, pos in enumerate(sample_positions):
            predict_pos(pos)
            if verbose and (idx + 1) % 10 == 0:
                n_pos_so_far = sum(1 for p in preds if p == 1)
                print(f"    [{idx+1}/{n_sampled} sampled]  "
                      f"pred_pos_so_far={n_pos_so_far}", flush=True)

        if stride > 0:
            # ── edges: predict individually ───────────────────────────────────
            for pos in range(0, sample_positions[0]):
                predict_pos(pos)
            for pos in range(sample_positions[-1] + 1, n):
                predict_pos(pos)

            # ── fill gaps between sampled positions ───────────────────────────
            for s_idx in range(len(sample_positions) - 1):
                left_pos  = sample_positions[s_idx]
                right_pos = sample_positions[s_idx + 1]
                gap       = list(range(left_pos + 1, right_pos))

                if not gap:
                    continue

                left_pred  = preds[left_pos]
                right_pred = preds[right_pos]

                if left_pred == right_pred:
                    # Agreement: interpolate — do NOT add to prior_preds
                    for pos in gap:
                        preds[pos]     = left_pred
                        responses[pos] = f"[filled:{left_pred}]"
                        retries[pos]   = 0
                        sampled[pos]   = False
                else:
                    # Disagreement: predict each individually
                    # These DO go into prior_preds for subsequent calls
                    for pos in gap:
                        predict_pos(pos)

        # ── collect results ───────────────────────────────────────────────────
        for i in range(n):
            pred  = preds[i] if preds[i] is not None else 0
            label = rec["labels"][i]
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
                "response":   responses[i].strip() if responses[i] else "",
                "n_retries":  retries[i],
                "sampled":    sampled[i],
            })

        print(f"  Done: {fname}  —  "
              f"true_pos={transcript_true_pos}  "
              f"pred_pos={transcript_pred_pos}", flush=True)

    return y_true_all, y_pred_all, pred_rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "LLM-based per-DA importance classifier.\n"
            "Few-shot examples are built from training patients using context\n"
            "windows around important blocks (with surrounding non-important\n"
            "DAs shown and labeled). Block position is randomly shifted within\n"
            "the context window to prevent positional shortcuts.\n"
            "Model: AutoModelForImageTextToText (Gemma 4 E4B)."
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
                        help="Patients used for few-shot example pool. "
                             "(default: 5)")
    parser.add_argument("--context_window",   type=int, default=5,
                        help="DAs of surrounding context shown per example "
                             "and around the target DA. (default: 5)")
    parser.add_argument("--n_few_shot",       type=int, default=-1,
                        help="Number of few-shot examples to include. "
                             "-1 = all available. (default: -1)")

    parser.add_argument("--model_id",    default="google/gemma-4-E4B-it",
                        help="HuggingFace model ID. "
                             "(default: google/gemma-4-E4B-it)")
    parser.add_argument("--hf_cache_dir", default=None,
                        help="HuggingFace cache directory. Useful on HPC.")
    parser.add_argument("--temperature", type=float, default=0.0,
                        help="Sampling temperature. 0.0 = deterministic. "
                             "(default: 0.0)")
    parser.add_argument("--max_tokens",  type=int,   default=10,
                        help="Max new tokens for LLM response. (default: 10)")
    parser.add_argument("--max_input_tokens", type=int, default=4096,
                        help="Truncate prompt to this many tokens before "
                             "passing to the model. Prevents OOM on long "
                             "prompts. 0 = no limit. (default: 4096)")
    parser.add_argument("--max_retries", type=int,   default=3,
                        help="Max retries for unparseable responses. (default: 3)")
    parser.add_argument("--use_pred_label", action="store_true",
                        help="Show prior predictions in context as "
                             "'predicted important/not important' rather than "
                             "plain 'important/not important'. Helps the model "
                             "distinguish ground-truth examples from its own "
                             "prior decisions. (default: off)")
    parser.add_argument("--stride",      type=int,   default=4,
                        help="Predict every N DAs, fill gaps by interpolation. "
                             "0 = predict every DA (original behaviour). (default: 4)")
    parser.add_argument("--pos_proportion", type=float, default=2/3,
                        help="Proportion of few-shot examples that are positive "
                             "(contain an important block). Remainder are fully "
                             "negative windows. (default: 0.667)")

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

    print(f"LLM Importance Classifier (context-window examples)", flush=True)
    print(f"target={args.target}  granularity={args.granularity}", flush=True)
    print(f"text_col={args.text_col}  context_window={args.context_window}",
          flush=True)
    print(f"n_train_patients={args.n_train_patients}  "
          f"n_few_shot={args.n_few_shot}  "
          f"pos_proportion={args.pos_proportion:.2f}", flush=True)
    print(f"stride={args.stride}  use_pred_label={args.use_pred_label}", flush=True)
    print(f"model_id={args.model_id}", flush=True)
    print(f"hf_cache_dir={args.hf_cache_dir}", flush=True)
    print(f"temperature={args.temperature}  max_retries={args.max_retries}",
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
    print(f"\nBuilding few-shot examples from training transcripts …",
          flush=True)
    rng      = random.Random(SEED)
    examples = build_examples_from_transcripts(
        train_transcripts, args.context_window, args.n_few_shot, rng,
        pos_proportion=args.pos_proportion,
    )
    print(f"  Built {len(examples)} examples "
          f"({'all available' if args.n_few_shot == -1 else str(args.n_few_shot)})",
          flush=True)

    if not examples:
        print("  Warning: no examples found in training set. "
              "Proceeding with no few-shot context.", flush=True)

    # ── load model ────────────────────────────────────────────────────────────
    print(f"\nLoading model {args.model_id} …", flush=True)
    model_and_processor = load_model(args.model_id, args.hf_cache_dir)
    print(f"Model ready.", flush=True)

    # ── predict ───────────────────────────────────────────────────────────────
    total_das = sum(len(r["labels"]) for r in test_transcripts.values())
    print(f"\nRunning predictions on {len(test_transcripts)} test transcripts "
          f"({total_das} DAs) …", flush=True)

    y_true, y_pred, pred_rows = run_predictions(
        test_transcripts=test_transcripts,
        examples=examples,
        model_and_processor=model_and_processor,
        context_window=args.context_window,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_retries=args.max_retries,
        max_input_tokens=args.max_input_tokens,
        stride=args.stride,
        use_pred_label=args.use_pred_label,
        verbose=args.verbose,
    )

    # ── evaluate ──────────────────────────────────────────────────────────────
    metrics = evaluate(y_true, y_pred)

    # ── save ──────────────────────────────────────────────────────────────────
    ts    = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
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
            "label":            label,
            "target":           args.target,
            "granularity":      args.granularity,
            "context_window":   args.context_window,
            "n_few_shot":       args.n_few_shot,
            "n_train_patients": args.n_train_patients,
            "model_id":         args.model_id,
            "temperature":      args.temperature,
            "n_test_das":       len(y_true),
            "n_test_pos":       sum(y_true),
            "n_examples_used":  len(examples),
            "stride":           args.stride,
            "use_pred_label":   args.use_pred_label,
            "pos_proportion":   args.pos_proportion,
            "max_retries":      args.max_retries,
            "n_retried":        int(sum(1 for r in pred_rows
                                       if r["n_retries"] > 0)),
            **metrics,
        }, f, indent=2)
    print(f"  Saved: {metrics_path}", flush=True)

    print(f"\nDone. Outputs in: {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
