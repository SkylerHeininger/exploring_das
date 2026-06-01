"""
Turn-level LLM importance classifier.

Converts DA-level transcript data to turn level by grouping DAs sharing the
same (timestamp, speaker) pair. Classifies each turn as important or not
important using a sliding few-shot prompt with Gemma 4.

Turn label: important if ANY DA in the turn is labeled important.
Train/test split: by patient ID (same as other pipeline scripts).
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
from collections import defaultdict

import numpy as np
import pandas as pd
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
)

from plotting.common_patterns import DA_COLUMN, load_da_level, get_label

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

logger = logging.getLogger(__name__)


def parse_patient_id(filename: str) -> str | None:
    match = re.search(r"AC(\d{2})", Path(filename).stem, re.IGNORECASE)
    return match.group(1) if match else None


def _normalise_speaker(raw: str) -> str:
    if isinstance(raw, str) and raw.strip().lower() == "therapist":
        return "therapist"
    return "patient"


def das_to_turns(df, target_col, text_col, granularity):
    df = df[df[DA_COLUMN] != "I-"].reset_index(drop=True)
    df[target_col] = df[target_col].fillna(0).astype(int)
    speakers = [_normalise_speaker(str(r.get("speaker", "patient")))
                for _, r in df.iterrows()]
    turns, turn_id, prev_key = [], 0, None
    cur_texts, cur_labels, cur_ts, cur_spkr = [], [], None, None
    for idx, row in df.iterrows():
        ts, spkr = row.get("timestamp", idx), speakers[idx]
        key = (ts, spkr)
        text, lbl = str(row.get(text_col, "")).strip(), int(row[target_col])
        if key != prev_key:
            if prev_key is not None:
                turns.append({"turn_id": turn_id, "timestamp": cur_ts,
                               "speaker": cur_spkr,
                               "text": " ".join(t for t in cur_texts if t),
                               "label": int(any(l == 1 for l in cur_labels)),
                               "n_das": len(cur_texts)})
                turn_id += 1
            cur_texts, cur_labels, cur_ts, cur_spkr, prev_key = [text], [lbl], ts, spkr, key
        else:
            cur_texts.append(text); cur_labels.append(lbl)
    if cur_texts:
        turns.append({"turn_id": turn_id, "timestamp": cur_ts, "speaker": cur_spkr,
                      "text": " ".join(t for t in cur_texts if t),
                      "label": int(any(l == 1 for l in cur_labels)),
                      "n_das": len(cur_texts)})
    return turns


def load_transcripts(dir_path, target, granularity, text_col):
    target_col, allowed_ext, transcripts = f"{target}_important", {".csv", ".tsv", ".xlsx"}, {}
    for fp in sorted(dir_path.iterdir()):
        if fp.suffix.lower() not in allowed_ext: continue
        patient_id = parse_patient_id(fp.name)
        if patient_id is None:
            print(f"Warning: could not parse patient ID from {fp.name} — skipping.", flush=True); continue
        print(f"Loading {fp.name}  (patient={patient_id}) ...", flush=True)
        df = load_da_level(fp)
        missing = [c for c in (target_col, text_col, "timestamp") if c not in df.columns]
        if missing:
            print(f"  Warning: missing columns {missing} — skipping.", flush=True); continue
        turns = das_to_turns(df, target_col, text_col, granularity)
        n_pos = sum(t["label"] for t in turns)
        print(f"  {len(turns)} turns  {n_pos} important ({100*n_pos/max(len(turns),1):.1f}%)", flush=True)
        transcripts[fp.name] = {"patient_id": patient_id, "turns": turns}
    return transcripts


def split_by_patient(transcripts, n_train_patients=5):
    patient_to_files = defaultdict(list)
    for fname, rec in transcripts.items():
        patient_to_files[rec["patient_id"]].append(fname)
    all_patients = sorted(patient_to_files.keys())
    rng = random.Random(SEED)
    train_patients = set(rng.sample(all_patients, min(n_train_patients, len(all_patients))))
    test_patients  = set(all_patients) - train_patients
    train = {f: transcripts[f] for p in train_patients for f in patient_to_files[p]}
    test  = {f: transcripts[f] for p in test_patients  for f in patient_to_files[p]}
    print(f"\n  Train patients: {sorted(train_patients)}  ({len(train)} transcripts)", flush=True)
    print(f"  Test  patients: {sorted(test_patients)}  ({len(test)} transcripts)", flush=True)
    return train, test


def load_from_jsonl(jsonl_dir, target):
    label_col = "label_p" if target == "patient" else "label_t"
    def _load(path):
        turns = []
        with open(path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                line = line.strip()
                if not line: continue
                obj = json.loads(line)
                if label_col not in obj: continue
                turns.append({"turn_id": i, "text": str(obj.get("prompt","")).strip(),
                               "label": int(obj[label_col]), "speaker": "unknown",
                               "timestamp": None, "n_das": 1})
        return turns
    train_path, test_path = os.path.join(jsonl_dir,"train.jsonl"), os.path.join(jsonl_dir,"test.jsonl")
    if not os.path.exists(train_path): raise FileNotFoundError(f"train.jsonl not found at {train_path}")
    if not os.path.exists(test_path):  raise FileNotFoundError(f"test.jsonl not found at {test_path}")
    train_turns, test_turns = _load(train_path), _load(test_path)
    n_tr = sum(t["label"] for t in train_turns); n_te = sum(t["label"] for t in test_turns)
    print(f"  jsonl train: {len(train_turns)} turns  {n_tr} important ({100*n_tr/max(len(train_turns),1):.1f}%)", flush=True)
    print(f"  jsonl test:  {len(test_turns)} turns  {n_te} important ({100*n_te/max(len(test_turns),1):.1f}%)", flush=True)
    return train_turns, test_turns


def subsample_test_balanced(test_turns, n_samples, rng):
    """
    Subsample test set to n_samples with 50/50 balance.
    Replicates the original script's return_file_info(90) logic:
      - first n_samples//2 positives
      - first n_samples//2 negatives
      - combine and shuffle
    """
    positives = [t for t in test_turns if t["label"] == 1][:n_samples // 2]
    negatives = [t for t in test_turns if t["label"] == 0][:n_samples // 2]
    combined  = positives + negatives
    rng.shuffle(combined)
    n_pos = sum(t["label"] for t in combined)
    print(f"  Balanced test subsample: {len(combined)} turns  "
          f"{n_pos} positive  {len(combined)-n_pos} negative", flush=True)
    return combined


def build_examples(train_transcripts, n_few_shot, pos_proportion, rng):
    all_turns = [t for rec in train_transcripts.values() for t in rec["turns"]]
    positives = [t for t in all_turns if t["label"] == 1]
    negatives = [t for t in all_turns if t["label"] == 0]
    rng.shuffle(positives); rng.shuffle(negatives)
    if n_few_shot == -1:
        n_pos = len(positives)
        n_neg = min(max(0, round(n_pos * (1-pos_proportion)/pos_proportion)), len(negatives))
    else:
        n_pos = min(round(n_few_shot * pos_proportion), len(positives))
        n_neg = min(n_few_shot - n_pos, len(negatives))
    examples = positives[:n_pos] + negatives[:n_neg]; rng.shuffle(examples)
    print(f"  Examples: {n_pos} positive + {n_neg} negative = {len(examples)} total", flush=True)
    return examples


def build_system_prompt(base_rate):
    return (
        "You are an expert behavioral psychologist analysing therapy session transcripts. "
        "You will be shown examples of therapist/patient turns labeled as important or not important. "
        "Then you must classify a single TARGET TURN.\n\n"
        "Important turns contain specific, intentional therapeutic acts — a deliberate intervention, "
        "a focused exchange, or a distinct shift in the conversation. The mere presence of a relevant "
        "topic does NOT make a turn important. Most turns are background conversation.\n\n"
        f"BASE RATE: Approximately {base_rate*100:.0f}% of turns are important. "
        "The vast majority are NOT important.\n\n"
        "When in doubt, answer 'not important'.\n"
        "Answer with exactly one of: 'important' or 'not important'.\n"
        "Do not explain your answer."
    )


def build_system_prompt_repr():
    return (
        "You are an expert behavioral psychologist conducting an important study. "
        "You are tasked with classifying text between a therapist and a patient as "
        "important or not important. You only answer with either the word ' important ' "
        "or the words ' not important ' based on the context."
    )


def build_examples_for_prediction(train_turns, rng, n_pos=16, neg_frac=0.0005):
    """
    Per-prediction example selection matching the original script exactly:
    1. Shuffle full train set.
    2. Take first n_pos positives as priority_items.
    3. Remove priority_items, shuffle remainder.
    4. Take neg_frac of remainder.
    5. Combine and shuffle.
    Called fresh for every test example.
    """
    shuffled = list(train_turns); rng.shuffle(shuffled)
    priority_items = [t for t in shuffled if t["label"] == 1][:n_pos]
    remaining      = [t for t in shuffled if t not in priority_items]; rng.shuffle(remaining)
    remaining      = remaining[:max(1, int(len(remaining) * neg_frac))]
    combined       = priority_items + remaining; rng.shuffle(combined)
    return combined


def construct_prompt(train_turns, turn, rng, n_pos=16, neg_frac=0.0005):
    """
    Prompt format matching the original script exactly. Fresh examples per call.

    Example: Classify the grammar of this sentence as ' important ' or ' not important ': [text]
    A: [label]
    ...
    Example: Classify the grammar of this sentence as ' important ' or ' not important ': [target]
    A:
    """
    examples = build_examples_for_prediction(train_turns, rng, n_pos, neg_frac)
    _lbl = lambda l: "important" if l == 1 else "not important"
    lines = []
    for ex in examples:
        lines.append(f"Example: Classify the grammar of this sentence as ' important ' or ' not important ': {ex['text']}")
        lines.append(f"A: {_lbl(ex['label'])}")
        lines.append("")
    lines.append(f"Example: Classify the grammar of this sentence as ' important ' or ' not important ': {turn['text']}")
    lines.append("A:")
    return "\n".join(lines)


def _is_multimodal(model_id):
    return bool(re.search(r"gemma.4.e[24]b", model_id.lower()))


def _get_tokenizer(processor):
    return processor.tokenizer if hasattr(processor, "tokenizer") else processor


def load_model(model_id, hf_cache_dir=None):
    import torch
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if hf_cache_dir:
        os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"] = os.environ["HF_DATASETS_CACHE"] = hf_cache_dir
        print(f"HuggingFace cache dir: {hf_cache_dir}", flush=True)
    multimodal = _is_multimodal(model_id)
    print(f"Model type: {'multimodal' if multimodal else 'text-only'}", flush=True)
    if multimodal:
        from transformers import AutoProcessor, AutoModelForImageTextToText
        processor = AutoProcessor.from_pretrained(model_id)
        model = AutoModelForImageTextToText.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto").eval()
    else:
        from transformers import AutoTokenizer, AutoModelForCausalLM
        processor = AutoTokenizer.from_pretrained(model_id)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16, device_map="auto").eval()
    device = next(model.parameters()).device
    print(f"Model loaded on device: {device}", flush=True)
    if torch.cuda.is_available():
        print(f"GPU memory: {torch.cuda.memory_allocated()/1024**3:.1f}GB allocated  {torch.cuda.memory_reserved()/1024**3:.1f}GB reserved", flush=True)
    return model, processor


def generate_prediction(prompt, system, model_and_processor, temperature=0.0, max_tokens=2, retry_note=""):
    import torch
    model, processor = model_and_processor
    is_multimodal    = hasattr(processor, "tokenizer")
    full_prompt      = prompt if not retry_note else f"{prompt}{retry_note}"
    if is_multimodal:
        messages = [{"role":"system","content":[{"type":"text","text":system}]},
                    {"role":"user","content":[{"type":"text","text":full_prompt}]}]
    else:
        messages = [{"role":"system","content":system},{"role":"user","content":full_prompt}]
    inputs    = processor.apply_chat_template(messages, add_generation_prompt=True,
                    tokenize=True, return_dict=True, return_tensors="pt").to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    gen_kwargs = dict(max_new_tokens=max_tokens, do_sample=temperature > 0.0, top_p=0.95)
    if temperature > 0.0: gen_kwargs["temperature"] = temperature
    with torch.inference_mode():
        output_ids = model.generate(**inputs, **gen_kwargs)
    tok = _get_tokenizer(processor)
    response = tok.decode(output_ids[0][input_len:], skip_special_tokens=True).strip()
    logger.debug(f"Response: {response}")
    return response


def parse_prediction(response):
    r = response.lower().strip()
    if re.search(r"\bnot important\b", r): return 0
    if re.search(r"\bimportant\b", r):     return 1
    return None


def evaluate(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    TP, TN, FP, FN = int(cm[1,1]), int(cm[0,0]), int(cm[0,1]), int(cm[1,0])
    sensitivity = TP/(TP+FN) if (TP+FN)>0 else 0.0
    specificity = TN/(TN+FP) if (TN+FP)>0 else 0.0
    precision   = TP/(TP+FP) if (TP+FP)>0 else 0.0
    f1_imp      = 2*precision*sensitivity/(precision+sensitivity) if (precision+sensitivity)>0 else 0.0
    f1_bal      = f1_score(y_true, y_pred, average="macro",    zero_division=0)
    f1_wt       = f1_score(y_true, y_pred, average="weighted", zero_division=0)
    print(f"\n{'='*60}", flush=True)
    print("EVALUATION RESULTS (turn level)", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"TP={TP}  TN={TN}  FP={FP}  FN={FN}", flush=True)
    print(f"Sensitivity (recall): {sensitivity:.4f}", flush=True)
    print(f"Specificity:          {specificity:.4f}", flush=True)
    print(f"Precision:            {precision:.4f}", flush=True)
    print(f"F1(important):        {f1_imp:.4f}", flush=True)
    print(f"F1(balanced/macro):   {f1_bal:.4f}", flush=True)
    print(f"F1(weighted):         {f1_wt:.4f}", flush=True)
    print(f"\n{classification_report(y_true, y_pred, labels=[0,1], target_names=['not_important','important'], zero_division=0)}", flush=True)
    return {"TP":TP,"TN":TN,"FP":FP,"FN":FN,
            "sensitivity":round(sensitivity,4),"specificity":round(specificity,4),
            "precision":round(precision,4),"f1_important":round(f1_imp,4),
            "f1_balanced":round(f1_bal,4),"f1_weighted":round(f1_wt,4)}


def run_predictions(test_transcripts, train_turns, rng, model_and_processor,
                    system_prompt, temperature, max_tokens, max_retries, verbose):
    y_true_all, y_pred_all, pred_rows = [], [], []
    for fname, rec in test_transcripts.items():
        turns = rec["turns"]
        print(f"\n  Transcript: {fname}  ({len(turns)} turns)", flush=True)
        tr_pos = tr_pred = 0
        for t_idx, turn in enumerate(turns):
            prompt = construct_prompt(train_turns, turn, rng)
            pred, response = None, ""
            for attempt in range(max_retries + 1):
                retry_note = ("" if attempt == 0 else
                    "\n\nIMPORTANT: Your previous response could not be parsed. "
                    "Answer with exactly one of: 'important' or 'not important'. Nothing else.")
                response = generate_prediction(prompt, system_prompt, model_and_processor,
                                               temperature, max_tokens, retry_note)
                pred = parse_prediction(response)
                if pred is not None: break
                print(f"    [retry {attempt+1}/{max_retries}] turn {t_idx} unparseable: '{response.strip()}'", flush=True)
            if pred is None:
                print(f"    [FAILED] turn {t_idx} defaulting to 0", flush=True); pred = 0
            y_true_all.append(turn["label"]); y_pred_all.append(pred)
            tr_pos += turn["label"]; tr_pred += pred
            pred_rows.append({"filename":fname,"patient_id":rec["patient_id"],
                               "turn_id":turn["turn_id"],"timestamp":turn["timestamp"],
                               "speaker":turn["speaker"],"text":turn["text"],
                               "n_das":turn["n_das"],"label":turn["label"],
                               "pred":pred,"response":response.strip()})
            if verbose and (t_idx+1) % 20 == 0:
                print(f"    [{t_idx+1}/{len(turns)} turns]", flush=True)
        print(f"  Done: {fname}  —  true_pos={tr_pos}  pred_pos={tr_pred}", flush=True)
    return y_true_all, y_pred_all, pred_rows


def main():
    parser = argparse.ArgumentParser(description="Turn-level LLM importance classifier.")
    parser.add_argument("--dir",            required=True)
    parser.add_argument("--granularity",    default="groups", choices=["groups","raw"])
    parser.add_argument("--target",         default="therapist", choices=["patient","therapist"])
    parser.add_argument("--text_col",       required=True)
    parser.add_argument("--use_jsonl",      action="store_true",
                        help="Load from tts/ jsonl files instead of DA-level transcripts.")
    parser.add_argument("--jsonl_dir",      default="tts/",
                        help="Directory with train.jsonl and test.jsonl. (default: tts/)")
    parser.add_argument("--balanced_test",  action="store_true",
                        help="Subsample test to --n_test_samples with 50/50 balance, "
                             "replicating the original script's evaluation.")
    parser.add_argument("--n_test_samples", type=int, default=90,
                        help="Test samples when --balanced_test is set. (default: 90)")
    parser.add_argument("--n_train_patients", type=int, default=5)
    parser.add_argument("--n_few_shot",       type=int, default=16)
    parser.add_argument("--pos_proportion",   type=float, default=0.75)
    parser.add_argument("--model_id",         default="google/gemma-4-26B-A4B-it")
    parser.add_argument("--hf_cache_dir",     default=None)
    parser.add_argument("--temperature",      type=float, default=0.0)
    parser.add_argument("--max_tokens",       type=int,   default=2)
    parser.add_argument("--max_retries",      type=int,   default=3)
    parser.add_argument("--outdir",  default="llm_turn_output/")
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--log",     action="store_true")
    args = parser.parse_args()

    if args.log:
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        os.makedirs(args.outdir, exist_ok=True)
        logging.basicConfig(filename=os.path.join(args.outdir, f"run-{ts}.log"), level=logging.DEBUG)

    if not Path(args.dir).exists(): raise ValueError(f"Directory not found: {args.dir}")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Turn-Level LLM Importance Classifier", flush=True)
    print(f"target={args.target}  use_jsonl={args.use_jsonl}  "
          f"balanced_test={args.balanced_test}  n_test_samples={args.n_test_samples}", flush=True)

    rng = random.Random(SEED)

    if args.use_jsonl:
        print(f"\nLoading from jsonl: {args.jsonl_dir}", flush=True)
        train_turns_flat, test_turns_flat = load_from_jsonl(args.jsonl_dir, args.target)
        train_transcripts = {"train.jsonl": {"patient_id": "jsonl_train", "turns": train_turns_flat}}
        test_transcripts  = {"test.jsonl":  {"patient_id": "jsonl_test",  "turns": test_turns_flat}}
        system_prompt = build_system_prompt_repr()
    else:
        transcripts = load_transcripts(Path(args.dir), args.target, args.granularity, args.text_col)
        print(f"\nLoaded {len(transcripts)} transcripts.", flush=True)
        train_transcripts, test_transcripts = split_by_patient(transcripts, args.n_train_patients)
        all_train = [t for rec in train_transcripts.values() for t in rec["turns"]]
        base_rate = sum(t["label"] for t in all_train) / max(len(all_train), 1)
        print(f"  Train base rate: {base_rate*100:.1f}%", flush=True)
        system_prompt = build_system_prompt(base_rate)

    print(f"  System prompt: {len(system_prompt)} chars", flush=True)

    if args.balanced_test:
        all_test = [t for rec in test_transcripts.values() for t in rec["turns"]]
        subsampled = subsample_test_balanced(all_test, args.n_test_samples, rng)
        test_transcripts = {"test_balanced": {"patient_id": "balanced", "turns": subsampled}}

    train_turns_flat = [t for rec in train_transcripts.values() for t in rec["turns"]]
    base_rate = sum(t["label"] for t in train_turns_flat) / max(len(train_turns_flat), 1)

    print(f"\nLoading model {args.model_id} ...", flush=True)
    model_and_processor = load_model(args.model_id, args.hf_cache_dir)
    print(f"Model ready.", flush=True)

    total_turns = sum(len(r["turns"]) for r in test_transcripts.values())
    print(f"\nRunning predictions on {len(test_transcripts)} transcripts ({total_turns} turns) ...", flush=True)

    y_true, y_pred, pred_rows = run_predictions(
        test_transcripts=test_transcripts, train_turns=train_turns_flat, rng=rng,
        model_and_processor=model_and_processor, system_prompt=system_prompt,
        temperature=args.temperature, max_tokens=args.max_tokens,
        max_retries=args.max_retries, verbose=args.verbose,
    )

    metrics = evaluate(y_true, y_pred)

    ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    label = (f"{args.target}_{args.granularity}_turn"
             f"_{'jsonl' if args.use_jsonl else 'da'}"
             f"_{'bal' if args.balanced_test else 'full'}"
             f"_{ts}")

    pred_path = os.path.join(args.outdir, f"llm_{label}_predictions.csv")
    pd.DataFrame(pred_rows).to_csv(pred_path, index=False)
    print(f"\n  Saved: {pred_path}", flush=True)

    metrics_path = os.path.join(args.outdir, f"llm_{label}_metrics.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump({"label": label, "target": args.target, "granularity": args.granularity,
                   "use_jsonl": args.use_jsonl, "jsonl_dir": args.jsonl_dir if args.use_jsonl else None,
                   "balanced_test": args.balanced_test,
                   "n_test_samples": args.n_test_samples if args.balanced_test else None,
                   "n_train_patients": args.n_train_patients, "n_few_shot": args.n_few_shot,
                   "pos_proportion": args.pos_proportion, "model_id": args.model_id,
                   "temperature": args.temperature, "max_tokens": args.max_tokens,
                   "base_rate_train": round(base_rate, 4), "n_test_turns": len(y_true),
                   "n_test_pos_turns": sum(y_true), "max_retries": args.max_retries,
                   **metrics}, f, indent=2)
    print(f"  Saved: {metrics_path}", flush=True)
    print(f"\nDone. Outputs in: {args.outdir}", flush=True)


if __name__ == "__main__":
    main()