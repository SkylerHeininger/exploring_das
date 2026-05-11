"""
Skyler Heininger
GBL

Single-file version of the daseg pipeline for deployment.
Processes one transcript file at a time, triggered by file upload.

Usage:
    python daseg_single.py \
        --transcript path/to/transcript.csv \
        --output_dir path/to/output/ \
        --col_with_text spoken_text
"""

import os
import argparse
import re
import time

import pandas as pd
import torch
from transformers import pipeline, AutoTokenizer
from datasets import Dataset


device = 0 if torch.cuda.is_available() else -1
print(f"Using device: {'GPU' if device == 0 else 'CPU'}", flush=True)

pipe      = pipeline("token-classification",
                     model="pzelasko/longformer-swda-nolower",
                     device=device)
tokenizer = AutoTokenizer.from_pretrained("pzelasko/longformer-swda-nolower")

IGNORE_MAX_LENGTH = False



def concatenate_words_to_length(words, max_length=512):
    """
    Concatenate words into sliding windows of up to max_length characters.
    If IGNORE_MAX_LENGTH is True, returns the whole transcript as one string.
    """
    if IGNORE_MAX_LENGTH:
        return [" ".join(map(str, words))]

    windows        = []
    current_window = []
    current_length = 0

    for word in words:
        word = str(word)
        if current_length + len(word) + 1 > max_length:
            windows.append(" ".join(current_window))
            current_window = [word]
            current_length = len(word)
        else:
            current_window.append(word)
            current_length += len(word) + 1

    if current_window:
        windows.append(" ".join(current_window))

    return windows



def scrub_word(word):
    """Remove non-ASCII and other characters that crash daseg."""
    chars_to_remove = ['Ä', 'Ġ', ' ', 'Š', '\n']
    original_word   = str(word).replace("â€", '')
    original_word   = original_word.replace('ÄŠ', '')
    original_word   = original_word.replace('Ä\x8a', '')
    original_word   = re.sub(r'[^\x00-\x7F]+', '', original_word)
    return ''.join(char for char in original_word if char not in chars_to_remove)


def clean_prediction(prediction, chars_to_remove=None):
    """Clean prediction artifacts from daseg token output."""
    if chars_to_remove is None:
        chars_to_remove = ['Ä', 'Ġ', ' ', 'Š', '\n']

    original_word = prediction['word'].strip()
    original_word = str(original_word).replace("â€", '')
    original_word = original_word.replace('ÄŠ', '')
    original_word = original_word.replace('Ä\x8a', '')
    cleaned_word  = ''.join(
        char for char in original_word if char not in chars_to_remove
    )
    prediction['word'] = cleaned_word
    return prediction, cleaned_word



def align_predictions_with_words_using_word_endings(words, predictions):
    """
    Align token-level predictions back to word level using word endings.
    Each word is matched by consuming predictions until the cleaned prediction
    text matches the end of the current word.
    """
    complete_predictions = []
    for predictions_list in predictions:
        complete_predictions.extend(predictions_list)
    print(f"Total tokens to align: {len(complete_predictions)}", flush=True)

    aligned_predictions = []
    prediction_index    = 0
    num_predictions     = len(complete_predictions)

    for word in words:
        word             = str(word)
        word_predictions = []
        cleaned_word     = scrub_word(word)

        while prediction_index < num_predictions:
            prediction             = complete_predictions[prediction_index]
            prediction, cleaned_pred = clean_prediction(prediction)

            if not cleaned_pred:
                prediction_index += 1
                continue

            word_predictions.append(prediction)
            prediction_index += 1

            if cleaned_word.endswith(cleaned_pred):
                break

        aligned_predictions.append(word_predictions if word_predictions else None)

    return aligned_predictions



def turn_df_to_word_df(df, col_with_text):
    return (
        df.assign(**{col_with_text: df[col_with_text].str.split(' ')})
        .explode(col_with_text)
        .loc[lambda d: d[col_with_text].notna() &
                       (d[col_with_text].str.strip() != '')]
        .reset_index(drop=True)
    )


def pad_list_to_dataframe_length(data, lst):
    df_length = len(data)
    if len(lst) < df_length:
        lst.extend([""] * (df_length - len(lst)))
    elif len(lst) > df_length:
        lst = lst[:df_length]
    return lst



def process_file(file_path, output_dir, col_with_text):
    """
    Load a single transcript file, run daseg, align predictions to words,
    and save output CSV to output_dir.
    """
    print(f"Processing: {file_path}", flush=True)

    if file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)
    elif file_path.endswith('.tsv'):
        data = pd.read_csv(file_path, delimiter='\t')
    else:
        raise ValueError(f"Unsupported file format: {file_path}. Use CSV, TSV, or XLSX.")

    data = data.dropna(how='all')
    data = data[
        data[col_with_text].notna() &
        (data[col_with_text].astype(str).str.strip() != '')
    ]
    data = data[~(data[col_with_text] == 'â€"')]
    data[col_with_text] = data[col_with_text].replace(
        {'—': ' ', '–': ' ', '-': ' '}, regex=True
    )

    data_word_level = turn_df_to_word_df(data, col_with_text)
    words = [scrub_word(str(w)) for w in data_word_level[col_with_text]]

    if not words:
        print(f"  No words found in {file_path} — skipping.", flush=True)
        return

    windows = concatenate_words_to_length(words)
    windows = [w for w in windows if w and w.strip()]

    if not windows:
        print(f"  No windows produced — skipping.", flush=True)
        return

    print(f"  {len(words)} words  →  {len(windows)} windows", flush=True)

    dataset     = Dataset.from_dict({col_with_text: windows})
    predictions = pipe(list(dataset[col_with_text]), batch_size=16)

    word_predictions = align_predictions_with_words_using_word_endings(
        words, predictions
    )
    print(f"  Aligned {len(word_predictions)} words", flush=True)

    entities       = []
    raw_entities   = []
    scores         = []
    raw_scores     = []
    words_pred     = []
    chunks         = []

    chunk_index        = 0
    previous_raw_label = "I-"
    temp_predictions   = []

    for pred_list in word_predictions:
        if pred_list is None:
            chunk_index += 1
            for lst in (entities, raw_entities, scores, raw_scores,
                        words_pred, chunks):
                lst.append("")
            continue

        pred   = pred_list[0]
        entity = pred['entity']
        score  = pred['score']

        temp_word = ''.join(p['word'] for p in pred_list)
        temp_predictions.append({
            'entity':     entity,
            'raw_entity': entity,
            'score':      score,
            'raw_score':  score,
            'word':       temp_word,
        })

        if entity != previous_raw_label and previous_raw_label != "I-":
            chunk_index += 1

        if entity != 'I-':
            # Fill preceding I- tokens with this label
            for tp in temp_predictions:
                if tp['entity'] == 'I-':
                    tp['entity'] = entity
                    tp['score']  = score

            for tp in temp_predictions:
                entities.append(tp['entity'])
                raw_entities.append(tp['raw_entity'])
                raw_scores.append(tp['raw_score'])
                scores.append(tp['score'])
                chunks.append(chunk_index)
                words_pred.append(tp['word'])

            temp_predictions = []

        previous_raw_label = entity

    # Flush any remaining I- predictions
    for tp in temp_predictions:
        entities.append(tp['entity'])
        raw_entities.append(tp['raw_entity'])
        raw_scores.append(tp['raw_score'])
        scores.append(tp['score'])
        chunks.append(chunk_index)
        words_pred.append(tp['word'])

    data_word_level['Pred_DA']         = pad_list_to_dataframe_length(data_word_level, raw_entities)
    data_word_level['Raw_Score']       = pad_list_to_dataframe_length(data_word_level, raw_scores)
    data_word_level['Proc_DA']         = pad_list_to_dataframe_length(data_word_level, entities)
    data_word_level['Score']           = pad_list_to_dataframe_length(data_word_level, scores)
    data_word_level['Words_Prediction']= pad_list_to_dataframe_length(data_word_level, words_pred)
    data_word_level['DA_number']       = pad_list_to_dataframe_length(data_word_level, chunks)

    none_count = sum(1 for p in word_predictions if p is None)
    print(f"  None predictions: {none_count}", flush=True)

    stem          = os.path.splitext(os.path.basename(file_path))[0]
    output_path   = os.path.join(output_dir, f"{stem}_with_predictions.csv")
    data_word_level.to_csv(output_path, index=False)
    print(f"  Saved: {output_path}", flush=True)


def main():
    parser = argparse.ArgumentParser(
        description="Run daseg on a single transcript file."
    )
    parser.add_argument("--transcript",    required=True,
                        help="Path to the transcript file (CSV, TSV, or XLSX).")
    parser.add_argument("--output_dir",    default="daseg_output/",
                        help="Directory to write the output CSV. (default: daseg_output/)")
    parser.add_argument("--col_with_text", required=True,
                        help="Column in the transcript file containing spoken text.")

    args = parser.parse_args()

    if not os.path.exists(args.transcript):
        raise FileNotFoundError(f"Transcript not found: {args.transcript}")

    os.makedirs(args.output_dir, exist_ok=True)

    start = time.time()
    process_file(args.transcript, args.output_dir, args.col_with_text)
    print(f"\nDone in {time.time() - start:.2f}s", flush=True)


if __name__ == "__main__":
    main()
