"""
Skyler Heininger
GBL

This pipeline uses Zelasko's daseg pipeline to perform dialogue act segmentation.

Note this saves DAs at the word level, converting utterance level to word level.
This does mean that start times, etc, are duplicated across.
"""

import os
import argparse
import random

import pandas as pd
from transformers import pipeline, AutoTokenizer
from datasets import Dataset
import tarfile
import numpy as np
import re
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize the pipeline for token classification
# Use device=0 for GPU usage if available
# pipe = pipeline("token-classification", model="pzelasko/longformer-swda-nolower", device=-1)
# tokenizer = AutoTokenizer.from_pretrained("pzelasko/longformer-swda-nolower", device=-1)

pipe = pipeline("token-classification", model="pzelasko/longformer-swda-nolower", device=0)
tokenizer = AutoTokenizer.from_pretrained("pzelasko/longformer-swda-nolower", device=0)

# Keeping this as true means all of zelasko's code does all the handling
# But, keeping this as true causes you to run into issue where inputting an entire transcript will cause the pipeline to just stop working
# This also is not outputting any specific error, but setting a max length and using a window seems to work well enough.
# This window is a sliding window with no overlap in windows.
IGNORE_MAX_LENGTH = False


def concatenate_words_to_length(words, max_length=512):
    """
    Concatenate words until reaching the maximum number of characters of `max_length`. This was done somewhat
    informally, using the number of characters, because it just seemed to work. Consider changing this in the future
    when comparing to ground-truthed labels.

    Option to ignore maximum length and concatenate all words together, using IGNORE_MAX_LENGTH global constant.

    :param words: List of strings
    "param max_length: Maximum number of characters in sequence, NOT number of tokens or words.
    """
    windows = []
    current_window = []
    current_length = 0

    if IGNORE_MAX_LENGTH:
        all_words = " ".join(map(str, words))
        return [all_words]

    for word in words:
        word = str(word)
        if current_length + len(word) + 1 > max_length:  # +1 for space or punctuation
            windows.append(" ".join(current_window))
            current_window = [word]
            current_length = len(word)
        else:
            current_window.append(word)
            current_length += len(word) + 1

    if current_window:
        windows.append(" ".join(current_window))

    return windows


def concatenate_words_to_turns(df):
    """
    Same as concatenate_words_to_length, other than it saves windows of words whenever the speaker changes. Would not
    recommend using, as otherwise daseg will not see the back and forth in conversation.

    :param df: Dataframe with 'speaker' and 'text' columns
    """
    turns = []
    current_speaker = None
    current_turn = []

    for _, row in df.iterrows():
        speaker = row['speaker']
        word = str(row['text'])
        # Happens sometimes, simply use the current speaker
        if pd.isna(speaker):
            speaker = current_speaker

        # If speaker changes store the current turn and reset
        if speaker != current_speaker:
            turns.append(" ".join(current_turn))  # Join the words into a single string
            current_speaker = speaker
            current_turn = [word]  # Start a new turn with the current word

        else:
            # Continue adding words for the same speaker
            current_turn.append(word)

    # Don't forget to append the last group of words
    if current_turn:
        turns.append(" ".join(current_turn))

    return turns


def align_predictions_with_words(words, predictions):
    """
    Align the predictions from tokenized text to the original words.
    This has slight issues where the tokenizer responds with the incorrect number of tokens for each word. It is
    recommended to use this version, if this issue can be fixed.
    Otherwise, use align_predictions_with_words_using_word_endings
    """
    complete_predictions = []
    for predictions_list in predictions:
        complete_predictions.extend(predictions_list)

    aligned_predictions = []

    token_index = 0

    for word in words:
        # Create a list to store predictions for the current word
        word_predictions = []

        # Get the number of tokens for the current word
        word_tokens = tokenizer.tokenize(word)
        num_word_tokens = len(word_tokens)

        # Collect predictions corresponding to the tokens of the word
        for _ in range(num_word_tokens):
            if token_index < len(complete_predictions):
                word_predictions.append(complete_predictions[token_index])
                token_index += 1
            else:
                break

        # If no predictions are found for the word, append None or an empty list
        if not word_predictions:
            aligned_predictions.append([])
        else:
            aligned_predictions.append(word_predictions)

    return aligned_predictions


def clean_prediction(prediction, chars_to_remove=None):
    """
    Used to clean the word level of a prediction to ensure there are no prediction
    'artifacts' left over (default chars_to_remove has the general artifacts). This is because
    daseg will for some reason output weird characters with predictions.
    :param prediction: dictionary output of daseg, for a single token
    :param chars_to_remove: Keep as None to remove common prediction artifacts.
    """
    if chars_to_remove is None:
        chars_to_remove = ['Ä', 'Ġ', ' ']

    original_word = prediction['word']
    original_word = str(original_word).replace("â€”", '')  # Remove another problem sequence

    # Remove unwanted characters but keep spaces
    cleaned_word = ''.join(char for char in original_word if char not in chars_to_remove)
    prediction['word'] = cleaned_word

    return prediction, cleaned_word


def scrub_word(word):
    """
    Scrub the word by removing non-ASCII characters and other unwanted symbols.
    Daseg struggles with non-ASCII characters and will sometimes crash, avoid using those.
    """
    word = re.sub(r'[^\x00-\x7F]+', '', word)
    return word


def align_predictions_with_words_using_word_endings(words, predictions):
    """
    Aligns predictions with words using the ends of words. This generally works well, but again an approach using a
    tokenizer would work better (the issue is the tokenizer is not consistent with the pipeline object). So, using
    endings of words allows us to tell what token matches the ending of a word.
    :param words: List of Strings
    :param predictions: list of dictionaries, each dictionary is an output from daseg for a token
    """
    # Note after the fact: I am unsure as to why I made a complete_predictions list, I must have made it for a different
    # version of the pipeline
    complete_predictions = []
    for predictions_list in predictions:
        complete_predictions.extend(predictions_list)
    print(len(complete_predictions), flush=True)

    aligned_predictions = []

    prediction_index = 0
    num_predictions = len(complete_predictions)

    for word in words:
        word = str(word)
        word_predictions = []
        current_concat = ""

        cleaned_word = scrub_word(word)

        # Continue processing while there are still predictions to check
        while prediction_index < num_predictions:
            # Remove unwanted characters from prediction
            prediction = complete_predictions[prediction_index]
            prediction, cleaned_pred = clean_prediction(prediction)

            # Word has nothing in it, skip
            if not cleaned_pred:
                prediction_index += 1
                continue

            if cleaned_word.endswith(cleaned_pred):
                # If the prediction matches the end of the current word segment
                word_predictions.append(prediction)
                current_concat += cleaned_pred
                prediction_index += 1
                break
            word_predictions.append(prediction)
            prediction_index += 1

        # If there are no matches, append None or an empty list
        if not word_predictions:
            aligned_predictions.append(None)
        else:
            aligned_predictions.append(word_predictions)

    return aligned_predictions


def load_and_split_text(file_path):
    """
    Custom method for loading data from a txt file where data is already in utterance form. This was for an experiment
    specifically using utterances, it is not recommended to use these (based on the results of the experiment)
    :param file_path: String
    """
    rows = []
    utterances = []
    if type(file_path) == str:
        with open(file_path, 'r') as file:
            for line in file:
                # Format for each line: "start_time end_time speaker_number utterance"
                parts = line.strip().split(maxsplit=3)

                if len(parts) < 4:
                    continue  # If line is malformed or doesn't have the required parts, skip it

                start_time, end_time, speaker_number, utterance = parts

                utterances.append(utterance)

                # Split the utterance by spaces into individual words
                words = utterance.split()

                # For each word, create a dictionary of the associated data
                for word in words:
                    row = {
                        'text': word,
                        'start_time': start_time,
                        'end_time': end_time,
                        'speaker': speaker_number
                    }
                    rows.append(row)
    else:
        for line in file_path:
            # Format for each line: "start_time end_time speaker_number utterance"
            parts = line.strip().split(maxsplit=3)

            if len(parts) < 4:
                continue  # If line is malformed or doesn't have the required parts, skip it

            start_time, end_time, speaker_number, utterance = parts
            # Decode data to remove binary encoding artifacts
            start_time = start_time.decode('utf-8', errors='ignore')
            end_time = end_time.decode('utf-8', errors='ignore')
            speaker_number = speaker_number.decode('utf-8', errors='ignore')
            utterance = utterance.decode('utf-8', errors='ignore')

            utterances.append(utterance)
            words = utterance.split()

            for word in words:
                row = {
                    'text': word,
                    'start_time': start_time,
                    'end_time': end_time,
                    'speaker': speaker_number
                }
                rows.append(row)
    # Create a DataFrame from the list of rows
    df = pd.DataFrame(rows)
    return df, utterances


def pad_list_to_dataframe_length(data, lst):
    """
    Method for ensuring dataframe is the same length as a list
    :param data: dataframe
    :param lst: List
    """
    df_length = len(data)
    # Pad with empty strings if shorter
    if len(lst) < df_length:
        lst.extend([""] * (df_length - len(lst)))
    # Truncate if it's longer
    elif len(lst) > df_length:
        lst = lst[:df_length]

    return lst


def turn_df_to_word_df(df, col_with_text):
    return df.assign(spoken_text=df[col_with_text].str.split(' ')).explode(col_with_text)


def process_file(file_path, output_dir, col_with_text, all_words_split_by_window=None, pre_loaded=False, filename_if_preloaded=""):
    """
    Opens file, loads words, uses daseg, aligns token level predictions with words, determines how to replace
    I- entities with first non-I- label, and saves file to output_dir.
    I- replacement in this is done using the first non-I- label, rather than last label of a DA. One recommended change
    would be to modify this to match the last label of a DA, rather than first non-I- label. (In practice there
    seems to be no difference)

    Also, only the prediction for the first token of each word is used. This mirrors Zelasko's usage. It is strongly
    recommended to not use the last token of each word, this will give highly unpredictable and incorrect predictions.

    :param file_path: String, filepath
    :param output_dir: String, will create output directory if it does not exist
    :param col_with_text: String, the column in data file corresponding to what contains the text
    :param all_words_split_by_window: List of Strings, each string is a chunk of text
        (currently getting this using a sliding window, within process_file)
    :param pre_loaded: Boolean, If the data has already been loaded (If so, input data using file_path)
    :param filename_if_preloaded: String, If pre-loaded data, This is the output filename
    """
    if all_words_split_by_window is None:
        all_words_split_by_window = []

    # Including this already split variable in case data loading pipeline has already split the words
    # using a sliding window for instance
    already_split = False
    # Load the file (assuming it's a CSV or Excel file)
    if pre_loaded:
        data = file_path
        # already_split = True
    elif file_path.startswith('='):
        # Wrong type of CSV file
        print("Issue with csv file", flush=True)
        return
    elif file_path.endswith('.csv'):
        data = pd.read_csv(file_path)
    elif file_path.endswith('.xlsx'):
        data = pd.read_excel(file_path)

    else:
        raise ValueError("Unsupported file format. Use CSV or Excel files.")

    # Remove rows with problem sequence (some of this parsing is likely redundant)
    # Replace dashes with spaces, rather than removing
    data = data[~((data[col_with_text] == 'â€”') | (data[col_with_text] == '—') | (data[col_with_text] == '–') | (data[col_with_text] == '-'))]

    # This is not required for daseg, more for labels afterwards
    data_word_level = turn_df_to_word_df(data, col_with_text)

    words = data_word_level[col_with_text] # data[col_with_text].tolist()
    # And remove problem sequence from words list directly
    words = [str(word).replace('â€”', '').replace('—', '') for word in words]

    words = [scrub_word(word) for word in words]

    # Concatenate words using a sliding window
    if not already_split:
        all_words_split_by_window = concatenate_words_to_length(words)

    if not all_words_split_by_window:
        return None
    
    all_words_split_by_window = [text for text in all_words_split_by_window if text and text.strip()]

    # Create a dataset from the sliding window list
    dataset = Dataset.from_dict({col_with_text: all_words_split_by_window})

    # Apply the pipeline to the dataset in batches
    predictions = pipe(list(dataset[col_with_text]), batch_size=16)

    # Map the predictions back to individual words
    word_predictions = align_predictions_with_words_using_word_endings(words, predictions)
    print("Aligned")
    entities = []
    raw_entities = []
    scores = []
    words_pred = []
    chunks = []
    raw_scores = []
    specific_raw_counts = []

    chunk_index = 0
    previous_raw_label = "I-"
    score = 0

    temp_predictions = []

    # print("Processing")

    for idx, pred_list in enumerate(word_predictions):
        if pred_list is None:
            chunk_index += 1
            entities.append("")
            raw_entities.append("")
            raw_scores.append("")
            scores.append("")
            chunks.append("")
            words_pred.append("")
            continue

        pred = pred_list[0]  # Modified from -1
        entity = pred['entity']
        score = pred['score']

        temp_word = ''
        for temp_pred in pred_list:
            temp_word += temp_pred['word']

        temp_predictions.append({
            'entity': entity,
            'raw_entity': entity,
            'score': score,
            'raw_score': score,
            'word': temp_word
        })

        if entity != previous_raw_label and previous_raw_label != "I-":
            chunk_index += 1

        if entity != 'I-':
            # This is the non-I- label, so we fill in all the previous I- labels with this one
            for temp_pred in temp_predictions:
                if temp_pred['entity'] == 'I-':
                    temp_pred['entity'] = entity
                    temp_pred['score'] = score

            # After filling, push all the filled predictions into the final results
            for temp_pred in temp_predictions:
                entities.append(temp_pred['entity'])
                raw_entities.append(temp_pred['raw_entity'])
                raw_scores.append(temp_pred['raw_score'])
                scores.append(temp_pred['score'])
                chunks.append(chunk_index)
                words_pred.append(temp_pred['word'])

            # Reset the temp_predictions list and increment the chunk index
            temp_predictions = []

        previous_raw_label = entity

    # Handle remaining entries by just adding them
    if temp_predictions:
        for temp_pred in temp_predictions:
            entities.append(temp_pred['entity'])
            raw_entities.append(temp_pred['raw_entity'])
            raw_scores.append(temp_pred['raw_score'])
            scores.append(temp_pred['score'])
            chunks.append(chunk_index)
            words_pred.append(temp_pred['word'])

    # Use if word-level
    data_word_level['Pred_DA'] = pad_list_to_dataframe_length(data_word_level, raw_entities)
    data_word_level['Raw_Score'] = pad_list_to_dataframe_length(data_word_level, raw_scores)
    data_word_level['Proc_DA'] = pad_list_to_dataframe_length(data_word_level, entities)
    data_word_level['Score'] = pad_list_to_dataframe_length(data_word_level, scores)
    data_word_level['Words_Prediction'] = pad_list_to_dataframe_length(data_word_level, words_pred)
    data_word_level['DA_number'] = pad_list_to_dataframe_length(data_word_level, chunks)

    # word_level_df = pd.DataFrame({
    #     'Word': words_pred,
    #     'Raw_DA': raw_entities,
    #     'Raw_Score': raw_scores,
    #     'Proc_DA': entities,
    #     'Score': scores,
    #     'DA_number': chunks,
    # })

    # Create a new filename for the output
    if not pre_loaded:
        file_name, file_extension = os.path.splitext(file_path)
    else:
        file_name, file_extension = os.path.splitext(filename_if_preloaded)
    file_name = file_name.split("/")[-1]
    new_file_path = f"{file_name}_with_predictions.csv"
    save_file_path = f"{file_name}_count_plot"

    # Output to given dir
    if output_dir is not None:
        new_file_path = os.path.join(output_dir, new_file_path)
        save_file_path = os.path.join(output_dir, save_file_path)

    # Save the updated dataframe to a new CSV file
    data_word_level.to_csv(new_file_path, index=False)
    print(f"Predictions saved to {new_file_path}", flush=True)

    return specific_raw_counts



def check_files_in_tar(tar_file, file_list, directory):
    """Check if filenames in file_list exist in the tarball."""
    tar_data = tarfile.open(tar_file, mode='r')

    tar_contents = tar_data.getnames()
    print(tar_contents)

    # Prepend the directory to each file in the file_list before checking
    matched_files = [
        os.path.join(directory, os.path.basename(file)) for file in file_list
        if os.path.join(directory, os.path.basename(file)) in tar_contents
    ]

    return matched_files, tar_data


def main():
    """
    Performs all of the data loading, and then calls process_file for each file.
    To note, this does check the output directory supplied in arguments for all files that have already been processed,
    and will not process those files again.

    Many previous usages are included below, commented out. In case you need to 
    load data differently, these may be helpful.
    """


    # There are a lot of arguments here. This pipeline and how to load the data changes a lot depending on what
    # form you are loading data from. Sometimes, you may need to pull names from one file, while getting the text from
    # another. Or, you want to look at a specific file. Change these as needed.

    parser = argparse.ArgumentParser(description="Token classification with Longformer pipeline.")

    parser.add_argument('--directory', type=str, help="Directory containing CSV, Excel, or txt files")
    parser.add_argument('--output_dir', type=str, default=None, help="Where output files will be stored")
    parser.add_argument('--col_with_text', type=str, default=None, help="Column in data with text for DASeg")


    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    transcribed_files = [os.path.join(args.directory, f) for f in os.listdir(args.directory)
                if f.endswith('.csv') or f.endswith('.xlsx') or f.endswit('.tsv')]

    already_daseg = [os.path.join(args.output_dir, f) for f in os.listdir(args.output_dir)
                     if f.endswith('.csv') or f.endswith('.xlsx') or f.endswit('.tsv')]

    files_to_daseg = {}
    files_already_daseg = {}

    print(f"{len(transcribed_files)} files were found within the given directory.")

    # Store filenames in dictionary
    for file in transcribed_files:
        filename = os.path.basename(file)  # Get filename without path
        base_name_parts = filename.split('_')  # Split by underscore

        # Select only first five indices - this typically gets just the names of the people in the interview
        combined_key = '_'.join(base_name_parts[:5])

        files_to_daseg[combined_key] = file

    # Skip files 
    for file in already_daseg:
        filename = os.path.basename(file)
        base_name_parts = filename.split('_')

        combined_key = '_'.join(base_name_parts[:5])

        # Add the audio file to the dictionary where the key is the part
        files_already_daseg[combined_key] = file
    
    print(files_to_daseg)
    print(f"{len(files_already_daseg)} files were already processed.\nBeginning processing now.", flush=True)
    for key in files_to_daseg.keys():
        # Check that the file has not already been processed
        if key not in files_already_daseg.keys():
            try:
                # Process the file
                print(f"Processing file: {key}")
                process_file(files_to_daseg[key], args.output_dir, col_with_text=args.col_with_text)
            except Exception as e:
                print(e)


if __name__ == "__main__":
    main()
