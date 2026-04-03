"""
Skyler Heininger
Open Recordings

This script performs analysis of where DAs happen in a file, 
in correspondence with important turns.

Probably run this after get_hists.py, so we generally know what we're working with

NOTE: This script is in progress, as I do not have IRB approval yet / access to
the data. Data loaders and other assumptions could be wrong. This assumes a word-level daseg output, 
with words in individual rows.
"""

import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import Path

# Defined groups of DAs
DA_GROUPS = {
    "questions": ["Wh-Question", 
                  "Yes-No-Question", 
                  "Open-Question", 
                  "Declarative-Yes-No-Question",
                  "Rhetorical-Questions",
                  "Backchannel-in-question-form",
                  "Tag-Question",
                  "Declarative-Wh-Question"],
    "backchannel": ["Acknowledge-Backchannel"],
    "answers": ["Yes-answers",
                "No-answers",
                "Affirmative-non-yes-answers",
                "Hold-before-answer-agreement",
                "Other-answers",
                "Negative-non-no-answers",  # this could be very interesting to see
                "Dispreferred-answers"],
    "misc": ["Conventional-opening", "Conventional-closing"]
}

# For negative non-answers - could be specifically interesting in cases where the patient is asked something,
# such as how they take their medication, and they identify that they have not been. This could be followed by things of
# importance
# The hold before agreement could also be interesting, as if a patient may not have fully understood something,
# it could be important for them to come back to it later


def graph_da_groups_through_transcript(transcript, show_da_class=False):
    """
    Graphs the different DA_GROUPS through a transcript. Option for showing the da class on the graph, 
    in addition to 
    """


# Code
def graph_file(filename):
    """
    Will call all appropriate graphing functions for a file.
    :param filename: String filename
    """




if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Token classification with Longformer pipeline.")

    parser.add_argument('--dir', type=str, 
                        help="Directory containing CSV, Excel, or txt files, of word-level daseg output")

    args = parser.parse_args()

    dir_path = Path(args.dir)
    if not dir_path.exists():
        raise ValueError(f"Error: The path '{args.dir}' does not exist.")
        
    # Iterate over dir and test all files
    allowed_file_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.tif'}
    for file in dir_path.iterdir():
        if file.suffix.lower() in allowed_file_extensions:
            graph_file(str(file))
