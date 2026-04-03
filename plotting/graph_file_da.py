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
from pathlib import Path
import numpy as np

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
    "statement": ["Statement-non-opinion", "Statement-opinion"],
    "misc": ["Conventional-opening", "Conventional-closing"]
}

GROUP_COLORS = {
    "questions":   "#4C72B0",
    "backchannel": "#DD8452",
    "answers":     "#55A868",
    "misc":        "#C44E52",
    "statement":    "#D6EF57"
}

# For negative non-answers - could be specifically interesting in cases where the patient is asked something,
# such as how they take their medication, and they identify that they have not been. This could be followed by things of
# importance
# The hold before agreement could also be interesting, as if a patient may not have fully understood something,
# it could be important for them to come back to it later


def graph_da_groups_through_transcript(transcript, das, show_da_class=False, importance="both"):
    """
    Graphs the different DA_GROUPS through a transcript. Option for showing the da class on the graph, 
    in addition to having different lines for the different da groups. This will also graph the important 
    turns alongside, with perhaps two different plots for patient and therapist importance
    :param transcript: String, all words concatenated
    :param show_da_class: Boolean, show the exact DA class alongside the colored line for da groups
    :param importance: String, "both", "patient", or "therapist". Determines which importance plot(s) to do
    """

    transcript = list(transcript)
    das = list(das)
    n = len(das)
    turns = np.arange(n)

    # Build binary signal per group
    group_signals = {}
    for group, members in DA_GROUPS.items():
        group_signals[group] = np.array([1 if da in members else 0 for da in das])

    n_groups = len(DA_GROUPS)
    fig, axes = plt.subplots(n_groups, 1, figsize=(16, 2.8 * n_groups), sharex=True)
    if n_groups == 1:
        axes = [axes]

    for ax, (group, signal) in zip(axes, group_signals.items()):
        color = GROUP_COLORS[group]

        # Step line: 0 or 1 across all turns
        ax.step(turns, signal, where="mid", color=color, linewidth=1.8, label=group)

        # Shade under active (=1) regions
        ax.fill_between(turns, signal, step="mid", alpha=0.25, color=color)

        if show_da_class:
            for i, (val, da_label) in enumerate(zip(signal, das)):
                if val == 1 and i % 2 == 0:
                    ax.text(
                        i, 1.05, da_label,
                        fontsize=6.5, ha="center", va="bottom",
                        color="#000000", rotation=0, clip_on=True
                    )

        ax.set_ylabel(group.capitalize(), fontsize=10, fontweight="bold")
        ax.set_ylim(-0.15, 1.45 if show_da_class else 1.25)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["Off", "On"], fontsize=8)
        ax.grid(axis="x", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    # X-axis: show a tick every ~20 turns to keep it readable
    tick_step = max(1, n // 20)
    axes[-1].set_xticks(turns[::tick_step])
    axes[-1].set_xticklabels(turns[::tick_step], fontsize=8)
    axes[-1].set_xlabel("Turn index", fontsize=10)

    fig.suptitle("Dialogue Act Groups Through Transcript", fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.show()


def graph_file(filename, word_col="spoken_text", target_col="Proc DA Class", show_da_class=False, importance="both"):
    """
    Will call all appropriate graphing functions for a file.
    :param filename: String filename
    :param show_da_class: Boolean, show the exact DA class alongside the colored line for da groups
    :param importance: String, "both", "patient", or "therapist". Determines which importance plot(s) to do
    """
    
    df = pd.read_csv(filename)

    graph_da_groups_through_transcript(transcript=df[word_col], das=df[target_col], show_da_class=show_da_class, importance=importance)


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Token classification with Longformer pipeline.")

    parser.add_argument('--dir', type=str, 
                        help="Directory containing CSV, Excel, or txt files, of word-level daseg output")
    parser.add_argument('--word_col', type=str, 
                        help="column containing the text/words",
                        default="spoken_text")
    parser.add_argument('--target_col', type=str, 
                        help="column with dialogue acts",
                        default="DA_Class_Proc")
    parser.add_argument('--show_da_class', type=bool, 
                        help="Show the exact DA class alonside DA groups",
                        default=True)
    parser.add_argument('--importance', type=str, 
                        help="therapist, patient, or both importance",
                        default="both")

    args = parser.parse_args()

    dir_path = Path(args.dir)
    if not dir_path.exists():
        raise ValueError(f"Error: The path '{args.dir}' does not exist.")
        
    # Iterate over dir and test all files
    allowed_file_extensions = {'.csv', '.tsv', '.xlsx'}
    for file in dir_path.iterdir():
        print(file)
        if file.suffix.lower() in allowed_file_extensions:
            graph_file(str(file), 
                       word_col=args.word_col, 
                       target_col=args.target_col, 
                       show_da_class=args.show_da_class, 
                       importance=args.importance)
