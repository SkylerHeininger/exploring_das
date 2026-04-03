"""
Skyler Heininger
Glass Brain Lab

Gets histograms of labels from a given directory
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

COL = "Raw DA Class"

SHOW = False
SAVE = True
SAVE_FOLDER = "plots"
SAVE_FOLDER_RI = "plots_RI"


def plot_countplot(data, title, filename):
    plt.figure(figsize=(10, 6))
    sns.countplot(data=data, x=COL, palette="Set2", edgecolor='black')
    plt.title(f'Frequencies of labels in a transcript')
    plt.xlabel(COL)
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)  # Rotate x-axis labels if needed
    # plt.grid(True)
    plt.tight_layout()
    if SAVE:
        plt.savefig(f"{SAVE_FOLDER}/{filename}_countplot.png")  # Save countplot as an image
    if SHOW:
        plt.show()


def plot_countplot_removeI(data, title, filename, col_name=COL):
    if type(data) != list:
        data = list(data[COL])
    data = [label for label in data if label != "I-"]

    plt.figure(figsize=(10, 6))
    sns.countplot(x=data, palette="Set2", edgecolor='black')
    plt.title(title if title else 'Frequencies of labels in a transcript')
    plt.xlabel(col_name)
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)  # Rotate x-axis labels if needed
    # plt.grid(True)

    plt.tight_layout()

    if SAVE:
        plt.savefig(f"{SAVE_FOLDER_RI}/{filename}_countplot.png")
    if SHOW:
        plt.show()


def plot_total_counts_histogram(all_labels, directory_name):
    plt.figure(figsize=(10, 6))
    sns.countplot(x=all_labels, palette="Set2", edgecolor='black')
    plt.title(f'Total counts of multiple transcripts')
    plt.xlabel(COL)
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    # plt.grid(True)
    plt.tight_layout()
    if SAVE:
        plt.savefig(f"{SAVE_FOLDER}/{directory_name}_total_countplot.png")  # Save histogram as an image
    if SHOW:
        plt.show()


def plot_total_counts_histogram_removeI(all_labels, directory_name):
    all_labels = [label for label in all_labels if label != "I-"]
    print(all_labels)
    plt.figure(figsize=(10, 6))
    sns.countplot(x=all_labels, palette="Set2", edgecolor='black')
    plt.title(f'Total counts of multiple transcripts')
    plt.xlabel(COL)
    plt.ylabel('Frequency')
    plt.xticks(rotation=90)
    # plt.grid(True)
    plt.tight_layout()
    if SAVE:
        plt.savefig(f"{SAVE_FOLDER_RI}/{directory_name}_countplot.png")  # Save histogram as an image
    if SHOW:
        plt.show()


def process_files(directory, title):
    all_labels = []

    # Loop through files in the given directory
    for filename in os.listdir(directory):
        if filename.endswith(".csv") or filename.endswith(".xlsx"):
            file_path = os.path.join(directory, filename)

            # Read the data based on file type (CSV or Excel)
            if filename.endswith(".csv"):
                data = pd.read_csv(file_path)
            elif filename.endswith(".xlsx"):
                data = pd.read_excel(file_path)

            # Check if the specified column exists in the file
            if COL in data.columns:
                # Add the labels to the all_labels list
                all_labels.extend(data[COL].dropna())  # Drop NaN values before adding
                # Plot individual file's countplot
                plot_countplot(data, title, f"{title}_{filename}")
                plot_countplot_removeI(data, title, f"{title}_{filename}")
            else:
                print(f"Column '{COL}' not found in file: {filename}")

    # Plot the total counts histogram for the directory after processing all files
    plot_total_counts_histogram(all_labels, title)
    plot_total_counts_histogram_removeI(all_labels, title)


if __name__ == "__main__":
    process_files(YOUR_DIRNAME, TITLE_OF_IMAGES)
