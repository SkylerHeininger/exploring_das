from itertools import product
from collections import Counter
from scipy.stats import entropy as scipy_entropy
from statsmodels.stats.multitest import multipletests
from scipy.stats import mannwhitneyu, chi2_contingency
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd
import os
import argparse
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from IPython.display import display, HTML
import tempfile
import webbrowser
from graph_file_da import DA_COLUMN, DA_GROUPS
from scipy.stats import mannwhitneyu


QUESTION_GROUPS = {"canonical_questions", "non_canonical_questions"}
RESPONSE_GROUPS = {"statements", "hedge"}


def map_da_to_group(da: str) -> str | None:
    for group, members in DA_GROUPS.items():
        if da in members:
            return group
    return None


def label_question_response_patterns(
    df: pd.DataFrame,
    da_col: str = DA_COLUMN,
) -> pd.DataFrame:
    """
    Labels each row of a DA-level DataFrame with pattern membership.

    Adds three columns:
      - da_group              : the DA_GROUP key for that DA (e.g. 'statements')
      - is_pattern_question   : True if this row is a question that is immediately
                                followed by ≥1 uninterrupted statement/hedge DA
      - is_pattern_response   : True if this row is a statement/hedge that is part
                                of a run immediately preceded by a question
      - pattern_id            : integer ID shared by a question and all its response
                                DAs (NaN for rows not in any pattern)
    """
    df = df.copy().reset_index(drop=True)
    df["da_group"] = df[da_col].map(map_da_to_group)

    df["is_pattern_question"] = False
    df["is_pattern_response"] = False
    df["pattern_id"] = np.nan

    groups_seq = df["da_group"].tolist()
    pattern_counter = 0
    i = 0

    while i < len(groups_seq):
        if groups_seq[i] in QUESTION_GROUPS:
            # find the uninterrupted run of response DAs immediately after
            j = i + 1
            while j < len(groups_seq) and groups_seq[j] in RESPONSE_GROUPS:
                j += 1

            run_length = j - i - 1
            if run_length > 0:
                df.loc[i, "is_pattern_question"] = True
                df.loc[i, "pattern_id"] = pattern_counter
                df.loc[i + 1 : j - 1, "is_pattern_response"] = True
                df.loc[i + 1 : j - 1, "pattern_id"] = pattern_counter
                pattern_counter += 1
            i = j  # skip past the run — its DAs can't start a new pattern
        else:
            i += 1

    return df

def graph_file_qsh(filename, word_col="spoken_text", target_col="Proc_DA", skip_individual_graphing=True,
               show_da_class=False, importance="both", save=True, show=False, render=False, outdir="qsh_output/"):
    """
    Will call all appropriate graphing functions for a file.
    :param filename: String filename
    :param show_da_class: Boolean, show the exact DA class alongside the colored line for da groups
    :param importance: String, "both", "patient", or "therapist". Determines which importance plot(s) to do
    """
    
    df = pd.read_csv(filename)

    # Pre-process DAs to have just one row per DA (if word-level, many duplicates)
    df_da_level = (df.groupby('DA_number', as_index=False)
         .agg({
             'speaker': 'first',
             'spoken_text': lambda x: ' '.join(x.dropna()),
             DA_COLUMN: 'last',
             'patient_important': 'max',
             'therapist_important': 'max',
             'patient_code': 'max',
             'therapist_code': 'max',
             'timestamp': 'max'
         }))
    
    df = label_question_response_patterns(df_da_level, da_col=DA_COLUMN)

    return df

SPEAKER_COMBOS = [
    ("Therapist", "Patient",    "T asks → P responds"),
    ("Patient",   "Therapist",  "P asks → T responds"),
    ("Therapist", "Therapist",  "T asks → T responds"),
    ("Patient",   "Patient",    "P asks → P responds"),
]

def plot_pattern_run_lengths(
    combined: pd.DataFrame,
    importance_col: str,
    title_prefix: str = "",
    outdir: str = "qsh_output/",
    save: bool = True,
    show: bool = False,
):
    """
    For each pattern, computes run length and breaks down by:
      - importance (important vs non-important question turn)
      - speaker combo (who asked, who responded — majority speaker in response run)

    One figure per speaker combo, each with two side-by-side histograms
    (important vs non-important).
    """
    os.makedirs(outdir, exist_ok=True)

    # ── build per-pattern summary ──────────────────────────────────────────────
    question_rows = combined[combined["is_pattern_question"]].copy()
    response_rows = combined[combined["is_pattern_response"]].copy()

    # Run length per (source_file, pattern_id)
    run_lengths = (
        response_rows
        .groupby(["source_file", "pattern_id"])
        .agg(
            run_length=("is_pattern_response", "count"),
            # majority speaker in the response run
            response_speaker=("speaker", lambda x: x.mode().iloc[0] if len(x) else None),
        )
        .reset_index()
    )

    # Bring in question speaker + importance flag
    question_meta = question_rows[["source_file", "pattern_id", "speaker", importance_col]].rename(
        columns={"speaker": "question_speaker"}
    )

    merged = run_lengths.merge(question_meta, on=["source_file", "pattern_id"], how="left")

    # ── overall figure (no speaker split) — same as before ────────────────────
    _plot_importance_histograms(
        merged,
        importance_col=importance_col,
        title=f"{title_prefix} — all speakers",
        outdir=outdir,
        fname=f"{title_prefix}_run_lengths_all.png",
        save=save,
        show=show,
    )

    # ── one figure per speaker combo ───────────────────────────────────────────
    for q_speaker, r_speaker, combo_label in SPEAKER_COMBOS:
        subset = merged[
            (merged["question_speaker"] == q_speaker) &
            (merged["response_speaker"] == r_speaker)
        ]
        if subset.empty:
            print(f"  No patterns found for: {combo_label}")
            continue

        safe_label = combo_label.replace(" ", "_").replace("→", "to")
        _plot_importance_histograms(
            subset,
            importance_col=importance_col,
            title=f"{title_prefix} — {combo_label}",
            outdir=outdir,
            fname=f"{title_prefix}_run_lengths_{safe_label}.png",
            save=save,
            show=show,
        )

def plot_code_da_group_breakdown(
    combined: pd.DataFrame,
    da_col: str = DA_COLUMN,
    title_prefix: str = "",
    outdir: str = "qsh_output/",
    save: bool = True,
    show: bool = False,
):
    """
    For each of patient_code and therapist_code, plots a stacked bar chart:
      - One bar per unique code value (parsed from comma-separated strings)
      - Each bar filled by the proportion of DA groups across all DAs with that code
    """
    os.makedirs(outdir, exist_ok=True)

    # Ensure da_group is present
    if "da_group" not in combined.columns:
        combined = combined.copy()
        combined["da_group"] = combined[da_col].map(map_da_to_group)

    da_group_order = list(DA_GROUPS.keys())
    cmap = plt.get_cmap("tab10")
    da_group_colors = {g: cmap(i / len(da_group_order)) for i, g in enumerate(da_group_order)}

    for speaker, code_col in [("Patient", "patient_code"), ("Therapist", "therapist_code")]:
        # Explode comma-separated codes into one row per code per DA
        rows = []
        for _, row in combined.iterrows():
            raw = str(row[code_col])
            codes = [c.strip() for c in raw.split(",") if c.strip() not in ("", "nan", "NaN", "None")]
            if not codes:
                codes = ["NA"]
            for code in codes:
                rows.append({
                    "code": code,
                    "da_group": row["da_group"] if pd.notna(row["da_group"]) else "unmapped",
                })

        exploded = pd.DataFrame(rows)

        # For each code, get the DA group distribution
        code_da_counts = (
            exploded
            .groupby(["code", "da_group"])
            .size()
            .unstack(fill_value=0)
        )

        # Reindex to consistent DA group order (add any missing)
        present_groups = [g for g in da_group_order if g in code_da_counts.columns]
        extra_groups = [g for g in code_da_counts.columns if g not in da_group_order]
        code_da_counts = code_da_counts.reindex(columns=present_groups + extra_groups, fill_value=0)

        # Row-normalise to proportions
        code_da_props = code_da_counts.div(code_da_counts.sum(axis=1), axis=0)

        # Sort bars by total count descending
        code_totals = code_da_counts.sum(axis=1).sort_values(ascending=False)
        code_da_props = code_da_props.loc[code_totals.index]
        code_da_counts = code_da_counts.loc[code_totals.index]

        n_codes = len(code_da_props)
        fig, ax = plt.subplots(figsize=(max(10, n_codes * 0.6), 5))

        bottoms = np.zeros(n_codes)
        x = np.arange(n_codes)

        for da_group in code_da_props.columns:
            vals = code_da_props[da_group].values
            color = da_group_colors.get(da_group, "red")
            ax.bar(x, vals, bottom=bottoms, label=da_group, color=color, alpha=0.85, edgecolor="white")
            bottoms += vals

        # Annotate each bar with its raw total count
        for xi, code in zip(x, code_da_props.index):
            total = code_totals[code]
            ax.text(xi, 1.01, str(total), ha="center", va="bottom", fontsize=7, rotation=45)

        ax.set_xticks(x)
        ax.set_xticklabels(code_da_props.index, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Proportion of DA groups")
        ax.set_ylim(0, 1.15)
        ax.set_xlabel("Code")
        title = f"{title_prefix} — {speaker} code DA group breakdown" if title_prefix else f"{speaker} code DA group breakdown"
        ax.set_title(title)
        ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8, title="DA group")
        ax.grid(True, axis="y", color="lightgrey", linewidth=0.5)

        plt.tight_layout()
        if save:
            fname = os.path.join(outdir, f"{title_prefix}_{speaker.lower()}_code_da_breakdown.png")
            plt.savefig(fname, bbox_inches="tight")
        if show:
            plt.show()
        plt.close()


def _plot_importance_histograms(
    merged: pd.DataFrame,
    importance_col: str,
    title: str,
    outdir: str,
    fname: str,
    save: bool,
    show: bool,
):
    """
    Two-panel histogram: important (left) vs non-important (right) run lengths.
    Prints summary stats + Mann-Whitney U test for each panel.
    """
    important     = merged[merged[importance_col] == 1]["run_length"]
    non_important = merged[merged[importance_col] != 1]["run_length"]

    # ── significance test ──────────────────────────────────────────────────────
    stat, p_val, test_note = None, None, ""
    if len(important) >= 2 and len(non_important) >= 2:
        # Mann-Whitney U: non-parametric, doesn't assume normality.
        # Good choice here since run lengths are integer-valued, right-skewed,
        # and group sizes may be unequal.
        stat, p_val = mannwhitneyu(important, non_important, alternative="two-sided")
        sig_label = "p < 0.05 ✓" if p_val < 0.05 else "p ≥ 0.05 ✗"
        test_note = f"Mann-Whitney U={stat:.1f}, p={p_val:.4f} ({sig_label})"
    else:
        test_note = "Mann-Whitney: insufficient data in one or both groups"

    print(f"\n── {title} ──")
    for label, series in [("Important", important), ("Non-important", non_important)]:
        if len(series):
            print(f"  {label}: n={len(series)}, mean={series.mean():.2f}, "
                  f"median={series.median():.0f}, max={series.max()}")
        else:
            print(f"  {label}: no data")
    print(f"  {test_note}")

    if merged.empty:
        return

    max_run = merged["run_length"].max()
    bins = range(1, max_run + 2)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    fig.suptitle(f"{title}\n{test_note}", fontsize=11)

    for ax, series, label, color in zip(
        axes,
        [important, non_important],
        ["Important", "Non-important"],
        ["steelblue", "tomato"],
    ):
        ax.hist(series, bins=bins, color=color, alpha=0.8, edgecolor="white", align="left")
        ax.set_title(f"{label} (n={len(series)})")
        ax.set_xlabel("Run length (# statement/hedge DAs)")
        ax.set_ylabel("Count")
        ax.set_xticks(list(bins)[:-1])
        ax.grid(True, color="lightgrey", linewidth=0.5)
        if len(series):
            ax.axvline(series.mean(), color="black", linestyle="--",
                       linewidth=1, label=f"mean={series.mean():.2f}")
            ax.legend()

    plt.tight_layout()
    if save:
        plt.savefig(os.path.join(outdir, fname), bbox_inches="tight")
    if show:
        plt.show()
    plt.close()


if __name__=="__main__":
    parser = argparse.ArgumentParser(description="Token classification with Longformer pipeline.")

    parser.add_argument('--dir', type=str, 
                        help="Directory containing CSV, Excel, or txt files, of word-level daseg output")
    parser.add_argument('--word_col', type=str, 
                        help="column containing the text/words",
                        default="spoken_text")
    parser.add_argument('--target_col', type=str, 
                        help="column with dialogue acts",
                        default="Proc_DA")
    parser.add_argument('--show_da_class', type=bool, 
                        help="Show the exact DA class alonside DA groups",
                        default=False)
    parser.add_argument('--importance', type=str, 
                        help="therapist, patient, or both importance",
                        default="both")
    parser.add_argument('--train_json', type=str, 
                        help="name of json training file",
                        default="train.json")
    parser.add_argument('--test_json', type=str, 
                        help="name of json testing file",
                        default="test.json")
    parser.add_argument('--graph_dir', type=str, 
                        help="output dir for graphs",
                        default='output/')

    args = parser.parse_args()

    dir_path = Path(args.dir)
    if not dir_path.exists():
        raise ValueError(f"Error: The path '{args.dir}' does not exist.")
    
    os.makedirs(args.graph_dir, exist_ok=True)
        
    # Iterate over dir and test all files
    dfs_at_da_level = {}
    allowed_file_extensions = {'.csv', '.tsv', '.xlsx'}
    for file in dir_path.iterdir():
        print(file)
        if file.suffix.lower() in allowed_file_extensions:
            out = graph_file_qsh(str(file), 
                             word_col=args.word_col, 
                             target_col=args.target_col, 
                             show_da_class=args.show_da_class, 
                             importance=args.importance,
                             outdir = args.graph_dir)
            out['source_file'] = file.name
            dfs_at_da_level[file.name] = out
    


    # Combine all - these are given the file name to distinguish eachother
    combined = pd.concat(dfs_at_da_level.values(), ignore_index=True)
    combined = combined[combined[DA_COLUMN] != 'I-']

    plot_code_da_group_breakdown(combined, title_prefix="Code DA groups", outdir="qsh_output/")


#     p_important = combined[combined['patient_important'] == 1]
#     non_p_important = combined[combined['patient_important'] != 1]

#     q_p_important = p_important[p_important['is_pattern_question'] == True]
#     a_p_important = p_important[p_important['is_pattern_response'] == True]
#     q_p_n_important = non_p_important[non_p_important['is_pattern_question'] == True]
#     a_p_n_important = non_p_important[non_p_important['is_pattern_response'] == True]
#     print(f"Important questions: {len(q_p_important)}\n\
# Important answers: {len(a_p_important)}\n\
# Non-important questions: {len(q_p_n_important)}\n\
# Non-important answes: {len(a_p_n_important)}")

#     # p_n_comparison = {'important': p_important, 'nonimportant': non_p_important}


#     plot_pattern_run_lengths(combined, importance_col='patient_important', 
#                             title_prefix='patient', outdir='qsh_output/')
#     plot_pattern_run_lengths(combined, importance_col='therapist_important', 
#                             title_prefix='therapist', outdir='qsh_output/')

