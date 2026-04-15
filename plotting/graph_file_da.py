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
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import json
from scipy.stats import mannwhitneyu
from statsmodels.stats.multitest import multipletests
from IPython.display import display, HTML
import tempfile
import webbrowser

DA_COLUMN = 'Proc_DA'
# These variables are for the file-level, not full dataset
SAVE_GRAPHS = True
SHOW_GRAPHS = False
RENDER_TRANSCRIPT = True

# Defined groups of DAs
DA_GROUPS = {
    # canonical questions follow the standard framing posed by questions
    # ex: Did you like the play?
    "canonical_questions": ["Wh-Question", 
                  "Yes-No-Question", 
                  "Open-Question", 
                  "Or-Clause"],
    # non-canonical questions don't follow standard framing
    # ex: You liked the play?
    "non_canonical_questions": ["Declarative-Yes-No-Question",
                  "Rhetorical-Questions",
                  "Backchannel-in-question-form",
                  "Tag-Question",
                  "Declarative-Wh-Question"],
    "backchannel": ["Hold-before-answer-agreement",
                            "Acknowledge-Backchannel"],
    # Canonical answers follow a standard response to a posed question (both canonical and noncanonical)
    # Ex: Yes, I liked the play.
    "canonical_answers": ["Yes-answers",
                "No-answers"],
    # Non-canonical answers follow a non-standard response to a posed question (both canonical and noncanonical)
    # Ex: The play was ok.
    "non_canonical_answers": ["Affirmative-non-yes-answers",
                "Other-answers",
                "Negative-non-no-answers",
                "Dispreferred-answers",
                "Reject"],
    # Statements are the most common of DAs, with opinionated ones following 
    # what someone thinks, while non-opinions follow facts
    "statements": ["Statement-non-opinion", "Statement-opinion"],
    # Hedges are diminished confidence of the speaker
    # Ex: I don't know...(followed by a statement, answer, quesstion, etc)
    "hedge": ["Hedge"]
}

GROUP_COLORS = {
    "questions":   "#4C72B0",
    "backchannel": "#DD8452",
    "answers":     "#55A868",
    "misc":        "#C44E52",
    "statement":    "#D6EF57"
}


CODES = {
    "BCS": "Building coping skills to avoid substance use",
    "ERT": "Emphasizing ongoing reality testing",
    "ECE": "Encouraging patients to engage in corrective experiences",
    "FTA": "Facilitating the therapeutic alliance",
    "FSU": "Focusing directly on substance use",
    "HPM": "Fostering hope, positive expectations, and motivation",
    "IDR": "Identifying resources",
    "IAI": "Increasing awareness and insight",
    "STK": "Social talk",
    "VLD": "Validating"
}

# For negative non-answers - could be specifically interesting in cases where the patient is asked something,
# such as how they take their medication, and they identify that they have not been. This could be followed by things of
# importance
# The hold before agreement could also be interesting, as if a patient may not have fully understood something,
# it could be important for them to come back to it later


def graph_da_groups_through_transcript(transcript, das, filename, show_da_class=False, importance="both", 
                                       save=SAVE_GRAPHS, show=SHOW_GRAPHS, outdir="output/" ):
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
    if save:
        plt.savefig(os.path.join(outdir, f"{os.path.splitext(os.path.basename(filename))[0]}.png"))
    if show:
        plt.show()


def graph_comparison_of_groups(groups: dict, sorted_columns=None, p_values_fdr=None, title_prefix="", save=SAVE_GRAPHS, show=SHOW_GRAPHS):
    """
    This function is responsible for looking at the different frequencies of all DAs in a group. 
    Since this may differ per-method, this needs to be split and get all DAs for a group prior to 
    sending to this function
    :param groups: dictionary containing all groups.
    """
    group_names = list(groups.keys())
    n_groups = len(group_names)

    # Compute normalized value counts for each group
    group_freqs = {
        name: series.value_counts(normalize=True)
        for name, series in groups.items()
    }

    # Infer columns (union of all labels) sorted by overall frequency if not provided
    if sorted_columns is None:
        all_counts = (pd.concat([s.value_counts() for s in groups.values()])
                .groupby(level=0)
                .sum())        
        sorted_columns = list(all_counts.sort_values(ascending=False).index)

    n_labels = len(sorted_columns)
    x = np.arange(n_labels)

    # cmap = cm.get_cmap('tab10' if n_groups <= 10 else 'tab20')
    # colors = [cmap(i / max(n_groups - 1, 1)) for i in range(n_groups)]
    colors = list(GROUP_COLORS.values())

    total_bar_width = 0.8
    bar_width = total_bar_width / n_groups
    offsets = np.linspace(
        -(total_bar_width - bar_width) / 2,
         (total_bar_width - bar_width) / 2,
        n_groups
    )

    plt.figure(figsize=[13, 5], clear=True)

    for name, color, offset in zip(group_names, colors, offsets):
        freqs = group_freqs[name].reindex(sorted_columns, fill_value=0)
        n = len(groups[name])
        plt.bar(
            x + offset, freqs,
            width=bar_width,
            label=f'{name} (n={n})',
            color=color, alpha=0.7
        )

    plt.title(f'{title_prefix} DA distribution' if title_prefix else 'DA distribution')
    plt.xlabel('DA Label')
    plt.ylabel('Fraction')
    plt.xticks(x, sorted_columns, rotation=90)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, color='lightgrey', linewidth=0.5)
    plt.tight_layout()
    plt.show()


def graph_comparison_of_groups_full(groups: dict, col: str, sorted_columns=None, p_values_fdr=None, outdir="output/", title_prefix="", save=SAVE_GRAPHS, show=SHOW_GRAPHS, add_nums=False):
    groups = {
        name: (df[col] if isinstance(df, pd.DataFrame) else df)
        for name, df in groups.items()
    }
    group_names = list(groups.keys())
    n_groups = len(group_names)

    # Compute normalized value counts for each group
    group_freqs = {
        name: series.value_counts(normalize=True)
        for name, series in groups.items()
    }

    # Also compute raw counts if needed
    if add_nums:
        group_counts = {
            name: series.value_counts(normalize=False)
            for name, series in groups.items()
        }

    # Infer columns (union of all labels) sorted by overall frequency if not provided
    if sorted_columns is None:
        all_counts = (pd.concat([s.value_counts() for s in groups.values()])
                        # .reset_index(level=0, drop=True)
                        .groupby(level=0)
                        .sum())
        sorted_columns = list(all_counts.sort_values(ascending=False).index)

    n_labels = len(sorted_columns)
    x = np.arange(n_labels)

    if n_groups > 5:
        cmap = cm.get_cmap('tab10' if n_groups <= 10 else 'tab20')
        colors = [cmap(i / max(n_groups - 1, 1)) for i in range(n_groups)]
    else:
        colors = list(GROUP_COLORS.values())

    total_bar_width = 0.8
    bar_width = total_bar_width / n_groups
    offsets = np.linspace(
        -(total_bar_width - bar_width) / 2,
         (total_bar_width - bar_width) / 2,
        n_groups
    )

    if n_groups == 2:
        name_a, name_b = group_names
        series_a, series_b = groups[name_a], groups[name_b]

        p_values = []
        for label in sorted_columns:
            a = (series_a == label).astype(int)
            b = (series_b == label).astype(int)
            _, p = mannwhitneyu(a, b, alternative='two-sided')
            p_values.append(p)

        if len(p_values) > 0:
            _, p_fdr, _, _ = multipletests(p_values, method='fdr_bh')
            significant = p_fdr < 0.05
        else:
            significant = [False] * n_labels
    else:
        significant = [False] * n_labels

    max_freq = max(
        freq.reindex(sorted_columns, fill_value=0).max()
        for freq in group_freqs.values()
    )

    plt.figure(figsize=[13, 5], clear=True)

    is_star_plotted = False
    for i, (label, sig) in enumerate(zip(sorted_columns, significant)):
        if sig:
            if not is_star_plotted:
                plt.plot(x[i], max_freq * 3, 'r*', markersize=10, label=r'$p_{fdr}<0.05$')
                is_star_plotted = True
            else:
                plt.plot(x[i], max_freq * 3, 'r*', markersize=10)

    for name, color, offset in zip(group_names, colors, offsets):
        freqs = group_freqs[name].reindex(sorted_columns, fill_value=0)
        n = len(groups[name])
        bars = plt.bar(
            x + offset, freqs,
            width=bar_width,
            label=f'{name} (n={n})',
            color=color, alpha=0.7
        )

        if add_nums:
            counts = group_counts[name].reindex(sorted_columns, fill_value=0)
            for bar, count in zip(bars, counts):
                if count > 0:
                    plt.text(
                        bar.get_x() + bar.get_width() / 2,
                        bar.get_height(),
                        str(count),
                        ha='center', va='bottom',
                        fontsize=6, rotation=0,
                        color="#000000"
                    )

    plt.title(f'{title_prefix} DA distribution' if title_prefix else 'DA distribution')
    plt.xlabel('DA Label')
    plt.ylabel('Fraction')
    plt.xticks(x, sorted_columns, rotation=90)
    plt.yscale('log')
    plt.legend()
    plt.grid(True, color='lightgrey', linewidth=0.5)

    plt.tight_layout()
    
    if save:
        plt.savefig(os.path.join(outdir, f'{title_prefix}_da_distribution.png'), bbox_inches='tight')
    if show:
        plt.show()
    plt.close()



def graph_file(filename, word_col="spoken_text", target_col="Proc_DA", skip_individual_graphing=True,
               show_da_class=False, importance="both", save=SAVE_GRAPHS, show=SHOW_GRAPHS, render=RENDER_TRANSCRIPT, outdir="output/"):
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
    
    if render:
        render_daseg(df_da_level, da_column=DA_COLUMN)
        # Wait fo rinput as otherwise this spams browser with 30 tabs
        input()
    
    # Just using this function for the graphing
    if skip_individual_graphing:
        return df_da_level

    graph_da_groups_through_transcript(transcript=df[word_col], das=df[target_col], filename=filename, show_da_class=show_da_class, importance=importance, outdir=outdir)

    # Split by speaker - not assuming two speakers
    speakers = df_da_level['speaker'].unique()
    speaker_groups = {
        speaker: df_da_level[df_da_level['speaker'] == speaker][DA_COLUMN]
        for speaker in speakers
    }
    graph_comparison_of_groups(speaker_groups, title_prefix="Speaker Comparison")


    # Triple split by patient importance, therapist importance, and non-importance
    # In case of overlapping importance, give DA to both therapist and importance
    p_important = df_da_level[df_da_level['patient_important'] == 1][DA_COLUMN]
    t_important = df_da_level[df_da_level['therapist_important'] == 1][DA_COLUMN]
    non_important = df_da_level[(df_da_level['therapist_important'] != 1) & (df_da_level['patient_important'] != 1)][DA_COLUMN]
    p_t_comparison = {'patient': p_important, 'therapist': t_important, 'nonimp': non_important}
    graph_comparison_of_groups(p_t_comparison, title_prefix="Patient-Therapist-NonImportant Comparison")

    # Compare importance across speaker
    # t_t_important is where the therapist thinks what the therapist said was important
    t_t_important = df_da_level[(df_da_level['therapist_important'] == 1) & (df_da_level['speaker'] == 'Therapist')][DA_COLUMN]
    # Therapist thinks patient said something important
    t_p_important = df_da_level[(df_da_level['therapist_important'] == 1) & (df_da_level['speaker'] != 'Therapist')][DA_COLUMN]
    p_t_important = df_da_level[(df_da_level['patient_important'] == 1) & (df_da_level['speaker'] == 'Therapist')][DA_COLUMN]
    p_p_important = df_da_level[(df_da_level['patient_important'] == 1) & (df_da_level['speaker'] != 'Therapist')][DA_COLUMN]
    
    cross_importance = {'t_t_important': t_t_important, 
                        't_p_important': t_p_important, 
                        'p_t_important': p_t_important,
                        'p_p_important': p_p_important,
                        'non_important': non_important}
    graph_comparison_of_groups(cross_importance, title_prefix="Cross Speaker Importance Comparison")


    # Split by importance code for therapists
    t_codes = df_da_level['therapist_code'].unique()
    df_t_codes = {code: df_da_level[df_da_level['therapist_code'] == code][DA_COLUMN] for code in t_codes}
    df_t_codes['noncoded'] = df_da_level[df_da_level['therapist_code'].isna() | (df['therapist_code'] == '')][DA_COLUMN]
    graph_comparison_of_groups(df_t_codes, title_prefix="Therapist Code Comparison")


    # Split by importance code for patients
    p_codes = df_da_level['patient_code'].unique()
    df_p_codes = {code: df_da_level[df_da_level['patient_code'] == code][DA_COLUMN] for code in p_codes}
    df_p_codes['noncoded'] = df_da_level[df_da_level['patient_code'].isna() | (df['patient_code'] == '')][DA_COLUMN]
    graph_comparison_of_groups(df_p_codes, title_prefix="Patient Code Comparison")
    
    return df_da_level

def render_daseg(df_da_level, da_column='Proc_DA'):
    """
    Renders important turns from df_da_level in a DAseg-style visualization.
    """
    turns = []
    for _, row in df_da_level.iterrows():
        p_t_codes = str(row["therapist_code"] if row["therapist_important"] == 1 else row["patient_code"]).split(",")
        codes = ""
        for p_t_code in p_t_codes:
            p_t_code = p_t_code.strip()
            if p_t_code != 'nan':
                codes += CODES[p_t_code] + ","
            else:
                codes += "NA,"
        turns.append({
            "speaker":     str(row['speaker']),
            "da_number":   int(row['DA_number']),
            "spoken_text":        str(row['spoken_text']),
            "da_tag":      str(row[da_column]),
            "therapist_important": int(row['therapist_important']),
            "patient_important": int(row['patient_important']),
            "code": codes
        })

    turns_json = json.dumps(turns)

    html = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>DAseg viewer</title>
<style>
  body {{ font-family: system-ui, sans-serif; max-width: 900px; margin: 2rem auto; padding: 0 1rem; background: #fff; color: #111; }}
  .turn-row {{ display: flex; align-items: flex-start; gap: 8px; margin-bottom: 8px; flex-wrap: wrap; }}
  .speaker-label {{ font-weight: 500; font-size: 13px; min-width: 18px; color: #666; padding-top: 4px; }}
  .segments-wrap {{ display: flex; flex-wrap: wrap; gap: 6px; flex: 1; }}
  .da-segment {{ display: inline-flex; flex-direction: column; padding: 5px 10px; border-radius: 6px; }}
  .da-text {{ font-size: 13px; line-height: 1.4; }}
  .da-label {{ font-size: 9px; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; margin-top: 3px; opacity: 0.85; }}
  .legend {{ display: flex; flex-wrap: wrap; gap: 8px; margin-bottom: 1.25rem; padding-bottom: 1rem; border-bottom: 1px solid #e5e5e5; }}
  .legend-item {{ display: flex; align-items: center; gap: 5px; font-size: 12px; color: #555; }}
  .legend-swatch {{ width: 12px; height: 12px; border-radius: 3px; }}
  .nav-bar {{ display: flex; align-items: center; gap: 12px; margin-bottom: 1.25rem; padding-bottom: 1rem; border-bottom: 1px solid #e5e5e5; }}
  .nav-bar button {{ padding: 4px 14px; font-size: 13px; cursor: pointer; border: 1px solid #ccc; border-radius: 4px; background: #fff; }}
  .nav-bar button:disabled {{ opacity: 0.35; cursor: default; }}
  .chunk-label {{ font-size: 13px; color: #666; }}
  .chunk-meta {{ font-size: 12px; color: #888; margin-bottom: 0.75rem; }}
</style>
</head>
<body>
<div id="legend" class="legend"></div>
<div class="nav-bar">
  <button id="btn-prev">← prev</button>
  <span class="chunk-label" id="chunk-label"></span>
  <button id="btn-next">next →</button>
</div>
<div class="chunk-meta" id="chunk-meta"></div>

<div id="da-output"></div>

<script>
const DA_COLORS = {{
  "STATEMENT-NON-OPINION":       {{bg:"#2563A8",text:"#EAF2FB",label:"#90BDE8"}},
  "STATEMENT-OPINION":           {{bg:"#1A3F7A",text:"#EAF2FB",label:"#5A8FD4"}},
  "ACKNOWLEDGE-BACKCHANNEL":     {{bg:"#7B3FA0",text:"#F5EEFF",label:"#C99EE0"}},
  "AFFIRMATIVE-NON-YES-ANSWERS": {{bg:"#1A7A3F",text:"#E8F7EE",label:"#6FCA97"}},
  "DECLARATIVE-YES-NO-QUESTION": {{bg:"#C0392B",text:"#FDECEA",label:"#F1A89E"}},
  "YES-NO-QUESTION":             {{bg:"#A93226",text:"#FDECEA",label:"#E8847C"}},
  "WH-QUESTION":                 {{bg:"#D35B8E",text:"#FEF0F6",label:"#F0A8C8"}},
  "OPEN-QUESTION":               {{bg:"#B03A6A",text:"#FEF0F6",label:"#E07AA8"}},
  "CONVENTIONAL-CLOSING":        {{bg:"#17718A",text:"#E5F5FA",label:"#72C4D8"}},
  "CONVENTIONAL-OPENING":        {{bg:"#0E4D63",text:"#E5F5FA",label:"#3FA3C0"}},
  "APOLOGY":                     {{bg:"#7A6A10",text:"#FDFBE8",label:"#D4C040"}},
  "THANKING":                    {{bg:"#4A7A10",text:"#F0FAE5",label:"#9FCC55"}},
  "OTHER":                       {{bg:"#888780",text:"#F1EFE8",label:"#D3D1C7"}},
  "UNINTERPRETABLE":             {{bg:"#5F5E5A",text:"#F1EFE8",label:"#B4B2A9"}},
  "HEDGE":                       {{bg:"#BA7517",text:"#FAEEDA",label:"#FAC775"}},
  "SIGNAL-NON-UNDERSTANDING":    {{bg:"#8B1A1A",text:"#FDECEA",label:"#D97070"}},
  "ACTION-DIRECTIVE":            {{bg:"#2E7D5E",text:"#E8F7EE",label:"#6ABFA0"}},
  "RHETORICAL-QUESTION":         {{bg:"#E8735A",text:"#FEF0ED",label:"#F4B5A5"}},
  "NO-ANSWER":                   {{bg:"#3A5A3A",text:"#EBF5EB",label:"#80B080"}},
  "REJECT":                      {{bg:"#6B1A1A",text:"#FDECEA",label:"#C06060"}},
}};

function getColor(da) {{
  return DA_COLORS[(da||"").toUpperCase().trim()] || {{bg:"#888780",text:"#F1EFE8",label:"#D3D1C7"}};
}}

const turns = {turns_json};

// Split into contiguous important chunks with nonimportant before and after
const CONTEXT = 15;
const chunks = [];

// First find all contiguous important blocks
const blocks = [];
let block = null;
turns.forEach((t, i) => {{
  if (t.therapist_important || t.patient_important) {{
    if (!block) {{ block = {{ start: i, end: i }}; }}
    else {{ block.end = i; }}
  }} else {{
    if (block) {{ blocks.push(block); block = null; }}
  }}
}});
if (block) blocks.push(block);

// Then build each chunk with clamped context that doesn't overlap adjacent blocks
blocks.forEach((b, bi) => {{
  const prevBlockEnd   = bi > 0 ? blocks[bi-1].end : -1;
  const nextBlockStart = bi < blocks.length-1 ? blocks[bi+1].start : turns.length;

  const contextStart = Math.max(prevBlockEnd + 1, b.start - CONTEXT);
  const contextEnd   = Math.min(nextBlockStart - 1, b.end + CONTEXT);

  chunks.push({{
    turns: turns.slice(contextStart, contextEnd + 1),
    importantStart: b.start,
    importantEnd: b.end,
  }});
}});

let idx = 0;

function renderChunk() {{
  const chunk = chunks[idx];
  document.getElementById("chunk-label").textContent = `chunk ${{idx+1}} of ${{chunks.length}}`;
  document.getElementById("btn-prev").disabled = idx === 0;
  document.getElementById("btn-next").disabled = idx === chunks.length - 1;

  const impTurns = chunk.turns.filter(t => t.therapist_important || t.patient_important);
  document.getElementById("chunk-meta").textContent =
    `turns ${{impTurns[0].da_number}}–${{impTurns[impTurns.length-1].da_number}}`;

  const firstCode = (impTurns[0].code || "").replace(/,$/, "");
  const important = impTurns[0].therapist_important && impTurns[0].patient_important
    ? "Important to: Therapist & Patient"
    : impTurns[0].therapist_important
      ? "Important to: Therapist"
      : "Important to: Patient";
  const codeDisplay = firstCode
    ? `<div style="font-size:12px;color:#555;margin-bottom:0.6rem;"><strong>Code:</strong> ${{firstCode}} &nbsp;|&nbsp; <strong>${{important}}</strong></div>`
    : `<div style="font-size:12px;color:#555;margin-bottom:0.6rem;"><strong>${{important}}</strong></div>`;

  const grouped = [];
  let g = null;
  chunk.turns.forEach(t => {{
    if (!g || g.speaker !== t.speaker || g.da_number !== t.da_number) {{
      g = {{speaker: t.speaker, da_number: t.da_number, segments: [], important: !!(t.therapist_important || t.patient_important)}};
      grouped.push(g);
    }}
    g.segments.push(t);
  }});

  let html = codeDisplay;
  let inImportant = false;

  grouped.forEach(g => {{
    if (g.important && !inImportant) {{
      html += `<div style="border-top:2px dashed #E24B4A;margin:10px 0 6px;padding-top:6px;font-size:11px;color:#E24B4A;font-weight:600;letter-spacing:0.05em;">▶ IMPORTANT TURNS BEGIN <br><span style="color:#555;font-weight:400;font-size:12px;">${{firstCode ? "<strong>Code:</strong> " + firstCode + " &nbsp;|&nbsp; " : ""}}<strong>${{important}}</strong></span></div>`;
      inImportant = true;
    }} else if (!g.important && inImportant) {{
      html += `<div style="border-top:2px dashed #E24B4A;margin:10px 0 6px;padding-top:6px;font-size:11px;color:#E24B4A;font-weight:600;letter-spacing:0.05em;">◀ IMPORTANT TURNS END</div>`;
      inImportant = false;
    }}

    const segs = g.segments.map(s => {{
      const c = getColor(s.da_tag);
      return `<span class="da-segment" style="background:${{c.bg}};${{g.important ? "" : "opacity:0.45;"}}">
        <span class="da-text" style="color:${{c.text}}">${{s.spoken_text||""}}</span>
        <span class="da-label" style="color:${{c.label}}">${{(s.da_tag||"").toUpperCase()}}</span>
      </span>`;
    }}).join("");

    html += `<div class="turn-row">
      <span class="speaker-label" style="${{g.important ? "" : "opacity:0.45;"}}">${{g.speaker}}:</span>
      <div class="segments-wrap">${{segs}}</div>
    </div>`;
  }});

  if (inImportant) {{
    html += `<div style="border-top:2px dashed #E24B4A;margin:10px 0 6px;font-size:11px;color:#E24B4A;font-weight:600;letter-spacing:0.05em;">◀ IMPORTANT TURNS END</div>`;
  }}

  document.getElementById("da-output").innerHTML = html;
}}

// Build legend from all turns
const seen = new Set(turns.map(t => (t.da_tag||"").toUpperCase().trim()).filter(Boolean));
const legend = document.getElementById("legend");
seen.forEach(da => {{
  const c = getColor(da);
  legend.innerHTML += `<div class="legend-item"><div class="legend-swatch" style="background:${{c.bg}}"></div>${{da}}</div>`;
}});

if (chunks.length) {{
  renderChunk();
}} else {{
  document.getElementById("da-output").textContent = "No important turns found.";
}}

document.getElementById("btn-prev").addEventListener("click", () => {{ if(idx>0){{idx--;renderChunk();}} }});
document.getElementById("btn-next").addEventListener("click", () => {{ if(idx<chunks.length-1){{idx++;renderChunk();}} }});
</script>
</body>
</html>"""

    # Write to a temp file and open in browser
    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False, encoding='utf-8')
    tmp.write(html)
    tmp.close()
    webbrowser.open(f'file://{tmp.name}')
    print(f"Opened: {tmp.name}")



def break_down_relationships(groups, target_col, title_prefix, outdir="group_output/", save=True, show=False):
    """
    This method is for grouping dialogue acts for comparisons
    Looks at the DA groups in each of DA_GROUPS.
    Will make a separate graph in a subdir of the base filename for the separate graphs
    """
    os.makedirs(os.path.join(outdir, title_prefix), exist_ok=True)
    grouped_data = {}
    for da_group, das in DA_GROUPS.items():
        subsetted_data = {}
        for group_id, data in groups.items():
            if isinstance(data, pd.DataFrame) and target_col in data.columns:
                selected_data = data[data[target_col].isin(das)].copy()
            else:
                data = pd.DataFrame({target_col: data})
                selected_data = data[data[target_col].isin(das)].copy()
            subsetted_data[group_id] = selected_data.copy()
            selected_data[target_col] = da_group

            if group_id in grouped_data:
                grouped_data[group_id] = pd.concat(
                    [grouped_data[group_id], selected_data],
                    ignore_index=True
                )
            else:
                grouped_data[group_id] = selected_data.copy()
        # Now pass to graph comparison
        graph_comparison_of_groups_full(subsetted_data, col=target_col, outdir=outdir, title_prefix=f"{title_prefix}/{da_group} Comparison", save=True, show=False, add_nums=True)
    
    # Now we use the data from each - the above re-does the data by col
    graph_comparison_of_groups_full(grouped_data, col=target_col, outdir=outdir, title_prefix=title_prefix, save=save, show=show, add_nums=True)
    




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
            out = graph_file(str(file), 
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

    # Split by therapist and non-therapist speakers
    speaker_groups = {'therapist': combined[combined['speaker'] == 'Therapist'],
                      'patient': combined[combined['speaker'] == 'Patient'],}
                    #   'SN': combined[(combined['speaker'] != 'Therapist') & (combined['speaker'] != 'Patient')]}
    
    graph_comparison_of_groups_full(speaker_groups, col=DA_COLUMN, title_prefix="Speaker Comparison", save=True, show=False)

    break_down_relationships(speaker_groups, target_col=DA_COLUMN, title_prefix="Speaker Comparison of DA groups", outdir="group_output/", save=True, show=False)


    speaker_groups = {'therapist': combined[combined['speaker'] == 'Therapist'],
                    'patient': combined[combined['speaker'] == 'Patient'],
                  'SN': combined[(combined['speaker'] != 'Therapist') & (combined['speaker'] != 'Patient')]}
    graph_comparison_of_groups_full(speaker_groups, col=DA_COLUMN, title_prefix="All Speaker Comparison", save=False, show=False)



    # Triple split by patient importance, therapist importance, and non-importance
    # In case of overlapping importance, give DA to both therapist and importance
    p_important = combined[combined['patient_important'] == 1][DA_COLUMN]
    t_important = combined[combined['therapist_important'] == 1][DA_COLUMN]
    non_important = combined[(combined['therapist_important'] != 1) & (combined['patient_important'] != 1)][DA_COLUMN]
    p_t_comparison = {'patient': p_important, 'therapist': t_important}
    graph_comparison_of_groups_full(p_t_comparison, col=DA_COLUMN, title_prefix="Patient-Therapist Importance Comparison", show=False, save=True)

    p_t_comparison = {'patient': p_important, 'therapist': t_important, 'nonimp': non_important}
    graph_comparison_of_groups_full(p_t_comparison, col=DA_COLUMN, title_prefix="Patient-Therapist-NonImportant Comparison", show=False, save=True)
    all_important = combined[(combined['patient_important'] == 1) | (combined['therapist_important'] == 1)][DA_COLUMN]
    a_t_comparison = {'Important': all_important, 'NonImportant': non_important}
    graph_comparison_of_groups_full(a_t_comparison, col=DA_COLUMN, title_prefix="Important-NonImportant Comparison", show=False, save=True)

    break_down_relationships(a_t_comparison, target_col=DA_COLUMN, title_prefix="Importance Comparison of DA groups", outdir="group_output", show=False, save=True)

    non_p_important = combined[combined['patient_important'] != 1][DA_COLUMN]
    p_n_comparison = {'important': p_important, 'nonimportant': non_p_important}
    break_down_relationships(p_n_comparison, target_col=DA_COLUMN, title_prefix="Patient Importance Comparison of DA groups", outdir="group_output", show=True, save=True)
    graph_comparison_of_groups_full(p_n_comparison, col=DA_COLUMN, title_prefix="Patient Importance Comparison", show=True, save=True)
    non_t_important = combined[combined['therapist_important'] != 1][DA_COLUMN]
    t_n_comparison = {'important': t_important, 'nonimportant': non_t_important}
    break_down_relationships(t_n_comparison, target_col=DA_COLUMN, title_prefix="Therapist Importance Comparison of DA groups", outdir="group_output", show=True, save=True)
    graph_comparison_of_groups_full(t_n_comparison, col=DA_COLUMN, title_prefix="Therapist Importance Comparison", show=True, save=True)


    # Compare importance across speaker
    # t_t_important is where the therapist thinks what the therapist said was important
    t_t_important = combined[(combined['therapist_important'] == 1) & (combined['speaker'] == 'Therapist')][DA_COLUMN]
    # Therapist thinks patient said something important
    t_p_important = combined[(combined['therapist_important'] == 1) & (combined['speaker'] == 'Patient')][DA_COLUMN]
    p_t_important = combined[(combined['patient_important'] == 1) & (combined['speaker'] == 'Therapist')][DA_COLUMN]
    p_p_important = combined[(combined['patient_important'] == 1) & (combined['speaker'] == 'Patient')][DA_COLUMN]
    print(f"TT importance num: {len(t_t_important)}\n \
TP importance num: {len(t_p_important)}\n \
PT importance num: {len(p_t_important)}\n \
PP importance num: {len(p_p_important)}\n")
    cross_importance = {'t_t_important': t_t_important, 
                        't_p_important': t_p_important, 
                        'p_t_important': p_t_important,
                        'p_p_important': p_p_important,
                        'non_important': non_important}
    graph_comparison_of_groups_full(cross_importance, col=DA_COLUMN, title_prefix="Cross Speaker Importance Comparison", show=False, save=True)


    # Combined importance across speakers:
    non_t_important = combined[(combined['speaker'] == 'Therapist') & (combined['therapist_important'] != 1) & 
                                                                   (combined['patient_important'] != 1)][DA_COLUMN]

    all_t_important = combined[(combined['speaker'] == 'Therapist') & ((combined['patient_important'] == 1) | 
                               (combined['therapist_important'] == 1))][DA_COLUMN]
    groups_t_imp = {"important": all_t_important, "nonimportant": non_t_important}
    graph_comparison_of_groups_full(groups_t_imp, col=DA_COLUMN, title_prefix="Importance over Therapist Speaking", show=False, save=True)

    break_down_relationships(groups_t_imp, target_col=DA_COLUMN, title_prefix="DA Groups over Therapist Speaking", show=False, save=True)

    non_p_important = combined[(combined['speaker'] == 'Patient') & ((combined['therapist_important'] != 1) & 
                                                                   (combined['patient_important'] != 1))][DA_COLUMN]
    all_p_important = combined[(combined['speaker'] == 'Patient') & ((combined['patient_important'] == 1) | 
                               (combined['therapist_important'] == 1))][DA_COLUMN]
    groups_p_imp = {"important": all_p_important, "nonimportant": non_p_important}
    graph_comparison_of_groups_full(groups_p_imp, col=DA_COLUMN, title_prefix="Importance over Patient Speaking", show=False, save=True)

    break_down_relationships(groups_p_imp, target_col=DA_COLUMN, title_prefix="DA Groups over Patient Speaking", show=False, save=True)


    # Combined split on multiple codes per thing
    combined_t_coded = combined.assign(
        therapist_code=combined['therapist_code'].str.split(', ')
    ).explode('therapist_code').reset_index(drop=True)

    # Split by importance code for therapists
    # Drop NaN from unique so it doesn't become a key
    t_codes = combined_t_coded['therapist_code'].dropna().unique()
    t_codes = [c for c in t_codes if c != '']
    # print(t_codes)
    df_t_codes = {code: combined_t_coded[combined_t_coded['therapist_code'] == code][DA_COLUMN] for code in t_codes}
    # Add noncoded separately and cleanly
    df_t_codes['noncoded'] = combined_t_coded[combined_t_coded['therapist_code'].isna() | (combined_t_coded['therapist_code'] == '')][DA_COLUMN]
    graph_comparison_of_groups_full(df_t_codes, col=DA_COLUMN, title_prefix="Therapist Code Comparison", show=False, save=True)



    # Split by importance code for patients
    combined_p_coded = combined.assign(
        patient_code=combined['patient_code'].str.split(', ')
    ).explode('patient_code').reset_index(drop=True)
    p_codes = combined_p_coded['patient_code'].dropna().unique()
    p_codes = [c for c in p_codes if c != '']
    # print(p_codes)
    df_p_codes = {code: combined_p_coded[combined_p_coded['patient_code'] == code][DA_COLUMN] for code in p_codes}
    df_p_codes['noncoded'] = combined_p_coded[combined_p_coded['patient_code'].isna() | (combined_p_coded['patient_code'] == '')][DA_COLUMN]
    graph_comparison_of_groups_full(df_p_codes, col=DA_COLUMN, title_prefix="Patient Code Comparison", show=False, save=True)
    




    # Number time!
    # Get questions from 
    question_data = combined[combined[DA_COLUMN].isin(DA_GROUPS['canonical_questions']) | combined[DA_COLUMN].isin(DA_GROUPS['non_canonical_questions'])].copy()

    t_questions = question_data[question_data['speaker'] == 'Therapist']
    p_questions = question_data[question_data['speaker'] == 'Patient']

    i_questions = question_data[(question_data['therapist_important'] == 1) | (question_data['patient_important'] == 1)]
    not_i_questions = question_data[(question_data['therapist_important'] != 1) & (question_data['patient_important'] != 1)]

    print(f"The therapist asked {len(t_questions)} questions\n\
Of the questions the patient asked, {len(t_questions[(t_questions["therapist_important"] == 1) | (t_questions["patient_important"] == 1)])} were important and {len(t_questions[(t_questions["therapist_important"] != 1) & (t_questions["patient_important"] != 1)])} were not important\n\
The patient asked {len(p_questions)}\n\
Of the questions the patient asked, {len(p_questions[(p_questions["therapist_important"] == 1) | (p_questions["patient_important"] == 1)])} were important and {len(p_questions[(p_questions["therapist_important"] != 1) & (p_questions["patient_important"] != 1)])} were not important")
    
    exit()

    # Train-test split comparison
    test_das = []
    train_das = []

    # Load json and iterate
    with open(args.test_json) as f:
        test_data = json.load(f)

    # Not super sure what this will exactly look like
    for item in data:
        df_file = dfs_at_da_level[item.key()]
        time = item.value()['timestamp']
        if time in df_file['timestamp']:
            das = df_file[df_file['timestamp'] == time][DA_COLUMN]
            test_das.append(das)

    
    # Load json and iterate
    with open(args.train_json) as f:
        test_data = json.load(f)

    # Not super sure what this will exactly look like
    for item in data:
        df_file = dfs_at_da_level[item.key()]
        time = item.value()['timestamp']
        if time in df_file['timestamp']:
            das = df_file[df_file['timestamp'] == time][DA_COLUMN]
            train_das.append(das)

    groups = {'train': train_das, 'test': test_das}
    graph_comparison_of_groups_full(groups, col=DA_COLUMN, title_prefix="Train-Test Comparison", show=True, save=True)
    