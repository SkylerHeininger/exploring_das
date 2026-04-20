"""
ngram_bow_da.py

N-gram bag-of-words analysis of dialogue act sequences, comparing across:
  1. Importance levels  patient_important, therapist_important, non_important
  2. Individual transcripts
  3. Therapists (transcripts pooled by therapist ID from filename)

For each comparison level and each n-gram size (2–5), produces:
  - A combined CSV  (rows = n-gram strings, cols = groups, values = norm. freq)
    with an extra "ngram_size" column so all sizes sit in one tall file
  - A heatmap PNG of the frequency matrix
  - Chi-squared + FDR correction flagging n-grams that differ across groups

Outputs go into subdirectories of --outdir/ngram_bow/:
  ngram_bow/
    importance/
      ngram_bow_importance.csv
      ngram_bow_importance_heatmap_n{2,3,4,5}.png
      ngram_bow_importance_chi2.csv
    transcripts/
      ngram_bow_transcripts.csv          (summary stats per n-gram, not per transcript)
      ngram_bow_transcripts_heatmap_n{N}.png
      ngram_bow_transcripts_chi2.csv
    therapists/
      ngram_bow_therapists.csv
      ngram_bow_therapists_heatmap_n{N}.png
      ngram_bow_therapists_chi2.csv

Granularity
-----------
--granularity groups  (default) DA group labels after mapping through
                                   EXTENDED_DA_GROUPS from common_patterns.py
--granularity raw     original DA class strings as-is

Usage
-----
python ngram_bow_da.py \\
    --dir /path/to/csv_dir \\
    --granularity groups \\
    --ngram_sizes 2 3 4 5 \\
    --outdir ngram_bow_output/

Drop alongside common_patterns.py and graph_file_da.py.
Requires: pandas, numpy, matplotlib, scipy, statsmodels
"""

from __future__ import annotations

import argparse
import os
import re
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

# ── shared imports ────────────────────────────────────────────────────────────
from plotting.common_patterns import (
    DA_COLUMN,
    EXTENDED_DA_GROUPS,
    DA_GROUP_ABBREV,
    load_da_level,
    map_da_to_group,
    get_label,
    rle_compress,
    extract_important_blocks,
    extract_nonimportant_sequences,
)

# ── helpers ───────────────────────────────────────────────────────────────────

NGRAM_SIZES = [2, 3, 4, 5]


def extract_therapist_id(filename: str) -> str:
    """Single character before '_with' in filename, or 'unknown'."""
    m = re.search(r"(.)_with", filename)
    return m.group(1) if m else "unknown"


def _subdir(base: str, *parts: str) -> str:
    path = os.path.join(base, *parts)
    os.makedirs(path, exist_ok=True)
    return path


def _ngrams(seq: Sequence[str], n: int):
    """Yield n-gram tuples from a sequence."""
    it     = iter(seq)
    window = tuple(islice(it, n))
    if len(window) == n:
        yield window
    for item in it:
        window = window[1:] + (item,)
        yield window


def _ngram_label(ngram: tuple[str, ...], granularity: str) -> str:
    """
    Human-readable n-gram string, e.g. 'ST-CQ-ST'.
    Uses abbreviations for group granularity, raw DA names for raw.
    """
    if granularity == "groups":
        parts = [DA_GROUP_ABBREV.get(g, g) for g in ngram]
    else:
        parts = list(ngram)
    return "-".join(parts)


def sequence_to_labels(
    df: pd.DataFrame,
    granularity: str,
) -> list[str]:
    """Convert a DA-level DataFrame to a flat list of group/raw labels."""
    return [
        get_label(row[DA_COLUMN], row["da_group"], granularity)
        for _, row in df.iterrows()
    ]


def count_ngrams(
    labels: list[str],
    n: int,
) -> Counter:
    """
    Count n-grams over the RLE-compressed label sequence.
    RLE ensures n-grams span contiguous *regions* not individual DA rows.
    """
    compressed = rle_compress(labels)
    return Counter(_ngrams(compressed, n))


# ── frequency matrix builder ──────────────────────────────────────────────────

def build_frequency_matrix(
    group_sequences: dict[str, list[str]],
    n: int,
    granularity: str,
    normalise: bool = True,
) -> pd.DataFrame:
    """
    Build a frequency matrix for one n-gram size.

    Parameters
    ----------
    group_sequences : dict mapping group_name -> flat label list
    n               : n-gram size
    granularity     : 'groups' or 'raw'
    normalise       : if True, each column sums to 1 (relative frequency)

    Returns
    -------
    DataFrame with:
      index   = n-gram strings  (e.g. "ST-CQ-ST")
      columns = group names
      values  = (normalised) counts
    """
    counters: dict[str, Counter] = {
        grp: count_ngrams(seq, n)
        for grp, seq in group_sequences.items()
    }

    # Union of all observed n-grams
    all_ngrams: set[tuple] = set()
    for c in counters.values():
        all_ngrams.update(c.keys())

    # Sort by total count descending for readability
    ngram_totals = Counter()
    for c in counters.values():
        ngram_totals.update(c)
    sorted_ngrams = [ng for ng, _ in ngram_totals.most_common()]

    groups = list(group_sequences.keys())
    matrix = pd.DataFrame(
        index=[_ngram_label(ng, granularity) for ng in sorted_ngrams],
        columns=groups,
        dtype=float,
    )
    for ng in sorted_ngrams:
        label = _ngram_label(ng, granularity)
        for grp in groups:
            matrix.loc[label, grp] = float(counters[grp].get(ng, 0))

    if normalise:
        col_sums = matrix.sum(axis=0)
        col_sums[col_sums == 0] = 1
        matrix = matrix.div(col_sums, axis=1)

    return matrix


def build_combined_csv(
    group_sequences: dict[str, list[str]],
    ngram_sizes: list[int],
    granularity: str,
) -> pd.DataFrame:
    """
    Build a single tall DataFrame combining all n-gram sizes.
    Rows = n-gram strings, with an extra 'ngram_size' and 'raw_count_total'
    column prepended so rows from different sizes are distinguishable.
    """
    frames = []
    for n in ngram_sizes:
        df_freq = build_frequency_matrix(group_sequences, n, granularity, normalise=True)
        df_raw  = build_frequency_matrix(group_sequences, n, granularity, normalise=False)
        df_freq.insert(0, "ngram_size",        n)
        df_freq.insert(1, "raw_count_total",   df_raw.sum(axis=1).values)
        frames.append(df_freq)
    return pd.concat(frames, axis=0)


# ── chi-squared comparison ────────────────────────────────────────────────────

def chi2_across_groups(
    group_sequences: dict[str, list[str]],
    ngram_sizes: list[int],
    granularity: str,
    min_count: int = 5,
) -> pd.DataFrame:
    """
    For each n-gram, test whether its frequency differs significantly across
    groups using a chi-squared test of independence with FDR correction.

    Only n-grams with at least min_count total observations are tested.

    Returns a DataFrame sorted by p_fdr with columns:
      ngram, ngram_size, chi2, p_raw, p_fdr, significant, *per_group_counts
    """
    results = []
    groups  = list(group_sequences.keys())

    for n in ngram_sizes:
        raw_matrix = build_frequency_matrix(
            group_sequences, n, granularity, normalise=False
        ).drop(columns=["ngram_size", "raw_count_total"], errors="ignore")

        for ngram_str in raw_matrix.index:
            counts = raw_matrix.loc[ngram_str, groups].values.astype(float)
            total  = counts.sum()
            if total < min_count:
                continue
            # 2×k contingency table: [this ngram counts, other ngram counts]
            col_totals = raw_matrix[groups].sum(axis=0).values.astype(float)
            other      = col_totals - counts
            # Avoid negative "other" from float imprecision
            other      = np.clip(other, 0, None)
            table      = np.vstack([counts, other])
            try:
                chi2, p, _, _ = chi2_contingency(table)
            except ValueError:
                continue
            row = {
                "ngram":      ngram_str,
                "ngram_size": n,
                "chi2":       round(chi2, 4),
                "p_raw":      p,
            }
            for grp, cnt in zip(groups, counts):
                row[f"count_{grp}"] = int(cnt)
            results.append(row)

    if not results:
        return pd.DataFrame()

    df  = pd.DataFrame(results)
    _, p_fdr, _, _ = multipletests(df["p_raw"], method="fdr_bh")
    df["p_fdr"]       = p_fdr
    df["significant"] = p_fdr < 0.05
    return df.sort_values("p_fdr").reset_index(drop=True)


# ── transcript summary stats ──────────────────────────────────────────────────

def build_transcript_summary(
    transcript_sequences: dict[str, list[str]],
    ngram_sizes: list[int],
    granularity: str,
) -> pd.DataFrame:
    """
    For the across-transcripts comparison, rather than one column per
    transcript (which would be very wide), compute summary statistics for
    each n-gram across all transcripts:
      mean_freq, std_freq, min_freq, max_freq, n_transcripts_present

    Returns a tall DataFrame (one row per n-gram per n-gram size).
    """
    frames = []
    for n in ngram_sizes:
        # Per-transcript normalised frequency vectors
        per_transcript: dict[str, Counter] = {
            fname: count_ngrams(seq, n)
            for fname, seq in transcript_sequences.items()
        }

        all_ngrams: set[tuple] = set()
        for c in per_transcript.values():
            all_ngrams.update(c.keys())

        # Normalise per-transcript
        norm: dict[str, dict[tuple, float]] = {}
        for fname, ctr in per_transcript.items():
            total = sum(ctr.values()) or 1
            norm[fname] = {ng: cnt / total for ng, cnt in ctr.items()}

        rows = []
        for ng in sorted(all_ngrams,
                         key=lambda x: sum(per_transcript[f].get(x, 0)
                                           for f in per_transcript),
                         reverse=True):
            freqs = [norm[f].get(ng, 0.0) for f in transcript_sequences]
            label = _ngram_label(ng, granularity)
            rows.append({
                "ngram":                 label,
                "ngram_size":            n,
                "raw_count_total":       sum(per_transcript[f].get(ng, 0)
                                             for f in per_transcript),
                "mean_freq":             round(float(np.mean(freqs)),  6),
                "std_freq":              round(float(np.std(freqs)),   6),
                "min_freq":              round(float(np.min(freqs)),   6),
                "max_freq":              round(float(np.max(freqs)),   6),
                "n_transcripts_present": sum(1 for f in freqs if f > 0),
            })
        frames.append(pd.DataFrame(rows))

    return pd.concat(frames, axis=0, ignore_index=True)


def build_difference_summary(
    group_sequences: dict[str, list[str]],
    ngram_sizes: list[int],
    granularity: str,
    min_count: int = 5,
) -> pd.DataFrame:
    """
    Build a single comprehensive summary CSV that captures all differences
    in the BoW representation across groups.

    For every n-gram and every n-gram size the summary includes:

    Identification
      ngram           n-gram string  e.g. "ST-CQ-ST"
      ngram_size      2, 3, 4, or 5

    Frequency per group  (one column each, normalised)
      freq_{group}    normalised frequency within that group

    Raw counts per group
      count_{group}   raw occurrence count

    Pairwise frequency differences  (all ordered pairs A vs B)
      diff_{A}_vs_{B} freq_A - freq_B  (positive = more common in A)
      absdiff_{A}_vs_{B} |freq_A - freq_B|

    Dispersion across groups
      freq_max        highest frequency across all groups
      freq_min        lowest frequency across all groups
      freq_range      freq_max - freq_min
      freq_std        standard deviation of frequencies across groups
      dominant_group  group with the highest frequency for this n-gram

    Statistical significance
      chi2            chi-squared statistic (NaN if not tested)
      p_raw           raw p-value
      p_fdr           FDR-corrected p-value
      significant     True if p_fdr < 0.05

    Rows are sorted by freq_range descending (most variable n-grams first),
    then by p_fdr ascending within ties.
    """
    groups = list(group_sequences.keys())
    frames = []

    for n in ngram_sizes:
        freq_mat = build_frequency_matrix(
            group_sequences, n, granularity, normalise=True
        )
        raw_mat  = build_frequency_matrix(
            group_sequences, n, granularity, normalise=False
        )

        # Chi2 results for this n-gram size
        chi2_df = chi2_across_groups(
            group_sequences, [n], granularity, min_count=min_count
        )
        chi2_lookup: dict[str, dict] = {}
        if not chi2_df.empty:
            for _, row in chi2_df.iterrows():
                chi2_lookup[row["ngram"]] = {
                    "chi2":        row["chi2"],
                    "p_raw":       row["p_raw"],
                    "p_fdr":       row["p_fdr"],
                    "significant": row["significant"],
                }

        rows = []
        for ngram_str in freq_mat.index:
            row: dict = {"ngram": ngram_str, "ngram_size": n}

            # Per-group normalised frequencies and raw counts
            freqs = {}
            for grp in groups:
                f = float(freq_mat.loc[ngram_str, grp])
                c = float(raw_mat.loc[ngram_str, grp])
                row[f"freq_{grp}"]  = round(f, 6)
                row[f"count_{grp}"] = int(c)
                freqs[grp] = f

            # Pairwise frequency differences
            for i, ga in enumerate(groups):
                for gb in groups[i + 1:]:
                    diff = freqs[ga] - freqs[gb]
                    row[f"diff_{ga}_vs_{gb}"]    = round(diff, 6)
                    row[f"absdiff_{ga}_vs_{gb}"] = round(abs(diff), 6)

            # Dispersion
            freq_vals = list(freqs.values())
            row["freq_max"]       = round(max(freq_vals), 6)
            row["freq_min"]       = round(min(freq_vals), 6)
            row["freq_range"]     = round(max(freq_vals) - min(freq_vals), 6)
            row["freq_std"]       = round(float(np.std(freq_vals)), 6)
            row["dominant_group"] = max(freqs, key=freqs.get)

            # Significance
            chi2_info = chi2_lookup.get(ngram_str, {})
            row["chi2"]        = round(chi2_info.get("chi2",  float("nan")), 4)
            row["p_raw"]       = chi2_info.get("p_raw",       float("nan"))
            row["p_fdr"]       = chi2_info.get("p_fdr",       float("nan"))
            row["significant"] = chi2_info.get("significant", False)

            rows.append(row)

        frames.append(pd.DataFrame(rows))

    df = pd.concat(frames, ignore_index=True)

    # Sort: most variable first, then most significant within ties
    df = df.sort_values(
        ["freq_range", "p_fdr"],
        ascending=[False, True],
    ).reset_index(drop=True)

    return df


def build_aggregate_summary(
    group_sequences: dict[str, list[str]],
    ngram_sizes: list[int],
    granularity: str,
    top_dominant: int = 10,
    min_count: int = 5,
) -> dict[str, pd.DataFrame]:
    """
    Compute four aggregate summaries of the BoW n-gram distributions.

    Returns a dict with keys:
      'tfidf'       TF-IDF scores per n-gram per group
      'cosine'      pairwise cosine similarity between group vectors
      'js'          pairwise Jensen-Shannon divergence between groups
      'dominant'    top_dominant n-grams per group per n-gram size
                      (ranked by TF-IDF score)

    All four DataFrames are tall (n-gram size as a column where relevant)
    and combine all requested n-gram sizes into one file.

    TF-IDF
    ------
    TF  = normalised frequency of n-gram in this group
    IDF = log( n_groups / (1 + n_groups_where_ngram_appears) )
    TF-IDF highlights n-grams that are frequent in one group but
    rare across others exactly what's distinctive per group.

    Cosine similarity
    -----------------
    Each group is a unit vector over the n-gram vocabulary.
    cos(A, B) = dot(A, B) / (||A|| * ||B||)
    1 = identical distribution, 0 = completely disjoint vocabulary.
    Computed per n-gram size so you can see whether groups diverge
    more at the bigram or trigram level.

    Jensen-Shannon divergence
    -------------------------
    Symmetric, bounded [0, 1] version of KL divergence.
    0 = identical distributions, 1 = no overlap.
    More principled than cosine for probability distributions.

    Dominant n-gram ranking
    -----------------------
    For each group and each n-gram size, the top_dominant n-grams ranked
    by TF-IDF score. One row per (group, n-gram size, rank).
    """
    groups  = list(group_sequences.keys())
    n_grps  = len(groups)
    smooth  = 1e-9   # smoothing for JS divergence

    tfidf_frames:    list[pd.DataFrame] = []
    cosine_frames:   list[pd.DataFrame] = []
    js_frames:       list[pd.DataFrame] = []
    dominant_frames: list[pd.DataFrame] = []

    for n in ngram_sizes:
        freq_mat = build_frequency_matrix(
            group_sequences, n, granularity, normalise=True
        )   # index=ngrams, columns=groups

        ngram_index = freq_mat.index.tolist()

        # ── TF-IDF ────────────────────────────────────────────────────────────
        # IDF: how many groups contain this n-gram at all
        presence = (freq_mat > 0).sum(axis=1)   # Series: ngram -> count of groups present
        idf      = np.log(n_grps / (1 + presence.values)).reshape(-1, 1)

        tfidf_mat = freq_mat.values * idf   # broadcast: (n_ngrams, n_groups)
        df_tfidf  = pd.DataFrame(
            tfidf_mat,
            index=ngram_index,
            columns=[f"tfidf_{g}" for g in groups],
        )
        df_tfidf.insert(0, "ngram",      ngram_index)
        df_tfidf.insert(1, "ngram_size", n)
        tfidf_frames.append(df_tfidf.reset_index(drop=True))

        # ── Cosine similarity ─────────────────────────────────────────────────
        vecs = freq_mat.values.T   # shape (n_groups, n_ngrams)
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1
        unit_vecs = vecs / norms

        cos_rows = []
        for i, ga in enumerate(groups):
            for j, gb in enumerate(groups):
                if j <= i:
                    continue
                cos_sim = float(np.dot(unit_vecs[i], unit_vecs[j]))
                cos_rows.append({
                    "ngram_size": n,
                    "group_A":    ga,
                    "group_B":    gb,
                    "cosine_similarity": round(cos_sim, 6),
                })
        cosine_frames.append(pd.DataFrame(cos_rows))

        # ── Jensen-Shannon divergence ─────────────────────────────────────────
        js_rows = []
        for i, ga in enumerate(groups):
            for j, gb in enumerate(groups):
                if j <= i:
                    continue
                p = freq_mat[ga].values.astype(float) + smooth
                q = freq_mat[gb].values.astype(float) + smooth
                p /= p.sum();  q /= q.sum()
                m  = 0.5 * (p + q)
                def _kl(a, b):
                    mask = a > 0
                    return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
                jsd = 0.5 * _kl(p, m) + 0.5 * _kl(q, m)
                js_rows.append({
                    "ngram_size":          n,
                    "group_A":             ga,
                    "group_B":             gb,
                    "js_divergence":       round(jsd, 6),
                    "js_similarity":       round(1 - jsd, 6),
                })
        js_frames.append(pd.DataFrame(js_rows))

        # ── Dominant n-gram ranking by TF-IDF ─────────────────────────────────
        dom_rows = []
        for gi, grp in enumerate(groups):
            tfidf_col = tfidf_mat[:, gi]
            top_idx   = np.argsort(tfidf_col)[::-1][:top_dominant]
            for rank, idx in enumerate(top_idx, 1):
                dom_rows.append({
                    "ngram_size":   n,
                    "group":        grp,
                    "rank":         rank,
                    "ngram":        ngram_index[idx],
                    "tfidf_score":  round(float(tfidf_col[idx]), 6),
                    "freq":         round(float(freq_mat.iloc[idx][grp]), 6),
                })
        dominant_frames.append(pd.DataFrame(dom_rows))

    return {
        "tfidf":    pd.concat(tfidf_frames,    ignore_index=True),
        "cosine":   pd.concat(cosine_frames,   ignore_index=True),
        "js":       pd.concat(js_frames,       ignore_index=True),
        "dominant": pd.concat(dominant_frames, ignore_index=True),
    }


def plot_cosine_heatmap(
    df_cosine: pd.DataFrame,
    label: str,
    outdir: str,
    save: bool = True,
    show: bool = False,
):
    """
    Heatmap of pairwise cosine similarity, one subplot per n-gram size.
    """
    ngram_sizes = sorted(df_cosine["ngram_size"].unique())
    n_sizes     = len(ngram_sizes)
    groups      = sorted(set(df_cosine["group_A"]) | set(df_cosine["group_B"]))
    n_grps      = len(groups)
    grp_idx     = {g: i for i, g in enumerate(groups)}

    fig, axes = plt.subplots(1, n_sizes,
                             figsize=(max(4, n_grps * 1.2 + 1) * n_sizes, n_grps * 0.9 + 2),
                             squeeze=False)
    fig.suptitle(f"{label} pairwise cosine similarity", fontsize=11)

    for ax, n in zip(axes[0], ngram_sizes):
        mat = np.eye(n_grps)   # diagonal = 1 (self-similarity)
        sub = df_cosine[df_cosine["ngram_size"] == n]
        for _, row in sub.iterrows():
            i = grp_idx[row["group_A"]]
            j = grp_idx[row["group_B"]]
            mat[i, j] = row["cosine_similarity"]
            mat[j, i] = row["cosine_similarity"]

        im = ax.imshow(mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
        ax.set_xticks(range(n_grps))
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_grps))
        ax.set_yticklabels(groups, fontsize=8)
        ax.set_title(f"{n}-gram", fontsize=10)
        for i in range(n_grps):
            for j in range(n_grps):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="black" if mat[i,j] < 0.8 else "white")
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03).set_label("Cosine sim", fontsize=7)

    plt.tight_layout()
    if save:
        p = os.path.join(outdir, f"ngram_bow_{label}_cosine_heatmap.png")
        plt.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  Saved: {p}")
    if show:
        plt.show()
    plt.close()


def plot_js_heatmap(
    df_js: pd.DataFrame,
    label: str,
    outdir: str,
    save: bool = True,
    show: bool = False,
):
    """
    Heatmap of pairwise JS divergence (lower = more similar), one subplot
    per n-gram size.
    """
    ngram_sizes = sorted(df_js["ngram_size"].unique())
    n_sizes     = len(ngram_sizes)
    groups      = sorted(set(df_js["group_A"]) | set(df_js["group_B"]))
    n_grps      = len(groups)
    grp_idx     = {g: i for i, g in enumerate(groups)}

    fig, axes = plt.subplots(1, n_sizes,
                             figsize=(max(4, n_grps * 1.2 + 1) * n_sizes, n_grps * 0.9 + 2),
                             squeeze=False)
    fig.suptitle(f"{label} pairwise JS divergence (lower = more similar)", fontsize=11)

    for ax, n in zip(axes[0], ngram_sizes):
        mat = np.zeros((n_grps, n_grps))
        sub = df_js[df_js["ngram_size"] == n]
        for _, row in sub.iterrows():
            i = grp_idx[row["group_A"]]
            j = grp_idx[row["group_B"]]
            mat[i, j] = row["js_divergence"]
            mat[j, i] = row["js_divergence"]

        im = ax.imshow(mat, vmin=0, vmax=1, cmap="YlOrRd", aspect="auto")
        ax.set_xticks(range(n_grps))
        ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
        ax.set_yticks(range(n_grps))
        ax.set_yticklabels(groups, fontsize=8)
        ax.set_title(f"{n}-gram", fontsize=10)
        for i in range(n_grps):
            for j in range(n_grps):
                ax.text(j, i, f"{mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=7,
                        color="black" if mat[i,j] < 0.6 else "white")
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.03).set_label("JS div", fontsize=7)

    plt.tight_layout()
    if save:
        p = os.path.join(outdir, f"ngram_bow_{label}_js_heatmap.png")
        plt.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  Saved: {p}")
    if show:
        plt.show()
    plt.close()


def plot_dominant_ngrams(
    df_dominant: pd.DataFrame,
    label: str,
    outdir: str,
    top_n: int = 10,
    save: bool = True,
    show: bool = False,
):
    """
    Horizontal bar chart of top-N n-grams by TF-IDF score per group,
    one subplot row per group, one column per n-gram size.
    """
    ngram_sizes = sorted(df_dominant["ngram_size"].unique())
    groups      = df_dominant["group"].unique().tolist()
    n_sizes     = len(ngram_sizes)
    n_grps      = len(groups)
    cmap        = plt.get_cmap("tab10")

    fig, axes = plt.subplots(
        n_grps, n_sizes,
        figsize=(max(6, top_n * 0.55) * n_sizes, max(3, n_grps * 2.8)),
        squeeze=False,
    )
    fig.suptitle(f"{label} dominant n-grams by TF-IDF", fontsize=12, y=1.01)

    for gi, grp in enumerate(groups):
        color = cmap(gi / max(n_grps, 1))
        for ni, n in enumerate(ngram_sizes):
            ax  = axes[gi][ni]
            sub = (df_dominant[(df_dominant["group"] == grp) &
                               (df_dominant["ngram_size"] == n)]
                   .sort_values("tfidf_score", ascending=True)
                   .tail(top_n))
            if sub.empty:
                ax.axis("off")
                continue
            ax.barh(sub["ngram"], sub["tfidf_score"],
                    color=color, alpha=0.82, edgecolor="white")
            ax.set_title(f"{grp}  |  {n}-gram", fontsize=9)
            ax.set_xlabel("TF-IDF score", fontsize=8)
            ax.grid(True, axis="x", color="lightgrey", linewidth=0.5)

    plt.tight_layout()
    if save:
        p = os.path.join(outdir, f"ngram_bow_{label}_dominant_tfidf.png")
        plt.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  Saved: {p}")
    if show:
        plt.show()
    plt.close()


# ── heatmap ───────────────────────────────────────────────────────────────────

def plot_heatmap(
    df_freq: pd.DataFrame,
    n: int,
    title: str,
    outdir: str,
    fname: str,
    top_n: int = 40,
    save: bool = True,
    show: bool = False,
):
    """
    Heatmap of n-gram frequencies: rows = top_n most frequent n-grams,
    columns = groups.  Values are normalised frequencies (colour scale 0→max).
    """
    # Filter to this n-gram size and drop metadata columns
    meta_cols = {"ngram_size", "raw_count_total"}
    group_cols = [c for c in df_freq.columns if c not in meta_cols]

    if "ngram_size" in df_freq.columns:
        sub = df_freq[df_freq["ngram_size"] == n].copy()
        sub = sub[group_cols]
    else:
        sub = df_freq[group_cols].copy()

    # Keep top_n rows by row sum (most frequent n-grams overall)
    sub = sub.astype(float)
    sub = sub.loc[sub.sum(axis=1).nlargest(top_n).index]

    if sub.empty:
        return

    n_rows = len(sub)
    n_cols = len(group_cols)
    fig_h  = max(5, n_rows * 0.28 + 1.5)
    fig_w  = max(6, n_cols * 1.2 + 2)

    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    mat     = sub.values
    im      = ax.imshow(mat, aspect="auto", cmap="YlOrRd", vmin=0)

    ax.set_xticks(range(n_cols))
    ax.set_xticklabels(group_cols, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_rows))
    ax.set_yticklabels(sub.index, fontsize=8)
    ax.set_title(f"{title}  |  {n}-gram", fontsize=11)

    cbar = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02)
    cbar.set_label("Normalised frequency", fontsize=8)

    # Annotate cells with value if ≥ 0.005
    for i in range(n_rows):
        for j in range(n_cols):
            v = mat[i, j]
            if v >= 0.005:
                ax.text(j, i, f"{v:.3f}", ha="center", va="center",
                        fontsize=6,
                        color="black" if v < 0.6 * mat.max() else "white")

    plt.tight_layout()
    if save:
        fpath = os.path.join(outdir, fname)
        plt.savefig(fpath, bbox_inches="tight", dpi=150)
        print(f"  Saved: {fpath}")
    if show:
        plt.show()
    plt.close()


# ── top-n-grams bar chart ─────────────────────────────────────────────────────

def plot_top_ngrams_bar(
    df_freq: pd.DataFrame,
    n: int,
    title: str,
    outdir: str,
    fname: str,
    top_n: int = 20,
    save: bool = True,
    show: bool = False,
):
    """
    Grouped bar chart: top_n most frequent n-grams, one bar per group.
    Complements the heatmap for groups with few columns.
    """
    meta_cols  = {"ngram_size", "raw_count_total"}
    group_cols = [c for c in df_freq.columns if c not in meta_cols]

    if "ngram_size" in df_freq.columns:
        sub = df_freq[df_freq["ngram_size"] == n][group_cols].astype(float)
    else:
        sub = df_freq[group_cols].astype(float)

    sub = sub.loc[sub.sum(axis=1).nlargest(top_n).index]
    if sub.empty:
        return

    x      = np.arange(len(sub))
    n_grps = len(group_cols)
    width  = 0.8 / max(n_grps, 1)
    cmap   = plt.get_cmap("tab10")

    fig, ax = plt.subplots(figsize=(max(10, len(sub) * 0.5), 5))
    for gi, grp in enumerate(group_cols):
        offset = (gi - (n_grps - 1) / 2) * width
        ax.bar(x + offset, sub[grp].values, width,
               label=grp, color=cmap(gi / max(n_grps, 1)),
               alpha=0.82, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(sub.index, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Normalised frequency")
    ax.set_title(f"{title}  |  {n}-gram  (top {top_n})", fontsize=11)
    ax.legend(fontsize=9, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, axis="y", color="lightgrey", linewidth=0.5)
    plt.tight_layout()

    if save:
        fpath = os.path.join(outdir, fname)
        plt.savefig(fpath, bbox_inches="tight", dpi=150)
        print(f"  Saved: {fpath}")
    if show:
        plt.show()
    plt.close()


def plot_bow_dendrogram(
    group_sequences: dict[str, list[str]],
    ngram_sizes: list[int],
    granularity: str,
    label: str,
    outdir: str,
    save: bool = True,
    show: bool = False,
):
    """
    For each n-gram size, draw a hierarchical clustering dendrogram of the
    groups based on JS divergence between their n-gram frequency distributions.

    JS divergence is used as the distance metric because:
      - it is symmetric and bounded [0, 1]
      - it treats each group's frequency vector as a probability distribution,
        which is the natural representation for BoW comparisons
      - it is consistent with the JS scores already computed in build_aggregate_summary

    One PNG per n-gram size:
      {outdir}/ngram_bow_{label}_dendrogram_n{N}.png

    Leaf labels are the group names.  If a 'non_important' group is present
    its leaf is coloured orange so it stands out as the baseline.
    """
    from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram
    from scipy.spatial.distance import squareform

    smooth = 1e-9
    groups = list(group_sequences.keys())
    n_grps = len(groups)

    if n_grps < 3:
        # Need at least 3 leaves for a meaningful dendrogram
        print(f"  [dendrogram:{label}] Need ≥3 groups for dendrogram, "
              f"got {n_grps}. Skipping.")
        return

    for n in ngram_sizes:
        freq_mat = build_frequency_matrix(
            group_sequences, n, granularity, normalise=True
        )   # rows=ngrams, cols=groups

        # Build pairwise JS distance matrix
        dist_mat = np.zeros((n_grps, n_grps))
        for i, ga in enumerate(groups):
            for j, gb in enumerate(groups):
                if j <= i:
                    continue
                p = freq_mat[ga].values.astype(float) + smooth
                q = freq_mat[gb].values.astype(float) + smooth
                p /= p.sum();  q /= q.sum()
                m  = 0.5 * (p + q)
                mask_p = p > 0;  mask_q = q > 0
                kl_pm  = float(np.sum(p[mask_p] * np.log2(p[mask_p] / m[mask_p])))
                kl_qm  = float(np.sum(q[mask_q] * np.log2(q[mask_q] / m[mask_q])))
                jsd    = 0.5 * kl_pm + 0.5 * kl_qm
                dist_mat[i, j] = jsd
                dist_mat[j, i] = jsd

        condensed = squareform(dist_mat, checks=False)
        Z         = linkage(condensed, method="average")

        fig_w = max(7, n_grps * 0.8 + 2)
        fig, ax = plt.subplots(figsize=(fig_w, 5))

        # scipy_dendrogram returns ddata which includes "ivl" the leaf labels
        # in left-to-right display order.  We use this to colour tick labels
        # reliably without needing a canvas draw pass.
        ddata = scipy_dendrogram(
            Z,
            labels=groups,
            ax=ax,
            leaf_rotation=45,
            color_threshold=0.0,
            above_threshold_color="dimgray",
            link_color_func=lambda _: "dimgray",
        )

        # Build coloured tick labels using the display-order leaf list.
        # Pass strings first to set_xticklabels (which creates the Text objects),
        # then iterate and colour the now-populated Text objects.
        leaf_labels_ordered = ddata["ivl"]
        ax.set_xticklabels(leaf_labels_ordered, rotation=45, ha="right", fontsize=9)
        for tick in ax.get_xticklabels():
            if tick.get_text() == "non_important":
                tick.set_color("#e05c3a")
                tick.set_fontweight("bold")
            else:
                tick.set_color("#2a6099")

        ax.set_title(
            f"{label} JS divergence clustering  |  {n}-gram\n"
            f"(AVGLINK; distance = JS divergence between n-gram distributions)",
            fontsize=10,
        )
        ax.set_ylabel("JS divergence")
        ax.grid(True, axis="y", color="lightgrey", linewidth=0.5)

        # Legend
        from matplotlib.lines import Line2D
        handles = [
            Line2D([0], [0], color="#2a6099", linewidth=4, label="group"),
        ]
        if "non_important" in groups:
            handles.append(
                Line2D([0], [0], color="#e05c3a", linewidth=4,
                       label="non_important (baseline)")
            )
        ax.legend(handles=handles, fontsize=9, loc="upper right")

        plt.tight_layout()
        if save:
            p = os.path.join(outdir, f"ngram_bow_{label}_dendrogram_n{n}.png")
            plt.savefig(p, bbox_inches="tight", dpi=150)
            print(f"  Saved: {p}")
        if show:
            plt.show()
        plt.close()


# ── run one comparison level ──────────────────────────────────────────────────

def run_comparison(
    group_sequences: dict[str, list[str]],
    label: str,
    outdir: str,
    ngram_sizes: list[int],
    granularity: str,
    transcript_mode: bool = False,
    top_n_heatmap: int = 40,
    top_n_bar: int = 20,
    min_chi2_count: int = 5,
    save: bool = True,
    show: bool = False,
):
    """
    Full pipeline for one comparison level (importance / transcripts / therapists).

    Parameters
    ----------
    group_sequences : dict  group_name -> flat DA label list
    label           : str   subdirectory name and title prefix
    transcript_mode : bool  if True, produce summary-stats CSV instead of
                            per-group frequency matrix (for many-transcript case)
    """
    level_dir = _subdir(outdir, "ngram_bow", label)
    print(f"\n{'='*60}")
    print(f"  N-gram BoW: {label}  |  granularity: {granularity}")
    print(f"{'='*60}")

    if transcript_mode:
        # ── summary stats across transcripts ──────────────────────────────────
        df_summary = build_transcript_summary(group_sequences, ngram_sizes, granularity)
        csv_path   = os.path.join(level_dir, f"ngram_bow_{label}.csv")
        df_summary.to_csv(csv_path, index=False)
        print(f"  Saved: {csv_path}")

        # Heatmap and bar chart using mean_freq as the value
        for n in ngram_sizes:
            sub = df_summary[df_summary["ngram_size"] == n].set_index("ngram")
            # Pivot to a single-column frequency matrix for heatmap
            freq_col = sub[["mean_freq"]].copy()
            freq_col.columns = ["mean_freq_across_transcripts"]

            plot_heatmap(
                freq_col.reset_index().rename(columns={"ngram": "index"}).set_index("index"),
                n=n,
                title=f"{label} mean frequency",
                outdir=level_dir,
                fname=f"ngram_bow_{label}_heatmap_n{n}.png",
                top_n=top_n_heatmap,
                save=save, show=show,
            )

        # Chi-squared across transcripts
        df_chi2 = chi2_across_groups(
            group_sequences, ngram_sizes, granularity, min_count=min_chi2_count
        )
        if not df_chi2.empty:
            chi2_path = os.path.join(level_dir, f"ngram_bow_{label}_chi2.csv")
            df_chi2.to_csv(chi2_path, index=False)
            print(f"  Saved: {chi2_path}")
            sig = df_chi2[df_chi2["significant"]]
            print(f"  Chi-squared: {len(sig)}/{len(df_chi2)} n-grams differ "
                  f"significantly across transcripts (FDR<0.05)")

        # Difference summary (freq_range = std across transcripts in this mode)
        df_diff = build_aggregate_summary(
            group_sequences, ngram_sizes, granularity, min_count=min_chi2_count
        )
        for key, df_out in df_diff.items():
            if not df_out.empty:
                p = os.path.join(level_dir, f"ngram_bow_{label}_{key}.csv")
                df_out.to_csv(p, index=False)
                print(f"  Saved: {p}")

        # Dendrogram
        plot_bow_dendrogram(
            group_sequences, ngram_sizes, granularity,
            label=label, outdir=level_dir, save=save, show=show,
        )
        return

    # ── standard group-level frequency matrix ─────────────────────────────────
    df_combined = build_combined_csv(group_sequences, ngram_sizes, granularity)

    csv_path = os.path.join(level_dir, f"ngram_bow_{label}.csv")
    df_combined.to_csv(csv_path)
    print(f"  Saved: {csv_path}")

    # Per n-gram-size: heatmap + bar chart
    for n in ngram_sizes:
        plot_heatmap(
            df_combined, n=n,
            title=label,
            outdir=level_dir,
            fname=f"ngram_bow_{label}_heatmap_n{n}.png",
            top_n=top_n_heatmap,
            save=save, show=show,
        )
        plot_top_ngrams_bar(
            df_combined, n=n,
            title=label,
            outdir=level_dir,
            fname=f"ngram_bow_{label}_bar_n{n}.png",
            top_n=top_n_bar,
            save=save, show=show,
        )

    # Chi-squared
    df_chi2 = chi2_across_groups(
        group_sequences, ngram_sizes, granularity, min_count=min_chi2_count
    )
    if not df_chi2.empty:
        chi2_path = os.path.join(level_dir, f"ngram_bow_{label}_chi2.csv")
        df_chi2.to_csv(chi2_path, index=False)
        print(f"  Saved: {chi2_path}")
        sig = df_chi2[df_chi2["significant"]]
        print(f"  Chi-squared: {len(sig)}/{len(df_chi2)} n-grams differ "
              f"significantly (FDR<0.05).")
        if not sig.empty:
            print(sig[["ngram", "ngram_size", "chi2", "p_fdr"]]
                  .head(10).to_string(index=False))

    # ── aggregate summary: TF-IDF, cosine, JS, dominant n-grams ─────────────
    agg = build_aggregate_summary(
        group_sequences, ngram_sizes, granularity, min_count=min_chi2_count
    )

    for key, df_out in agg.items():
        if not df_out.empty:
            p = os.path.join(level_dir, f"ngram_bow_{label}_{key}.csv")
            df_out.to_csv(p, index=False)
            print(f"  Saved: {p}")

    # Cosine similarity heatmap
    if not agg["cosine"].empty:
        plot_cosine_heatmap(agg["cosine"], label, level_dir, save=save, show=show)

    # JS divergence heatmap
    if not agg["js"].empty:
        plot_js_heatmap(agg["js"], label, level_dir, save=save, show=show)

    # Dendrogram: one per n-gram size, JS distance between group distributions
    plot_bow_dendrogram(
        group_sequences, ngram_sizes, granularity,
        label=label, outdir=level_dir, save=save, show=show,
    )

    # Dominant n-grams bar chart
    if not agg["dominant"].empty:
        plot_dominant_ngrams(agg["dominant"], label, level_dir,
                             save=save, show=show)

    # Console summary: top 5 dominant n-grams per group at bigram level
    if not agg["dominant"].empty:
        n_preview = ngram_sizes[0]
        print(f"\n  Top-5 dominant n-grams at {n_preview}-gram level ({label}):")
        preview = (agg["dominant"][agg["dominant"]["ngram_size"] == n_preview]
                   .sort_values(["group", "rank"])
                   .groupby("group")
                   .head(5))
        print(preview[["group", "rank", "ngram", "tfidf_score", "freq"]]
              .to_string(index=False))


# ── importance sequence builder (with optional context window) ────────────────

def build_importance_sequences(
    dfs: dict[str, pd.DataFrame],
    granularity: str,
    context_window: int = 15,
    include_context: bool = False,
) -> dict[str, list[str]]:
    """
    Build flat DA label sequences for the three importance groups, operating
    per source file so context windows never bleed across session boundaries.

    Parameters
    ----------
    dfs             : dict  filename -> DA-level DataFrame
    granularity     : 'groups' or 'raw'
    context_window  : number of DA rows to include before/after each
                      important block when include_context=True
    include_context : if True, the ±context_window rows around each important
                      block are pulled into the important sequence (patient or
                      therapist depending on which block owns them) and
                      excluded from non_important.
                      if False, context rows stay in non_important (default).

    Returns
    -------
    dict with keys 'patient_important', 'therapist_important', 'non_important'
    each mapping to a flat list of label strings.
    """
    patient_seqs:    list[str] = []
    therapist_seqs:  list[str] = []
    nonimportant_seqs: list[str] = []

    for fname, df in dfs.items():
        df = df.reset_index(drop=True)

        # Extract patient important blocks with context
        p_blocks = extract_important_blocks(
            df,
            importance_col="patient_important",
            code_col="patient_code",
            granularity=granularity,
            context_window=context_window,
            include_context_in_block=include_context,
            bucketed_runs=False,
        )
        for blk in p_blocks:
            patient_seqs.extend(blk["full_sequence"])

        # Extract therapist important blocks with context
        t_blocks = extract_important_blocks(
            df,
            importance_col="therapist_important",
            code_col="therapist_code",
            granularity=granularity,
            context_window=context_window,
            include_context_in_block=include_context,
            bucketed_runs=False,
        )
        for blk in t_blocks:
            therapist_seqs.extend(blk["full_sequence"])

        # Non-important: all rows not claimed by either partition
        # (context rows excluded from non_important when include_context=True)
        ni_seqs = extract_nonimportant_sequences(
            df,
            granularity=granularity,
            context_window=context_window,
            include_context_in_block=include_context,
            bucketed_runs=False,
        )
        for seq in ni_seqs:
            nonimportant_seqs.extend(seq)

    return {
        "patient_important":   patient_seqs,
        "therapist_important": therapist_seqs,
        "non_important":       nonimportant_seqs,
    }


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="N-gram bag-of-words DA analysis across importance, "
                    "transcripts, and therapists."
    )
    parser.add_argument("--dir",            required=True,
                        help="Directory of word-level daseg CSV files.")
    parser.add_argument("--granularity",    default="groups",
                        choices=["groups", "raw"],
                        help="'groups' (default) or 'raw' DA class strings.")
    parser.add_argument("--ngram_sizes",    nargs="+", type=int,
                        default=[2, 3, 4, 5])
    parser.add_argument("--top_n_heatmap",  type=int, default=40,
                        help="Max n-grams shown in heatmap (default 40).")
    parser.add_argument("--top_n_bar",      type=int, default=20,
                        help="Max n-grams shown in bar chart (default 20).")
    parser.add_argument("--min_chi2_count", type=int, default=5,
                        help="Min total count for a n-gram to be chi2-tested.")
    parser.add_argument("--context_window",  type=int, default=15,
                        help="DA rows of context around each important block "
                             "(default: 15). Only used for importance comparison.")
    parser.add_argument("--include_context", action="store_true",
                        help="Absorb the ±context_window DAs around each "
                             "important block into that block's sequence, "
                             "excluding them from non_important. "
                             "Default: context rows stay in non_important.")
    parser.add_argument("--outdir",         default="ngram_bow_output/")
    parser.add_argument("--show",           action="store_true")
    args = parser.parse_args()

    dir_path = Path(args.dir)
    if not dir_path.exists():
        raise ValueError(f"Directory not found: {args.dir}")
    os.makedirs(args.outdir, exist_ok=True)

    # ── load all files ────────────────────────────────────────────────────────
    allowed_ext = {".csv", ".tsv", ".xlsx"}
    dfs: dict[str, pd.DataFrame] = {}
    for fp in sorted(dir_path.iterdir()):
        if fp.suffix.lower() in allowed_ext:
            print(f"Loading {fp.name} …")
            df = load_da_level(fp)
            df["source_file"] = fp.name
            dfs[fp.name] = df

    if not dfs:
        raise RuntimeError(f"No data files found in {args.dir}")

    # Add _label column to each per-file DataFrame before combining,
    # so transcript sequences can be built directly from dfs later.
    for fname, df in dfs.items():
        dfs[fname]["_label"] = [
            get_label(row[DA_COLUMN], row["da_group"], args.granularity)
            for _, row in df.iterrows()
        ]

    combined = pd.concat(dfs.values(), ignore_index=True)
    combined = combined[combined[DA_COLUMN] != "I-"].reset_index(drop=True)
    print(f"\nTotal DA-level rows: {len(combined)}")

    # ── 1. Importance-level comparison ───────────────────────────────────────
    # Three groups: patient_important, therapist_important, non_important.
    # A row can belong to both patient and therapist important simultaneously;
    # in that case it contributes to both groups (matching existing script logic).
    # When --include_context is set, the ±context_window DA rows around each
    # important block are pulled into that block's sequence and excluded from
    # non_important.  Context extraction is done per source file so windows
    # never bleed across session boundaries.
    print("\n--- Building importance-level sequences ---")
    context_note = (
        f"context_window={args.context_window}, included in block"
        if args.include_context
        else f"context_window={args.context_window}, stays in non_important"
    )
    print(f"  Context setting: {context_note}")

    importance_sequences = build_importance_sequences(
        dfs,
        granularity=args.granularity,
        context_window=args.context_window,
        include_context=args.include_context,
    )

    for grp, seq in importance_sequences.items():
        print(f"  {grp}: {len(seq)} DA rows")

    # Subdirectory name reflects context setting so different runs don't
    # overwrite each other
    importance_label = (
        f"importance_ctx{args.context_window}_included"
        if args.include_context
        else f"importance_ctx{args.context_window}_excluded"
    )

    run_comparison(
        importance_sequences,
        label=importance_label,
        outdir=args.outdir,
        ngram_sizes=args.ngram_sizes,
        granularity=args.granularity,
        transcript_mode=False,
        top_n_heatmap=args.top_n_heatmap,
        top_n_bar=args.top_n_bar,
        min_chi2_count=args.min_chi2_count,
        save=True, show=args.show,
    )

    # ── 2. Per-transcript comparison ──────────────────────────────────────────
    # Each transcript is one group; produces summary stats CSV rather than
    # a per-transcript frequency matrix (avoids an extremely wide table).
    print("\n--- Building per-transcript sequences ---")

    transcript_sequences: dict[str, list[str]] = {
        fname: df["_label"].tolist()
        for fname, df in dfs.items()
    }
    for fname, seq in transcript_sequences.items():
        print(f"  {fname}: {len(seq)} DA rows")

    run_comparison(
        transcript_sequences,
        label="transcripts",
        outdir=args.outdir,
        ngram_sizes=args.ngram_sizes,
        granularity=args.granularity,
        transcript_mode=True,      # summary stats, not per-group matrix
        top_n_heatmap=args.top_n_heatmap,
        top_n_bar=args.top_n_bar,
        min_chi2_count=args.min_chi2_count,
        save=True, show=args.show,
    )

    # ── 3. Per-therapist comparison ───────────────────────────────────────────
    # Pool all transcripts belonging to each therapist (both speakers combined).
    print("\n--- Building per-therapist sequences ---")

    therapist_label_groups: dict[str, list[str]] = {}
    for fname, df in dfs.items():
        tid = extract_therapist_id(fname)
        key = f"therapist_{tid}"
        therapist_label_groups.setdefault(key, [])
        therapist_label_groups[key].extend(df["_label"].tolist())

    therapist_sequences = dict(sorted(therapist_label_groups.items()))
    for key, seq in therapist_sequences.items():
        n_files = sum(
            1 for fn in dfs
            if extract_therapist_id(fn) == key.replace("therapist_", "")
        )
        print(f"  {key}: {len(seq)} DA rows across {n_files} transcript(s)")

    run_comparison(
        therapist_sequences,
        label="therapists",
        outdir=args.outdir,
        ngram_sizes=args.ngram_sizes,
        granularity=args.granularity,
        transcript_mode=False,
        top_n_heatmap=args.top_n_heatmap,
        top_n_bar=args.top_n_bar,
        min_chi2_count=args.min_chi2_count,
        save=True, show=args.show,
    )

    # ── 4. Patient importance codes vs non-important ──────────────────────────
    # One group per unique patient code (comma-separated codes exploded so each
    # code gets credit), plus non_important as a baseline group.
    print("\n--- Building patient-code sequences ---")

    patient_code_seqs: dict[str, list[str]] = {}
    for fname, df in dfs.items():
        df = df.reset_index(drop=True)
        p_blocks = extract_important_blocks(
            df,
            importance_col="patient_important",
            code_col="patient_code",
            granularity=args.granularity,
            context_window=args.context_window,
            include_context_in_block=args.include_context,
            bucketed_runs=False,
        )
        for blk in p_blocks:
            for code in blk["codes"]:
                code = code.strip()
                if code in ("", "nan", "NaN", "None", "NA"):
                    continue
                patient_code_seqs.setdefault(f"patient_{code}", [])
                patient_code_seqs[f"patient_{code}"].extend(blk["full_sequence"])

    # Add non-important as baseline reuse the already-built sequences
    patient_code_seqs["non_important"] = importance_sequences["non_important"]

    if len(patient_code_seqs) >= 2:
        for key, seq in patient_code_seqs.items():
            print(f"  {key}: {len(seq)} DA rows")
        run_comparison(
            patient_code_seqs,
            label="patient_codes_vs_nonimportant",
            outdir=args.outdir,
            ngram_sizes=args.ngram_sizes,
            granularity=args.granularity,
            transcript_mode=False,
            top_n_heatmap=args.top_n_heatmap,
            top_n_bar=args.top_n_bar,
            min_chi2_count=args.min_chi2_count,
            save=True, show=args.show,
        )
    else:
        print("  Not enough patient codes found skipping.")

    # ── 5. Therapist importance codes vs non-important ────────────────────────
    print("\n--- Building therapist-code sequences ---")

    therapist_code_seqs: dict[str, list[str]] = {}
    for fname, df in dfs.items():
        df = df.reset_index(drop=True)
        t_blocks = extract_important_blocks(
            df,
            importance_col="therapist_important",
            code_col="therapist_code",
            granularity=args.granularity,
            context_window=args.context_window,
            include_context_in_block=args.include_context,
            bucketed_runs=False,
        )
        for blk in t_blocks:
            for code in blk["codes"]:
                code = code.strip()
                if code in ("", "nan", "NaN", "None", "NA"):
                    continue
                therapist_code_seqs.setdefault(f"therapist_{code}", [])
                therapist_code_seqs[f"therapist_{code}"].extend(blk["full_sequence"])

    # Add non-important as baseline
    therapist_code_seqs["non_important"] = importance_sequences["non_important"]

    if len(therapist_code_seqs) >= 2:
        for key, seq in therapist_code_seqs.items():
            print(f"  {key}: {len(seq)} DA rows")
        run_comparison(
            therapist_code_seqs,
            label="therapist_codes_vs_nonimportant",
            outdir=args.outdir,
            ngram_sizes=args.ngram_sizes,
            granularity=args.granularity,
            transcript_mode=False,
            top_n_heatmap=args.top_n_heatmap,
            top_n_bar=args.top_n_bar,
            min_chi2_count=args.min_chi2_count,
            save=True, show=args.show,
        )
    else:
        print("  Not enough therapist codes found skipping.")

    # ── directory summary ─────────────────────────────────────────────────────
    print(f"\nDone. Outputs in: {args.outdir}")
    print("\nOutput directory layout:")
    for root, dirs, files in os.walk(args.outdir):
        depth = root.replace(args.outdir, "").count(os.sep)
        print("  " * depth + os.path.basename(root) + "/")
        for f in sorted(files):
            print("  " * (depth + 1) + f)


if __name__ == "__main__":
    main()