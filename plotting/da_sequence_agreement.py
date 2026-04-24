
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


import argparse
import os
import re
import warnings
from collections import defaultdict
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency

from plotting.common_patterns import (
    DA_COLUMN,
    DA_GROUP_ABBREV,
    DA_GROUP_COLORS,
    load_da_level,
    get_label,
    _parse_codes,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── therapist ID parsing ──────────────────────────────────────────────────────

def parse_therapist_id(filename: str) -> str:
    """
    Extract therapist ID from filename.
    Finds the digit(s) immediately following 'T' before the first underscore.
    E.g. 'randomT1_session.csv' → '1', 'sessionT12_abc.csv' → '12'.
    Falls back to 'unknown' if no match.
    """
    stem = Path(filename).stem
    before_first_underscore = stem.split("_")[0]
    match = re.search(r"T(\d+)", before_first_underscore, re.IGNORECASE)
    if match:
        return match.group(1)
    return "unknown"


# ── data loading ──────────────────────────────────────────────────────────────

def load_transcripts(
    dir_path:    Path,
    target:      str,
    granularity: str,
) -> list[dict]:
    """
    Load all transcripts from dir_path.  Returns a list of dicts:
      {
        filename:     str,
        therapist_id: str,
        df:           pd.DataFrame,   # DA-level with importance + code cols
        target_col:   str,            # e.g. 'patient_important'
        code_col:     str,            # e.g. 'patient_code'
        granularity:  str,
      }
    Skips files missing the target column.
    """
    target_col  = f"{target}_important"
    code_col    = f"{target}_code"
    allowed_ext = {".csv", ".tsv", ".xlsx"}
    records     = []

    for fp in sorted(dir_path.iterdir()):
        if fp.suffix.lower() not in allowed_ext:
            continue
        print(f"Loading {fp.name} …")
        df = load_da_level(fp)
        df = df[df[DA_COLUMN] != "I-"].reset_index(drop=True)

        if target_col not in df.columns:
            print(f"  Warning: '{target_col}' not found — skipping.")
            continue

        df[target_col] = df[target_col].fillna(0).astype(int)
        n_pos = int(df[target_col].sum())
        n_tot = len(df)

        if n_pos == 0:
            print(f"  No important labels — skipping.")
            continue

        therapist_id = parse_therapist_id(fp.name)
        print(f"  therapist={therapist_id}  {n_tot} DAs  "
              f"{n_pos} important ({100*n_pos/max(n_tot,1):.1f}%)")

        records.append({
            "filename":     fp.name,
            "therapist_id": therapist_id,
            "df":           df,
            "target_col":   target_col,
            "code_col":     code_col,
            "granularity":  granularity,
        })

    return records


# ── important block extraction ────────────────────────────────────────────────

def _normalise_speaker(raw: str) -> str:
    """Return 'therapist' or 'patient' (catches all non-therapist speakers)."""
    if isinstance(raw, str) and raw.strip().lower() == "therapist":
        return "therapist"
    return "patient"


def extract_blocks(
    rec:            dict,
    context_window: int = 0,
) -> list[dict]:
    """
    Extract contiguous important blocks from one transcript.
    Returns list of:
      {
        therapist_id:        str,
        filename:            str,
        codes:               list[str],
        da_sequence:         list[str],   # all DAs within block (+ optional context)
        da_sequence_therapist: list[str], # only therapist-speaker DAs
        da_sequence_patient:   list[str], # only non-therapist DAs
      }
    Speaker is normalised: 'therapist' stays; everything else becomes 'patient'.
    Context window rows use the same speaker split.
    """
    df          = rec["df"]
    target_col  = rec["target_col"]
    code_col    = rec["code_col"]
    granularity = rec["granularity"]

    importance = df[target_col].values
    da_labels  = [
        get_label(row[DA_COLUMN], row["da_group"], granularity)
        for _, row in df.iterrows()
    ]
    speakers = [
        _normalise_speaker(str(row.get("speaker", "patient")))
        for _, row in df.iterrows()
    ]
    n      = len(df)
    blocks = []

    i = 0
    while i < n:
        if importance[i] == 1:
            j = i
            while j < n and importance[j] == 1:
                j += 1

            # Codes for this block
            codes: list[str] = []
            if code_col in df.columns:
                for raw in df[code_col].iloc[i:j]:
                    for c in _parse_codes(raw):
                        if c not in codes:
                            codes.append(c)
            if not codes:
                codes = ["NA"]

            # DA sequence: block ± optional context
            start = max(0, i - context_window)
            end   = min(n, j + context_window)

            da_sequence           = da_labels[start:end]
            da_sequence_therapist = [
                da for da, spkr in zip(da_labels[start:end], speakers[start:end])
                if spkr == "therapist"
            ]
            da_sequence_patient = [
                da for da, spkr in zip(da_labels[start:end], speakers[start:end])
                if spkr == "patient"
            ]

            blocks.append({
                "therapist_id":          rec["therapist_id"],
                "filename":              rec["filename"],
                "codes":                 codes,
                "da_sequence":           da_sequence,
                "da_sequence_therapist": da_sequence_therapist,
                "da_sequence_patient":   da_sequence_patient,
            })
            i = j
        else:
            i += 1

    return blocks


# ── rate analysis ─────────────────────────────────────────────────────────────

def analyse_rates(
    records: list[dict],
    outdir:  str,
) -> pd.DataFrame:
    """
    Compute per-session important rates, summarise per therapist,
    and produce a boxplot.
    """
    rates_dir = os.path.join(outdir, "rates")
    os.makedirs(rates_dir, exist_ok=True)

    rows = []
    for rec in records:
        df         = rec["df"]
        target_col = rec["target_col"]
        n_total    = len(df)
        n_imp      = int(df[target_col].sum())
        rows.append({
            "therapist_id": rec["therapist_id"],
            "filename":     rec["filename"],
            "n_das":        n_total,
            "n_important":  n_imp,
            "rate":         round(n_imp / max(n_total, 1), 4),
        })

    df_rates = pd.DataFrame(rows).sort_values(["therapist_id", "filename"])
    df_rates.to_csv(os.path.join(rates_dir, "important_rates.csv"), index=False)
    print(f"\n  Saved: {os.path.join(rates_dir, 'important_rates.csv')}")

    # Per-therapist summary
    summary = (
        df_rates.groupby("therapist_id")["rate"]
        .agg(
            n_sessions="count",
            mean_rate="mean",
            std_rate="std",
            min_rate="min",
            max_rate="max",
        )
        .reset_index()
        .round(4)
    )
    summary.to_csv(os.path.join(rates_dir, "rate_summary.csv"), index=False)
    print(f"  Saved: {os.path.join(rates_dir, 'rate_summary.csv')}")

    print("\n  Important rate summary per therapist:")
    print(summary.to_string(index=False))

    # Boxplot
    therapists = sorted(df_rates["therapist_id"].unique())
    fig, ax    = plt.subplots(figsize=(max(5, len(therapists) * 1.2 + 1), 4))
    data       = [df_rates[df_rates["therapist_id"] == t]["rate"].values
                  for t in therapists]
    bp = ax.boxplot(data, labels=[f"T{t}" for t in therapists],
                    patch_artist=True, notch=False)
    colors = plt.cm.Set2(np.linspace(0, 1, len(therapists)))
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_xlabel("Therapist")
    ax.set_ylabel("Proportion of DAs labeled important")
    ax.set_title("Important labeling rate per therapist")
    ax.grid(True, axis="y", color="lightgrey", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(os.path.join(rates_dir, "rate_boxplot.png"),
                bbox_inches="tight", dpi=150)
    print(f"  Saved: {os.path.join(rates_dir, 'rate_boxplot.png')}")
    plt.close()

    return df_rates


# ── code usage analysis ───────────────────────────────────────────────────────

def analyse_code_usage(
    all_blocks: list[dict],
    outdir:     str,
) -> pd.DataFrame:
    """
    Count how many important blocks each therapist assigns each code to.
    Produces a count table and heatmap.
    """
    codes_dir = os.path.join(outdir, "codes")
    os.makedirs(codes_dir, exist_ok=True)

    # One row per (therapist, code) pair
    rows = []
    for blk in all_blocks:
        for code in blk["codes"]:
            rows.append({
                "therapist_id": blk["therapist_id"],
                "code":         code,
            })

    if not rows:
        print("  No coded blocks found.")
        return pd.DataFrame()

    df_long = pd.DataFrame(rows)
    df_counts = (
        df_long.groupby(["therapist_id", "code"])
        .size()
        .reset_index(name="n_blocks")
    )
    df_counts.to_csv(os.path.join(codes_dir, "code_counts.csv"), index=False)
    print(f"\n  Saved: {os.path.join(codes_dir, 'code_counts.csv')}")

    # Pivot for heatmap
    pivot = df_counts.pivot(
        index="therapist_id", columns="code", values="n_blocks"
    ).fillna(0).astype(int)

    # Normalise row-wise so therapists with more sessions aren't dominant
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0).round(3)

    therapists = sorted(pivot.index.tolist())
    codes      = sorted(pivot.columns.tolist())
    n_t        = len(therapists)
    n_c        = len(codes)

    fig, axes = plt.subplots(1, 2, figsize=(max(8, n_c * 0.8 + 3),
                                             max(4, n_t * 0.7 + 2)))

    for ax, data, title, fmt in zip(
        axes,
        [pivot.values, pivot_norm.values],
        ["Block counts per therapist × code",
         "Normalised proportions per therapist × code"],
        [".0f", ".2f"],
    ):
        im = ax.imshow(data, aspect="auto", cmap="YlOrRd")
        ax.set_xticks(range(n_c))
        ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n_t))
        ax.set_yticklabels([f"T{t}" for t in therapists], fontsize=9)
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        for i in range(n_t):
            for j in range(n_c):
                val = data[i, j]
                ax.text(j, i, format(val, fmt),
                        ha="center", va="center", fontsize=8,
                        color="black")

    plt.suptitle("Code usage across therapists", fontsize=11, y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(codes_dir, "code_heatmap.png"),
                bbox_inches="tight", dpi=150)
    print(f"  Saved: {os.path.join(codes_dir, 'code_heatmap.png')}")
    plt.close()

    print("\n  Code usage pivot (counts):")
    print(pivot.to_string())

    return df_counts


# ── sequence distribution helpers ────────────────────────────────────────────

def _da_counts(sequences: list[list[str]], all_da_types: list[str]) -> np.ndarray:
    """
    Count occurrences of each DA type across a list of sequences.
    Returns a 1-D array aligned to all_da_types.
    """
    counter: dict[str, int] = defaultdict(int)
    for seq in sequences:
        for da in seq:
            counter[da] += 1
    return np.array([counter.get(da, 0) for da in all_da_types], dtype=float)


def _js_divergence(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon divergence between two count vectors (not yet normalised)."""
    p = p + 1e-10
    q = q + 1e-10
    p = p / p.sum()
    q = q / q.sum()
    m = 0.5 * (p + q)
    def _kl(a, b):
        return float(np.sum(a * np.log(a / b)))
    return 0.5 * _kl(p, m) + 0.5 * _kl(q, m)


def _chi_square_pair(
    counts_a: np.ndarray,
    counts_b: np.ndarray,
    label_a:  str,
    label_b:  str,
) -> dict:
    """
    Chi-square test of independence between two DA count vectors.
    Merges rare cells (expected < 5) into an 'other' bin.
    """
    contingency = np.vstack([counts_a, counts_b])

    # Drop columns where both are zero
    nonzero = contingency.sum(axis=0) > 0
    contingency = contingency[:, nonzero]

    if contingency.shape[1] < 2:
        return {
            "label_a": label_a, "label_b": label_b,
            "chi2": np.nan, "p_value": np.nan,
            "dof": np.nan, "js_divergence": np.nan,
            "n_a": int(counts_a.sum()), "n_b": int(counts_b.sum()),
            "note": "insufficient_data",
        }

    try:
        chi2, p, dof, _ = chi2_contingency(contingency)
    except ValueError as e:
        return {
            "label_a": label_a, "label_b": label_b,
            "chi2": np.nan, "p_value": np.nan,
            "dof": np.nan, "js_divergence": np.nan,
            "n_a": int(counts_a.sum()), "n_b": int(counts_b.sum()),
            "note": str(e),
        }

    js = _js_divergence(counts_a.copy(), counts_b.copy())

    return {
        "label_a":      label_a,
        "label_b":      label_b,
        "chi2":         round(float(chi2), 4),
        "p_value":      round(float(p),    6),
        "dof":          int(dof),
        "js_divergence": round(js,         4),
        "n_a":          int(counts_a.sum()),
        "n_b":          int(counts_b.sum()),
        "note":         "",
    }


# ── per-code sequence analysis ────────────────────────────────────────────────

def analyse_code_sequences(
    all_blocks:    list[dict],
    granularity:   str,
    outdir:        str,
    seq_key:       str = "da_sequence",
    speaker_label: str = "all",
) -> list[dict]:
    """
    For each code, compare the DA-type distributions of sequences associated
    with that code:
      - Between therapists (pairwise chi-square + JS divergence)
      - Within each therapist across their sessions (pairwise chi-square)
      - Per-therapist code × code JS heatmap

    seq_key controls which sequence field is used:
      "da_sequence"           — all DAs in the block
      "da_sequence_therapist" — therapist-speaker DAs only
      "da_sequence_patient"   — patient-speaker DAs only

    speaker_label is used for output subdirectory naming.

    Returns a list of summary rows for the top-level summary.csv.
    """
    seq_dir = os.path.join(outdir, "sequences", speaker_label)
    os.makedirs(seq_dir, exist_ok=True)

    # Collect all unique DA types for this speaker slice
    all_da_types = sorted({
        da
        for blk in all_blocks
        for da in blk.get(seq_key, [])
    })

    if not all_da_types:
        print(f"    [{speaker_label}] No DA types found — skipping.")
        return []

    # Organise blocks: code → therapist_id → list of sequences
    # Also: code → therapist_id → session (filename) → list of sequences
    by_code_therapist: dict[str, dict[str, list[list[str]]]] = defaultdict(
        lambda: defaultdict(list)
    )
    by_code_session: dict[str, dict[str, dict[str, list[list[str]]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )

    for blk in all_blocks:
        seq = blk.get(seq_key, [])
        if not seq:
            continue
        for code in blk["codes"]:
            by_code_therapist[code][blk["therapist_id"]].append(seq)
            by_code_session[code][blk["therapist_id"]][blk["filename"]].append(seq)

    all_codes    = sorted(by_code_therapist.keys())
    all_therapists = sorted({
        t for code in all_codes for t in by_code_therapist[code]
    })
    summary_rows = []

    for code in all_codes:
        print(f"\n  {'─'*50}")
        print(f"  Code: {code}  [{speaker_label}]")

        code_dir = os.path.join(seq_dir, str(code).replace("/", "-").replace(" ", "_"))
        os.makedirs(code_dir, exist_ok=True)

        therapist_seqs = by_code_therapist[code]
        therapists     = sorted(therapist_seqs.keys())

        if len(therapists) < 2:
            print(f"    Only one therapist has this code — skipping between-therapist analysis.")

        # ── DA distribution table per therapist ───────────────────────────────
        dist_rows = []
        for t in therapists:
            counts = _da_counts(therapist_seqs[t], all_da_types)
            total  = counts.sum()
            row    = {"therapist_id": t, "n_sequences": len(therapist_seqs[t]),
                      "n_das": int(total)}
            for da, cnt in zip(all_da_types, counts):
                row[da] = int(cnt)
            dist_rows.append(row)

        df_dist = pd.DataFrame(dist_rows)
        df_dist.to_csv(os.path.join(code_dir, "da_distributions.csv"), index=False)
        print(f"    Saved: da_distributions.csv")

        # ── between-therapist pairwise chi-square ─────────────────────────────
        between_rows = []
        if len(therapists) >= 2:
            for ta, tb in combinations(therapists, 2):
                counts_a = _da_counts(therapist_seqs[ta], all_da_types)
                counts_b = _da_counts(therapist_seqs[tb], all_da_types)
                result   = _chi_square_pair(counts_a, counts_b,
                                            f"T{ta}", f"T{tb}")
                between_rows.append(result)
                sig = "**" if result["p_value"] < 0.05 else ""
                print(f"    T{ta} vs T{tb}:  "
                      f"chi2={result['chi2']:.3f}  "
                      f"p={result['p_value']:.4f}{sig}  "
                      f"JS={result['js_divergence']:.3f}  "
                      f"(n={result['n_a']}, {result['n_b']})")

                summary_rows.append({
                    "code":          code,
                    "comparison":    "between_therapists",
                    "label_a":       result["label_a"],
                    "label_b":       result["label_b"],
                    "chi2":          result["chi2"],
                    "p_value":       result["p_value"],
                    "js_divergence": result["js_divergence"],
                    "n_a":           result["n_a"],
                    "n_b":           result["n_b"],
                    "note":          result["note"],
                })

            pd.DataFrame(between_rows).to_csv(
                os.path.join(code_dir, "chi_square_between.csv"), index=False
            )
            print(f"    Saved: chi_square_between.csv")

        # ── within-therapist pairwise chi-square (across sessions) ────────────
        within_rows = []
        for t in therapists:
            sessions = sorted(by_code_session[code][t].keys())
            if len(sessions) < 2:
                continue

            for sa, sb in combinations(sessions, 2):
                seqs_a   = by_code_session[code][t][sa]
                seqs_b   = by_code_session[code][t][sb]
                counts_a = _da_counts(seqs_a, all_da_types)
                counts_b = _da_counts(seqs_b, all_da_types)
                result   = _chi_square_pair(counts_a, counts_b, sa, sb)
                result["therapist_id"] = t
                within_rows.append(result)

                sig = "**" if result["p_value"] < 0.05 else ""
                print(f"    T{t} within:  {Path(sa).stem} vs {Path(sb).stem}  "
                      f"chi2={result['chi2']:.3f}  "
                      f"p={result['p_value']:.4f}{sig}  "
                      f"JS={result['js_divergence']:.3f}")

                summary_rows.append({
                    "code":          code,
                    "comparison":    f"within_T{t}",
                    "label_a":       sa,
                    "label_b":       sb,
                    "chi2":          result["chi2"],
                    "p_value":       result["p_value"],
                    "js_divergence": result["js_divergence"],
                    "n_a":           result["n_a"],
                    "n_b":           result["n_b"],
                    "note":          result["note"],
                })

        if within_rows:
            df_within = pd.DataFrame(within_rows)
            df_within.to_csv(
                os.path.join(code_dir, "chi_square_within.csv"), index=False
            )
            print(f"    Saved: chi_square_within.csv")

        # ── JS divergence matrix between therapists ───────────────────────────
        if len(therapists) >= 2:
            js_mat = np.zeros((len(therapists), len(therapists)))
            for i, ta in enumerate(therapists):
                for j, tb in enumerate(therapists):
                    if i != j:
                        ca = _da_counts(therapist_seqs[ta], all_da_types)
                        cb = _da_counts(therapist_seqs[tb], all_da_types)
                        js_mat[i, j] = _js_divergence(ca, cb)
            df_js = pd.DataFrame(
                js_mat,
                index=[f"T{t}" for t in therapists],
                columns=[f"T{t}" for t in therapists],
            ).round(4)
            df_js.to_csv(os.path.join(code_dir, "js_divergence.csv"))
            print(f"    Saved: js_divergence.csv")

        # ── DA distribution bar chart ─────────────────────────────────────────
        _plot_da_distribution(
            therapist_seqs=therapist_seqs,
            all_da_types=all_da_types,
            code=code,
            granularity=granularity,
            outpath=os.path.join(code_dir, "da_distribution_plot.png"),
        )
        print(f"    Saved: da_distribution_plot.png")

    # ── per-therapist code × code JS heatmap (one per therapist) ─────────────
    _plot_js_heatmap_per_therapist_code(
        by_code_therapist=by_code_therapist,
        all_da_types=all_da_types,
        therapists=all_therapists,
        speaker_label=speaker_label,
        code_dir_fn=lambda code: os.path.join(
            seq_dir, str(code).replace("/", "-").replace(" ", "_")
        ),
        outdir=outdir,
    )

    # ── per-code therapist × therapist JS heatmap (one per code) ─────────────
    _plot_js_heatmap_per_code(
        by_code_therapist=by_code_therapist,
        all_da_types=all_da_types,
        speaker_label=speaker_label,
        outdir=outdir,
    )

    # ── combined figure: all codes in one canvas ──────────────────────────────
    _plot_combined_therapist_heatmaps(
        by_code_therapist=by_code_therapist,
        all_da_types=all_da_types,
        speaker_label=speaker_label,
        outdir=outdir,
    )

    return summary_rows


# ── DA distribution bar chart ─────────────────────────────────────────────────

def _plot_da_distribution(
    therapist_seqs: dict[str, list[list[str]]],
    all_da_types:   list[str],
    code:           str,
    granularity:    str,
    outpath:        str,
):
    """
    Grouped bar chart showing normalised DA-type frequencies per therapist
    for a given code.
    """
    therapists = sorted(therapist_seqs.keys())
    n_t        = len(therapists)
    n_da       = len(all_da_types)

    if n_da == 0 or n_t == 0:
        return

    # Build normalised count matrix: (n_therapists, n_da_types)
    mat = np.zeros((n_t, n_da))
    for i, t in enumerate(therapists):
        counts  = _da_counts(therapist_seqs[t], all_da_types)
        total   = counts.sum()
        mat[i]  = counts / max(total, 1)

    # Only plot DA types that appear at least once for any therapist
    nonzero_cols = mat.sum(axis=0) > 0
    mat_plot     = mat[:, nonzero_cols]
    da_plot      = [da for da, nz in zip(all_da_types, nonzero_cols) if nz]
    n_da_plot    = len(da_plot)

    if n_da_plot == 0:
        return

    bar_width = 0.8 / max(n_t, 1)
    x         = np.arange(n_da_plot)
    colors    = plt.cm.Set2(np.linspace(0, 1, n_t))

    fig_w = max(8, n_da_plot * 0.6 * n_t + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 4))

    for i, (t, color) in enumerate(zip(therapists, colors)):
        offset = (i - n_t / 2 + 0.5) * bar_width
        bars   = ax.bar(
            x + offset, mat_plot[i], bar_width,
            label=f"T{t}", color=color, alpha=0.8, edgecolor="white",
        )

    # X-axis labels: use abbreviations if groups granularity
    if granularity == "groups":
        x_labels = [DA_GROUP_ABBREV.get(da, da) for da in da_plot]
    else:
        x_labels = da_plot

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Proportion of DAs")
    ax.set_title(f"DA distribution for code: {code}\n(normalised per therapist)")
    ax.legend(title="Therapist", fontsize=8, loc="upper right")
    ax.grid(True, axis="y", color="lightgrey", linewidth=0.5)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close()


# ── per-code therapist × therapist JS heatmap ────────────────────────────────

def _plot_js_heatmap_per_code(
    by_code_therapist: dict[str, dict[str, list[list[str]]]],
    all_da_types:      list[str],
    speaker_label:     str,
    outdir:            str,
) -> None:
    """
    For each code, produce a therapist × therapist JS divergence heatmap
    showing how similar each pair of therapists are in their DA distributions
    for that code.

    Axes: therapist ID × therapist ID (only therapists who used this code).
    Saved to: sequences/{speaker_label}/{code}/js_heatmap_therapists.png

    Normalised to the actual off-diagonal min/max so colour contrast is
    maximised within each code's matrix.
    """
    all_codes = sorted(by_code_therapist.keys())

    for code in all_codes:
        therapist_seqs = by_code_therapist[code]
        therapists     = sorted(therapist_seqs.keys())

        if len(therapists) < 2:
            continue

        n      = len(therapists)
        js_mat = np.zeros((n, n))
        for i, ta in enumerate(therapists):
            for j, tb in enumerate(therapists):
                if i != j:
                    ca = _da_counts(therapist_seqs[ta], all_da_types)
                    cb = _da_counts(therapist_seqs[tb], all_da_types)
                    js_mat[i, j] = _js_divergence(ca, cb)

        # Normalise to actual off-diagonal range
        off_diag = js_mat[~np.eye(n, dtype=bool)]
        vmin     = float(off_diag.min()) if len(off_diag) else 0.0
        vmax     = float(off_diag.max()) if len(off_diag) else 1.0
        if vmax - vmin < 1e-6:
            vmin, vmax = 0.0, 1.0

        t_labels = [f"T{t}" for t in therapists]

        fig, ax = plt.subplots(figsize=(max(4, n * 0.9 + 1.5), max(3.5, n * 0.9)))
        im = ax.imshow(js_mat, vmin=vmin, vmax=vmax, cmap="RdYlGn_r", aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(t_labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n))
        ax.set_yticklabels(t_labels, fontsize=9)
        ax.set_title(
            f"Code: {code} — therapist × therapist JS divergence\n"
            f"({speaker_label}; normalised to [{vmin:.2f}, {vmax:.2f}])",
            fontsize=9,
        )
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label(
            "JS divergence (normalised)"
        )
        mid = (vmin + vmax) / 2
        for i in range(n):
            for j in range(n):
                if i != j:
                    ax.text(j, i, f"{js_mat[i,j]:.2f}", ha="center", va="center",
                            fontsize=8,
                            color="black")
                else:
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=8, color="grey")

        plt.tight_layout()
        safe_code = str(code).replace("/", "-").replace(" ", "_")
        code_dir  = os.path.join(outdir, "sequences", speaker_label, safe_code)
        os.makedirs(code_dir, exist_ok=True)
        fname = os.path.join(code_dir, "js_heatmap_therapists.png")
        plt.savefig(fname, bbox_inches="tight", dpi=150)
        print(f"    Saved: {code}/js_heatmap_therapists.png  ({speaker_label})")
        plt.close()


# ── combined all-codes therapist × therapist JS heatmap ──────────────────────

def _plot_combined_therapist_heatmaps(
    by_code_therapist: dict[str, dict[str, list[list[str]]]],
    all_da_types:      list[str],
    speaker_label:     str,
    outdir:            str,
) -> None:
    """
    One combined figure per speaker slice with one subplot per code.
    Each subplot is the therapist x therapist JS divergence heatmap for
    that code (same data as js_heatmap_therapists.png but all on one canvas).

    All therapists are always shown on every subplot — those with no data for
    a given code have their rows/cols filled with NaN and rendered grey so
    the axes are consistent across subplots, making visual comparison easier.

    Layout: up to 5 columns, rows as needed.
    Saved to: sequences/{speaker_label}/combined_therapist_heatmaps.png
    """
    all_codes = sorted(by_code_therapist.keys())

    # All therapists seen across any code
    all_therapists = sorted({
        t
        for code in all_codes
        for t in by_code_therapist[code]
    })

    if len(all_therapists) < 2:
        print(f"    [{speaker_label}] Fewer than 2 therapists — skipping combined plot.")
        return

    plottable_codes = [c for c in all_codes if len(by_code_therapist[c]) > 0]
    if not plottable_codes:
        return

    n_codes  = len(plottable_codes)
    n_cols   = min(5, n_codes)
    n_rows   = (n_codes + n_cols - 1) // n_cols
    n_t      = len(all_therapists)
    t_labels = [f"T{t}" for t in all_therapists]

    # ── pre-compute all JS matrices so we can find a global colour scale ─────
    all_js_mats: dict[str, np.ndarray] = {}
    for code in plottable_codes:
        therapist_seqs = by_code_therapist[code]
        js_mat = np.full((n_t, n_t), np.nan)
        for i, ta in enumerate(all_therapists):
            for j, tb in enumerate(all_therapists):
                if i == j:
                    continue
                seqs_a = therapist_seqs.get(ta, [])
                seqs_b = therapist_seqs.get(tb, [])
                if seqs_a and seqs_b:
                    ca = _da_counts(seqs_a, all_da_types)
                    cb = _da_counts(seqs_b, all_da_types)
                    js_mat[i, j] = _js_divergence(ca, cb)
        all_js_mats[code] = js_mat

    # Global scale: min/max across all finite off-diagonal values in all codes
    all_finite = np.concatenate([
        mat[~np.eye(n_t, dtype=bool)][np.isfinite(mat[~np.eye(n_t, dtype=bool)])]
        for mat in all_js_mats.values()
    ])
    global_vmin = float(all_finite.min()) if len(all_finite) else 0.0
    global_vmax = float(all_finite.max()) if len(all_finite) else 1.0
    if global_vmax - global_vmin < 1e-6:
        global_vmin, global_vmax = 0.0, 1.0

    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad(color="#cccccc")

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.5, n_rows * 3.2),
        squeeze=False,
    )

    for ax_idx, code in enumerate(plottable_codes):
        row = ax_idx // n_cols
        col = ax_idx  % n_cols
        ax  = axes[row][col]

        js_mat = all_js_mats[code]
        masked = np.ma.masked_invalid(js_mat)
        im = ax.imshow(masked, vmin=global_vmin, vmax=global_vmax,
                       cmap=cmap, aspect="auto")
        ax.set_xticks(range(n_t))
        ax.set_xticklabels(t_labels, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(n_t))
        ax.set_yticklabels(t_labels, fontsize=7)
        ax.set_title(f"{code}", fontsize=9, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(n_t):
            for j in range(n_t):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=6, color="grey")
                elif np.isfinite(js_mat[i, j]):
                    ax.text(j, i, f"{js_mat[i,j]:.2f}",
                            ha="center", va="center", fontsize=6, color="black")
                else:
                    ax.text(j, i, "N/A", ha="center", va="center",
                            fontsize=5, color="#888888")

    # Hide any unused subplots
    for ax_idx in range(n_codes, n_rows * n_cols):
        axes[ax_idx // n_cols][ax_idx % n_cols].set_visible(False)

    fig.suptitle(
        f"Therapist x therapist JS divergence per code ({speaker_label})",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()

    seq_spkr_dir = os.path.join(outdir, "sequences", speaker_label)
    os.makedirs(seq_spkr_dir, exist_ok=True)
    fname = os.path.join(seq_spkr_dir, "combined_therapist_heatmaps.png")
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    print(f"    Saved: combined_therapist_heatmaps.png  ({speaker_label})")
    plt.close()



def _plot_js_heatmap_per_therapist_code(
    by_code_therapist: dict[str, dict[str, list[list[str]]]],
    all_da_types:      list[str],
    therapists:        list[str],
    speaker_label:     str,
    code_dir_fn,
    outdir:            str,
) -> None:
    """
    For each therapist, produce a code x code JS divergence heatmap.
    All codes are always shown — codes the therapist has no data for are
    filled with NaN and rendered grey so axes are consistent across therapists.

    A single global colour scale is used across all therapist heatmaps so
    they are directly comparable.

    Individual PNGs saved to: sequences/{speaker_label}/js_heatmap_T{n}.png
    Combined PNG saved to:    sequences/{speaker_label}/combined_therapist_code_heatmaps.png
    """
    all_codes = sorted(by_code_therapist.keys())
    n_codes   = len(all_codes)

    if n_codes < 2 or len(therapists) < 1:
        return

    # ── pre-compute all matrices (all codes x all codes per therapist) ────────
    all_mats: dict[str, np.ndarray] = {}
    for t in therapists:
        js_mat = np.full((n_codes, n_codes), np.nan)
        for i, ca in enumerate(all_codes):
            for j, cb in enumerate(all_codes):
                if i == j:
                    continue
                seqs_a = by_code_therapist[ca].get(t, [])
                seqs_b = by_code_therapist[cb].get(t, [])
                if seqs_a and seqs_b:
                    js_mat[i, j] = _js_divergence(
                        _da_counts(seqs_a, all_da_types),
                        _da_counts(seqs_b, all_da_types),
                    )
        all_mats[t] = js_mat

    # Global colour scale across all finite off-diagonal values
    off_mask = ~np.eye(n_codes, dtype=bool)
    all_finite = np.concatenate([
        mat[off_mask][np.isfinite(mat[off_mask])]
        for mat in all_mats.values()
    ])
    global_vmin = float(all_finite.min()) if len(all_finite) else 0.0
    global_vmax = float(all_finite.max()) if len(all_finite) else 1.0
    if global_vmax - global_vmin < 1e-6:
        global_vmin, global_vmax = 0.0, 1.0

    cmap = plt.cm.RdYlGn_r.copy()
    cmap.set_bad(color="#cccccc")

    seq_spkr_dir = os.path.join(outdir, "sequences", speaker_label)
    os.makedirs(seq_spkr_dir, exist_ok=True)

    # ── individual plots ──────────────────────────────────────────────────────
    for t in therapists:
        js_mat = all_mats[t]
        masked = np.ma.masked_invalid(js_mat)

        fig, ax = plt.subplots(
            figsize=(max(4, n_codes * 0.9 + 1.5), max(3.5, n_codes * 0.9))
        )
        im = ax.imshow(masked, vmin=global_vmin, vmax=global_vmax,
                       cmap=cmap, aspect="auto")
        ax.set_xticks(range(n_codes))
        ax.set_xticklabels(all_codes, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n_codes))
        ax.set_yticklabels(all_codes, fontsize=9)
        ax.set_title(
            f"T{t} — code x code JS divergence\n"
            f"({speaker_label}; scale [{global_vmin:.2f}, {global_vmax:.2f}])",
            fontsize=9,
        )
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label(
            "JS divergence"
        )
        for i in range(n_codes):
            for j in range(n_codes):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=8, color="grey")
                elif np.isfinite(js_mat[i, j]):
                    ax.text(j, i, f"{js_mat[i,j]:.2f}", ha="center", va="center",
                            fontsize=8, color="black")
                else:
                    ax.text(j, i, "N/A", ha="center", va="center",
                            fontsize=6, color="#888888")

        plt.tight_layout()
        fname = os.path.join(seq_spkr_dir, f"js_heatmap_T{t}.png")
        plt.savefig(fname, bbox_inches="tight", dpi=150)
        print(f"    Saved: js_heatmap_T{t}.png  ({speaker_label})")
        plt.close()

    # ── combined plot: one subplot per therapist ──────────────────────────────
    n_t    = len(therapists)
    n_cols = min(5, n_t)
    n_rows = (n_t + n_cols - 1) // n_cols

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(n_cols * 3.5, n_rows * 3.2),
        squeeze=False,
    )

    for ax_idx, t in enumerate(therapists):
        row = ax_idx // n_cols
        col = ax_idx  % n_cols
        ax  = axes[row][col]

        masked = np.ma.masked_invalid(all_mats[t])
        im = ax.imshow(masked, vmin=global_vmin, vmax=global_vmax,
                       cmap=cmap, aspect="auto")
        ax.set_xticks(range(n_codes))
        ax.set_xticklabels(all_codes, fontsize=7, rotation=45, ha="right")
        ax.set_yticks(range(n_codes))
        ax.set_yticklabels(all_codes, fontsize=7)
        ax.set_title(f"T{t}", fontsize=9, fontweight="bold")
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        for i in range(n_codes):
            for j in range(n_codes):
                if i == j:
                    ax.text(j, i, "—", ha="center", va="center",
                            fontsize=6, color="grey")
                elif np.isfinite(all_mats[t][i, j]):
                    ax.text(j, i, f"{all_mats[t][i,j]:.2f}",
                            ha="center", va="center", fontsize=6, color="black")
                else:
                    ax.text(j, i, "N/A", ha="center", va="center",
                            fontsize=5, color="#888888")

    for ax_idx in range(n_t, n_rows * n_cols):
        axes[ax_idx // n_cols][ax_idx % n_cols].set_visible(False)

    fig.suptitle(
        f"Code x code JS divergence per therapist ({speaker_label})",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    fname = os.path.join(seq_spkr_dir, "combined_code_heatmaps.png")
    plt.savefig(fname, bbox_inches="tight", dpi=150)
    print(f"    Saved: combined_code_heatmaps.png  ({speaker_label})")
    plt.close()



def analyse_cross_code(
    all_blocks:    list[dict],
    outdir:        str,
    seq_key:       str = "da_sequence",
    speaker_label: str = "all",
) -> None:
    """
    Compare DA distributions between different codes (pooled across all
    therapists).  Answers: do different codes correspond to genuinely
    different DA patterns, or is the coding inconsistent?
    """
    cross_dir = os.path.join(outdir, "cross_code", speaker_label)
    os.makedirs(cross_dir, exist_ok=True)

    all_da_types = sorted({
        da for blk in all_blocks for da in blk.get(seq_key, [])
    })

    if not all_da_types:
        print(f"    [{speaker_label}] No DA types for cross-code — skipping.")
        return

    # Pool sequences per code across all therapists
    by_code: dict[str, list[list[str]]] = defaultdict(list)
    for blk in all_blocks:
        seq = blk.get(seq_key, [])
        if seq:
            for code in blk["codes"]:
                by_code[code].append(seq)

    codes = sorted(by_code.keys())
    if len(codes) < 2:
        return

    print(f"\n  Cross-code DA distribution comparison ({len(codes)} codes)")

    # Pairwise chi-square between codes
    rows = []
    for ca, cb in combinations(codes, 2):
        counts_a = _da_counts(by_code[ca], all_da_types)
        counts_b = _da_counts(by_code[cb], all_da_types)
        result   = _chi_square_pair(counts_a, counts_b, ca, cb)
        rows.append(result)
        sig = "**" if result["p_value"] < 0.05 else ""
        print(f"    {ca} vs {cb}:  "
              f"chi2={result['chi2']:.3f}  "
              f"p={result['p_value']:.4f}{sig}  "
              f"JS={result['js_divergence']:.3f}")

    df_cross = pd.DataFrame(rows).sort_values("js_divergence", ascending=False)
    df_cross.to_csv(os.path.join(cross_dir, "cross_code_chi_square.csv"), index=False)
    print(f"  Saved: {os.path.join(cross_dir, 'cross_code_chi_square.csv')}")

    # JS divergence heatmap between codes
    n = len(codes)
    js_mat = np.zeros((n, n))
    for i, ca in enumerate(codes):
        for j, cb in enumerate(codes):
            if i != j:
                js_mat[i, j] = _js_divergence(
                    _da_counts(by_code[ca], all_da_types),
                    _da_counts(by_code[cb], all_da_types),
                )

    # Normalise heatmap to actual off-diagonal min/max
    off_diag   = js_mat[~np.eye(n, dtype=bool)]
    vmin_js    = float(off_diag.min()) if len(off_diag) else 0.0
    vmax_js    = float(off_diag.max()) if len(off_diag) else 1.0
    if vmax_js - vmin_js < 1e-6:
        vmin_js, vmax_js = 0.0, 1.0

    fig, ax = plt.subplots(figsize=(max(5, n * 0.8 + 1.5), max(4, n * 0.8)))
    im = ax.imshow(js_mat, vmin=vmin_js, vmax=vmax_js, cmap="RdYlGn_r", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(codes, fontsize=9)
    ax.set_title(
        f"Cross-code JS divergence\n"
        f"(higher = more distinct; normalised to [{vmin_js:.2f}, {vmax_js:.2f}])"
    )
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label(
        "JS divergence (normalised)"
    )
    mid_js = (vmin_js + vmax_js) / 2
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{js_mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=8,
                        color="black")
            else:
                ax.text(j, i, "—", ha="center", va="center",
                        fontsize=8, color="grey")
    plt.tight_layout()
    plt.savefig(os.path.join(cross_dir, "cross_code_js_heatmap.png"),
                bbox_inches="tight", dpi=150)
    print(f"  Saved: {os.path.join(cross_dir, 'cross_code_js_heatmap.png')}")
    plt.close()


# ── Krippendorff's alpha ──────────────────────────────────────────────────────

def krippendorff_alpha_rates(
    records: list[dict],
    outdir:  str,
) -> None:
    """
    Compute Krippendorff's alpha on per-session important rates per therapist.
    Each therapist is treated as a 'rater'; their sessions are the 'units'.
    Since therapists rate different sessions, this is computed per-therapist
    (internal consistency) using interval-level alpha on their own session rates.

    Interpretation: alpha > 0.8 = good, 0.67–0.8 = tentative, < 0.67 = unreliable.
    """
    rates_dir = os.path.join(outdir, "rates")

    # Group session rates per therapist
    therapist_rates: dict[str, list[float]] = defaultdict(list)
    for rec in records:
        df         = rec["df"]
        target_col = rec["target_col"]
        rate       = df[target_col].sum() / max(len(df), 1)
        therapist_rates[rec["therapist_id"]].append(float(rate))

    rows = []
    for t, rates in sorted(therapist_rates.items()):
        if len(rates) < 2:
            rows.append({"therapist_id": t, "n_sessions": len(rates),
                         "krippendorff_alpha": np.nan,
                         "note": "need_>=2_sessions"})
            continue

        # Interval-level Krippendorff's alpha for one rater across sessions
        # Here we treat each session as a unit and compute agreement
        # across repeated measurement of the same rater (self-consistency).
        # For a single rater, alpha reduces to 1 - (observed disagreement /
        # expected disagreement under random assignment).
        rates_arr = np.array(rates)
        n         = len(rates_arr)

        # Observed disagreement: mean squared difference between all pairs
        obs_pairs = [
            (rates_arr[i] - rates_arr[j]) ** 2
            for i in range(n) for j in range(i + 1, n)
        ]
        D_o = float(np.mean(obs_pairs)) if obs_pairs else 0.0

        # Expected disagreement: based on overall variance
        D_e = float(np.var(rates_arr)) * (n / (n - 1)) if n > 1 else 0.0

        alpha = 1.0 - (D_o / D_e) if D_e > 1e-10 else 1.0

        rows.append({
            "therapist_id":       t,
            "n_sessions":         n,
            "mean_rate":          round(float(rates_arr.mean()), 4),
            "std_rate":           round(float(rates_arr.std()),  4),
            "krippendorff_alpha": round(alpha, 4),
            "note": (
                "good"      if alpha > 0.8  else
                "tentative" if alpha > 0.67 else
                "unreliable"
            ),
        })

    df_alpha = pd.DataFrame(rows)
    df_alpha.to_csv(os.path.join(rates_dir, "krippendorff_alpha.csv"), index=False)
    print(f"\n  Saved: {os.path.join(rates_dir, 'krippendorff_alpha.csv')}")
    print("\n  Krippendorff alpha (within-therapist rate consistency):")
    print(df_alpha.to_string(index=False))



# ── performance ceiling analysis ─────────────────────────────────────────────

def _extract_ngrams(seq: list[str], n: int) -> list[tuple]:
    """Extract all n-grams from a DA sequence."""
    return [tuple(seq[i:i+n]) for i in range(len(seq) - n + 1)]


def _ngram_js(seqs_a: list[list[str]],
              seqs_b: list[list[str]],
              n: int) -> float:
    """JS divergence between n-gram frequency distributions of two groups."""
    all_ngrams = sorted({
        ng
        for seqs in (seqs_a, seqs_b)
        for seq in seqs
        for ng in _extract_ngrams(seq, n)
    })
    if not all_ngrams:
        return np.nan

    def _vec(seqs):
        counts = defaultdict(int)
        for seq in seqs:
            for ng in _extract_ngrams(seq, n):
                counts[ng] += 1
        return np.array([counts.get(ng, 0) for ng in all_ngrams], dtype=float)

    va = _vec(seqs_a) + 1e-10
    vb = _vec(seqs_b) + 1e-10
    va /= va.sum();  vb /= vb.sum()
    m   = 0.5 * (va + vb)
    kl  = lambda p, q: float(np.sum(p * np.log(p / q)))
    return 0.5 * kl(va, m) + 0.5 * kl(vb, m)


def _js_to_kappa(js: float) -> float:
    """
    Convert JS divergence to an approximate kappa analogue.
    JS=0 (identical) -> kappa=1.0  (perfect agreement)
    JS=1 (maximally different) -> kappa=0.0  (chance)
    Linear mapping: kappa ≈ 1 - JS
    """
    return max(0.0, 1.0 - js)


def _kappa_to_f1_ceiling(kappa: float) -> float:
    """
    Approximate F1 ceiling given a kappa-based agreement level.
    Derived from the relationship: F1_max ≈ (1 + kappa) / 2
    This gives F1=1.0 at kappa=1 and F1=0.5 at kappa=0 (chance).
    """
    return (1.0 + max(0.0, kappa)) / 2.0


def analyse_performance_ceiling(
    all_blocks:  list[dict],
    outdir:      str,
    ngram_ns:    list[int] = (1, 2, 3),
) -> None:
    """
    Estimate the maximum achievable classifier performance given label
    inconsistency across therapists.

    For each code and speaker slice:
      - Pairwise JS divergence between therapists' DA frequency distributions
        for that code (already done elsewhere — recomputed here for the ceiling)
      - Pairwise JS divergence on bigram/trigram distributions (supplemental
        sequence consistency measure)
      - JS -> kappa analogue -> F1 ceiling conversion
      - Aggregate (mean across all therapist pairs) and per-pair breakdown

    Interpretation guide
    --------------------
    kappa > 0.8  -> F1 ceiling ~0.90+  labels are consistent, model can do well
    kappa 0.6-0.8 -> F1 ceiling ~0.80  moderate consistency, moderate ceiling
    kappa 0.4-0.6 -> F1 ceiling ~0.70  noisy labels, ceiling is real constraint
    kappa < 0.4  -> F1 ceiling <0.70   high noise, even a perfect model struggles

    Outputs
    -------
    {outdir}/ceiling/performance_ceiling.csv   — full per-code per-pair table
    {outdir}/ceiling/performance_ceiling.txt   — human-readable summary
    """
    ceiling_dir = os.path.join(outdir, "ceiling")
    os.makedirs(ceiling_dir, exist_ok=True)

    speaker_slices = [
        ("da_sequence",           "all"),
        ("da_sequence_therapist", "therapist_spkr"),
        ("da_sequence_patient",   "patient_spkr"),
    ]

    all_rows  = []
    txt_lines = [
        "PERFORMANCE CEILING ANALYSIS",
        "=" * 60,
        "",
        "Methodology:",
        "  - JS divergence between therapist DA distributions for each code",
        "    is used as a proxy for label noise.",
        "  - kappa_analogue = 1 - JS  (JS=0 -> perfect agreement)",
        "  - F1_ceiling = (1 + kappa) / 2",
        "  - N-gram JS divergence is a supplemental sequence-order measure.",
        "",
        "Interpretation:",
        "  kappa > 0.8  -> F1 ceiling ~0.90+  (consistent labels)",
        "  kappa 0.6-0.8 -> F1 ceiling ~0.80  (moderate consistency)",
        "  kappa 0.4-0.6 -> F1 ceiling ~0.70  (noisy labels)",
        "  kappa < 0.4  -> F1 ceiling <0.70   (high noise)",
        "",
    ]

    def h(t, c="-"):
        txt_lines.append(f"\n{c*60}\n{t}\n{c*60}")

    for seq_key, speaker_label in speaker_slices:
        h(f"SPEAKER SLICE: {speaker_label.upper()}", "=")

        # Collect sequences per (therapist, code)
        by_tc: dict[tuple, list[list[str]]] = defaultdict(list)
        for blk in all_blocks:
            seq = blk.get(seq_key, [])
            if not seq:
                continue
            for code in blk["codes"]:
                by_tc[(blk["therapist_id"], code)].append(seq)

        if not by_tc:
            txt_lines.append("  No data.")
            continue

        all_da      = sorted({da for seqs in by_tc.values()
                               for seq in seqs for da in seq})
        all_codes      = sorted({tc[1] for tc in by_tc})
        all_therapists = sorted({tc[0] for tc in by_tc})

        for code in all_codes:
            therapists_with = sorted(
                t for t in all_therapists if (t, code) in by_tc
            )
            if len(therapists_with) < 2:
                continue

            h(f"Code: {code}  [{speaker_label}]  "
              f"(therapists: {', '.join('T'+t for t in therapists_with)})")

            pair_rows = []

            for ta, tb in combinations(therapists_with, 2):
                seqs_a = by_tc[(ta, code)]
                seqs_b = by_tc[(tb, code)]

                # Frequency JS
                va      = _da_counts(seqs_a, all_da) + 1e-10
                vb      = _da_counts(seqs_b, all_da) + 1e-10
                va_n    = va / va.sum();  vb_n = vb / vb.sum()
                m       = 0.5 * (va_n + vb_n)
                kl      = lambda p, q: float(np.sum(p * np.log(p / q)))
                js_freq = 0.5 * kl(va_n, m) + 0.5 * kl(vb_n, m)

                # N-gram JS for each n
                ngram_js_vals = {
                    n: _ngram_js(seqs_a, seqs_b, n)
                    for n in ngram_ns
                }

                kappa    = _js_to_kappa(js_freq)
                f1_ceil  = _kappa_to_f1_ceiling(kappa)

                row = {
                    "speaker":      speaker_label,
                    "code":         code,
                    "T_a":          ta,
                    "T_b":          tb,
                    "n_seqs_a":     len(seqs_a),
                    "n_seqs_b":     len(seqs_b),
                    "js_freq":      round(js_freq, 4),
                    "kappa_analogue": round(kappa, 4),
                    "f1_ceiling":   round(f1_ceil, 4),
                }
                for n, js_ng in ngram_js_vals.items():
                    row[f"js_{n}gram"] = round(js_ng, 4) if np.isfinite(js_ng) else None
                pair_rows.append(row)
                all_rows.append(row)

                ngram_str = "  ".join(
                    f"{n}-gram JS={js_ng:.3f}" if np.isfinite(js_ng) else f"{n}-gram JS=N/A"
                    for n, js_ng in ngram_js_vals.items()
                )
                txt_lines.append(
                    f"  T{ta} vs T{tb}:  "
                    f"freq JS={js_freq:.3f}  "
                    f"kappa={kappa:.3f}  "
                    f"F1_ceil={f1_ceil:.3f}  |  {ngram_str}  "
                    f"(n={len(seqs_a)}, {len(seqs_b)})"
                )

            if not pair_rows:
                continue

            # Aggregate across all pairs for this code
            js_vals    = [r["js_freq"] for r in pair_rows]
            kappa_vals = [r["kappa_analogue"] for r in pair_rows]
            f1_vals    = [r["f1_ceiling"] for r in pair_rows]

            mean_js    = float(np.mean(js_vals))
            mean_kappa = float(np.mean(kappa_vals))
            mean_f1    = float(np.mean(f1_vals))
            min_f1     = float(np.min(f1_vals))
            max_f1     = float(np.max(f1_vals))

            if mean_kappa > 0.8:
                interp = "CONSISTENT — labels are reliable, model ceiling is high"
            elif mean_kappa > 0.6:
                interp = "MODERATE — some inconsistency, ceiling is real but not severe"
            elif mean_kappa > 0.4:
                interp = "NOISY — significant inconsistency, ceiling will limit performance"
            else:
                interp = "HIGH NOISE — labels are inconsistent, ceiling is a hard constraint"

            txt_lines.append(
                f"\n  AGGREGATE for {code} [{speaker_label}]:\n"
                f"    mean freq JS={mean_js:.3f}  "
                f"mean kappa={mean_kappa:.3f}  "
                f"mean F1_ceil={mean_f1:.3f}  "
                f"[min={min_f1:.3f}, max={max_f1:.3f}]\n"
                f"    Interpretation: {interp}"
            )

            # N-gram aggregates
            for n in ngram_ns:
                col  = f"js_{n}gram"
                vals = [r[col] for r in pair_rows
                        if r.get(col) is not None and np.isfinite(r[col])]
                if vals:
                    txt_lines.append(
                        f"    {n}-gram JS (supplemental):  "
                        f"mean={np.mean(vals):.3f}  "
                        f"min={np.min(vals):.3f}  max={np.max(vals):.3f}"
                    )

    # Overall aggregate across all codes and speaker slices
    if all_rows:
        df_all = pd.DataFrame(all_rows)

        h("OVERALL AGGREGATE ACROSS ALL CODES", "=")

        for speaker_label in ["all", "therapist_spkr", "patient_spkr"]:
            sub = df_all[df_all["speaker"] == speaker_label]
            if sub.empty:
                continue
            mean_f1    = sub["f1_ceiling"].mean()
            mean_kappa = sub["kappa_analogue"].mean()
            mean_js    = sub["js_freq"].mean()
            txt_lines.append(
                f"\n  [{speaker_label}]  "
                f"mean kappa={mean_kappa:.3f}  "
                f"mean F1_ceiling={mean_f1:.3f}  "
                f"mean JS={mean_js:.3f}"
            )
            # Per-code summary line
            for code, grp in sub.groupby("code"):
                txt_lines.append(
                    f"    {code}:  "
                    f"kappa={grp['kappa_analogue'].mean():.3f}  "
                    f"F1_ceil={grp['f1_ceiling'].mean():.3f}"
                )

        df_all.to_csv(
            os.path.join(ceiling_dir, "performance_ceiling.csv"), index=False
        )
        print(f"  Saved: ceiling/performance_ceiling.csv")

    txt_path = os.path.join(ceiling_dir, "performance_ceiling.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt_lines))
    print(f"  Saved: ceiling/performance_ceiling.txt")



# ── non-important window extraction ──────────────────────────────────────────

def extract_nonimportant_windows(
    records:     list[dict],
    window_size: int | None = None,
) -> dict[str, list[list[str]]]:
    """
    Extract DA sequences from non-important rows in each transcript,
    keyed by therapist_id.

    window_size=None (default): each contiguous non-important run is kept
    as a single sequence (no chunking).  Since JS divergence operates on
    normalised distributions, length differences don't affect the comparison.

    window_size=N: chunk runs into non-overlapping windows of N DAs.
    Partial trailing windows are discarded.  Use this only if you want
    length-matched comparisons.

    Returns: {therapist_id: [[da, da, ...], ...]}
    """
    by_therapist: dict[str, list[list[str]]] = defaultdict(list)

    for rec in records:
        df          = rec["df"]
        target_col  = rec["target_col"]
        granularity = rec["granularity"]
        t_id        = rec["therapist_id"]

        importance = df[target_col].values
        da_labels  = [
            get_label(row[DA_COLUMN], row["da_group"], granularity)
            for _, row in df.iterrows()
        ]
        speakers = [
            _normalise_speaker(str(row.get("speaker", "patient")))
            for _, row in df.iterrows()
        ]
        n = len(df)

        # Collect contiguous non-important runs
        run: list[tuple] = []   # (da_label, speaker)
        def _flush_run(run):
            if not run:
                return
            if window_size is None:
                by_therapist[t_id].append([da for da, _ in run])
            else:
                for start in range(0, len(run) - window_size + 1, window_size):
                    chunk = run[start:start + window_size]
                    by_therapist[t_id].append([da for da, _ in chunk])

        for i in range(n):
            if importance[i] == 0:
                run.append((da_labels[i], speakers[i]))
            else:
                _flush_run(run)
                run = []
        _flush_run(run)

    return dict(by_therapist)


def extract_nonimportant_windows_by_speaker(
    records:     list[dict],
    window_size: int | None = None,
) -> dict[str, dict[str, list[list[str]]]]:
    """
    Same as extract_nonimportant_windows but split by speaker within each
    sequence.  Returns {therapist_id: {"therapist_spkr": [...], "patient_spkr": [...]}}

    window_size=None (default): each contiguous run kept as one sequence.
    window_size=N: chunk into fixed-size windows.
    """
    by_therapist: dict[str, dict[str, list[list[str]]]] = defaultdict(
        lambda: {"therapist_spkr": [], "patient_spkr": []}
    )

    for rec in records:
        df          = rec["df"]
        target_col  = rec["target_col"]
        granularity = rec["granularity"]
        t_id        = rec["therapist_id"]

        importance = df[target_col].values
        da_labels  = [
            get_label(row[DA_COLUMN], row["da_group"], granularity)
            for _, row in df.iterrows()
        ]
        speakers = [
            _normalise_speaker(str(row.get("speaker", "patient")))
            for _, row in df.iterrows()
        ]
        n = len(df)

        run: list[tuple] = []

        def _flush_run_spkr(run):
            if not run:
                return
            chunks = (
                [run] if window_size is None
                else [run[s:s + window_size]
                      for s in range(0, len(run) - window_size + 1, window_size)]
            )
            for chunk in chunks:
                by_therapist[t_id]["therapist_spkr"].append(
                    [da for da, sp in chunk if sp == "therapist"]
                )
                by_therapist[t_id]["patient_spkr"].append(
                    [da for da, sp in chunk if sp == "patient"]
                )

        for i in range(n):
            if importance[i] == 0:
                run.append((da_labels[i], speakers[i]))
            else:
                _flush_run_spkr(run)
                run = []
        _flush_run_spkr(run)

    return dict(by_therapist)


# ── extended ceiling: important vs non-important + code discrimination ────────

def analyse_extended_ceiling(
    all_blocks:   list[dict],
    records:      list[dict],
    outdir:       str,
    window_size:  int | None = None,
    ngram_ns:     list[int]  = (1, 2, 3),
) -> None:
    """
    Two additional ceiling estimates:

    TASK 1 — Important vs Non-important classification
    ---------------------------------------------------
    How separable are important blocks from non-important windows?
    High JS = distinct distributions = classifier can do well.
    Low JS = similar distributions = hard to distinguish = low ceiling.

    Computed:
      - Per therapist: their important seqs vs their non-important windows
      - Pooled: all important seqs vs all non-important windows

    TASK 2 — Code discrimination (given important, which code?)
    -----------------------------------------------------------
    How separable are the per-code distributions from each other?
    High mean pairwise JS between codes = codes are distinct = high ceiling.
    Low JS = codes overlap in DA patterns = classifier will confuse them.

    Computed:
      - Per therapist: pairwise JS between their codes
      - Pooled across therapists: pairwise JS between pooled code distributions

    Both tasks run for all three speaker slices.

    Outputs
    -------
    {outdir}/ceiling/important_vs_nonimportant.csv
    {outdir}/ceiling/code_discrimination.csv
    {outdir}/ceiling/extended_ceiling.txt
    """
    ceiling_dir = os.path.join(outdir, "ceiling")
    os.makedirs(ceiling_dir, exist_ok=True)

    def _js(a: np.ndarray, b: np.ndarray) -> float:
        a = a + 1e-10;  b = b + 1e-10
        a /= a.sum();   b /= b.sum()
        m = 0.5 * (a + b)
        kl = lambda p, q: float(np.sum(p * np.log(p / q)))
        return 0.5 * kl(a, m) + 0.5 * kl(b, m)

    speaker_slices = [
        ("da_sequence",           "all"),
        ("da_sequence_therapist", "therapist_spkr"),
        ("da_sequence_patient",   "patient_spkr"),
    ]

    # Non-important windows
    nonim_all      = extract_nonimportant_windows(records, window_size)
    nonim_by_spkr  = extract_nonimportant_windows_by_speaker(records, window_size)

    txt  = ["EXTENDED CEILING ANALYSIS", "=" * 60, "",
            f"Non-important windowing: {'%d DAs' % window_size if window_size else 'disabled (full runs)'}",
            "",
            "TASK 1: Important vs Non-important",
            "  JS = how separable important blocks are from background.",
            "  High JS -> classifier can distinguish -> higher F1 ceiling.",
            "",
            "TASK 2: Code discrimination (given important)",
            "  JS = how separable codes are from each other.",
            "  High mean pairwise JS -> codes are distinct -> higher ceiling.",
            ""]

    imp_rows  = []   # task 1
    disc_rows = []   # task 2

    for seq_key, speaker_label in speaker_slices:
        txt.append(f"\n{'='*60}")
        txt.append(f"SPEAKER SLICE: {speaker_label.upper()}")
        txt.append("=" * 60)

        # ── collect important sequences per therapist and pooled ──────────────
        imp_by_t: dict[str, list[list[str]]] = defaultdict(list)
        imp_by_code_t: dict[str, dict[str, list[list[str]]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for blk in all_blocks:
            seq = blk.get(seq_key, [])
            if not seq:
                continue
            t = blk["therapist_id"]
            imp_by_t[t].append(seq)
            for code in blk["codes"]:
                imp_by_code_t[t][code].append(seq)

        # ── non-important sequences for this speaker slice ────────────────────
        if speaker_label == "all":
            nonim_by_t = nonim_all
        else:
            nonim_by_t = {
                t: nonim_by_spkr.get(t, {}).get(speaker_label, [])
                for t in nonim_all
            }

        all_da = sorted({
            da
            for seqs in list(imp_by_t.values()) + list(nonim_by_t.values())
            for seq in seqs for da in seq
        })
        if not all_da:
            txt.append("  No DA types found.")
            continue

        all_therapists = sorted(set(imp_by_t) | set(nonim_by_t))
        all_codes      = sorted({
            c for t_codes in imp_by_code_t.values() for c in t_codes
        })

        # ── TASK 1: important vs non-important ────────────────────────────────
        txt.append("\n--- TASK 1: Important vs Non-important ---\n")

        # Per therapist
        for t in all_therapists:
            imp_seqs  = imp_by_t.get(t, [])
            nim_seqs  = nonim_by_t.get(t, [])
            if not imp_seqs or not nim_seqs:
                continue

            va = _da_counts(imp_seqs,  all_da)
            vb = _da_counts(nim_seqs,  all_da)
            js = _js(va.copy(), vb.copy())
            k  = js   # Task 1: high JS = separable = high kappa (JS used directly)
            f1 = _kappa_to_f1_ceiling(k)

            ngram_str = "  ".join(
                f"{n}-gram JS={_ngram_js(imp_seqs, nim_seqs, n):.3f}"
                if np.isfinite(_ngram_js(imp_seqs, nim_seqs, n)) else f"{n}-gram N/A"
                for n in ngram_ns
            )
            txt.append(
                f"  T{t}:  JS={js:.3f}  kappa={k:.3f}  F1_ceil={f1:.3f}"
                f"  n_imp={len(imp_seqs)}  n_nonim={len(nim_seqs)}"
                f"  |  {ngram_str}"
            )
            imp_rows.append({
                "speaker": speaker_label, "therapist": t, "pooled": False,
                "js": round(js,4), "kappa": round(k,4), "f1_ceiling": round(f1,4),
                "n_important": len(imp_seqs), "n_nonimportant": len(nim_seqs),
                **{f"js_{n}gram": round(_ngram_js(imp_seqs, nim_seqs, n), 4)
                   if np.isfinite(_ngram_js(imp_seqs, nim_seqs, n)) else None
                   for n in ngram_ns},
            })

        # Pooled
        all_imp  = [s for seqs in imp_by_t.values()  for s in seqs]
        all_nonim= [s for seqs in nonim_by_t.values() for s in seqs]
        if all_imp and all_nonim:
            va = _da_counts(all_imp,   all_da)
            vb = _da_counts(all_nonim, all_da)
            js = _js(va.copy(), vb.copy())
            k  = js   # Task 1: high JS = separable = high kappa (JS used directly)
            f1 = _kappa_to_f1_ceiling(k)
            ngram_str = "  ".join(
                f"{n}-gram JS={_ngram_js(all_imp, all_nonim, n):.3f}"
                if np.isfinite(_ngram_js(all_imp, all_nonim, n)) else f"{n}-gram N/A"
                for n in ngram_ns
            )
            txt.append(
                f"\n  POOLED:  JS={js:.3f}  kappa={k:.3f}  F1_ceil={f1:.3f}"
                f"  n_imp={len(all_imp)}  n_nonim={len(all_nonim)}"
                f"  |  {ngram_str}"
            )
            imp_rows.append({
                "speaker": speaker_label, "therapist": "pooled", "pooled": True,
                "js": round(js,4), "kappa": round(k,4), "f1_ceiling": round(f1,4),
                "n_important": len(all_imp), "n_nonimportant": len(all_nonim),
                **{f"js_{n}gram": round(_ngram_js(all_imp, all_nonim, n), 4)
                   if np.isfinite(_ngram_js(all_imp, all_nonim, n)) else None
                   for n in ngram_ns},
            })

        # ── TASK 2: code discrimination ───────────────────────────────────────
        txt.append("\n--- TASK 2: Code Discrimination (given important) ---\n")

        # Per therapist
        for t in sorted(imp_by_code_t.keys()):
            t_codes = sorted(imp_by_code_t[t].keys())
            if len(t_codes) < 2:
                continue
            pairs = list(combinations(t_codes, 2))
            js_vals = []
            pair_strs = []
            for ca, cb in pairs:
                va = _da_counts(imp_by_code_t[t][ca], all_da)
                vb = _da_counts(imp_by_code_t[t][cb], all_da)
                js = _js(va.copy(), vb.copy())
                js_vals.append(js)
                pair_strs.append(f"{ca}vs{cb}={js:.3f}")
                disc_rows.append({
                    "speaker": speaker_label, "therapist": t, "pooled": False,
                    "code_a": ca, "code_b": cb,
                    "js": round(js,4),
                    "kappa": round(_js_to_kappa(js),4),
                    "f1_ceiling": round(_kappa_to_f1_ceiling(_js_to_kappa(js)),4),
                    "n_a": len(imp_by_code_t[t][ca]),
                    "n_b": len(imp_by_code_t[t][cb]),
                })
            mean_js = float(np.mean(js_vals))
            mean_k  = _js_to_kappa(mean_js)
            mean_f1 = _kappa_to_f1_ceiling(mean_k)
            txt.append(
                f"  T{t}:  mean JS={mean_js:.3f}  kappa={mean_k:.3f}"
                f"  F1_ceil={mean_f1:.3f}  "
                f"codes={t_codes}  |  " + "  ".join(pair_strs)
            )

        # Pooled across therapists
        pooled_code: dict[str, list[list[str]]] = defaultdict(list)
        for t_codes in imp_by_code_t.values():
            for code, seqs in t_codes.items():
                pooled_code[code].extend(seqs)

        p_codes = sorted(pooled_code.keys())
        if len(p_codes) >= 2:
            pairs = list(combinations(p_codes, 2))
            js_vals = []
            pair_strs = []
            for ca, cb in pairs:
                va = _da_counts(pooled_code[ca], all_da)
                vb = _da_counts(pooled_code[cb], all_da)
                js = _js(va.copy(), vb.copy())
                js_vals.append(js)
                pair_strs.append(f"{ca}vs{cb}={js:.3f}")
                disc_rows.append({
                    "speaker": speaker_label, "therapist": "pooled", "pooled": True,
                    "code_a": ca, "code_b": cb,
                    "js": round(js,4),
                    "kappa": round(_js_to_kappa(js),4),
                    "f1_ceiling": round(_kappa_to_f1_ceiling(_js_to_kappa(js)),4),
                    "n_a": len(pooled_code[ca]),
                    "n_b": len(pooled_code[cb]),
                })
            mean_js = float(np.mean(js_vals))
            mean_k  = _js_to_kappa(mean_js)
            mean_f1 = _kappa_to_f1_ceiling(mean_k)
            txt.append(
                f"\n  POOLED:  mean JS={mean_js:.3f}  kappa={mean_k:.3f}"
                f"  F1_ceil={mean_f1:.3f}  codes={p_codes}"
                f"\n    pairs: " + "  ".join(pair_strs)
            )

    # ── save outputs ──────────────────────────────────────────────────────────
    if imp_rows:
        pd.DataFrame(imp_rows).to_csv(
            os.path.join(ceiling_dir, "important_vs_nonimportant.csv"), index=False
        )
        print(f"  Saved: ceiling/important_vs_nonimportant.csv")

    if disc_rows:
        pd.DataFrame(disc_rows).to_csv(
            os.path.join(ceiling_dir, "code_discrimination.csv"), index=False
        )
        print(f"  Saved: ceiling/code_discrimination.csv")

    txt_path = os.path.join(ceiling_dir, "extended_ceiling.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt))
    print(f"  Saved: ceiling/extended_ceiling.txt")


# ── narrative labeling analysis ───────────────────────────────────────────────

def write_labeling_narrative(
    all_blocks:     list[dict],
    outdir:         str,
    p_threshold:    float = 0.05,
    js_agree:       float = 0.3,
    js_disagree:    float = 0.5,
) -> None:
    """
    For every (therapist, code) pair and every speaker slice, compute JS
    divergence against ALL other (therapist, code) pairs and write a ranked
    similarity list.  This shows, for each therapist's code, which codes from
    other therapists are most and least similar in terms of DA sequences.

    Also writes a per-code agreement/disagreement summary between therapist
    pairs using chi-square p-value and JS divergence.

    Output: {outdir}/labeling_narrative.txt
             {outdir}/cross_therapist_code_similarity_{speaker_label}.csv  (one per slice)
    """
    speaker_slices = [
        ("da_sequence",           "all"),
        ("da_sequence_therapist", "therapist_spkr"),
        ("da_sequence_patient",   "patient_spkr"),
    ]

    def _js(a: np.ndarray, b: np.ndarray) -> float:
        a = a + 1e-10;  b = b + 1e-10
        a /= a.sum();   b /= b.sum()
        m = 0.5 * (a + b)
        kl = lambda p, q: float(np.sum(p * np.log(p / q)))
        return 0.5 * kl(a, m) + 0.5 * kl(b, m)

    out_lines = [
        "LABELING CONSISTENCY NARRATIVE",
        f"p_threshold={p_threshold}  js_agree<{js_agree}  js_disagree>{js_disagree}",
        "",
    ]

    def section(title, char="="):
        bar = char * 60
        out_lines.append(f"\n{bar}\n{title}\n{bar}")

    for seq_key, speaker_label in speaker_slices:
        section(f"SPEAKER SLICE: {speaker_label.upper()}")

        # ── build DA count vectors per (therapist, code) ──────────────────────
        # Collect sequences
        by_tc: dict[tuple, list[list[str]]] = defaultdict(list)
        for blk in all_blocks:
            seq = blk.get(seq_key, [])
            if not seq:
                continue
            for code in blk["codes"]:
                by_tc[(blk["therapist_id"], code)].append(seq)

        if not by_tc:
            out_lines.append("  No data for this speaker slice.")
            continue

        # Unified DA vocabulary
        all_da = sorted({da for seqs in by_tc.values() for seq in seqs for da in seq})
        if not all_da:
            out_lines.append("  No DA types found.")
            continue

        # Count vectors
        tc_keys = sorted(by_tc.keys())   # list of (therapist_id, code)
        vecs = {
            tc: _da_counts(by_tc[tc], all_da)
            for tc in tc_keys
        }

        all_therapists = sorted({tc[0] for tc in tc_keys})
        all_codes      = sorted({tc[1] for tc in tc_keys})

        # ── SECTION 1: per-code between-therapist agreement ───────────────────
        section(f"[{speaker_label}] 1. PER-CODE AGREEMENT BETWEEN THERAPIST PAIRS", "-")

        for code in all_codes:
            therapists_with_code = sorted(
                t for t in all_therapists if (t, code) in vecs
            )
            out_lines.append(f"\n  Code: {code}  "
                             f"(therapists with data: {', '.join('T'+t for t in therapists_with_code)})")

            if len(therapists_with_code) < 2:
                out_lines.append("    Only one therapist has this code — no comparison possible.")
                continue

            for ta, tb in combinations(therapists_with_code, 2):
                va = vecs[(ta, code)]
                vb = vecs[(tb, code)]
                js = _js(va.copy(), vb.copy())

                # Chi-square
                result = _chi_square_pair(va, vb, f"T{ta}", f"T{tb}")
                p      = result["p_value"]
                sig    = " **sig**" if (not np.isnan(p) and p < p_threshold) else ""
                n_a    = int(va.sum())
                n_b    = int(vb.sum())

                if js < js_agree:
                    verdict = "AGREE"
                elif js > js_disagree:
                    verdict = "DISAGREE"
                else:
                    verdict = "MIXED"

                out_lines.append(
                    f"    T{ta} vs T{tb}: [{verdict}]  "
                    f"JS={js:.3f}  p={p:.4f}{sig}  (n={n_a}, {n_b})"
                )

        # ── SECTION 2: ranked similarity for every (therapist, code) ─────────
        section(f"[{speaker_label}] 2. RANKED SIMILARITY: FOR EACH THERAPIST x CODE, "
                f"ALL OTHER THERAPIST x CODE PAIRS", "-")
        out_lines.append(
            "  Read as: given T_a's sequences labeled as Code_a, how similar\n"
            "  are they to every other therapist's code sequences?\n"
            "  Lower JS = more similar.  Same-code rows marked with [=].\n"
        )

        sim_rows = []   # for CSV output

        for ta in all_therapists:
            out_lines.append(f"\n  Therapist T{ta}:")
            ta_codes = sorted(c for c in all_codes if (ta, c) in vecs)

            if not ta_codes:
                out_lines.append("    No coded sequences.")
                continue

            for code_a in ta_codes:
                va   = vecs[(ta, code_a)]
                n_ta = int(va.sum())
                out_lines.append(f"\n    Code {code_a}  (n_das={n_ta}):")

                # Compute JS to all other (therapist, code) pairs
                comparisons = []
                for tb in all_therapists:
                    if tb == ta:
                        continue
                    for code_b in all_codes:
                        if (tb, code_b) not in vecs:
                            continue
                        vb   = vecs[(tb, code_b)]
                        n_tb = int(vb.sum())
                        js   = _js(va.copy(), vb.copy())
                        comparisons.append((js, tb, code_b, n_tb))
                        sim_rows.append({
                            "speaker":  speaker_label,
                            "T_a":      ta,
                            "code_a":   code_a,
                            "T_b":      tb,
                            "code_b":   code_b,
                            "js":       round(js, 4),
                            "n_a":      n_ta,
                            "n_b":      n_tb,
                            "same_code": code_a == code_b,
                        })

                # Sort by JS ascending (most similar first)
                comparisons.sort(key=lambda x: x[0])

                for js, tb, code_b, n_tb in comparisons:
                    same = " [=]" if code_b == code_a else ""
                    out_lines.append(
                        f"      T{tb}:{code_b}{same}  JS={js:.3f}  (n={n_tb})"
                    )

        # Save similarity table as CSV
        if sim_rows:
            df_sim = pd.DataFrame(sim_rows).sort_values(
                ["T_a", "code_a", "js"]
            )
            csv_path = os.path.join(
                outdir, f"cross_therapist_code_similarity_{speaker_label}.csv"
            )
            df_sim.to_csv(csv_path, index=False)
            print(f"  Saved: cross_therapist_code_similarity_{speaker_label}.csv")

    # Write text file
    out_path = os.path.join(outdir, "labeling_narrative.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(out_lines))
    print(f"\n  Saved: {out_path}")



# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Coding consistency analysis across therapists and sessions."
    )
    parser.add_argument("--dir",            required=True,
                        help="Directory containing transcript CSV/TSV/XLSX files.")
    parser.add_argument("--granularity",    default="groups",
                        choices=["groups", "raw"])
    parser.add_argument("--target",         default="patient",
                        choices=["patient", "therapist"],
                        help="Which importance column to analyse (default: patient).")
    parser.add_argument("--context_window", type=int, default=0,
                        help="DAs of context to include around each important block "
                             "when building DA sequences.  0 = block only (default).")
    parser.add_argument("--outdir",         default="coding_consistency_output/")
    parser.add_argument("--p_threshold",   type=float, default=0.05,
                        help="Chi-square p-value threshold for significance "
                             "(default: 0.05).")
    parser.add_argument("--js_agree",      type=float, default=0.3,
                        help="JS divergence below which therapists agree "
                             "(default: 0.3).")
    parser.add_argument("--js_disagree",   type=float, default=0.5,
                        help="JS divergence above which therapists disagree "
                             "(default: 0.5).")
    parser.add_argument("--ngram_ns",      type=str,   default="1,2,3,4,5,6,7,8,9,10",
                        help="Comma-separated n-gram sizes for sequence "
                             "consistency analysis (default: 1,2,3).")
    parser.add_argument("--window_size",   type=int,   default=0,
                        help="Fixed window size for chunking non-important "
                             "sequences.  0 = disabled, use full runs (default). "
                             "Set e.g. 24 to chunk into fixed-length windows.")
    args = parser.parse_args()

    dir_path = Path(args.dir)
    if not dir_path.exists():
        raise ValueError(f"Directory not found: {args.dir}")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"  Coding Consistency Analysis")
    print(f"  target={args.target}  granularity={args.granularity}  "
          f"context_window={args.context_window}")
    print(f"{'='*60}")

    # ── load ──────────────────────────────────────────────────────────────────
    records = load_transcripts(dir_path, args.target, args.granularity)
    if len(records) < 2:
        raise RuntimeError(
            f"Need ≥2 transcripts with important labels, found {len(records)}."
        )

    therapist_ids = sorted({r["therapist_id"] for r in records})
    print(f"\nLoaded {len(records)} transcripts across "
          f"{len(therapist_ids)} therapist(s): {therapist_ids}")

    # ── rate analysis ─────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  RATE ANALYSIS")
    print(f"{'─'*60}")
    analyse_rates(records, args.outdir)
    krippendorff_alpha_rates(records, args.outdir)

    # ── extract all important blocks ──────────────────────────────────────────
    all_blocks: list[dict] = []
    for rec in records:
        all_blocks.extend(extract_blocks(rec, context_window=args.context_window))

    print(f"\nTotal important blocks extracted: {len(all_blocks)}")

    # ── code usage ────────────────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  CODE USAGE")
    print(f"{'─'*60}")
    analyse_code_usage(all_blocks, args.outdir)

    # ── per-code sequence analysis (all speakers + per speaker) ─────────────
    print(f"\n{'─'*60}")
    print("  SEQUENCE ANALYSIS PER CODE")
    print(f"{'─'*60}")

    summary_rows = []
    for seq_key, speaker_label in [
        ("da_sequence",           "all"),
        ("da_sequence_therapist", "therapist_spkr"),
        ("da_sequence_patient",   "patient_spkr"),
    ]:
        print(f"\n  [{speaker_label}]")
        rows = analyse_code_sequences(
            all_blocks, args.granularity, args.outdir,
            seq_key=seq_key, speaker_label=speaker_label,
        )
        for r in rows:
            r["speaker"] = speaker_label
        summary_rows.extend(rows)

    # ── cross-code comparison (all speakers + per speaker) ───────────────────
    print(f"\n{'─'*60}")
    print("  CROSS-CODE COMPARISON")
    print(f"{'─'*60}")
    for seq_key, speaker_label in [
        ("da_sequence",           "all"),
        ("da_sequence_therapist", "therapist_spkr"),
        ("da_sequence_patient",   "patient_spkr"),
    ]:
        print(f"\n  [{speaker_label}]")
        analyse_cross_code(
            all_blocks, args.outdir,
            seq_key=seq_key, speaker_label=speaker_label,
        )

    # ── performance ceiling ──────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  PERFORMANCE CEILING ANALYSIS")
    print(f"{'─'*60}")
    ngram_ns = [int(n.strip()) for n in args.ngram_ns.split(",") if n.strip()]
    analyse_performance_ceiling(
        all_blocks=all_blocks,
        outdir=args.outdir,
        ngram_ns=ngram_ns,
    )

    # ── extended ceiling: imp vs non-imp + code discrimination ───────────────
    print(f"\n{'─'*60}")
    print("  EXTENDED CEILING ANALYSIS")
    print(f"{'─'*60}")
    analyse_extended_ceiling(
        all_blocks=all_blocks,
        records=records,
        outdir=args.outdir,
        window_size=args.window_size if args.window_size > 0 else None,
        ngram_ns=ngram_ns,
    )

    # ── labeling narrative ───────────────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("  LABELING NARRATIVE")
    print(f"{'─'*60}")
    write_labeling_narrative(
        all_blocks=all_blocks,
        outdir=args.outdir,
        p_threshold=args.p_threshold,
        js_agree=args.js_agree,
        js_disagree=args.js_disagree,
    )

    # ── top-level summary ─────────────────────────────────────────────────────
    if summary_rows:
        df_summary = (
            pd.DataFrame(summary_rows)
            .sort_values(["speaker", "code", "comparison", "p_value"])
        )
        df_summary["significant_p05"] = df_summary["p_value"] < 0.05
        summary_path = os.path.join(args.outdir, "summary.csv")
        df_summary.to_csv(summary_path, index=False)
        print(f"\n  Saved: {summary_path}")

        n_sig = int(df_summary["significant_p05"].sum())
        n_tot = len(df_summary)
        print(f"\n  {n_sig}/{n_tot} comparisons significant at p<0.05")
        print("\n  Most divergent comparisons (by JS divergence):")
        print(
            df_summary.nlargest(10, "js_divergence")[
                ["speaker", "code", "comparison", "label_a", "label_b",
                 "chi2", "p_value", "js_divergence", "n_a", "n_b"]
            ].to_string(index=False)
        )

    print(f"\nDone. Outputs in: {args.outdir}")


if __name__ == "__main__":
    main()