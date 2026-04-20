"""
analyze_code_da_patterns.py
───────────────────────────
Analyzes dialogue-act (DA) group patterns within and around "important" turns,
broken down by code label (patient_code / therapist_code).

For each important turn the script collects:
  • up to CONTEXT_WINDOW DAs before the turn  (pre-context)
  • all DAs that belong to the turn itself     (core)
  • up to CONTEXT_WINDOW DAs after the turn    (post-context)

It then counts DA-group frequencies per code and compares codes via chi-square.

Usage
-----
python analyze_code_da_patterns.py \
    --dir path/to/csv_folder \
    [--context 15] \
    [--graph_dir output/] \
    [--word_col spoken_text] \
    [--target_col Proc_DA]
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

# ── import shared constants ────────────────────────────────────────────────────
from plotting.graph_file_da import DA_COLUMN, DA_GROUPS   # noqa: E402  (project-local)

# ── constants ─────────────────────────────────────────────────────────────────
CONTEXT_WINDOW = 15          # DAs of context to grab on each side
OTHER_GROUP    = "other"     # label for DAs not in any DA_GROUP

# Pre-build a flat DA → group mapping (includes OTHER_GROUP for unmapped)
DA_TO_GROUP: dict[str, str] = {}
for _group, _members in DA_GROUPS.items():
    for _da in _members:
        DA_TO_GROUP[_da] = _group

DA_GROUP_ORDER = list(DA_GROUPS.keys()) + [OTHER_GROUP]



def load_da_level(filepath: str) -> pd.DataFrame:
    """
    Read one transcript CSV and collapse to DA-level (one row per DA_number).
    Returns a DataFrame with at least:
        DA_number, speaker, spoken_text, DA_COLUMN,
        patient_important, therapist_important,
        patient_code, therapist_code, timestamp
    """
    df = pd.read_csv(filepath)

    df_da = (
        df.groupby("DA_number", as_index=False)
        .agg(
            {
                "speaker":            "first",
                "spoken_text":        lambda x: " ".join(x.dropna()),
                DA_COLUMN:            "last",
                "patient_important":  "max",
                "therapist_important":"max",
                "patient_code":       "max",
                "therapist_code":     "max",
                "timestamp":          "max",
            }
        )
        .reset_index(drop=True)
    )

    # Drop sentinel non-DA rows
    df_da = df_da[df_da[DA_COLUMN] != "I-"].reset_index(drop=True)

    return df_da


def map_group(da: str) -> str:
    return DA_TO_GROUP.get(da, OTHER_GROUP)

def _parse_codes(raw) -> list[str]:
    """Split a (possibly comma-separated) code cell into a clean list."""
    s = str(raw)
    codes = [c.strip() for c in s.split(",") if c.strip() not in ("", "nan", "NaN", "None")]
    return codes if codes else ["NA"]


def extract_important_windows(
    df: pd.DataFrame,
    importance_col: str,
    code_col: str,
    context: int = CONTEXT_WINDOW,
) -> list[dict]:
    """
    For each contiguous run of rows where importance_col == 1, extract:
        - pre_context : up to `context` DA rows immediately before the run
        - core        : the run itself
        - post_context: up to `context` DA rows immediately after the run

    Returns a list of dicts, one per (run × code) combination so that a turn
    tagged with multiple codes produces one entry per code.
    """
    rows = []
    n = len(df)
    importance_flags = df[importance_col].fillna(0).astype(int).tolist()

    i = 0
    while i < n:
        if importance_flags[i] == 1:
            # find the end of this contiguous important run
            j = i
            while j < n and importance_flags[j] == 1:
                j += 1
            # i..j-1 is the core run

            pre_start  = max(0, i - context)
            post_end   = min(n, j + context)

            core_df    = df.iloc[i:j]
            pre_df     = df.iloc[pre_start:i]
            post_df    = df.iloc[j:post_end]

            # Get codes from ANY row in the core (they should share a code)
            raw_codes  = core_df[code_col].dropna()
            all_codes: list[str] = []
            for raw in raw_codes:
                all_codes.extend(_parse_codes(raw))
            unique_codes = list(dict.fromkeys(all_codes))  # preserve order, dedupe
            if not unique_codes:
                unique_codes = ["NA"]

            for code in unique_codes:
                rows.append(
                    {
                        "code":         code,
                        "core_df":      core_df,
                        "pre_df":       pre_df,
                        "post_df":      post_df,
                        "core_len":     len(core_df),
                        "pre_len":      len(pre_df),
                        "post_len":     len(post_df),
                        "source_file":  df["source_file"].iloc[i] if "source_file" in df.columns else "",
                    }
                )

            i = j  # skip past the run
        else:
            i += 1

    return rows


def count_da_groups(sub_df: pd.DataFrame) -> Counter:
    """Return a Counter of DA group labels for a sub-DataFrame of DAs."""
    return Counter(sub_df[DA_COLUMN].map(map_group).tolist())


def build_code_profile_table(
    windows: list[dict],
    region: str = "core",
) -> pd.DataFrame:
    """
    Aggregate DA-group counts per code for a given region
    ('pre', 'core', or 'post').  Returns a DataFrame:
        rows    = codes
        columns = DA groups
        values  = raw counts
    Also adds a 'n_turns' column = number of important turns with that code.
    """
    assert region in ("pre", "core", "post")
    df_col = f"{region}_df"

    code_counts: dict[str, Counter] = {}
    code_turn_counts: dict[str, int] = {}

    for w in windows:
        code = w["code"]
        counts = count_da_groups(w[df_col])
        if code not in code_counts:
            code_counts[code] = Counter()
            code_turn_counts[code] = 0
        code_counts[code] += counts
        code_turn_counts[code] += 1

    records = []
    for code, cnt in code_counts.items():
        row = {"code": code, "n_turns": code_turn_counts[code]}
        for g in DA_GROUP_ORDER:
            row[g] = cnt.get(g, 0)
        records.append(row)

    result = pd.DataFrame(records).set_index("code")
    # Fill any missing group columns with 0
    for g in DA_GROUP_ORDER:
        if g not in result.columns:
            result[g] = 0

    return result.sort_values("n_turns", ascending=False)

def chi2_pairwise(count_table: pd.DataFrame) -> pd.DataFrame:
    """
    For every pair of codes (rows of count_table), run a chi-square test on
    their DA-group count vectors.  Returns a tidy DataFrame with columns:
        code_a, code_b, chi2, p_value, p_corrected, significant
    Multiple-testing correction: Benjamini-Hochberg FDR.
    """
    group_cols = [c for c in DA_GROUP_ORDER if c in count_table.columns]
    codes = list(count_table.index)

    results = []
    for a, b in combinations(codes, 2):
        row_a = count_table.loc[a, group_cols].values
        row_b = count_table.loc[b, group_cols].values
        contingency = np.array([row_a, row_b], dtype=float)

        # drop all-zero columns to avoid degenerate chi2
        non_zero_cols = contingency.sum(axis=0) > 0
        contingency = contingency[:, non_zero_cols]

        if contingency.shape[1] < 2:
            # can't run chi2 with <2 categories
            results.append({"code_a": a, "code_b": b, "chi2": np.nan, "p_value": np.nan})
            continue

        chi2, p, _, _ = chi2_contingency(contingency)
        results.append({"code_a": a, "code_b": b, "chi2": chi2, "p_value": p})

    df_res = pd.DataFrame(results)
    if df_res.empty or df_res["p_value"].isna().all():
        return df_res

    valid_mask = df_res["p_value"].notna()
    if valid_mask.sum() > 0:
        _, p_corr, _, _ = multipletests(
            df_res.loc[valid_mask, "p_value"], method="fdr_bh"
        )
        df_res.loc[valid_mask, "p_corrected"] = p_corr
        df_res.loc[valid_mask, "significant"] = p_corr < 0.05

    return df_res


_REGION_COLORS = {
    "pre":  "#7fbfff",   # soft blue
    "core": "#ff9f7f",   # soft orange
    "post": "#7fff9f",   # soft green
}

_CMAP = plt.get_cmap("tab10")
_GROUP_COLOR = {g: _CMAP(i / len(DA_GROUP_ORDER)) for i, g in enumerate(DA_GROUP_ORDER)}


def plot_stacked_proportions(
    count_table: pd.DataFrame,
    title: str,
    outdir: str,
    fname: str,
    save: bool = True,
    show: bool = False,
    min_turns: int = 2,
):
    """
    Stacked proportional bar chart one bar per code, coloured by DA group.
    Bars with fewer than `min_turns` turns are shown with hatching to indicate
    low support.
    """
    group_cols = [c for c in DA_GROUP_ORDER if c in count_table.columns]
    table = count_table[count_table["n_turns"] >= 1].copy()
    if table.empty:
        return

    counts  = table[group_cols].values.astype(float)
    totals  = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1          # avoid div/0
    props   = counts / totals

    codes   = list(table.index)
    n_codes = len(codes)
    n_turns = table["n_turns"].values

    fig, ax = plt.subplots(figsize=(max(10, n_codes * 0.7), 5))
    bottoms = np.zeros(n_codes)
    x       = np.arange(n_codes)

    for gi, group in enumerate(group_cols):
        vals   = props[:, gi]
        colors = [_GROUP_COLOR.get(group, "grey")] * n_codes
        ax.bar(x, vals, bottom=bottoms, label=group, color=colors,
               alpha=0.85, edgecolor="white")
        bottoms += vals

    # hatch bars with very few turns
    for xi, nt in enumerate(n_turns):
        if nt < min_turns:
            ax.bar(xi, 1.0, color="none", edgecolor="black",
                   linewidth=0.5, hatch="////", alpha=0.3)

    # annotate n_turns above each bar
    for xi, nt in enumerate(n_turns):
        ax.text(xi, 1.02, f"n={nt}", ha="center", va="bottom",
                fontsize=7, rotation=45)

    ax.set_xticks(x)
    ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Proportion of DA groups")
    ax.set_ylim(0, 1.20)
    ax.set_title(title, fontsize=11)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left",
              fontsize=8, title="DA group")
    ax.grid(True, axis="y", color="lightgrey", linewidth=0.5)

    plt.tight_layout()
    if save:
        path = os.path.join(outdir, fname)
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close()


def plot_region_comparison(
    pre_table: pd.DataFrame,
    core_table: pd.DataFrame,
    post_table: pd.DataFrame,
    code: str,
    title_prefix: str,
    outdir: str,
    save: bool = True,
    show: bool = False,
):
    """
    For a single code, show a grouped bar chart comparing DA-group proportions
    across pre / core / post regions.
    """
    group_cols = [c for c in DA_GROUP_ORDER
                  if c in pre_table.columns
                  and c in core_table.columns
                  and c in post_table.columns]

    def _get_props(table: pd.DataFrame) -> np.ndarray:
        if code not in table.index:
            return np.zeros(len(group_cols))
        row = table.loc[code, group_cols].values.astype(float)
        total = row.sum()
        return row / total if total > 0 else row

    pre_props  = _get_props(pre_table)
    core_props = _get_props(core_table)
    post_props = _get_props(post_table)

    x      = np.arange(len(group_cols))
    width  = 0.25
    fig, ax = plt.subplots(figsize=(max(10, len(group_cols) * 0.9), 5))

    ax.bar(x - width, pre_props,  width, label="Pre-context",  color=_REGION_COLORS["pre"],  alpha=0.85, edgecolor="white")
    ax.bar(x,         core_props, width, label="Core turn",    color=_REGION_COLORS["core"], alpha=0.85, edgecolor="white")
    ax.bar(x + width, post_props, width, label="Post-context", color=_REGION_COLORS["post"], alpha=0.85, edgecolor="white")

    ax.set_xticks(x)
    ax.set_xticklabels(group_cols, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Proportion")
    ax.set_ylim(0, 1.0)
    ax.set_title(f"{title_prefix} code '{code}' pre/core/post DA profile", fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, axis="y", color="lightgrey", linewidth=0.5)

    plt.tight_layout()
    if save:
        safe_code = str(code).replace("/", "-").replace(" ", "_")
        fname = os.path.join(outdir, f"{title_prefix}_region_profile_{safe_code}.png")
        plt.savefig(fname, bbox_inches="tight")
        print(f"  Saved: {fname}")
    if show:
        plt.show()
    plt.close()


def plot_heatmap_da_by_code(
    count_table: pd.DataFrame,
    title: str,
    outdir: str,
    fname: str,
    save: bool = True,
    show: bool = False,
):
    """
    Heatmap of z-scored DA-group proportions across codes.
    Useful for spotting which codes over/under-use specific DA groups.
    """
    group_cols = [c for c in DA_GROUP_ORDER if c in count_table.columns]
    counts  = count_table[group_cols].values.astype(float)
    totals  = counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    props   = counts / totals

    # z-score across codes per DA group
    with np.errstate(invalid="ignore"):
        means = props.mean(axis=0)
        stds  = props.std(axis=0)
        stds[stds == 0] = 1
        z_props = (props - means) / stds

    codes = list(count_table.index)
    fig, ax = plt.subplots(figsize=(max(8, len(group_cols) * 1.0),
                                    max(4, len(codes) * 0.5)))
    im = ax.imshow(z_props, aspect="auto", cmap="RdBu_r", vmin=-2.5, vmax=2.5)
    plt.colorbar(im, ax=ax, label="z-score (proportion)")

    ax.set_xticks(range(len(group_cols)))
    ax.set_xticklabels(group_cols, rotation=30, ha="right", fontsize=9)
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels(codes, fontsize=8)

    # annotate with raw proportions
    for i in range(len(codes)):
        for j in range(len(group_cols)):
            ax.text(j, i, f"{props[i, j]:.2f}", ha="center", va="center",
                    fontsize=7, color="black")

    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    if save:
        path = os.path.join(outdir, fname)
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close()


def plot_chi2_significance_matrix(
    chi2_df: pd.DataFrame,
    codes: list[str],
    title: str,
    outdir: str,
    fname: str,
    save: bool = True,
    show: bool = False,
):
    if chi2_df.empty or "p_corrected" not in chi2_df.columns:
        return

    n = len(codes)
    mat = np.full((n, n), np.nan)
    code_idx = {c: i for i, c in enumerate(codes)}

    for _, row in chi2_df.iterrows():
        if pd.isna(row.get("p_corrected")):
            continue
        i = code_idx.get(row["code_a"])
        j = code_idx.get(row["code_b"])
        if i is None or j is None:
            continue
        val = -np.log10(max(row["p_corrected"], 1e-10))
        mat[i, j] = val
        mat[j, i] = val

    fig, ax = plt.subplots(figsize=(max(6, n * 0.7), max(5, n * 0.7)))
    im = ax.imshow(mat, cmap="YlOrRd", vmin=0)
    plt.colorbar(im, ax=ax, label="-log10(p_corrected FDR)")

    ax.set_xticks(range(n))
    ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(n))
    ax.set_yticklabels(codes, fontsize=8)

    # mark significant pairs with a star
    threshold = -np.log10(0.05)
    for i in range(n):
        for j in range(n):
            if i != j and not np.isnan(mat[i, j]):
                marker = "★" if mat[i, j] >= threshold else ""
                ax.text(j, i, marker, ha="center", va="center",
                        fontsize=10, color="black")

    ax.set_title(title, fontsize=11)
    plt.tight_layout()
    if save:
        path = os.path.join(outdir, fname)
        plt.savefig(path, bbox_inches="tight")
        print(f"  Saved: {path}")
    if show:
        plt.show()
    plt.close()

def run_speaker_analysis(
    combined: pd.DataFrame,
    importance_col: str,
    code_col: str,
    speaker_label: str,
    context: int = CONTEXT_WINDOW,
    outdir: str = "output/",
    save: bool = True,
    show: bool = False,
):
    """
    Full pipeline for one speaker role (patient or therapist):
      1. Extract important windows (with context)
      2. Build count tables per region
      3. Plot proportional stacked bars per region
      4. Plot per-code pre/core/post region comparisons
      5. Heatmaps for core region
      6. Chi-square pairwise tests on core DA-group profiles
      7. Chi-square significance matrix
      8. Save summary CSVs
    """
    prefix = f"{speaker_label}"
    print(f"\n{'='*60}")
    print(f"  {speaker_label.upper()} important turns via '{importance_col}'")
    print(f"{'='*60}")

    # 1. Extract windows
    windows = extract_important_windows(
        combined, importance_col=importance_col,
        code_col=code_col, context=context,
    )

    if not windows:
        print(f"  No important turns found for {speaker_label}. Skipping.")
        return

    print(f"  Found {len(windows)} (turn × code) entries across "
          f"{combined['source_file'].nunique() if 'source_file' in combined.columns else '?'} files.")

    # 2. Build count tables
    pre_table  = build_code_profile_table(windows, region="pre")
    core_table = build_code_profile_table(windows, region="core")
    post_table = build_code_profile_table(windows, region="post")

    # Handle zero-context case: if no pre/post rows exist at all, tables will
    # be populated but every group count will be 0 that's fine, plots will
    # just show flat zero bars for those regions.

    print(f"\n  Codes found: {list(core_table.index)}")
    print(f"  Turns per code:\n{core_table['n_turns'].to_string()}")

    # 3. Stacked proportion plots for each region
    for region, table in [("pre", pre_table), ("core", core_table), ("post", post_table)]:
        if region == "pre" and context == 0:
            continue
        if region == "post" and context == 0:
            continue
        plot_stacked_proportions(
            table,
            title=f"{prefix} {region} DA group proportions by code",
            outdir=outdir,
            fname=f"{prefix}_{region}_da_proportions.png",
            save=save,
            show=show,
        )

    # 4. Per-code region comparison (pre vs core vs post)
    if context > 0:
        all_codes = sorted(
            set(core_table.index) | set(pre_table.index) | set(post_table.index)
        )
        for code in all_codes:
            plot_region_comparison(
                pre_table, core_table, post_table,
                code=code,
                title_prefix=prefix,
                outdir=outdir,
                save=save,
                show=show,
            )

    # 5. Core heatmap (z-scored)
    if len(core_table) >= 2:
        plot_heatmap_da_by_code(
            core_table,
            title=f"{prefix} core turn DA group z-score heatmap",
            outdir=outdir,
            fname=f"{prefix}_core_da_heatmap.png",
            save=save,
            show=show,
        )

    # 6. Pairwise chi-square on core DA profiles
    chi2_results = chi2_pairwise(core_table)

    if not chi2_results.empty and "p_corrected" in chi2_results.columns:
        sig = chi2_results[chi2_results.get("significant", False) == True]
        print(f"\n  Pairwise chi-square (core region) "
              f"{len(sig)} significant pairs (FDR-corrected p < 0.05):")
        if not sig.empty:
            print(sig[["code_a", "code_b", "chi2", "p_value", "p_corrected"]].to_string(index=False))
        else:
            print("  None significant after correction.")

    # 7. Significance matrix
    if len(core_table) >= 2:
        plot_chi2_significance_matrix(
            chi2_results,
            codes=list(core_table.index),
            title=f"{prefix} pairwise chi-square (core) significance",
            outdir=outdir,
            fname=f"{prefix}_core_chi2_matrix.png",
            save=save,
            show=show,
        )

    # 8. Save CSVs
    for region, table in [("pre", pre_table), ("core", core_table), ("post", post_table)]:
        csv_path = os.path.join(outdir, f"{prefix}_{region}_da_counts.csv")
        table.to_csv(csv_path)
        print(f"  Saved CSV: {csv_path}")

    if not chi2_results.empty:
        chi2_csv = os.path.join(outdir, f"{prefix}_core_chi2_pairwise.csv")
        chi2_results.to_csv(chi2_csv, index=False)
        print(f"  Saved CSV: {chi2_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyse DA group patterns in important turns, by code."
    )
    parser.add_argument("--dir",      type=str, required=True,
                        help="Directory of transcript CSVs (word-level daseg output).")
    parser.add_argument("--context",  type=int, default=CONTEXT_WINDOW,
                        help=f"Context window size (DAs before/after). 0 = no context. Default: {CONTEXT_WINDOW}.")
    parser.add_argument("--graph_dir", type=str, default="code_da_output/",
                        help="Output directory for plots and CSVs.")
    parser.add_argument("--word_col",  type=str, default="spoken_text")
    parser.add_argument("--target_col", type=str, default="Proc_DA")
    parser.add_argument("--show",      action="store_true",
                        help="Display plots interactively (in addition to saving).")
    args = parser.parse_args()

    dir_path = Path(args.dir)
    if not dir_path.exists():
        raise ValueError(f"Path does not exist: {args.dir}")

    os.makedirs(args.graph_dir, exist_ok=True)

    # ── load all files ────────────────────────────────────────────────────────
    allowed_exts = {".csv", ".tsv", ".xlsx"}
    dfs: list[pd.DataFrame] = []
    for file in sorted(dir_path.iterdir()):
        if file.suffix.lower() in allowed_exts:
            print(f"Loading: {file.name}")
            try:
                df = load_da_level(str(file))
                df["source_file"] = file.name
                dfs.append(df)
            except Exception as e:
                print(f"  WARNING: could not load {file.name}: {e}")

    if not dfs:
        raise RuntimeError("No valid files loaded. Check --dir.")

    combined = pd.concat(dfs, ignore_index=True)
    combined = combined[combined[DA_COLUMN] != "I-"].reset_index(drop=True)
    print(f"\nTotal DA rows loaded: {len(combined)} across {len(dfs)} file(s).")

    # ── run analysis for patient and therapist ────────────────────────────────
    run_speaker_analysis(
        combined,
        importance_col="patient_important",
        code_col="patient_code",
        speaker_label="patient",
        context=args.context,
        outdir=args.graph_dir,
        save=True,
        show=args.show,
    )

    run_speaker_analysis(
        combined,
        importance_col="therapist_important",
        code_col="therapist_code",
        speaker_label="therapist",
        context=args.context,
        outdir=args.graph_dir,
        save=True,
        show=args.show,
    )

    print(f"\nDone. All outputs written to: {args.graph_dir}")