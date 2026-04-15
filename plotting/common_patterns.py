"""
analyze_da_patterns.py

Discovers common dialogue-act (DA) group patterns within and around "important"
turns, broken down by code (patient_code / therapist_code).

For each speaker (therapist / patient):
  - Finds contiguous blocks of flagged important DAs
  - Attaches a configurable context window of DAs before and after each block
  - Maps every DA to a DA group (unmapped DAs fall into purpose-built groups
    or "other"); questions are merged into one group, answers into one group
  - RLE-compresses each sequence so consecutive DAs of the same group are
    collapsed into one region before n-gramming
    (e.g. ST-opinion, ST-non-opinion, HG  ->  ST -> HG as a bigram)
  - Extracts n-gram sequences over those compressed regions
  - Compares n-gram frequency distributions across codes
  - Plots a directed transition graph: nodes = DA labels, directed edges =
    observed transitions, edge weight = count (heavier / shorter = more common)

Granularity
-----------
--granularity groups   (default) -- work with DA group labels (e.g. "questions")
--granularity raw      -- work with individual DA class labels as-is

Usage
-----
python analyze_da_patterns.py \\
    --dir /path/to/csv_dir \\
    --ngram_sizes 2 3 \\
    --context_window 15 \\
    --top_n 15 \\
    --granularity groups \\
    --outdir da_pattern_output/

Drop this file alongside graph_file_da.py (provides DA_COLUMN, DA_GROUPS).
Requires: pandas, numpy, matplotlib, scipy, statsmodels, networkx
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from itertools import islice
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pandas as pd
import networkx as nx
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests

from graph_file_da import DA_COLUMN, DA_GROUPS

# ── extended DA group definitions ─────────────────────────────────────────────
# Questions (canonical + non-canonical) merged into one group.
# Answers  (canonical + non-canonical) merged into one group.
# Five new groups cover previously unmapped DAs from daseg.

EXTENDED_DA_GROUPS: dict[str, list[str]] = {
    # merged from original two question groups
    "questions": [
        "Wh-Question", "Yes-No-Question", "Open-Question", "Or-Clause",
        "Declarative-Yes-No-Question", "Rhetorical-Questions",
        "Backchannel-in-question-form", "Tag-Question",
        "Declarative-Wh-Question",
    ],
    # merged from original two answer groups
    "answers": [
        "Yes-answers", "No-answers",
        "Affirmative-non-yes-answers", "Other-answers",
        "Negative-non-no-answers", "Dispreferred-answers", "Reject",
    ],
    # kept from original
    "backchannel": [
        "Hold-before-answer-agreement", "Acknowledge-Backchannel",
        "Response-Acknowledgement",
    ],
    "statements": ["Statement-non-opinion", "Statement-opinion"],
    "hedge":      ["Hedge"],
    # Social ritual: conversational glue, no propositional content.
    # In therapy marks session boundaries and softening moves.
    "social_ritual": [
        "Conventional-opening", "Conventional-closing",
        "Thanking", "Apology", "Appreciation", "Downplayer",
    ],
    # Acknowledgement: signals heard/understood, stronger than backchannel.
    "acknowledgement": [
        "Agree-Accept", "Signal-non-understanding",
    ],
    # Elaboration: reformulating or extending prior speech.
    # Particularly relevant for therapeutic reflection techniques.
    "elaboration": [
        "Summarize/reformulate", "Repeat-phrase",
        "Collaborative-Completion", "Quotation",
    ],
    # Action: directives or offers that push the conversation forward.
    "action": [
        "Action-directive", "Offers-Options-Commits",
    ],
    # Noise: non-linguistic or unclassifiable rows.
    "noise": [
        "Non-verbal", "Uninterpretable", "Other", "3rd-party-talk",
    ],
}

# Short display abbreviations for axis labels and graph nodes
DA_GROUP_ABBREV: dict[str, str] = {
    "questions":      "Q",
    "answers":        "A",
    "backchannel":    "BC",
    "statements":     "ST",
    "hedge":          "HG",
    "social_ritual":  "SR",
    "acknowledgement": "ACK",
    "elaboration":    "EL",
    "action":         "ACT",
    "noise":          "NS",
    "other":          "OT",
}

_N = len(DA_GROUP_ABBREV)
_CMAP = plt.get_cmap("tab20")
DA_GROUP_COLORS: dict[str, str] = {
    g: mcolors.to_hex(_CMAP(i / _N)) for i, g in enumerate(DA_GROUP_ABBREV)
}

# ── DA -> group mapping ───────────────────────────────────────────────────────

def _build_da_to_group() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for group, members in EXTENDED_DA_GROUPS.items():
        for da in members:
            mapping[da] = group
    return mapping

DA_TO_GROUP: dict[str, str] = _build_da_to_group()


def map_da_to_group(da: str) -> str:
    return DA_TO_GROUP.get(da, "other")


# ── data loading ──────────────────────────────────────────────────────────────

def load_da_level(filepath: str | Path) -> pd.DataFrame:
    """
    Read a word-level CSV, collapse to one row per DA_number, and add a
    'da_group' column.
    """
    df = pd.read_csv(filepath)

    agg_spec: dict = {
        "speaker":             "first",
        "spoken_text":         lambda x: " ".join(x.dropna()),
        DA_COLUMN:             "last",
        "patient_important":   "max",
        "therapist_important": "max",
        "patient_code":        "max",
        "therapist_code":      "max",
    }
    if "timestamp" in df.columns:
        agg_spec["timestamp"] = "max"

    df_da = (
        df.groupby("DA_number", as_index=False)
          .agg(agg_spec)
          .reset_index(drop=True)
    )
    df_da = df_da[df_da[DA_COLUMN] != "I-"].reset_index(drop=True)
    df_da["da_group"] = df_da[DA_COLUMN].map(map_da_to_group)
    return df_da


# ── label resolution (groups vs raw) ─────────────────────────────────────────

def get_label(row_da: str, row_group: str, granularity: str) -> str:
    """
    'groups' -> DA group name (e.g. 'questions')
    'raw'    -> original DA class string (e.g. 'Wh-Question')
    """
    return row_group if granularity == "groups" else row_da


def abbrev(label: str, granularity: str) -> str:
    """Short string for axis labels and graph node text."""
    if granularity == "groups":
        return DA_GROUP_ABBREV.get(label, label)
    # For raw DAs, abbreviate common suffixes so labels fit in graph nodes
    return (label
            .replace("-Question", "-Q")
            .replace("-answers", "-A")
            .replace("Statement-", "ST-")
            .replace("Acknowledge-", "Ack-")
            .replace("Conventional-", "Conv-")
            .replace("Collaborative-", "Collab-"))


def node_color(label: str, granularity: str) -> str:
    if granularity == "groups":
        return DA_GROUP_COLORS.get(label, "#aaaaaa")
    grp = map_da_to_group(label)
    return DA_GROUP_COLORS.get(grp, "#aaaaaa")


# ── important-block extraction ────────────────────────────────────────────────

def _parse_codes(raw) -> list[str]:
    s = str(raw)
    codes = [c.strip() for c in s.split(",") if c.strip() not in ("", "nan", "NaN", "None")]
    return codes if codes else ["NA"]


def extract_important_blocks(
    df: pd.DataFrame,
    importance_col: str,
    code_col: str,
    granularity: str,
    context_window: int = 15,
    include_context_in_block: bool = False,
) -> list[dict]:
    """
    Find contiguous runs where importance_col == 1.

    Returns list of dicts with keys:
      codes, block_indices, pre_indices, post_indices,
      full_sequence, block_sequence, pre_sequence, post_sequence
    All sequence lists contain label strings per granularity.
    """
    importance = df[importance_col].fillna(0).astype(int)
    labels = [
        get_label(row[DA_COLUMN], row["da_group"], granularity)
        for _, row in df.iterrows()
    ]
    n = len(df)
    blocks: list[dict] = []

    i = 0
    while i < n:
        if importance.iloc[i] == 1:
            j = i
            while j < n and importance.iloc[j] == 1:
                j += 1
            block_idx = list(range(i, j))
            pre_idx   = list(range(max(0, i - context_window), i))
            post_idx  = list(range(j, min(n, j + context_window)))

            raw_codes = df[code_col].iloc[block_idx]
            codes: list[str] = []
            for raw in raw_codes:
                for c in _parse_codes(raw):
                    if c not in codes:
                        codes.append(c)
            if not codes:
                codes = ["NA"]

            block_seq = [labels[k] for k in block_idx]
            pre_seq   = [labels[k] for k in pre_idx]
            post_seq  = [labels[k] for k in post_idx]
            full_seq  = (pre_seq + block_seq + post_seq
                         if include_context_in_block else block_seq)

            blocks.append({
                "codes":          codes,
                "block_indices":  block_idx,
                "pre_indices":    pre_idx,
                "post_indices":   post_idx,
                "full_sequence":  full_seq,
                "block_sequence": block_seq,
                "pre_sequence":   pre_seq,
                "post_sequence":  post_seq,
            })
            i = j
        else:
            i += 1

    return blocks


# ── non-important sequence extraction ────────────────────────────────────────

def extract_nonimportant_sequences(
    df: pd.DataFrame,
    granularity: str,
    context_window: int = 15,
    include_context_in_block: bool = False,
) -> list[list[str]]:
    """
    Return the DA sequences from rows that are NOT part of any important block
    for either speaker (patient_important or therapist_important).

    When include_context_in_block is True the ±context_window rows around
    every important block are also excluded, because those rows are already
    "owned" by the important partition.  When False, those context rows remain
    in the non-important pool (consistent with how run_analysis treats them).

    Contiguous runs of non-important rows are kept as separate sequences so
    session boundaries and topic changes are not accidentally bridged.

    Returns a list of per-run label sequences (one list per contiguous run).
    """
    pat_imp  = df["patient_important"].fillna(0).astype(int)
    ther_imp = df["therapist_important"].fillna(0).astype(int)

    # Build a boolean mask of rows that are "claimed" by either important partition
    n       = len(df)
    claimed = np.zeros(n, dtype=bool)

    for imp_col in ("patient_important", "therapist_important"):
        importance = df[imp_col].fillna(0).astype(int)
        i = 0
        while i < n:
            if importance.iloc[i] == 1:
                j = i
                while j < n and importance.iloc[j] == 1:
                    j += 1
                # Always claim the block itself
                claimed[i:j] = True
                # Claim context too when it is merged into the block
                if include_context_in_block:
                    pre_start = max(0, i - context_window)
                    post_end  = min(n, j + context_window)
                    claimed[pre_start:i] = True
                    claimed[j:post_end]  = True
                i = j
            else:
                i += 1

    labels = [
        get_label(row[DA_COLUMN], row["da_group"], granularity)
        for _, row in df.iterrows()
    ]

    # Split unclaimed rows into contiguous runs
    sequences: list[list[str]] = []
    current_run: list[str] = []
    for idx in range(n):
        if not claimed[idx]:
            current_run.append(labels[idx])
        else:
            if current_run:
                sequences.append(current_run)
                current_run = []
    if current_run:
        sequences.append(current_run)

    return sequences


# ── sequence helpers ──────────────────────────────────────────────────────────

def rle_compress(seq: Sequence[str]) -> list[str]:
    """
    Collapse consecutive identical labels into one region.
    ["ST", "ST", "HG", "HG", "Q"] -> ["ST", "HG", "Q"]
    Applied before n-gramming so patterns capture group transitions,
    not individual DA repetitions.
    """
    out: list[str] = []
    for item in seq:
        if not out or item != out[-1]:
            out.append(item)
    return out


def _ngrams(seq: Sequence[str], n: int):
    it = iter(seq)
    window = tuple(islice(it, n))
    if len(window) == n:
        yield window
    for item in it:
        window = window[1:] + (item,)
        yield window


def ngram_counter(sequences: list[list[str]], n: int) -> Counter:
    """Count n-grams over RLE-compressed sequences."""
    c: Counter = Counter()
    for seq in sequences:
        c.update(_ngrams(rle_compress(seq), n))
    return c


def _label_str(ngram: tuple[str, ...], granularity: str) -> str:
    return " -> ".join(abbrev(g, granularity) for g in ngram)


# ── statistical comparison ────────────────────────────────────────────────────

def compare_codes_chi2(
    code_counters: dict[str, Counter],
    top_patterns: list[tuple],
    granularity: str,
) -> pd.DataFrame:
    results = []
    codes = list(code_counters.keys())
    for pat in top_patterns:
        counts = [code_counters[c].get(pat, 0) for c in codes]
        totals = [sum(code_counters[c].values()) for c in codes]
        table  = [counts, [t - ct for t, ct in zip(totals, counts)]]
        if sum(counts) == 0:
            continue
        try:
            chi2, p, _, _ = chi2_contingency(table)
        except ValueError:
            continue
        results.append({"pattern": _label_str(pat, granularity), "chi2": chi2, "p_raw": p})
    if not results:
        return pd.DataFrame()
    df_res = pd.DataFrame(results)
    _, p_fdr, _, _ = multipletests(df_res["p_raw"], method="fdr_bh")
    df_res["p_fdr"]       = p_fdr
    df_res["significant"] = p_fdr < 0.05
    return df_res.sort_values("p_fdr")


# ── transition graph ──────────────────────────────────────────────────────────

def _node_label_str(node: str | tuple, granularity: str) -> str:
    """
    Render a graph node as a display string.

    Order-1 nodes are plain group/DA strings  (e.g. "ST").
    Higher-order nodes are tuples of k groups  (e.g. ("ST", "Q"))
    and are rendered as a short chain          (e.g. "ST→Q").
    """
    if isinstance(node, tuple):
        return "→".join(abbrev(g, granularity) for g in node)
    return abbrev(node, granularity)


def _node_base_group(node: str | tuple, granularity: str) -> str:
    """
    Return the DA group that determines this node's colour.
    For higher-order nodes we use the *last* element of the tuple,
    so the colour reflects where the sequence is heading.
    """
    leaf = node[-1] if isinstance(node, tuple) else node
    return leaf if granularity == "groups" else map_da_to_group(leaf)


def build_higher_order_graph(
    sequences: list[list[str]],
    order: int,
    min_edge_weight: int = 1,
) -> nx.DiGraph:
    """
    Build a k-th order Markov transition graph.

    order=1  (default / current behaviour)
    -----------------------------------------
    Nodes  = individual DA labels, e.g. "ST", "Q"
    Edges  = bigram transitions  A -> B
    Encodes sequences of length 2.

    order=2
    -----------------------------------------
    Nodes  = consecutive DA pairs (bigrams), e.g. ("ST","Q"), ("Q","A")
    Edges  = overlapping-bigram transitions  (ST,Q) -> (Q,A)
    An edge exists iff the trigram ST→Q→A was observed.
    Encodes sequences of length 3.

    order=k  (general)
    -----------------------------------------
    Nodes  = k-tuples of consecutive DAs
    Edges  = (k+1)-gram transitions between overlapping k-tuples
    Encodes sequences of length k+1.

    In all cases RLE compression is applied first so nodes represent
    contiguous regions, not individual DA rows.

    The resulting graph is returned with edge attribute "weight" = count.
    Edges below min_edge_weight are pruned.
    """
    G = nx.DiGraph()

    for seq in sequences:
        compressed = rle_compress(seq)
        if len(compressed) < order + 1:
            continue

        # Slide a window of width (order+1) across the compressed sequence.
        # Each window produces one source node (first k elements) and one
        # destination node (last k elements), which overlap by k-1 elements.
        for i in range(len(compressed) - order):
            window = tuple(compressed[i : i + order + 1])
            src = window[:order]   # k-tuple
            dst = window[1:]       # k-tuple, shifted by one
            # For order=1, unwrap single-element tuples to plain strings
            # so the graph is identical to the original first-order graph.
            src = src[0] if order == 1 else src
            dst = dst[0] if order == 1 else dst
            if G.has_edge(src, dst):
                G[src][dst]["weight"] += 1.0
            else:
                G.add_edge(src, dst, weight=1.0)

    # Prune rare edges
    remove = [e for e in G.edges() if G[e[0]][e[1]]["weight"] < min_edge_weight]
    G.remove_edges_from(remove)
    # Remove isolated nodes left by pruning
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


def plot_transition_graph(
    sequences: list[list[str]],
    granularity: str,
    title: str,
    outdir: str,
    fname: str,
    graph_order: int = 1,
    min_edge_weight: int = 2,
    save: bool = True,
    show: bool = False,
) -> nx.DiGraph | None:
    """
    Directed transition graph over RLE-compressed sequences.

    graph_order=1  (default — original behaviour)
        Nodes = individual DA group/class labels
        Edges = bigram (A->B) transitions
        Captures sequences of length 2.

    graph_order=2
        Nodes = consecutive DA pairs, e.g. (ST, Q)
        Edges = trigram transitions, e.g. (ST,Q) -> (Q,A)
        Captures sequences of length 3 — ST then Q then A shows up as
        a path  (ST,Q) -> (Q,A)  rather than two separate edges.

    graph_order=k  (general)
        Nodes = k-tuples; edges encode (k+1)-gram sequences.

    Higher-order nodes are coloured by their *last* element (where the
    sequence is heading) and labelled as a compact chain (e.g. "ST→Q").

    Layout: spring/Fruchterman-Reingold with edge weight as spring
    strength — heavily-used transitions pull their nodes together.
    """
    os.makedirs(outdir, exist_ok=True)

    G = build_higher_order_graph(sequences, order=graph_order,
                                 min_edge_weight=min_edge_weight)

    if G.number_of_edges() == 0:
        print(f"  [{title}] No transitions at order={graph_order}, "
              f"min_edge_weight={min_edge_weight}. Skipping graph.")
        return None

    weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    max_w   = weights.max() if len(weights) else 1.0

    pos = nx.spring_layout(
        G,
        weight="weight",
        k=2.0 / max(len(G), 1) ** 0.5,
        iterations=200,
        seed=42,
    )

    edge_widths = [0.8 + 6.0 * (G[u][v]["weight"] / max_w) for u, v in G.edges()]
    edge_alphas = [0.2  + 0.7 * (G[u][v]["weight"] / max_w) for u, v in G.edges()]

    node_sizes = []
    for node in G.nodes():
        out_total = sum(d["weight"] for _, _, d in G.out_edges(node, data=True))
        node_sizes.append(500 + 2500 * (out_total / (max_w * max(len(G), 1))))

    node_colors = [
        DA_GROUP_COLORS.get(_node_base_group(n, granularity), "#aaaaaa")
        for n in G.nodes()
    ]
    node_labels = {n: _node_label_str(n, granularity) for n in G.nodes()}

    # Higher-order graphs can have many nodes — scale figure size accordingly
    fig_size = (14, 10) if len(G) <= 30 else (18, 14)
    font_size = 9 if len(G) <= 20 else (7 if len(G) <= 40 else 6)

    fig, ax = plt.subplots(figsize=fig_size)
    order_note = (f"  (order {graph_order}: nodes = {graph_order}-grams, "
                  f"edges = {graph_order+1}-gram sequences)")
    ax.set_title(title + order_note, fontsize=11, pad=12)
    ax.axis("off")

    for (u, v), lw, alpha in zip(G.edges(), edge_widths, edge_alphas):
        nx.draw_networkx_edges(
            G, pos, edgelist=[(u, v)], ax=ax,
            width=lw, alpha=alpha,
            edge_color="dimgray",
            arrows=True,
            arrowstyle="-|>",
            arrowsize=14,
            connectionstyle="arc3,rad=0.12",
            min_source_margin=20,
            min_target_margin=20,
        )

    nx.draw_networkx_nodes(
        G, pos, ax=ax,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.92,
        linewidths=0.8,
        edgecolors="white",
    )
    nx.draw_networkx_labels(
        G, pos, labels=node_labels, ax=ax,
        font_size=font_size,
        font_weight="bold",
        font_color="black",
    )

    # Label top-20% heaviest edges with their count
    threshold = np.percentile(weights, 80) if len(weights) else max_w
    heavy_labels = {
        (u, v): str(int(G[u][v]["weight"]))
        for u, v in G.edges()
        if G[u][v]["weight"] >= threshold
    }
    if heavy_labels:
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=heavy_labels, ax=ax,
            font_size=7, alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, lw=0),
        )

    # Legend: one dot per DA group colour present in this graph
    seen_groups: set[str] = {_node_base_group(n, granularity) for n in G.nodes()}
    legend_handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=DA_GROUP_COLORS.get(g, "#aaa"),
                   markersize=10,
                   label=f"{DA_GROUP_ABBREV.get(g, g)}  {g}")
        for g in DA_GROUP_ABBREV if g in seen_groups
    ]
    if legend_handles:
        ax.legend(handles=legend_handles, loc="upper left",
                  fontsize=8, title="DA groups (last element)", framealpha=0.85)

    plt.tight_layout()
    if save:
        fpath = os.path.join(outdir, fname)
        plt.savefig(fpath, bbox_inches="tight", dpi=150)
        print(f"  Saved: {fpath}")
    if show:
        plt.show()
    plt.close()
    return G


# ── bar / heatmap / context plots ─────────────────────────────────────────────

def _bar_color(ngram: tuple[str, ...], granularity: str) -> str:
    return node_color(ngram[-1], granularity)


def plot_top_patterns_per_code(
    code_counters: dict[str, Counter],
    ngram_size: int,
    top_n: int,
    title_prefix: str,
    granularity: str,
    outdir: str,
    save: bool = True,
    show: bool = False,
) -> list[tuple]:
    os.makedirs(outdir, exist_ok=True)
    codes = list(code_counters.keys())

    total_counter: Counter = Counter()
    for c in codes:
        total_counter.update(code_counters[c])
    top_patterns = [pat for pat, _ in total_counter.most_common(top_n)]
    if not top_patterns:
        print(f"  [{title_prefix}] No patterns for n={ngram_size}.")
        return []

    fig, axes = plt.subplots(len(codes), 1,
                             figsize=(max(10, top_n * 0.55), max(4, len(codes) * 3.5)),
                             squeeze=False)
    fig.suptitle(f"{title_prefix}  |  {ngram_size}-gram patterns per code  [{granularity}]",
                 fontsize=12, y=1.01)

    for ax, code in zip(axes[:, 0], codes):
        counter = code_counters[code]
        labels  = [_label_str(p, granularity) for p in top_patterns]
        counts  = [counter.get(p, 0) for p in top_patterns]
        colors  = [_bar_color(p, granularity) for p in top_patterns]
        bars = ax.barh(labels, counts, color=colors, alpha=0.82, edgecolor="white")
        ax.set_title(f"Code: {code}  (total n-grams: {sum(counter.values())})", fontsize=10)
        ax.set_xlabel("Count")
        ax.invert_yaxis()
        ax.grid(True, axis="x", color="lightgrey", linewidth=0.5)
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(bar.get_width() + max(counts) * 0.01,
                        bar.get_y() + bar.get_height() / 2,
                        str(cnt), va="center", fontsize=8)

    present_groups = {
        (n if granularity == "groups" else map_da_to_group(n))
        for pat in top_patterns for n in pat
    }
    legend_lines = [
        plt.Line2D([0], [0], color=DA_GROUP_COLORS.get(g, "#aaa"), linewidth=6,
                   label=f"{DA_GROUP_ABBREV.get(g, g)} = {g}")
        for g in DA_GROUP_ABBREV if g in present_groups
    ]
    if legend_lines:
        fig.legend(handles=legend_lines, bbox_to_anchor=(1.01, 0.5),
                   loc="center left", fontsize=8, title="DA groups")

    plt.tight_layout()
    if save:
        fname = os.path.join(outdir,
            f"{title_prefix}_{granularity}_ngram{ngram_size}_per_code.png")
        plt.savefig(fname, bbox_inches="tight")
        print(f"  Saved: {fname}")
    if show:
        plt.show()
    plt.close()
    return top_patterns


def plot_pattern_heatmap(
    code_counters: dict[str, Counter],
    top_patterns: list[tuple],
    ngram_size: int,
    title_prefix: str,
    granularity: str,
    outdir: str,
    save: bool = True,
    show: bool = False,
):
    os.makedirs(outdir, exist_ok=True)
    codes = list(code_counters.keys())
    if not codes or not top_patterns:
        return

    matrix = np.zeros((len(codes), len(top_patterns)))
    for i, code in enumerate(codes):
        for j, pat in enumerate(top_patterns):
            matrix[i, j] = code_counters[code].get(pat, 0)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    prop_matrix = matrix / row_sums
    col_labels  = [_label_str(p, granularity) for p in top_patterns]

    fig, ax = plt.subplots(figsize=(max(10, len(top_patterns) * 0.7),
                                    max(3, len(codes) * 0.6 + 1.5)))
    im = ax.imshow(prop_matrix, aspect="auto", cmap="YlOrRd", vmin=0)
    ax.set_xticks(range(len(col_labels)))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticks(range(len(codes)))
    ax.set_yticklabels(codes, fontsize=9)
    ax.set_xlabel("DA n-gram  (proportion within code)")
    fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02).set_label("Proportion", fontsize=8)
    ax.set_title(f"{title_prefix}  |  {ngram_size}-gram proportions across codes  [{granularity}]",
                 fontsize=11)
    for i in range(len(codes)):
        for j in range(len(top_patterns)):
            v = prop_matrix[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=7, color="black" if v < 0.6 else "white")
    plt.tight_layout()
    if save:
        fname = os.path.join(outdir,
            f"{title_prefix}_{granularity}_ngram{ngram_size}_heatmap.png")
        plt.savefig(fname, bbox_inches="tight")
        print(f"  Saved: {fname}")
    if show:
        plt.show()
    plt.close()


def plot_context_breakdown(
    pre_counters:   dict[str, Counter],
    block_counters: dict[str, Counter],
    post_counters:  dict[str, Counter],
    ngram_size: int,
    title_prefix: str,
    granularity: str,
    outdir: str,
    top_n: int = 10,
    save: bool = True,
    show: bool = False,
):
    os.makedirs(outdir, exist_ok=True)
    codes = list(block_counters.keys())
    if not codes:
        return

    total_block: Counter = Counter()
    for c in codes:
        total_block.update(block_counters[c])
    top_patterns = [p for p, _ in total_block.most_common(top_n)]
    if not top_patterns:
        return

    col_labels = [_label_str(p, granularity) for p in top_patterns]
    x     = np.arange(len(top_patterns))
    width = 0.25

    def _norm(counts):
        s = sum(counts)
        return [c / s if s else 0 for c in counts]

    for code in codes:
        pre_c   = [pre_counters.get(code, Counter()).get(p, 0)  for p in top_patterns]
        block_c = [block_counters[code].get(p, 0)               for p in top_patterns]
        post_c  = [post_counters.get(code, Counter()).get(p, 0) for p in top_patterns]

        fig, ax = plt.subplots(figsize=(max(10, len(top_patterns) * 0.7), 4.5))
        ax.bar(x - width, _norm(pre_c),   width, label="Pre-context",
               color="steelblue", alpha=0.8, edgecolor="white")
        ax.bar(x,          _norm(block_c), width, label="Important block",
               color="tomato",    alpha=0.8, edgecolor="white")
        ax.bar(x + width, _norm(post_c),  width, label="Post-context",
               color="seagreen",  alpha=0.8, edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Proportion within region")
        ax.set_title(f"{title_prefix}  |  Code: {code}  |  {ngram_size}-gram  "
                     f"pre / block / post  [{granularity}]", fontsize=10)
        ax.legend(fontsize=9)
        ax.grid(True, axis="y", color="lightgrey", linewidth=0.5)
        plt.tight_layout()
        if save:
            safe_code = str(code).replace("/", "-").replace(" ", "_")
            fname = os.path.join(outdir,
                f"{title_prefix}_{granularity}_ngram{ngram_size}_code{safe_code}_context.png")
            plt.savefig(fname, bbox_inches="tight")
            print(f"  Saved: {fname}")
        if show:
            plt.show()
        plt.close()


# ── output directory helpers ──────────────────────────────────────────────────

def _subdir(base: str, *parts: str) -> str:
    """Join and create a nested output directory, return the path string."""
    path = os.path.join(base, *parts)
    os.makedirs(path, exist_ok=True)
    return path


# ── graph similarity: shared helpers ─────────────────────────────────────────

def _edge_vocab(graphs: dict[str, nx.DiGraph]) -> list[tuple[str, str]]:
    """Union of all (src, dst) pairs across all graphs, sorted for stability."""
    vocab: set[tuple[str, str]] = set()
    for G in graphs.values():
        vocab.update(G.edges())
    return sorted(vocab)


def _transition_vector(G: nx.DiGraph, vocab: list[tuple[str, str]],
                       normalise: bool = True) -> np.ndarray:
    """
    Represent G as a 1-D array over *vocab*.
    Each entry is the raw edge weight (or 0 if the edge is absent).
    If normalise=True the vector is divided by its sum so it is a
    probability distribution (safe for JS divergence).
    """
    vec = np.array([G[u][v]["weight"] if G.has_edge(u, v) else 0.0
                    for u, v in vocab], dtype=float)
    if normalise:
        s = vec.sum()
        if s > 0:
            vec /= s
    return vec


# ── graph similarity: Method 1 — Jensen-Shannon divergence ───────────────────

def _js_divergence(p: np.ndarray, q: np.ndarray,
                   smoothing: float = 1e-9) -> float:
    """
    Jensen-Shannon divergence between two probability distributions p and q.

    JSD(p‖q) = 0.5 * KL(p‖m) + 0.5 * KL(q‖m)  where m = 0.5*(p+q)

    Bounded in [0, 1] (using log base 2).
    Smoothing adds a tiny constant before normalising to handle zero entries
    on edges present in one graph but absent in the other.
    """
    p = p + smoothing;  p /= p.sum()
    q = q + smoothing;  q /= q.sum()
    m = 0.5 * (p + q)
    def _kl(a, b):
        mask = a > 0
        return np.sum(a[mask] * np.log2(a[mask] / b[mask]))
    return float(0.5 * _kl(p, m) + 0.5 * _kl(q, m))


def compute_js_similarity(
    code_graphs: dict[str, nx.DiGraph],
    speaker_label: str,
    outdir: str,
    save: bool = True,
    show: bool = False,
) -> pd.DataFrame:
    """
    Compute pairwise Jensen-Shannon divergence between every pair of per-code
    transition graphs, then convert to similarity = 1 - divergence.

    Outputs
    -------
    {speaker}/similarity/js_divergence.csv     — raw divergence matrix
    {speaker}/similarity/js_similarity.png     — heatmap of similarity scores
    {speaker}/similarity/js_dendrogram.png     — hierarchical clustering

    Returns a DataFrame of pairwise similarities (codes × codes).
    """
    sim_dir = _subdir(outdir, speaker_label, "similarity")
    codes   = sorted(code_graphs.keys())
    n       = len(codes)

    if n < 2:
        print(f"  [JS] Need ≥2 codes to compare, got {n}. Skipping.")
        return pd.DataFrame()

    vocab = _edge_vocab(code_graphs)
    if not vocab:
        print(f"  [JS] No edges found across graphs. Skipping.")
        return pd.DataFrame()

    # Build distribution matrix  (n_codes × n_vocab)
    dist_matrix = np.vstack([
        _transition_vector(code_graphs[c], vocab, normalise=True)
        for c in codes
    ])

    # Pairwise JSD  →  similarity = 1 - JSD
    div_mat  = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _js_divergence(dist_matrix[i].copy(), dist_matrix[j].copy())
            div_mat[i, j] = d
            div_mat[j, i] = d
    sim_mat = 1.0 - div_mat

    df_div = pd.DataFrame(div_mat, index=codes, columns=codes)
    df_sim = pd.DataFrame(sim_mat, index=codes, columns=codes)

    # ── save CSV ──────────────────────────────────────────────────────────────
    if save:
        csv_path = os.path.join(sim_dir, "js_divergence.csv")
        df_div.to_csv(csv_path, float_format="%.4f")
        print(f"  Saved: {csv_path}")

    # ── heatmap ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(5, n * 0.7 + 1.5), max(4, n * 0.7)))
    im = ax.imshow(sim_mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(codes, fontsize=9)
    ax.set_title(f"{speaker_label} — JS transition similarity\n"
                 f"(1 = identical distribution, 0 = no overlap)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("Similarity", fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim_mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if sim_mat[i,j] > 0.35 else "white")
    plt.tight_layout()
    if save:
        p = os.path.join(sim_dir, "js_similarity.png")
        plt.savefig(p, bbox_inches="tight", dpi=150);  print(f"  Saved: {p}")
    if show:
        plt.show()
    plt.close()

    # ── dendrogram (linkage on divergence distances) ──────────────────────────
    if n >= 3:
        from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram
        from scipy.spatial.distance  import squareform

        # squareform expects a condensed distance vector
        condensed = squareform(div_mat, checks=False)
        Z = linkage(condensed, method="average")

        fig, ax = plt.subplots(figsize=(max(6, n * 0.9), 4))
        scipy_dendrogram(Z, labels=codes, ax=ax, leaf_rotation=45,
                         color_threshold=0.6 * div_mat.max())
        ax.set_title(f"{speaker_label} — JS divergence clustering", fontsize=10)
        ax.set_ylabel("JS divergence")
        ax.grid(True, axis="y", color="lightgrey", linewidth=0.5)
        plt.tight_layout()
        if save:
            p = os.path.join(sim_dir, "js_dendrogram.png")
            plt.savefig(p, bbox_inches="tight", dpi=150);  print(f"  Saved: {p}")
        if show:
            plt.show()
        plt.close()

    print(f"\n  JS similarity summary ({speaker_label}):")
    # Print most and least similar pairs
    pairs = [(codes[i], codes[j], sim_mat[i,j])
             for i in range(n) for j in range(i+1, n)]
    pairs.sort(key=lambda x: x[2], reverse=True)
    print("  Most similar pairs:")
    for a, b, s in pairs[:3]:
        print(f"    {a} <-> {b}  sim={s:.3f}  (div={1-s:.3f})")
    print("  Least similar pairs:")
    for a, b, s in pairs[-3:]:
        print(f"    {a} <-> {b}  sim={s:.3f}  (div={1-s:.3f})")

    return df_sim


# ── graph similarity: Method 2 — NetLSD (Laplacian spectral distance) ─────────

def _heat_trace(G: nx.DiGraph, time_points: np.ndarray) -> np.ndarray:
    """
    Compute the heat trace signature of *G* at each value in *time_points*.

    The heat trace at time t is:  h(t) = sum_i  exp(-t * lambda_i)
    where lambda_i are the eigenvalues of the normalised Laplacian of G.

    For directed graphs we symmetrise the adjacency matrix before computing
    the Laplacian so the eigenvalues are real-valued.  This is equivalent to
    treating the graph as undirected for the purposes of spectral comparison
    (the directed structure is already captured in which edges exist and their
    weights — the eigenspectrum measures global structural topology).

    Returns a 1-D array of length len(time_points).
    """
    nodes = sorted(G.nodes())
    n     = len(nodes)
    if n == 0:
        return np.zeros(len(time_points))

    idx = {v: i for i, v in enumerate(nodes)}

    # Weighted adjacency (symmetrised)
    A = np.zeros((n, n))
    for u, v, d in G.edges(data=True):
        i, j = idx[u], idx[v]
        A[i, j] += d["weight"]
        A[j, i] += d["weight"]   # symmetrise

    # Normalised Laplacian  L = I - D^{-1/2} A D^{-1/2}
    deg  = A.sum(axis=1)
    # Guard against isolated nodes
    with np.errstate(divide="ignore", invalid="ignore"):
        d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)
    L = np.eye(n) - D_inv_sqrt @ A @ D_inv_sqrt

    # Eigenvalues of a symmetric real matrix are always real
    eigenvalues = np.linalg.eigvalsh(L)
    eigenvalues = np.clip(eigenvalues, 0, None)   # numerical noise can give tiny negatives

    # Heat trace:  h(t) = sum_i exp(-t * lambda_i)
    # shape: (n_eigenvalues,) outer (n_times,)  →  sum over eigenvalues
    heat_trace = np.exp(-np.outer(eigenvalues, time_points)).sum(axis=0)
    return heat_trace


def compute_netlsd_similarity(
    code_graphs: dict[str, nx.DiGraph],
    speaker_label: str,
    outdir: str,
    n_time_points: int = 250,
    save: bool = True,
    show: bool = False,
) -> pd.DataFrame:
    """
    Compute pairwise NetLSD distance between per-code transition graphs.

    The heat trace of each graph is computed at *n_time_points* logarithmically
    spaced time values in [0.01, 10].  Distance between two graphs is the L2
    norm of the difference of their normalised heat traces.

    Normalisation: divide each heat trace by its value at t→0 (which equals
    the number of nodes) so graphs of different sizes are comparable.

    Outputs
    -------
    {speaker}/similarity/netlsd_distance.csv    — raw distance matrix
    {speaker}/similarity/netlsd_similarity.png  — heatmap (1/(1+dist))
    {speaker}/similarity/netlsd_dendrogram.png  — hierarchical clustering
    {speaker}/similarity/netlsd_heat_traces.png — heat trace curves per code

    Returns a DataFrame of pairwise similarities (1 / (1 + distance)).
    """
    sim_dir     = _subdir(outdir, speaker_label, "similarity")
    codes       = sorted(code_graphs.keys())
    n           = len(codes)

    if n < 2:
        print(f"  [NetLSD] Need ≥2 codes, got {n}. Skipping.")
        return pd.DataFrame()

    time_points = np.logspace(-2, 1, n_time_points)   # [0.01 … 10] log-spaced

    # Compute heat traces
    traces: dict[str, np.ndarray] = {}
    for code in codes:
        ht = _heat_trace(code_graphs[code], time_points)
        # Normalise by n_nodes so graphs of different sizes are comparable
        n_nodes = max(code_graphs[code].number_of_nodes(), 1)
        traces[code] = ht / n_nodes

    # ── heat trace plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    for code in codes:
        ax.plot(time_points, traces[code], label=str(code), linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Diffusion time  t  (log scale)")
    ax.set_ylabel("Normalised heat trace  h(t) / n")
    ax.set_title(f"{speaker_label} — NetLSD heat traces per code\n"
                 f"(curves that overlap have similar graph structure)", fontsize=10)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, color="lightgrey", linewidth=0.5)
    plt.tight_layout()
    if save:
        p = os.path.join(sim_dir, "netlsd_heat_traces.png")
        plt.savefig(p, bbox_inches="tight", dpi=150);  print(f"  Saved: {p}")
    if show:
        plt.show()
    plt.close()

    # Pairwise L2 distance between heat traces
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(traces[codes[i]] - traces[codes[j]]))
            dist_mat[i, j] = d
            dist_mat[j, i] = d

    # Similarity: 1 / (1 + distance)  — monotonically decreasing, in (0,1]
    sim_mat = 1.0 / (1.0 + dist_mat)

    df_dist = pd.DataFrame(dist_mat, index=codes, columns=codes)
    df_sim  = pd.DataFrame(sim_mat,  index=codes, columns=codes)

    # ── save CSV ──────────────────────────────────────────────────────────────
    if save:
        csv_path = os.path.join(sim_dir, "netlsd_distance.csv")
        df_dist.to_csv(csv_path, float_format="%.4f")
        print(f"  Saved: {csv_path}")

    # ── heatmap ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(5, n * 0.7 + 1.5), max(4, n * 0.7)))
    im = ax.imshow(sim_mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(codes, fontsize=9)
    ax.set_title(f"{speaker_label} — NetLSD structural similarity\n"
                 f"(1/(1+dist), higher = more similar graph topology)", fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("Similarity", fontsize=8)
    for i in range(n):
        for j in range(n):
            ax.text(j, i, f"{sim_mat[i,j]:.2f}", ha="center", va="center",
                    fontsize=8, color="black" if sim_mat[i,j] > 0.35 else "white")
    plt.tight_layout()
    if save:
        p = os.path.join(sim_dir, "netlsd_similarity.png")
        plt.savefig(p, bbox_inches="tight", dpi=150);  print(f"  Saved: {p}")
    if show:
        plt.show()
    plt.close()

    # ── dendrogram ────────────────────────────────────────────────────────────
    if n >= 3:
        from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram
        from scipy.spatial.distance  import squareform

        condensed = squareform(dist_mat, checks=False)
        Z = linkage(condensed, method="average")

        fig, ax = plt.subplots(figsize=(max(6, n * 0.9), 4))
        scipy_dendrogram(Z, labels=codes, ax=ax, leaf_rotation=45,
                         color_threshold=0.6 * dist_mat.max())
        ax.set_title(f"{speaker_label} — NetLSD spectral distance clustering", fontsize=10)
        ax.set_ylabel("L2 distance between heat traces")
        ax.grid(True, axis="y", color="lightgrey", linewidth=0.5)
        plt.tight_layout()
        if save:
            p = os.path.join(sim_dir, "netlsd_dendrogram.png")
            plt.savefig(p, bbox_inches="tight", dpi=150);  print(f"  Saved: {p}")
        if show:
            plt.show()
        plt.close()

    print(f"\n  NetLSD similarity summary ({speaker_label}):")
    pairs = [(codes[i], codes[j], sim_mat[i,j], dist_mat[i,j])
             for i in range(n) for j in range(i+1, n)]
    pairs.sort(key=lambda x: x[2], reverse=True)
    print("  Most similar pairs (structurally):")
    for a, b, s, d in pairs[:3]:
        print(f"    {a} <-> {b}  sim={s:.3f}  (L2 dist={d:.3f})")
    print("  Least similar pairs (structurally):")
    for a, b, s, d in pairs[-3:]:
        print(f"    {a} <-> {b}  sim={s:.3f}  (L2 dist={d:.3f})")

    return df_sim


# ── combined similarity report ────────────────────────────────────────────────

def run_similarity_analysis(
    code_graphs: dict[str, nx.DiGraph],
    speaker_label: str,
    outdir: str,
    save: bool = True,
    show: bool = False,
):
    """
    Run both JS divergence and NetLSD similarity on *code_graphs* and write
    a combined summary CSV to {speaker}/similarity/combined_summary.csv.

    The combined summary has one row per code-pair with both scores side by
    side, making it easy to check whether the two methods agree.
    """
    if len(code_graphs) < 2:
        print(f"  [similarity] Need ≥2 code graphs for {speaker_label}, skipping.")
        return

    print(f"\n{'─'*60}")
    print(f"  Graph similarity — {speaker_label.upper()}")
    print(f"{'─'*60}")

    df_js     = compute_js_similarity(code_graphs, speaker_label, outdir, save, show)
    df_netlsd = compute_netlsd_similarity(code_graphs, speaker_label, outdir,
                                          save=save, show=show)

    if df_js.empty or df_netlsd.empty:
        return

    # Combined summary: one row per ordered pair (i < j)
    codes = sorted(code_graphs.keys())
    rows  = []
    for i, ca in enumerate(codes):
        for j, cb in enumerate(codes):
            if j <= i:
                continue
            rows.append({
                "code_A":        ca,
                "code_B":        cb,
                "js_similarity": round(float(df_js.loc[ca, cb]), 4),
                "js_divergence": round(1 - float(df_js.loc[ca, cb]), 4),
                "netlsd_similarity": round(float(df_netlsd.loc[ca, cb]), 4),
            })

    df_summary = pd.DataFrame(rows).sort_values("js_similarity", ascending=False)

    sim_dir = _subdir(outdir, speaker_label, "similarity")
    if save:
        p = os.path.join(sim_dir, "combined_summary.csv")
        df_summary.to_csv(p, index=False)
        print(f"\n  Saved combined summary: {p}")

    print(f"\n  Combined similarity table ({speaker_label}):")
    print(df_summary.to_string(index=False))


# ── main analysis driver ──────────────────────────────────────────────────────

_NGRAM_NAMES = {1: "unigrams", 2: "bigrams", 3: "trigrams",
                4: "4-grams",  5: "5-grams"}


def _run_partition(
    sequences: list[list[str]],
    partition_label: str,
    code_map: dict[str, list[list[str]]] | None,
    granularity: str,
    ngram_sizes: list[int],
    top_n: int,
    outdir: str,
    min_edge_weight: int,
    graph_order: int,
    save: bool,
    show: bool,
) -> dict[str, nx.DiGraph]:
    """
    Shared plotting pipeline for one partition (patient-important,
    therapist-important, or non-important).

    *sequences*  — all block sequences pooled (used for the "all" graph)
    *code_map*   — per-code sequences for code-level breakdown; pass None
                   for non-important (no code dimension)

    Returns {code: DiGraph} (empty dict for non-important).
    """
    graphs_dir = _subdir(outdir, partition_label, "graphs")

    # ── transition graph: overall ─────────────────────────────────────────────
    plot_transition_graph(
        sequences, granularity,
        title=f"{partition_label} — all — DA transitions [{granularity}]",
        outdir=graphs_dir,
        fname="transition_graph_all.png",
        graph_order=graph_order,
        min_edge_weight=min_edge_weight,
        save=save, show=show,
    )

    # ── transition graphs + collect DiGraphs: per code ────────────────────────
    code_graphs: dict[str, nx.DiGraph] = {}
    if code_map:
        for code, seqs in code_map.items():
            safe_code = str(code).replace("/", "-").replace(" ", "_")
            G = plot_transition_graph(
                seqs, granularity,
                title=f"{partition_label} — code: {code} — DA transitions [{granularity}]",
                outdir=graphs_dir,
                fname=f"transition_graph_code_{safe_code}.png",
                graph_order=graph_order,
                min_edge_weight=min_edge_weight,
                save=save, show=show,
            )
            if G is not None:
                code_graphs[str(code)] = G

    # ── n-gram analyses ────────────────────────────────────────────────────────
    for ngram_size in ngram_sizes:
        ngram_name = _NGRAM_NAMES.get(ngram_size, f"{ngram_size}-grams")
        ngram_dir  = _subdir(outdir, partition_label, "ngrams", ngram_name)
        print(f"    -- {ngram_size}-gram -> {ngram_dir} --")

        if code_map:
            code_counters = {c: ngram_counter(s, ngram_size) for c, s in code_map.items()}
        else:
            # Non-important has no code dimension: treat the whole pool as one bucket
            code_counters = {"non_important": ngram_counter(sequences, ngram_size)}

        # No pre/post context for this pooled view — pass empty counters
        empty: dict[str, Counter] = {c: Counter() for c in code_counters}

        top_patterns = plot_top_patterns_per_code(
            code_counters, ngram_size, top_n,
            title_prefix=partition_label, granularity=granularity,
            outdir=ngram_dir, save=save, show=show,
        )
        if len(code_counters) > 1 and top_patterns:
            plot_pattern_heatmap(
                code_counters, top_patterns, ngram_size,
                title_prefix=partition_label, granularity=granularity,
                outdir=ngram_dir, save=save, show=show,
            )
        # Context breakdown only makes sense for important partitions
        # (non-important has no pre/post structure)
        if code_map:
            # We need the original pre/post counters — caller must supply them.
            # They are passed via the optional pre/post dicts below.
            pass

        if len(code_counters) > 1 and top_patterns:
            chi2_df = compare_codes_chi2(code_counters, top_patterns, granularity)
            if not chi2_df.empty:
                sig = chi2_df[chi2_df["significant"]]
                print(f"    Chi² (FDR) {ngram_size}-grams: "
                      f"{len(sig)}/{len(chi2_df)} patterns differ (p<0.05)")
                if not sig.empty:
                    print(sig[["pattern", "chi2", "p_fdr"]].to_string(index=False))

    return code_graphs


def run_analysis(
    combined: pd.DataFrame,
    importance_col: str,
    code_col: str,
    speaker_label: str,
    granularity: str,
    ngram_sizes: list[int],
    context_window: int,
    top_n: int,
    outdir: str,
    include_context_in_block: bool = False,
    graph_order: int = 1,
    min_edge_weight: int = 2,
    save: bool = True,
    show: bool = False,
) -> dict[str, nx.DiGraph]:
    """
    Run the full pipeline for one important-speaker partition.
    Returns {code: DiGraph} for downstream similarity analysis.

    Output directory layout
    -----------------------
    outdir/
      {speaker}/
        graphs/
        ngrams/
          bigrams/
          trigrams/
          ...
        similarity/
    """
    print(f"\n{'='*60}")
    print(f" Speaker: {speaker_label.upper()}  |  granularity: {granularity}")
    print(f"{'='*60}")

    all_blocks: list[dict] = []
    for _, grp in combined.groupby("source_file"):
        grp = grp.reset_index(drop=True)
        all_blocks.extend(
            extract_important_blocks(
                grp,
                importance_col=importance_col,
                code_col=code_col,
                granularity=granularity,
                context_window=context_window,
                include_context_in_block=include_context_in_block,
            )
        )

    if not all_blocks:
        print("  No important blocks found.")
        return {}

    print(f"  Total important blocks found: {len(all_blocks)}")

    # Explode to (block, code) rows
    rows = []
    for blk in all_blocks:
        for code in blk["codes"]:
            rows.append({
                "code":           code,
                "block_sequence": blk["block_sequence"],
                "pre_sequence":   blk["pre_sequence"],
                "post_sequence":  blk["post_sequence"],
            })
    df_blocks = pd.DataFrame(rows)
    print("\n  Block counts per code:")
    print(df_blocks["code"].value_counts().to_string())

    # Build per-code sequence lists
    code_map: dict[str, list[list[str]]] = {
        str(code): grp["block_sequence"].tolist()
        for code, grp in df_blocks.groupby("code")
    }
    all_sequences = df_blocks["block_sequence"].tolist()

    code_graphs = _run_partition(
        sequences=all_sequences,
        partition_label=speaker_label,
        code_map=code_map,
        granularity=granularity,
        ngram_sizes=ngram_sizes,
        top_n=top_n,
        outdir=outdir,
        min_edge_weight=min_edge_weight,
        graph_order=graph_order,
        save=save,
        show=show,
    )

    # Context breakdown needs pre/post — run it separately here
    for ngram_size in ngram_sizes:
        ngram_name = _NGRAM_NAMES.get(ngram_size, f"{ngram_size}-grams")
        ngram_dir  = _subdir(outdir, speaker_label, "ngrams", ngram_name)
        pre_counters  = {
            str(code): ngram_counter(grp["pre_sequence"].tolist(), ngram_size)
            for code, grp in df_blocks.groupby("code")
        }
        block_counters = {
            str(code): ngram_counter(grp["block_sequence"].tolist(), ngram_size)
            for code, grp in df_blocks.groupby("code")
        }
        post_counters = {
            str(code): ngram_counter(grp["post_sequence"].tolist(), ngram_size)
            for code, grp in df_blocks.groupby("code")
        }
        plot_context_breakdown(
            pre_counters, block_counters, post_counters,
            ngram_size=ngram_size,
            title_prefix=speaker_label, granularity=granularity,
            outdir=ngram_dir, top_n=min(top_n, 10),
            save=save, show=show,
        )

    return code_graphs


def run_nonimportant_analysis(
    combined: pd.DataFrame,
    granularity: str,
    ngram_sizes: list[int],
    context_window: int,
    top_n: int,
    outdir: str,
    include_context_in_block: bool = False,
    graph_order: int = 1,
    min_edge_weight: int = 2,
    save: bool = True,
    show: bool = False,
) -> nx.DiGraph | None:
    """
    Run the same graphs/ngrams pipeline on the non-important portion of every
    transcript — i.e. all DA rows not claimed by patient_important or
    therapist_important (and not claimed by their context windows when
    include_context_in_block is True).

    There is no per-code split for non-important turns (codes only exist on
    important blocks).  The overall transition graph is returned so it can be
    included in cross-partition similarity analysis.

    Output directory
    ----------------
    outdir/non_important/graphs/
    outdir/non_important/ngrams/{bigrams,trigrams,...}/
    """
    print(f"\n{'='*60}")
    print(f" NON-IMPORTANT partition  |  granularity: {granularity}")
    print(f"{'='*60}")

    all_sequences: list[list[str]] = []
    for _, grp in combined.groupby("source_file"):
        grp = grp.reset_index(drop=True)
        all_sequences.extend(
            extract_nonimportant_sequences(
                grp,
                granularity=granularity,
                context_window=context_window,
                include_context_in_block=include_context_in_block,
            )
        )

    total_das = sum(len(s) for s in all_sequences)
    print(f"  Non-important runs: {len(all_sequences)}  |  total DAs: {total_das}")

    if not all_sequences:
        print("  No non-important rows found.")
        return None

    _run_partition(
        sequences=all_sequences,
        partition_label="non_important",
        code_map=None,
        granularity=granularity,
        ngram_sizes=ngram_sizes,
        top_n=top_n,
        outdir=outdir,
        min_edge_weight=min_edge_weight,
        graph_order=graph_order,
        save=save,
        show=show,
    )

    # Build and return the overall transition graph for similarity comparisons
    G = build_higher_order_graph(all_sequences, order=graph_order,
                                 min_edge_weight=min_edge_weight)
    return G if G.number_of_edges() > 0 else None


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="DA pattern analysis across importance codes.")
    parser.add_argument("--dir",             required=True,
                        help="Directory of word-level daseg CSV files.")
    parser.add_argument("--granularity",     default="groups",
                        choices=["groups", "raw"],
                        help="'groups' (default): DA group labels. "
                             "'raw': individual DA class strings.")
    parser.add_argument("--ngram_sizes",     nargs="+", type=int, default=[2, 3])
    parser.add_argument("--context_window",  type=int, default=15)
    parser.add_argument("--top_n",           type=int, default=15)
    parser.add_argument("--min_edge_weight", type=int, default=2,
                        help="Prune transition graph edges below this count (default: 2).")
    parser.add_argument("--graph_order",     type=int, default=1,
                        help="Order of the Markov transition graph (default: 1). "
                             "order=1: nodes are individual DA groups, edges are "
                             "bigram transitions — the original behaviour. "
                             "order=2: nodes are DA-pair bigrams e.g. (ST,Q), edges "
                             "encode trigram sequences — ST→Q→A becomes the path "
                             "(ST,Q)->(Q,A). "
                             "order=k: nodes are k-tuples, edges encode (k+1)-grams. "
                             "Recommended range: 1–3. With --granularity raw, keep "
                             "order=1 to avoid an explosion of node combinations.")
    parser.add_argument("--include_context", action="store_true",
                        help="Absorb the ±context_window DAs around each important "
                             "block into that block's sequences (and exclude them "
                             "from non-important).  Default: context stays in "
                             "non-important pool.")
    parser.add_argument("--outdir",          default="da_pattern_output/")
    parser.add_argument("--show",            action="store_true")
    args = parser.parse_args()

    dir_path = Path(args.dir)
    if not dir_path.exists():
        raise ValueError(f"Directory not found: {args.dir}")
    os.makedirs(args.outdir, exist_ok=True)

    allowed_ext = {".csv", ".tsv", ".xlsx"}
    dfs: dict[str, pd.DataFrame] = {}
    for fp in dir_path.iterdir():
        if fp.suffix.lower() in allowed_ext:
            print(f"Loading {fp.name} ...")
            df = load_da_level(fp)
            df["source_file"] = fp.name
            dfs[fp.name] = df

    if not dfs:
        raise RuntimeError(f"No data files found in {args.dir}")

    combined = pd.concat(dfs.values(), ignore_index=True)
    print(f"\nTotal DA-level rows: {len(combined)}")

    # ── three partitions ──────────────────────────────────────────────────────
    all_speaker_graphs: dict[str, dict[str, nx.DiGraph]] = {}
    for importance_col, code_col, speaker_label in [
        ("patient_important",   "patient_code",   "patient"),
        ("therapist_important", "therapist_code", "therapist"),
    ]:
        code_graphs = run_analysis(
            combined,
            importance_col=importance_col,
            code_col=code_col,
            speaker_label=speaker_label,
            granularity=args.granularity,
            ngram_sizes=args.ngram_sizes,
            context_window=args.context_window,
            top_n=args.top_n,
            outdir=args.outdir,
            include_context_in_block=args.include_context,
            graph_order=args.graph_order,
            min_edge_weight=args.min_edge_weight,
            save=True,
            show=args.show,
        )
        all_speaker_graphs[speaker_label] = code_graphs

    nonimportant_graph = run_nonimportant_analysis(
        combined,
        granularity=args.granularity,
        ngram_sizes=args.ngram_sizes,
        context_window=args.context_window,
        top_n=args.top_n,
        outdir=args.outdir,
        include_context_in_block=args.include_context,
        graph_order=args.graph_order,
        min_edge_weight=args.min_edge_weight,
        save=True,
        show=args.show,
    )

    # ── within-speaker similarity across codes ────────────────────────────────
    for speaker_label, code_graphs in all_speaker_graphs.items():
        run_similarity_analysis(
            code_graphs,
            speaker_label=speaker_label,
            outdir=args.outdir,
            save=True,
            show=args.show,
        )

    # ── cross-partition similarity: patient vs therapist vs non-important ──────
    # Build one pooled graph per important speaker (all codes merged) to compare
    # at the partition level against non-important.
    print(f"\n{'='*60}")
    print(f" Cross-partition similarity")
    print(f"{'='*60}")

    partition_graphs: dict[str, nx.DiGraph] = {}
    for speaker_label, code_graphs in all_speaker_graphs.items():
        if not code_graphs:
            continue
        # Merge all per-code graphs into one pooled graph by summing edge weights
        G_pool = nx.DiGraph()
        for G in code_graphs.values():
            for u, v, d in G.edges(data=True):
                if G_pool.has_edge(u, v):
                    G_pool[u][v]["weight"] += d["weight"]
                else:
                    G_pool.add_edge(u, v, weight=d["weight"])
        partition_graphs[speaker_label] = G_pool

    if nonimportant_graph is not None:
        partition_graphs["non_important"] = nonimportant_graph

    cross_dir = _subdir(args.outdir, "cross_partition")
    run_similarity_analysis(
        partition_graphs,
        speaker_label="cross_partition",
        outdir=args.outdir,
        save=True,
        show=args.show,
    )

    print(f"\nDone. Outputs in: {args.outdir}")
    print("\nOutput directory layout:")
    for root, dirs, files in os.walk(args.outdir):
        depth = root.replace(args.outdir, "").count(os.sep)
        print("  " * depth + os.path.basename(root) + "/")
        for f in sorted(files):
            print("  " * (depth + 1) + f)


if __name__ == "__main__":
    main()