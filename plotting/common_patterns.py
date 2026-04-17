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

EXTENDED_DA_GROUPS: dict[str, list[str]] = {
    # canonical questions follow standard framing  e.g. "Did you like the play?"
    "canonical_questions": [
        "Wh-Question", "Yes-No-Question", "Open-Question", "Or-Clause",
    ],
    # non-canonical questions don't follow standard framing  e.g. "You liked the play?"
    "non_canonical_questions": [
        "Declarative-Yes-No-Question", "Rhetorical-Questions",
        "Backchannel-in-question-form", "Tag-Question", "Declarative-Wh-Question",
    ],
    # canonical answers follow a standard response  e.g. "Yes, I liked the play."
    "canonical_answers": [
        "Yes-answers", "No-answers",
    ],
    # non-canonical answers follow a non-standard response  e.g. "The play was ok."
    "non_canonical_answers": [
        "Affirmative-non-yes-answers", "Other-answers",
        "Negative-non-no-answers", "Dispreferred-answers", "Reject",
    ],
    "backchannel": [
        "Hold-before-answer-agreement", "Acknowledge-Backchannel",
        "Response-Acknowledgement", "Non-verbal", "Uninterpretable", "Other",
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
        "3rd-party-talk",
    ],
}

# Short display abbreviations for axis labels and graph nodes
DA_GROUP_ABBREV: dict[str, str] = {
    "canonical_questions":     "CQ",
    "non_canonical_questions": "NCQ",
    "canonical_answers":       "CA",
    "non_canonical_answers":   "NCA",
    "backchannel":             "BC",
    "statements":              "ST",
    "hedge":                   "HG",
    "social_ritual":           "SR",
    "acknowledgement":         "ACK",
    "elaboration":             "EL",
    "action":                  "ACT",
    "noise":                   "NS",
    "other":                   "OT",
}

_N = len(DA_GROUP_ABBREV)
_CMAP = plt.get_cmap("tab20")
DA_GROUP_COLORS: dict[str, str] = {
    g: mcolors.to_hex(_CMAP(i / _N)) for i, g in enumerate(DA_GROUP_ABBREV)
}

# ── run-length bucketing ──────────────────────────────────────────────────────
# Per-group thresholds: (short_max, medium_max).
# A run of length L is bucketed as:
#   short  : 1  <= L <= short_max
#   medium : short_max < L <= medium_max
#   long   : L  >  medium_max
#
# Statements are typically longer, so they get wider bins.
# Every group not listed here uses the DEFAULT_BUCKET entry.
# Edit this dict to tune thresholds without touching any other code.

RUN_LENGTH_BUCKETS: dict[str, tuple[int, int]] = {
    # group_name          : (short_max, medium_max)
    "statements"          : (2, 8),
    # all other groups use DEFAULT_BUCKET below
}
_DEFAULT_BUCKET: tuple[int, int] = (1, 3)  # short 1, medium 2-3, long 4+


def _bucket_label(base_label: str, run_length: int, granularity: str) -> str:
    """
    Return a bucketed node label like "ST_medium" or "Wh-Q_short".

    *base_label* is the plain group or raw DA string (already resolved by
    get_label before bucketing).  *run_length* is the number of consecutive
    DA rows in this contiguous region.

    For raw granularity, bucket thresholds are looked up by the *group* the
    DA belongs to, so Statement-opinion and Statement-non-opinion both use
    the "statements" thresholds.
    """
    # Determine which threshold to use
    group_key = base_label if granularity == "groups" else map_da_to_group(base_label)
    short_max, medium_max = RUN_LENGTH_BUCKETS.get(group_key, _DEFAULT_BUCKET)

    if run_length <= short_max:
        bucket = "short"
    elif run_length <= medium_max:
        bucket = "medium"
    else:
        bucket = "long"

    return f"{base_label}_{bucket}"


def _strip_bucket(label: str) -> str:
    """Remove the _short / _medium / _long suffix from a bucketed label."""
    for suffix in ("_short", "_medium", "_long"):
        if label.endswith(suffix):
            return label[: -len(suffix)]
    return label

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
    """Short string for axis labels and graph node text.
    For bucketed labels (e.g. 'statements_medium') the suffix is kept but
    the base is abbreviated: 'ST_med'.
    """
    # Detect and split bucket suffix
    bucket_suffix = ""
    for sfx, short in (("_short", "_S"), ("_medium", "_M"), ("_long", "_L")):
        if label.endswith(sfx):
            bucket_suffix = short
            label = label[: -len(sfx)]
            break

    if granularity == "groups":
        base = DA_GROUP_ABBREV.get(label, label)
    else:
        base = (label
                .replace("-Question", "-Q")
                .replace("-answers", "-A")
                .replace("Statement-", "ST-")
                .replace("Acknowledge-", "Ack-")
                .replace("Conventional-", "Conv-")
                .replace("Collaborative-", "Collab-"))
    return base + bucket_suffix


def node_color(label: str, granularity: str) -> str:
    """Return hex colour for a node, stripping any bucket suffix first."""
    base = _strip_bucket(label)
    if granularity == "groups":
        return DA_GROUP_COLORS.get(base, "#aaaaaa")
    grp = map_da_to_group(base)
    return DA_GROUP_COLORS.get(grp, "#aaaaaa")


# ── run-length analysis helpers ───────────────────────────────────────────────

def compute_node_mean_run_lengths(
    sequences: list[list[str]],
) -> dict[str, float]:
    """
    For each unique base label (stripping any bucket suffix), compute the
    mean run length across all contiguous regions in *sequences*.

    Used for Option D: encoding mean run length as node size in the
    transition graph, independent of whether bucketing is active.

    Returns {base_label: mean_run_length}.
    """
    run_totals: dict[str, list[int]] = {}
    for seq in sequences:
        if not seq:
            continue
        current = seq[0]
        count   = 1
        for item in seq[1:]:
            if item == current:
                count += 1
            else:
                base = _strip_bucket(current)
                run_totals.setdefault(base, []).append(count)
                current = item
                count   = 1
        base = _strip_bucket(current)
        run_totals.setdefault(base, []).append(count)

    return {label: float(np.mean(runs)) for label, runs in run_totals.items()}


# ── important-block extraction ────────────────────────────────────────────────

def _parse_codes(raw) -> list[str]:
    s = str(raw)
    codes = [c.strip() for c in s.split(",") if c.strip() not in ("", "nan", "NaN", "None")]
    return codes if codes else ["NA"]


def _labels_from_df(
    df: pd.DataFrame,
    granularity: str,
    bucketed_runs: bool,
) -> list[str]:
    """
    Build the per-row label list for a DA-level DataFrame.

    Plain mode (bucketed_runs=False):
        Each row gets its group or raw DA string unchanged.

    Bucketed mode (bucketed_runs=True):
        We first identify every contiguous run of identical base labels,
        measure its length, then emit a bucketed label (e.g. "ST_medium")
        for every row in that run.  This means rows that are part of a long
        statement run all get "ST_long" rather than just "ST", so downstream
        RLE compression collapses the whole run to a single "ST_long" node.
    """
    base_labels = [
        get_label(row[DA_COLUMN], row["da_group"], granularity)
        for _, row in df.iterrows()
    ]
    if not bucketed_runs:
        return base_labels

    # Two-pass: find run boundaries and lengths, then emit bucketed labels
    n      = len(base_labels)
    result = [""] * n
    i      = 0
    while i < n:
        j = i + 1
        while j < n and base_labels[j] == base_labels[i]:
            j += 1
        run_len  = j - i
        bucketed = _bucket_label(base_labels[i], run_len, granularity)
        for k in range(i, j):
            result[k] = bucketed
        i = j
    return result


def extract_important_blocks(
    df: pd.DataFrame,
    importance_col: str,
    code_col: str,
    granularity: str,
    context_window: int = 15,
    include_context_in_block: bool = False,
    bucketed_runs: bool = False,
) -> list[dict]:
    """
    Find contiguous runs where importance_col == 1.

    When bucketed_runs=True each label is suffixed with its run-length
    bucket (_short / _medium / _long) so downstream RLE compression
    produces nodes like "ST_long" instead of plain "ST".

    Returns list of dicts with keys:
      codes, block_indices, pre_indices, post_indices,
      full_sequence, block_sequence, pre_sequence, post_sequence
    """
    importance = df[importance_col].fillna(0).astype(int)
    labels     = _labels_from_df(df, granularity, bucketed_runs)
    n          = len(df)
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
    bucketed_runs: bool = False,
) -> list[list[str]]:
    """
    Return DA sequences from rows not claimed by any important partition.
    When bucketed_runs=True labels include run-length suffixes.
    """
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
                claimed[i:j] = True
                if include_context_in_block:
                    pre_start = max(0, i - context_window)
                    post_end  = min(n, j + context_window)
                    claimed[pre_start:i] = True
                    claimed[j:post_end]  = True
                i = j
            else:
                i += 1

    labels = _labels_from_df(df, granularity, bucketed_runs)

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
    node_run_lengths: dict[str, float] | None = None,
    save: bool = True,
    show: bool = False,
) -> nx.DiGraph | None:
    """
    Directed transition graph over RLE-compressed sequences.

    node_run_lengths (Option D)
        Optional dict mapping base label -> mean run length (from
        compute_node_mean_run_lengths).  When provided, node SIZE encodes
        mean run length rather than outgoing edge weight, giving an immediate
        visual answer to "which groups tend to appear in long runs?".
        Node colour still encodes DA group.  Works in both plain and bucketed
        modes: in bucketed mode the node "ST_medium" looks up "ST" in the dict.
    
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

    # Option D: node size encodes mean run length when supplied;
    # otherwise falls back to total outgoing edge weight (original behaviour).
    if node_run_lengths:
        max_rl = max(node_run_lengths.values()) if node_run_lengths else 1.0
        node_sizes = []
        for node in G.nodes():
            base = _strip_bucket(node[-1] if isinstance(node, tuple) else node)
            rl   = node_run_lengths.get(base, 1.0)
            node_sizes.append(400 + 2800 * (rl / max(max_rl, 1.0)))
    else:
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
    if node_run_lengths:
        legend_handles.append(
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#888888", markersize=6,
                       label="node size = mean run length")
        )
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


# ── graph similarity: Method 3 — Magnetic Laplacian spectral distance ─────────

def _magnetic_heat_trace(
    G: nx.DiGraph,
    time_points: np.ndarray,
    q: float = 0.25,
) -> np.ndarray:
    """
    Compute the heat trace of the magnetic Laplacian of *G*.

    The magnetic Laplacian L^q is a Hermitian complex matrix that encodes
    both the undirected connectivity of G *and* the direction of each edge
    via a complex phase factor exp(i·2π·q·s_uv), where s_uv = +1 if the
    directed edge goes u→v and -1 for v→u.

    Construction
    ------------
    For each pair (u, v) where at least one directed edge exists:

        w_sym  = (w_uv + w_vu) / 2        # symmetric weight
        s_uv   = (w_uv - w_vu) / (w_uv + w_vu + ε)   # net directionality in [-1,1]

    The magnetic adjacency entry:
        H_uv = w_sym · exp( i·2π·q·s_uv )
        H_vu = conj(H_uv)

    The degree matrix D_ii = sum_j w_sym_ij (real, same as undirected degree).

    L^q = D - Re(H)  … for the eigenvalue computation we use the real
    part of the normalised version, which keeps eigenvalues real and ≥ 0
    while still encoding direction via the phase-weighted degree.

    Why this works
    --------------
    When w_uv = w_vu (undirected edge) the phase is 0 and L^q reduces to
    the standard normalised Laplacian — NetLSD and magnetic-LSD give the
    same answer.  When edges are strongly asymmetric the phase shifts the
    eigenvalues away from those of the symmetrised graph, so two graphs
    that look identical after symmetrisation but have opposite directionality
    (e.g. Q→ST everywhere vs ST→Q everywhere) will produce different spectra.

    The parameter q controls phase sensitivity; q=0.25 is the canonical
    choice (quarter-turn per unit of net directionality).  q=0 recovers
    the undirected NetLSD.

    Returns a 1-D real array of length len(time_points).
    """
    nodes = sorted(G.nodes())
    n     = len(nodes)
    if n == 0:
        return np.zeros(len(time_points))

    idx = {v: i for i, v in enumerate(nodes)}

    # Build symmetric weight and net-directionality matrices
    W_sym  = np.zeros((n, n))   # (w_uv + w_vu) / 2
    S      = np.zeros((n, n))   # net directionality  (w_uv - w_vu) / (w_uv + w_vu)

    for u, v, d in G.edges(data=True):
        i, j   = idx[u], idx[v]
        w_fwd  = d["weight"]
        w_bwd  = G[v][u]["weight"] if G.has_edge(v, u) else 0.0
        total  = w_fwd + w_bwd
        W_sym[i, j]  = total / 2.0
        W_sym[j, i]  = total / 2.0
        if total > 0:
            S[i, j]  =  (w_fwd - w_bwd) / total   # +1 if purely fwd, -1 if purely bwd
            S[j, i]  = -S[i, j]                    # antisymmetric

    # Magnetic adjacency  H_ij = W_sym_ij * exp(i * 2π * q * S_ij)
    phase = 2.0 * np.pi * q * S
    H = W_sym * (np.cos(phase) + 1j * np.sin(phase))   # complex Hermitian

    # Degree matrix (real, from symmetric weights)
    deg       = W_sym.sum(axis=1)
    d_inv_sqrt = np.where(deg > 0, 1.0 / np.sqrt(deg), 0.0)
    D_inv_sqrt = np.diag(d_inv_sqrt)

    # Normalised magnetic Laplacian  L^q = I - D^{-1/2} H D^{-1/2}
    # L^q is Hermitian → eigenvalues are real
    L_mag = np.eye(n) - D_inv_sqrt @ H @ D_inv_sqrt

    # Eigenvalues of a Hermitian matrix are real
    eigenvalues = np.linalg.eigvalsh(L_mag)
    eigenvalues = np.clip(eigenvalues.real, 0, None)

    heat_trace = np.exp(-np.outer(eigenvalues, time_points)).sum(axis=0)
    return heat_trace.real


def compute_magnetic_similarity(
    code_graphs: dict[str, nx.DiGraph],
    speaker_label: str,
    outdir: str,
    q: float = 0.25,
    n_time_points: int = 250,
    save: bool = True,
    show: bool = False,
) -> pd.DataFrame:
    """
    Compute pairwise magnetic-Laplacian spectral distance between per-code
    transition graphs.

    Unlike NetLSD (which symmetrises the adjacency matrix and loses all
    directional information), the magnetic Laplacian encodes edge direction
    as a complex phase.  Two graphs that look identical after symmetrisation
    but have opposite dominant directions — e.g. one where statements mostly
    lead to questions vs one where questions mostly lead to statements — will
    produce different spectra and therefore different distances here.

    The q parameter (default 0.25) controls how much weight the phase
    component receives.  q=0 reduces to standard NetLSD; q=0.5 gives
    maximum directional sensitivity.

    Outputs
    -------
    {speaker}/similarity/magnetic_distance.csv       — raw distance matrix
    {speaker}/similarity/magnetic_similarity.png     — heatmap (1/(1+dist))
    {speaker}/similarity/magnetic_dendrogram.png     — hierarchical clustering
    {speaker}/similarity/magnetic_heat_traces.png    — heat trace curves per code

    Returns a DataFrame of pairwise similarities (1 / (1 + distance)).
    """
    sim_dir     = _subdir(outdir, speaker_label, "similarity")
    codes       = sorted(code_graphs.keys())
    n           = len(codes)

    if n < 2:
        print(f"  [Magnetic] Need ≥2 codes, got {n}. Skipping.")
        return pd.DataFrame()

    time_points = np.logspace(-2, 1, n_time_points)

    traces: dict[str, np.ndarray] = {}
    for code in codes:
        ht      = _magnetic_heat_trace(code_graphs[code], time_points, q=q)
        n_nodes = max(code_graphs[code].number_of_nodes(), 1)
        traces[code] = ht / n_nodes

    # ── heat trace plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    for code in codes:
        ax.plot(time_points, traces[code], label=str(code), linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Diffusion time  t  (log scale)")
    ax.set_ylabel("Normalised magnetic heat trace  h(t) / n")
    ax.set_title(f"{speaker_label} — Magnetic Laplacian heat traces per code  (q={q})\n"
                 f"(encodes directed structure; overlap = similar directed topology)",
                 fontsize=10)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, color="lightgrey", linewidth=0.5)
    plt.tight_layout()
    if save:
        p = os.path.join(sim_dir, "magnetic_heat_traces.png")
        plt.savefig(p, bbox_inches="tight", dpi=150);  print(f"  Saved: {p}")
    if show:
        plt.show()
    plt.close()

    # Pairwise L2 distance
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(traces[codes[i]] - traces[codes[j]]))
            dist_mat[i, j] = d
            dist_mat[j, i] = d

    sim_mat = 1.0 / (1.0 + dist_mat)
    df_dist = pd.DataFrame(dist_mat, index=codes, columns=codes)
    df_sim  = pd.DataFrame(sim_mat,  index=codes, columns=codes)

    # ── save CSV ──────────────────────────────────────────────────────────────
    if save:
        p = os.path.join(sim_dir, "magnetic_distance.csv")
        df_dist.to_csv(p, float_format="%.4f")
        print(f"  Saved: {p}")

    # ── heatmap ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(5, n * 0.7 + 1.5), max(4, n * 0.7)))
    im = ax.imshow(sim_mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(codes, fontsize=9)
    ax.set_title(f"{speaker_label} — Magnetic Laplacian similarity  (q={q})\n"
                 f"(directed topology: higher = more similar directed flow)",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("Similarity", fontsize=8)
    for i in range(n):
        for j in range(n):
            v = sim_mat[i, j]
            if i != j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if v > 0.35 else "white")
    plt.tight_layout()
    if save:
        p = os.path.join(sim_dir, "magnetic_similarity.png")
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
        ax.set_title(f"{speaker_label} — Magnetic Laplacian clustering  (q={q})",
                     fontsize=10)
        ax.set_ylabel("L2 distance between magnetic heat traces")
        ax.grid(True, axis="y", color="lightgrey", linewidth=0.5)
        plt.tight_layout()
        if save:
            p = os.path.join(sim_dir, "magnetic_dendrogram.png")
            plt.savefig(p, bbox_inches="tight", dpi=150);  print(f"  Saved: {p}")
        if show:
            plt.show()
        plt.close()

    print(f"\n  Magnetic similarity summary ({speaker_label}, q={q}):")
    pairs = [(codes[i], codes[j], sim_mat[i, j], dist_mat[i, j])
             for i in range(n) for j in range(i + 1, n)]
    pairs.sort(key=lambda x: x[2], reverse=True)
    print("  Most similar pairs (directed topology):")
    for a, b, s, d in pairs[:3]:
        print(f"    {a} <-> {b}  sim={s:.3f}  (L2 dist={d:.3f})")
    print("  Least similar pairs (directed topology):")
    for a, b, s, d in pairs[-3:]:
        print(f"    {a} <-> {b}  sim={s:.3f}  (L2 dist={d:.3f})")

    return df_sim


# ── graph similarity: Method 4 — Hashimoto (non-backtracking) operator ────────

def _hashimoto_heat_trace(
    G: nx.DiGraph,
    time_points: np.ndarray,
) -> np.ndarray:
    """
    Compute the heat trace of the Hashimoto (non-backtracking) operator of G.

    The non-backtracking operator B is defined on the set of *directed edges*
    of G.  For each directed edge e = (u→v), B maps it to all outgoing edges
    f = (v→w) where w ≠ u — i.e. walks that don't immediately reverse.

    This has two key properties that make it useful here:

    1.  It is intrinsically directed — B(u→v) and B(v→u) are completely
        different rows of the matrix, so asymmetric edge weights naturally
        produce different spectra.  Unlike the magnetic Laplacian, no
        symmetrisation or phase parameter is needed; directionality is baked
        in structurally.

    2.  It eliminates backtracking artefacts.  Standard random-walk Laplacians
        double-count short back-and-forth cycles; the non-backtracking walk
        is blind to those, so the spectrum better reflects longer-range
        conversational flow patterns.

    Construction
    ------------
    Nodes of B are directed edges of G: each (u, v) with w_uv > 0 becomes
    a row/column index.  Entry B[(u→v), (v→w)] = w_vw  for w ≠ u, else 0.
    B is a real (but generally non-symmetric) matrix of size |E| × |E|.

    For the heat trace we use the *real parts* of the eigenvalues of B,
    which are guaranteed real when B is symmetrised via (B + B^T)/2.
    The full complex eigendecomposition would be expensive and the imaginary
    parts carry less structural signal; the real part of the spectrum still
    differentiates directed graphs that the magnetic Laplacian misses when
    phase cancellation occurs.

    Heat trace:  h(t) = Σ_i exp(-t · |λ_i|)   (absolute value stabilises
    negative eigenvalues that arise from directed cycles).

    Normalised by number of edges so graphs of different sizes are comparable.

    Returns a 1-D real array of length len(time_points).
    """
    nodes  = sorted(G.nodes())
    # Collect directed edges with positive weight
    edges  = [(u, v, G[u][v]["weight"]) for u, v in G.edges()
              if G[u][v]["weight"] > 0]
    m      = len(edges)

    if m == 0:
        return np.zeros(len(time_points))

    # Index edges
    edge_idx = {(u, v): i for i, (u, v, _) in enumerate(edges)}

    # Build B as a dense matrix (fine for the graph sizes we have here;
    # for very large raw-granularity graphs this could be made sparse)
    B = np.zeros((m, m))
    for i, (u, v, _) in enumerate(edges):
        for w in G.successors(v):
            if w == u:
                continue                    # no backtracking
            if (v, w) in edge_idx:
                j = edge_idx[(v, w)]
                B[i, j] = G[v][w]["weight"]

    # Symmetrise to get real eigenvalues cheaply
    B_sym = (B + B.T) / 2.0
    eigenvalues = np.linalg.eigvalsh(B_sym)

    # Heat trace using |λ| for stability
    heat_trace = np.exp(-np.outer(np.abs(eigenvalues), time_points)).sum(axis=0)
    return heat_trace


def compute_hashimoto_similarity(
    code_graphs: dict[str, nx.DiGraph],
    speaker_label: str,
    outdir: str,
    n_time_points: int = 250,
    save: bool = True,
    show: bool = False,
) -> pd.DataFrame:
    """
    Compute pairwise Hashimoto (non-backtracking) spectral distance between
    per-code transition graphs.

    Compared to the other three methods:

    JS divergence      — edge weight distribution (no topology, no direction)
    NetLSD             — undirected topology (loses direction entirely)
    Magnetic Laplacian — directed topology via phase encoding (needs q tuning)
    Hashimoto          — directed topology via non-backtracking walks
                         (no free parameters; structurally captures direction)

    The Hashimoto operator is defined on *edges* rather than nodes, so its
    matrix size is |E|×|E|.  For typical per-code transition graphs with
    ≤50 edges this is fast.  For large raw-granularity graphs with many
    edges, runtime increases quadratically — prefer groups granularity.

    Outputs
    -------
    {speaker}/similarity/hashimoto_distance.csv
    {speaker}/similarity/hashimoto_similarity.png
    {speaker}/similarity/hashimoto_dendrogram.png
    {speaker}/similarity/hashimoto_heat_traces.png

    Returns a DataFrame of pairwise similarities (1 / (1 + distance)).
    """
    sim_dir = _subdir(outdir, speaker_label, "similarity")
    codes   = sorted(code_graphs.keys())
    n       = len(codes)

    if n < 2:
        print(f"  [Hashimoto] Need ≥2 codes, got {n}. Skipping.")
        return pd.DataFrame()

    time_points = np.logspace(-2, 1, n_time_points)

    traces: dict[str, np.ndarray] = {}
    for code in codes:
        ht     = _hashimoto_heat_trace(code_graphs[code], time_points)
        n_edges = max(code_graphs[code].number_of_edges(), 1)
        traces[code] = ht / n_edges   # normalise by edge count (operator lives on edges)

    # ── heat trace plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4))
    for code in codes:
        ax.plot(time_points, traces[code], label=str(code), linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Diffusion time  t  (log scale)")
    ax.set_ylabel("Normalised Hashimoto heat trace  h(t) / |E|")
    ax.set_title(f"{speaker_label} — Hashimoto (non-backtracking) heat traces\n"
                 f"(directed; no backtracking artefacts; overlap = similar flow)",
                 fontsize=10)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, color="lightgrey", linewidth=0.5)
    plt.tight_layout()
    if save:
        p = os.path.join(sim_dir, "hashimoto_heat_traces.png")
        plt.savefig(p, bbox_inches="tight", dpi=150);  print(f"  Saved: {p}")
    if show:
        plt.show()
    plt.close()

    # Pairwise L2 distance
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(traces[codes[i]] - traces[codes[j]]))
            dist_mat[i, j] = d
            dist_mat[j, i] = d

    sim_mat = 1.0 / (1.0 + dist_mat)
    df_dist = pd.DataFrame(dist_mat, index=codes, columns=codes)
    df_sim  = pd.DataFrame(sim_mat,  index=codes, columns=codes)

    if save:
        p = os.path.join(sim_dir, "hashimoto_distance.csv")
        df_dist.to_csv(p, float_format="%.4f")
        print(f"  Saved: {p}")

    # ── heatmap ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(max(5, n * 0.7 + 1.5), max(4, n * 0.7)))
    im = ax.imshow(sim_mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(n)); ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n)); ax.set_yticklabels(codes, fontsize=9)
    ax.set_title(f"{speaker_label} — Hashimoto similarity\n"
                 f"(non-backtracking directed topology; higher = more similar)",
                 fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("Similarity", fontsize=8)
    for i in range(n):
        for j in range(n):
            v = sim_mat[i, j]
            if i != j:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if v > 0.35 else "white")
    plt.tight_layout()
    if save:
        p = os.path.join(sim_dir, "hashimoto_similarity.png")
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
        ax.set_title(f"{speaker_label} — Hashimoto clustering", fontsize=10)
        ax.set_ylabel("L2 distance between Hashimoto heat traces")
        ax.grid(True, axis="y", color="lightgrey", linewidth=0.5)
        plt.tight_layout()
        if save:
            p = os.path.join(sim_dir, "hashimoto_dendrogram.png")
            plt.savefig(p, bbox_inches="tight", dpi=150);  print(f"  Saved: {p}")
        if show:
            plt.show()
        plt.close()

    print(f"\n  Hashimoto similarity summary ({speaker_label}):")
    pairs = [(codes[i], codes[j], sim_mat[i, j], dist_mat[i, j])
             for i in range(n) for j in range(i + 1, n)]
    pairs.sort(key=lambda x: x[2], reverse=True)
    print("  Most similar pairs (non-backtracking directed topology):")
    for a, b, s, d in pairs[:3]:
        print(f"    {a} <-> {b}  sim={s:.3f}  (L2 dist={d:.3f})")
    print("  Least similar pairs:")
    for a, b, s, d in pairs[-3:]:
        print(f"    {a} <-> {b}  sim={s:.3f}  (L2 dist={d:.3f})")

    return df_sim



def run_similarity_analysis(
    code_graphs: dict[str, nx.DiGraph],
    speaker_label: str,
    outdir: str,
    save: bool = True,
    show: bool = False,
):
    """
    Run JS divergence, NetLSD, Magnetic Laplacian, and Hashimoto (non-
    backtracking) similarity on *code_graphs* and write a combined summary CSV.

    What each method captures
    -------------------------
    JS divergence      — edge weight distribution overlap.  Similar if the
                         same transitions appear with similar relative
                         frequency.  Ignores topology; sensitive to vocab.

    NetLSD             — undirected structural topology via symmetrised
                         Laplacian eigenspectrum.  Similar if graphs have
                         similar degree distributions, clustering, path
                         lengths.  Loses all directional information.

    Magnetic Laplacian — directed topology via complex phase encoding (q=0.25).
                         Retains direction; can show phase cancellation on
                         highly symmetric graphs.

    Hashimoto          — directed topology via non-backtracking walks on edges.
                         No free parameters; naturally directed; eliminates
                         short back-and-forth cycle artefacts.

    Agreement patterns to look for
    --------------------------------
    both_similar                  → same transitions, same directed topology
    both_different                → genuinely different in every sense
    same_transitions_diff_direction → JS high, directed methods low
    diff_transitions_same_topology → JS low, directed methods high
    directed_methods_disagree     → magnetic and Hashimoto differ by >0.25;
                                     trust Hashimoto more in this case
    mixed                         → no clean pattern
    """
    if len(code_graphs) < 2:
        print(f"  [similarity] Need ≥2 code graphs for {speaker_label}, skipping.")
        return

    print(f"\n{'─'*60}")
    print(f"  Graph similarity — {speaker_label.upper()}")
    print(f"{'─'*60}")

    df_js       = compute_js_similarity(code_graphs, speaker_label, outdir, save, show)
    df_netlsd   = compute_netlsd_similarity(code_graphs, speaker_label, outdir,
                                            save=save, show=show)
    df_magnetic = compute_magnetic_similarity(code_graphs, speaker_label, outdir,
                                              save=save, show=show)
    df_hashimoto = compute_hashimoto_similarity(code_graphs, speaker_label, outdir,
                                                save=save, show=show)

    # Need at least JS and one spectral method to produce a summary
    if df_js.empty:
        return

    codes = sorted(code_graphs.keys())
    rows  = []
    for i, ca in enumerate(codes):
        for j, cb in enumerate(codes):
            if j <= i:
                continue
            row = {
                "code_A":        ca,
                "code_B":        cb,
                "js_similarity": round(float(df_js.loc[ca, cb]), 4),
                "js_divergence": round(1 - float(df_js.loc[ca, cb]), 4),
            }
            if not df_netlsd.empty:
                row["netlsd_similarity"] = round(float(df_netlsd.loc[ca, cb]), 4)
            if not df_magnetic.empty:
                row["magnetic_similarity"] = round(float(df_magnetic.loc[ca, cb]), 4)
            if not df_hashimoto.empty:
                row["hashimoto_similarity"] = round(float(df_hashimoto.loc[ca, cb]), 4)

            # Agreement flag across all four measures
            js_s   = row["js_similarity"]
            mag_s  = row.get("magnetic_similarity")
            hsh_s  = row.get("hashimoto_similarity")
            # Use directed methods (magnetic + hashimoto) when both available,
            # fall back to magnetic alone, then just JS
            directed_scores = [s for s in [mag_s, hsh_s] if s is not None]
            dir_mean = float(np.mean(directed_scores)) if directed_scores else None

            if dir_mean is not None:
                if js_s > 0.7 and dir_mean > 0.7:
                    row["agreement"] = "both_similar"
                elif js_s < 0.4 and dir_mean < 0.4:
                    row["agreement"] = "both_different"
                elif js_s > 0.6 and dir_mean < 0.5:
                    row["agreement"] = "same_transitions_diff_direction"
                elif js_s < 0.5 and dir_mean > 0.6:
                    row["agreement"] = "diff_transitions_same_topology"
                elif mag_s is not None and hsh_s is not None and abs(mag_s - hsh_s) > 0.25:
                    # Magnetic and Hashimoto disagree with each other — phase
                    # cancellation in magnetic may be masking real differences
                    row["agreement"] = "directed_methods_disagree"
                else:
                    row["agreement"] = "mixed"
            else:
                row["agreement"] = "js_only"
            rows.append(row)

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


def plot_run_length_bin_counts(
    sequences: list[list[str]],
    granularity: str,
    title_prefix: str,
    outdir: str,
    code_label: str = "all",
    save: bool = True,
    show: bool = False,
):
    """
    For each base DA label, count how many contiguous runs fell into each
    bucket (short / medium / long) and plot a grouped bar chart.

    Only meaningful when bucketed_runs=True — the labels in *sequences* must
    already carry bucket suffixes (e.g. "statements_medium").  Rows with no
    suffix (plain labels) are counted under an implicit "short" bucket so the
    plot still renders gracefully if called with un-bucketed sequences.

    Saved to:  {outdir}/{code_label}_run_length_bins.png
    """
    os.makedirs(outdir, exist_ok=True)

    # Count occurrences of each (base, bucket) pair after RLE compression
    # (each contiguous run = one occurrence of its bucketed label)
    bin_counts: dict[str, dict[str, int]] = {}   # base -> {short/medium/long -> count}
    buckets_seen: set[str] = set()

    for seq in sequences:
        compressed = rle_compress(seq)
        for label in compressed:
            base   = _strip_bucket(label)
            bucket = "short"   # default if no suffix
            for sfx in ("_short", "_medium", "_long"):
                if label.endswith(sfx):
                    bucket = sfx.lstrip("_")
                    break
            buckets_seen.add(bucket)
            bin_counts.setdefault(base, {})
            bin_counts[base][bucket] = bin_counts[base].get(bucket, 0) + 1

    if not bin_counts:
        return

    # Order bases by total count descending, buckets in logical order
    bucket_order = [b for b in ("short", "medium", "long") if b in buckets_seen]
    bases        = sorted(bin_counts, key=lambda b: sum(bin_counts[b].values()),
                          reverse=True)

    x     = np.arange(len(bases))
    n_b   = len(bucket_order)
    width = 0.8 / max(n_b, 1)

    # Colour each bucket with a shade of the base group's colour
    bucket_shade = {"short": 0.55, "medium": 0.80, "long": 1.0}
    bucket_hatch = {"short": "",   "medium": "//",  "long": "xx"}

    fig, ax = plt.subplots(figsize=(max(10, len(bases) * 0.7), 5))

    for bi, bucket in enumerate(bucket_order):
        counts = [bin_counts[base].get(bucket, 0) for base in bases]
        colors = []
        for base in bases:
            hex_c = node_color(base, granularity)
            # Darken/lighten by blending with white (lighter) or black (darker)
            rgb   = mcolors.to_rgb(hex_c)
            shade = bucket_shade[bucket]
            blended = tuple(c * shade + (1 - shade) * 0.95 for c in rgb)
            colors.append(blended)

        offsets = (bi - (n_b - 1) / 2) * width
        bars = ax.bar(
            x + offsets, counts, width,
            label=bucket,
            color=colors,
            hatch=bucket_hatch[bucket],
            edgecolor="white",
            alpha=0.9,
        )
        for bar, cnt in zip(bars, counts):
            if cnt > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + ax.get_ylim()[1] * 0.005,
                    str(cnt),
                    ha="center", va="bottom", fontsize=7, rotation=45,
                )

    ax.set_xticks(x)
    ax.set_xticklabels(
        [abbrev(b, granularity) for b in bases],
        rotation=45, ha="right", fontsize=9,
    )
    ax.set_ylabel("Number of contiguous runs")
    ax.set_title(
        f"{title_prefix}  |  {code_label}  —  run-length bin counts per DA\n"
        f"(after RLE compression; each bar = one contiguous run)",
        fontsize=10,
    )
    ax.legend(title="Bin", fontsize=9)
    ax.grid(True, axis="y", color="lightgrey", linewidth=0.5)

    plt.tight_layout()
    if save:
        safe = code_label.replace("/", "-").replace(" ", "_")
        fpath = os.path.join(outdir, f"{safe}_run_length_bins.png")
        plt.savefig(fpath, bbox_inches="tight", dpi=150)
        print(f"  Saved: {fpath}")
    if show:
        plt.show()
    plt.close()


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
    bucketed_runs: bool,
    save: bool,
    show: bool,
) -> dict[str, nx.DiGraph]:
    """
    Shared plotting pipeline for one partition.

    bucketed_runs controls whether sequences already contain bucketed labels
    (e.g. "ST_medium").  When True, node_run_lengths is NOT passed to
    plot_transition_graph — the bucket suffix already encodes length in the
    node label itself (Option B).  When False, node_run_lengths IS computed
    and passed so node size encodes mean run length visually (Option D).
    Both options are therefore automatically active based on this single flag.
    """
    graphs_dir = _subdir(outdir, partition_label, "graphs")

    # Option D: compute mean run lengths from raw sequences for node sizing.
    # Only used when bucketed_runs=False (when True, the bucket suffix in the
    # node label already communicates run length, so sizing would be redundant).
    node_run_lengths = (
        None if bucketed_runs
        else compute_node_mean_run_lengths(sequences)
    )

    # ── transition graph: overall ─────────────────────────────────────────────
    plot_transition_graph(
        sequences, granularity,
        title=f"{partition_label} — all — DA transitions [{granularity}]",
        outdir=graphs_dir,
        fname="transition_graph_all.png",
        graph_order=graph_order,
        min_edge_weight=min_edge_weight,
        node_run_lengths=node_run_lengths,
        save=save, show=show,
    )

    # ── transition graphs + collect DiGraphs: per code ────────────────────────
    code_graphs: dict[str, nx.DiGraph] = {}
    if code_map:
        for code, seqs in code_map.items():
            safe_code = str(code).replace("/", "-").replace(" ", "_")
            code_rl = (
                None if bucketed_runs
                else compute_node_mean_run_lengths(seqs)
            )
            G = plot_transition_graph(
                seqs, granularity,
                title=f"{partition_label} — code: {code} — DA transitions [{granularity}]",
                outdir=graphs_dir,
                fname=f"transition_graph_code_{safe_code}.png",
                graph_order=graph_order,
                min_edge_weight=min_edge_weight,
                node_run_lengths=code_rl,
                save=save, show=show,
            )
            if G is not None:
                code_graphs[str(code)] = G

    # ── n-gram analyses ────────────────────────────────────────────────────────
    # ── run-length bin counts (only when bucketed_runs is active) ─────────────
    if bucketed_runs:
        plot_run_length_bin_counts(
            sequences, granularity,
            title_prefix=partition_label,
            outdir=graphs_dir,
            code_label="all",
            save=save, show=show,
        )
        if code_map:
            for code, seqs in code_map.items():
                plot_run_length_bin_counts(
                    seqs, granularity,
                    title_prefix=partition_label,
                    outdir=graphs_dir,
                    code_label=str(code),
                    save=save, show=show,
                )

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
    bucketed_runs: bool = False,
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
                bucketed_runs=bucketed_runs,
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
        bucketed_runs=bucketed_runs,
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
    bucketed_runs: bool = False,
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
                bucketed_runs=bucketed_runs,
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
        bucketed_runs=bucketed_runs,
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
    parser.add_argument("--bucketed_runs",   action="store_true",
                        help="Option B: suffix each DA label with its run-length "
                             "bucket (_short/_medium/_long) before building graphs "
                             "and n-grams, so e.g. a long statement run becomes "
                             "'statements_long'. Thresholds per group are defined in "
                             "RUN_LENGTH_BUCKETS at the top of the file. "
                             "When this flag is NOT set, Option D is active instead: "
                             "node SIZE in the transition graph encodes mean run "
                             "length visually without changing the node labels.")
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
            bucketed_runs=args.bucketed_runs,
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
        bucketed_runs=args.bucketed_runs,
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