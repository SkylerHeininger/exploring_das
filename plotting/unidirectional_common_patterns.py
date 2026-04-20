"""
path_graph_da.py

Builds dual-node directed path graphs from DA sequences, where each DA group
gets exactly two nodes (DA_a, DA_b).  This avoids the single-node collapse
problem of standard transition graphs a sequence ST→Q→ST becomes a genuine
3-node path ST_a→Q_a→ST_b rather than a loop ST→Q→ST.

Node assignment rules
---------------------
1. Each sequence starts fresh: whatever DA it begins with maps to DA_a.
2. Within a sequence, each DA alternates between its _a and _b node:
   first visit → _a, second → _b, third → _a, …
3. No antiparallel edges: if src→dst already exists as an edge in the graph,
   and the new transition would create dst→src (reversing it), the destination
   is bumped to its alternate (dst_a ↔ dst_b) to avoid the reversal.
   The source node is fixed by the sequence-position alternation.
4. All sequences (important blocks or non-important chunks) reset to _a at
   their start, so _a nodes are always "entry" nodes for their DA group.

Similarity
----------
The same four methods from analyze_da_patterns.py are reused:
  JS divergence, NetLSD, Magnetic Laplacian, Hashimoto.

Usage
-----
python path_graph_da.py \\
    --dir /path/to/csv_dir \\
    --granularity groups \\
    --context_window 15 \\
    --outdir path_graph_output/

Drop alongside analyze_da_patterns.py (imported as common_patterns) and
graph_file_da.py.
Requires: pandas, numpy, matplotlib, scipy, statsmodels, networkx
"""

from __future__ import annotations

import argparse
import os
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram
from scipy.spatial.distance import squareform

# ── imports from existing script ─────────────────────────────────────────────
from plotting.common_patterns import (
    DA_COLUMN,
    EXTENDED_DA_GROUPS,
    DA_GROUP_ABBREV,
    DA_GROUP_COLORS,
    load_da_level,
    map_da_to_group,
    get_label,
    rle_compress,
    _parse_codes,
    _subdir,
    _heat_trace,
    _magnetic_heat_trace,
    _hashimoto_heat_trace,
    _js_divergence,
    _transition_vector,
    _edge_vocab,
)

# ── node naming ───────────────────────────────────────────────────────────────

_SLOTS = ("a", "b")


def _node(da: str, slot: int) -> str:
    """Return the node name for a DA group and slot index (0=a, 1=b)."""
    return f"{da}_{_SLOTS[slot % 2]}"


def _base(node: str) -> str:
    """Strip the _a / _b suffix to get the base DA label."""
    if node.endswith("_a") or node.endswith("_b"):
        return node[:-2]
    return node


def _slot(node: str) -> int:
    """Return 0 for _a nodes, 1 for _b nodes."""
    return 0 if node.endswith("_a") else 1


def _alt(node: str) -> str:
    """Return the alternate node (_a ↔ _b)."""
    return _node(_base(node), 1 - _slot(node))


# ── sequence → path building ──────────────────────────────────────────────────

def sequence_to_path(
    seq: list[str],
    G: nx.DiGraph,
) -> list[str]:
    """
    Convert an RLE-compressed DA sequence to a list of node names following
    the dual-node assignment rules, respecting edges already in G.

    Rules applied in order:
      1. First DA in seq → DA_a  (fresh start)
      2. Each subsequent DA alternates _a/_b based on visit count within seq
      3. If the chosen (src, dst) edge would create an antiparallel to an
         existing edge in G, bump dst to its alternate to avoid the reversal.

    Returns the list of node names (length = len(seq)).
    """
    if not seq:
        return []

    # Track visit counts for alternation within this sequence
    visit_count: dict[str, int] = {}

    path: list[str] = []
    for da in seq:
        count      = visit_count.get(da, 0)
        visit_count[da] = count + 1
        path.append(_node(da, count))   # even count → _a, odd → _b

    # Apply antiparallel-avoidance: walk the path and fix destinations
    # We fix in-place; src is locked, only dst can be bumped.
    for i in range(len(path) - 1):
        src = path[i]
        dst = path[i + 1]
        # Would this create an antiparallel edge?
        if G.has_edge(dst, src):
            alt_dst = _alt(dst)
            # Only bump if alt_dst doesn't also create an antiparallel
            if not G.has_edge(alt_dst, src):
                # Update dst and propagate: the next step in the path that
                # uses the same DA must start from the bumped slot
                path[i + 1] = alt_dst
                # Recalculate all subsequent visits to this DA so alternation
                # stays consistent from this point forward
                da   = _base(alt_dst)
                slot = _slot(alt_dst)
                # Re-walk remaining path entries for this DA
                for j in range(i + 2, len(path)):
                    if _base(path[j]) == da:
                        slot += 1
                        path[j] = _node(da, slot)

    return path


def build_path_graph(
    sequences: list[list[str]],
    min_edge_weight: int = 1,
) -> nx.DiGraph:
    """
    Build a dual-node directed path graph from a list of RLE-compressed
    DA sequences.

    Each sequence is converted to a node path (via sequence_to_path) and its
    edges are accumulated with weights.  min_edge_weight prunes rare edges
    and their isolated nodes.
    """
    G = nx.DiGraph()

    for seq in sequences:
        compressed = rle_compress(seq)
        if len(compressed) < 2:
            continue
        path = sequence_to_path(compressed, G)
        for src, dst in zip(path, path[1:]):
            if G.has_edge(src, dst):
                G[src][dst]["weight"] += 1.0
            else:
                G.add_edge(src, dst, weight=1.0)

    # Prune rare edges and resulting isolates
    to_remove = [(u, v) for u, v, d in G.edges(data=True)
                 if d["weight"] < min_edge_weight]
    G.remove_edges_from(to_remove)
    G.remove_nodes_from(list(nx.isolates(G)))
    return G


# ── visual helpers ────────────────────────────────────────────────────────────

def _node_display(node: str, granularity: str) -> str:
    """Short display label: 'ST_a', 'CQ_b', etc."""
    base = _base(node)
    slot = "a" if node.endswith("_a") else "b"
    if granularity == "groups":
        abbr = DA_GROUP_ABBREV.get(base, base)
    else:
        abbr = (base
                .replace("-Question", "-Q")
                .replace("-answers", "-A")
                .replace("Statement-", "ST-"))
    return f"{abbr}_{slot}"


def _node_color(node: str, granularity: str) -> str:
    """Colour by base DA group; _b nodes are slightly desaturated."""
    base = _base(node)
    grp  = base if granularity == "groups" else map_da_to_group(base)
    hex_c = DA_GROUP_COLORS.get(grp, "#aaaaaa")
    if node.endswith("_b"):
        # Desaturate _b nodes slightly so _a/_b pairs are visually related
        # but distinguishable
        rgb     = mcolors.to_rgb(hex_c)
        grey    = sum(rgb) / 3
        factor  = 0.55
        blended = tuple(c * factor + grey * (1 - factor) for c in rgb)
        return mcolors.to_hex(blended)
    return hex_c


# ── plotting ──────────────────────────────────────────────────────────────────

def plot_path_graph(
    sequences: list[list[str]],
    granularity: str,
    title: str,
    outdir: str,
    fname: str,
    min_edge_weight: int = 1,
    node_run_lengths: dict[str, float] | None = None,
    save: bool = True,
    show: bool = False,
) -> nx.DiGraph | None:
    """
    Build and draw a dual-node directed path graph.

    Visual encoding
    ---------------
    - _a nodes: full DA group colour
    - _b nodes: desaturated version of same colour (visually paired)
    - Node size: mean run length when node_run_lengths is supplied,
                 otherwise out-degree weight
    - Edge width + opacity: scale with edge weight
    - Top-20% heaviest edges labelled with count
    - Curved arcs (arc3) distinguish direction
    - Legend shows DA groups + a note that filled = _a, faded = _b
    """
    os.makedirs(outdir, exist_ok=True)

    G = build_path_graph(sequences, min_edge_weight=min_edge_weight)
    if G.number_of_edges() == 0:
        print(f"  [{title}] No edges after pruning. Skipping.")
        return None

    weights = np.array([d["weight"] for _, _, d in G.edges(data=True)])
    max_w   = weights.max() if len(weights) else 1.0

    pos = nx.spring_layout(
        G,
        weight="weight",
        k=2.5 / max(len(G), 1) ** 0.5,
        iterations=200,
        seed=42,
    )

    edge_widths = [0.8 + 6.0 * (G[u][v]["weight"] / max_w) for u, v in G.edges()]
    edge_alphas = [0.2  + 0.7 * (G[u][v]["weight"] / max_w) for u, v in G.edges()]

    if node_run_lengths:
        max_rl    = max(node_run_lengths.values(), default=1.0)
        node_sizes = [
            400 + 2800 * (node_run_lengths.get(_base(n), 1.0) / max(max_rl, 1.0))
            for n in G.nodes()
        ]
    else:
        node_sizes = [
            500 + 2500 * (
                sum(d["weight"] for _, _, d in G.out_edges(n, data=True))
                / (max_w * max(len(G), 1))
            )
            for n in G.nodes()
        ]

    node_colors = [_node_color(n, granularity) for n in G.nodes()]
    node_labels = {n: _node_display(n, granularity) for n in G.nodes()}

    fig_size  = (14, 10) if len(G) <= 40 else (20, 15)
    font_size = 9 if len(G) <= 25 else (7 if len(G) <= 50 else 6)

    fig, ax = plt.subplots(figsize=fig_size)
    ax.set_title(title + "\n(dual-node directed path graph; _a=entry, _b=return)",
                 fontsize=11, pad=12)
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

    threshold = np.percentile(weights, 80) if len(weights) else max_w
    heavy = {
        (u, v): str(int(G[u][v]["weight"]))
        for u, v in G.edges()
        if G[u][v]["weight"] >= threshold
    }
    if heavy:
        nx.draw_networkx_edge_labels(
            G, pos, edge_labels=heavy, ax=ax,
            font_size=7, alpha=0.8,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", alpha=0.6, lw=0),
        )

    # Legend
    seen_groups = {
        (n if granularity == "groups" else map_da_to_group(n))
        for node in G.nodes() for n in [_base(node)]
    }
    handles = [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor=DA_GROUP_COLORS.get(g, "#aaa"),
                   markersize=10,
                   label=f"{DA_GROUP_ABBREV.get(g, g)}  {g}")
        for g in DA_GROUP_ABBREV if g in seen_groups
    ]
    handles += [
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#666", markersize=8,
                   label="_a = entry node (full colour)"),
        plt.Line2D([0], [0], marker="o", color="w",
                   markerfacecolor="#aaa", markersize=8,
                   label="_b = return node (desaturated)"),
    ]
    if node_run_lengths:
        handles.append(
            plt.Line2D([0], [0], marker="o", color="w",
                       markerfacecolor="#888", markersize=6,
                       label="node size = mean run length")
        )
    ax.legend(handles=handles, loc="upper left", fontsize=8,
              title="DA groups", framealpha=0.85)

    plt.tight_layout()
    if save:
        fpath = os.path.join(outdir, fname)
        plt.savefig(fpath, bbox_inches="tight", dpi=150)
        print(f"  Saved: {fpath}")
    if show:
        plt.show()
    plt.close()
    return G


# ── sequence extraction ───────────────────────────────────────────────────────

def extract_important_sequences(
    df: pd.DataFrame,
    importance_col: str,
    code_col: str,
    granularity: str,
    context_window: int = 15,
    include_context: bool = False,
) -> list[dict]:
    """
    Extract contiguous important blocks as labelled sequences.
    Returns list of {codes, sequence} dicts.
    Each block is a fresh sequence (resets to _a on graph building).
    Context is optionally prepended/appended to the sequence but NOT
    visualised separately it becomes part of the path.
    """
    importance = df[importance_col].fillna(0).astype(int)
    labels     = [
        get_label(row[DA_COLUMN], row["da_group"], granularity)
        for _, row in df.iterrows()
    ]
    n      = len(df)
    blocks = []

    i = 0
    while i < n:
        if importance.iloc[i] == 1:
            j = i
            while j < n and importance.iloc[j] == 1:
                j += 1

            raw_codes = df[code_col].iloc[i:j]
            codes: list[str] = []
            for raw in raw_codes:
                for c in _parse_codes(raw):
                    if c not in codes:
                        codes.append(c)
            if not codes:
                codes = ["NA"]

            block_seq = labels[i:j]
            if include_context:
                pre  = labels[max(0, i - context_window):i]
                post = labels[j:min(n, j + context_window)]
                seq  = pre + block_seq + post
            else:
                seq = block_seq

            blocks.append({"codes": codes, "sequence": seq})
            i = j
        else:
            i += 1

    return blocks


def extract_nonimportant_sequences(
    df: pd.DataFrame,
    granularity: str,
    context_window: int = 15,
    include_context: bool = False,
) -> list[list[str]]:
    """
    Extract DA sequences from non-important rows (rows not claimed by
    patient_important or therapist_important).

    When include_context=True the ±context_window rows around each important
    block are also excluded (they belong to the important partition).

    Each contiguous run of unclaimed rows becomes one sequence.
    """
    n       = len(df)
    claimed = np.zeros(n, dtype=bool)

    for col in ("patient_important", "therapist_important"):
        imp = df[col].fillna(0).astype(int)
        i   = 0
        while i < n:
            if imp.iloc[i] == 1:
                j = i
                while j < n and imp.iloc[j] == 1:
                    j += 1
                claimed[i:j] = True
                if include_context:
                    claimed[max(0, i - context_window):i] = True
                    claimed[j:min(n, j + context_window)] = True
                i = j
            else:
                i += 1

    labels = [
        get_label(row[DA_COLUMN], row["da_group"], granularity)
        for _, row in df.iterrows()
    ]

    seqs: list[list[str]] = []
    run:  list[str]       = []
    for idx in range(n):
        if not claimed[idx]:
            run.append(labels[idx])
        else:
            if run:
                seqs.append(run)
                run = []
    if run:
        seqs.append(run)
    return seqs


# ── mean run-length helper (for Option D node sizing) ────────────────────────

def compute_node_mean_run_lengths(sequences: list[list[str]]) -> dict[str, float]:
    """Mean run length per base DA label across all sequences."""
    runs: dict[str, list[int]] = {}
    for seq in sequences:
        if not seq:
            continue
        cur, cnt = seq[0], 1
        for item in seq[1:]:
            if item == cur:
                cnt += 1
            else:
                runs.setdefault(cur, []).append(cnt)
                cur, cnt = item, 1
        runs.setdefault(cur, []).append(cnt)
    return {lbl: float(np.mean(v)) for lbl, v in runs.items()}


# ── similarity methods (reuse spectral helpers from common_patterns) ──────────

def _sim_heatmap_and_dendrogram(
    dist_mat: np.ndarray,
    codes: list[str],
    sim_mat: np.ndarray,
    title: str,
    sim_dir: str,
    fname_prefix: str,
    ylabel: str,
    save: bool,
    show: bool,
):
    """Shared heatmap + dendrogram output for all similarity methods."""
    n = len(codes)

    fig, ax = plt.subplots(figsize=(max(5, n * 0.7 + 1.5), max(4, n * 0.7)))
    im = ax.imshow(sim_mat, vmin=0, vmax=1, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(n))
    ax.set_xticklabels(codes, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n))
    ax.set_yticklabels(codes, fontsize=9)
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02).set_label("Similarity", fontsize=8)
    for i in range(n):
        for j in range(n):
            if i != j:
                ax.text(j, i, f"{sim_mat[i,j]:.2f}", ha="center", va="center",
                        fontsize=8, color="black" if sim_mat[i,j] > 0.35 else "white")
    plt.tight_layout()
    if save:
        p = os.path.join(sim_dir, f"{fname_prefix}_similarity.png")
        plt.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  Saved: {p}")
    if show:
        plt.show()
    plt.close()

    if n >= 3:
        condensed = squareform(dist_mat, checks=False)
        Z = linkage(condensed, method="average")
        fig, ax = plt.subplots(figsize=(max(6, n * 0.9), 4))
        scipy_dendrogram(Z, labels=codes, ax=ax, leaf_rotation=45,
                         color_threshold=0.6 * dist_mat.max())
        ax.set_title(f"{title} clustering", fontsize=10)
        ax.set_ylabel(ylabel)
        ax.grid(True, axis="y", color="lightgrey", linewidth=0.5)
        plt.tight_layout()
        if save:
            p = os.path.join(sim_dir, f"{fname_prefix}_dendrogram.png")
            plt.savefig(p, bbox_inches="tight", dpi=150)
            print(f"  Saved: {p}")
        if show:
            plt.show()
        plt.close()


def compute_js_similarity(
    code_graphs: dict[str, nx.DiGraph],
    speaker_label: str,
    outdir: str,
    save: bool = True,
    show: bool = False,
) -> pd.DataFrame:
    sim_dir = _subdir(outdir, speaker_label, "similarity")
    codes   = sorted(code_graphs.keys())
    n       = len(codes)
    if n < 2:
        return pd.DataFrame()

    vocab      = _edge_vocab(code_graphs)
    dist_mat   = np.zeros((n, n))
    dist_vecs  = np.vstack([
        _transition_vector(code_graphs[c], vocab, normalise=True) for c in codes
    ])

    for i in range(n):
        for j in range(i + 1, n):
            d = _js_divergence(dist_vecs[i].copy(), dist_vecs[j].copy())
            dist_mat[i, j] = d
            dist_mat[j, i] = d

    sim_mat = 1.0 - dist_mat
    df_div  = pd.DataFrame(dist_mat, index=codes, columns=codes)
    df_sim  = pd.DataFrame(sim_mat,  index=codes, columns=codes)
    if save:
        df_div.to_csv(os.path.join(sim_dir, "js_divergence.csv"), float_format="%.4f")

    _sim_heatmap_and_dendrogram(
        dist_mat, codes, sim_mat,
        title=f"{speaker_label} JS transition similarity",
        sim_dir=sim_dir, fname_prefix="js",
        ylabel="JS divergence",
        save=save, show=show,
    )

    _print_sim_summary("JS", codes, sim_mat, dist_mat)
    return df_sim


def compute_netlsd_similarity(
    code_graphs: dict[str, nx.DiGraph],
    speaker_label: str,
    outdir: str,
    n_time_points: int = 250,
    save: bool = True,
    show: bool = False,
) -> pd.DataFrame:
    sim_dir     = _subdir(outdir, speaker_label, "similarity")
    codes       = sorted(code_graphs.keys())
    n           = len(codes)
    if n < 2:
        return pd.DataFrame()

    time_points = np.logspace(-2, 1, n_time_points)
    traces      = {
        c: _heat_trace(code_graphs[c], time_points) /
           max(code_graphs[c].number_of_nodes(), 1)
        for c in codes
    }

    _plot_heat_traces(traces, time_points, f"{speaker_label} NetLSD heat traces",
                      "Normalised heat trace  h(t) / n",
                      os.path.join(sim_dir, "netlsd_heat_traces.png"), save, show)

    dist_mat = _pairwise_l2(codes, traces)
    sim_mat  = 1.0 / (1.0 + dist_mat)
    df_dist  = pd.DataFrame(dist_mat, index=codes, columns=codes)
    df_sim   = pd.DataFrame(sim_mat,  index=codes, columns=codes)
    if save:
        df_dist.to_csv(os.path.join(sim_dir, "netlsd_distance.csv"), float_format="%.4f")

    _sim_heatmap_and_dendrogram(
        dist_mat, codes, sim_mat,
        title=f"{speaker_label} NetLSD structural similarity",
        sim_dir=sim_dir, fname_prefix="netlsd",
        ylabel="L2 distance",
        save=save, show=show,
    )

    _print_sim_summary("NetLSD", codes, sim_mat, dist_mat)
    return df_sim


def compute_magnetic_similarity(
    code_graphs: dict[str, nx.DiGraph],
    speaker_label: str,
    outdir: str,
    q: float = 0.25,
    n_time_points: int = 250,
    save: bool = True,
    show: bool = False,
) -> pd.DataFrame:
    sim_dir     = _subdir(outdir, speaker_label, "similarity")
    codes       = sorted(code_graphs.keys())
    n           = len(codes)
    if n < 2:
        return pd.DataFrame()

    time_points = np.logspace(-2, 1, n_time_points)
    traces      = {
        c: _magnetic_heat_trace(code_graphs[c], time_points, q=q) /
           max(code_graphs[c].number_of_nodes(), 1)
        for c in codes
    }

    _plot_heat_traces(traces, time_points,
                      f"{speaker_label} Magnetic Laplacian heat traces (q={q})",
                      "Normalised magnetic heat trace  h(t) / n",
                      os.path.join(sim_dir, "magnetic_heat_traces.png"), save, show)

    dist_mat = _pairwise_l2(codes, traces)
    sim_mat  = 1.0 / (1.0 + dist_mat)
    df_dist  = pd.DataFrame(dist_mat, index=codes, columns=codes)
    df_sim   = pd.DataFrame(sim_mat,  index=codes, columns=codes)
    if save:
        df_dist.to_csv(os.path.join(sim_dir, "magnetic_distance.csv"), float_format="%.4f")

    _sim_heatmap_and_dendrogram(
        dist_mat, codes, sim_mat,
        title=f"{speaker_label} Magnetic Laplacian similarity (q={q})",
        sim_dir=sim_dir, fname_prefix="magnetic",
        ylabel="L2 distance",
        save=save, show=show,
    )

    _print_sim_summary("Magnetic", codes, sim_mat, dist_mat)
    return df_sim


def compute_hashimoto_similarity(
    code_graphs: dict[str, nx.DiGraph],
    speaker_label: str,
    outdir: str,
    n_time_points: int = 250,
    save: bool = True,
    show: bool = False,
) -> pd.DataFrame:
    sim_dir     = _subdir(outdir, speaker_label, "similarity")
    codes       = sorted(code_graphs.keys())
    n           = len(codes)
    if n < 2:
        return pd.DataFrame()

    time_points = np.logspace(-2, 1, n_time_points)
    traces      = {
        c: _hashimoto_heat_trace(code_graphs[c], time_points) /
           max(code_graphs[c].number_of_edges(), 1)
        for c in codes
    }

    _plot_heat_traces(traces, time_points,
                      f"{speaker_label} Hashimoto heat traces",
                      "Normalised Hashimoto heat trace  h(t) / |E|",
                      os.path.join(sim_dir, "hashimoto_heat_traces.png"), save, show)

    dist_mat = _pairwise_l2(codes, traces)
    sim_mat  = 1.0 / (1.0 + dist_mat)
    df_dist  = pd.DataFrame(dist_mat, index=codes, columns=codes)
    df_sim   = pd.DataFrame(sim_mat,  index=codes, columns=codes)
    if save:
        df_dist.to_csv(os.path.join(sim_dir, "hashimoto_distance.csv"), float_format="%.4f")

    _sim_heatmap_and_dendrogram(
        dist_mat, codes, sim_mat,
        title=f"{speaker_label} Hashimoto similarity",
        sim_dir=sim_dir, fname_prefix="hashimoto",
        ylabel="L2 distance",
        save=save, show=show,
    )

    _print_sim_summary("Hashimoto", codes, sim_mat, dist_mat)
    return df_sim


# ── similarity shared utilities ───────────────────────────────────────────────

def _pairwise_l2(codes: list[str], traces: dict[str, np.ndarray]) -> np.ndarray:
    n        = len(codes)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = float(np.linalg.norm(traces[codes[i]] - traces[codes[j]]))
            dist_mat[i, j] = d
            dist_mat[j, i] = d
    return dist_mat


def _plot_heat_traces(
    traces: dict[str, np.ndarray],
    time_points: np.ndarray,
    title: str,
    ylabel: str,
    fpath: str,
    save: bool,
    show: bool,
):
    fig, ax = plt.subplots(figsize=(9, 4))
    for code, trace in traces.items():
        ax.plot(time_points, trace, label=str(code), linewidth=1.5)
    ax.set_xscale("log")
    ax.set_xlabel("Diffusion time  t  (log scale)")
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01, 1), loc="upper left")
    ax.grid(True, color="lightgrey", linewidth=0.5)
    plt.tight_layout()
    if save:
        plt.savefig(fpath, bbox_inches="tight", dpi=150)
        print(f"  Saved: {fpath}")
    if show:
        plt.show()
    plt.close()


def _print_sim_summary(
    method: str,
    codes: list[str],
    sim_mat: np.ndarray,
    dist_mat: np.ndarray,
):
    n     = len(codes)
    pairs = [
        (codes[i], codes[j], sim_mat[i, j], dist_mat[i, j])
        for i in range(n) for j in range(i + 1, n)
    ]
    pairs.sort(key=lambda x: x[2], reverse=True)
    print(f"\n  {method} similarity summary:")
    print("  Most similar:")
    for a, b, s, d in pairs[:3]:
        print(f"    {a} <-> {b}  sim={s:.3f}  dist={d:.3f}")
    print("  Least similar:")
    for a, b, s, d in pairs[-3:]:
        print(f"    {a} <-> {b}  sim={s:.3f}  dist={d:.3f}")


def run_similarity_analysis(
    code_graphs: dict[str, nx.DiGraph],
    speaker_label: str,
    outdir: str,
    save: bool = True,
    show: bool = False,
):
    """Run all four similarity methods and write combined_summary.csv."""
    if len(code_graphs) < 2:
        print(f"  [similarity] Need ≥2 graphs for {speaker_label}, skipping.")
        return

    print(f"\n{'─'*60}")
    print(f"  Graph similarity {speaker_label.upper()}")
    print(f"{'─'*60}")

    df_js       = compute_js_similarity(code_graphs, speaker_label, outdir, save, show)
    df_netlsd   = compute_netlsd_similarity(code_graphs, speaker_label, outdir,
                                            save=save, show=show)
    df_magnetic = compute_magnetic_similarity(code_graphs, speaker_label, outdir,
                                              save=save, show=show)
    df_hashimoto = compute_hashimoto_similarity(code_graphs, speaker_label, outdir,
                                                save=save, show=show)

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
                row["netlsd_similarity"]   = round(float(df_netlsd.loc[ca, cb]),   4)
            if not df_magnetic.empty:
                row["magnetic_similarity"] = round(float(df_magnetic.loc[ca, cb]), 4)
            if not df_hashimoto.empty:
                row["hashimoto_similarity"] = round(float(df_hashimoto.loc[ca, cb]), 4)

            js_s  = row["js_similarity"]
            mags  = [row.get(k) for k in ("magnetic_similarity", "hashimoto_similarity")
                     if row.get(k) is not None]
            dir_m = float(np.mean(mags)) if mags else None

            if dir_m is not None:
                if js_s > 0.7 and dir_m > 0.7:
                    row["agreement"] = "both_similar"
                elif js_s < 0.4 and dir_m < 0.4:
                    row["agreement"] = "both_different"
                elif js_s > 0.6 and dir_m < 0.5:
                    row["agreement"] = "same_transitions_diff_direction"
                elif js_s < 0.5 and dir_m > 0.6:
                    row["agreement"] = "diff_transitions_same_topology"
                elif (row.get("magnetic_similarity") and row.get("hashimoto_similarity") and
                      abs(row["magnetic_similarity"] - row["hashimoto_similarity"]) > 0.25):
                    row["agreement"] = "directed_methods_disagree"
                else:
                    row["agreement"] = "mixed"
            else:
                row["agreement"] = "js_only"

            rows.append(row)

    df_summary = pd.DataFrame(rows).sort_values("js_similarity", ascending=False)
    sim_dir    = _subdir(outdir, speaker_label, "similarity")
    if save:
        p = os.path.join(sim_dir, "combined_summary.csv")
        df_summary.to_csv(p, index=False)
        print(f"\n  Saved: {p}")
    print(f"\n  Combined similarity table ({speaker_label}):")
    print(df_summary.to_string(index=False))


def plot_codes_vs_nonimportant_dendrogram(
    code_graphs: dict[str, nx.DiGraph],
    nonimportant_graph: nx.DiGraph,
    speaker_label: str,
    outdir: str,
    n_time_points: int = 250,
    save: bool = True,
    show: bool = False,
):
    """
    Compute Hashimoto distances between every per-code graph and the
    non-important graph, then draw a single dendrogram showing how all
    codes cluster relative to each other and to non-important.

    Saved to:
      {outdir}/{speaker}/similarity/codes_vs_nonimportant_dendrogram.png
      {outdir}/{speaker}/similarity/codes_vs_nonimportant_distances.csv

    Non-important appears as a leaf labelled "non_important" so its position
    in the tree shows which codes are most similar to baseline conversation.
    """
    if not code_graphs:
        print(f"  [codes_vs_nonimportant] No code graphs for {speaker_label}, skipping.")
        return

    sim_dir = _subdir(outdir, speaker_label, "similarity")

    # Build the combined graph dict: all per-code graphs + non_important
    all_graphs: dict[str, nx.DiGraph] = {**code_graphs, "non_important": nonimportant_graph}
    labels     = sorted(all_graphs.keys())
    n          = len(labels)

    if n < 3:
        print(f"  [codes_vs_nonimportant] Need ≥3 entries for a dendrogram, got {n}.")
        return

    time_points = np.logspace(-2, 1, n_time_points)

    # Compute normalised Hashimoto heat traces for each graph
    traces: dict[str, np.ndarray] = {}
    for lbl, G in all_graphs.items():
        ht = _hashimoto_heat_trace(G, time_points)
        traces[lbl] = ht / max(G.number_of_edges(), 1)

    dist_mat = _pairwise_l2(labels, traces)
    df_dist  = pd.DataFrame(dist_mat, index=labels, columns=labels)

    if save:
        p = os.path.join(sim_dir, "codes_vs_nonimportant_distances.csv")
        df_dist.to_csv(p, float_format="%.4f")
        print(f"  Saved: {p}")

    # ── dendrogram ────────────────────────────────────────────────────────────
    condensed = squareform(dist_mat, checks=False)
    Z         = linkage(condensed, method="average")

    # Colour non_important leaf distinctly so it stands out
    def _leaf_color(label: str) -> str:
        return "#e05c3a" if label == "non_important" else "#4477aa"

    fig, ax = plt.subplots(figsize=(max(7, n * 0.9 + 1), 5))

    ddata = scipy_dendrogram(
        Z,
        labels=labels,
        ax=ax,
        leaf_rotation=45,
        color_threshold=0.0,   # all links same colour we colour leaves manually
        above_threshold_color="dimgray",
        link_color_func=lambda _: "dimgray",
    )

    # Colour the x-tick labels: non_important in orange, codes in blue
    for tick_label in ax.get_xticklabels():
        tick_label.set_color(_leaf_color(tick_label.get_text()))
        tick_label.set_fontweight(
            "bold" if tick_label.get_text() == "non_important" else "normal"
        )

    ax.set_title(
        f"{speaker_label} per-code vs non-important\n"
        f"(Hashimoto spectral distance; orange = non-important baseline)",
        fontsize=10,
    )
    ax.set_ylabel("Hashimoto L2 distance")
    ax.grid(True, axis="y", color="lightgrey", linewidth=0.5)

    # Add a small legend explaining the colours
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color="#4477aa", linewidth=4, label="code"),
        Line2D([0], [0], color="#e05c3a", linewidth=4, label="non_important"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right")

    plt.tight_layout()
    if save:
        p = os.path.join(sim_dir, "codes_vs_nonimportant_dendrogram.png")
        plt.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  Saved: {p}")
    if show:
        plt.show()
    plt.close()

    # Print where non_important sits relative to codes
    ni_idx = labels.index("non_important")
    dists_to_ni = [
        (labels[i], dist_mat[i, ni_idx])
        for i in range(n) if i != ni_idx
    ]
    dists_to_ni.sort(key=lambda x: x[1])
    print(f"\n  Codes closest to non_important ({speaker_label}):")
    for code, d in dists_to_ni[:3]:
        print(f"    {code}  dist={d:.3f}")
    print(f"  Codes furthest from non_important ({speaker_label}):")
    for code, d in dists_to_ni[-3:]:
        print(f"    {code}  dist={d:.3f}")


def plot_all_codes_dendrogram(
    patient_graphs: dict[str, nx.DiGraph],
    therapist_graphs: dict[str, nx.DiGraph],
    nonimportant_graph: nx.DiGraph | None,
    outdir: str,
    n_time_points: int = 250,
    save: bool = True,
    show: bool = False,
):
    """
    Compute Hashimoto distances across all individual per-code graphs from
    both speakers and non-important, then draw a single dendrogram.

    Node labels on the x-axis are prefixed with their speaker:
      patient_CODE, therapist_CODE, non_important

    Leaf colours:
      patient codes   blue  (#4477aa)
      therapist codes green (#2a9d5c)
      non_important   orange (#e05c3a)

    Saved to:
      {outdir}/cross_partition/similarity/all_codes_dendrogram.png
      {outdir}/cross_partition/similarity/all_codes_distances.csv
    """
    sim_dir = _subdir(outdir, "cross_partition", "similarity")

    # Build labelled graph dict with speaker-prefixed keys
    all_graphs: dict[str, nx.DiGraph] = {}
    for code, G in patient_graphs.items():
        all_graphs[f"patient_{code}"] = G
    for code, G in therapist_graphs.items():
        all_graphs[f"therapist_{code}"] = G
    if nonimportant_graph is not None:
        all_graphs["non_important"] = nonimportant_graph

    labels = sorted(all_graphs.keys())
    n      = len(labels)

    if n < 3:
        print(f"  [all_codes_dendrogram] Need ≥3 entries, got {n}. Skipping.")
        return

    print(f"\n  Building all-codes dendrogram ({n} leaves) …")

    time_points = np.logspace(-2, 1, n_time_points)

    traces: dict[str, np.ndarray] = {}
    for lbl, G in all_graphs.items():
        ht = _hashimoto_heat_trace(G, time_points)
        traces[lbl] = ht / max(G.number_of_edges(), 1)

    dist_mat = _pairwise_l2(labels, traces)
    df_dist  = pd.DataFrame(dist_mat, index=labels, columns=labels)

    if save:
        p = os.path.join(sim_dir, "all_codes_distances.csv")
        df_dist.to_csv(p, float_format="%.4f")
        print(f"  Saved: {p}")

    # ── colour map for leaf labels ────────────────────────────────────────────
    _LEAF_COLORS = {
        "patient":      "#4477aa",
        "therapist":    "#2a9d5c",
        "non_important": "#e05c3a",
    }

    def _leaf_color(lbl: str) -> str:
        if lbl == "non_important":
            return _LEAF_COLORS["non_important"]
        if lbl.startswith("patient_"):
            return _LEAF_COLORS["patient"]
        if lbl.startswith("therapist_"):
            return _LEAF_COLORS["therapist"]
        return "#888888"

    # ── dendrogram ────────────────────────────────────────────────────────────
    condensed = squareform(dist_mat, checks=False)
    Z         = linkage(condensed, method="average")

    # Figure width scales with number of leaves
    fig_w = max(10, n * 0.65 + 2)
    fig, ax = plt.subplots(figsize=(fig_w, 5))

    scipy_dendrogram(
        Z,
        labels=labels,
        ax=ax,
        leaf_rotation=55,
        color_threshold=0.0,
        above_threshold_color="dimgray",
        link_color_func=lambda _: "dimgray",
    )

    # Colour and style x-tick labels by speaker
    for tick_label in ax.get_xticklabels():
        txt = tick_label.get_text()
        tick_label.set_color(_leaf_color(txt))
        tick_label.set_fontsize(8)
        tick_label.set_fontweight(
            "bold" if txt == "non_important" else "normal"
        )

    ax.set_title(
        "All codes patient, therapist, and non-important\n"
        "(Hashimoto spectral distance; AVGLINK clustering)",
        fontsize=11,
    )
    ax.set_ylabel("Hashimoto L2 distance")
    ax.grid(True, axis="y", color="lightgrey", linewidth=0.5)

    # Legend
    from matplotlib.lines import Line2D
    legend_handles = [
        Line2D([0], [0], color=_LEAF_COLORS["patient"],
               linewidth=4, label="patient codes"),
        Line2D([0], [0], color=_LEAF_COLORS["therapist"],
               linewidth=4, label="therapist codes"),
        Line2D([0], [0], color=_LEAF_COLORS["non_important"],
               linewidth=4, label="non_important"),
    ]
    ax.legend(handles=legend_handles, fontsize=9, loc="upper right")

    plt.tight_layout()
    if save:
        p = os.path.join(sim_dir, "all_codes_dendrogram.png")
        plt.savefig(p, bbox_inches="tight", dpi=150)
        print(f"  Saved: {p}")
    if show:
        plt.show()
    plt.close()

    # Console summary: codes closest to and furthest from non_important
    if "non_important" in labels:
        ni_idx = labels.index("non_important")
        dists  = [
            (labels[i], dist_mat[i, ni_idx])
            for i in range(n) if i != ni_idx
        ]
        dists.sort(key=lambda x: x[1])
        print("\n  All-codes: closest to non_important:")
        for lbl, d in dists[:3]:
            print(f"    {lbl}  dist={d:.3f}")
        print("  All-codes: furthest from non_important:")
        for lbl, d in dists[-3:]:
            print(f"    {lbl}  dist={d:.3f}")


# ── partition runners ─────────────────────────────────────────────────────────

def run_important_partition(
    combined: pd.DataFrame,
    importance_col: str,
    code_col: str,
    speaker_label: str,
    granularity: str,
    context_window: int,
    include_context: bool,
    outdir: str,
    min_edge_weight: int,
    save: bool,
    show: bool,
) -> dict[str, nx.DiGraph]:
    """
    Build dual-node path graphs for one important speaker partition.
    Returns {code: DiGraph} for downstream similarity analysis.

    Output layout:
      {outdir}/{speaker}/graphs/
      {outdir}/{speaker}/similarity/
    """
    print(f"\n{'='*60}")
    print(f" Speaker: {speaker_label.upper()}  |  granularity: {granularity}")
    print(f"{'='*60}")

    # Collect blocks from all source files
    all_blocks: list[dict] = []
    for _, grp in combined.groupby("source_file"):
        grp = grp.reset_index(drop=True)
        all_blocks.extend(
            extract_important_sequences(
                grp,
                importance_col=importance_col,
                code_col=code_col,
                granularity=granularity,
                context_window=context_window,
                include_context=include_context,
            )
        )

    if not all_blocks:
        print("  No important blocks found.")
        return {}

    print(f"  Total important blocks: {len(all_blocks)}")

    # Explode to (block, code) rows
    rows = []
    for blk in all_blocks:
        for code in blk["codes"]:
            rows.append({"code": code, "sequence": blk["sequence"]})
    df_blocks = pd.DataFrame(rows)

    print("\n  Block counts per code:")
    print(df_blocks["code"].value_counts().to_string())

    graphs_dir   = _subdir(outdir, speaker_label, "graphs")
    code_graphs: dict[str, nx.DiGraph] = {}
    all_sequences = df_blocks["sequence"].tolist()

    # Mean run lengths for node sizing (Option D)
    all_rl = compute_node_mean_run_lengths(all_sequences)

    # Overall graph (all codes pooled)
    plot_path_graph(
        all_sequences, granularity,
        title=f"{speaker_label} all codes dual-node path graph [{granularity}]",
        outdir=graphs_dir,
        fname="path_graph_all.png",
        min_edge_weight=min_edge_weight,
        node_run_lengths=all_rl,
        save=save, show=show,
    )

    # Per-code graphs
    for code, grp in df_blocks.groupby("code"):
        seqs      = grp["sequence"].tolist()
        code_rl   = compute_node_mean_run_lengths(seqs)
        safe_code = str(code).replace("/", "-").replace(" ", "_")
        G = plot_path_graph(
            seqs, granularity,
            title=f"{speaker_label} code: {code} dual-node path graph [{granularity}]",
            outdir=graphs_dir,
            fname=f"path_graph_code_{safe_code}.png",
            min_edge_weight=min_edge_weight,
            node_run_lengths=code_rl,
            save=save, show=show,
        )
        if G is not None:
            code_graphs[str(code)] = G

    return code_graphs


def run_nonimportant_partition(
    combined: pd.DataFrame,
    granularity: str,
    context_window: int,
    include_context: bool,
    outdir: str,
    min_edge_weight: int,
    save: bool,
    show: bool,
) -> nx.DiGraph | None:
    """
    Build a dual-node path graph for the non-important portion of all
    transcripts.  Returns the pooled DiGraph for cross-partition similarity.

    Output layout:
      {outdir}/non_important/graphs/
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
                include_context=include_context,
            )
        )

    total = sum(len(s) for s in all_sequences)
    print(f"  Non-important runs: {len(all_sequences)}  |  total DAs: {total}")

    if not all_sequences:
        print("  No non-important rows found.")
        return None

    graphs_dir = _subdir(outdir, "non_important", "graphs")
    all_rl     = compute_node_mean_run_lengths(all_sequences)

    G = plot_path_graph(
        all_sequences, granularity,
        title=f"non_important dual-node path graph [{granularity}]",
        outdir=graphs_dir,
        fname="path_graph_all.png",
        min_edge_weight=min_edge_weight,
        node_run_lengths=all_rl,
        save=save, show=show,
    )
    return G


# ── CLI ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Dual-node directed path graph analysis of DA sequences."
    )
    parser.add_argument("--dir",             required=True,
                        help="Directory of word-level daseg CSV files.")
    parser.add_argument("--granularity",     default="groups",
                        choices=["groups", "raw"])
    parser.add_argument("--context_window",  type=int, default=15)
    parser.add_argument("--include_context", action="store_true",
                        help="Include ±context_window DAs in each important "
                             "block's path (and exclude them from non-important).")
    parser.add_argument("--min_edge_weight", type=int, default=1,
                        help="Prune path graph edges below this count (default: 1).")
    parser.add_argument("--outdir",          default="path_graph_output/")
    parser.add_argument("--show",            action="store_true")
    args = parser.parse_args()

    dir_path = Path(args.dir)
    if not dir_path.exists():
        raise ValueError(f"Directory not found: {args.dir}")
    os.makedirs(args.outdir, exist_ok=True)

    # ── load ──────────────────────────────────────────────────────────────────
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
        code_graphs = run_important_partition(
            combined,
            importance_col=importance_col,
            code_col=code_col,
            speaker_label=speaker_label,
            granularity=args.granularity,
            context_window=args.context_window,
            include_context=args.include_context,
            outdir=args.outdir,
            min_edge_weight=args.min_edge_weight,
            save=True,
            show=args.show,
        )
        all_speaker_graphs[speaker_label] = code_graphs

    nonimportant_graph = run_nonimportant_partition(
        combined,
        granularity=args.granularity,
        context_window=args.context_window,
        include_context=args.include_context,
        outdir=args.outdir,
        min_edge_weight=args.min_edge_weight,
        save=True,
        show=args.show,
    )

    # ── within-speaker similarity across codes ────────────────────────────────
    # Run therapist first so its combined_summary.csv is available for the
    # clustered cross-partition analysis below.
    speaker_order = ["therapist", "patient"]
    for speaker_label in speaker_order:
        code_graphs = all_speaker_graphs.get(speaker_label, {})
        run_similarity_analysis(
            code_graphs,
            speaker_label=speaker_label,
            outdir=args.outdir,
            save=True,
            show=args.show,
        )
        # Dendrogram showing each code vs the non-important baseline
        if nonimportant_graph is not None and code_graphs:
            plot_codes_vs_nonimportant_dendrogram(
                code_graphs,
                nonimportant_graph=nonimportant_graph,
                speaker_label=speaker_label,
                outdir=args.outdir,
                save=True,
                show=args.show,
            )

    # ── cross-partition similarity (original, pooled) ─────────────────────────
    print(f"\n{'='*60}")
    print(f" Cross-partition similarity (pooled)")
    print(f"{'='*60}")

    partition_graphs: dict[str, nx.DiGraph] = {}
    for speaker_label, code_graphs in all_speaker_graphs.items():
        if not code_graphs:
            continue
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

    run_similarity_analysis(
        partition_graphs,
        speaker_label="cross_partition",
        outdir=args.outdir,
        save=True,
        show=args.show,
    )

    # ── cross-partition similarity (clustered) ────────────────────────────────
    # Codes are split into two clusters based on domain knowledge:
    #   cluster_a : IAI, BCS, FSU, VLD
    #   cluster_b : everything else
    # This split is derived from the therapist Hashimoto similarity (less noisy
    # than patient) and then applied symmetrically to the patient codes.
    # The same split is used for both speakers so the comparison is consistent.
    print(f"\n{'='*60}")
    print(f" Cross-partition similarity (clustered)")
    print(f"{'='*60}")

    _cluster_a_codes = {"IAI", "BCS", "FSU", "VLD"}

    # Print the Hashimoto clustering from the therapist summary for reference
    therapist_sim_csv = os.path.join(
        args.outdir, "therapist", "similarity", "combined_summary.csv"
    )
    if os.path.exists(therapist_sim_csv):
        df_tsim = pd.read_csv(therapist_sim_csv)
        if "hashimoto_similarity" in df_tsim.columns:
            print("\n  Therapist Hashimoto similarity (for reference):")
            print(df_tsim[["code_A", "code_B", "hashimoto_similarity"]]
                  .sort_values("hashimoto_similarity", ascending=False)
                  .to_string(index=False))
    else:
        print(f"  (therapist combined_summary.csv not found at {therapist_sim_csv})")

    def _pool_codes(
        code_graphs: dict[str, nx.DiGraph],
        code_set: set[str],
        label: str,
    ) -> nx.DiGraph | None:
        """Merge graphs for codes in code_set into one pooled DiGraph."""
        matched = {c: G for c, G in code_graphs.items() if c in code_set}
        if not matched:
            print(f"  [{label}] No matching codes found in {set(code_graphs.keys())}")
            return None
        print(f"  [{label}] pooling codes: {sorted(matched.keys())}")
        G_pool = nx.DiGraph()
        for G in matched.values():
            for u, v, d in G.edges(data=True):
                if G_pool.has_edge(u, v):
                    G_pool[u][v]["weight"] += d["weight"]
                else:
                    G_pool.add_edge(u, v, weight=d["weight"])
        return G_pool if G_pool.number_of_edges() > 0 else None

    clustered_graphs: dict[str, nx.DiGraph] = {}

    for speaker_label in speaker_order:
        code_graphs = all_speaker_graphs.get(speaker_label, {})
        if not code_graphs:
            continue

        all_codes   = set(code_graphs.keys())
        cluster_b   = all_codes - _cluster_a_codes

        G_a = _pool_codes(code_graphs, _cluster_a_codes, f"{speaker_label}_cluster_a")
        G_b = _pool_codes(code_graphs, cluster_b,        f"{speaker_label}_cluster_b")

        if G_a is not None:
            clustered_graphs[f"{speaker_label}_cluster_a"] = G_a
        if G_b is not None:
            clustered_graphs[f"{speaker_label}_cluster_b"] = G_b

    if nonimportant_graph is not None:
        clustered_graphs["non_important"] = nonimportant_graph

    if len(clustered_graphs) >= 2:
        run_similarity_analysis(
            clustered_graphs,
            speaker_label="cross_partition_clustered",
            outdir=args.outdir,
            save=True,
            show=args.show,
        )
    else:
        print("  Not enough clustered graphs for similarity analysis.")

    # ── all-codes dendrogram: patient + therapist + non-important ─────────────
    print(f"\n{'='*60}")
    print(f" All-codes dendrogram (patient + therapist + non-important)")
    print(f"{'='*60}")

    plot_all_codes_dendrogram(
        patient_graphs=all_speaker_graphs.get("patient", {}),
        therapist_graphs=all_speaker_graphs.get("therapist", {}),
        nonimportant_graph=nonimportant_graph,
        outdir=args.outdir,
        save=True,
        show=args.show,
    )

    # ── directory tree summary ────────────────────────────────────────────────
    print(f"\nDone. Outputs in: {args.outdir}")
    print("\nOutput directory layout:")
    for root, dirs, files in os.walk(args.outdir):
        depth = root.replace(args.outdir, "").count(os.sep)
        print("  " * depth + os.path.basename(root) + "/")
        for f in sorted(files):
            print("  " * (depth + 1) + f)


if __name__ == "__main__":
    main()