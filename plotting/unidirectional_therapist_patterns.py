"""
transcript_graph_da.py

Builds one dual-node directed path graph per transcript (entire transcript
as a single sequence, no important/non-important split), computes a suite
of graph-level metrics, and then compares therapists by pooling all their
transcript graphs.

Therapist ID
------------
Extracted from the filename as the single character immediately before
"_with" (e.g. "session3_with_patient.csv" → therapist "3").
Files that do not contain "_with" are assigned therapist id "unknown".

Graph model
-----------
Same dual-node directed path graph as path_graph_da.py:
  - 2 nodes per DA group: DA_a (entry) and DA_b (return)
  - Each transcript is one sequence, starting fresh at DA_a
  - No antiparallel edges; destination is bumped to alternate when a
    reversal would otherwise be created

Metrics (one row per transcript in transcript_metrics.csv)
-----------------------------------------------------------
  filename, therapist_id,
  n_nodes, n_edges, density,
  n_weakly_connected_components, n_strongly_connected_components,
  largest_wcc_frac,          # fraction of nodes in largest WCC
  largest_scc_frac,          # fraction of nodes in largest SCC
  mean_in_degree, mean_out_degree,
  max_in_degree_node, max_out_degree_node,
  clustering_coeff,          # on symmetrised graph (directed has no standard defn)
  avg_shortest_path_wcc,     # avg shortest path on largest WCC (unweighted)
  top_edge_1, top_edge_1_weight,
  top_edge_2, top_edge_2_weight,
  top_edge_3, top_edge_3_weight

Similarity
----------
Four methods (JS, NetLSD, Magnetic Laplacian, Hashimoto) applied at the
therapist level (pooled across all their transcripts) and at the individual
transcript level.

Usage
-----
python transcript_graph_da.py \\
    --dir /path/to/csv_dir \\
    --granularity groups \\
    --outdir transcript_graph_output/

Requires: pandas, numpy, matplotlib, scipy, statsmodels, networkx
Drop alongside path_graph_da.py and analyze_da_patterns.py.
"""

from __future__ import annotations

import argparse
import os
import re
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import networkx as nx
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, dendrogram as scipy_dendrogram
from scipy.spatial.distance import squareform

# ── shared imports ────────────────────────────────────────────────────────────
from common_patterns import (
    DA_COLUMN,
    DA_GROUP_ABBREV,
    DA_GROUP_COLORS,
    load_da_level,
    map_da_to_group,
    get_label,
    rle_compress,
    _subdir,
    _heat_trace,
    _magnetic_heat_trace,
    _hashimoto_heat_trace,
    _js_divergence,
    _transition_vector,
    _edge_vocab,
)
from unidirectional_common_patterns import (
    _node,
    _base,
    _slot,
    _alt,
    sequence_to_path,
    build_path_graph,
    plot_path_graph,
    compute_node_mean_run_lengths,
    _pairwise_l2,
    _plot_heat_traces,
    _sim_heatmap_and_dendrogram,
    _print_sim_summary,
    _node_color,
    _node_display,
)

# ── therapist ID extraction ───────────────────────────────────────────────────

def extract_therapist_id(filename: str) -> str:
    """
    Return the single character immediately before '_with' in *filename*.
    Falls back to 'unknown' if '_with' is not present.

    Examples
    --------
    'session3_with_patient.csv'  → '3'
    'T2_with_client_01.csv'      → '2'
    'no_marker_here.csv'         → 'unknown'
    """
    m = re.search(r"(.)_with", filename)
    return m.group(1) if m else "unknown"


# ── per-transcript sequence building ─────────────────────────────────────────

def transcript_to_sequence(
    df: pd.DataFrame,
    granularity: str,
) -> list[str]:
    """
    Convert an entire DA-level transcript DataFrame into a flat label sequence.
    The whole transcript is treated as one uninterrupted sequence — no
    important/non-important split.
    """
    return [
        get_label(row[DA_COLUMN], row["da_group"], granularity)
        for _, row in df.iterrows()
    ]


# ── graph metrics ─────────────────────────────────────────────────────────────

def compute_graph_metrics(G: nx.DiGraph, filename: str, therapist_id: str) -> dict:
    """
    Compute a suite of structural metrics for a single transcript graph.

    Returns a flat dict suitable for a single CSV row.
    """
    n = G.number_of_nodes()
    e = G.number_of_edges()

    # Density: e / (n*(n-1)) for directed graphs
    density = nx.density(G) if n > 1 else 0.0

    # Connected components
    wcc_list = list(nx.weakly_connected_components(G))
    scc_list = list(nx.strongly_connected_components(G))
    n_wcc    = len(wcc_list)
    n_scc    = len(scc_list)

    largest_wcc      = max(wcc_list, key=len) if wcc_list else set()
    largest_scc      = max(scc_list, key=len) if scc_list else set()
    largest_wcc_frac = len(largest_wcc) / n if n else 0.0
    largest_scc_frac = len(largest_scc) / n if n else 0.0

    # Degree statistics
    in_degrees  = dict(G.in_degree())
    out_degrees = dict(G.out_degree())
    mean_in     = float(np.mean(list(in_degrees.values())))  if in_degrees  else 0.0
    mean_out    = float(np.mean(list(out_degrees.values()))) if out_degrees else 0.0
    max_in_node  = max(in_degrees,  key=in_degrees.get)  if in_degrees  else ""
    max_out_node = max(out_degrees, key=out_degrees.get) if out_degrees else ""

    # Clustering coefficient on the symmetrised (undirected) graph
    G_undirected     = G.to_undirected()
    clustering_coeff = nx.average_clustering(G_undirected) if n > 1 else 0.0

    # Average shortest path on the largest WCC (unweighted)
    avg_sp = float("nan")
    if len(largest_wcc) > 1:
        G_wcc  = G.subgraph(largest_wcc).copy()
        G_wcc_ud = G_wcc.to_undirected()
        try:
            avg_sp = nx.average_shortest_path_length(G_wcc_ud)
        except nx.NetworkXError:
            avg_sp = float("nan")

    # Top-3 edges by weight
    edges_sorted = sorted(
        G.edges(data=True),
        key=lambda x: x[2].get("weight", 0),
        reverse=True,
    )
    top_edges = edges_sorted[:3]
    top: dict = {}
    for rank, (u, v, d) in enumerate(top_edges, 1):
        top[f"top_edge_{rank}"]        = f"{u} -> {v}"
        top[f"top_edge_{rank}_weight"] = d.get("weight", 0)
    # Fill missing slots if fewer than 3 edges
    for rank in range(len(top_edges) + 1, 4):
        top[f"top_edge_{rank}"]        = ""
        top[f"top_edge_{rank}_weight"] = ""

    return {
        "filename":                      filename,
        "therapist_id":                  therapist_id,
        "n_nodes":                       n,
        "n_edges":                       e,
        "density":                       round(density, 6),
        "n_weakly_connected_components": n_wcc,
        "n_strongly_connected_components": n_scc,
        "largest_wcc_frac":              round(largest_wcc_frac, 4),
        "largest_scc_frac":              round(largest_scc_frac, 4),
        "mean_in_degree":                round(mean_in,  4),
        "mean_out_degree":               round(mean_out, 4),
        "max_in_degree_node":            max_in_node,
        "max_out_degree_node":           max_out_node,
        "clustering_coeff":              round(clustering_coeff, 4),
        "avg_shortest_path_wcc":         round(avg_sp, 4) if not np.isnan(avg_sp) else "nan",
        **top,
    }


# ── similarity helpers ────────────────────────────────────────────────────────

def _pool_graphs(graphs: list[nx.DiGraph]) -> nx.DiGraph:
    """Merge a list of DiGraphs by summing edge weights."""
    G_pool = nx.DiGraph()
    for G in graphs:
        for u, v, d in G.edges(data=True):
            if G_pool.has_edge(u, v):
                G_pool[u][v]["weight"] += d["weight"]
            else:
                G_pool.add_edge(u, v, weight=d["weight"])
    return G_pool


def run_similarity_analysis(
    named_graphs: dict[str, nx.DiGraph],
    label: str,
    outdir: str,
    n_time_points: int = 250,
    save: bool = True,
    show: bool = False,
):
    """
    Run all four similarity methods on *named_graphs* and write outputs to
    {outdir}/{label}/similarity/.

    Produces per-method heatmaps, dendrograms, distance CSVs, and a
    combined_summary.csv with all four scores per pair.
    """
    sim_dir = _subdir(outdir, label, "similarity")
    codes   = sorted(named_graphs.keys())
    n       = len(codes)

    if n < 2:
        print(f"  [similarity:{label}] Need ≥2 graphs, got {n}. Skipping.")
        return

    print(f"\n{'─'*60}")
    print(f"  Similarity — {label.upper()}")
    print(f"{'─'*60}")

    time_points = np.logspace(-2, 1, n_time_points)
    vocab       = _edge_vocab(named_graphs)

    # ── JS divergence ─────────────────────────────────────────────────────────
    js_vecs  = np.vstack([
        _transition_vector(named_graphs[c], vocab, normalise=True) for c in codes
    ])
    js_dist  = np.zeros((n, n))
    for i in range(n):
        for j in range(i + 1, n):
            d = _js_divergence(js_vecs[i].copy(), js_vecs[j].copy())
            js_dist[i, j] = d
            js_dist[j, i] = d
    js_sim = 1.0 - js_dist
    pd.DataFrame(js_dist, index=codes, columns=codes).to_csv(
        os.path.join(sim_dir, "js_divergence.csv"), float_format="%.4f")
    _sim_heatmap_and_dendrogram(
        js_dist, codes, js_sim,
        title=f"{label} — JS transition similarity",
        sim_dir=sim_dir, fname_prefix="js",
        ylabel="JS divergence", save=save, show=show,
    )
    _print_sim_summary("JS", codes, js_sim, js_dist)

    # ── NetLSD ────────────────────────────────────────────────────────────────
    netlsd_traces = {
        c: _heat_trace(named_graphs[c], time_points) /
           max(named_graphs[c].number_of_nodes(), 1)
        for c in codes
    }
    _plot_heat_traces(
        netlsd_traces, time_points,
        f"{label} — NetLSD heat traces",
        "Normalised heat trace  h(t) / n",
        os.path.join(sim_dir, "netlsd_heat_traces.png"), save, show,
    )
    netlsd_dist = _pairwise_l2(codes, netlsd_traces)
    netlsd_sim  = 1.0 / (1.0 + netlsd_dist)
    pd.DataFrame(netlsd_dist, index=codes, columns=codes).to_csv(
        os.path.join(sim_dir, "netlsd_distance.csv"), float_format="%.4f")
    _sim_heatmap_and_dendrogram(
        netlsd_dist, codes, netlsd_sim,
        title=f"{label} — NetLSD structural similarity",
        sim_dir=sim_dir, fname_prefix="netlsd",
        ylabel="L2 distance", save=save, show=show,
    )
    _print_sim_summary("NetLSD", codes, netlsd_sim, netlsd_dist)

    # ── Magnetic Laplacian ────────────────────────────────────────────────────
    mag_traces = {
        c: _magnetic_heat_trace(named_graphs[c], time_points) /
           max(named_graphs[c].number_of_nodes(), 1)
        for c in codes
    }
    _plot_heat_traces(
        mag_traces, time_points,
        f"{label} — Magnetic Laplacian heat traces (q=0.25)",
        "Normalised magnetic heat trace  h(t) / n",
        os.path.join(sim_dir, "magnetic_heat_traces.png"), save, show,
    )
    mag_dist = _pairwise_l2(codes, mag_traces)
    mag_sim  = 1.0 / (1.0 + mag_dist)
    pd.DataFrame(mag_dist, index=codes, columns=codes).to_csv(
        os.path.join(sim_dir, "magnetic_distance.csv"), float_format="%.4f")
    _sim_heatmap_and_dendrogram(
        mag_dist, codes, mag_sim,
        title=f"{label} — Magnetic Laplacian similarity (q=0.25)",
        sim_dir=sim_dir, fname_prefix="magnetic",
        ylabel="L2 distance", save=save, show=show,
    )
    _print_sim_summary("Magnetic", codes, mag_sim, mag_dist)

    # ── Hashimoto ─────────────────────────────────────────────────────────────
    hash_traces = {
        c: _hashimoto_heat_trace(named_graphs[c], time_points) /
           max(named_graphs[c].number_of_edges(), 1)
        for c in codes
    }
    _plot_heat_traces(
        hash_traces, time_points,
        f"{label} — Hashimoto heat traces",
        "Normalised Hashimoto heat trace  h(t) / |E|",
        os.path.join(sim_dir, "hashimoto_heat_traces.png"), save, show,
    )
    hash_dist = _pairwise_l2(codes, hash_traces)
    hash_sim  = 1.0 / (1.0 + hash_dist)
    pd.DataFrame(hash_dist, index=codes, columns=codes).to_csv(
        os.path.join(sim_dir, "hashimoto_distance.csv"), float_format="%.4f")
    _sim_heatmap_and_dendrogram(
        hash_dist, codes, hash_sim,
        title=f"{label} — Hashimoto similarity",
        sim_dir=sim_dir, fname_prefix="hashimoto",
        ylabel="L2 distance", save=save, show=show,
    )
    _print_sim_summary("Hashimoto", codes, hash_sim, hash_dist)

    # ── combined summary ──────────────────────────────────────────────────────
    rows = []
    for i, ca in enumerate(codes):
        for j, cb in enumerate(codes):
            if j <= i:
                continue
            js_s   = js_sim[i, j]
            net_s  = netlsd_sim[i, j]
            mag_s  = mag_sim[i, j]
            hash_s = hash_sim[i, j]
            dir_m  = float(np.mean([mag_s, hash_s]))

            if js_s > 0.7 and dir_m > 0.7:
                agreement = "both_similar"
            elif js_s < 0.4 and dir_m < 0.4:
                agreement = "both_different"
            elif js_s > 0.6 and dir_m < 0.5:
                agreement = "same_transitions_diff_direction"
            elif js_s < 0.5 and dir_m > 0.6:
                agreement = "diff_transitions_same_topology"
            elif abs(mag_s - hash_s) > 0.25:
                agreement = "directed_methods_disagree"
            else:
                agreement = "mixed"

            rows.append({
                "label_A":              ca,
                "label_B":              cb,
                "js_similarity":        round(js_s,   4),
                "js_divergence":        round(1 - js_s, 4),
                "netlsd_similarity":    round(net_s,  4),
                "magnetic_similarity":  round(mag_s,  4),
                "hashimoto_similarity": round(hash_s, 4),
                "agreement":            agreement,
            })

    df_summary = pd.DataFrame(rows).sort_values("js_similarity", ascending=False)
    p = os.path.join(sim_dir, "combined_summary.csv")
    df_summary.to_csv(p, index=False)
    print(f"  Saved: {p}")
    print(df_summary.to_string(index=False))


# ── main pipeline ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Per-transcript dual-node path graph metrics and therapist similarity."
    )
    parser.add_argument("--dir",             required=True,
                        help="Directory of word-level daseg CSV files.")
    parser.add_argument("--granularity",     default="groups",
                        choices=["groups", "raw"],
                        help="'groups' (default) or 'raw' DA class strings.")
    parser.add_argument("--min_edge_weight", type=int, default=1,
                        help="Prune path graph edges below this count (default: 1).")
    parser.add_argument("--outdir",          default="transcript_graph_output/")
    parser.add_argument("--show",            action="store_true")
    args = parser.parse_args()

    dir_path = Path(args.dir)
    if not dir_path.exists():
        raise ValueError(f"Directory not found: {args.dir}")
    os.makedirs(args.outdir, exist_ok=True)

    graphs_dir  = _subdir(args.outdir, "graphs")
    allowed_ext = {".csv", ".tsv", ".xlsx"}

    # ── per-transcript processing ─────────────────────────────────────────────
    metrics_rows:   list[dict]            = []
    transcript_graphs: dict[str, nx.DiGraph] = {}   # filename → graph
    therapist_graph_lists: dict[str, list[nx.DiGraph]] = {}  # tid → [graphs]

    files = sorted(
        fp for fp in dir_path.iterdir()
        if fp.suffix.lower() in allowed_ext
    )
    if not files:
        raise RuntimeError(f"No data files found in {args.dir}")

    for fp in files:
        print(f"\nProcessing {fp.name} …")
        therapist_id = extract_therapist_id(fp.name)
        print(f"  Therapist ID: {therapist_id}")

        df = load_da_level(fp)

        # Build the full-transcript label sequence (no splits)
        seq = transcript_to_sequence(df, args.granularity)
        print(f"  Sequence length (DA rows): {len(seq)}")

        # Compute mean run lengths for node sizing (Option D)
        run_lengths = compute_node_mean_run_lengths([seq])

        # Build graph
        G = build_path_graph([seq], min_edge_weight=args.min_edge_weight)
        print(f"  Graph: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

        # Save graph plot
        safe_name = fp.stem.replace(" ", "_")
        plot_path_graph(
            [seq], args.granularity,
            title=f"{fp.name}  (therapist {therapist_id})",
            outdir=graphs_dir,
            fname=f"{safe_name}_path_graph.png",
            min_edge_weight=args.min_edge_weight,
            node_run_lengths=run_lengths,
            save=True,
            show=args.show,
        )

        # Compute and store metrics
        metrics = compute_graph_metrics(G, fp.name, therapist_id)
        metrics_rows.append(metrics)

        # Store for therapist-level pooling
        transcript_graphs[fp.name] = G
        therapist_graph_lists.setdefault(therapist_id, []).append(G)

    # ── save metrics CSV ──────────────────────────────────────────────────────
    df_metrics = pd.DataFrame(metrics_rows)
    metrics_path = os.path.join(args.outdir, "transcript_metrics.csv")
    df_metrics.to_csv(metrics_path, index=False)
    print(f"\nSaved transcript metrics: {metrics_path}")
    print(df_metrics.to_string(index=False))

    # ── therapist-level similarity ────────────────────────────────────────────
    # Pool all transcript graphs per therapist then compare across therapists
    therapist_graphs: dict[str, nx.DiGraph] = {
        f"therapist_{tid}": _pool_graphs(graphs)
        for tid, graphs in sorted(therapist_graph_lists.items())
        if graphs
    }

    print(f"\nTherapists found: {sorted(therapist_graph_lists.keys())}")
    for tid, graphs in sorted(therapist_graph_lists.items()):
        print(f"  Therapist {tid}: {len(graphs)} transcript(s)")

    if len(therapist_graphs) >= 2:
        run_similarity_analysis(
            therapist_graphs,
            label="therapist_similarity",
            outdir=args.outdir,
            save=True,
            show=args.show,
        )
    else:
        print("\nFewer than 2 therapists found — skipping therapist similarity.")

    # ── per-transcript similarity (all transcripts vs each other) ────────────
    # Only run if there are a manageable number of transcripts (≤30) to avoid
    # very large distance matrices; print a warning otherwise.
    if 2 <= len(transcript_graphs) <= 30:
        run_similarity_analysis(
            transcript_graphs,
            label="transcript_similarity",
            outdir=args.outdir,
            save=True,
            show=args.show,
        )
    elif len(transcript_graphs) > 30:
        print(f"\n  {len(transcript_graphs)} transcripts found — skipping "
              f"all-vs-all transcript similarity (>30 files). "
              f"Run on a subset or increase the limit in the script.")

    # ── directory tree summary ────────────────────────────────────────────────
    print(f"\nDone. Outputs in: {args.outdir}")
    print("\nOutput directory layout:")
    for root, dirs, files_in_root in os.walk(args.outdir):
        depth = root.replace(args.outdir, "").count(os.sep)
        print("  " * depth + os.path.basename(root) + "/")
        for f in sorted(files_in_root):
            print("  " * (depth + 1) + f)


if __name__ == "__main__":
    main()