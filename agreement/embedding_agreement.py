"""
Embedding-based inter-rater agreement analysis.

Embeds contiguous important/non-important blocks as text using a local
sentence-transformer model, then computes pairwise cosine and euclidean
similarities across all comparison axes:

  1. Between therapists (same code, different therapists)
  2. Within therapist (same therapist, different sessions)
  3. Patient-important vs therapist-important vs non-important
  4. Between codes (do different codes occupy different embedding spaces?)
  5. Within codes (do same-code blocks cluster together?)

For each comparison, reports:
  - Mean cosine similarity + std
  - Mean euclidean distance + std
  - Intraclass Correlation Coefficient (ICC, two-way mixed, absolute agreement)
  - Heatmaps saved to outdir

Each block is embedded by:
  1. Attempting to embed the full concatenated utterance text as one string.
  2. If the text exceeds the model's token limit, embedding each DA utterance
     separately and averaging the resulting vectors.

Usage:
    python embedding_agreement.py \\
        --dir /path/to/csv_dir \\
        --granularity groups \\
        --text_col spoken_text \\
        --outdir embedding_output/ \\
        --model BAAI/bge-large-en-v1.5
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
import warnings
from collections import defaultdict
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_dist
from scipy.stats import pearsonr

from plotting.common_patterns import (
    DA_COLUMN,
    load_da_level,
    get_label,
    _parse_codes,
)

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ── ID parsing ────────────────────────────────────────────────────────────────

def parse_therapist_id(filename: str) -> str:
    import re
    stem  = Path(filename).stem
    match = re.search(r"T(\d+)", stem.split("_")[0], re.IGNORECASE)
    return match.group(1) if match else "unknown"


def _normalise_speaker(raw: str) -> str:
    if isinstance(raw, str) and raw.strip().lower() == "therapist":
        return "therapist"
    return "patient"


# ── data loading ──────────────────────────────────────────────────────────────

def load_records_both(
    dir_path:    Path,
    granularity: str,
    text_col:    str,
) -> list[dict]:
    """
    Load all transcripts with both patient_important and therapist_important.
    Skips files missing either column.
    Each record carries the full df plus both importance arrays.
    """
    allowed_ext = {".csv", ".tsv", ".xlsx"}
    records     = []

    for fp in sorted(dir_path.iterdir()):
        if fp.suffix.lower() not in allowed_ext:
            continue

        print(f"Loading {fp.name} …", flush=True)
        df = load_da_level(fp)
        df = df[df[DA_COLUMN] != "I-"].reset_index(drop=True)

        missing = [c for c in ("patient_important", "therapist_important")
                   if c not in df.columns]
        if missing:
            print(f"  Missing columns {missing} — skipping.", flush=True)
            continue

        if text_col not in df.columns:
            print(f"  Missing text_col '{text_col}' — skipping.", flush=True)
            continue

        df["patient_important"]   = df["patient_important"].fillna(0).astype(int)
        df["therapist_important"] = df["therapist_important"].fillna(0).astype(int)

        therapist_id = parse_therapist_id(fp.name)
        n_pat  = int(df["patient_important"].sum())
        n_ther = int(df["therapist_important"].sum())
        n_tot  = len(df)
        print(f"  therapist={therapist_id}  {n_tot} DAs  "
              f"pat_imp={n_pat}  ther_imp={n_ther}", flush=True)

        records.append({
            "filename":     fp.name,
            "therapist_id": therapist_id,
            "df":           df,
            "granularity":  granularity,
            "text_col":     text_col,
        })

    return records


# ── block extraction ──────────────────────────────────────────────────────────

def extract_blocks_with_text(
    records:    list[dict],
    target_col: str,
    code_col:   str | None = None,
) -> list[dict]:
    """
    Extract contiguous important blocks for a given target column.
    Each block carries:
      - therapist_id, filename
      - texts: list of utterance strings within the block
      - codes: list of codes assigned to the block
      - n_das: block length
    """
    blocks = []

    for rec in records:
        df          = rec["df"]
        gran        = rec["granularity"]
        text_col    = rec["text_col"]

        if target_col not in df.columns:
            continue

        importance = df[target_col].values
        texts      = [str(row.get(text_col, "")).strip()
                      for _, row in df.iterrows()]
        n          = len(df)

        i = 0
        while i < n:
            if importance[i] == 1:
                j = i
                while j < n and importance[j] == 1:
                    j += 1

                block_texts = [t for t in texts[i:j] if t]

                codes: list[str] = []
                if code_col and code_col in df.columns:
                    for raw in df[code_col].iloc[i:j]:
                        for c in _parse_codes(raw):
                            if c not in codes:
                                codes.append(c)
                if not codes:
                    codes = ["NA"]

                blocks.append({
                    "therapist_id": rec["therapist_id"],
                    "filename":     rec["filename"],
                    "texts":        block_texts,
                    "codes":        codes,
                    "n_das":        j - i,
                    "target":       target_col,
                })
                i = j
            else:
                i += 1

    return blocks


def extract_nonimportant_blocks_with_text(
    records:    list[dict],
    target_col: str,
    min_length: int = 3,
    chunk_size: int = 20,
) -> list[dict]:
    """
    Extract contiguous non-important blocks (relative to target_col).

    chunk_size: long non-important runs are split into non-overlapping chunks
      of this many DAs. This prevents very long runs from producing one
      massive averaged embedding that loses specificity. Partial trailing
      chunks shorter than min_length are discarded.
    min_length: minimum chunk/block length to include.
    """
    blocks = []

    for rec in records:
        df       = rec["df"]
        text_col = rec["text_col"]

        if target_col not in df.columns:
            continue

        importance = df[target_col].values
        texts      = [str(row.get(text_col, "")).strip()
                      for _, row in df.iterrows()]
        n          = len(df)

        i = 0
        while i < n:
            if importance[i] == 0:
                j = i
                while j < n and importance[j] == 0:
                    j += 1
                run_len = j - i

                # Chunk the run into fixed-size pieces
                for chunk_start in range(i, j, chunk_size):
                    chunk_end = min(chunk_start + chunk_size, j)
                    if chunk_end - chunk_start < min_length:
                        continue
                    block_texts = [t for t in texts[chunk_start:chunk_end] if t]
                    if block_texts:
                        blocks.append({
                            "therapist_id": rec["therapist_id"],
                            "filename":     rec["filename"],
                            "texts":        block_texts,
                            "n_das":        chunk_end - chunk_start,
                            "target":       target_col,
                        })
                i = j
            else:
                i += 1

    return blocks


# ── embedding model ───────────────────────────────────────────────────────────

def load_embedding_model(model_name: str):
    """
    Load a SentenceTransformer model.
    Returns the model object.
    """
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        raise ImportError(
            "sentence-transformers is required. "
            "Install with: pip install sentence-transformers"
        )
    print(f"Loading embedding model: {model_name} …", flush=True)
    model = SentenceTransformer(model_name)
    print(f"  Max sequence length: {model.max_seq_length}", flush=True)
    return model


def embed_block(
    texts:       list[str],
    model,
    max_tokens:  int = 512,
) -> np.ndarray:
    """
    Embed a block of utterances.

    Strategy:
      1. Try embedding the full concatenated text as one string.
      2. If the concatenated text is likely to exceed max_tokens
         (estimated by character count heuristic), embed each utterance
         separately and return the mean vector.

    Returns a 1-D numpy array of shape (embedding_dim,).
    """
    if not texts:
        return None

    full_text = " ".join(texts)

    # Rough heuristic: ~4 chars per token on average
    if len(full_text) / 4 <= max_tokens:
        try:
            vec = model.encode(full_text, show_progress_bar=False,
                               convert_to_numpy=True)
            return vec
        except Exception:
            pass

    # Fallback: embed per-utterance and average
    vecs = []
    for text in texts:
        if text.strip():
            try:
                v = model.encode(text, show_progress_bar=False,
                                 convert_to_numpy=True)
                vecs.append(v)
            except Exception:
                continue

    if not vecs:
        return None

    return np.mean(vecs, axis=0)


def embed_blocks(
    blocks: list[dict],
    model,
    max_tokens: int = 512,
) -> list[dict]:
    """
    Embed all blocks in-place, adding an 'embedding' key.
    Blocks where embedding fails are filtered out.
    """
    embedded = []
    for blk in blocks:
        vec = embed_block(blk["texts"], model, max_tokens)
        if vec is not None:
            embedded.append({**blk, "embedding": vec})
    print(f"  Embedded {len(embedded)}/{len(blocks)} blocks", flush=True)
    return embedded


# ── similarity helpers ────────────────────────────────────────────────────────

def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine similarity between two vectors."""
    return 1.0 - float(cosine_dist(a, b))


def euclidean_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance between two vectors."""
    return float(np.linalg.norm(a - b))


def pairwise_similarities(
    vecs_a: list[np.ndarray],
    vecs_b: list[np.ndarray],
) -> tuple[list[float], list[float]]:
    """
    Compute all pairwise cosine similarities and euclidean distances
    between two lists of vectors. Returns (cosines, euclideans).
    """
    cosines   = []
    euclideans = []
    for a in vecs_a:
        for b in vecs_b:
            cosines.append(cosine_sim(a, b))
            euclideans.append(euclidean_dist(a, b))
    return cosines, euclideans


def summary_stats(values: list[float]) -> dict:
    """Return mean, std, median, min, max of a list."""
    if not values:
        return {"mean": None, "std": None, "median": None,
                "min": None, "max": None, "n": 0}
    arr = np.array(values)
    return {
        "mean":   round(float(arr.mean()), 4),
        "std":    round(float(arr.std()),  4),
        "median": round(float(np.median(arr)), 4),
        "min":    round(float(arr.min()),  4),
        "max":    round(float(arr.max()),  4),
        "n":      len(values),
    }


# ── ICC ───────────────────────────────────────────────────────────────────────

def compute_icc(
    group_a_vecs: list[np.ndarray],
    group_b_vecs: list[np.ndarray],
) -> float | None:
    """
    Compute ICC(2,1) — two-way mixed, absolute agreement — between two groups
    of embeddings.

    Approach: project each embedding onto the first principal component of the
    combined set to get a scalar per block, then compute ICC on those scalars.
    This gives a single ICC value summarising whether the two raters agree on
    the relative ordering/magnitude of blocks in embedding space.

    Returns ICC value in [-1, 1], or None if insufficient data.
    """
    if len(group_a_vecs) < 2 or len(group_b_vecs) < 2:
        return None

    try:
        all_vecs = np.stack(group_a_vecs + group_b_vecs)

        # PCA to get scalar projections
        mean_vec  = all_vecs.mean(axis=0)
        centered  = all_vecs - mean_vec
        _, _, vt  = np.linalg.svd(centered, full_matrices=False)
        pc1       = vt[0]

        proj_a = np.array([float(v @ pc1) for v in group_a_vecs])
        proj_b = np.array([float(v @ pc1) for v in group_b_vecs])

        # Truncate to same length for paired ICC
        min_len = min(len(proj_a), len(proj_b))
        if min_len < 2:
            return None
        proj_a = proj_a[:min_len]
        proj_b = proj_b[:min_len]

        # ICC(2,1) formula
        n     = min_len
        grand = np.concatenate([proj_a, proj_b])
        mean  = grand.mean()

        # Between-subject SS
        means_s = (proj_a + proj_b) / 2
        ss_s    = 2 * np.sum((means_s - mean) ** 2)

        # Between-rater SS
        ss_r    = n * ((proj_a.mean() - mean) ** 2 +
                       (proj_b.mean() - mean) ** 2)

        # Residual SS
        ss_e    = np.sum((proj_a - means_s) ** 2) + \
                  np.sum((proj_b - means_s) ** 2)

        ms_s = ss_s / (n - 1)
        ms_e = ss_e / (n - 1)   # (k-1)(n-1) with k=2
        ms_r = ss_r / 1          # k-1 = 1

        icc  = (ms_s - ms_e) / (ms_s + ms_e + 2 * (ms_r - ms_e) / n)
        return round(float(icc), 4)

    except Exception as e:
        print(f"    ICC computation failed: {e}", flush=True)
        return None


# ── heatmap plotting ──────────────────────────────────────────────────────────

def plot_similarity_heatmap(
    matrix:    np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title:     str,
    outpath:   str,
    cmap:      str = "RdYlGn",
    vmin:      float | None = None,
    vmax:      float | None = None,
) -> None:
    """Save a labelled heatmap of a similarity/distance matrix."""
    n_r, n_c = len(row_labels), len(col_labels)
    fig, ax  = plt.subplots(figsize=(max(5, n_c * 0.9 + 1.5),
                                      max(4, n_r * 0.9)))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto",
                   vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n_c))
    ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_r))
    ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    for i in range(n_r):
        for j in range(n_c):
            val = matrix[i, j]
            if np.isfinite(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                        fontsize=7, color="black")
            else:
                ax.text(j, i, "N/A", ha="center", va="center",
                        fontsize=6, color="grey")

    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close()


# ── analysis functions ────────────────────────────────────────────────────────

def analyse_between_therapists(
    blocks:  list[dict],
    outdir:  str,
    label:   str = "important",
) -> list[dict]:
    """
    Between-therapist comparison: for each code, compare embeddings of blocks
    labeled with that code across different therapists.

    Returns summary rows.
    """
    print(f"\n  Between-therapist analysis ({label}) …", flush=True)
    os.makedirs(outdir, exist_ok=True)
    rows = []

    # Group by code → therapist → embeddings
    by_code_therapist: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for blk in blocks:
        for code in blk.get("codes", ["NA"]):
            by_code_therapist[code][blk["therapist_id"]].append(
                blk["embedding"]
            )

    all_codes = sorted(by_code_therapist.keys())

    for code in all_codes:
        therapists = sorted(by_code_therapist[code].keys())
        if len(therapists) < 2:
            continue

        n_t   = len(therapists)
        cos_m = np.full((n_t, n_t), np.nan)
        euc_m = np.full((n_t, n_t), np.nan)

        for i, ta in enumerate(therapists):
            for j, tb in enumerate(therapists):
                if i >= j:
                    continue
                vecs_a = by_code_therapist[code][ta]
                vecs_b = by_code_therapist[code][tb]
                cos, euc = pairwise_similarities(vecs_a, vecs_b)
                icc_val  = compute_icc(vecs_a, vecs_b)

                cos_m[i, j] = cos_m[j, i] = np.mean(cos) if cos else np.nan
                euc_m[i, j] = euc_m[j, i] = np.mean(euc) if euc else np.nan

                s_cos = summary_stats(cos)
                s_euc = summary_stats(euc)
                print(f"    {code} T{ta} vs T{tb}: "
                      f"cos={s_cos['mean']:.3f}±{s_cos['std']:.3f}  "
                      f"euc={s_euc['mean']:.3f}  ICC={icc_val}", flush=True)

                rows.append({
                    "comparison": "between_therapists",
                    "label":      label,
                    "code":       code,
                    "group_a":    f"T{ta}",
                    "group_b":    f"T{tb}",
                    "n_a":        len(vecs_a),
                    "n_b":        len(vecs_b),
                    "cos_mean":   s_cos["mean"],
                    "cos_std":    s_cos["std"],
                    "euc_mean":   s_euc["mean"],
                    "euc_std":    s_euc["std"],
                    "icc":        icc_val,
                })

        t_labels = [f"T{t}" for t in therapists]
        np.fill_diagonal(cos_m, 1.0)
        np.fill_diagonal(euc_m, 0.0)

        safe_code = str(code).replace("/", "-").replace(" ", "_")
        plot_similarity_heatmap(
            cos_m, t_labels, t_labels,
            f"Between-therapist cosine similarity — code: {code} ({label})",
            os.path.join(outdir, f"between_therapists_cos_{safe_code}.png"),
            cmap="RdYlGn", vmin=0, vmax=1,
        )
        plot_similarity_heatmap(
            euc_m, t_labels, t_labels,
            f"Between-therapist euclidean distance — code: {code} ({label})",
            os.path.join(outdir, f"between_therapists_euc_{safe_code}.png"),
            cmap="RdYlGn_r",
        )

    return rows


def analyse_within_therapist(
    blocks:  list[dict],
    outdir:  str,
    label:   str = "important",
) -> list[dict]:
    """
    Within-therapist comparison: for each therapist, compare embeddings
    between their sessions (cross-session consistency).
    """
    print(f"\n  Within-therapist analysis ({label}) …", flush=True)
    os.makedirs(outdir, exist_ok=True)
    rows = []

    by_therapist_session: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
        lambda: defaultdict(list)
    )
    for blk in blocks:
        by_therapist_session[blk["therapist_id"]][blk["filename"]].append(
            blk["embedding"]
        )

    for therapist, sessions in sorted(by_therapist_session.items()):
        session_list = sorted(sessions.keys())
        if len(session_list) < 2:
            print(f"    T{therapist}: only 1 session — skipping.", flush=True)
            continue

        n_s   = len(session_list)
        cos_m = np.full((n_s, n_s), np.nan)
        euc_m = np.full((n_s, n_s), np.nan)

        for i, sa in enumerate(session_list):
            for j, sb in enumerate(session_list):
                if i >= j:
                    continue
                vecs_a = sessions[sa]
                vecs_b = sessions[sb]
                cos, euc = pairwise_similarities(vecs_a, vecs_b)
                icc_val  = compute_icc(vecs_a, vecs_b)

                cos_m[i, j] = cos_m[j, i] = np.mean(cos) if cos else np.nan
                euc_m[i, j] = euc_m[j, i] = np.mean(euc) if euc else np.nan

                s_cos = summary_stats(cos)
                s_euc = summary_stats(euc)
                sa_s  = Path(sa).stem
                sb_s  = Path(sb).stem
                print(f"    T{therapist}: {sa_s} vs {sb_s}: "
                      f"cos={s_cos['mean']:.3f}  ICC={icc_val}", flush=True)

                rows.append({
                    "comparison": "within_therapist",
                    "label":      label,
                    "therapist":  therapist,
                    "group_a":    sa,
                    "group_b":    sb,
                    "n_a":        len(vecs_a),
                    "n_b":        len(vecs_b),
                    "cos_mean":   s_cos["mean"],
                    "cos_std":    s_cos["std"],
                    "euc_mean":   s_euc["mean"],
                    "euc_std":    s_euc["std"],
                    "icc":        icc_val,
                })

        s_labels = [Path(s).stem for s in session_list]
        np.fill_diagonal(cos_m, 1.0)
        np.fill_diagonal(euc_m, 0.0)

        plot_similarity_heatmap(
            cos_m, s_labels, s_labels,
            f"Within-therapist cosine similarity — T{therapist} ({label})",
            os.path.join(outdir, f"within_T{therapist}_cos.png"),
            cmap="RdYlGn", vmin=0, vmax=1,
        )
        plot_similarity_heatmap(
            euc_m, s_labels, s_labels,
            f"Within-therapist euclidean distance — T{therapist} ({label})",
            os.path.join(outdir, f"within_T{therapist}_euc.png"),
            cmap="RdYlGn_r",
        )

    return rows


def analyse_three_way(
    pat_imp_blocks:  list[dict],
    ther_imp_blocks: list[dict],
    nonim_blocks:    list[dict],
    outdir:          str,
) -> list[dict]:
    """
    Three-way comparison: patient-important vs therapist-important vs
    non-important embeddings. Non-important blocks are those where the
    relevant target label is 0 (computed separately for each target).

    Compares all three pairings:
      - patient_imp vs therapist_imp
      - patient_imp vs non_important
      - therapist_imp vs non_important
    """
    print(f"\n  Three-way importance comparison …", flush=True)
    os.makedirs(outdir, exist_ok=True)
    rows = []

    pat_vecs  = [b["embedding"] for b in pat_imp_blocks]
    ther_vecs = [b["embedding"] for b in ther_imp_blocks]
    nonim_vecs = [b["embedding"] for b in nonim_blocks]

    pairs = [
        ("patient_imp",   "therapist_imp",  pat_vecs,  ther_vecs),
        ("patient_imp",   "non_important",  pat_vecs,  nonim_vecs),
        ("therapist_imp", "non_important",  ther_vecs, nonim_vecs),
    ]

    group_labels = ["patient_imp", "therapist_imp", "non_important"]
    all_vecs_map = {
        "patient_imp":   pat_vecs,
        "therapist_imp": ther_vecs,
        "non_important": nonim_vecs,
    }

    # 3×3 summary matrices
    cos_m = np.full((3, 3), np.nan)
    euc_m = np.full((3, 3), np.nan)

    # Diagonal is always 1.0 cosine / 0.0 euclidean — a group is perfectly
    # similar to itself by definition. Off-diagonal captures between-group
    # separability.
    np.fill_diagonal(cos_m, 1.0)
    np.fill_diagonal(euc_m, 0.0)

    for gi, g in enumerate(group_labels):
        for gj, h in enumerate(group_labels):
            if gi == gj:
                continue
            elif gi < gj:
                vecs_a = all_vecs_map[g]
                vecs_b = all_vecs_map[h]
                if vecs_a and vecs_b:
                    cos, euc    = pairwise_similarities(vecs_a, vecs_b)
                    icc_val     = compute_icc(vecs_a, vecs_b)
                    s_cos       = summary_stats(cos)
                    s_euc       = summary_stats(euc)
                    cos_m[gi, gj] = cos_m[gj, gi] = s_cos["mean"]
                    euc_m[gi, gj] = euc_m[gj, gi] = s_euc["mean"]

                    print(f"    {g} vs {h}: "
                          f"cos={s_cos['mean']:.3f}±{s_cos['std']:.3f}  "
                          f"euc={s_euc['mean']:.3f}  ICC={icc_val}",
                          flush=True)

                    rows.append({
                        "comparison": "three_way",
                        "label":      "three_way",
                        "group_a":    g,
                        "group_b":    h,
                        "n_a":        len(vecs_a),
                        "n_b":        len(vecs_b),
                        "cos_mean":   s_cos["mean"],
                        "cos_std":    s_cos["std"],
                        "euc_mean":   s_euc["mean"],
                        "euc_std":    s_euc["std"],
                        "icc":        icc_val,
                    })

    plot_similarity_heatmap(
        cos_m, group_labels, group_labels,
        "Three-way cosine similarity (pat_imp / ther_imp / non_imp)",
        os.path.join(outdir, "three_way_cos.png"),
        cmap="RdYlGn", vmin=0, vmax=1,
    )
    plot_similarity_heatmap(
        euc_m, group_labels, group_labels,
        "Three-way euclidean distance (pat_imp / ther_imp / non_imp)",
        os.path.join(outdir, "three_way_euc.png"),
        cmap="RdYlGn_r",
    )

    return rows


def analyse_between_codes(
    blocks:  list[dict],
    outdir:  str,
    label:   str = "important",
) -> list[dict]:
    """
    Between-code comparison: are different codes separable in embedding space?
    Pools all blocks per code across all therapists.
    """
    print(f"\n  Between-code analysis ({label}) …", flush=True)
    os.makedirs(outdir, exist_ok=True)
    rows = []

    by_code: dict[str, list[np.ndarray]] = defaultdict(list)
    for blk in blocks:
        for code in blk.get("codes", ["NA"]):
            by_code[code].append(blk["embedding"])

    codes = sorted(by_code.keys())
    if len(codes) < 2:
        print("    Fewer than 2 codes — skipping between-code analysis.",
              flush=True)
        return rows

    n_c   = len(codes)
    cos_m = np.full((n_c, n_c), np.nan)
    euc_m = np.full((n_c, n_c), np.nan)

    np.fill_diagonal(cos_m, 1.0)
    np.fill_diagonal(euc_m, 0.0)

    for i, ca in enumerate(codes):
        for j, cb in enumerate(codes):
            if i == j:
                continue
            elif i < j:
                vecs_a = by_code[ca]
                vecs_b = by_code[cb]
                if vecs_a and vecs_b:
                    cos, euc   = pairwise_similarities(vecs_a, vecs_b)
                    icc_val    = compute_icc(vecs_a, vecs_b)
                    s_cos      = summary_stats(cos)
                    s_euc      = summary_stats(euc)
                    cos_m[i, j] = cos_m[j, i] = s_cos["mean"]
                    euc_m[i, j] = euc_m[j, i] = s_euc["mean"]

                    print(f"    {ca} vs {cb}: "
                          f"cos={s_cos['mean']:.3f}  ICC={icc_val}",
                          flush=True)

                    rows.append({
                        "comparison": "between_codes",
                        "label":      label,
                        "group_a":    ca,
                        "group_b":    cb,
                        "n_a":        len(vecs_a),
                        "n_b":        len(vecs_b),
                        "cos_mean":   s_cos["mean"],
                        "cos_std":    s_cos["std"],
                        "euc_mean":   s_euc["mean"],
                        "euc_std":    s_euc["std"],
                        "icc":        icc_val,
                    })

    plot_similarity_heatmap(
        cos_m, codes, codes,
        f"Between-code cosine similarity ({label})",
        os.path.join(outdir, f"between_codes_cos.png"),
        cmap="RdYlGn", vmin=0, vmax=1,
    )
    plot_similarity_heatmap(
        euc_m, codes, codes,
        f"Between-code euclidean distance ({label})",
        os.path.join(outdir, f"between_codes_euc.png"),
        cmap="RdYlGn_r",
    )

    return rows


def analyse_within_code(
    blocks:  list[dict],
    outdir:  str,
    label:   str = "important",
) -> list[dict]:
    """
    Within-code comparison: for each code, how similar are blocks from
    the same code across therapists and sessions?
    Per-code mean self-similarity and ICC.
    """
    print(f"\n  Within-code analysis ({label}) …", flush=True)
    os.makedirs(outdir, exist_ok=True)
    rows = []

    by_code: dict[str, list[np.ndarray]] = defaultdict(list)
    for blk in blocks:
        for code in blk.get("codes", ["NA"]):
            by_code[code].append(blk["embedding"])

    code_labels  = []
    cos_means    = []
    euc_means    = []

    for code, vecs in sorted(by_code.items()):
        if len(vecs) < 2:
            continue
        cos, euc  = pairwise_similarities(vecs, vecs)
        cos_self  = [c for c in cos if c < 0.9999]
        euc_self  = [e for e in euc if e > 1e-6]
        s_cos     = summary_stats(cos_self)
        s_euc     = summary_stats(euc_self)

        # ICC of all pairs within code — split into even/odd halves
        half      = len(vecs) // 2
        icc_val   = compute_icc(vecs[:half], vecs[half:2*half]) \
                    if half >= 2 else None

        print(f"    {code}: n={len(vecs)}  "
              f"cos={s_cos['mean']:.3f}±{s_cos['std']:.3f}  "
              f"ICC={icc_val}", flush=True)

        rows.append({
            "comparison": "within_code",
            "label":      label,
            "code":       code,
            "n":          len(vecs),
            "cos_mean":   s_cos["mean"],
            "cos_std":    s_cos["std"],
            "euc_mean":   s_euc["mean"],
            "euc_std":    s_euc["std"],
            "icc":        icc_val,
        })
        code_labels.append(code)
        cos_means.append(s_cos["mean"] or 0)
        euc_means.append(s_euc["mean"] or 0)

    if code_labels:
        fig, axes = plt.subplots(1, 2, figsize=(max(8, len(code_labels) * 0.8 + 2), 4))
        for ax, vals, title, ylabel in zip(
            axes,
            [cos_means, euc_means],
            [f"Within-code cosine similarity ({label})",
             f"Within-code euclidean distance ({label})"],
            ["Mean cosine similarity", "Mean euclidean distance"],
        ):
            ax.bar(range(len(code_labels)), vals,
                   color="steelblue", alpha=0.8)
            ax.set_xticks(range(len(code_labels)))
            ax.set_xticklabels(code_labels, rotation=45, ha="right", fontsize=9)
            ax.set_ylabel(ylabel)
            ax.set_title(title, fontsize=9)
            ax.grid(True, axis="y", color="lightgrey", linewidth=0.5)
        plt.tight_layout()
        plt.savefig(os.path.join(outdir, "within_code_summary.png"),
                    bbox_inches="tight", dpi=150)
        plt.close()
        print(f"  Saved: within_code_summary.png", flush=True)

    return rows


# ── main ──────────────────────────────────────────────────────────────────────


def _icc_interpretation(icc: float | None) -> str:
    """Return a human-readable reliability label for an ICC value."""
    if icc is None:
        return "N/A"
    if icc < 0:
        return "poor (negative — raters more different than chance)"
    if icc < 0.50:
        return "poor"
    if icc < 0.75:
        return "moderate"
    if icc < 0.90:
        return "good"
    return "excellent"


def analyse_overall_therapist_similarity(
    blocks:  list[dict],
    outdir:  str,
    label:   str = "important",
) -> list[dict]:
    """
    Overall between-therapist comparison pooling ALL blocks regardless of code.
    Produces:
      - Therapist x therapist cosine similarity matrix (heatmap)
      - Therapist x therapist euclidean distance matrix (heatmap)
      - Per-pair ICC
      - ICC interpretation labels
      - F1 ceiling estimate per pair and overall: F1_ceil = (1 + cos_mean) / 2

    This gives a single-number summary of how similar each therapist pair is
    overall, without breaking down by code.
    """
    print(f"\n  Overall therapist similarity ({label}) …", flush=True)
    os.makedirs(outdir, exist_ok=True)
    rows = []

    # Pool all blocks per therapist regardless of code
    by_therapist: dict[str, list[np.ndarray]] = defaultdict(list)
    for blk in blocks:
        by_therapist[blk["therapist_id"]].append(blk["embedding"])

    therapists = sorted(by_therapist.keys())
    if len(therapists) < 2:
        print("    Fewer than 2 therapists — skipping.", flush=True)
        return rows

    n_t   = len(therapists)
    cos_m = np.full((n_t, n_t), np.nan)
    euc_m = np.full((n_t, n_t), np.nan)
    f1_m  = np.full((n_t, n_t), np.nan)

    print(f"    {'Pair':<20} {'cos':>6} {'euc':>6} {'ICC':>7} {'F1_ceil':>8}  Reliability",
          flush=True)
    print(f"    {'─'*65}", flush=True)

    for i, ta in enumerate(therapists):
        cos_m[i, i] = 1.0
        euc_m[i, i] = 0.0
        f1_m[i, i]  = 1.0
        for j, tb in enumerate(therapists):
            if i >= j:
                continue
            vecs_a = by_therapist[ta]
            vecs_b = by_therapist[tb]
            cos, euc = pairwise_similarities(vecs_a, vecs_b)
            icc_val  = compute_icc(vecs_a, vecs_b)
            s_cos    = summary_stats(cos)
            s_euc    = summary_stats(euc)
            f1_ceil  = round((1.0 + s_cos["mean"]) / 2.0, 4)                        if s_cos["mean"] is not None else None

            cos_m[i, j] = cos_m[j, i] = s_cos["mean"]
            euc_m[i, j] = euc_m[j, i] = s_euc["mean"]
            f1_m[i, j]  = f1_m[j, i]  = f1_ceil

            interp = _icc_interpretation(icc_val)
            pair   = f"T{ta} vs T{tb}"
            print(f"    {pair:<20} {s_cos['mean']:>6.3f} {s_euc['mean']:>6.3f} "
                  f"{str(icc_val):>7}  F1≤{f1_ceil:.3f}  {interp}", flush=True)

            rows.append({
                "comparison":    "overall_between_therapists",
                "label":         label,
                "code":          "ALL",
                "group_a":       f"T{ta}",
                "group_b":       f"T{tb}",
                "n_a":           len(vecs_a),
                "n_b":           len(vecs_b),
                "cos_mean":      s_cos["mean"],
                "cos_std":       s_cos["std"],
                "euc_mean":      s_euc["mean"],
                "euc_std":       s_euc["std"],
                "icc":           icc_val,
                "icc_interp":    interp,
                "f1_ceiling":    f1_ceil,
            })

    # Overall aggregate
    all_cos = [r["cos_mean"] for r in rows if r["cos_mean"] is not None]
    all_f1  = [r["f1_ceiling"] for r in rows if r["f1_ceiling"] is not None]
    if all_cos:
        mean_cos   = round(float(np.mean(all_cos)), 4)
        mean_f1    = round(float(np.mean(all_f1)), 4)
        all_icc    = [r["icc"] for r in rows if r["icc"] is not None]
        mean_icc   = round(float(np.mean(all_icc)), 4) if all_icc else None
        interp     = _icc_interpretation(mean_icc)
        print(f"\n    {'OVERALL':<20} {mean_cos:>6.3f}  {'':>6} "
              f"{str(mean_icc):>7}  F1≤{mean_f1:.3f}  {interp}", flush=True)
        rows.append({
            "comparison":  "overall_between_therapists",
            "label":       label,
            "code":        "ALL",
            "group_a":     "OVERALL",
            "group_b":     "OVERALL",
            "n_a":         sum(len(v) for v in by_therapist.values()),
            "n_b":         sum(len(v) for v in by_therapist.values()),
            "cos_mean":    mean_cos,
            "cos_std":     None,
            "euc_mean":    None,
            "euc_std":     None,
            "icc":         mean_icc,
            "icc_interp":  interp,
            "f1_ceiling":  mean_f1,
        })

    # Heatmaps
    t_labels = [f"T{t}" for t in therapists]
    plot_similarity_heatmap(
        cos_m, t_labels, t_labels,
        f"Overall therapist cosine similarity ({label})",
        os.path.join(outdir, f"overall_therapist_cos_{label}.png"),
        cmap="RdYlGn", vmin=0, vmax=1,
    )
    plot_similarity_heatmap(
        euc_m, t_labels, t_labels,
        f"Overall therapist euclidean distance ({label})",
        os.path.join(outdir, f"overall_therapist_euc_{label}.png"),
        cmap="RdYlGn_r",
    )
    plot_similarity_heatmap(
        f1_m, t_labels, t_labels,
        f"Overall F1 ceiling per therapist pair ({label})\n"
        f"F1_ceil = (1 + cosine) / 2",
        os.path.join(outdir, f"overall_therapist_f1_ceiling_{label}.png"),
        cmap="RdYlGn", vmin=0.5, vmax=1.0,
    )
    print(f"  Saved: overall heatmaps ({label})", flush=True)

    return rows


def compute_embedding_f1_ceiling(
    pat_imp_blocks:   list[dict],
    ther_imp_blocks:  list[dict],
    nonim_pat_blocks: list[dict],
    nonim_ther_blocks: list[dict],
    outdir:           str,
) -> list[dict]:
    """
    Compute F1 ceiling estimates from embedding cosine similarity.

    Three separability-based ceilings (most important):
      A. patient_imp vs non_important (patient target)
         — how separable is patient importance from background in text space?
      B. therapist_imp vs non_important (therapist target)
         — same for therapist importance
      C. patient_imp vs therapist_imp
         — are the two types of important moments distinguishable?

    Formula: F1_ceil = (1 + mean_cosine_between_groups) / 2
      High cosine = groups are similar = NOT separable = LOW ceiling
      Low cosine  = groups are distinct = separable = HIGH ceiling

    NOTE: For separability, LOWER cosine = HIGHER ceiling (unlike the
    between-therapist consistency ceiling where higher cosine = higher ceiling).
    For separability we use: F1_ceil = (1 + (1 - cos)) / 2 = 1 - cos/2

    Also computes per-code between-therapist consistency ceiling
    (as before) for comparison.

    Saves a summary text file to outdir.
    """
    print(f"\n  Embedding F1 ceiling …", flush=True)
    os.makedirs(outdir, exist_ok=True)
    rows = []

    pat_vecs   = [b["embedding"] for b in pat_imp_blocks]
    ther_vecs  = [b["embedding"] for b in ther_imp_blocks]
    nonim_pat  = [b["embedding"] for b in nonim_pat_blocks]
    nonim_ther = [b["embedding"] for b in nonim_ther_blocks]

    txt = [
        "EMBEDDING F1 CEILING ANALYSIS",
        "=" * 60,
        "",
        "SEPARABILITY CEILINGS (primary — how well can a classifier distinguish",
        "important from non-important based on text content?)",
        "",
        "Formula: F1_ceil = 1 - cos/2",
        "  Low cosine between groups = groups are distinct = high ceiling",
        "  High cosine between groups = groups overlap = low ceiling",
        "",
    ]

    # ── A/B/C: separability ceilings ─────────────────────────────────────────
    sep_pairs = [
        ("patient_imp",   "non_important(pat)",  pat_vecs,  nonim_pat),
        ("therapist_imp", "non_important(ther)",  ther_vecs, nonim_ther),
        ("patient_imp",   "therapist_imp",        pat_vecs,  ther_vecs),
    ]

    txt.append(f"{'Comparison':<40} {'cos':>7} {'F1_ceil':>8} "
               f"{'ICC':>7}  Reliability")
    txt.append("─" * 70)

    for label_a, label_b, vecs_a, vecs_b in sep_pairs:
        if not vecs_a or not vecs_b:
            txt.append(f"  {label_a} vs {label_b}: insufficient data")
            continue
        cos, _   = pairwise_similarities(vecs_a, vecs_b)
        icc_val  = compute_icc(vecs_a, vecs_b)
        s_cos    = summary_stats(cos)
        # Separability ceiling: lower cosine = more distinct = higher ceiling
        f1_ceil  = round(1.0 - (s_cos["mean"] or 0) / 2.0, 4)
        interp   = _icc_interpretation(icc_val)

        pair_str = f"{label_a} vs {label_b}"
        txt.append(f"  {pair_str:<38} {s_cos['mean']:>7.4f} {f1_ceil:>8.4f} "
                   f"{str(icc_val):>7}  {interp}")
        print(f"    {pair_str}: cos={s_cos['mean']:.4f}  "
              f"F1_ceil={f1_ceil:.4f}  ICC={icc_val}  ({interp})", flush=True)

        rows.append({
            "comparison":  "embedding_separability_ceiling",
            "group_a":     label_a,
            "group_b":     label_b,
            "n_a":         len(vecs_a),
            "n_b":         len(vecs_b),
            "cos_mean":    s_cos["mean"],
            "cos_std":     s_cos["std"],
            "icc":         icc_val,
            "icc_interp":  interp,
            "f1_ceiling":  f1_ceil,
            "ceiling_type": "separability",
        })

    # Overall separability ceiling = mean of A and B (imp vs non-imp)
    sep_rows = [r for r in rows
                if r["ceiling_type"] == "separability"
                and "non_important" in r["group_b"]]
    if sep_rows:
        overall_sep = round(float(np.mean([r["f1_ceiling"] for r in sep_rows])), 4)
        txt += [
            "─" * 70,
            f"  {'Overall (imp vs non-imp mean)':<38} {'':>7} {overall_sep:>8.4f}",
            "",
        ]
        print(f"    Overall separability ceiling: {overall_sep:.4f}", flush=True)
        rows.append({
            "comparison":   "embedding_separability_ceiling",
            "group_a":      "OVERALL",
            "group_b":      "non_important",
            "n_a":          0, "n_b": 0,
            "cos_mean":     None, "cos_std": None,
            "icc":          None, "icc_interp": "N/A",
            "f1_ceiling":   overall_sep,
            "ceiling_type": "separability",
        })

    # ── Per-code between-therapist consistency ceiling ────────────────────────
    txt += [
        "",
        "CONSISTENCY CEILING (per code — do therapists agree on what each",
        "code looks like? Higher cosine = more agreement = higher ceiling)",
        "",
        "Formula: F1_ceil = (1 + cos) / 2",
        "",
        f"{'Code':<12} {'n_pairs':>7} {'cos_mean':>9} {'F1_ceil':>8} "
        f"{'ICC':>9}  Reliability",
        "─" * 65,
    ]

    for target_label, blocks in [("patient_imp", pat_imp_blocks),
                                   ("therapist_imp", ther_imp_blocks)]:
        by_code_t: dict[str, dict[str, list[np.ndarray]]] = defaultdict(
            lambda: defaultdict(list)
        )
        for blk in blocks:
            for code in blk.get("codes", ["NA"]):
                by_code_t[code][blk["therapist_id"]].append(blk["embedding"])

        code_ceilings = []
        txt.append(f"  [{target_label}]")

        for code in sorted(by_code_t.keys()):
            therapists = sorted(by_code_t[code].keys())
            if len(therapists) < 2:
                continue
            pair_cos, pair_icc = [], []
            for ta, tb in combinations(therapists, 2):
                vecs_a = by_code_t[code][ta]
                vecs_b = by_code_t[code][tb]
                cos, _ = pairwise_similarities(vecs_a, vecs_b)
                icc    = compute_icc(vecs_a, vecs_b)
                if cos:
                    pair_cos.append(float(np.mean(cos)))
                if icc is not None:
                    pair_icc.append(icc)
            if not pair_cos:
                continue
            mean_cos  = round(float(np.mean(pair_cos)), 4)
            mean_icc  = round(float(np.mean(pair_icc)), 4) if pair_icc else None
            f1_ceil   = round((1.0 + mean_cos) / 2.0, 4)
            interp    = _icc_interpretation(mean_icc)
            code_ceilings.append(f1_ceil)
            txt.append(f"  {code:<12} {len(pair_cos):>7} {mean_cos:>9.4f} "
                       f"{f1_ceil:>8.4f} {str(mean_icc):>9}  {interp}")
            rows.append({
                "comparison":   "embedding_consistency_ceiling",
                "group_a":      target_label,
                "group_b":      code,
                "n_a":          len(pair_cos), "n_b": 0,
                "cos_mean":     mean_cos, "cos_std": None,
                "icc":          mean_icc, "icc_interp": interp,
                "f1_ceiling":   f1_ceil,
                "ceiling_type": "consistency",
            })

        if code_ceilings:
            overall_cons = round(float(np.mean(code_ceilings)), 4)
            txt.append(f"  {'OVERALL':<12} {'':>7} {'':>9} "
                       f"{overall_cons:>8.4f}")

    # ── save txt ──────────────────────────────────────────────────────────────
    txt_path = os.path.join(outdir, "embedding_f1_ceiling.txt")
    with open(txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(txt))
    print(f"  Saved: {txt_path}", flush=True)

    return rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Embedding-based inter-rater agreement analysis.\n"
            "Embeds contiguous important/non-important blocks and compares\n"
            "them across therapists, sessions, targets, and codes."
        )
    )

    parser.add_argument("--dir",         required=True,
                        help="Directory containing transcript CSV/TSV/XLSX files.")
    parser.add_argument("--granularity", default="groups",
                        choices=["groups", "raw"])
    parser.add_argument("--text_col",    required=True,
                        help="Column containing spoken text for each DA.")
    parser.add_argument("--outdir",      default="embedding_output/")
    parser.add_argument("--model",       default="BAAI/bge-large-en-v1.5",
                        help="SentenceTransformer model name. "
                             "(default: BAAI/bge-large-en-v1.5)")
    parser.add_argument("--max_tokens",  type=int, default=512,
                        help="Estimated token limit for full-text embedding. "
                             "Falls back to per-DA averaging above this. "
                             "(default: 512)")
    parser.add_argument("--min_nonim_length", type=int, default=3,
                        help="Minimum DA length for non-important blocks. "
                             "(default: 3)")
    parser.add_argument("--nonim_chunk_size", type=int, default=20,
                        help="Chunk non-important runs into fixed-size pieces "
                             "of this many DAs before embedding. Prevents long "
                             "runs producing a single over-averaged embedding. "
                             "(default: 20)")

    args = parser.parse_args()

    dir_path = Path(args.dir)
    if not dir_path.exists():
        raise ValueError(f"Directory not found: {args.dir}")
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Embedding Inter-Rater Agreement Analysis", flush=True)
    print(f"model={args.model}  granularity={args.granularity}  "
          f"text_col={args.text_col}", flush=True)

    # ── load ──────────────────────────────────────────────────────────────────
    records = load_records_both(dir_path, args.granularity, args.text_col)
    if not records:
        raise RuntimeError("No valid transcripts found.")
    print(f"\nLoaded {len(records)} transcripts.", flush=True)

    # ── embedding model ───────────────────────────────────────────────────────
    emb_model = load_embedding_model(args.model)

    # ── extract and embed blocks ──────────────────────────────────────────────
    print(f"\nExtracting and embedding blocks …", flush=True)

    print(f"  Patient-important blocks …", flush=True)
    pat_imp_raw  = extract_blocks_with_text(
        records, "patient_important", "patient_code"
    )
    pat_imp      = embed_blocks(pat_imp_raw, emb_model, args.max_tokens)

    print(f"  Therapist-important blocks …", flush=True)
    ther_imp_raw = extract_blocks_with_text(
        records, "therapist_important", "therapist_code"
    )
    ther_imp     = embed_blocks(ther_imp_raw, emb_model, args.max_tokens)

    print(f"  Non-important blocks (patient target) …", flush=True)
    nonim_pat_raw  = extract_nonimportant_blocks_with_text(
        records, "patient_important", args.min_nonim_length,
        chunk_size=args.nonim_chunk_size
    )
    nonim_pat      = embed_blocks(nonim_pat_raw, emb_model, args.max_tokens)

    print(f"  Non-important blocks (therapist target) …", flush=True)
    nonim_ther_raw = extract_nonimportant_blocks_with_text(
        records, "therapist_important", args.min_nonim_length,
        chunk_size=args.nonim_chunk_size
    )
    nonim_ther     = embed_blocks(nonim_ther_raw, emb_model, args.max_tokens)

    print(f"\n  Blocks ready:")
    print(f"    patient_important:    {len(pat_imp)}", flush=True)
    print(f"    therapist_important:  {len(ther_imp)}", flush=True)
    print(f"    non_important (pat):  {len(nonim_pat)}", flush=True)
    print(f"    non_important (ther): {len(nonim_ther)}", flush=True)

    all_rows: list[dict] = []

    # ── 1. overall between-therapist (no code breakdown) ─────────────────────
    print(f"\n{'─'*60}", flush=True)
    print("  1a. OVERALL BETWEEN THERAPISTS (all codes pooled)", flush=True)
    ot_dir = os.path.join(args.outdir, "overall_therapist")
    all_rows += analyse_overall_therapist_similarity(
        pat_imp,  ot_dir, "patient_imp"
    )
    all_rows += analyse_overall_therapist_similarity(
        ther_imp, ot_dir, "therapist_imp"
    )

    # ── 1b. between therapists per code ───────────────────────────────────────
    print(f"\n{'─'*60}", flush=True)
    print("  1b. BETWEEN THERAPISTS PER CODE", flush=True)
    bt_dir = os.path.join(args.outdir, "between_therapists")
    all_rows += analyse_between_therapists(
        pat_imp,  os.path.join(bt_dir, "patient_imp"),  "patient_imp"
    )
    all_rows += analyse_between_therapists(
        ther_imp, os.path.join(bt_dir, "therapist_imp"), "therapist_imp"
    )

    # ── 2. within therapist ───────────────────────────────────────────────────
    print(f"\n{'─'*60}", flush=True)
    print("  2. WITHIN THERAPIST", flush=True)
    wt_dir = os.path.join(args.outdir, "within_therapist")
    all_rows += analyse_within_therapist(
        pat_imp,  os.path.join(wt_dir, "patient_imp"),  "patient_imp"
    )
    all_rows += analyse_within_therapist(
        ther_imp, os.path.join(wt_dir, "therapist_imp"), "therapist_imp"
    )

    # ── 3. three-way: pat_imp vs ther_imp vs non_imp ──────────────────────────
    print(f"\n{'─'*60}", flush=True)
    print("  3. THREE-WAY: patient_imp vs therapist_imp vs non_important",
          flush=True)

    # Use non-important pooled from both targets
    nonim_pooled = nonim_pat + nonim_ther
    # Deduplicate by filename+position if needed — here just pool both
    tw_dir = os.path.join(args.outdir, "three_way")
    all_rows += analyse_three_way(
        pat_imp, ther_imp, nonim_pooled, tw_dir
    )

    # Also run separately per target for non-important
    all_rows += analyse_three_way(
        pat_imp, ther_imp, nonim_pat,
        os.path.join(tw_dir, "nonim_patient_target")
    )
    all_rows += analyse_three_way(
        pat_imp, ther_imp, nonim_ther,
        os.path.join(tw_dir, "nonim_therapist_target")
    )

    # ── 4. between codes ──────────────────────────────────────────────────────
    print(f"\n{'─'*60}", flush=True)
    print("  4. BETWEEN CODES", flush=True)
    bc_dir = os.path.join(args.outdir, "between_codes")
    all_rows += analyse_between_codes(
        pat_imp,  os.path.join(bc_dir, "patient_imp"),  "patient_imp"
    )
    all_rows += analyse_between_codes(
        ther_imp, os.path.join(bc_dir, "therapist_imp"), "therapist_imp"
    )

    # ── 5. within codes ───────────────────────────────────────────────────────
    print(f"\n{'─'*60}", flush=True)
    print("  5. WITHIN CODES", flush=True)
    wc_dir = os.path.join(args.outdir, "within_codes")
    all_rows += analyse_within_code(
        pat_imp,  os.path.join(wc_dir, "patient_imp"),  "patient_imp"
    )
    all_rows += analyse_within_code(
        ther_imp, os.path.join(wc_dir, "therapist_imp"), "therapist_imp"
    )

    # ── 6. embedding f1 ceiling ──────────────────────────────────────────────
    print(f"\n{'─'*60}", flush=True)
    print("  6. EMBEDDING F1 CEILING", flush=True)
    ceil_dir = os.path.join(args.outdir, "embedding_ceiling")
    all_rows += compute_embedding_f1_ceiling(
        pat_imp_blocks=pat_imp,
        ther_imp_blocks=ther_imp,
        nonim_pat_blocks=nonim_pat,
        nonim_ther_blocks=nonim_ther,
        outdir=ceil_dir,
    )

    # ── summary CSV ───────────────────────────────────────────────────────────
    if all_rows:
        df_summary = pd.DataFrame(all_rows)
        summary_path = os.path.join(args.outdir, "embedding_summary.csv")
        df_summary.to_csv(summary_path, index=False)
        print(f"\n  Saved: {summary_path}", flush=True)

        print(f"\n  Summary ({len(df_summary)} comparisons):", flush=True)
        # Show overall and ceiling rows prominently
        key_rows = df_summary[
            df_summary["comparison"].isin([
                "overall_between_therapists", "embedding_f1_ceiling"
            ])
        ]
        if not key_rows.empty:
            print("\n  Key results:", flush=True)
            show_cols = [c for c in ["comparison", "label", "code",
                                     "group_a", "group_b", "cos_mean",
                                     "icc", "icc_interp", "f1_ceiling"]
                         if c in key_rows.columns]
            print(key_rows[show_cols].to_string(index=False), flush=True)

    print(f"\nDone. Outputs in: {args.outdir}", flush=True)


if __name__ == "__main__":
    main()
