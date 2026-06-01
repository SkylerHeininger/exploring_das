"""
Acoustic feature extraction and ceiling analysis.

Extracts openSMILE features per turn using turn-level start/end timestamps,
then analyses separability of important vs non-important turns using those
features.

WORKFLOW:
  1. Load transcript CSVs (same format as other pipeline scripts).
  2. For each transcript, find the corresponding audio file.
  3. Slice audio per turn using (start_time, end_time) timestamps.
  4. Extract openSMILE features (eGeMAPS by default) per turn slice.
  5. Aggregate features per turn (mean + std across frames).
  6. Compute ceiling analysis:
     - patient_imp vs non_important (patient target)
     - therapist_imp vs non_important (therapist target)
     - patient_imp vs therapist_imp
     Between-therapist consistency per comparison.
  7. Save features CSV, heatmaps, and summary.

AUDIO FILE PLACEHOLDER:
  Audio files are assumed to live in --audio_dir.
  Matching is by transcript stem: transcript AC01_session1.csv ->
  audio AC01_session1.wav (or .mp3, .flac, .m4a).
  Adjust _find_audio() if your naming convention differs.

FEATURE SETS:
  eGeMAPS   — 88 features, pitch, loudness, spectral, voice quality (default)
  ComParE   — 6373 features, comprehensive but slow
  custom    — pitch + jitter/shimmer + speech rate proxy only (fast)

Usage:
    python acoustic_analysis.py \\
        --dir output/ \\
        --audio_dir audio/ \\
        --outdir acoustic_output/ \\
        --feature_set egemaps
"""

from __future__ import annotations

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import os
import re
import warnings
from collections import defaultdict
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial.distance import cosine as cosine_dist
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold

from plotting.common_patterns import (
    DA_COLUMN,
    load_da_level,
    _parse_codes,
)

warnings.filterwarnings("ignore")

AUDIO_EXTENSIONS = [".wav", ".mp3", ".flac", ".m4a", ".ogg"]

# Feature names for the custom feature set
CUSTOM_FEATURE_NAMES = [
    # Pitch (F0)
    "f0_mean", "f0_std", "f0_min", "f0_max", "f0_range",
    # Loudness
    "loudness_mean", "loudness_std",
    # Jitter / shimmer (voice quality)
    "jitter_local", "shimmer_local",
    # Spectral
    "spectral_centroid_mean", "spectral_centroid_std",
    "hnr_mean",
    # Temporal / speech rate proxy
    "duration_s",
    "voiced_frac",
    "pause_count",
]

def parse_therapist_id(filename: str) -> str:
    stem  = Path(filename).stem
    match = re.search(r"T(\d+)", stem.split("_")[0], re.IGNORECASE)
    return match.group(1) if match else "unknown"


def parse_patient_id(filename: str) -> str | None:
    match = re.search(r"AC(\d{2})", Path(filename).stem, re.IGNORECASE)
    return match.group(1) if match else None


def _find_audio(transcript_path: Path, audio_dir: Path) -> Path | None:
    """
    Find audio file matching a transcript by stem name.
    Looks in audio_dir for any file with the same stem and a known audio
    extension. Returns None if not found.

    Adjust this function if your audio/transcript naming convention differs.
    """
    stem = transcript_path.stem
    for ext in AUDIO_EXTENSIONS:
        candidate = audio_dir / f"{stem}{ext}"
        if candidate.exists():
            return candidate
    # Fallback: try case-insensitive match
    if audio_dir.exists():
        for f in audio_dir.iterdir():
            if f.stem.lower() == stem.lower() and f.suffix.lower() in AUDIO_EXTENSIONS:
                return f
    return None


def load_audio_segment(
    audio_path: Path,
    start_s:    float,
    end_s:      float,
    sr:         int = 16000,
) -> tuple[np.ndarray, int] | None:
    """
    Load a segment of an audio file between start_s and end_s seconds.
    Returns (samples, sample_rate) or None on failure.
    Requires librosa.
    """
    try:
        import librosa
        y, sr_actual = librosa.load(
            str(audio_path),
            sr=sr,
            offset=start_s,
            duration=max(end_s - start_s, 0.1),
            mono=True,
        )
        return y, sr_actual
    except Exception as e:
        print(f"    Warning: could not load audio segment {start_s:.2f}-{end_s:.2f} "
              f"from {audio_path.name}: {e}", flush=True)
        return None



def extract_features_opensmile(
    audio_path:  Path,
    start_s:     float,
    end_s:       float,
    feature_set: str = "egemaps",
) -> np.ndarray | None:
    """
    Extract openSMILE features from an audio segment.
    Returns a 1-D feature vector or None on failure.

    feature_set:
      "egemaps"  — GeMAPS/eGeMAPS (88 features), recommended
      "compare"  — ComParE 2016 (6373 features), comprehensive but slow
    """
    try:
        import opensmile
    except ImportError:
        raise ImportError(
            "opensmile-python is required. "
            "Install with: pip install opensmile"
        )

    smile_set = {
        "egemaps": opensmile.FeatureSet.eGeMAPSv02,
        "compare": opensmile.FeatureSet.ComParE_2016,
    }.get(feature_set.lower())

    if smile_set is None:
        raise ValueError(f"Unknown feature_set '{feature_set}'. Use 'egemaps' or 'compare'.")

    try:
        smile = opensmile.Smile(
            feature_set=smile_set,
            feature_level=opensmile.FeatureLevel.Functionals,
        )
        duration = max(end_s - start_s, 0.1)
        features = smile.process_file(
            str(audio_path),
            start=start_s,
            end=start_s + duration,
        )
        if features.empty:
            return None
        return features.values[0].astype(np.float32)
    except Exception as e:
        print(f"    Warning: openSMILE failed for {audio_path.name} "
              f"{start_s:.2f}-{end_s:.2f}: {e}", flush=True)
        return None


def extract_features_custom(
    audio_path: Path,
    start_s:    float,
    end_s:      float,
    sr:         int = 16000,
) -> np.ndarray | None:
    """
    Extract a focused custom feature set using librosa:
      - Pitch (F0): mean, std, min, max, range
      - Loudness (RMS): mean, std
      - Jitter / shimmer (approximated via period-to-period variation)
      - Spectral centroid: mean, std
      - HNR (harmonics-to-noise ratio via autocorrelation approximation)
      - Duration
      - Voiced fraction (speech rate proxy)
      - Pause count (unvoiced gaps > 200ms)

    Returns a 1-D numpy array of length len(CUSTOM_FEATURE_NAMES).
    """
    try:
        import librosa
    except ImportError:
        raise ImportError(
            "librosa is required for the custom feature set. "
            "Install with: pip install librosa"
        )

    result = load_audio_segment(audio_path, start_s, end_s, sr)
    if result is None:
        return None
    y, sr_actual = result

    if len(y) < sr_actual * 0.05:   # less than 50ms — too short
        return None

    frame_len = int(sr_actual * 0.025)   # 25ms frames
    hop_len   = int(sr_actual * 0.010)   # 10ms hop

    try:
        f0, voiced_flag, _ = librosa.pyin(
            y, fmin=50, fmax=500,
            sr=sr_actual, frame_length=frame_len, hop_length=hop_len,
        )
        f0_voiced = f0[voiced_flag & np.isfinite(f0)] if voiced_flag is not None else f0[np.isfinite(f0)]
        if len(f0_voiced) > 0:
            f0_mean  = float(np.mean(f0_voiced))
            f0_std   = float(np.std(f0_voiced))
            f0_min   = float(np.min(f0_voiced))
            f0_max   = float(np.max(f0_voiced))
            f0_range = f0_max - f0_min
        else:
            f0_mean = f0_std = f0_min = f0_max = f0_range = 0.0
        voiced_frac = float(np.mean(voiced_flag)) if voiced_flag is not None else 0.0
    except Exception:
        f0_mean = f0_std = f0_min = f0_max = f0_range = voiced_frac = 0.0

    rms = librosa.feature.rms(y=y, frame_length=frame_len, hop_length=hop_len)[0]
    loudness_mean = float(np.mean(rms))
    loudness_std  = float(np.std(rms))

    try:
        f0_all = f0 if 'f0' in dir() else np.array([])
        f0_valid = f0_all[np.isfinite(f0_all) & (f0_all > 0)]
        if len(f0_valid) > 1:
            periods     = 1.0 / f0_valid
            jitter_local = float(np.mean(np.abs(np.diff(periods))) / np.mean(periods))
        else:
            jitter_local = 0.0
    except Exception:
        jitter_local = 0.0

    try:
        if len(f0_valid) > 1:
            # Use RMS envelope sampled at F0 periods as amplitude proxy
            rms_interp     = np.interp(
                np.linspace(0, len(rms)-1, len(f0_valid)),
                np.arange(len(rms)), rms
            )
            shimmer_local  = float(
                np.mean(np.abs(np.diff(rms_interp))) / (np.mean(rms_interp) + 1e-10)
            )
        else:
            shimmer_local = 0.0
    except Exception:
        shimmer_local = 0.0

    cent = librosa.feature.spectral_centroid(
        y=y, sr=sr_actual, n_fft=frame_len, hop_length=hop_len
    )[0]
    spectral_centroid_mean = float(np.mean(cent))
    spectral_centroid_std  = float(np.std(cent))

    try:
        ac    = np.correlate(y, y, mode='full')
        ac    = ac[len(ac)//2:]
        ac   /= (ac[0] + 1e-10)
        # peak in voiced range (2ms–20ms at 16kHz = samples 32–320)
        lo    = int(sr_actual * 0.002)
        hi    = int(sr_actual * 0.020)
        peak  = float(np.max(ac[lo:hi])) if hi < len(ac) else 0.0
        hnr   = 10 * np.log10((peak + 1e-10) / (1 - peak + 1e-10))
        hnr_mean = float(np.clip(hnr, -20, 40))
    except Exception:
        hnr_mean = 0.0

    duration_s = float(end_s - start_s)

    try:
        pause_threshold_frames = int(0.200 / (hop_len / sr_actual))
        if voiced_flag is not None and len(voiced_flag) > 0:
            unvoiced         = (~voiced_flag).astype(int)
            # count runs of unvoiced > threshold
            in_pause, count  = False, 0
            run              = 0
            for v in unvoiced:
                if v == 1:
                    run += 1
                else:
                    if in_pause and run >= pause_threshold_frames:
                        count += 1
                    run, in_pause = 0, False
                    in_pause = True if run > 0 else False
            pause_count = float(count)
        else:
            pause_count = 0.0
    except Exception:
        pause_count = 0.0

    return np.array([
        f0_mean, f0_std, f0_min, f0_max, f0_range,
        loudness_mean, loudness_std,
        jitter_local, shimmer_local,
        spectral_centroid_mean, spectral_centroid_std,
        hnr_mean,
        duration_s, voiced_frac, pause_count,
    ], dtype=np.float32)


def extract_turn_features(
    audio_path:  Path,
    start_s:     float,
    end_s:       float,
    feature_set: str,
) -> np.ndarray | None:
    """Dispatch to the right feature extractor."""
    if feature_set.lower() == "custom":
        return extract_features_custom(audio_path, start_s, end_s)
    else:
        return extract_features_opensmile(audio_path, start_s, end_s, feature_set)


def get_feature_names(feature_set: str, n_features: int) -> list[str]:
    """Return feature names for the given feature set."""
    if feature_set.lower() == "custom":
        return CUSTOM_FEATURE_NAMES
    return [f"feat_{i:04d}" for i in range(n_features)]



def load_records(
    dir_path:    Path,
    audio_dir:   Path,
    feature_set: str,
    start_col:   str,
    end_col:     str,
) -> list[dict]:
    """
    Load all transcripts and extract acoustic features per turn.

    Each record:
    {
      filename:       str,
      therapist_id:   str,
      patient_id:     str,
      turn_id:        int,
      start_s:        float,
      end_s:          float,
      speaker:        str,
      pat_label:      int,
      ther_label:     int,
      features:       np.ndarray | None,
      has_audio:      bool,
    }
    """
    allowed_ext = {".csv", ".tsv", ".xlsx"}
    records     = []

    for fp in sorted(dir_path.iterdir()):
        if fp.suffix.lower() not in allowed_ext:
            continue

        therapist_id = parse_therapist_id(fp.name)
        patient_id   = parse_patient_id(fp.name)
        if patient_id is None:
            print(f"Warning: could not parse patient ID from {fp.name} — skipping.",
                  flush=True)
            continue

        print(f"Loading {fp.name}  (T{therapist_id}  AC{patient_id}) …",
              flush=True)
        df = load_da_level(fp)

        missing = [c for c in ("patient_important", "therapist_important",
                               start_col, end_col)
                   if c not in df.columns]
        if missing:
            print(f"  Missing columns {missing} — skipping.", flush=True)
            continue

        df["patient_important"]   = df["patient_important"].fillna(0).astype(int)
        df["therapist_important"] = df["therapist_important"].fillna(0).astype(int)

        audio_path = _find_audio(fp, audio_dir)
        if audio_path is None:
            print(f"  Warning: no audio found for {fp.name} in {audio_dir} — "
                  f"features will be None.", flush=True)
            has_audio = False
        else:
            print(f"  Audio: {audio_path.name}", flush=True)
            has_audio = True

        # Use the DA-level data to get turn groupings and aggregate labels
        if "timestamp" in df.columns and "speaker" in df.columns:
            turns = _group_das_to_turns(df, start_col, end_col)
        else:
            # Fall back to treating each row as a turn
            turns = []
            for idx, row in df.iterrows():
                try:
                    start_s = float(row[start_col])
                    end_s   = float(row[end_col])
                except (ValueError, TypeError):
                    continue
                turns.append({
                    "turn_id":    idx,
                    "start_s":    start_s,
                    "end_s":      end_s,
                    "speaker":    str(row.get("speaker", "unknown")),
                    "pat_label":  int(row["patient_important"]),
                    "ther_label": int(row["therapist_important"]),
                })

        n_turns = len(turns)
        print(f"  {n_turns} turns", flush=True)

        n_extracted = 0
        for turn in turns:
            feat = None
            if has_audio and turn["end_s"] > turn["start_s"]:
                feat = extract_turn_features(
                    audio_path, turn["start_s"], turn["end_s"], feature_set
                )
                if feat is not None:
                    n_extracted += 1

            records.append({
                "filename":     fp.name,
                "therapist_id": therapist_id,
                "patient_id":   patient_id,
                "turn_id":      turn["turn_id"],
                "start_s":      turn["start_s"],
                "end_s":        turn["end_s"],
                "speaker":      turn["speaker"],
                "pat_label":    turn["pat_label"],
                "ther_label":   turn["ther_label"],
                "features":     feat,
                "has_audio":    has_audio,
            })

        print(f"  Extracted features for {n_extracted}/{n_turns} turns",
              flush=True)

    print(f"\nTotal records: {len(records)}", flush=True)
    return records


def _group_das_to_turns(
    df:        pd.DataFrame,
    start_col: str,
    end_col:   str,
) -> list[dict]:
    """
    Group DA-level rows into turns by (timestamp, speaker).
    Turn start = min DA start, end = max DA end.
    Turn label = 1 if ANY DA in the turn is labeled important.
    """
    def _norm_speaker(raw):
        if isinstance(raw, str) and raw.strip().lower() == "therapist":
            return "therapist"
        return "patient"

    turns     = []
    turn_id   = 0
    prev_key  = None
    cur_rows  = []

    for _, row in df.iterrows():
        ts   = row.get("timestamp", 0)
        spkr = _norm_speaker(str(row.get("speaker", "patient")))
        key  = (ts, spkr)

        if key != prev_key:
            if cur_rows:
                turns.append(_aggregate_turn(cur_rows, turn_id, start_col, end_col))
                turn_id += 1
            cur_rows = [row]
            prev_key = key
        else:
            cur_rows.append(row)

    if cur_rows:
        turns.append(_aggregate_turn(cur_rows, turn_id, start_col, end_col))

    return turns


def _aggregate_turn(
    rows:      list,
    turn_id:   int,
    start_col: str,
    end_col:   str,
) -> dict:
    """Aggregate a list of DA rows into a single turn dict."""
    starts = []
    ends   = []
    for row in rows:
        try:
            starts.append(float(row[start_col]))
            ends.append(float(row[end_col]))
        except (ValueError, TypeError):
            pass

    start_s = min(starts) if starts else 0.0
    end_s   = max(ends)   if ends   else 0.0
    spkr    = str(rows[0].get("speaker", "unknown"))

    pat_label  = int(any(int(r["patient_important"]) == 1   for r in rows))
    ther_label = int(any(int(r["therapist_important"]) == 1 for r in rows))

    return {
        "turn_id":    turn_id,
        "start_s":    start_s,
        "end_s":      end_s,
        "speaker":    spkr,
        "pat_label":  pat_label,
        "ther_label": ther_label,
    }


def get_feature_matrix(
    records:    list[dict],
    label_col:  str,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Build X (feature matrix) and y (labels) from records.
    Drops records with None features.
    Returns (X, y, therapist_ids).
    """
    valid = [r for r in records if r["features"] is not None]
    if not valid:
        return np.empty((0, 0)), np.array([]), []

    X    = np.stack([r["features"] for r in valid])
    y    = np.array([r[label_col] for r in valid], dtype=int)
    tids = [r["therapist_id"] for r in valid]
    return X, y, tids


def impute_and_scale(X_train: np.ndarray, X_test: np.ndarray):
    """Replace NaN/Inf with column medians from train, then StandardScale."""
    X_train = X_train.copy().astype(float)
    X_test  = X_test.copy().astype(float)

    # Replace inf
    X_train[~np.isfinite(X_train)] = np.nan
    X_test[~np.isfinite(X_test)]   = np.nan

    # Impute with train medians
    medians = np.nanmedian(X_train, axis=0)
    for j in range(X_train.shape[1]):
        mask = np.isnan(X_train[:, j])
        X_train[mask, j] = medians[j]
        mask_te = np.isnan(X_test[:, j])
        X_test[mask_te, j] = medians[j]

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)
    return X_train, X_test


def compute_icc(vecs_a: list[np.ndarray], vecs_b: list[np.ndarray]) -> float | None:
    """ICC(2,1) via PCA projection — same as embedding_agreement.py."""
    if len(vecs_a) < 2 or len(vecs_b) < 2:
        return None
    try:
        all_vecs = np.stack(vecs_a + vecs_b)
        mean_vec = all_vecs.mean(axis=0)
        centered = all_vecs - mean_vec
        _, _, vt = np.linalg.svd(centered, full_matrices=False)
        pc1      = vt[0]
        proj_a   = np.array([float(v @ pc1) for v in vecs_a])
        proj_b   = np.array([float(v @ pc1) for v in vecs_b])
        min_len  = min(len(proj_a), len(proj_b))
        if min_len < 2:
            return None
        proj_a, proj_b = proj_a[:min_len], proj_b[:min_len]
        n     = min_len
        grand = np.concatenate([proj_a, proj_b])
        mean  = grand.mean()
        means_s = (proj_a + proj_b) / 2
        ss_s    = 2 * np.sum((means_s - mean) ** 2)
        ss_r    = n * ((proj_a.mean() - mean)**2 + (proj_b.mean() - mean)**2)
        ss_e    = (np.sum((proj_a - means_s)**2) +
                   np.sum((proj_b - means_s)**2))
        ms_s, ms_e, ms_r = ss_s/(n-1), ss_e/(n-1), ss_r/1
        icc  = (ms_s - ms_e) / (ms_s + ms_e + 2*(ms_r - ms_e)/n)
        return round(float(icc), 4)
    except Exception:
        return None


def _icc_interpretation(icc: float | None) -> str:
    if icc is None: return "N/A"
    if icc < 0:     return "poor (negative)"
    if icc < 0.50:  return "poor"
    if icc < 0.75:  return "moderate"
    if icc < 0.90:  return "good"
    return "excellent"


def acoustic_f1_ceiling_classifier(
    X_a:   np.ndarray,
    X_b:   np.ndarray,
    label: str,
    n_splits: int = 5,
) -> dict:
    """
    Estimate F1 ceiling for separating two groups using acoustic features.
    Fits a logistic regression classifier with stratified k-fold CV.
    Returns mean F1 and std across folds.
    """
    if len(X_a) == 0 or len(X_b) == 0:
        return {"f1_mean": None, "f1_std": None, "n_a": 0, "n_b": 0}

    X = np.vstack([X_a, X_b])
    y = np.array([0]*len(X_a) + [1]*len(X_b))

    n_splits = min(n_splits, min(
        np.sum(y == 0), np.sum(y == 1)
    ))
    if n_splits < 2:
        # Not enough data for CV — fit on all, evaluate on all
        scaler = StandardScaler()
        X_s    = scaler.fit_transform(np.nan_to_num(X))
        clf    = LogisticRegression(max_iter=1000, class_weight="balanced")
        clf.fit(X_s, y)
        pred   = clf.predict(X_s)
        f1     = f1_score(y, pred, pos_label=1, zero_division=0)
        return {"f1_mean": round(float(f1), 4), "f1_std": None,
                "n_a": len(X_a), "n_b": len(X_b), "note": "no_cv"}

    kf      = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    f1s     = []
    X_clean = np.nan_to_num(X)

    for train_idx, test_idx in kf.split(X_clean, y):
        X_tr, X_te = impute_and_scale(X_clean[train_idx], X_clean[test_idx])
        clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        try:
            clf.fit(X_tr, y[train_idx])
            pred = clf.predict(X_te)
            f1s.append(f1_score(y[test_idx], pred, pos_label=1, zero_division=0))
        except Exception:
            pass

    if not f1s:
        return {"f1_mean": None, "f1_std": None, "n_a": len(X_a), "n_b": len(X_b)}

    return {
        "f1_mean": round(float(np.mean(f1s)), 4),
        "f1_std":  round(float(np.std(f1s)),  4),
        "n_a":     len(X_a),
        "n_b":     len(X_b),
    }



def plot_heatmap(
    matrix:     np.ndarray,
    row_labels: list[str],
    col_labels: list[str],
    title:      str,
    outpath:    str,
    cmap:       str = "RdYlGn",
    vmin=None,
    vmax=None,
) -> None:
    n_r, n_c = len(row_labels), len(col_labels)
    fig, ax  = plt.subplots(figsize=(max(5, n_c*0.9+1.5), max(4, n_r*0.9)))
    im = ax.imshow(matrix, cmap=cmap, aspect="auto", vmin=vmin, vmax=vmax)
    ax.set_xticks(range(n_c)); ax.set_xticklabels(col_labels, rotation=45, ha="right", fontsize=9)
    ax.set_yticks(range(n_r)); ax.set_yticklabels(row_labels, fontsize=9)
    ax.set_title(title, fontsize=9)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    for i in range(n_r):
        for j in range(n_c):
            v = matrix[i, j]
            txt = f"{v:.2f}" if np.isfinite(v) else "N/A"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7)
    plt.tight_layout()
    plt.savefig(outpath, bbox_inches="tight", dpi=150)
    plt.close()



def analyse_separability(
    records:  list[dict],
    outdir:   str,
) -> list[dict]:
    """
    Three-way separability ceiling:
      A. patient_imp vs non_important (patient target)
      B. therapist_imp vs non_important (therapist target)
      C. patient_imp vs therapist_imp

    Uses logistic regression classifier CV.
    """
    print(f"\n  Separability ceiling …", flush=True)
    os.makedirs(outdir, exist_ok=True)
    rows = []

    def _get(label_col, value):
        return np.stack([r["features"] for r in records
                         if r["features"] is not None
                         and r[label_col] == value])

    pat_imp   = _get("pat_label",  1)
    pat_nonim = _get("pat_label",  0)
    thr_imp   = _get("ther_label", 1)
    thr_nonim = _get("ther_label", 0)

    pairs = [
        ("patient_imp",    "non_imp_pat",   pat_imp,  pat_nonim),
        ("therapist_imp",  "non_imp_ther",  thr_imp,  thr_nonim),
        ("patient_imp",    "therapist_imp", pat_imp,  thr_imp),
    ]

    print(f"  {'Comparison':<40} {'F1_ceil':>8} {'±std':>6}  n_a  n_b", flush=True)
    print(f"  {'─'*65}", flush=True)

    for label_a, label_b, X_a, X_b in pairs:
        result = acoustic_f1_ceiling_classifier(X_a, X_b, f"{label_a}_vs_{label_b}")
        f1_str  = f"{result['f1_mean']:.4f}" if result["f1_mean"] is not None else "N/A"
        std_str = f"{result['f1_std']:.4f}"  if result["f1_std"]  is not None else "N/A"
        print(f"  {label_a+' vs '+label_b:<40} {f1_str:>8} {std_str:>6}  "
              f"{result['n_a']:>4}  {result['n_b']:>4}", flush=True)
        rows.append({
            "comparison":  "acoustic_separability",
            "group_a":     label_a,
            "group_b":     label_b,
            "f1_ceiling":  result["f1_mean"],
            "f1_std":      result["f1_std"],
            "n_a":         result["n_a"],
            "n_b":         result["n_b"],
        })

    return rows


def analyse_between_therapists(
    records:  list[dict],
    label_col: str,
    label_name: str,
    outdir:    str,
) -> list[dict]:
    """
    Between-therapist consistency on acoustic features.
    Uses ICC and cosine similarity on mean feature vectors per therapist.
    """
    print(f"\n  Between-therapist acoustic consistency ({label_name}) …",
          flush=True)
    os.makedirs(outdir, exist_ok=True)
    rows = []

    by_therapist: dict[str, list[np.ndarray]] = defaultdict(list)
    for r in records:
        if r["features"] is not None and r[label_col] == 1:
            by_therapist[r["therapist_id"]].append(r["features"])

    therapists = sorted(by_therapist.keys())
    if len(therapists) < 2:
        print("    Fewer than 2 therapists with data — skipping.", flush=True)
        return rows

    n_t   = len(therapists)
    cos_m = np.full((n_t, n_t), np.nan)
    np.fill_diagonal(cos_m, 1.0)

    print(f"  {'Pair':<20} {'cos':>6}  {'ICC':>7}  Reliability", flush=True)
    print(f"  {'─'*55}", flush=True)

    for i, ta in enumerate(therapists):
        for j, tb in enumerate(therapists):
            if i >= j:
                continue
            vecs_a = by_therapist[ta]
            vecs_b = by_therapist[tb]

            # Mean vectors per therapist for cosine
            mu_a  = np.nanmean(np.stack(vecs_a), axis=0)
            mu_b  = np.nanmean(np.stack(vecs_b), axis=0)
            cos   = float(1.0 - cosine_dist(mu_a, mu_b))
            icc   = compute_icc(vecs_a, vecs_b)
            interp = _icc_interpretation(icc)

            cos_m[i, j] = cos_m[j, i] = cos
            pair = f"T{ta} vs T{tb}"
            print(f"  {pair:<20} {cos:>6.3f}  {str(icc):>7}  {interp}",
                  flush=True)

            rows.append({
                "comparison":    "acoustic_between_therapists",
                "label":         label_name,
                "group_a":       f"T{ta}",
                "group_b":       f"T{tb}",
                "n_a":           len(vecs_a),
                "n_b":           len(vecs_b),
                "cos_mean":      round(cos, 4),
                "icc":           icc,
                "icc_interp":    interp,
                "f1_ceiling":    round((1.0 + cos) / 2.0, 4),
            })

    t_labels = [f"T{t}" for t in therapists]
    plot_heatmap(
        cos_m, t_labels, t_labels,
        f"Between-therapist acoustic cosine similarity ({label_name})",
        os.path.join(outdir, f"between_therapists_cos_{label_name}.png"),
        cmap="RdYlGn", vmin=0, vmax=1,
    )

    return rows


def analyse_feature_importance(
    records:   list[dict],
    label_col: str,
    label_name: str,
    feat_names: list[str],
    outdir:    str,
) -> list[dict]:
    """
    Fit logistic regression on all features and rank by coefficient magnitude.
    Gives a sense of which acoustic features drive separability.
    """
    print(f"\n  Feature importance ({label_name}) …", flush=True)
    os.makedirs(outdir, exist_ok=True)

    valid = [r for r in records if r["features"] is not None]
    if not valid:
        return []

    X = np.nan_to_num(np.stack([r["features"] for r in valid]))
    y = np.array([r[label_col] for r in valid], dtype=int)

    if len(np.unique(y)) < 2:
        print("    Only one class present — skipping.", flush=True)
        return []

    scaler = StandardScaler()
    X_s    = scaler.fit_transform(X)
    clf    = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_s, y)

    coefs = clf.coef_[0]
    order = np.argsort(np.abs(coefs))[::-1]

    names_ordered = [feat_names[i] if i < len(feat_names)
                     else f"feat_{i:04d}" for i in order]
    coefs_ordered = coefs[order]

    print(f"  Top 10 features by |coef|:", flush=True)
    for name, coef in zip(names_ordered[:10], coefs_ordered[:10]):
        print(f"    {name:<35} {coef:>+.4f}", flush=True)

    rows = [{"comparison": "feature_importance", "label": label_name,
             "feature": names_ordered[k], "coef": round(float(coefs_ordered[k]), 4),
             "rank": k+1}
            for k in range(len(names_ordered))]

    # Bar chart of top 20
    top_n = min(20, len(names_ordered))
    fig, ax = plt.subplots(figsize=(8, max(4, top_n * 0.4)))
    colors = ["steelblue" if c > 0 else "tomato" for c in coefs_ordered[:top_n]]
    ax.barh(range(top_n), coefs_ordered[:top_n], color=colors, alpha=0.8)
    ax.set_yticks(range(top_n))
    ax.set_yticklabels(names_ordered[:top_n], fontsize=8)
    ax.invert_yaxis()
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Logistic regression coefficient")
    ax.set_title(f"Acoustic feature importance — {label_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(outdir, f"feature_importance_{label_name}.png"),
                bbox_inches="tight", dpi=150)
    plt.close()

    return rows


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Acoustic feature extraction and ceiling analysis.\n"
            "Extracts openSMILE/librosa features per turn, then computes\n"
            "separability ceiling for important vs non-important turns."
        )
    )

    parser.add_argument("--dir",         required=True,
                        help="Directory containing transcript CSV/TSV/XLSX files.")
    parser.add_argument("--audio_dir",   required=True,
                        help="Directory containing audio files (one per transcript). "
                             "Matched by transcript stem name.")
    parser.add_argument("--outdir",      default="acoustic_output/",
                        help="Directory for analysis outputs (heatmaps, summary CSV). "
                             "(default: acoustic_output/)")
    parser.add_argument("--features_dir", default="acoustic_features/",
                        help="Directory for per-transcript feature CSVs, one file "
                             "per transcript matching the transcript stem name. "
                             "(default: acoustic_features/)")
    parser.add_argument("--feature_set", default="egemaps",
                        choices=["egemaps", "compare", "custom"],
                        help="openSMILE feature set. 'egemaps' (88 features, "
                             "recommended), 'compare' (6373 features, slow), "
                             "'custom' (15 librosa features, no openSMILE needed). "
                             "(default: egemaps)")
    parser.add_argument("--start_col",   default="start_time",
                        help="Column name for turn start time in seconds. "
                             "(default: start_time)")
    parser.add_argument("--end_col",     default="end_time",
                        help="Column name for turn end time in seconds. "
                             "(default: end_time)")
    parser.add_argument("--n_cv_folds",  type=int, default=5,
                        help="Number of CV folds for ceiling classifier. "
                             "(default: 5)")

    args = parser.parse_args()

    dir_path   = Path(args.dir)
    audio_dir  = Path(args.audio_dir)
    if not dir_path.exists():
        raise ValueError(f"Transcript directory not found: {args.dir}")
    if not audio_dir.exists():
        print(f"Warning: audio directory not found: {args.audio_dir}. "
              f"Features will be None for all turns.", flush=True)
    os.makedirs(args.outdir, exist_ok=True)

    print(f"Acoustic Feature Analysis", flush=True)
    print(f"feature_set={args.feature_set}  "
          f"start_col={args.start_col}  end_col={args.end_col}", flush=True)

    print(f"\nExtracting features …", flush=True)
    records = load_records(
        dir_path, audio_dir, args.feature_set,
        args.start_col, args.end_col,
    )

    valid   = [r for r in records if r["features"] is not None]
    n_valid = len(valid)
    print(f"\nValid turns with features: {n_valid}/{len(records)}", flush=True)

    if n_valid == 0:
        print("No features extracted — check audio directory and column names.",
              flush=True)
        return

    sample_feat = valid[0]["features"]
    feat_names  = get_feature_names(args.feature_set, len(sample_feat))

    os.makedirs(args.features_dir, exist_ok=True)

    # Group records by transcript filename
    by_transcript: dict[str, list[dict]] = defaultdict(list)
    for r in records:
        by_transcript[r["filename"]].append(r)

    print(f"\n  Saving per-transcript feature CSVs to: {args.features_dir}",
          flush=True)
    for fname, recs in sorted(by_transcript.items()):
        feat_rows = []
        for r in recs:
            row = {k: v for k, v in r.items() if k != "features"}
            if r["features"] is not None:
                for name, val in zip(feat_names, r["features"]):
                    row[name] = float(val)
            feat_rows.append(row)

        stem     = Path(fname).stem
        out_path = os.path.join(args.features_dir, f"{stem}_acoustic.csv")
        pd.DataFrame(feat_rows).to_csv(out_path, index=False)
        n_with_feats = sum(1 for r in recs if r["features"] is not None)
        print(f"    {stem}_acoustic.csv  ({n_with_feats}/{len(recs)} turns "
              f"with features)", flush=True)

    all_rows: list[dict] = []

    # Separability ceiling (three-way)
    print(f"\n{'─'*60}", flush=True)
    print("  1. SEPARABILITY CEILING", flush=True)
    sep_dir = os.path.join(args.outdir, "separability")
    all_rows += analyse_separability(records, sep_dir)

    # Between-therapist consistency
    print(f"\n{'─'*60}", flush=True)
    print("  2. BETWEEN-THERAPIST CONSISTENCY", flush=True)
    bt_dir = os.path.join(args.outdir, "between_therapists")
    all_rows += analyse_between_therapists(
        records, "pat_label",  "patient_imp",   bt_dir
    )
    all_rows += analyse_between_therapists(
        records, "ther_label", "therapist_imp",  bt_dir
    )

    # Feature importance
    print(f"\n{'─'*60}", flush=True)
    print("  3. FEATURE IMPORTANCE", flush=True)
    fi_dir = os.path.join(args.outdir, "feature_importance")
    all_rows += analyse_feature_importance(
        records, "pat_label",  "patient_imp",   feat_names, fi_dir
    )
    all_rows += analyse_feature_importance(
        records, "ther_label", "therapist_imp",  feat_names, fi_dir
    )

    if all_rows:
        summary_path = os.path.join(args.outdir, "acoustic_summary.csv")
        pd.DataFrame(all_rows).to_csv(summary_path, index=False)
        print(f"\n  Saved: {summary_path}", flush=True)

        sep_rows = [r for r in all_rows if r["comparison"] == "acoustic_separability"]
        if sep_rows:
            print(f"\n  Separability ceiling summary:", flush=True)
            for r in sep_rows:
                f1_str = f"{r['f1_ceiling']:.4f}" if r["f1_ceiling"] is not None else "N/A"
                print(f"    {r['group_a']+' vs '+r['group_b']:<40} F1_ceil={f1_str}",
                      flush=True)

    print(f"\nDone. Outputs in: {args.outdir}", flush=True)


if __name__ == "__main__":
    main()