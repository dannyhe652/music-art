"""
Stage 1 audio quality check.

For each decoded WAV:
  - sample rate, channels, bit depth, duration
  - peak level and clipping count (samples within 0.001 of ±1.0)
  - silence fractions (RMS below -70 dBFS per 100 ms frame)
  - dynamic range summary (10th–90th pRMS percentile, excluding silence/edges)
  - RMS envelope over time
  - recommended 25-second segment with highest timbral variety
    (maximise sum of spectral centroid std-dev across 1-second frames)

Writes:
  stage1_qc/
    report.txt          human-readable summary
    <piece>_rms.png     RMS waveform with recommended segment highlighted
"""

import json, pathlib, textwrap
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

AUDIO_DIR = pathlib.Path("1. Music Extraction/outputs/decoded-audio")
OUT_DIR   = pathlib.Path("stage1_qc")
OUT_DIR.mkdir(exist_ok=True)

SILENCE_THRESH_DBFS = -70.0
CLIP_MARGIN         = 0.001          # |sample| >= 1 - margin counts as clip
FRAME_HOP_S         = 0.1            # 100 ms frames for silence/RMS
TARGET_SEGMENT_S    = 25             # target segment length in seconds
VARIETY_FRAME_S     = 1.0            # window for centroid std-dev scoring
SR_ANALYSIS         = 22050          # resample to this for librosa analysis

pieces = sorted(AUDIO_DIR.glob("*.wav"))
assert pieces, f"No WAV files found in {AUDIO_DIR}"

report_lines = []
manifest     = []

def rms_dbfs(samples):
    rms = np.sqrt(np.mean(samples ** 2))
    if rms < 1e-10:
        return -120.0
    return 20 * np.log10(rms)


def windowed_rms_dbfs(y, sr, hop_s=0.1):
    hop = int(sr * hop_s)
    n_frames = len(y) // hop
    vals = []
    for i in range(n_frames):
        chunk = y[i * hop : (i + 1) * hop]
        vals.append(rms_dbfs(chunk))
    return np.array(vals), np.arange(n_frames) * hop_s


def best_segment(y, sr, seg_s=TARGET_SEGMENT_S, frame_s=VARIETY_FRAME_S):
    """Return (start_s, end_s) of the most tonally varied segment of length seg_s."""
    frame_len = int(sr * frame_s)
    n_frames  = len(y) // frame_len
    centroids = []
    for i in range(n_frames):
        chunk = y[i * frame_len : (i + 1) * frame_len]
        c = librosa.feature.spectral_centroid(y=chunk, sr=sr)[0]
        centroids.append(float(np.std(c)))
    centroids = np.array(centroids)

    seg_frames = int(seg_s / frame_s)
    if len(centroids) < seg_frames:
        return 0.0, float(len(y) / sr)

    # sliding window sum
    window_scores = np.convolve(centroids, np.ones(seg_frames), mode="valid")
    best_start_frame = int(np.argmax(window_scores))
    start_s = best_start_frame * frame_s
    end_s   = start_s + seg_s
    return start_s, end_s


for wav_path in pieces:
    name = wav_path.stem
    print(f"\n{'='*60}\n{name}")

    # --- load full file ---
    data, sr = sf.read(str(wav_path), dtype="float32", always_2d=True)
    duration  = len(data) / sr
    channels  = data.shape[1]

    # mix to mono for analysis
    mono = data.mean(axis=1)

    # --- clipping ---
    clip_count = int(np.sum(np.abs(mono) >= 1.0 - CLIP_MARGIN))
    peak_dbfs  = 20 * np.log10(np.max(np.abs(mono)) + 1e-10)

    # --- RMS per 100 ms frame ---
    frame_rms, frame_times = windowed_rms_dbfs(mono, sr, FRAME_HOP_S)
    silence_mask   = frame_rms < SILENCE_THRESH_DBFS
    silence_frac   = silence_mask.mean()
    active_rms     = frame_rms[~silence_mask]

    if len(active_rms) > 0:
        p10 = float(np.percentile(active_rms, 10))
        p90 = float(np.percentile(active_rms, 90))
        dyn_range = p90 - p10
        median_rms = float(np.median(active_rms))
    else:
        p10 = p90 = dyn_range = median_rms = float("nan")

    # --- resample for librosa analysis ---
    y_lr  = librosa.resample(mono, orig_sr=sr, target_sr=SR_ANALYSIS)
    start_s, end_s = best_segment(y_lr, SR_ANALYSIS)

    # --- build text report for this piece ---
    lines = [
        f"\n## {name}",
        f"  Duration      : {duration:.1f} s",
        f"  Sample rate   : {sr} Hz  |  Channels: {channels}",
        f"  Peak level    : {peak_dbfs:.1f} dBFS",
        f"  Clip samples  : {clip_count}  {'⚠ CLIPPING' if clip_count > 0 else '(none)'}",
        f"  Silence       : {silence_frac*100:.1f}% of frames (< {SILENCE_THRESH_DBFS} dBFS)",
        f"  Median RMS    : {median_rms:.1f} dBFS  (active frames only)",
        f"  Dynamic range : {dyn_range:.1f} dB  (p90–p10 of active frames)",
        f"  Recommended segment: {start_s:.1f}–{end_s:.1f} s  (most timbral variety)",
    ]
    report_lines.extend(lines)
    for ln in lines:
        print(ln)

    # --- RMS plot ---
    fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(frame_times, frame_rms, lw=0.8, color="#444")
    ax.axhline(SILENCE_THRESH_DBFS, color="steelblue", lw=0.8, ls="--", label="silence threshold")
    ax.axvspan(start_s, end_s, alpha=0.18, color="orange", label=f"recommended {TARGET_SEGMENT_S}s")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("RMS (dBFS)")
    ax.set_title(name)
    ax.set_ylim(-80, 2)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_png = OUT_DIR / f"{name.replace(' ', '_')}_rms.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"  Plot saved → {out_png}")

    manifest.append({
        "name": name,
        "wav": str(wav_path),
        "duration_s": round(duration, 2),
        "sample_rate": sr,
        "channels": channels,
        "peak_dbfs": round(peak_dbfs, 2),
        "clip_samples": clip_count,
        "silence_fraction": round(float(silence_frac), 4),
        "median_rms_dbfs": round(median_rms, 2),
        "dynamic_range_db": round(dyn_range, 2),
        "recommended_segment": {"start_s": round(start_s, 2), "end_s": round(end_s, 2)},
    })

# --- write report ---
header = textwrap.dedent(f"""\
    # Stage 1 Audio Quality Check
    Source: {AUDIO_DIR}
    Files checked: {len(pieces)}
    Silence threshold: {SILENCE_THRESH_DBFS} dBFS
    Clipping threshold: |sample| >= {1 - CLIP_MARGIN:.3f}
    Recommended segment length: {TARGET_SEGMENT_S} s
    Selection criterion: maximum spectral-centroid standard deviation
      (most timbral variety within a {TARGET_SEGMENT_S}-second window)
""")

report_text = header + "\n".join(report_lines)
(OUT_DIR / "report.txt").write_text(report_text)
(OUT_DIR / "manifest.json").write_text(json.dumps(manifest, indent=2, default=lambda x: float(x) if hasattr(x, "__float__") else str(x)))
print(f"\nReport → {OUT_DIR}/report.txt")
print(f"Manifest → {OUT_DIR}/manifest.json")
