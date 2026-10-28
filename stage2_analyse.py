"""
Stage 2 — acoustic analysis of the 25-second segments.

For each segment WAV we compute six families of features from the roadmap:
  1. Tempo and beat positions         (librosa.beat.beat_track)
  2. Onset density per 1-s window     (librosa.onset.onset_detect)
  3. RMS energy per frame             (librosa.feature.rms)
  4. Spectral centroid per frame      (librosa.feature.spectral_centroid)
  5. Chroma (12 pitch classes)        (librosa.feature.chroma_cqt)
  6. Spectral flux per frame          (sum of positive STFT magnitude differences)

Features are computed over short frames (~23 ms at sr=22050, hop 512 ≈ 23 ms)
AND summarised over 2.5-second windows with 50% overlap (keeping the timeline).

Per-piece output in stage2_analysis/<piece>/:
  features.npz           raw per-frame arrays
  windowed.json          2.5s-window summaries + whole-segment summary
  diagnostic.png         6-panel plot (one per feature family)

Cross-piece output in stage2_analysis/:
  comparison.png         1-panel-per-feature grid comparing all 5 pieces
  summary.json           whole-segment summary table
  summary.csv            same, human readable
"""
import json, pathlib, csv
import numpy as np
import soundfile as sf
import librosa
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SEGMENTS_DIR = pathlib.Path("stage1_qc/segments")
OUT_DIR      = pathlib.Path("stage2_analysis")
OUT_DIR.mkdir(exist_ok=True)

SR          = 22050      # analysis sample rate
HOP         = 512        # ~23 ms at 22050 Hz
N_FFT       = 2048       # ~93 ms STFT window
WIN_S       = 2.5        # summary window length
WIN_OVERLAP = 0.5        # 50 %

segments_manifest = json.loads((SEGMENTS_DIR / "segments_manifest.json").read_text())

summary_rows = []

for item in segments_manifest:
    name = item["name"]
    wav  = pathlib.Path(item["segment_wav"])
    piece_dir = OUT_DIR / name.replace(" ", "_").replace("/", "-")
    piece_dir.mkdir(exist_ok=True)
    print(f"\n=== {name} ===")

    # --- Load segment, mono, resampled ---
    data, sr0 = sf.read(str(wav), dtype="float32", always_2d=True)
    y = data.mean(axis=1)
    if sr0 != SR:
        y = librosa.resample(y, orig_sr=sr0, target_sr=SR)
    duration = len(y) / SR
    times = librosa.times_like(y[::HOP], sr=SR, hop_length=HOP, n_fft=N_FFT)

    # ---------- 1. Tempo and beats ----------
    tempo, beat_frames = librosa.beat.beat_track(y=y, sr=SR, hop_length=HOP)
    beat_times = librosa.frames_to_time(beat_frames, sr=SR, hop_length=HOP)
    tempo_bpm  = float(np.atleast_1d(tempo)[0])
    print(f"  tempo = {tempo_bpm:.1f} BPM   ({len(beat_times)} beats detected)")

    # ---------- 2. Onsets and onset density ----------
    onset_frames = librosa.onset.onset_detect(y=y, sr=SR, hop_length=HOP)
    onset_times  = librosa.frames_to_time(onset_frames, sr=SR, hop_length=HOP)
    # per-1-s onset density across the segment timeline
    bin_edges = np.arange(0, duration + 1.0, 1.0)
    onset_density, _ = np.histogram(onset_times, bins=bin_edges)
    print(f"  onsets = {len(onset_times)}  (mean {len(onset_times)/duration:.2f}/s)")

    # ---------- 3. RMS energy ----------
    rms = librosa.feature.rms(y=y, frame_length=N_FFT, hop_length=HOP)[0]
    rms_dbfs = 20 * np.log10(np.maximum(rms, 1e-10))

    # ---------- 4. Spectral centroid ----------
    centroid = librosa.feature.spectral_centroid(y=y, sr=SR, n_fft=N_FFT, hop_length=HOP)[0]

    # ---------- 5. Chroma (CQT-based, stable for mixed music) ----------
    chroma = librosa.feature.chroma_cqt(y=y, sr=SR, hop_length=HOP)   # (12, T)
    chroma_mean = chroma.mean(axis=1)
    dominant_pc = int(np.argmax(chroma_mean))
    pc_names = ["C", "C#", "D", "D#", "E", "F", "F#", "G", "G#", "A", "A#", "B"]
    print(f"  dominant pitch class = {pc_names[dominant_pc]}")

    # ---------- 6. Spectral flux ----------
    S = np.abs(librosa.stft(y, n_fft=N_FFT, hop_length=HOP))
    flux = np.sum(np.maximum(0.0, np.diff(S, axis=1)), axis=0)
    flux = np.concatenate([[0.0], flux])   # align to T frames

    # Make per-frame times align
    T = min(len(rms), len(centroid), chroma.shape[1], len(flux))
    rms      = rms[:T]
    rms_dbfs = rms_dbfs[:T]
    centroid = centroid[:T]
    chroma   = chroma[:, :T]
    flux     = flux[:T]
    frame_times = librosa.frames_to_time(np.arange(T), sr=SR, hop_length=HOP)

    # ---------- 2.5-s window summaries (50% overlap) ----------
    step_s = WIN_S * (1 - WIN_OVERLAP)
    starts = np.arange(0, max(1e-6, duration - WIN_S) + 1e-6, step_s)
    windows = []
    for s in starts:
        e = s + WIN_S
        mask = (frame_times >= s) & (frame_times < e)
        if mask.sum() < 2:
            continue
        onset_mask = (onset_times >= s) & (onset_times < e)
        windows.append({
            "start_s":   float(round(s, 2)),
            "end_s":     float(round(e, 2)),
            "rms_dbfs_med":   float(round(np.median(rms_dbfs[mask]), 2)),
            "rms_dbfs_p90p10":float(round(np.percentile(rms_dbfs[mask], 90) -
                                          np.percentile(rms_dbfs[mask], 10), 2)),
            "centroid_hz_med":float(round(np.median(centroid[mask]), 1)),
            "centroid_hz_std":float(round(np.std(centroid[mask]), 1)),
            "onset_count":    int(onset_mask.sum()),
            "onsets_per_s":   float(round(onset_mask.sum() / WIN_S, 2)),
            "flux_mean":      float(round(np.mean(flux[mask]), 3)),
            "chroma_mean":    [float(round(x, 3)) for x in chroma[:, mask].mean(axis=1)],
        })

    # Whole-segment summary (for cross-piece comparison)
    whole = {
        "piece": name,
        "duration_s": round(duration, 2),
        "tempo_bpm":  round(tempo_bpm, 1),
        "n_beats":    len(beat_times),
        "n_onsets":   len(onset_times),
        "onsets_per_s": round(len(onset_times) / duration, 2),
        "rms_dbfs_med":    round(float(np.median(rms_dbfs)), 2),
        "rms_dbfs_p90p10": round(float(np.percentile(rms_dbfs, 90) -
                                       np.percentile(rms_dbfs, 10)), 2),
        "centroid_hz_med": round(float(np.median(centroid)), 1),
        "flux_mean":       round(float(np.mean(flux)), 3),
        "dominant_pc":     pc_names[dominant_pc],
    }
    summary_rows.append(whole)
    print(f"  summary → {whole}")

    # ---------- Save arrays and windowed summary ----------
    np.savez(piece_dir / "features.npz",
             frame_times=frame_times, rms=rms, rms_dbfs=rms_dbfs,
             centroid=centroid, chroma=chroma, flux=flux,
             beat_times=beat_times, onset_times=onset_times,
             onset_density=onset_density, bin_edges=bin_edges)
    (piece_dir / "windowed.json").write_text(json.dumps({
        "piece": name, "duration_s": round(duration, 2),
        "tempo_bpm": round(tempo_bpm, 1),
        "window_s": WIN_S, "overlap": WIN_OVERLAP,
        "windows": windows, "whole_segment": whole,
    }, indent=2))

    # ---------- Diagnostic plot (6 panels) ----------
    fig, axes = plt.subplots(6, 1, figsize=(12, 11), sharex=True)

    ax = axes[0]
    ax.plot(frame_times, rms_dbfs, lw=0.8, color="#333")
    for b in beat_times:
        ax.axvline(b, color="tab:red", alpha=0.25, lw=0.5)
    ax.set_ylabel("RMS (dBFS)")
    ax.set_title(f"{name}  —  tempo ≈ {tempo_bpm:.1f} BPM (red = beats)",
                 fontsize=11, loc="left")
    ax.set_ylim(-60, 0)

    ax = axes[1]
    bin_centres = (bin_edges[:-1] + bin_edges[1:]) / 2
    ax.bar(bin_centres, onset_density, width=0.9, color="#4a90c7", alpha=0.8)
    for o in onset_times:
        ax.axvline(o, color="black", alpha=0.15, lw=0.3)
    ax.set_ylabel("onsets / s")

    ax = axes[2]
    ax.plot(frame_times, centroid, lw=0.8, color="#b35900")
    ax.set_ylabel("centroid (Hz)")
    ax.set_ylim(0, max(4000, float(np.max(centroid)) * 1.05))

    ax = axes[3]
    ax.plot(frame_times, flux, lw=0.6, color="#5a5")
    ax.set_ylabel("spectral flux")

    ax = axes[4]
    im = ax.imshow(chroma, aspect="auto", origin="lower",
                   extent=[frame_times[0], frame_times[-1], 0, 12],
                   cmap="magma", vmin=0, vmax=1)
    ax.set_yticks(np.arange(12) + 0.5)
    ax.set_yticklabels(pc_names, fontsize=8)
    ax.set_ylabel("chroma")

    ax = axes[5]
    ax.bar(np.arange(12), chroma_mean, color="#6a4c93")
    ax.set_xticks(np.arange(12))
    ax.set_xticklabels(pc_names)
    ax.set_ylabel("mean chroma")
    ax.set_xlabel("Time (s)   |   bottom panel: pitch-class distribution over whole segment")

    fig.tight_layout()
    fig.savefig(piece_dir / "diagnostic.png", dpi=140)
    plt.close(fig)
    print(f"  wrote → {piece_dir}/diagnostic.png")

# ---------- Cross-piece comparison ----------
print("\n=== Cross-piece comparison ===")
(OUT_DIR / "summary.json").write_text(json.dumps(summary_rows, indent=2))
with open(OUT_DIR / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    w.writeheader()
    for r in summary_rows:
        w.writerow(r)

# Grid plot: RMS, centroid, chroma per piece, aligned columns
n_pieces = len(summary_rows)
fig, axes = plt.subplots(3, n_pieces, figsize=(4 * n_pieces, 7), sharey="row")
for col, item in enumerate(segments_manifest):
    name = item["name"]
    piece_dir = OUT_DIR / name.replace(" ", "_").replace("/", "-")
    arr = np.load(piece_dir / "features.npz")
    t = arr["frame_times"]

    axes[0, col].plot(t, arr["rms_dbfs"], lw=0.7, color="#333")
    axes[0, col].set_ylim(-60, 0)
    axes[0, col].set_title(name, fontsize=9)
    if col == 0: axes[0, col].set_ylabel("RMS (dBFS)")

    axes[1, col].plot(t, arr["centroid"], lw=0.7, color="#b35900")
    axes[1, col].set_ylim(0, 5000)
    if col == 0: axes[1, col].set_ylabel("centroid (Hz)")

    axes[2, col].imshow(arr["chroma"], aspect="auto", origin="lower",
                       extent=[t[0], t[-1], 0, 12], cmap="magma", vmin=0, vmax=1)
    axes[2, col].set_yticks(np.arange(12) + 0.5)
    axes[2, col].set_yticklabels(["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"],
                                 fontsize=7)
    axes[2, col].set_xlabel("time (s)")
    if col == 0: axes[2, col].set_ylabel("chroma")

fig.suptitle("Stage 2 — RMS, spectral centroid, chroma across all 5 pieces", fontsize=11)
fig.tight_layout(rect=[0, 0, 1, 0.97])
fig.savefig(OUT_DIR / "comparison.png", dpi=140)
plt.close(fig)
print(f"wrote → {OUT_DIR}/comparison.png")
print(f"wrote → {OUT_DIR}/summary.json and summary.csv")
