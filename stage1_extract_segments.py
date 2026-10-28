"""
Stage 1 — extract the recommended 25-second segments as lossless 32-bit float WAVs.
Reads segment times from stage1_qc/manifest.json.
Writes to stage1_qc/segments/.

Each segment file records:
  - source WAV, segment boundaries, reason for selection
  - peak and RMS levels of the extracted segment
"""

import json, pathlib
import numpy as np
import soundfile as sf

MANIFEST = pathlib.Path("stage1_qc/manifest.json")
OUT_DIR  = pathlib.Path("stage1_qc/segments")
OUT_DIR.mkdir(exist_ok=True)

# Manual overrides where the algorithm's max-variety criterion chose a poor segment.
# Algorithm maximises spectral-centroid std-dev, which is attracted to level transitions
# and silence edges rather than coherent musical content.
#
# Discipline: algorithm chose 48–73s (quieter second section ending in silent edit).
#   Override to 5–30s: the characteristic dense interlocking rhythms, no silence risk.
#
# Starless: algorithm chose 39–64s (climax decay → brief near-silence at 63s → brief
#   return). Override to 25–50s: captures the sustained loud section plus the very
#   start of the dramatic drop — a cleaner arc without the fragmentary epilogue.
MANUAL_OVERRIDES = {
    "King Crimson - Discipline": (5.0, 30.0),
    "King Crimson - Starless":   (25.0, 50.0),
}

manifest = json.loads(MANIFEST.read_text())
segment_manifest = []

for item in manifest:
    name   = item["name"]
    wav_in = pathlib.Path(item["wav"])
    if name in MANUAL_OVERRIDES:
        start_s, end_s = MANUAL_OVERRIDES[name]
        selection_note = "manual override — see script for reasoning"
    else:
        seg    = item["recommended_segment"]
        start_s, end_s = seg["start_s"], seg["end_s"]
        selection_note = f"maximum spectral-centroid std-dev in a {end_s-start_s:.0f}s sliding window"

    data, sr = sf.read(str(wav_in), dtype="float32", always_2d=True)
    start_i  = int(start_s * sr)
    end_i    = int(end_s   * sr)
    segment  = data[start_i:end_i, :]

    mono     = segment.mean(axis=1)
    peak_dbfs  = float(20 * np.log10(max(np.max(np.abs(mono)), 1e-10)))
    rms_dbfs   = float(20 * np.log10(max(np.sqrt(np.mean(mono**2)), 1e-10)))
    duration_s = float(len(segment) / sr)

    out_name = name.replace(" ", "_").replace("/", "-") + "_segment.wav"
    out_path = OUT_DIR / out_name
    sf.write(str(out_path), segment, sr, subtype="FLOAT")

    print(f"{name}: {start_s:.1f}–{end_s:.1f}s → {out_path.name}  peak={peak_dbfs:.1f}dBFS  rms={rms_dbfs:.1f}dBFS")

    segment_manifest.append({
        "name": name,
        "source_wav": str(wav_in),
        "source_sha256": item.get("source_sha256", "see decoded-audio/manifest.json"),
        "segment_wav": str(out_path),
        "start_s": start_s,
        "end_s": end_s,
        "duration_s": round(duration_s, 3),
        "sample_rate": sr,
        "channels": data.shape[1],
        "subtype": "32-bit float PCM",
        "peak_dbfs": round(peak_dbfs, 2),
        "rms_dbfs": round(rms_dbfs, 2),
        "selection": selection_note,
    })

(OUT_DIR / "segments_manifest.json").write_text(json.dumps(segment_manifest, indent=2))
print(f"\nSegment manifest → {OUT_DIR}/segments_manifest.json")
