"""
Stage 4 — map Stage 2/3 features to Kandinsky-inspired visual parameters.

NO RENDERING in this stage. We only produce a `visual_parameters.json`
per piece plus a 'preview card' PNG so the mapping can be inspected
before Stage 5 commits paint to canvas.

## Grounding (research-reviewed, not Kandinsky-dogmatic)

Only one of Kandinsky's shape-colour pairs survives empirical replication:
  circle ↔ blue (Jacobsen 2002; Makin & Wuerger 2013).
Triangle-yellow and square-red are treated as *stylistic homage*, not
perceptual claims.

The empirically grounded cross-modal rules we use:
  • Palmer et al. 2013 PNAS — "Music-colour associations mediated by emotion":
      faster + major  → saturated, light, warm hue
      slower + minor  → desaturated, dark, cool hue
  • Spence 2011 — crossmodal correspondences:
      higher pitch  ↔ brighter, smaller, higher-on-canvas
  • Isbilen & Krumhansl 2016 — timbre → colour (brass→yellow/red,
      strings→blue), used only in soft weighting.
  • Albertazzi et al. 2013 JEP:HPP — curved↔cool, angular↔warm
      (partial Kandinsky support).

## Design axes

For each piece we compute three family parameters:

  PALETTE   : background hex + 3 element hexes
              Lightness driven by valence (gated by valence_confidence)
              Saturation driven by arousal
              Hue angle driven by spectral centroid (cool↔warm)
              Within-palette hue spread driven by chroma entropy
  GRAMMAR   : weighted vocabulary of Kandinsky forms
              {point, line, curve, circle, triangle, square}
              Weights driven by onset rate, flux, beat regularity
  COMPOSITION: how marks are placed on the canvas
              - total mark count: scales with onset count
              - size: base from RMS; variation from dynamic range
              - layout: grid / scatter / gestural, from beat regularity
              - timeline: 2.5-s windows mapped left→right

The result file for Stage 5 is JSON that fully determines the renderer
state (except seed and renderer version, which Stage 5 supplies).
"""

import json, pathlib, csv, colorsys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

STAGE2_DIR = pathlib.Path("stage2_analysis")
STAGE3_DIR = pathlib.Path("stage3_emotion")
OUT_DIR    = pathlib.Path("stage4_mapping")
OUT_DIR.mkdir(exist_ok=True)


# ----- helper mappings -----
def clip01(x, lo=0.0, hi=1.0):
    return float(np.clip(x, lo, hi))


def centroid_to_warmth_shift(centroid_hz):
    """Centroid shifts hue warm (small positive deg) or cool (negative).
    500 Hz → −20° (cooler), 4500 Hz → +20° (warmer)."""
    t = clip01((centroid_hz - 500) / (4500 - 500))
    return (t - 0.5) * 40.0


# Scriabin-informed pitch-class → hue angle (artistic hypothesis, declared).
# Not an empirical claim; chosen for differentiation across the corpus.
PC_HUE = {
    "C":  0,    "C#": 30,   "D":  60,   "D#": 90,
    "E":  120,  "F":  150,  "F#": 180,  "G":  210,
    "G#": 240,  "A":  270,  "A#": 300,  "B":  330,
}


def key_to_main_hue(dominant_pc_name, centroid_hz):
    """Primary hue from dominant pitch class; shifted ±20° by centroid."""
    base = PC_HUE.get(dominant_pc_name, 180)
    return (base + centroid_to_warmth_shift(centroid_hz)) % 360.0


def hsl_to_hex(h_deg, s, l):
    """h_deg in [0,360), s and l in [0,1]."""
    r, g, b = colorsys.hls_to_rgb(h_deg / 360.0, l, s)
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


def chroma_entropy(chroma_mean):
    p = np.asarray(chroma_mean, dtype=float)
    p = p / (p.sum() + 1e-9)
    p = np.clip(p, 1e-9, 1.0)
    return float(-(p * np.log2(p)).sum()) / np.log2(12)  # 0..1


def onset_regularity(onset_times):
    """Based on inter-onset intervals from the detector, not librosa's
    quantised beat_track output. 1 = metronomic; 0 = chaotic."""
    if len(onset_times) < 4:
        return 0.5
    ioi = np.diff(onset_times)
    cv  = float(np.std(ioi) / (np.mean(ioi) + 1e-9))
    # Typical musical cv: metronomic ≈ 0.1, swinging jazz ≈ 0.4,
    # polyrhythmic ≈ 0.6+. Map cv=0.1→1, cv=0.6→0.
    return clip01(1.0 - (cv - 0.1) / 0.5)


# ----- Palette generation -----
def build_palette(arousal, valence, valence_confidence,
                  centroid_hz, chroma_entropy_norm, dominant_pc_name):
    """Return dict with background + 3 element hex colours and the raw HSL.

    Palmer 2013 drives the empirically supported mappings:
      arousal → saturation,  valence → lightness (gated by confidence).
    Hue is picked from dominant pitch class (Scriabin-style; declared as
    an artistic hypothesis, not an empirical claim) and shifted ±20° by
    centroid (Spence 2011: brighter → warmer).
    Chroma entropy controls hue spread within the palette (unified vs varied).
    """
    sat = 0.30 + 0.60 * arousal

    push       = (valence - 0.5) * 2.0
    base_light = 0.50 + 0.22 * push * valence_confidence
    base_light = clip01(base_light + 0.05 * (arousal - 0.5))

    main_hue = key_to_main_hue(dominant_pc_name, centroid_hz)
    # Narrower spread: 12°..36° (analogous palette, not rainbow)
    hue_spread_deg = 12 + 24 * chroma_entropy_norm

    bg_light = clip01(base_light - 0.30)
    bg_sat   = clip01(sat * 0.35)
    bg_hex   = hsl_to_hex(main_hue, bg_sat, bg_light)

    # Analogous triad around main hue
    offsets  = [-hue_spread_deg, 0.0, +hue_spread_deg]
    elements = [hsl_to_hex((main_hue + o) % 360, sat, base_light)
                for o in offsets]

    accent = hsl_to_hex((main_hue + 180) % 360,
                        clip01(sat * 1.05), clip01(base_light + 0.18))

    return {
        "background": bg_hex,
        "elements":   elements,
        "accent":     accent,
        "hsl": {
            "main_hue_deg":    round(main_hue, 1),
            "dominant_pc":     dominant_pc_name,
            "saturation":      round(sat, 3),
            "base_lightness":  round(base_light, 3),
            "hue_spread_deg":  round(hue_spread_deg, 1),
        },
    }


# ----- Kandinsky-style vocabulary weights -----
def build_grammar(onsets_per_s, flux_mean, centroid_hz, arousal,
                  onset_reg, dynamic_range_db, rms_dbfs_med):
    """Weighted vocabulary of six Kandinsky-inspired forms.
    The dominant form is picked by decisive rules first, then every form
    gets a continuous weight so Stage-5 can still sample a mix."""

    flux_n   = clip01((np.log(max(flux_mean, 1.0)) - 2.0) / 5.0)
    onset_n  = clip01((onsets_per_s - 1.0) / 7.0)
    bright_n = clip01((centroid_hz - 800) / 3500)
    quiet_n  = clip01((-rms_dbfs_med - 15) / 20.0)   # 0 if loud, 1 if quiet

    angular = clip01(0.5 * flux_n + 0.3 * bright_n + 0.2 * onset_n)
    curvy   = 1.0 - angular

    # Decisive dominant-form rules — reflect musical character, not just
    # weighted sums (which collapsed on this corpus).
    if onsets_per_s >= 5.0:
        dominant = "point"                 # percussive density
    elif flux_mean > 400:
        dominant = "triangle"              # strong spectral change / aggression
    elif quiet_n > 0.6:
        dominant = "curve"                 # quiet lyrical texture
    elif dynamic_range_db > 10:
        dominant = "line"                  # dynamic contrast
    elif onset_reg > 0.6:
        dominant = "square"                # regular + mid-energy
    else:
        dominant = "circle"                # sustained mid-register

    # Soft weights (for sampling, not just the dominant)
    weights = {
        "point":    0.2 + 0.7 * onset_n,
        "line":     0.2 + 0.5 * clip01(dynamic_range_db / 15.0),
        "curve":    0.2 + 0.5 * curvy * quiet_n,
        "circle":   0.2 + 0.5 * curvy * (1 - quiet_n),
        "triangle": 0.15 + 0.7 * angular,
        "square":   0.15 + 0.6 * onset_reg * (1 - angular),
    }
    # Give the dominant a boost so sampling leans in the right direction
    weights[dominant] += 0.3
    total   = sum(weights.values())
    weights = {k: round(v / total, 3) for k, v in weights.items()}

    return {"weights": weights, "dominant_form": dominant,
            "angularity": round(angular, 3),
            "curvature":  round(curvy, 3)}


# ----- composition: layout, sizes, counts -----
def build_composition(onsets_total, duration_s, rms_dbfs_med, rms_spread_db,
                      onset_reg, arousal, tempo_bpm):
    mark_count = int(np.clip(onsets_total, 8, 400))

    size_base = 10 + 90 * clip01((rms_dbfs_med + 45) / 40)
    size_std  = size_base * clip01(rms_spread_db / 20.0) * 0.8

    if onset_reg > 0.65:
        layout = "grid"
    elif onset_reg > 0.35:
        layout = "scatter_metered"
    else:
        layout = "gestural"

    # Timeline mapping: 25-second segment maps left→right across canvas
    timeline = {
        "duration_s":  duration_s,
        "x_axis":      "left_to_right",
        "n_windows":   int(round(duration_s / 2.5)),
    }

    return {
        "mark_count":   mark_count,
        "size_base_px": round(size_base, 1),
        "size_std_px":  round(size_std, 1),
        "layout":       layout,
        "onset_regularity": round(onset_reg, 3),
        "arousal_density_scale": round(0.5 + 0.5 * arousal, 3),
        "tempo_bpm":    tempo_bpm,
        "timeline":     timeline,
    }


# ----- iterate -----
summary_rows = []
pieces_dirs = sorted([p for p in STAGE2_DIR.iterdir()
                      if p.is_dir() and (p / "features.npz").exists()])

for piece_dir in pieces_dirs:
    windowed = json.loads((piece_dir / "windowed.json").read_text())
    name   = windowed["piece"]
    whole2 = windowed["whole_segment"]
    arr    = np.load(piece_dir / "features.npz")

    traj   = json.loads((STAGE3_DIR / piece_dir.name / "trajectory.json").read_text())
    whole3 = traj["whole_segment"]

    print(f"\n=== {name} ===")

    chroma_mean = arr["chroma"].mean(axis=1).tolist()
    chroma_ent  = chroma_entropy(chroma_mean)
    onset_reg   = onset_regularity(arr["onset_times"])

    palette = build_palette(
        arousal=whole3["arousal"],
        valence=whole3["valence"],
        valence_confidence=whole3["valence_confidence"],
        centroid_hz=whole2["centroid_hz_med"],
        chroma_entropy_norm=chroma_ent,
        dominant_pc_name=whole2["dominant_pc"],
    )

    grammar = build_grammar(
        onsets_per_s=whole2["onsets_per_s"],
        flux_mean=whole2["flux_mean"],
        centroid_hz=whole2["centroid_hz_med"],
        arousal=whole3["arousal"],
        onset_reg=onset_reg,
        dynamic_range_db=whole2["rms_dbfs_p90p10"],
        rms_dbfs_med=whole2["rms_dbfs_med"],
    )

    composition = build_composition(
        onsets_total=whole2["n_onsets"],
        duration_s=whole2["duration_s"],
        rms_dbfs_med=whole2["rms_dbfs_med"],
        rms_spread_db=whole2["rms_dbfs_p90p10"],
        onset_reg=onset_reg,
        arousal=whole3["arousal"],
        tempo_bpm=whole2["tempo_bpm"],
    )

    # Per-window intensity trajectory, usable by Stage-5 renderer
    window_track = []
    for s3w, s2w in zip(traj["trajectory"], windowed["windows"]):
        window_track.append({
            "t_mid_s":   s3w["t_mid_s"],
            "arousal":   s3w["arousal"],
            "valence":   s3w["valence"],
            "valence_confidence": s3w["valence_confidence"],
            "onsets_per_s": s2w["onsets_per_s"],
            "rms_dbfs":  s2w["rms_dbfs_med"],
            "centroid_hz": s2w["centroid_hz_med"],
        })

    params = {
        "piece": name,
        "source": {
            "segment_wav": str((STAGE2_DIR / piece_dir.name).resolve()),
            "stage2_summary": whole2,
            "stage3_summary": whole3,
        },
        "palette":     palette,
        "grammar":     grammar,
        "composition": composition,
        "onset_regularity": round(onset_reg, 3),
        "chroma_entropy":   round(chroma_ent, 3),
        "windows":     window_track,
        "uncertainty": {
            "valence_confidence": whole3["valence_confidence"],
            "notes": [
                "Palette lightness scaled by valence_confidence; low "
                "confidence → palette collapses toward mid-lightness, "
                "preventing confident hue claims when the valence "
                "estimate is weak.",
                "Shape vocabulary weights are proportional, not absolute: "
                "Stage 5 should sample from this distribution with a seed.",
                "Only circle↔blue has strong empirical support "
                "(Jacobsen 2002); other Kandinsky shape↔colour pairings "
                "are stylistic.",
            ],
        },
        "mapping_version": "stage4-v1",
    }

    out_dir = OUT_DIR / piece_dir.name
    out_dir.mkdir(exist_ok=True)
    (out_dir / "visual_parameters.json").write_text(json.dumps(params, indent=2))

    print(f"  bg={palette['background']} elems={palette['elements']} "
          f"accent={palette['accent']}")
    print(f"  dominant form = {grammar['dominant_form']}   "
          f"layout = {composition['layout']}")
    print(f"  marks = {composition['mark_count']}  size≈{composition['size_base_px']}px  "
          f"onset-reg={composition['onset_regularity']:.2f}")

    summary_rows.append({
        "piece": name,
        "bg": palette["background"],
        "elem1": palette["elements"][0],
        "elem2": palette["elements"][1],
        "elem3": palette["elements"][2],
        "accent": palette["accent"],
        "dominant_form": grammar["dominant_form"],
        "layout": composition["layout"],
        "mark_count": composition["mark_count"],
        "arousal": whole3["arousal"],
        "valence": whole3["valence"],
        "valence_confidence": whole3["valence_confidence"],
    })

    # ---------- Per-piece preview card ----------
    fig = plt.figure(figsize=(10, 6))
    gs  = fig.add_gridspec(3, 6, hspace=0.5, wspace=0.4)

    # Palette strip
    ax = fig.add_subplot(gs[0, :])
    ax.set_xlim(0, 5); ax.set_ylim(0, 1); ax.axis("off")
    for i, hx in enumerate([palette["background"]] + palette["elements"] + [palette["accent"]]):
        ax.add_patch(mpatches.Rectangle((i, 0), 0.95, 1, facecolor=hx, edgecolor="black"))
        label = ["background", "elem 1", "elem 2", "elem 3", "accent"][i]
        ax.text(i + 0.475, -0.15, f"{label}\n{hx}", ha="center", fontsize=8)
    ax.text(0, 1.15, f"{name}   A={whole3['arousal']:.2f}  V={whole3['valence']:.2f}"
                     f"  (conf={whole3['valence_confidence']:.2f})",
            fontsize=11, weight="bold")

    # Grammar bar chart
    ax = fig.add_subplot(gs[1, 0:3])
    forms = list(grammar["weights"].keys())
    vals  = [grammar["weights"][f] for f in forms]
    ax.barh(forms, vals, color="#444")
    ax.set_xlim(0, max(vals) * 1.3)
    ax.set_title("Shape vocabulary weights", fontsize=10, loc="left")
    ax.invert_yaxis()
    for i, v in enumerate(vals):
        ax.text(v + 0.005, i, f"{v:.2f}", va="center", fontsize=8)

    # Composition facts
    ax = fig.add_subplot(gs[1, 3:])
    ax.axis("off")
    txt = [
        f"Dominant form : {grammar['dominant_form']}",
        f"Layout        : {composition['layout']}",
        f"Marks (onsets): {composition['mark_count']}",
        f"Size base     : {composition['size_base_px']} px (±{composition['size_std_px']})",
        f"Onset reg.    : {composition['onset_regularity']:.2f}  (1=metronomic)",
        f"Angular/curvy : {grammar['angularity']:.2f} / {grammar['curvature']:.2f}",
        f"Key estimate  : {whole3['key_estimate']}",
    ]
    ax.text(0.0, 1.0, "\n".join(txt), va="top", family="monospace", fontsize=9)

    # Window intensity timeline (arousal + onsets/s)
    ax = fig.add_subplot(gs[2, :])
    t = [w["t_mid_s"] for w in window_track]
    A = [w["arousal"] for w in window_track]
    O = [w["onsets_per_s"] / 8.0 for w in window_track]
    ax.plot(t, A, color="#c0392b", lw=1.8, label="arousal")
    ax.plot(t, O, color="#2874a6", lw=1.2, ls="--", label="onsets/s ÷ 8")
    ax.set_ylim(0, 1); ax.set_xlabel("time within segment (s)")
    ax.set_title("Per-window visual-intensity drivers", fontsize=10, loc="left")
    ax.legend(fontsize=8)

    fig.savefig(out_dir / "preview.png", dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote → {out_dir}/visual_parameters.json, preview.png")


# ----- cross-piece table and overview -----
(OUT_DIR / "summary.json").write_text(json.dumps(summary_rows, indent=2))
with open(OUT_DIR / "summary.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(summary_rows[0].keys()))
    w.writeheader()
    for r in summary_rows: w.writerow(r)

# Overview grid: one row per piece, palette + grammar + layout tag
fig, axes = plt.subplots(len(summary_rows), 1, figsize=(11, 1.6 * len(summary_rows)))
for ax, row in zip(axes, summary_rows):
    ax.set_xlim(0, 10); ax.set_ylim(0, 1); ax.axis("off")
    swatches = [row["bg"], row["elem1"], row["elem2"], row["elem3"], row["accent"]]
    for i, hx in enumerate(swatches):
        ax.add_patch(mpatches.Rectangle((i * 0.9, 0.15), 0.85, 0.7,
                                         facecolor=hx, edgecolor="black", lw=0.5))
    ax.text(5.0, 0.55,
            f"dominant: {row['dominant_form']:>8s}   "
            f"layout: {row['layout']:>16s}   "
            f"marks: {row['mark_count']:>3d}   "
            f"A={row['arousal']:.2f}  V={row['valence']:.2f}",
            family="monospace", fontsize=9, va="center")
    ax.text(0.0, 1.02, row["piece"], fontsize=11, weight="bold")

fig.suptitle("Stage 4 — visual-parameter overview", fontsize=12, y=1.00)
fig.tight_layout()
fig.savefig(OUT_DIR / "overview.png", dpi=140, bbox_inches="tight")
plt.close(fig)
print(f"\nwrote → {OUT_DIR}/overview.png")
print(f"wrote → {OUT_DIR}/summary.json  and  summary.csv")
