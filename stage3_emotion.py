"""
Stage 3 — emotional trajectory (expressive-feature V/A score).

NOT a validated music emotion recognition system. This is a transparent
rule-based estimate built on acoustic features whose links to perceived
arousal and valence are supported in MIR research:

  Eerola & Vuoskoski (2011),  Topics in Cognitive Science
  Panda et al. (2018),        "Novel audio features for MER"
  Deaf-perception work:       Caldwell (2015), Russo (2012), Tranchant (2017)

Design decisions, based on the research:
  * AROUSAL is treated as high-confidence. The acoustic cues (RMS, centroid,
    onset rate, flux) are well established, and arousal ratings converge
    between deaf and hearing listeners (Caldwell et al. 2015).
  * VALENCE is treated as LOW-confidence. Mode-based cues (major/minor) are
    largely inaccessible to CI users and profoundly deaf listeners; harmonic
    complexity is noisy on mixed recordings. We compute a valence score but
    flag its uncertainty, and the output explicitly carries a
    'valence_confidence' field.

Per-piece output:  stage3_emotion/<piece>/
  trajectory.json    windowed V/A plus the per-feature evidence
  diagnostic.png     3-panel plot — arousal, valence, components

Cross-piece output: stage3_emotion/
  summary.json / summary.csv
  circumplex.png     V/A scatter + trajectory curves
  arousal_bars.png   mean A per piece with 10-90 pct bars
"""
import json, pathlib, csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

STAGE2_DIR = pathlib.Path("stage2_analysis")
OUT_DIR    = pathlib.Path("stage3_emotion")
OUT_DIR.mkdir(exist_ok=True)

WIN_S        = 2.5
WIN_OVERLAP  = 0.5

# ----- normalization reference ranges (dBFS, Hz, onsets/s, flux) -----
# Chosen from the research literature and sanity-checked against this corpus.
# Any value outside the range is clipped. Not statistically calibrated.
RMS_DBFS_RANGE   = (-45.0,  -8.0)   # softer vs louder
CENTROID_RANGE   = (500.0, 4500.0)  # darker vs brighter
ONSETS_PER_S_RANGE = (0.5,  8.0)
FLUX_RANGE_LOG   = (1.5, 7.0)       # natural log of flux

# Krumhansl-Kessler key profiles (perception-weighted pitch-class salience)
KK_MAJOR = np.array([6.35,2.23,3.48,2.33,4.38,4.09,2.52,5.19,2.39,3.66,2.29,2.88])
KK_MINOR = np.array([6.33,2.68,3.52,5.38,2.60,3.53,2.54,4.75,3.98,2.69,3.34,3.17])

PC_NAMES = ["C","C#","D","D#","E","F","F#","G","G#","A","A#","B"]


def clip01(x, lo, hi):
    return float(np.clip((x - lo) / (hi - lo), 0.0, 1.0))


def krumhansl_key(chroma_mean):
    """Return (key_name, mode, correlation_strength, major_minor_margin).
    major_minor_margin > 0 means major fits better; < 0 means minor.
    """
    chroma_mean = np.asarray(chroma_mean)
    chroma_mean = chroma_mean / (chroma_mean.sum() + 1e-9)
    best = {"corr": -2.0}
    for tonic in range(12):
        for name, prof in (("major", KK_MAJOR), ("minor", KK_MINOR)):
            rotated = np.roll(prof, tonic) / prof.sum()
            corr = float(np.corrcoef(chroma_mean, rotated)[0, 1])
            if corr > best["corr"]:
                best = {"tonic": tonic, "mode": name, "corr": corr}
    # Also compute best major vs best minor to get the margin
    best_major = max(
        float(np.corrcoef(chroma_mean, np.roll(KK_MAJOR, t) / KK_MAJOR.sum())[0, 1])
        for t in range(12))
    best_minor = max(
        float(np.corrcoef(chroma_mean, np.roll(KK_MINOR, t) / KK_MINOR.sum())[0, 1])
        for t in range(12))
    return {
        "key": f"{PC_NAMES[best['tonic']]} {best['mode']}",
        "mode": best["mode"],
        "corr": best["corr"],
        "major_minor_margin": best_major - best_minor,  # + = major-sounding
    }


def chroma_entropy(chroma_vec):
    """Entropy of the 12-dim pitch-class distribution (log2, 0-log2(12))."""
    p = np.asarray(chroma_vec)
    p = p / (p.sum() + 1e-9)
    p = np.clip(p, 1e-9, 1.0)
    return float(-(p * np.log2(p)).sum())


def arousal_components(rms_dbfs_med, centroid_med, onsets_per_s, flux_mean):
    """4 component scores in [0,1] whose mean is the arousal estimate."""
    a_rms      = clip01(rms_dbfs_med,       *RMS_DBFS_RANGE)
    a_centroid = clip01(centroid_med,       *CENTROID_RANGE)
    a_onset    = clip01(onsets_per_s,       *ONSETS_PER_S_RANGE)
    a_flux     = clip01(np.log(max(flux_mean, 1.0)), *FLUX_RANGE_LOG)
    return {"rms": a_rms, "centroid": a_centroid,
            "onset": a_onset, "flux": a_flux,
            "arousal": float(np.mean([a_rms, a_centroid, a_onset, a_flux]))}


def valence_components(chroma_mean_vec, flux_mean, centroid_med):
    """Rule-based valence proxy. Mode gives sign, chroma entropy gives
    tonal clarity, flux + centroid feed small corrections. Returns a dict
    with v in [0,1] (0.5 = neutral) and a 'confidence' in [0,1]."""
    key = krumhansl_key(chroma_mean_vec)

    # Mode polarity. Research warns this is weak; weight small.
    mode_vote = 0.5 + 0.25 * np.tanh(3.0 * key["major_minor_margin"])

    # Tonal clarity (lower entropy = clearer harmonic focus = more "resolved")
    entropy = chroma_entropy(chroma_mean_vec)        # 0 .. log2(12) ≈ 3.585
    tonal_clarity = 1.0 - entropy / np.log2(12)
    tonal_vote = 0.5 + 0.15 * (tonal_clarity - 0.3)  # small push around neutral

    # Brightness contributes mildly to perceived positive valence
    bright_vote = 0.5 + 0.10 * (clip01(centroid_med, *CENTROID_RANGE) - 0.5)

    v = float(np.clip(0.4 * mode_vote + 0.35 * tonal_vote + 0.25 * bright_vote,
                      0.0, 1.0))

    # Confidence: Krumhansl-Schmuckler correlation and tonal clarity drive this.
    # Low tonal clarity (busy mix) and low key correlation = low confidence.
    conf = float(np.clip(0.5 * key["corr"] + 0.5 * tonal_clarity, 0.0, 1.0))

    return {"valence": v, "confidence": conf,
            "key_estimate": key["key"],
            "mode_margin": float(key["major_minor_margin"]),
            "tonal_clarity": float(tonal_clarity),
            "mode_vote": float(mode_vote),
            "tonal_vote": float(tonal_vote),
            "bright_vote": float(bright_vote)}


# ----- iterate over Stage-2 outputs -----
pieces_dirs = sorted([p for p in STAGE2_DIR.iterdir()
                      if p.is_dir() and (p / "features.npz").exists()])

summary_rows = []

for piece_dir in pieces_dirs:
    windowed = json.loads((piece_dir / "windowed.json").read_text())
    name = windowed["piece"]
    arr  = np.load(piece_dir / "features.npz")
    print(f"\n=== {name} ===")

    out_dir = OUT_DIR / piece_dir.name
    out_dir.mkdir(exist_ok=True)

    # per-window A and V
    traj = []
    for w in windowed["windows"]:
        a = arousal_components(w["rms_dbfs_med"], w["centroid_hz_med"],
                               w["onsets_per_s"], w["flux_mean"])
        v = valence_components(w["chroma_mean"], w["flux_mean"], w["centroid_hz_med"])
        mid = 0.5 * (w["start_s"] + w["end_s"])
        traj.append({
            "t_mid_s": round(mid, 3),
            "start_s": w["start_s"], "end_s": w["end_s"],
            "arousal": round(a["arousal"], 3),
            "arousal_components": {k: round(v2, 3) for k, v2 in a.items() if k != "arousal"},
            "valence": round(v["valence"], 3),
            "valence_confidence": round(v["confidence"], 3),
            "valence_components": {k: (round(v2, 3) if isinstance(v2, float) else v2)
                                   for k, v2 in v.items() if k != "valence"},
        })

    # whole-segment summary from stage 2 'whole_segment'
    whole = windowed["whole_segment"]
    a_whole = arousal_components(whole["rms_dbfs_med"], whole["centroid_hz_med"],
                                 whole["onsets_per_s"], whole["flux_mean"])
    chroma_mean_whole = arr["chroma"].mean(axis=1).tolist()
    v_whole = valence_components(chroma_mean_whole, whole["flux_mean"],
                                 whole["centroid_hz_med"])

    quadrant = {
        (True, True):   "exuberant (high A, high V)",
        (True, False):  "tense/agitated (high A, low V)",
        (False, True):  "peaceful/tender (low A, high V)",
        (False, False): "melancholic (low A, low V)",
    }[(a_whole["arousal"] >= 0.5, v_whole["valence"] >= 0.5)]

    summary = {
        "piece": name,
        "arousal": round(a_whole["arousal"], 3),
        "arousal_components": {k: round(x, 3) for k, x in a_whole.items() if k != "arousal"},
        "valence": round(v_whole["valence"], 3),
        "valence_confidence": round(v_whole["confidence"], 3),
        "key_estimate": v_whole["key_estimate"],
        "quadrant_hint": quadrant,
    }
    print(f"  arousal={summary['arousal']:.2f}  valence={summary['valence']:.2f}  "
          f"(conf={summary['valence_confidence']:.2f})  key≈{summary['key_estimate']}  "
          f"→ {quadrant}")
    summary_rows.append(summary)

    (out_dir / "trajectory.json").write_text(json.dumps({
        "piece": name,
        "warning": ("These are expressive-feature scores, not validated MER. "
                    "Arousal is more reliable than valence; see Stage 3 docs."),
        "whole_segment": summary,
        "trajectory": traj,
    }, indent=2))

    # ----- per-piece plot -----
    t = np.array([p["t_mid_s"] for p in traj])
    A = np.array([p["arousal"] for p in traj])
    V = np.array([p["valence"] for p in traj])
    Vc = np.array([p["valence_confidence"] for p in traj])
    comps_A = {k: np.array([p["arousal_components"][k] for p in traj])
               for k in ("rms", "centroid", "onset", "flux")}

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True)

    ax = axes[0]
    ax.plot(t, A, lw=2, color="#c0392b", label="arousal (0-1)")
    ax.axhline(0.5, color="grey", lw=0.5, ls="--")
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel("arousal")
    ax.set_title(f"{name}  —  whole-segment A={summary['arousal']:.2f}, "
                 f"V={summary['valence']:.2f} (conf={summary['valence_confidence']:.2f}), "
                 f"key≈{summary['key_estimate']}", fontsize=11, loc="left")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[1]
    # alpha of the line follows the confidence
    for i in range(len(t) - 1):
        ax.plot(t[i:i+2], V[i:i+2],
                color="#2874a6",
                alpha=0.3 + 0.7 * float(Vc[i:i+2].mean()),
                lw=2)
    ax.plot([], [], color="#2874a6", lw=2, label="valence (0-1); dim = low confidence")
    ax.axhline(0.5, color="grey", lw=0.5, ls="--")
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel("valence")
    ax.legend(loc="upper right", fontsize=9)

    ax = axes[2]
    ax.plot(t, comps_A["rms"],      lw=1.0, label="RMS (loudness)")
    ax.plot(t, comps_A["centroid"], lw=1.0, label="centroid (brightness)")
    ax.plot(t, comps_A["onset"],    lw=1.0, label="onsets/s")
    ax.plot(t, comps_A["flux"],     lw=1.0, label="spectral flux")
    ax.set_ylim(-0.05, 1.05); ax.set_ylabel("arousal components")
    ax.set_xlabel("time within segment (s)")
    ax.legend(loc="upper right", fontsize=8, ncol=2)

    fig.tight_layout()
    fig.savefig(out_dir / "diagnostic.png", dpi=140)
    plt.close(fig)
    print(f"  wrote → {out_dir}/diagnostic.png")


# ----- cross-piece summary -----
(OUT_DIR / "summary.json").write_text(json.dumps(summary_rows, indent=2))
with open(OUT_DIR / "summary.csv", "w", newline="") as f:
    flat = []
    for r in summary_rows:
        flat.append({"piece": r["piece"],
                     "arousal": r["arousal"], "valence": r["valence"],
                     "valence_confidence": r["valence_confidence"],
                     "key_estimate": r["key_estimate"],
                     "quadrant_hint": r["quadrant_hint"]})
    w = csv.DictWriter(f, fieldnames=list(flat[0].keys()))
    w.writeheader()
    for r in flat: w.writerow(r)

# ---- Circumplex scatter with trajectory curves ----
fig, ax = plt.subplots(figsize=(8, 8))
ax.axvline(0.5, color="grey", lw=0.5); ax.axhline(0.5, color="grey", lw=0.5)

colors = plt.cm.tab10(np.linspace(0, 1, len(summary_rows)))
for i, piece_dir in enumerate(pieces_dirs):
    name = summary_rows[i]["piece"]
    traj = json.loads((OUT_DIR / piece_dir.name / "trajectory.json").read_text())["trajectory"]
    V_arr = np.array([p["valence"] for p in traj])
    A_arr = np.array([p["arousal"] for p in traj])
    ax.plot(V_arr, A_arr, "-", color=colors[i], alpha=0.4, lw=1)
    ax.plot(V_arr[0],  A_arr[0],  "o", color=colors[i], ms=8)   # start
    ax.plot(V_arr[-1], A_arr[-1], "s", color=colors[i], ms=8)   # end
    ax.plot(summary_rows[i]["valence"], summary_rows[i]["arousal"], "*",
            color=colors[i], ms=18, label=name,
            markeredgecolor="black", markeredgewidth=0.5)

ax.set_xlabel("Valence  →  (negative ↔ positive)")
ax.set_ylabel("Arousal  →  (subdued ↔ energetic)")
ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
ax.text(0.75, 0.95, "exuberant",   ha="center", fontsize=10, color="#555")
ax.text(0.25, 0.95, "tense",       ha="center", fontsize=10, color="#555")
ax.text(0.75, 0.05, "peaceful",    ha="center", fontsize=10, color="#555")
ax.text(0.25, 0.05, "melancholic", ha="center", fontsize=10, color="#555")
ax.set_title("Stage 3 — V/A trajectories (★ = whole-segment, ○ = start, ■ = end)")
ax.legend(loc="lower left", fontsize=8)
ax.grid(alpha=0.2)
fig.tight_layout()
fig.savefig(OUT_DIR / "circumplex.png", dpi=140)
plt.close(fig)

# ---- Arousal bars with within-segment 10-90 pct spread ----
fig, ax = plt.subplots(figsize=(9, 4))
for i, piece_dir in enumerate(pieces_dirs):
    traj = json.loads((OUT_DIR / piece_dir.name / "trajectory.json").read_text())["trajectory"]
    A_arr = np.array([p["arousal"] for p in traj])
    V_arr = np.array([p["valence"] for p in traj])
    med = np.median(A_arr); lo = np.percentile(A_arr, 10); hi = np.percentile(A_arr, 90)
    ax.bar(i, med, color="#c0392b", alpha=0.8)
    ax.errorbar(i, med, yerr=[[med - lo], [hi - med]], color="black", capsize=4, lw=1)
ax.set_xticks(range(len(summary_rows)))
ax.set_xticklabels([r["piece"].replace(" - ", "\n") for r in summary_rows],
                   rotation=0, fontsize=8)
ax.set_ylim(0, 1); ax.set_ylabel("arousal (0-1)")
ax.set_title("Within-segment arousal: median + 10-90 pct spread")
fig.tight_layout()
fig.savefig(OUT_DIR / "arousal_bars.png", dpi=140)
plt.close(fig)

print(f"\nwrote → {OUT_DIR}/circumplex.png")
print(f"wrote → {OUT_DIR}/arousal_bars.png")
print(f"wrote → {OUT_DIR}/summary.json  and  summary.csv")
