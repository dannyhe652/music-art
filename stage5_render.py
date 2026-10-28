"""
Stage 5 — Render the painting.

Simulates Gerhard Richter's squeegee technique from his "Cage 1-6" paintings
(2006, Tate Modern). The six "Cage" canvases were dedicated to the composer
John Cage after Cage visited Richter's studio; each panel is built by
layering 5-7 colors of oil paint, then dragging a wide Perspex squeegee
across the still-wet surface. Where the blade presses hard, deep layers
reveal; where it slides, paint from one side of the canvas gets carried and
redeposited meters away.

This module reproduces that process in numpy. One painting per music piece.

Audio-to-visual mapping, in brief:
    tempo / onset density  → number of passes, band thickness
    arousal                → default drag direction (H calm, V/mixed agitated)
    flux variance          → direction-switch probability
    local RMS at a pass    → how hard the squeegee presses
    local centroid at a pass → which underlayer gets revealed
                               (bright = yellow/red, dark = black/piece-blue)
    peak-moment            → forces a dramatic red or black source layer
    dominant pitch class   → piece-specific hue in the underlayer stack
"""
import colorsys
import json
import pathlib

import numpy as np
from PIL import Image, ImageDraw, ImageFont

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------

STAGE2 = pathlib.Path("stage2_analysis")
STAGE4 = pathlib.Path("stage4_mapping")
OUT_DIR = pathlib.Path("stage5_artworks")
OUT_DIR.mkdir(exist_ok=True)

CANVAS_SIZE = 1800  # square, matching Richter's Cage panels' 290×290 cm ratio

# Scriabin-style pitch-class to hue mapping (declared artistic choice, not an
# empirical claim — used because centroid alone cannot differentiate palettes
# when the corpus has a narrow spectral range).
PC_HUE = {
    "C": 0, "C#": 30, "D": 60, "D#": 90, "E": 120, "F": 150,
    "F#": 180, "G": 210, "G#": 240, "A": 270, "A#": 300, "B": 330,
}
PC_NAMES = list(PC_HUE)

# Richter's own Cage-palette: muted olive ground with saturated reveals.
RICHTER_PALETTE = {
    "ground": "#5a5c4e",  # muted olive-grey (the dominant surface tone)
    "red":    "#c43b2a",  # cadmium red
    "yellow": "#d7b53c",  # ochre-yellow
    "blue":   "#2a4c75",  # cobalt deep
    "cream":  "#d9cdb2",  # warm cream
    "black":  "#1f1b16",  # near-black
}


# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------

def hex_to_rgb(hex_str):
    """Return an (r, g, b) tuple of ints 0-255 from "#rrggbb"."""
    h = hex_str.lstrip("#")
    return int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16)


def hex_from_hls(hue_deg, lightness, saturation):
    """Build "#rrggbb" from HLS values (hue in degrees, L and S in 0-1)."""
    r, g, b = colorsys.hls_to_rgb(
        (hue_deg / 360.0) % 1.0,
        float(np.clip(lightness, 0, 1)),
        float(np.clip(saturation, 0, 1)),
    )
    return "#{:02x}{:02x}{:02x}".format(int(r * 255), int(g * 255), int(b * 255))


# ---------------------------------------------------------------------------
# Peak-moment detection — Richter's compositions anchor on one event
# ---------------------------------------------------------------------------

def find_peak_moment(frame_times, rms_dbfs, flux, chroma):
    """Find the single most salient instant of the segment.

    Peak = max of (loudness × spectral novelty), smoothed over ~250 ms to
    avoid single-frame flicker. The chosen moment anchors the painting:
    passes near this time use a dramatic red or black source layer.

    Returns a dict with t (seconds), i (frame index), rms_dbfs, flux,
    pc (0-11), and pc_strength.
    """
    r = (rms_dbfs - rms_dbfs.min()) / (rms_dbfs.max() - rms_dbfs.min() + 1e-9)
    f = flux - flux.min()
    f = f / (f.max() + 1e-9)
    score = r * (0.4 + 0.6 * f)  # loudness dominates, flux lifts peaks

    kernel_size = max(3, int(0.25 / (frame_times[1] - frame_times[0] + 1e-9)))
    smoothed = np.convolve(score, np.ones(kernel_size) / kernel_size, mode="same")

    i_peak = int(np.argmax(smoothed))
    chr_peak = chroma[:, i_peak]
    pc_peak = int(np.argmax(chr_peak))
    return {
        "t": float(frame_times[i_peak]),
        "i": i_peak,
        "rms_dbfs": float(rms_dbfs[i_peak]),
        "flux": float(flux[i_peak]),
        "pc": pc_peak,
        "pc_strength": float(chr_peak[pc_peak] / (chr_peak.sum() + 1e-9)),
    }


# ---------------------------------------------------------------------------
# Underlayer stack — the colours the squeegee reveals
# ---------------------------------------------------------------------------

def build_base_layers(size, piece_hues):
    """Build the six colored underlayers.

    Five come from Richter's own palette; one slot is tinted toward the
    piece's dominant pitch class so each piece has a visually identifiable
    hue showing through.
    """
    hexes = [
        RICHTER_PALETTE["ground"],
        RICHTER_PALETTE["red"],
        RICHTER_PALETTE["yellow"],
        hex_from_hls(piece_hues[0], 0.42, 0.70),  # tinted piece hue
        hex_from_hls(piece_hues[1], 0.72, 0.42),  # tinted piece complement
        RICHTER_PALETTE["black"],
    ]

    rng = np.random.default_rng(13)
    layers = []
    yy, xx = np.indices((size, size), dtype=np.float32)
    low_freq = (
        np.sin(xx * 0.004) * np.cos(yy * 0.003)
        + 0.5 * np.sin(xx * 0.009 + 1.3) * np.cos(yy * 0.007 + 0.7)
    ) * 7.0

    for hx in hexes:
        base_rgb = np.array(hex_to_rgb(hx), dtype=np.float32)
        noise = rng.normal(0, 1, (size, size)).astype(np.float32) * 2.0
        plate = base_rgb[None, None, :] + (low_freq + noise)[:, :, None]
        layers.append(np.clip(plate, 0, 255))
    return layers


# ---------------------------------------------------------------------------
# Squeegee passes — horizontal and vertical
# ---------------------------------------------------------------------------
#
# A pass does four things, matching the physical squeegee:
#   1. Deposit the source paint, modulated by random "pressure" along the
#      blade and by a tapered longitudinal profile.
#   2. Smear horizontally (or vertically) by iterating a weighted shift.
#   3. Pick up colour from the leading edge, redeposit it on the trailing
#      edge — Richter's "paint teleport."
#   4. Drop gravity-driven drip streaks from the band edge.
#
# Variety parameters (added so the result doesn't read as a clean stripe
# grid):
#   taper_frac    — gradient alpha at both endpoints
#   y_drift       — slight angle via per-column y-centre offset
#   patchy_amount — low-freq intensity variation along the pass length

def pass_horizontal(canvas, source, y0, y1, x0, x1, *,
                    reveal_strength=0.75, smear_len=25, drip_amount=0.3,
                    taper_frac=0.14, y_drift=0, patchy_amount=0.35, rng=None):
    """Pull a horizontal squeegee pass across the band [y0:y1, x0:x1]."""
    if rng is None:
        rng = np.random.default_rng()
    band_h = y1 - y0
    pass_width = x1 - x0
    if band_h < 2 or pass_width < 10:
        return canvas

    # Longitudinal intensity envelope: low-frequency noise × endpoint taper.
    # This is what breaks the uniform-stripe reading.
    patch = _smoothed_noise(pass_width, rng, window=max(1, pass_width // 25))
    patch = (1 - patchy_amount) + patchy_amount * patch
    taper = _taper_profile(pass_width, taper_frac)
    long_profile = patch * taper  # shape (pass_width,)

    # Deposit the source colour in short column slices so y-drift bends the
    # pass gradually along a slight diagonal.
    for col_start in range(0, pass_width, 10):
        col_end = min(col_start + 10, pass_width)
        frac = (col_start + col_end) / 2 / pass_width
        y_offset = int(round((frac - 0.5) * 2 * y_drift))
        yy0 = max(0, y0 + y_offset)
        yy1 = min(canvas.shape[0], y1 + y_offset)
        if yy1 - yy0 < 2:
            continue
        # Pressure across the blade (perpendicular to drag direction).
        pressure = _smoothed_noise(yy1 - yy0, rng, window=9)
        strip = canvas[yy0:yy1, x0 + col_start:x0 + col_end].copy()
        src = source[yy0:yy1, x0 + col_start:x0 + col_end]
        blend = (reveal_strength
                 * pressure[:, None]
                 * long_profile[col_start:col_end][None, :])[:, :, None]
        canvas[yy0:yy1, x0 + col_start:x0 + col_end] = (
            strip * (1 - blend) + src * blend
        )

    # Smear the whole deposited strip along the drag direction.
    strip = canvas[y0:y1, x0:x1].copy()
    smeared = strip.copy()
    for shift in range(2, smear_len + 1, 3):
        shifted = np.roll(strip, shift, axis=1)
        shifted[:, :shift] = strip[:, :shift]
        smeared = smeared * 0.88 + shifted * 0.12

    # Pick up paint from the leading edge and redeposit on the trailing edge.
    if pass_width > 60:
        pickup = strip[:, :20].mean(axis=1, keepdims=True)
        tail = min(50, pass_width - 20)
        smeared[:, -tail:] = smeared[:, -tail:] * 0.72 + pickup * 0.28

    canvas[y0:y1, x0:x1] = smeared
    _add_drips(canvas, strip, x0, x1, y1, band_h, drip_amount, rng,
               orientation="h")
    return canvas


def pass_vertical(canvas, source, x0, x1, y0, y1, *,
                  reveal_strength=0.75, smear_len=25, drip_amount=0.0,
                  taper_frac=0.14, x_drift=0, patchy_amount=0.35, rng=None):
    """Pull a vertical squeegee pass across the band [y0:y1, x0:x1]."""
    if rng is None:
        rng = np.random.default_rng()
    pass_h = y1 - y0
    pass_w = x1 - x0
    if pass_h < 10 or pass_w < 2:
        return canvas

    patch = _smoothed_noise(pass_h, rng, window=max(1, pass_h // 25))
    patch = (1 - patchy_amount) + patchy_amount * patch
    taper = _taper_profile(pass_h, taper_frac)
    long_profile = patch * taper  # shape (pass_h,)

    for row_start in range(0, pass_h, 10):
        row_end = min(row_start + 10, pass_h)
        frac = (row_start + row_end) / 2 / pass_h
        x_offset = int(round((frac - 0.5) * 2 * x_drift))
        xx0 = max(0, x0 + x_offset)
        xx1 = min(canvas.shape[1], x1 + x_offset)
        if xx1 - xx0 < 2:
            continue
        pressure = _smoothed_noise(xx1 - xx0, rng, window=9)
        strip = canvas[y0 + row_start:y0 + row_end, xx0:xx1].copy()
        src = source[y0 + row_start:y0 + row_end, xx0:xx1]
        blend = (reveal_strength
                 * long_profile[row_start:row_end][:, None]
                 * pressure[None, :])[:, :, None]
        canvas[y0 + row_start:y0 + row_end, xx0:xx1] = (
            strip * (1 - blend) + src * blend
        )

    # Smear vertically — transpose, apply horizontal smear, transpose back.
    strip = canvas[y0:y1, x0:x1].copy()
    strip_t = np.transpose(strip, (1, 0, 2))
    smeared = strip_t.copy()
    for shift in range(2, smear_len + 1, 3):
        shifted = np.roll(strip_t, shift, axis=1)
        shifted[:, :shift] = strip_t[:, :shift]
        smeared = smeared * 0.88 + shifted * 0.12
    if strip_t.shape[1] > 60:
        pickup = strip_t[:, :20].mean(axis=1, keepdims=True)
        tail = min(50, strip_t.shape[1] - 20)
        smeared[:, -tail:] = smeared[:, -tail:] * 0.72 + pickup * 0.28
    canvas[y0:y1, x0:x1] = np.transpose(smeared, (1, 0, 2))

    _add_drips(canvas, canvas[y0:y1, x0:x1], x0, x1, y1, pass_h, drip_amount,
               rng, orientation="v")
    return canvas


def _smoothed_noise(length, rng, window):
    """Low-frequency 0-1 noise profile of given length, smoothed by boxcar."""
    raw = rng.random(length).astype(np.float32)
    smoothed = np.convolve(raw, np.ones(window) / window, mode="same")
    return (smoothed - smoothed.min()) / (smoothed.max() - smoothed.min() + 1e-9)


def _taper_profile(length, fraction):
    """Gradient alpha profile: fade in at the start, fade out at the end."""
    taper = np.ones(length, dtype=np.float32)
    w = int(length * fraction)
    if w > 1:
        taper[:w] = np.linspace(0.0, 1.0, w)
        taper[-w:] = np.linspace(1.0, 0.0, w)
    return taper


def _add_drips(canvas, strip, x0, x1, y1, band_h, drip_amount, rng, orientation):
    """Drop a few gravity-driven drip streaks below the band."""
    n_drips = int(drip_amount * 6)
    for _ in range(n_drips):
        drip_x = x0 + int(rng.integers(0, max(1, x1 - x0)))
        drip_color = strip[int(rng.integers(0, max(1, band_h if orientation == "h"
                                                   else strip.shape[0]))),
                           -1 if orientation == "h" else int(rng.integers(0, max(1, strip.shape[1])))]
        drip_h = int(60 + rng.random() * (180 if orientation == "h" else 140))
        thickness = 1 + int(rng.random() * 2)
        for dy in range(y1, min(canvas.shape[0], y1 + drip_h)):
            alpha = (0.22 if orientation == "h" else 0.20) * (1 - (dy - y1) / drip_h)
            if rng.random() < 0.6:
                for t in range(thickness):
                    xx = drip_x + t
                    if 0 <= xx < canvas.shape[1]:
                        canvas[dy, xx] = canvas[dy, xx] * (1 - alpha) + drip_color * alpha


# ---------------------------------------------------------------------------
# Finishing details — scraped-through windows and micro-streaks
# ---------------------------------------------------------------------------

def add_reveal_spots(canvas, layers, flux_mean, rng):
    """Soft elliptical "windows" where a deeper underlayer shows through.

    These are the little paint-lifted wells visible in real Cages. Count
    scales with mean flux: more agitated pieces get more scrape-throughs.
    """
    n = int(np.clip(6 + flux_mean * 0.008, 6, 20))
    size = canvas.shape[0]
    for _ in range(n):
        cx, cy = int(rng.integers(100, size - 100)), int(rng.integers(100, size - 100))
        w, h = int(rng.integers(40, 180)), int(rng.integers(20, 80))
        src = layers[int(rng.integers(1, len(layers)))][cy:cy + h, cx:cx + w]
        dst = canvas[cy:cy + h, cx:cx + w]
        if src.shape != dst.shape:
            continue
        yy, xx = np.indices(src.shape[:2], dtype=np.float32)
        yc, xc = src.shape[0] / 2, src.shape[1] / 2
        dist = np.sqrt(((xx - xc) / xc) ** 2 + ((yy - yc) / yc) ** 2)
        mask = (np.clip(1 - dist, 0, 1) ** 1.5)[:, :, None] * 0.78
        canvas[cy:cy + h, cx:cx + w] = dst * (1 - mask) + src * mask


def add_micro_streaks(canvas, layers, flux_mean, default_dir, rng):
    """Short perpendicular hatches giving fine within-pass paint grain."""
    n = int(np.clip(30 + flux_mean * 0.03, 30, 90))
    size = canvas.shape[0]
    for _ in range(n):
        sx = int(rng.integers(60, size - 60))
        sy = int(rng.integers(60, size - 60))
        length = int(rng.integers(30, 140))
        color = layers[int(rng.integers(1, len(layers)))][sy, sx]
        strength = 0.4 + 0.3 * rng.random()
        if default_dir == "H":
            for ty in range(sy, min(size - 10, sy + length)):
                if rng.random() < 0.85:
                    canvas[ty, sx] = canvas[ty, sx] * (1 - strength) + color * strength
                    if sx + 1 < size:
                        canvas[ty, sx + 1] = (canvas[ty, sx + 1] * (1 - strength * 0.5)
                                              + color * strength * 0.5)
        else:
            for tx in range(sx, min(size - 10, sx + length)):
                if rng.random() < 0.85:
                    canvas[sy, tx] = canvas[sy, tx] * (1 - strength) + color * strength
                    if sy + 1 < size:
                        canvas[sy + 1, tx] = (canvas[sy + 1, tx] * (1 - strength * 0.5)
                                              + color * strength * 0.5)


def apply_fill_noise(image, strength=7):
    """Multiplicative multi-octave noise so solid regions aren't digitally flat."""
    rgb = np.array(image.convert("RGB")).astype(np.float32)
    H, W = rgb.shape[:2]
    y, x = np.indices((H, W), dtype=np.float32)
    rng = np.random.default_rng(11)
    noise = (
        np.sin(x * 0.018 + y * 0.013) * np.cos(x * 0.009 - y * 0.015)
        + np.sin(x * 0.055 + rng.normal(0, 0.3, (H, W))) * 0.5
        + rng.normal(0, 1, (H, W)) * 0.4
    ) / 2.0
    rgb += noise[:, :, None] * strength
    return Image.fromarray(np.clip(rgb, 0, 255).astype(np.uint8), "RGB")


# ---------------------------------------------------------------------------
# Main render
# ---------------------------------------------------------------------------

def render_piece(piece_dir_name):
    """Render one painting from the piece's Stage 2 + Stage 4 output."""
    params = json.loads((STAGE4 / piece_dir_name / "visual_parameters.json").read_text())
    feats = np.load(STAGE2 / piece_dir_name / "features.npz")

    name = params["piece"]
    arousal = params["source"]["stage3_summary"]["arousal"]
    valence = params["source"]["stage3_summary"]["valence"]
    print(f"\n→ {name}")

    # Pull the feature arrays we need.
    onsets = np.asarray(feats["onset_times"], dtype=np.float64)
    frame_times = np.asarray(feats["frame_times"], dtype=np.float64)
    rms = np.asarray(feats["rms_dbfs"], dtype=np.float64)
    centroid = np.asarray(feats["centroid"], dtype=np.float64)
    flux = np.asarray(feats["flux"], dtype=np.float64)
    chroma = feats["chroma"]
    duration = float(frame_times[-1])

    peak = find_peak_moment(frame_times, rms, flux, chroma)
    top_pcs = np.argsort(chroma.mean(axis=1))[::-1][:3]
    piece_hues = [PC_HUE[PC_NAMES[int(pc)]] for pc in top_pcs]

    layers = build_base_layers(CANVAS_SIZE, piece_hues)
    canvas = layers[0].copy()  # start on the olive ground

    # Normalisation ranges we'll reuse per pass.
    rms_min, rms_max = float(rms.min()), float(rms.max())
    flux_min, flux_max = float(flux.min()), float(flux.max())
    cen_min, cen_max = float(centroid.min()), float(centroid.max())

    rng = np.random.default_rng(seed=int(peak["t"] * 1000) % 2**31)

    # ------------------------------------------------------------------
    # Decide the piece's visual character from its features
    # ------------------------------------------------------------------
    dynamic_range_db = params["source"]["stage2_summary"]["rms_dbfs_p90p10"]
    onsets_per_sec = len(onsets) / duration
    dense_score = min(1.0, onsets_per_sec / 7.0)
    # Flux variance drives direction switching — chaotic pieces mix H and V.
    flux_std_norm = flux.std() / (flux.std() + 100 + 1e-9)
    mix_direction_prob = float(np.clip(0.15 + 0.60 * flux_std_norm, 0.15, 0.70))
    # Calm pieces get horizontal strata; agitated ones get vertical slabs.
    default_dir = "H" if arousal < 0.6 else ("V" if peak["pc"] % 2 == 0 else "H")
    n_passes = int(np.clip(14 + len(onsets) * 0.18 * (1 + dense_score), 18, 70))

    print(f"  default_dir={default_dir}  n_passes={n_passes}  "
          f"dense={dense_score:.2f}  mix_dir_p={mix_direction_prob:.2f}")

    # ------------------------------------------------------------------
    # Stage A — cover the ground with 7 full-width base passes
    # ------------------------------------------------------------------
    # Finished Cages show very little bare ground; the base layer is almost
    # entirely over-painted. We lay this down before any music-driven marks.
    _lay_base_coverage(canvas, layers, default_dir, rng)

    # ------------------------------------------------------------------
    # Stage B — music-driven character passes
    # ------------------------------------------------------------------
    pass_times = _pick_pass_times(onsets, duration, n_passes, rng)
    for t_pass in pass_times:
        _one_music_pass(canvas, layers, t_pass, peak, valence,
                        default_dir, mix_direction_prob, dense_score,
                        frame_times, rms, centroid, flux,
                        rms_min, rms_max, flux_min, flux_max, cen_min, cen_max,
                        rng)

    # ------------------------------------------------------------------
    # Stage C — finishing details
    # ------------------------------------------------------------------
    add_reveal_spots(canvas, layers, flux.mean(), rng)
    add_micro_streaks(canvas, layers, flux.mean(), default_dir, rng)

    # Light paint texture and a gentle contrast lift.
    img = Image.fromarray(canvas.astype(np.uint8), "RGB")
    img = apply_fill_noise(img, strength=7)
    final_rgb = np.array(img).astype(np.float32)
    final_rgb = np.clip((final_rgb - 128) * 1.08 + 128, 0, 255).astype(np.uint8)
    final = Image.fromarray(final_rgb, "RGB")

    out_path = OUT_DIR / piece_dir_name / "artwork.png"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    final.save(out_path)
    print(f"  wrote → {out_path}")
    return out_path


def _lay_base_coverage(canvas, layers, default_dir, rng):
    """Seven full-canvas passes cycling through the colour layers."""
    n = 7
    for bp in range(n):
        src = layers[1 + (bp % 5)]
        band_h = int(CANVAS_SIZE / n * 1.4)  # bands overlap
        center = int((bp + 0.5) * CANVAS_SIZE / n)
        c0 = max(10, center - band_h // 2)
        c1 = min(CANVAS_SIZE - 10, center + band_h // 2)
        drift = int(rng.integers(-18, 19))
        if default_dir == "H":
            pass_horizontal(canvas, src, c0, c1, 20, CANVAS_SIZE - 20,
                            reveal_strength=0.78, smear_len=55, drip_amount=0.18,
                            taper_frac=0.06, y_drift=drift, patchy_amount=0.32,
                            rng=rng)
        else:
            pass_vertical(canvas, src, c0, c1, 20, CANVAS_SIZE - 20,
                          reveal_strength=0.78, smear_len=55, drip_amount=0.14,
                          taper_frac=0.06, x_drift=drift, patchy_amount=0.32,
                          rng=rng)


def _pick_pass_times(onsets, duration, target_count, rng):
    """Choose pass times from onsets + a few evenly-spaced extras.

    Pure onset lists concentrate at the dense start of some pieces, so we
    pad with evenly-spaced extras to make sure late portions of the piece
    get painted too.
    """
    times = sorted(list(onsets) + list(np.linspace(0.5, duration - 0.5, 10)))
    while len(times) < target_count:
        times.append(float(rng.random() * duration))
    return sorted(times)[:target_count]


def _one_music_pass(canvas, layers, t_pass, peak, valence,
                    default_dir, mix_direction_prob, dense_score,
                    frame_times, rms, centroid, flux,
                    rms_min, rms_max, flux_min, flux_max, cen_min, cen_max,
                    rng):
    """Perform one music-driven pass at the given time."""
    i = int(np.argmin(np.abs(frame_times - t_pass)))
    rms_n = (rms[i] - rms_min) / (rms_max - rms_min + 1e-9)
    cen_n = (centroid[i] - cen_min) / (cen_max - cen_min + 1e-9)
    flux_n = (flux[i] - flux_min) / (flux_max - flux_min + 1e-9)  # noqa: F841

    direction = default_dir
    if rng.random() < mix_direction_prob:
        direction = "V" if default_dir == "H" else "H"

    # Pick which underlayer this pass reveals. Bright centroids pull up warm
    # layers (yellow / red); dark centroids pull up blue / black. Near the
    # peak moment we force the dramatic red (positive valence) or black
    # (negative valence) regardless of local centroid.
    if cen_n > 0.7:
        src_idx = 2  # yellow
    elif cen_n > 0.5:
        src_idx = 1  # red
    elif cen_n > 0.3:
        src_idx = 3  # piece hue 1
    elif cen_n > 0.15:
        src_idx = 4  # piece hue 2
    else:
        src_idx = 5  # black
    if abs(t_pass - peak["t"]) < 1.5:
        src_idx = 1 if valence >= 0.5 else 5
    source = layers[src_idx]

    # Band thickness scales with density and loudness.
    thickness = int(np.clip(140 + 160 * (1 - dense_score), 140, 320))
    thickness = int(thickness * (0.75 + 0.55 * rms_n))

    # Direction-specific pass setup. Dense pieces get partial-width "cliff
    # edges"; sparse pieces get full-width sustained plates.
    reveal = 0.65 + 0.30 * rms_n
    smear = int(np.clip(35 + 50 * (1 - dense_score), 30, 100))
    drift = int(rng.integers(-25, 26))
    taper = float(0.08 + 0.14 * rng.random())
    patchy = float(0.25 + 0.35 * rng.random())

    if direction == "H":
        y_c = int(rng.integers(thickness // 2 + 10, CANVAS_SIZE - thickness // 2 - 10))
        y0, y1 = y_c - thickness // 2, y_c + thickness // 2
        if dense_score > 0.5 and rng.random() < 0.7:
            w = int(CANVAS_SIZE * (0.30 + 0.45 * rng.random()))
            x_start = int(rng.integers(10, CANVAS_SIZE - w - 10))
            x0, x1 = x_start, x_start + w
        else:
            x0, x1 = 10, CANVAS_SIZE - 10
        pass_horizontal(canvas, source, y0, y1, x0, x1,
                        reveal_strength=reveal, smear_len=smear,
                        drip_amount=0.20 + 0.4 * rms_n, taper_frac=taper,
                        y_drift=drift, patchy_amount=patchy, rng=rng)
    else:
        x_c = int(rng.integers(thickness // 2 + 10, CANVAS_SIZE - thickness // 2 - 10))
        x0, x1 = x_c - thickness // 2, x_c + thickness // 2
        if dense_score > 0.5 and rng.random() < 0.7:
            h = int(CANVAS_SIZE * (0.30 + 0.45 * rng.random()))
            y_start = int(rng.integers(10, CANVAS_SIZE - h - 10))
            y0, y1 = y_start, y_start + h
        else:
            y0, y1 = 10, CANVAS_SIZE - 10
        pass_vertical(canvas, source, x0, x1, y0, y1,
                      reveal_strength=reveal, smear_len=smear,
                      drip_amount=0.15 + 0.3 * rms_n, taper_frac=taper,
                      x_drift=drift, patchy_amount=patchy, rng=rng)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def build_gallery(paths, font_path=None):
    """Lay the paintings out in a 3-column grid on a white ground with
    Josefin Sans labels centred beneath each panel.
    """
    thumb_w = thumb_h = 540
    gap_x, gap_y, label_h = 36, 60, 60
    n = len(paths)
    rows = int(np.ceil(n / 3))
    cols = min(n, 3)
    width = cols * thumb_w + (cols + 1) * gap_x
    height = rows * (thumb_h + label_h) + (rows + 1) * gap_y
    gallery = Image.new("RGB", (width, height), (255, 255, 255))
    draw = ImageDraw.Draw(gallery)

    try:
        font = ImageFont.truetype(font_path or ".fonts/JosefinSans.ttf", size=26)
        try:
            font.set_variation_by_axes([500])  # medium weight
        except Exception:
            pass
    except Exception:
        font = ImageFont.load_default()

    for i, path in enumerate(paths):
        im = Image.open(path).resize((thumb_w, thumb_h), Image.LANCZOS)
        col, row = i % cols, i // cols
        x = gap_x + col * (thumb_w + gap_x)
        y = gap_y + row * (thumb_h + label_h + gap_y)
        gallery.paste(im, (x, y))
        label = pathlib.Path(path).parent.name.replace("_", " ")
        bbox = draw.textbbox((0, 0), label, font=font)
        tw = bbox[2] - bbox[0]
        draw.text((x + (thumb_w - tw) // 2, y + thumb_h + 16),
                  label, fill=(40, 36, 32), font=font)
    out_path = OUT_DIR / "gallery.png"
    gallery.save(out_path)
    return out_path


def main():
    pieces = sorted(p.name for p in STAGE4.iterdir()
                    if p.is_dir() and (p / "visual_parameters.json").exists())
    paths = [render_piece(p) for p in pieces]
    gallery_path = build_gallery(paths)
    print(f"\nGallery → {gallery_path}")


if __name__ == "__main__":
    main()
