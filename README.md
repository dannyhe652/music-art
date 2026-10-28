# Music → Painting

A procedural visual translator for music. Takes a short audio excerpt,
extracts rhythmic and tonal features with *librosa*, scores it on an
arousal–valence plane, and renders a painting by simulating Gerhard
Richter's squeegee technique from his **Cage 1–6** paintings (2006, Tate
Modern), six oils dedicated to the composer John Cage.

![Six panels: Richter-after-Cage abstractions of six art-rock
pieces.](gallery.png)

The six panels above are, in reading order: *GYBE — Sleep*, *King Crimson —
Discipline*, *King Crimson — Starless*, *Magma — De Futura*, *Pat Metheny
Group — San Lorenzo*, *Yes — Close to the Edge*. Each panel is a
25-second-or-so excerpt projected into the shared visual grammar.

## Why Richter

The *Cage 1–6* canvases work by layering five to seven colours of oil
paint, then dragging a wide Perspex squeegee across the still-wet surface.
Where the blade presses hard, deep layers reveal; where it slides, paint
from one side of the canvas gets carried and redeposited meters away.
Richter chose Cage for the dedication because Cage's compositional
philosophy — strict system with chance operations — matched the technique:
the direction of the drag is deliberate, but the variability of pressure,
pickup, and redeposit is uncertain. The six finished panels have very
different compositional feel because Richter pulled the squeegee
differently each day: *Cage 1* reads as calm horizontal strata, *Cage 3*
as agitated short mixed-direction passes.

That duality — system plus chance — is why the technique maps cleanly onto
music: the piece supplies the system, the simulated squeegee supplies the
variability, and the finished painting carries the piece's character.

## What maps to what

The dominant pitch class of the piece sets the warm or cool color family
in the underlayer stack. The spectral centroid at any given moment decides
which underlayer the squeegee reveals — bright moments pull up yellow or
red, dark moments pull up near-black or deep blue. Loudness at each moment
controls how hard the squeegee presses. The density of onsets per second
decides how many passes get painted — dense percussive pieces end up with
dozens of thin bands, sparse lyrical pieces with a few wide plates.

Arousal (how energetic the piece feels overall) chooses the default drag
direction: calm pieces get horizontal strata reminiscent of *Cage 1*,
agitated pieces get vertical slabs or mixed directions like *Cage 3*. The
variability of spectral flux decides how often the direction switches, so
chaotic pieces visibly chop between horizontal and vertical. The piece's
single peak moment — the loudest, most spectrally novel instant — forces a
dramatic source color at the matching passes: red if the piece is
bright-valenced, black if dark.

Dynamic range becomes negative space: a piece with wide quiet-loud contrast
drops its quietest third of onsets so the olive-ground base reads through
as rest, while a piece with narrow dynamics covers almost every pixel.
Pass position is deliberately randomized across the canvas rather than
following time left-to-right.

## Pipeline

```
Stage 1   Audio quality check and 25-second segment selection
Stage 2   librosa feature extraction (tempo, onsets, RMS, centroid, chroma, flux)
Stage 3   Valence–arousal scores from the Stage 2 features
Stage 4   Visual parameters: palette, dominant form, layout
Stage 5   Richter squeegee render → one painting per piece
```

Each stage reads from the output of the earlier stage and writes to its own
directory. The intermediate directories are gitignored and will be
regenerated from audio.

## Reproducing

```bash
# Set up a Python environment
uv venv .venv && source .venv/bin/activate
uv pip install -r requirements.txt

# Drop lossless audio into stage1_qc/segments/<Piece_Name>_segment.wav
# then run each stage in order:
python stage2_analyse.py
python stage3_emotion.py
python stage4_mapping.py
python stage5_render.py
```

The six paintings in `artworks/` and the composite in `gallery.png` are the
exact outputs produced by the five stages for the six excerpts described
above.

## Files

```
stage1_audio_qc.py         Audio quality check (clipping, silence, dynamics)
stage1_extract_segments.py Select the 25-second segment with most timbral variety
stage2_analyse.py          librosa feature extraction + diagnostic plots
stage3_emotion.py          Transparent rule-based arousal–valence estimator
stage4_mapping.py          Features → visual parameters JSON
stage5_render.py           Richter squeegee simulator (the painter)
gallery.png                The six finished panels composed on a white ground
artworks/                  The six individual paintings
.fonts/                    Josefin Sans Medium for the gallery labels
```

## Caveats

Everything in `stage3_emotion.py` is a transparent rule-based estimator,
not a validated music-emotion-recognition system. Arousal estimation is
reasonably supported by acoustic cues (RMS, centroid, onset rate, flux);
valence estimation is weaker — mode detection from chroma is noisy on
mixed recordings, and the current best pretrained MER models still show a
clear arousal-stronger-than-valence gap. The renderer deliberately uses
valence sparingly as a result.

The pitch-class to hue mapping is Scriabin-style and declared as an
artistic choice, not an empirical claim. Using the dominant chroma to
tint one slot in the underlayer stack produces visually identifiable
palettes across pieces, which pure spectral centroid cannot do when the
corpus centroids cluster closely.
