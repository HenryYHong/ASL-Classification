#!/usr/bin/env bash
#
# Turn a screen recording of the live demo into the GIF README.md embeds.
#
#   ./docs/make_demo.sh ~/Desktop/recording.mov     convert a recording you already made
#   ./docs/make_demo.sh record 25                   record the screen for 25 s, then convert
#
# Two environment variables, because a raw screen recording is almost never the shot:
#
#   CROP=w:h:x:y   crop before scaling. A recording of the whole screen leaves the page column
#                  filling well under half the width, and everything that matters -- the letters
#                  row especially -- is then scaled into illegibility. Crop first, scale second.
#   TRIM=from:to   seconds. Cuts the reach for the mouse at the end, and anything before the
#                  first letter at the start.
#
# Three more, for cutting a second, smaller version of the same recording -- the one on my
# personal site is shorter and narrower than the one in this README:
#
#   OUT_GIF=path  where to write the GIF. Anything other than the default also skips the README
#                 embed, since that line should point at the committed demo and nothing else.
#   OUT_MP4=path  same, for the mp4.
#   LADDER=...    "width fps colors" triples separated by |, tried in order until one fits
#                 under 10 MB.
#
# The values that produced the committed demo.gif are in the README table below this block.
# GitHub renders an animated GIF inline from a path in the repo; it does NOT render an <video>
# tag or a committed .mp4, which is why the GIF is the artifact that matters here. The mp4 is
# written too, because it is a tenth of the size and is the better thing to link elsewhere.
#
# What to sign: the alphabet in order is the honest demo, because it shows the letters this
# thing is weakest at (M, N, S) alongside the ones it nails. Include J and Z -- they are the
# reason temporal/ exists, and a demo without them is a demo of the old 24-class project.
set -euo pipefail

cd "$(dirname "$0")/.."
DEFAULT_GIF="docs/demo.gif"
OUT_GIF="${OUT_GIF:-$DEFAULT_GIF}"
OUT_MP4="${OUT_MP4:-docs/demo.mp4}"
LADDER="${LADDER:-800 12 96|720 10 96|720 10 64|640 10 64}"
MAX_BYTES=$((10 * 1024 * 1024))     # GitHub renders larger files, but slowly and badly on mobile

# The shot used for the committed demo, from a 2968x2026 "Record Selected Portion" capture:
#   CROP=1370:1090:790:60 TRIM=0:24.1 ./docs/make_demo.sh ~/Desktop/'Screen Recording ....mov'
CROP="${CROP:-}"
TRIM="${TRIM:-}"
PRE="${CROP:+crop=$CROP,}"
CUT=()
if [ -n "$TRIM" ]; then
  FROM="${TRIM%%:*}"; TO="${TRIM##*:}"
  CUT=(-ss "$FROM" -t "$(awk -v a="$TO" -v b="$FROM" 'BEGIN{printf "%.3f", a-b}')")
fi

if ! command -v ffmpeg >/dev/null; then
  echo "ffmpeg not found.  brew install ffmpeg" >&2
  exit 1
fi

if [ "${1:-}" = "record" ]; then
  SECS="${2:-25}"
  SRC="$(mktemp -t asldemo).mov"
  echo "Recording the full screen for ${SECS}s. Put the demo page in front now."
  for i in 3 2 1; do printf '%s... ' "$i"; sleep 1; done; echo "go"
  # -v video, -V duration. macOS will ask for Screen Recording permission the first time; if the
  # file comes out empty, that prompt is why. System Settings > Privacy & Security.
  screencapture -v -V "$SECS" "$SRC"
elif [ -f "${1:-}" ]; then
  SRC="$1"
else
  echo "usage: $0 <recording.mov> | record [seconds]" >&2
  exit 1
fi

echo "source: $SRC ($(du -h "$SRC" | cut -f1))"
ffmpeg -nostdin -loglevel error -y "${CUT[@]}" -i "$SRC" \
  -vf "${PRE}scale=1280:-2:flags=lanczos" -c:v libx264 -crf 22 -preset slow -an \
  -movflags +faststart -pix_fmt yuv420p "$OUT_MP4"
# The mp4 is gitignored: it is for dragging into the GitHub editor, which rehosts it as an
# inline player at better quality than any GIF. The committed GIF is what a clone gets.
echo "wrote $OUT_MP4 ($(du -h "$OUT_MP4" | cut -f1)) -- not committed; drag it into a GitHub editor for an inline player"

# Two passes: one to build a palette from the clip's own colors, one to apply it. A single-pass
# GIF uses the default 216-color web palette and turns the camera feed into mud.
#
# Three settings do the real work on a clip of a camera feed, and they are not obvious:
#
#   hqdn3d      A webcam's sensor noise changes every pixel every frame, so GIF's inter-frame
#               compression finds nothing to skip. Denoising first took this clip from 10.4 MB
#               to 7.1 MB at identical resolution -- a third of the file was grain.
#   max_colors  96 rather than the full 256. Below about 64 the wall behind the signer posterizes
#               into visible bands; above ~96 the extra entries are spent on noise.
#   dither=none Dithering scatters per-pixel error, which is exactly the high-frequency detail
#               that does not compress. On flat UI plus one camera rectangle it costs little and
#               saves a third again.
PAL="$(mktemp -t aslpal).png"
IFS='|' read -ra RUNGS <<< "$LADDER"
for spec in "${RUNGS[@]}"; do
  set -- $spec; W="$1"; F="$2"; C="$3"
  BASE="${PRE}hqdn3d=4:4:6:6,fps=$F,scale=$W:-2:flags=lanczos"
  ffmpeg -nostdin -loglevel error -y "${CUT[@]}" -i "$SRC" \
    -vf "$BASE,palettegen=max_colors=$C:stats_mode=diff" "$PAL"
  ffmpeg -nostdin -loglevel error -y "${CUT[@]}" -i "$SRC" -i "$PAL" \
    -lavfi "$BASE[v];[v][1:v]paletteuse=dither=none:diff_mode=rectangle" "$OUT_GIF"
  BYTES=$(wc -c < "$OUT_GIF")
  echo "  ${W}px @ ${F}fps, ${C} colors -> $((BYTES / 1024)) KB"
  [ "$BYTES" -le "$MAX_BYTES" ] && break
done
rm -f "$PAL"

if [ "$(wc -c < "$OUT_GIF")" -gt "$MAX_BYTES" ]; then
  echo "Still over 10 MB at the smallest setting -- record a shorter clip." >&2
fi

# Deliberately NOT speeding the clip up to save frames. A recognizer's demo is partly a claim
# about its latency, and 1.25x would quietly overstate it.

# Wire it into the README the first time only, right under the title, and only for the demo the
# README is actually about.
if [ "$OUT_GIF" = "$DEFAULT_GIF" ] && ! grep -q 'docs/demo.gif' README.md; then
  /usr/bin/python3 - <<'PY'
lines = open('README.md').read().split('\n')
i = next(k for k, l in enumerate(lines) if l.startswith('# '))
lines.insert(i + 1, '\n![The live demo reading fingerspelled letters](docs/demo.gif)\n')
open('README.md', 'w').write('\n'.join(lines))
print('README.md: embedded docs/demo.gif')
PY
fi
echo "done: $OUT_GIF ($(du -h "$OUT_GIF" | cut -f1))"
