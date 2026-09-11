#!/usr/bin/env bash
#
# Turn a screen recording of the live demo into the GIF README.md embeds.
#
#   ./docs/make_demo.sh ~/Desktop/recording.mov     convert a recording you already made
#   ./docs/make_demo.sh record 25                   record the screen for 25 s, then convert
#
# GitHub renders an animated GIF inline from a path in the repo; it does NOT render an <video>
# tag or a committed .mp4, which is why the GIF is the artifact that matters here. The mp4 is
# written too, because it is a tenth of the size and is the better thing to link elsewhere.
#
# What to sign: the alphabet in order is the honest demo, because it shows the letters this
# thing is weakest at (M, N, S) alongside the ones it nails. Include J and Z -- they are the
# reason temporal/ exists, and a demo without them is a demo of the old 24-class project.
set -euo pipefail

cd "$(dirname "$0")/.."
OUT_GIF="docs/demo.gif"
OUT_MP4="docs/demo.mp4"
MAX_BYTES=$((10 * 1024 * 1024))     # GitHub renders larger files, but slowly and badly on mobile

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
ffmpeg -nostdin -loglevel error -y -i "$SRC" \
  -vf "scale=960:-2:flags=lanczos" -c:v libx264 -crf 24 -preset slow -an -movflags +faststart \
  "$OUT_MP4"
echo "wrote $OUT_MP4 ($(du -h "$OUT_MP4" | cut -f1))"

# Two passes: one to build a palette from the clip's own colours, one to apply it. A single-pass
# GIF uses the default 216-colour web palette and turns the camera feed into mud.
PAL="$(mktemp -t aslpal).png"
for spec in "900 15" "820 13" "720 12" "640 10"; do
  set -- $spec; W="$1"; F="$2"
  ffmpeg -nostdin -loglevel error -y -i "$SRC" \
    -vf "fps=$F,scale=$W:-2:flags=lanczos,palettegen=stats_mode=diff" "$PAL"
  ffmpeg -nostdin -loglevel error -y -i "$SRC" -i "$PAL" \
    -lavfi "fps=$F,scale=$W:-2:flags=lanczos[v];[v][1:v]paletteuse=dither=bayer:bayer_scale=3:diff_mode=rectangle" \
    "$OUT_GIF"
  BYTES=$(wc -c < "$OUT_GIF")
  echo "  ${W}px @ ${F}fps -> $((BYTES / 1024)) KB"
  [ "$BYTES" -le "$MAX_BYTES" ] && break
done
rm -f "$PAL"

if [ "$(wc -c < "$OUT_GIF")" -gt "$MAX_BYTES" ]; then
  echo "Still over 10 MB at the smallest setting -- record a shorter clip." >&2
fi

# Wire it into the README the first time only, right under the title.
if ! grep -q 'docs/demo.gif' README.md; then
  /usr/bin/python3 - <<'PY'
lines = open('README.md').read().split('\n')
i = next(k for k, l in enumerate(lines) if l.startswith('# '))
lines.insert(i + 1, '\n![The live demo reading fingerspelled letters](docs/demo.gif)\n')
open('README.md', 'w').write('\n'.join(lines))
print('README.md: embedded docs/demo.gif')
PY
fi
echo "done: $OUT_GIF ($(du -h "$OUT_GIF" | cut -f1))"
