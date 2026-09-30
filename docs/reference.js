// The alphabet chart under the camera: one hand per letter, drawn from real landmarks.
//
// reference.json carries, for each letter, the MEDOID of that letter's training frames --
// 21 palm-normalized (x, y) pairs from a frame that actually happened. make_reference.py says
// why a medoid and not a mean. Nothing here is a photograph or a drawing: the chart is the same
// geometry the classifier consumes, so a letter that looks wrong here is a letter the model was
// taught wrong, which is the useful failure mode to have.
//
// MediaPipe's landmark order is fixed (0 wrist, 1-4 thumb, 5-8 index, 9-12 middle, 13-16 ring,
// 17-20 pinky), so the bone list below is a property of the landmarker, not of this data.

// Fingers as chunky capsules, the palm as a filled slab. An earlier version drew the 21
// landmarks as dots joined by thin bones, which is how MediaPipe visualizes a hand and reads
// as an x-ray: the point of this chart is to be friendly to somebody who does not fingerspell
// yet, and a skeleton is not that. Same landmarks, drawn as a hand instead of as its bones.

//: MediaPipe's landmark order is fixed: 0 wrist, 1-4 thumb, 5-8 index, 9-12 middle,
//: 13-16 ring, 17-20 pinky. These chains are a property of the landmarker, not of this data.
//: Each finger's chain, and how thick it is relative to the base width. A hand is not five
//: identical rods: the thumb is much the thickest, middle and index are close behind it, the
//: ring is slightly slimmer and the little finger is noticeably so. Drawing them all one width
//: is most of what made the first version read as a cartoon glove.
const FINGERS = [
  { chain: [1, 2, 3, 4], w: 1.24 },      // thumb
  { chain: [5, 6, 7, 8], w: 1.02 },      // index
  { chain: [9, 10, 11, 12], w: 1.06 },   // middle
  { chain: [13, 14, 15, 16], w: 0.95 },  // ring
  { chain: [17, 18, 19, 20], w: 0.80 },  // pinky
];
//: Base stroke in viewBox units, and how much each successive bone narrows toward the tip.
//: A real finger tapers; a polyline cannot, so each finger is drawn bone by bone instead.
const FINGER_BASE = 15;
const TAPER = 0.87;
//: The palm slab. Through the thumb's CMC rather than straight across the knuckles, because a
//: hand's palm includes the muscle at the base of the thumb; without it the silhouette is a
//: narrow bar with five sausages on it.
const PALM = [0, 1, 5, 9, 13, 17];
//: Every bone, for the skeleton drawn INSIDE the silhouette.
const BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4],
  [0, 5], [5, 6], [6, 7], [7, 8],
  [0, 9], [9, 10], [10, 11], [11, 12],
  [0, 13], [13, 14], [14, 15], [15, 16],
  [0, 17], [17, 18], [18, 19], [19, 20],
  [5, 9], [9, 13], [13, 17],
];

/** Fit the 21 points into a `size` box with padding, preserving aspect. */
function fit(lm, size, pad) {
  const xs = lm.map((p) => p[0]);
  const ys = lm.map((p) => p[1]);
  const x0 = Math.min(...xs);
  const x1 = Math.max(...xs);
  const y0 = Math.min(...ys);
  const y1 = Math.max(...ys);
  // One scale for both axes, or the hand shears and the letter stops being the letter.
  const s = (size - 2 * pad) / Math.max(x1 - x0, y1 - y0, 1e-6);
  const ox = pad + ((size - 2 * pad) - (x1 - x0) * s) / 2;
  const oy = pad + ((size - 2 * pad) - (y1 - y0) * s) / 2;
  return lm.map((p) => [ox + (p[0] - x0) * s, oy + (p[1] - y0) * s]);
}

/**
 * One <svg> hand: a filled silhouette with the landmarks drawn inside it.
 *
 * The silhouette is what makes it read as a hand -- a palm slab through the wrist, the thumb's
 * base and the four knuckles, with each finger laid over it as a round-capped capsule, all one
 * colour so the joins disappear. The skeleton on top is the actual MediaPipe graph, which is
 * the point of the chart: these are the 21 points the classifier sees, not an artist's idea of
 * the letter. Drawn alone it reads as an x-ray; drawn inside the hand it reads as a diagram.
 */
export function handSvg(lm, { size = 96, motion = false } = {}) {
  const p = fit(lm, size, 15);
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${size} ${size}`);
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
  svg.setAttribute("aria-hidden", "true");
  const at = (i) => `${p[i][0].toFixed(2)},${p[i][1].toFixed(2)}`;
  const add = (tag, cls, attrs) => {
    const el = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
    el.setAttribute("class", cls);
    svg.appendChild(el);
    return el;
  };

  // 1. the silhouette, palm first so the finger capsules overlap it. Widths are set here
  // rather than in CSS because they are geometry, not theme: the colour stays in the
  // stylesheet so the chart still follows the light and dark switch.
  add("polygon", "palm", { points: PALM.map(at).join(" ") });
  const scale = size / 96;
  for (const { chain, w } of FINGERS) {
    // The thumb is drawn from the wrist so its base merges into the palm; the other four start
    // at their own knuckle, which the palm slab already reaches. Prefixing chain[0] here would
    // emit a zero-length bone and shift the taper by one joint.
    const pts = chain[0] === 1 ? [0, ...chain] : chain;
    for (let j = 0; j < pts.length - 1; j++) {
      // Round caps on every bone, so the segments of a tapering finger blend into one shape.
      add("line", "finger", {
        x1: p[pts[j]][0].toFixed(2), y1: p[pts[j]][1].toFixed(2),
        x2: p[pts[j + 1]][0].toFixed(2), y2: p[pts[j + 1]][1].toFixed(2),
        "stroke-width": (FINGER_BASE * w * scale * TAPER ** j).toFixed(2),
      });
    }
  }

  // 2. the landmarks, inside it
  for (const [a, b] of BONES) {
    add("line", "bone", { x1: p[a][0].toFixed(2), y1: p[a][1].toFixed(2),
                          x2: p[b][0].toFixed(2), y2: p[b][1].toFixed(2) });
  }
  for (let i = 0; i < p.length; i++) {
    add("circle", "joint", { cx: p[i][0].toFixed(2), cy: p[i][1].toFixed(2), r: "1.9" });
  }

  if (motion) {
    add("path", "trail", { d: `M ${(size * 0.60).toFixed(1)} ${(size * 0.84).toFixed(1)} `
      + `q ${(size * 0.16).toFixed(1)} ${(size * 0.11).toFixed(1)} `
      + `${(size * 0.27).toFixed(1)} ${(-size * 0.05).toFixed(1)}` });
  }
  return svg;
}

/**
 * Build the chart into `host`. Returns { highlight(letter) } so the page can light up the
 * letter it just emitted -- which is the whole reason the chart is on the same screen as the
 * camera rather than in the README.
 */
export function buildChart(host, data) {
  const cells = new Map();
  const order = Object.keys(data.letters).sort();
  for (const letter of order) {
    const entry = data.letters[letter];
    const cell = document.createElement("figure");
    cell.className = "refcell";
    cell.appendChild(handSvg(entry.lm, { motion: Boolean(entry.motion) }));
    const cap = document.createElement("figcaption");
    const name = document.createElement("b");
    name.textContent = letter;
    cap.appendChild(name);
    if (entry.motion) {
      const dot = document.createElement("span");
      dot.className = "movetag";
      dot.textContent = "moves";
      cap.appendChild(dot);
    }
    // The sentence is not decoration. A projected skeleton cannot separate A from S from T,
    // or G from H, because what separates them is where the thumb is and which way the hand
    // points; make_reference.HOW carries that and the drawing carries the orientation.
    if (entry.how) {
      const how = document.createElement("span");
      how.className = "refhow";
      how.textContent = entry.how;
      cap.appendChild(how);
    }
    cell.appendChild(cap);
    if (entry.motion) cell.title = entry.motion;
    host.appendChild(cell);
    cells.set(letter, cell);
  }
  let lit = null;
  return {
    count: cells.size,
    highlight(letter) {
      if (lit && cells.has(lit)) cells.get(lit).classList.remove("lit");
      lit = cells.has(letter) ? letter : null;
      if (lit) cells.get(lit).classList.add("lit");
    },
  };
}
