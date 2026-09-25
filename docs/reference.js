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
const FINGERS = [
  [1, 2, 3, 4],        // thumb
  [5, 6, 7, 8],        // index
  [9, 10, 11, 12],     // middle
  [13, 14, 15, 16],    // ring
  [17, 18, 19, 20],    // pinky
];
//: The palm slab: wrist, then across the knuckles and back. Drawn filled and behind the
//: fingers so the joins disappear into it.
const PALM = [0, 17, 13, 9, 5];

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

/** One <svg> hand. `motion` draws the trail marker J and Z need. */
export function handSvg(lm, { size = 84, motion = false } = {}) {
  const p = fit(lm, size, 11);
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${size} ${size}`);
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
  svg.setAttribute("aria-hidden", "true");
  const at = (i) => `${p[i][0].toFixed(2)},${p[i][1].toFixed(2)}`;

  // Palm first, so the finger capsules overlap it and the joins vanish.
  const palm = document.createElementNS(ns, "polygon");
  palm.setAttribute("points", PALM.map(at).join(" "));
  palm.setAttribute("class", "palm");
  svg.appendChild(palm);

  for (const chain of FINGERS) {
    const f = document.createElementNS(ns, "polyline");
    // Start the capsule at the knuckle's parent so the finger reads as joined to the palm.
    const from = chain[0] === 1 ? 0 : chain[0];
    f.setAttribute("points", [from, ...chain].map(at).join(" "));
    f.setAttribute("class", "finger");
    svg.appendChild(f);
  }

  if (motion) {
    const a = document.createElementNS(ns, "path");
    a.setAttribute("d", `M ${(size * 0.60).toFixed(1)} ${(size * 0.82).toFixed(1)} q ${(size * 0.16).toFixed(1)} ${(size * 0.11).toFixed(1)} ${(size * 0.27).toFixed(1)} ${(-size * 0.05).toFixed(1)}`);
    a.setAttribute("class", "trail");
    svg.appendChild(a);
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
