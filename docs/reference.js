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

const BONES = [
  [0, 1], [1, 2], [2, 3], [3, 4],            // thumb
  [0, 5], [5, 6], [6, 7], [7, 8],            // index
  [0, 9], [9, 10], [10, 11], [11, 12],       // middle
  [0, 13], [13, 14], [14, 15], [15, 16],     // ring
  [0, 17], [17, 18], [18, 19], [19, 20],     // pinky
  [5, 9], [9, 13], [13, 17],                 // knuckle bridge
];
//: Fingertips get a slightly larger dot: they are what distinguishes most of the confusable pairs.
const TIPS = new Set([4, 8, 12, 16, 20]);

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
  const p = fit(lm, size, 9);
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${size} ${size}`);
  svg.setAttribute("width", String(size));
  svg.setAttribute("height", String(size));
  svg.setAttribute("aria-hidden", "true");
  for (const [a, b] of BONES) {
    const l = document.createElementNS(ns, "line");
    l.setAttribute("x1", p[a][0].toFixed(2));
    l.setAttribute("y1", p[a][1].toFixed(2));
    l.setAttribute("x2", p[b][0].toFixed(2));
    l.setAttribute("y2", p[b][1].toFixed(2));
    l.setAttribute("class", "bone");
    svg.appendChild(l);
  }
  for (let i = 0; i < p.length; i++) {
    const c = document.createElementNS(ns, "circle");
    c.setAttribute("cx", p[i][0].toFixed(2));
    c.setAttribute("cy", p[i][1].toFixed(2));
    c.setAttribute("r", TIPS.has(i) ? "2.6" : "1.7");
    c.setAttribute("class", TIPS.has(i) ? "joint tip" : "joint");
    svg.appendChild(c);
  }
  if (motion) {
    const a = document.createElementNS(ns, "path");
    a.setAttribute("d", `M ${size * 0.62} ${size * 0.80} q ${size * 0.16} ${size * 0.10} ${size * 0.26} ${-size * 0.06}`);
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
