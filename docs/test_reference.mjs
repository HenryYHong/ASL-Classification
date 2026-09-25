// The alphabet chart: reference.json's shape, and that reference.js turns it into hands.
//
// The chart is the one thing on the page that a visitor who cannot already fingerspell has to
// rely on, so the failure that matters is a quiet one -- a letter missing, a description lost,
// a skeleton drawn from the wrong number of landmarks. Every check here is against the
// committed reference.json, so a regenerate that drops a letter fails the suite.
import { readFileSync } from 'node:fs';

// A DOM stub, not jsdom: this repository has no node_modules and every other suite builds the
// few methods it needs by hand. reference.js touches exactly what is implemented below, so a
// method it starts using that is missing here throws rather than silently doing nothing.
class El {
  constructor(tag = 'div') {
    this.tagName = tag.toUpperCase(); this.children = []; this.attrs = {};
    this.classList = new Set(); this._text = ''; this.parent = null; this.listeners = {};
  }
  set className(v) { this.classList = new Set(String(v).split(/\s+/).filter(Boolean)); }
  get className() { return [...this.classList].join(' '); }
  set textContent(v) { this._text = String(v); this.children = []; }
  get textContent() { return this._text + this.children.map((c) => c.textContent).join(''); }
  // The real setAttribute('class', ...) is the same channel as className: reference.js sets
  // classes on SVG nodes that way, and a stub that kept them apart would pass a .trail query
  // that the browser fails.
  setAttribute(k, v) {
    this.attrs[k] = String(v);
    if (k === 'class') this.className = v;
  }
  getAttribute(k) { return k in this.attrs ? this.attrs[k] : null; }
  appendChild(el) { el.parent = this; this.children.push(el); return el; }
  addEventListener(ev, fn) { (this.listeners[ev] ||= []).push(fn); }
  _walk(out = []) { for (const c of this.children) { out.push(c); c._walk(out); } return out; }
  querySelectorAll(sel) {
    const want = sel.replace('.', '');
    return this._walk().filter((e) => (sel.startsWith('.')
      ? e.classList.has(want) : e.tagName.toLowerCase() === sel.toLowerCase()));
  }
  querySelector(sel) { return this.querySelectorAll(sel)[0] || null; }
}
// classList in the real DOM has add/remove; Set has add/delete. Bridge the one name that differs.
Object.defineProperty(Set.prototype, 'remove', { value(v) { return this.delete(v); }, configurable: true });
Object.defineProperty(Set.prototype, 'contains', { value(v) { return this.has(v); }, configurable: true });
globalThis.document = {
  createElement: (t) => new El(t),
  createElementNS: (_ns, t) => new El(t),
  getElementById: () => null,
};

let pass = 0, fail = 0;
const check = (name, ok, extra = '') => {
  if (ok) { pass++; console.log(`pass  ${name}${extra ? '  ' + extra : ''}`); }
  else { fail++; console.log(`FAIL  ${name}${extra ? '  ' + extra : ''}`); }
};

const data = JSON.parse(readFileSync(new URL('./reference.json', import.meta.url)));
const LETTERS = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'.split('');
const MOTION = ['J', 'Z'];

check('all 26 letters present', LETTERS.every((L) => data.letters[L]),
  `have ${Object.keys(data.letters).length}`);
check('every letter has 21 landmarks',
  LETTERS.every((L) => Array.isArray(data.letters[L]?.lm) && data.letters[L].lm.length === 21));
check('every landmark is a finite (x, y) pair',
  LETTERS.every((L) => data.letters[L].lm.every(
    (p) => p.length === 2 && p.every((v) => Number.isFinite(v)))));
// A medoid is a real frame, so no letter may be degenerate: a collapsed hand means the letter
// had one frame, or the palm scale divided by something near zero.
check('no letter collapses to a point', LETTERS.every((L) => {
  const xs = data.letters[L].lm.map((p) => p[0]), ys = data.letters[L].lm.map((p) => p[1]);
  return (Math.max(...xs) - Math.min(...xs)) > 0.2 && (Math.max(...ys) - Math.min(...ys)) > 0.2;
}));
check('every letter carries a written description',
  LETTERS.every((L) => typeof data.letters[L].how === 'string' && data.letters[L].how.length > 15));
check('exactly J and Z are marked as moving',
  LETTERS.filter((L) => data.letters[L].motion).join('') === MOTION.join(''));
check('J starts from I and Z from D',
  data.letters.J.pose === 'I' && data.letters.Z.pose === 'D');
// The six fists are the letters the drawing cannot separate. Their descriptions are what does,
// so each must actually say something different about the thumb.
const fists = ['A', 'E', 'M', 'N', 'S', 'T'];
check('the six fists have distinct descriptions',
  new Set(fists.map((L) => data.letters[L].how)).size === fists.length);
check('every fist description mentions the thumb',
  fists.every((L) => /thumb/i.test(data.letters[L].how)));

const { buildChart, handSvg } = await import('./reference.js');
const host = new El('div');
const chart = buildChart(host, data);
check('buildChart renders one cell per letter', chart.count === 26 && host.children.length === 26);

const first = host.children[0];
check('a cell is an svg plus a caption',
  first.tagName.toLowerCase() === 'figure' && first.querySelector('svg') && first.querySelector('figcaption'));
// One filled palm and five capsule fingers -- the shapes, not the skeleton the first version
// drew. A hand that loses a finger here is a hand that lost a landmark chain.
check('a hand is one palm and five fingers',
  first.querySelectorAll('polygon').length === 1 && first.querySelectorAll('polyline').length === 5);
check('no bare joints or bones are drawn (it must not read as an x-ray)',
  first.querySelectorAll('circle').length === 0 && first.querySelectorAll('line').length === 0);
check('only the moving letters draw a trail',
  [...host.children].filter((c) => c.querySelector('.trail')).length === 2);

// The drawing must fit its box whatever the landmarks are, or a cell overlaps its neighbour.
const svg = handSvg(data.letters.L.lm, { size: 84 });
const pts = (el) => el.getAttribute('points').trim().split(/\s+/).map((pair) => pair.split(',').map(Number));
const coords = [...svg.querySelectorAll('polygon'), ...svg.querySelectorAll('polyline')]
  .flatMap((el) => pts(el).flat());
check('the hand is fitted inside its box',
  coords.every((v) => v >= 0 && v <= 84), `range ${Math.min(...coords).toFixed(1)}..${Math.max(...coords).toFixed(1)}`);
// One scale for both axes: a per-axis fit would stretch L's long thumb and stop it being an L.
const xs = coords.filter((_, i) => i % 2 === 0), ys = coords.filter((_, i) => i % 2 === 1);
const src = data.letters.L.lm;
const sx = (Math.max(...xs) - Math.min(...xs)) / (Math.max(...src.map((p) => p[0])) - Math.min(...src.map((p) => p[0])));
const sy = (Math.max(...ys) - Math.min(...ys)) / (Math.max(...src.map((p) => p[1])) - Math.min(...src.map((p) => p[1])));
// Coordinates are written with toFixed(2), so the two recovered scales agree only to within
// that rounding -- a relative tolerance, not an exact one.
check('aspect is preserved (one scale for both axes)',
  Math.abs(sx - sy) / Math.max(sx, sy) < 2e-3, `sx ${sx.toFixed(4)} sy ${sy.toFixed(4)}`);

chart.highlight('K');
check('highlight lights exactly one cell',
  [...host.children].filter((c) => c.classList.has('lit')).length === 1);
chart.highlight('7');
check('a letter with no cell (a digit) lights nothing',
  [...host.children].filter((c) => c.classList.has('lit')).length === 0);

// index.html and app.js must actually use it, or the chart is dead code that still passes.
const html = readFileSync(new URL('./index.html', import.meta.url), 'utf8');
const app = readFileSync(new URL('./app.js', import.meta.url), 'utf8');
check('index.html hosts the chart', /id="refchart"/.test(html) && /id="reftoggle"/.test(html));
check('app.js builds it and highlights emissions',
  /buildChart/.test(app) && /chart\.highlight\(em\.letter\)/.test(app));
check('the note admits the drawings are flat and one signer',
  /one signer's hands/.test(html) && /flat/.test(html));

console.log(`\n${pass} pass, ${fail} FAIL`);
process.exit(fail ? 1 : 0);
