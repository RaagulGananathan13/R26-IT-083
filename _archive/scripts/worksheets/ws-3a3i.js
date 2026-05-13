/**
 * scripts/worksheets/ws-3a3i.js
 *
 * Renders individual number-sequence questions for the 3A3I section.
 * Supports 11 variation types (A–K).
 *
 * Exports: window.WORKSHEET_3A3I
 *   renderQuestion(nums, variationId, qIdx, blankCount) → HTML string
 */
(function () {

  /** Box colour rotation (10 options for extended colour variety) */
  const BOX_COLORS = [
    'col-purple', 'col-blue', 'col-amber', 'col-green', 'col-pink',
    'col-teal', 'col-red', 'col-indigo', 'col-lime', 'col-sky',
  ];

  const CIRCLE_COLORS = [
    'circle-teal', 'circle-purple', 'circle-amber', 'circle-green', 'circle-pink',
  ];

  /** Line colours for Variation K (number-line style) */
  const LINE_COLORS = [
    { line: '#22c55e', border: '#16a34a', blankBorder: '#86efac', bg: '#f0fdf4' },  // green
    { line: '#ec4899', border: '#db2777', blankBorder: '#f9a8d4', bg: '#fdf2f8' },  // pink
    { line: '#3b82f6', border: '#2563eb', blankBorder: '#93c5fd', bg: '#eff6ff' },  // blue
    { line: '#f59e0b', border: '#d97706', blankBorder: '#fcd34d', bg: '#fffbeb' },  // amber
    { line: '#8b5cf6', border: '#7c3aed', blankBorder: '#c4b5fd', bg: '#f5f3ff' },  // purple
    { line: '#14b8a6', border: '#0d9488', blankBorder: '#5eead4', bg: '#f0fdfa' },  // teal
    { line: '#ef4444', border: '#dc2626', blankBorder: '#fca5a5', bg: '#fef2f2' },  // red
    { line: '#6366f1', border: '#4f46e5', blankBorder: '#a5b4fc', bg: '#eef2ff' },  // indigo
    { line: '#84cc16', border: '#65a30d', blankBorder: '#bef264', bg: '#f7fee7' },  // lime
    { line: '#06b6d4', border: '#0891b2', blankBorder: '#67e8f9', bg: '#ecfeff' },  // cyan
  ];

  // ── Helpers ──────────────────────────────────────────────────

  function colorForQ(qIdx) {
    return BOX_COLORS[qIdx % BOX_COLORS.length];
  }

  function circleColorForIdx(idx) {
    return CIRCLE_COLORS[idx % CIRCLE_COLORS.length];
  }

  function makeBox(n, colorClass) {
    return `<div class="num-box ${colorClass}">${n}</div>`;
  }

  function makeBlankBox() {
    return `<div class="num-box num-blank"></div>`;
  }

  function makeCircle(n, colorClass) {
    return `<div class="num-circle ${colorClass}">${n}</div>`;
  }

  function makeBlankCircle() {
    return `<div class="num-circle circle-blank"></div>`;
  }

  function row(items) {
    return `<div class="num-row">${items.join('')}</div>`;
  }

  function wrapBlock(qIdx, inner) {
    return `<div class="q-block"><div class="q-num">(${qIdx + 1})</div>${inner}</div>`;
  }

  // ── Variation Renderers ───────────────────────────────────────

  /**
   * A — Show all numbers, blank row below for copying.
   */
  function renderA(nums, qIdx) {
    const col = colorForQ(qIdx);
    const topRow  = row(nums.map(n => makeBox(n, col)));
    const botRow  = row(nums.map(() => makeBlankBox()));
    return wrapBlock(qIdx, topRow + botRow);
  }

  /**
   * B — Fill in the first N blanks from the left.
   */
  function renderB(nums, qIdx, blankCount) {
    const col = colorForQ(qIdx);
    const bc  = Math.min(blankCount, nums.length - 1);
    const topRow = row(nums.map(n => makeBox(n, col)));
    const botRow = row(nums.map((n, i) =>
      i < bc ? makeBlankBox() : makeBox(n, col)
    ));
    return wrapBlock(qIdx, topRow + botRow);
  }

  /**
   * C — Circles with randomly missing values.
   */
  function renderC(nums, qIdx, blankCount) {
    const bc = Math.min(blankCount, Math.floor(nums.length * 0.4));
    const blankSet = new Set();
    while (blankSet.size < bc) blankSet.add(Math.floor(Math.random() * nums.length));

    const topRow = row(nums.map((n, i) =>
      blankSet.has(i) ? makeBlankCircle() : makeCircle(n, circleColorForIdx(i))
    ));
    const botRow = row(nums.map((n, i) =>
      makeCircle(n + nums.length, circleColorForIdx(i))
    ));
    return wrapBlock(qIdx, topRow + botRow);
  }

  /**
   * D — Fill in the last N blanks from the right.
   */
  function renderD(nums, qIdx, blankCount) {
    const col = colorForQ(qIdx);
    const bc  = Math.min(blankCount, nums.length - 1);
    const topRow = row(nums.map(n => makeBox(n, col)));
    const botRow = row(nums.map((n, i) => {
      const fromEnd = nums.length - 1 - i;
      return fromEnd < bc ? makeBlankBox() : makeBox(n, col);
    }));
    return wrapBlock(qIdx, topRow + botRow);
  }

  /**
   * E — Sequence in reverse (countdown) order, write forward below.
   */
  function renderE(nums, qIdx) {
    const col  = colorForQ(qIdx);
    const rev  = [...nums].reverse();
    const topRow = row(rev.map(n => makeBox(n, col)));
    const botRow = row(nums.map(() => makeBlankBox()));
    return wrapBlock(qIdx, topRow + botRow);
  }

  /**
   * F — Show only odd-indexed numbers; even-indexed are blank.
   */
  function renderF(nums, qIdx) {
    const col = colorForQ(qIdx);
    const topRow = row(nums.map((n, i) =>
      i % 2 === 0 ? makeBox(n, col) : makeBlankBox()
    ));
    const botRow = row(nums.map(n => makeBlankBox()));
    return wrapBlock(qIdx, topRow + botRow);
  }

  /**
   * G — Show only even-indexed numbers; odd-indexed are blank.
   */
  function renderG(nums, qIdx) {
    const col = colorForQ(qIdx);
    const topRow = row(nums.map((n, i) =>
      i % 2 !== 0 ? makeBox(n, col) : makeBlankBox()
    ));
    const botRow = row(nums.map(n => makeBlankBox()));
    return wrapBlock(qIdx, topRow + botRow);
  }

  /**
   * H — Skip-count by 2: show every other value, blank the rest.
   */
  function renderH(nums, qIdx) {
    const col = colorForQ(qIdx);
    const topRow = row(nums.map((n, i) =>
      i % 2 === 0 ? makeBox(n, col) : makeBlankBox()
    ));
    const botRow = row(nums.map(n => makeBlankBox()));
    return wrapBlock(qIdx, topRow + botRow);
  }

  /**
   * I — Skip-count by 5: show every 5th, blank the rest.
   */
  function renderI(nums, qIdx) {
    const col = colorForQ(qIdx);
    const topRow = row(nums.map((n, i) =>
      i % 5 === 0 ? makeBox(n, col) : makeBlankBox()
    ));
    const botRow = row(nums.map(n => makeBlankBox()));
    return wrapBlock(qIdx, topRow + botRow);
  }

  /**
   * J — Random blanks scattered throughout a single row.
   */
  function renderJ(nums, qIdx, blankCount) {
    const col = colorForQ(qIdx);
    const bc  = Math.max(1, Math.min(blankCount, Math.floor(nums.length * 0.45)));
    const blankSet = new Set();
    while (blankSet.size < bc) blankSet.add(Math.floor(Math.random() * nums.length));

    const topRow = row(nums.map((n, i) =>
      blankSet.has(i) ? makeBlankBox() : makeBox(n, col)
    ));
    return wrapBlock(qIdx, topRow);
  }

  /**
   * K — Number-line: boxes connected by thick coloured horizontal lines.
   *     Some boxes filled, some blank for students to write in.
   *     Each sub-row gets a distinct line colour (green, pink, blue, etc.)
   */
  function renderK(nums, qIdx, blankCount) {
    const bc = Math.max(2, Math.min(blankCount, Math.floor(nums.length * 0.35)));
    const blankSet = new Set();

    // Ensure blanks are not at position 0 or last so the line looks continuous
    while (blankSet.size < bc) {
      const idx = Math.floor(Math.random() * nums.length);
      blankSet.add(idx);
    }

    const lineColor = LINE_COLORS[qIdx % LINE_COLORS.length];

    const cells = nums.map((n, i) => {
      const isBlank = blankSet.has(i);
      const connector = i < nums.length - 1
        ? `<div class="num-line-connector" style="background:${lineColor.line}"></div>`
        : '';

      if (isBlank) {
        return `<div class="num-line-cell">
          <div class="num-line-box num-line-blank" style="border-color:${lineColor.blankBorder}; background:${lineColor.bg}"></div>
          ${connector}
        </div>`;
      }
      return `<div class="num-line-cell">
        <div class="num-line-box num-line-filled" style="border-color:${lineColor.border}">${n}</div>
        ${connector}
      </div>`;
    });

    const lineRow = `<div class="num-line-row">${cells.join('')}</div>`;
    return wrapBlock(qIdx, lineRow);
  }

  // ── Public API ────────────────────────────────────────────────

  /**
   * renderQuestion(nums, variationId, qIdx, blankCount) → HTML string
   *
   * @param {number[]} nums        - The number sequence for this question
   * @param {string}   variationId - One of A-K
   * @param {number}   qIdx        - 0-based question position (for colour cycling)
   * @param {number}   blankCount  - How many blanks to leave (used by B, C, D, J, K)
   */
  function renderQuestion(nums, variationId, qIdx, blankCount) {
    switch (variationId) {
      case 'A': return renderA(nums, qIdx);
      case 'B': return renderB(nums, qIdx, blankCount);
      case 'C': return renderC(nums, qIdx, blankCount);
      case 'D': return renderD(nums, qIdx, blankCount);
      case 'E': return renderE(nums, qIdx);
      case 'F': return renderF(nums, qIdx);
      case 'G': return renderG(nums, qIdx);
      case 'H': return renderH(nums, qIdx);
      case 'I': return renderI(nums, qIdx);
      case 'J': return renderJ(nums, qIdx, blankCount);
      case 'K': return renderK(nums, qIdx, blankCount);
      default:  return renderA(nums, qIdx);
    }
  }

  window.WORKSHEET_3A3I = { renderQuestion };

})();
