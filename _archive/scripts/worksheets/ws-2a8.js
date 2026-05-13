/**
 * scripts/worksheets/ws-2a8.js
 *
 * Renders equation-based questions for the 2A8 section.
 * Supports 10 variation types.
 *
 * Exports: window.WORKSHEET_2A8
 *   parseIntegerList(rawText)                                    → number[]
 *   renderEquationQuestion(a, b, operator, qIdx, style)          → HTML string
 *   buildEquationGrid(totalQ, variationId, seedNumbers, addend)  → HTML string
 */
(function () {

  // ── Helpers ──────────────────────────────────────────────────

  /**
   * Parses a comma/space-separated string of integers into an array.
   */
  function parseIntegerList(rawText) {
    return (rawText || '')
      .split(/[^\d-]+/)
      .map(v => parseInt(v, 10))
      .filter(v => Number.isFinite(v));
  }

  function clamp(val, min, max) {
    return Math.max(min, Math.min(max, val));
  }

  function randomInt(min, max) {
    return Math.floor(Math.random() * (max - min + 1)) + min;
  }

  // ── Equation HTML builders ────────────────────────────────────

  /**
   * Renders a standard  a ○ b = ___  equation row.
   * @param {number|string} a        - left operand (or blank placeholder)
   * @param {number|string} b        - right operand
   * @param {string}        operator - '+', '-'
   * @param {number}        qIdx     - 0-based index for numbering
   * @param {'answer'|'addend'} style - what gets the blank line
   */
  function renderEquationQuestion(a, b, operator, qIdx, style) {
    style = style || 'answer';
    const answer = operator === '+' ? (a + b) : (a - b);

    let left, right, result;
    if (style === 'addend') {
      // Missing addend: a + ___ = sum
      left   = a;
      right  = `<span class="eq-answer-line"></span>`;
      result = answer;
    } else {
      left   = a;
      right  = b;
      result = `<span class="eq-answer-line"></span>`;
    }

    return `
      <div class="q-block q-block-equation">
        <div class="q-num">(${qIdx + 1})</div>
        <div class="equation-row">
          <span>${left}</span>
          <span>${operator}</span>
          <span>${right}</span>
          <span>=</span>
          <span>${result}</span>
        </div>
      </div>
    `;
  }

  // ── Variation grid builders ───────────────────────────────────

  function renderSingleVariation(base, addend, variationId, qIdx) {
    let result = '';
    switch (variationId) {
      // E — Addition: base + addend = ___
      case 'E':
      default:
        result = renderEquationQuestion(base, addend, '+', qIdx, 'answer');
        break;

      // E2 — Missing addend: base + ___ = sum
      case 'E2':
        result = renderEquationQuestion(base, addend, '+', qIdx, 'addend');
        break;

      // E3 — Subtraction: base - addend = ___   (keep result ≥ 0)
      case 'E3': {
        const minuend    = Math.max(base, addend);
        const subtrahend = Math.min(base, addend);
        result = renderEquationQuestion(minuend, subtrahend, '-', qIdx, 'answer');
        break;
      }

      // E4 — Mixed: alternate + and -
      case 'E4': {
        const op = qIdx % 2 === 0 ? '+' : '-';
        const a  = op === '-' ? Math.max(base, addend) : base;
        const b  = op === '-' ? Math.min(base, addend) : addend;
        result = renderEquationQuestion(a, b, op, qIdx, 'answer');
        break;
      }

      // E5 — Doubles: base + base = ___
      case 'E5':
        result = renderEquationQuestion(base, base, '+', qIdx, 'answer');
        break;

      // E6 — Near doubles: base + (base ± 1) = ___
      case 'E6': {
        const nearB = base + (qIdx % 2 === 0 ? 1 : -1);
        const safeB = Math.max(0, nearB);
        result = renderEquationQuestion(base, safeB, '+', qIdx, 'answer');
        break;
      }

      // E7 — Adding 10: base + 10 = ___
      case 'E7':
        result = renderEquationQuestion(base, 10, '+', qIdx, 'answer');
        break;

      // E8 — Adding 0: base + 0 = ___
      case 'E8':
        result = renderEquationQuestion(base, 0, '+', qIdx, 'answer');
        break;

      // E9 — Number bonds to 10: base + ___ = 10
      case 'E9': {
        const bondBase = clamp(base, 0, 10);
        result = renderEquationQuestion(bondBase, 10 - bondBase, '+', qIdx, 'addend');
        break;
      }

      // E10 — Three-number addition: a + b + c = ___
      case 'E10': {
        const c = Math.max(0, addend - 1);
        result = `
          <div class="q-block q-block-equation">
            <div class="q-num">(${qIdx + 1})</div>
            <div class="equation-row">
              <span>${base}</span>
              <span>+</span>
              <span>${addend}</span>
              <span>+</span>
              <span>${c}</span>
              <span>=</span>
              <span class="eq-answer-line"></span>
            </div>
          </div>
        `;
        break;
      }
    }
    return result;
  }

  /**
   * Builds an equation grid for a given variation.
   */
  function buildEquationGrid(totalQ, variationId, seedNumbers, addend) {
    const seeds  = seedNumbers.length ? seedNumbers : [1];
    const safeQ  = Math.max(1, totalQ);
    const items  = [];

    for (let q = 0; q < safeQ; q++) {
      const base = seeds[q % seeds.length];
      items.push(renderSingleVariation(base, addend, variationId, q));
    }

    return `<div class="equation-grid">${items.join('')}</div>`;
  }

  // ── Public API ────────────────────────────────────────────────

  window.WORKSHEET_2A8 = {
    parseIntegerList,
    renderEquationQuestion,
    buildEquationGrid,
    renderSingleVariation,
  };

})();
