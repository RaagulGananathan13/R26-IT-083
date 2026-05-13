/**
 * scripts/worksheets/ws-4a.js
 *
 * Handles number-tracing questions for the 4A section (numbers 1–10).
 * Each "type" (4A1–4A10) generates a 2-question tracing page with
 * optional image strips above each question.
 *
 * Exports: window.WORKSHEET_4A
 */
(function () {

  /* ── Constants ─────────────────────────────────────────────── */

  const NUMBER_WORDS = {
    1: 'one',  2: 'two',   3: 'three', 4: 'four', 5: 'five',
    6: 'six',  7: 'seven', 8: 'eight', 9: 'nine', 10: 'ten',
  };

  const TRACE_COLORS = [
    '#1d4ed8', '#db2777', '#0f766e', '#4c1d95', '#dc2626',
    '#0ea5e9', '#b45309', '#be185d', '#047857', '#0369a1',
  ];

  const TRACE_GUIDES = {
    1:  { start: { x: 50, y: 16 }, end: { x: 50, y: 86 }, hint: 'Start at the dot and draw straight down to the star.', paths: ['M50 16 L50 86'] },
    2:  { start: { x: 40, y: 22 }, end: { x: 66, y: 80 }, hint: 'Start at the dot, curve around, then slide down and finish with a short line.', paths: ['M40 22 C52 10 75 14 70 35 C66 50 52 60 42 72 L35 80 L66 80'] },
    3:  { start: { x: 60, y: 25 }, end: { x: 40, y: 79 }, hint: 'Start at the dot, make the top curve, then the bottom curve to the star.', paths: ['M60 25 C56 14 40 14 38 28 C36 40 47 45 56 48 C65 52 68 62 60 71 C54 78 47 80 40 79'] },
    4:  { start: { x: 39, y: 21 }, end: { x: 66, y: 86 }, hint: 'Start at the dot, draw down and across, then go down to the star.', paths: ['M39 21 L39 58 L66 58', 'M66 21 L66 86'] },
    5:  { start: { x: 62, y: 21 }, end: { x: 42, y: 84 }, hint: 'Start at the dot, go across, down, and curve around to the star.', paths: ['M62 21 L40 21 L40 49 C48 42 60 44 64 54 C68 68 60 84 42 84'] },
    6:  { start: { x: 60, y: 22 }, end: { x: 57, y: 84 }, hint: 'Start at the dot, curve around, and close the loop at the star.', paths: ['M60 22 C46 20 36 34 38 52 C40 74 58 88 68 74 C74 64 66 52 54 56 C46 58 44 72 57 84'] },
    7:  { start: { x: 35, y: 22 }, end: { x: 62, y: 84 }, hint: 'Start at the dot, draw the top line, then slant down to the star.', paths: ['M35 22 L70 22 L62 84'] },
    8:  { start: { x: 52, y: 20 }, end: { x: 52, y: 84 }, hint: 'Start at the dot, make the top loop, then the bottom loop to the star.', paths: ['M52 20 C40 20 36 36 48 42 C60 48 68 36 60 24 C58 20 54 20 52 20', 'M48 46 C34 50 34 72 52 84 C68 74 70 54 56 48 C52 46 50 46 48 46'] },
    9:  { start: { x: 59, y: 36 }, end: { x: 49, y: 84 }, hint: 'Start at the dot, make a loop, then draw down to the star.', paths: ['M59 36 C58 20 44 14 34 28 C26 40 30 56 44 60 C56 62 64 52 60 40', 'M60 36 L49 84'] },
    10: { start: { x: 35, y: 16 }, end: { x: 35, y: 84 }, hint: 'Start at the dot, trace 1 first, then trace 0 and finish at the star.', paths: ['M35 16 L35 84', 'M72 84 C84 74 84 28 72 18 C56 8 42 24 44 50 C46 76 60 90 72 84'] },
  };

  /* ── Helpers ───────────────────────────────────────────────── */

  function escapeHtml(value) {
    return String(value || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;')
      .replace(/'/g, '&#39;');
  }

  function getTraceNumberFromType(type) {
    const match = /^4A(10|[1-9])$/.exec(String(type || '').trim());
    return match ? parseInt(match[1], 10) : null;
  }

  function getWorksheetCode(traceNumber) {
    return `4A${traceNumber}`;
  }

  function getTraceWord(traceNumber) {
    return NUMBER_WORDS[traceNumber] || String(traceNumber);
  }

  function normalizeUrl(url) {
    return String(url || '').trim();
  }

  function getTraceGuide(traceNumber) {
    return TRACE_GUIDES[traceNumber] || {
      start: { x: 40, y: 18 }, end: { x: 62, y: 84 },
      hint: 'Start at the dot and trace to the star.', paths: ['M40 18 L62 84'],
    };
  }

  /* ── Image field helpers (sidebar inputs) ──────────────────── */

  function normalizeImageValues(traceNumber, values) {
    const expected = traceNumber * 2;
    const normalized = [];
    for (let i = 0; i < expected; i++) {
      normalized.push(normalizeUrl(values?.[i]));
    }
    return normalized;
  }

  function splitImageValuesByQuestion(traceNumber, values) {
    const normalized = normalizeImageValues(traceNumber, values);
    return [
      normalized.slice(0, traceNumber),
      normalized.slice(traceNumber, traceNumber * 2),
    ];
  }

  function buildImageFieldsHTML(traceNumber, initialValues) {
    const safeNumber = Math.max(1, Math.min(10, parseInt(traceNumber, 10) || 1));
    const values = normalizeImageValues(safeNumber, initialValues);
    let html = '';

    for (let qi = 0; qi < 2; qi++) {
      html += `<div class="trace-image-group">`;
      html += `<div class="trace-image-group-title">Question ${qi + 1} images (${safeNumber})</div>`;

      for (let slot = 0; slot < safeNumber; slot++) {
        const flatIndex = qi * safeNumber + slot;
        const safeValue = escapeHtml(values[flatIndex]);
        html += `
          <label class="trace-image-field">
            <span class="trace-image-label">Q${qi + 1} Image ${slot + 1}</span>
            <input
              type="url"
              class="field-input trace-image-url"
              data-image-index="${flatIndex}"
              data-question-index="${qi}"
              data-slot-index="${slot}"
              value="${safeValue}"
              placeholder="https://example.com/image-${slot + 1}.png"
            />
          </label>
        `;
      }
      html += `</div>`;
    }
    return html;
  }

  function readImageValuesFromDOM(traceNumber, root) {
    const safeNumber = Math.max(1, Math.min(10, parseInt(traceNumber, 10) || 1));
    const scope = root || document;
    const inputs = scope.querySelectorAll('.trace-image-url');
    const values = Array.from(inputs).map(input => normalizeUrl(input.value));
    return normalizeImageValues(safeNumber, values);
  }

  /* ── Question HTML ─────────────────────────────────────────── */

  function buildImageTiles(traceNumber, questionIndex, urls) {
    return urls.map((url, idx) => {
      const safeUrl        = normalizeUrl(url);
      const safeAttrUrl    = escapeHtml(safeUrl);
      const fallbackHidden = safeUrl ? ' style="display:none;"' : '';
      return `
        <div class="trace4a-image-tile" role="img" aria-label="Image ${idx + 1}">
          ${safeUrl ? `<img class="trace4a-image" src="${safeAttrUrl}" alt="Question ${questionIndex + 1} image ${idx + 1}" onerror="this.style.display='none';this.nextElementSibling.style.display='flex';"/>` : ''}
          <span class="trace4a-image-fallback"${fallbackHidden}>Image ${idx + 1}</span>
        </div>
      `;
    }).join('');
  }

  function buildQuestionHTML(traceNumber, questionIndex, imageUrls) {
    const worksheetCode = getWorksheetCode(traceNumber);
    const subCode       = `${worksheetCode}${questionIndex === 0 ? 'a' : 'b'}`;
    const numberWord    = escapeHtml(getTraceWord(traceNumber));
    const color         = TRACE_COLORS[(traceNumber - 1) % TRACE_COLORS.length];
    const guide         = getTraceGuide(traceNumber);

    return `
      <div class="trace4a-question">
        <div class="trace4a-code">${subCode}</div>
        <div class="trace4a-image-strip" style="--trace-count:${traceNumber};">
          ${buildImageTiles(traceNumber, questionIndex, imageUrls)}
        </div>
        <div class="trace4a-body">
          <div class="trace4a-left">
            <div class="trace4a-caption">Write the number.</div>
            <div class="trace4a-word">${numberWord}</div>
            <div class="trace4a-digit">${traceNumber}</div>
          </div>
          <div class="trace4a-template" style="background:${color};">
            <div class="trace4a-template-guide" aria-hidden="true">
              <span class="trace4a-start-dot" style="left:${guide.start.x}%;top:${guide.start.y}%;"></span>
              <span class="trace4a-end-star"  style="left:${guide.end.x}%;top:${guide.end.y}%;">★</span>
            </div>
            <div class="trace4a-template-digit">${traceNumber}</div>
          </div>
        </div>
      </div>
    `;
  }

  /* ── Worksheet data builder ────────────────────────────────── */

  function buildWorksheetData(options) {
    const traceNumber = Math.max(1, Math.min(10, parseInt(options?.traceNumber, 10) || 1));
    const imageSets   = splitImageValuesByQuestion(traceNumber, options?.imageValues || []);
    const guide       = getTraceGuide(traceNumber);

    return {
      worksheetCode:   getWorksheetCode(traceNumber),
      activityName:    'Number Tracing',
      instructionText: 'Trace the number.',
      hintText:        guide.hint,
      questionsHtml: `
        <div class="trace4a-questions">
          ${buildQuestionHTML(traceNumber, 0, imageSets[0])}
          ${buildQuestionHTML(traceNumber, 1, imageSets[1])}
        </div>
      `,
      questionCount: 2,
    };
  }

  /* ── Exports ───────────────────────────────────────────────── */

  window.WORKSHEET_4A = {
    getTraceNumberFromType,
    getWorksheetCode,
    buildImageFieldsHTML,
    readImageValuesFromDOM,
    buildWorksheetData,
  };

})();
