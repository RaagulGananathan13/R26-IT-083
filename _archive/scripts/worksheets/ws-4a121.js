/**
 * scripts/worksheets/ws-4a121.js
 *
 * Handles image-counting questions for the 4A121 section.
 * Supports 10 variation types (F, F2 … F10).
 *
 * Exports: window.WORKSHEET_4A121
 */
(function () {

  /* ── Configuration ─────────────────────────────────────────── */
  const WORKSHEET_IMAGES_CONFIG = {
    sectionImages: {
      '4A121': '',
    },
    sectionGrid: {
      '4A121': { count: 15, perRow: 5 },
    },
    sectionPrompts: {
      '4A121': 'How many butterflies are there? Write the number in the box.',
    },
  };

  /* ── Helpers ───────────────────────────────────────────────── */

  function parseBoundedInt(rawValue, fallbackValue, minValue, maxValue) {
    const parsed = parseInt(rawValue, 10);
    if (!Number.isFinite(parsed)) return fallbackValue;
    return Math.max(minValue, Math.min(maxValue, parsed));
  }

  function getConfiguredImageForSection(sectionName) {
    const raw = WORKSHEET_IMAGES_CONFIG?.sectionImages?.[sectionName];
    if (typeof raw !== 'string' || !raw.trim()) return '';
    return /^https?:\/\//i.test(raw.trim()) ? raw.trim() : '';
  }

  function getConfiguredPromptForSection(sectionName) {
    const raw = WORKSHEET_IMAGES_CONFIG?.sectionPrompts?.[sectionName];
    return typeof raw === 'string' ? raw.trim() : '';
  }

  function getConfiguredGridForSection(sectionName) {
    const raw = WORKSHEET_IMAGES_CONFIG?.sectionGrid?.[sectionName];
    if (!raw || typeof raw !== 'object') return { count: 15, perRow: 5 };
    return {
      count:  parseBoundedInt(raw.count,  15, 1, 40),
      perRow: parseBoundedInt(raw.perRow, 5,  1, 8),
    };
  }

  function getSuitableGridRange(perRowValue) {
    const safePerRow = parseBoundedInt(perRowValue, 5, 1, 8);
    return { perRow: safePerRow, minCount: 1, maxCount: safePerRow * 5 };
  }

  function getAdaptiveImageTileSize(imageCount, imagePerRow) {
    const safeCount  = parseBoundedInt(imageCount,  15, 1, 40);
    const safePerRow = parseBoundedInt(imagePerRow, 5,  1, 8);
    const rows       = Math.ceil(safeCount / safePerRow);
    const gapSize    = safePerRow <= 4 ? 14 : (safePerRow <= 6 ? 12 : 10);
    const byCols     = Math.floor((640 - gapSize * (safePerRow - 1)) / safePerRow);
    const byRows     = Math.floor((480 - gapSize * (rows - 1)) / Math.max(rows, 1));
    return Math.max(56, Math.min(160, Math.min(byCols, byRows)));
  }

  /* ── Image Grid HTML ───────────────────────────────────────── */

  function buildImageGrid(imageUrl, imageCount, imagePerRow) {
    const safeCount  = parseBoundedInt(imageCount,  15, 1, 40);
    const safePerRow = parseBoundedInt(imagePerRow, 5,  1, 8);
    const tileSize   = getAdaptiveImageTileSize(safeCount, safePerRow);
    const gapSize    = safePerRow <= 4 ? 14 : (safePerRow <= 6 ? 12 : 10);

    if (!imageUrl) {
      return `<div class="ws-image-missing">
        Image link is missing. Paste a valid URL in the 4A121 Image Link field in the sidebar.
      </div>`;
    }

    const tiles = Array.from({ length: safeCount }, () =>
      `<img class="ws-image-item" src="${imageUrl}" alt="Worksheet object counting visual"/>`
    ).join('');

    return `<div class="ws-image-grid" style="grid-template-columns:repeat(${safePerRow},${tileSize}px);gap:${gapSize}px;">${tiles}</div>`;
  }

  /* ── Variation Renderers ───────────────────────────────────── */

  /**
   * F — Standard: count repeated images → write in box.
   */
  function renderF(imageUrl, imageCount, imagePerRow) {
    const grid = buildImageGrid(imageUrl, imageCount, imagePerRow);
    return `
      <div class="ws-image-question">
        <div class="ws-image-frame">
          ${grid}
          <div class="ws-answer-box-row"><div class="ws-answer-box" aria-label="Answer box"></div></div>
        </div>
      </div>
    `;
  }

  /**
   * F2 — Circle the correct numeral (shows 3 choices).
   */
  function renderF2(imageUrl, imageCount, imagePerRow) {
    const count   = parseBoundedInt(imageCount, 15, 1, 40);
    const wrong1  = count + 1;
    const wrong2  = Math.max(1, count - 1);
    const choices = [wrong2, count, wrong1].sort((a, b) => a - b);
    const grid    = buildImageGrid(imageUrl, count, imagePerRow);
    const choiceHTML = choices.map(c =>
      `<span style="display:inline-block;border:2px solid #9ca3af;border-radius:8px;padding:4px 14px;font-size:22px;font-weight:800;font-family:'Nunito',sans-serif;">${c}</span>`
    ).join('');
    return `
      <div class="ws-image-question">
        <div class="ws-image-frame">
          ${grid}
          <div style="display:flex;gap:16px;align-items:center;padding:4px;">${choiceHTML}</div>
        </div>
      </div>
    `;
  }

  /**
   * F3 — Count → write the number word and numeral.
   */
  function renderF3(imageUrl, imageCount, imagePerRow) {
    const grid = buildImageGrid(imageUrl, imageCount, imagePerRow);
    return `
      <div class="ws-image-question">
        <div class="ws-image-frame">
          ${grid}
          <div style="display:flex;gap:24px;align-items:center;">
            <div class="ws-fill-line" style="font-size:13px;">Word:<span class="dots" style="min-width:80px;"></span></div>
            <div class="ws-fill-line" style="font-size:13px;">Number:<span class="dots" style="min-width:50px;"></span></div>
          </div>
        </div>
      </div>
    `;
  }

  /**
   * F4 — Two groups: count each and add.
   */
  function renderF4(imageUrl, imageCount, imagePerRow) {
    const half1 = Math.ceil(parseBoundedInt(imageCount, 15, 1, 40) / 2);
    const half2 = Math.floor(parseBoundedInt(imageCount, 15, 1, 40) / 2);
    const grid1 = buildImageGrid(imageUrl, half1, imagePerRow);
    const grid2 = buildImageGrid(imageUrl, half2, imagePerRow);
    return `
      <div class="ws-image-question">
        <div class="ws-image-frame" style="flex-direction:row;gap:12px;">
          <div style="flex:1;">${grid1}</div>
          <div style="font-size:28px;font-weight:800;align-self:center;">+</div>
          <div style="flex:1;">${grid2}</div>
          <div style="font-size:28px;font-weight:800;align-self:center;">=</div>
          <div class="ws-answer-box" style="align-self:center;"></div>
        </div>
      </div>
    `;
  }

  /**
   * F5–F10 — Additional variations (simplified renderers that reuse grid + custom label).
   */
  function renderFGeneric(imageUrl, imageCount, imagePerRow, promptOverride) {
    const grid = buildImageGrid(imageUrl, imageCount, imagePerRow);
    return `
      <div class="ws-image-question">
        <div class="ws-image-frame">
          ${grid}
          <div style="font-size:13px;font-weight:700;color:#374151;text-align:center;">${promptOverride}</div>
          <div class="ws-answer-box-row"><div class="ws-answer-box" aria-label="Answer box"></div></div>
        </div>
      </div>
    `;
  }

  /* ── Public router ─────────────────────────────────────────── */

  /**
   * renderImageQuestion(imageUrl, imageCount, imagePerRow, variationId) → HTML
   */
  function renderImageQuestion(imageUrl, imageCount, imagePerRow, variationId) {
    switch (variationId) {
      case 'F':   return renderF(imageUrl, imageCount, imagePerRow);
      case 'F2':  return renderF2(imageUrl, imageCount, imagePerRow);
      case 'F3':  return renderF3(imageUrl, imageCount, imagePerRow);
      case 'F4':  return renderF4(imageUrl, imageCount, imagePerRow);
      case 'F5':  return renderFGeneric(imageUrl, imageCount, imagePerRow, 'Match the count to the number line.');
      case 'F6':  return renderFGeneric(imageUrl, imageCount, imagePerRow, 'Which group has more? Circle it.');
      case 'F7':  return renderFGeneric(imageUrl, imageCount, imagePerRow, 'Count in groups of 2 and write the total.');
      case 'F8':  return renderFGeneric(imageUrl, imageCount, imagePerRow, 'Count in groups of 5 and write the total.');
      case 'F9':  return renderFGeneric(imageUrl, imageCount, imagePerRow, 'Count total, cross out 2, write how many are left.');
      case 'F10': return renderFGeneric(imageUrl, imageCount, imagePerRow, 'Order the three sets from least to greatest.');
      default:    return renderF(imageUrl, imageCount, imagePerRow);
    }
  }

  /* ── Exports ───────────────────────────────────────────────── */

  window.WORKSHEET_4A121 = {
    parseBoundedInt,
    getConfiguredImageForSection,
    getConfiguredPromptForSection,
    getConfiguredGridForSection,
    getSuitableGridRange,
    renderImageQuestion,
  };

})();
