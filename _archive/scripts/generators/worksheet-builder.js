/**
 * scripts/generators/worksheet-builder.js
 *
 * Builds the full worksheet HTML string from a pool of question descriptors.
 * Handles all 4 sections and their variations.
 *
 * Exports: window.WORKSHEET_BUILDER
 *   buildWorksheetHTML(questions, globalOptions) → HTML string
 *   doesFitA4(html) → boolean
 */
(function () {

  const WORKSHEET_LOGO_SRC = 'https://www.geniusbees.com/assets/icons/gb-logo.svg';

  /** Page accent colours — 10 themes for 50 pages (cycle every 5 pages) */
  const PAGE_ACCENTS = [
    { bg: '#faf5ff', border: '#c084fc' },  // purple
    { bg: '#eff6ff', border: '#60a5fa' },  // blue
    { bg: '#fffbeb', border: '#fbbf24' },  // amber
    { bg: '#f0fdf4', border: '#34d399' },  // green
    { bg: '#fdf2f8', border: '#f472b6' },  // pink
    { bg: '#f0fdfa', border: '#2dd4bf' },  // teal
    { bg: '#fef2f2', border: '#f87171' },  // red
    { bg: '#eef2ff', border: '#818cf8' },  // indigo
    { bg: '#f7fee7', border: '#a3e635' },  // lime
    { bg: '#ecfeff', border: '#38bdf8' },  // sky
  ];

  /* ── Helpers ───────────────────────────────────────────────── */

  function escapeHtml(val) {
    return String(val || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  /* ── Marks box (top-right corner) ──────────────────────────── */

  function buildMarksBox() {
    return `
      <div class="ws-marks-box">
        <div class="ws-marks-label">Marks</div>
        <div class="ws-marks-value">
          <span class="ws-marks-dots"></span>
          <span class="ws-marks-slash">/</span>
          <span class="ws-marks-dots"></span>
        </div>
      </div>
    `;
  }

  /* ── Header fields ─────────────────────────────────────────── */

  function buildHeaderFields(opts) {
    const { showName, showDate } = opts;
    return `
      ${showName ? `
        <div class="ws-fill-line"><span>First Name:</span><span class="dots"></span></div>
        <div class="ws-fill-line"><span>Last Name:</span><span class="dots"></span></div>
      ` : ''}
      <div class="ws-fill-line"><span>GeniusBees ID:</span><span class="dots"></span></div>
      ${showDate ? `<div class="ws-fill-line"><span>Date:</span><span class="dots"></span></div>` : ''}
    `;
  }

  /* ── Worksheet page header bar ─────────────────────────────── */

  function buildHeaderBar(opts) {
    const { levelLabel, activityTitle, fields } = opts;
    return `
      <div class="ws-header-bar">
        <div class="ws-brand-column">
          <div class="ws-brand-logo-wrap">
            <img class="ws-brand-logo" src="${WORKSHEET_LOGO_SRC}" alt="GeniusBees"/>
          </div>
        </div>
        <div class="ws-center-column">
          <div class="ws-main-title">WORK<span class="muted">SHEET</span></div>
          <div class="ws-meta-line ws-level-line">Level: ${escapeHtml(levelLabel)}</div>
          <div class="ws-meta-line">Activity: ${escapeHtml(activityTitle)}</div>
        </div>
        <div class="ws-right-column">
          ${fields}
          ${buildMarksBox()}
        </div>
      </div>
    `;
  }

  /* ── Footer ────────────────────────────────────────────────── */

  function buildFooter(pageNum, showKidSticker) {
    const sticker = showKidSticker ? `
      <span class="ws-kid-sticker" aria-hidden="true">
        <svg class="ws-kid-svg" viewBox="0 0 120 36" xmlns="http://www.w3.org/2000/svg" role="img" aria-label="Fun decoration">
          <circle cx="16" cy="18" r="10" fill="#60a5fa"/>
          <circle cx="36" cy="18" r="10" fill="#34d399"/>
          <circle cx="56" cy="18" r="10" fill="#fbbf24"/>
          <circle cx="76" cy="18" r="10" fill="#f472b6"/>
          <circle cx="96" cy="18" r="10" fill="#c084fc"/>
          <circle cx="106" cy="9" r="5" fill="#fde68a"/>
          <circle cx="104" cy="8" r="1" fill="#111827"/>
          <circle cx="108" cy="8" r="1" fill="#111827"/>
          <path d="M103 10.5 Q106 13 109 10.5" stroke="#111827" stroke-width="1.4" fill="none" stroke-linecap="round"/>
        </svg>
      </span>` : '';

    return `
      <div class="ws-footer">
        <span class="ws-footer-left">${sticker}</span>
        <span>Page: ${pageNum}</span>
      </div>
      <div class="ws-bottom-contact">
        <div class="ws-bottom-contact-line"><span class="u">www.geniusbees.com</span></div>
        <div class="ws-bottom-contact-line"><span class="u">info@geniusbees.com</span></div>
        <div class="ws-bottom-contact-line">Copyright &copy; 2025 <span class="u">GeniusBees</span> Inc. All rights reserved</div>
      </div>
    `;
  }

  /* ── Per-question HTML renderers by section ────────────────── */

  function renderQuestionHTML(descriptor, qIdx) {
    const { section, variationId, data } = descriptor;

    switch (section) {

      case '3A3I': {
        if (window.WORKSHEET_3A3I?.renderQuestion) {
          return window.WORKSHEET_3A3I.renderQuestion(data.nums, variationId, qIdx, data.blankCount);
        }
        return '';
      }

      case '2A8': {
        if (window.WORKSHEET_2A8?.renderSingleVariation && data.bases && data.bases.length) {
          const items = data.bases.map((base, idx) => 
            window.WORKSHEET_2A8.renderSingleVariation(base, data.addend, variationId, idx)
          );
          return `<div class="equation-grid">\n${items.join('')}\n</div>`;
        }
        return '';
      }

      case '4A121': {
        if (window.WORKSHEET_4A121?.renderImageQuestion) {
          return window.WORKSHEET_4A121.renderImageQuestion(
            data.imageUrl, data.imageCount, data.imagePerRow, variationId
          );
        }
        return '';
      }

      case '4A': {
        if (window.WORKSHEET_4A?.buildWorksheetData) {
          return window.WORKSHEET_4A.buildWorksheetData({
            traceNumber:  data.traceNumber,
            imageValues:  data.imageValues || [],
          }).questionsHtml || '';
        }
        return '';
      }

      case 'UPLOADED': {
        return `<div class="ws-uploaded-wrapper" style="flex: 1; display: flex; flex-direction: column; width: 100%; min-height: 100%; position: relative;">
          ${data.rawHtml || ''}
        </div>`;
      }

      default:
        return '';
    }
  }

  /* ── Layout class selector ─────────────────────────────────── */

  function getLayoutClass(section, variationId, qCount) {
    if (section === '2A8') return 'type-e-layout';
    if (section === '4A121') return 'type-f-layout';
    if (section === '4A') return 'type-4a-layout';
    if (qCount === 1) return 'one-question-layout';
    if (qCount === 2) return 'two-question-layout';
    return '';
  }

  function getQuestionsWrapClass(section) {
    if (section === '2A8') return 'ws-questions-e';
    if (section === '4A121') return 'ws-questions-f';
    return 'ws-questions-default';
  }

  /* ── buildWorksheetHTML ────────────────────────────────────── */

  function buildWorksheetHTML(questions, globalOptions) {
    if (!questions || !questions.length) return '';

    const opts = globalOptions || {};
    const {
      showName   = true,
      showDate   = true,
      levelLabel = '',
      title      = 'Write the numbers',
      section    = '3A3I',
      variationId = 'A',
      poolMode   = false,
      globalImageUrl = '',
      activityText = '',
    } = opts;

    const fields      = buildHeaderFields({ showName, showDate });
    const activityTitle = activityText || title;
    const showSticker = section === '3A3I';

    const decorationHtml = globalImageUrl && section !== '4A121' && section !== '4A' ? 
      `<div class="ws-global-decoration" style="text-align:center; padding-bottom:15px; margin-top:-10px;">
         <img src="${escapeHtml(globalImageUrl)}" style="max-height:140px; max-width:90%; object-fit:contain; border-radius:8px;" alt=""/>
       </div>` : '';

    if (poolMode) {
      // Pool mode: render 2 questions per page, 50 pages total (unless UPLOADED)
      const pages = [];
      const isUploaded = section === 'UPLOADED';
      const step = isUploaded ? 1 : 2;

      for (let i = 0; i < questions.length; i += step) {
        const pageIdx = Math.floor(i / step);
        const pageQuestions = questions.slice(i, i + step);
        const qCount = pageQuestions.length;

        // Page accent colour
        const accent = PAGE_ACCENTS[pageIdx % PAGE_ACCENTS.length];
        const accentStyle = `border-left: 5px solid ${accent.border}; background: linear-gradient(135deg, ${accent.bg} 0%, white 40%)`;

        const qHtmlArr = pageQuestions.map((q, qi) => renderQuestionHTML(q, qi));
        const layoutClass = getLayoutClass(pageQuestions[0].section, pageQuestions[0].variationId, qCount);
        const wrapClass   = getQuestionsWrapClass(pageQuestions[0].section);
        const headerBar   = buildHeaderBar({ levelLabel, activityTitle, fields });
        const footer      = buildFooter(pageIdx + 1, showSticker);

        let itemPrompt = pageQuestions[0].data?.prompt;
        if (!itemPrompt) {
          itemPrompt = pageQuestions[0].section === '4A121' 
            ? (window.WORKSHEET_4A121?.getConfiguredPromptForSection?.('4A121') || title)
            : `${activityTitle}.`;
        }

        pages.push(`
          <div class="ws-page ${layoutClass}" style="${accentStyle}" id="wsPage-${pageIdx + 1}">
            ${headerBar}
            <div class="ws-instruction">${escapeHtml(itemPrompt)}</div>
            ${decorationHtml}
            <div class="ws-questions ${wrapClass}">${qHtmlArr.join('')}</div>
            ${footer}
          </div>
        `);
      }
      return pages.join('\n');
    }

    // Standard mode: render all questions on one page
    const questionsHTML = questions.map((q, qIdx) => renderQuestionHTML(q, qIdx)).join('');

    const layoutClass = getLayoutClass(section, variationId, questions.length);
    const wrapClass   = getQuestionsWrapClass(section);

    // Special handling for 4A section
    if (section === '4A') {
      const traceData = window.WORKSHEET_4A?.buildWorksheetData({
        traceNumber: questions[0]?.data?.traceNumber || 1,
        imageValues: questions[0]?.data?.imageValues || [],
      });
      if (!traceData) return '';

      const tracingCode  = traceData.worksheetCode || '';
      const tracingTitle = traceData.activityName || 'Number Tracing';
      const tracingInstr = traceData.instructionText || 'Trace the number.';
      const tracingHint  = (traceData.hintText || '').trim();

      return `
        <div class="ws-page type-4a-layout" id="wsPage">
          <div class="ws-header-bar">
            <div class="ws-brand-column">
              <div class="ws-brand-logo-wrap">
                <img class="ws-brand-logo" src="${WORKSHEET_LOGO_SRC}" alt="GeniusBees"/>
              </div>
            </div>
            <div class="ws-center-column">
              <div class="ws-main-title">WORK<span class="muted">SHEET</span></div>
              <div class="ws-meta-line ws-level-line">English Level:${escapeHtml(tracingCode)}</div>
              <div class="ws-meta-line">Activity: ${escapeHtml(tracingTitle)}</div>
            </div>
            <div class="ws-right-column">
              ${fields}
              ${buildMarksBox()}
            </div>
          </div>
          <div class="ws-instruction">${escapeHtml(tracingInstr)}</div>
          ${tracingHint ? `<div class="trace4a-global-hint">Hint: ${escapeHtml(tracingHint)}</div>` : ''}
          <div class="ws-questions ws-questions-default">
            ${traceData.questionsHtml || ''}
          </div>
          ${buildFooter(1, false)}
        </div>
      `;
    }

    const headerBar = buildHeaderBar({ levelLabel, activityTitle, fields });
    const footer    = buildFooter(1, showSticker);
    
    let defaultPrompt = section === '4A121'
      ? (window.WORKSHEET_4A121?.getConfiguredPromptForSection?.('4A121') || title)
      : `${activityTitle}.`;
      
    const prompt = questions[0]?.data?.prompt || defaultPrompt;

    return `
      <div class="ws-page ${layoutClass}" id="wsPage">
        ${headerBar}
        <div class="ws-instruction">${escapeHtml(prompt)}</div>
        ${decorationHtml}
        <div class="ws-questions ${wrapClass}">${questionsHTML}</div>
        ${footer}
      </div>
    `;
  }

  /* ── A4-fit resolver ───────────────────────────────────────── */

  function doesFitA4(html) {
    const probe = document.createElement('div');
    probe.style.cssText = 'position:fixed;left:-100000px;top:0;visibility:hidden;pointer-events:none;width:210mm;';
    probe.innerHTML = html;
    document.body.appendChild(probe);

    const page = probe.querySelector('.ws-page');
    let fits = true;
    if (page) {
      page.style.height    = '297mm';
      page.style.minHeight = '297mm';
      page.style.maxHeight = '297mm';
      page.style.overflow  = 'hidden';
      fits = page.scrollHeight <= page.clientHeight + 1;
    }

    probe.remove();
    return fits;
  }

  /* ── Exports ───────────────────────────────────────────────── */

  window.WORKSHEET_BUILDER = {
    buildWorksheetHTML,
    doesFitA4,
  };

})();
