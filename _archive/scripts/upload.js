/**
 * scripts/upload.js
 *
 * Worksheet Upload Platform logic.
 * Handles login, multi-member slots, per-member variations,
 * code editing, and preview with shared header/footer.
 */
(function () {

  /* ── Constants ─────────────────────────────────────────────────── */

  const CREDENTIALS = { username: 'admin1', password: '1234' };
  const TOTAL_SLOTS = 5;
  const LOGO_URL = 'https://www.geniusbees.com/assets/icons/gb-logo.svg';
  const STORAGE_KEY = 'ws_upload_data';

  const CSS_FILES = [
    'styles/base.css',
    'styles/worksheet.css',
    'styles/components/num-box.css',
    'styles/components/equation.css',
    'styles/components/image-question.css',
    'styles/components/trace4a.css',
  ];

  /* ── State ─────────────────────────────────────────────────────── */

  let activeSlot = -1;
  let activeVariation = 0;
  let debounceTimer = null;

  let members = [];
  let config = { 
    level: '3A3I', 
    activity: 'Write the numbers',
    varCount: 8,
    activityCode: '',
    grade: '',
    essentialDetails: ''
  };

  /* ── DOM helpers ───────────────────────────────────────────────── */

  function $(id) { return document.getElementById(id); }

  /* ── Data helpers ──────────────────────────────────────────────── */

  function makeEmptyVariations() {
    return Array.from({ length: config.varCount }, () => ({ code: '', lastEdit: null }));
  }

  function ensureMemberVariations(member) {
    if (!Array.isArray(member.variations) || member.variations.length !== config.varCount) {
      const existing = Array.isArray(member.variations) ? member.variations : [];
      // Migrate old single-code format
      if (!existing.length && member.code) {
        existing.push({ code: member.code, lastEdit: member.lastEdit || null });
      }
      while (existing.length < config.varCount) {
        existing.push({ code: '', lastEdit: null });
      }
      member.variations = existing.slice(0, config.varCount);
    }
    return member;
  }

  function getFilledCount(member) {
    if (!member?.variations) return 0;
    return member.variations.filter(v => v.code && v.code.trim().length > 0).length;
  }

  function getActiveCode() {
    const member = members[activeSlot];
    if (!member?.variations?.[activeVariation]) return '';
    return member.variations[activeVariation].code || '';
  }

  /* ── Persistence ───────────────────────────────────────────────── */

  function loadData() {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (raw) {
        const data = JSON.parse(raw);
        if (Array.isArray(data.members)) members = data.members;
        if (data.config) config = data.config;
      }
    } catch (e) { /* ignore */ }

    // Ensure we have the right number of slots with variations
    while (members.length < TOTAL_SLOTS) {
      members.push({
        name: 'Member ' + (members.length + 1),
        variations: makeEmptyVariations(),
      });
    }
    members = members.slice(0, TOTAL_SLOTS);
    members.forEach(m => ensureMemberVariations(m));
  }

  function saveData() {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify({ members, config }));
    } catch (e) { /* storage full */ }
  }

  function showDashboard() {
    const app = $('appDashboard');
    if (app) app.classList.add('visible');
  }

  /* ── Member Slots ──────────────────────────────────────────────── */

  function renderSlots() {
    const container = $('slotList');
    if (!container) return;

    container.innerHTML = members.map((m, i) => {
      const filled = getFilledCount(m);
      const isActive = i === activeSlot;
      const hasAny = filled > 0;
      const classes = ['slot-card'];
      if (isActive) classes.push('active');
      if (hasAny) classes.push('has-code');

      let statusText = 'Empty';
      if (filled > 0) {
        statusText = `${filled} / ${config.varCount} variations`;
      }

      return `
        <div class="${classes.join(' ')}" data-slot="${i}" id="slotCard-${i}">
          <div class="slot-dot"></div>
          <div class="slot-info">
            <div class="slot-name">${escapeHtml(m.name)}</div>
            <div class="slot-status">${statusText}</div>
          </div>
        </div>
      `;
    }).join('');

    // Bind click events
    container.querySelectorAll('.slot-card').forEach(card => {
      card.addEventListener('click', () => {
        const idx = parseInt(card.dataset.slot, 10);
        selectSlot(idx);
      });
    });
  }

  function selectSlot(index) {
    if (index < 0 || index >= TOTAL_SLOTS) return;

    // Save current variation code before switching
    saveCurrentCode();

    activeSlot = index;
    activeVariation = 0; // Reset to first variation
    renderSlots();
    showWorkspace();
  }

  function saveCurrentCode() {
    if (activeSlot >= 0) {
      const textarea = $('codeTextarea');
      if (textarea && members[activeSlot]?.variations?.[activeVariation]) {
        members[activeSlot].variations[activeVariation].code = textarea.value;
      }
      saveData();
    }
  }

  function showWorkspace() {
    const empty = $('mainEmpty');
    const workspace = $('workspace');
    if (empty) empty.style.display = 'none';
    if (workspace) workspace.classList.add('visible');

    const member = members[activeSlot];
    if (!member) return;

    // Update toolbar
    const label = $('wsToolbarLabel');
    if (label) label.textContent = member.name;

    // Render variation tabs
    renderVariationTabs();

    // Load the active variation's code into the textarea
    const textarea = $('codeTextarea');
    if (textarea) textarea.value = getActiveCode();

    // Update status & preview
    updateStatus();
    updatePreview();
  }

  /* ── Variation Tabs ────────────────────────────────────────────── */

  function renderVariationTabs() {
    const container = $('varTabs');
    const countEl = $('varBarCount');
    const member = members[activeSlot];
    if (!container || !member) return;

    container.innerHTML = member.variations.map((v, i) => {
      const hasCd = v.code && v.code.trim().length > 0;
      const isActive = i === activeVariation;
      const classes = ['var-tab'];
      if (isActive) classes.push('active');
      if (hasCd) classes.push('filled');

      return `<button class="${classes.join(' ')}" data-var="${i}">
        <span class="var-tab-dot"></span>V${i + 1}
      </button>`;
    }).join('');

    // Bind click events
    container.querySelectorAll('.var-tab').forEach(tab => {
      tab.addEventListener('click', () => {
        const idx = parseInt(tab.dataset.var, 10);
        selectVariation(idx);
      });
    });

    // Update count
    const filled = getFilledCount(member);
    if (countEl) countEl.textContent = `${filled} / ${config.varCount} filled`;
  }

  function selectVariation(index) {
    if (index < 0 || index >= config.varCount) return;

    // Save current variation code
    saveCurrentCode();

    activeVariation = index;

    // Update tabs visual state
    renderVariationTabs();

    // Load this variation's code
    const textarea = $('codeTextarea');
    if (textarea) textarea.value = getActiveCode();

    // Update status & preview
    updateStatus();
    updatePreview();
  }

  /* ── Status ────────────────────────────────────────────────────── */

  function updateStatus() {
    const statusEl = $('wsToolbarStatus');
    if (!statusEl) return;

    const code = getActiveCode();
    const hasCode = code.trim().length > 0;

    if (hasCode) {
      statusEl.className = 'ws-toolbar-status live';
      statusEl.innerHTML = `<span class="ws-toolbar-dot"></span> V${activeVariation + 1} — Live Preview`;
    } else {
      statusEl.className = 'ws-toolbar-status empty';
      statusEl.innerHTML = `<span class="ws-toolbar-dot"></span> V${activeVariation + 1} — No code`;
    }
  }

  /* ── Preview ───────────────────────────────────────────────────── */

  function buildPreviewHTML(contentCode) {
    const levelLabel = $('cfgLevel')?.value || config.level || '3A3I';
    const activityTitle = $('cfgActivity')?.value || config.activity || 'Write the numbers';

    const cssLinks = CSS_FILES.map(f => `<link rel="stylesheet" href="${f}"/>`).join('\n');

    const headerBar = `
      <div class="ws-header-bar">
        <div class="ws-brand-column">
          <div class="ws-brand-logo-wrap">
            <img class="ws-brand-logo" src="${LOGO_URL}" alt="GeniusBees"/>
          </div>
        </div>
        <div class="ws-center-column">
          <div class="ws-main-title">WORK<span class="muted">SHEET</span></div>
          <div class="ws-meta-line ws-level-line">Level: ${escapeHtml(levelLabel)}</div>
          <div class="ws-meta-line">Activity: ${escapeHtml(activityTitle)}</div>
        </div>
        <div class="ws-right-column">
          <div class="ws-fill-line"><span>First Name:</span><span class="dots"></span></div>
          <div class="ws-fill-line"><span>Last Name:</span><span class="dots"></span></div>
          <div class="ws-fill-line"><span>GeniusBees ID:</span><span class="dots"></span></div>
          <div class="ws-fill-line"><span>Date:</span><span class="dots"></span></div>
          <div class="ws-marks-box">
            <div class="ws-marks-label">Marks</div>
            <div class="ws-marks-value">
              <span class="ws-marks-dots"></span>
              <span class="ws-marks-slash">/</span>
              <span class="ws-marks-dots"></span>
            </div>
          </div>
        </div>
      </div>`;

    const footer = `
      <div class="ws-footer">
        <span class="ws-footer-left"></span>
        <span>Page: 1</span>
      </div>
      <div class="ws-bottom-contact">
        <div class="ws-bottom-contact-line"><span class="u">www.geniusbees.com</span></div>
        <div class="ws-bottom-contact-line"><span class="u">info@geniusbees.com</span></div>
        <div class="ws-bottom-contact-line">Copyright &copy; 2025 <span class="u">GeniusBees</span> Inc. All rights reserved</div>
      </div>`;

    const instruction = `<div class="ws-instruction">${escapeHtml(activityTitle)}.</div>`;

    return `<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  ${cssLinks}
  <style>
    body {
      margin: 0;
      padding: 20px;
      background: #f0f4ff;
      display: flex;
      justify-content: center;
    }
    .ws-page {
      box-shadow: 0 2px 20px rgba(0,0,0,0.08);
      border-radius: 4px;
      width: 210mm;
      min-height: 297mm;
    }
  </style>
</head>
<body>
  <div class="ws-page">
    ${headerBar}
    ${instruction}
    <div class="ws-questions ws-questions-default">
      ${contentCode}
    </div>
    ${footer}
  </div>
</body>
</html>`;
  }

  function buildEmptyPreviewHTML() {
    return `<!DOCTYPE html>
<html><head>
<style>
  body {
    margin: 0; display: flex; align-items: center; justify-content: center;
    height: 100vh; font-family: 'Poppins', sans-serif; color: #9ca3af;
    background: #f9fafb;
  }
  .wrap { text-align: center; }
  .icon {
    width: 56px; height: 56px; margin: 0 auto 12px; border-radius: 14px;
    background: #f3f4f6; display: flex; align-items: center; justify-content: center;
  }
  svg { width: 28px; height: 28px; color: #d1d5db; }
  h3 { font-size: 15px; font-weight: 700; margin: 0 0 4px; color: #6b7280; }
  p { font-size: 12px; margin: 0; }
</style>
</head><body>
<div class="wrap">
  <div class="icon">
    <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="1.5">
      <path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8z"/>
      <polyline points="14 2 14 8 20 8"/>
    </svg>
  </div>
  <h3>No Content Yet</h3>
  <p>Paste or upload worksheet content code to see a preview</p>
</div>
</body></html>`;
  }

  function updatePreview() {
    const code = getActiveCode();
    const iframe = $('previewIframe');
    if (!iframe) return;

    if (!code.trim()) {
      iframe.srcdoc = buildEmptyPreviewHTML();
      return;
    }

    iframe.srcdoc = buildPreviewHTML(code);
  }

  /* ── Code editing ──────────────────────────────────────────────── */

  function onCodeInput(e) {
    if (activeSlot < 0) return;
    const member = members[activeSlot];
    if (!member?.variations?.[activeVariation]) return;

    member.variations[activeVariation].code = e.target.value;
    member.variations[activeVariation].lastEdit = new Date().toISOString();

    clearTimeout(debounceTimer);
    debounceTimer = setTimeout(() => {
      updatePreview();
      updateStatus();
      renderVariationTabs();
      renderSlots();
      saveData();
    }, 400);
  }

  function clearCode() {
    if (activeSlot < 0) return;
    const member = members[activeSlot];
    if (!member?.variations?.[activeVariation]) return;

    member.variations[activeVariation].code = '';
    member.variations[activeVariation].lastEdit = null;
    const textarea = $('codeTextarea');
    if (textarea) textarea.value = '';
    updatePreview();
    updateStatus();
    renderVariationTabs();
    renderSlots();
    saveData();
  }

  /* ── File upload ───────────────────────────────────────────────── */

  function handleFileUpload(file) {
    if (!file || activeSlot < 0) return;
    const member = members[activeSlot];
    if (!member?.variations?.[activeVariation]) return;

    const reader = new FileReader();
    reader.onload = function (e) {
      const content = e.target.result;
      member.variations[activeVariation].code = content;
      member.variations[activeVariation].lastEdit = new Date().toISOString();
      const textarea = $('codeTextarea');
      if (textarea) textarea.value = content;
      updatePreview();
      updateStatus();
      renderVariationTabs();
      renderSlots();
      saveData();
    };
    reader.readAsText(file);
  }

  function setupDropzone() {
    const dropArea = $('dropArea');
    const fileInput = $('fileInput');
    if (!dropArea || !fileInput) return;

    dropArea.addEventListener('click', () => fileInput.click());
    dropArea.addEventListener('dragover', e => { e.preventDefault(); dropArea.classList.add('drag-over'); });
    dropArea.addEventListener('dragleave', () => dropArea.classList.remove('drag-over'));
    dropArea.addEventListener('drop', e => {
      e.preventDefault();
      dropArea.classList.remove('drag-over');
      if (e.dataTransfer.files[0]) handleFileUpload(e.dataTransfer.files[0]);
    });

    fileInput.addEventListener('change', e => {
      if (e.target.files[0]) handleFileUpload(e.target.files[0]);
      e.target.value = '';
    });
  }

  /* ── Fullscreen preview ────────────────────────────────────────── */

  function toggleFullscreen() {
    const overlay = $('fsOverlay');
    if (!overlay) return;

    const isOpen = overlay.classList.contains('visible');
    if (isOpen) {
      overlay.classList.remove('visible');
      return;
    }

    const code = getActiveCode();
    const fsIframe = $('fsIframe');
    if (!fsIframe) return;

    fsIframe.srcdoc = code.trim() ? buildPreviewHTML(code) : buildEmptyPreviewHTML();
    overlay.classList.add('visible');
  }

  /* ── Config change ─────────────────────────────────────────────── */

  function onConfigChange() {
    config.level = $('cfgLevel')?.value || '3A3I';
    config.activity = $('cfgActivity')?.value || 'Write the numbers';
    config.varCount = Math.max(1, Math.min(50, parseInt($('cfgVarCount')?.value, 10) || 8));
    config.activityCode = $('cfgActivityCode')?.value || '';
    config.grade = $('cfgGrade')?.value || '';
    config.essentialDetails = $('cfgEssentialDetails')?.value || '';

    // If varCount changed, we must ensure all members have the right number of variations
    members.forEach(m => ensureMemberVariations(m));
    
    // If active variation is now out of bounds, select the last valid one
    if (activeVariation >= config.varCount) {
      activeVariation = config.varCount - 1;
    }

    saveData();
    if (activeSlot >= 0) {
      renderVariationTabs();
      updatePreview();
      renderSlots();
    }
  }

  /* ── Helpers ───────────────────────────────────────────────────── */

  function escapeHtml(val) {
    return String(val || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;');
  }

  /* ── Initialize ────────────────────────────────────────────────── */

  function init() {
    loadData();

    // Sync config inputs from loaded data
    const cfgLevel = $('cfgLevel');
    const cfgActivity = $('cfgActivity');
    const cfgVarCount = $('cfgVarCount');
    const cfgActivityCode = $('cfgActivityCode');
    const cfgGrade = $('cfgGrade');
    const cfgEssentialDetails = $('cfgEssentialDetails');
    
    if (cfgLevel) cfgLevel.value = config.level;
    if (cfgActivity) cfgActivity.value = config.activity;
    if (cfgVarCount) cfgVarCount.value = config.varCount;
    if (cfgActivityCode) cfgActivityCode.value = config.activityCode;
    if (cfgGrade) cfgGrade.value = config.grade;
    if (cfgEssentialDetails) cfgEssentialDetails.value = config.essentialDetails;

    // Always show dashboard since login is removed
    showDashboard();

    // Render member slots
    renderSlots();

    // Code textarea
    $('codeTextarea')?.addEventListener('input', onCodeInput);
    $('codeTextarea')?.addEventListener('keydown', e => {
      if (e.key === 'Tab') {
        e.preventDefault();
        const ta = e.target;
        const start = ta.selectionStart;
        const end = ta.selectionEnd;
        ta.value = ta.value.substring(0, start) + '  ' + ta.value.substring(end);
        ta.selectionStart = ta.selectionEnd = start + 2;
        onCodeInput({ target: ta });
      }
    });

    // File upload
    setupDropzone();

    // Toolbar buttons
    $('toolClear')?.addEventListener('click', clearCode);
    $('toolFullscreen')?.addEventListener('click', toggleFullscreen);
    $('fsClose')?.addEventListener('click', toggleFullscreen);

    $('toolPublish')?.addEventListener('click', () => {
      if (activeSlot < 0) return;
      const member = members[activeSlot];
      if (!member) return;

      // Extract filled variations
      const filledVariations = member.variations
        .map(v => v.code?.trim())
        .filter(c => c && c.length > 0);

      if (filledVariations.length === 0) {
        alert('No variations to publish! Please add code to at least one variation.');
        return;
      }

      try {
        const published = JSON.parse(localStorage.getItem('ws_published_worksheets') || '{}');
        published[member.name] = {
          level: config.level,
          activity: config.activity,
          activityCode: config.activityCode,
          grade: config.grade,
          essentialDetails: config.essentialDetails,
          variations: filledVariations
        };
        localStorage.setItem('ws_published_worksheets', JSON.stringify(published));
        alert(`Successfully published ${filledVariations.length} variations for ${member.name}!`);
        window.location.href = 'index.html?section=UPLOADED';
      } catch (e) {
        alert('Failed to publish. Storage might be full.');
      }
    });

    // Config changes
    const configInputs = ['cfgLevel', 'cfgActivity', 'cfgVarCount', 'cfgActivityCode', 'cfgGrade', 'cfgEssentialDetails'];
    configInputs.forEach(id => {
      $(id)?.addEventListener('change', onConfigChange);
      $(id)?.addEventListener('input', () => {
        clearTimeout(debounceTimer);
        debounceTimer = setTimeout(onConfigChange, 500);
      });
    });

    // Escape to close fullscreen
    document.addEventListener('keydown', e => {
      if (e.key === 'Escape') {
        const overlay = $('fsOverlay');
        if (overlay?.classList.contains('visible')) toggleFullscreen();
      }
    });
  }

  document.addEventListener('DOMContentLoaded', init);

})();
