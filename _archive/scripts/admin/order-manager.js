/**
 * scripts/admin/order-manager.js
 *
 * Admin drag-and-drop question order panel.
 *
 * Renders a list of mini-cards (one per question in the pool) inside a
 * given container element. The admin can drag to reorder them. The current
 * order is persisted to localStorage and returned via getOrderedQuestions().
 *
 * Exports: window.ORDER_MANAGER
 *   renderPanel(questions, container)  — draws the sortable card list
 *   getOrderedQuestions()              — returns questions in current admin order
 *   shufflePool()                      — randomises current order
 *   resetOrder()                       — restores original pool order
 *   saveOrder(order)                   — persists to localStorage
 *   loadOrder()                        — restores from localStorage
 */
(function () {

  const LS_KEY = 'ws_question_order';
  let _pool    = [];   // Original pool (as generated)
  let _order   = [];   // Current admin-controlled order (array of pool indices)

  /* ── Persistence ───────────────────────────────────────────── */

  function saveOrder(order) {
    try {
      localStorage.setItem(LS_KEY, JSON.stringify(order));
    } catch (_) { /* storage unavailable */ }
  }

  function loadOrder() {
    try {
      const raw = localStorage.getItem(LS_KEY);
      if (raw) return JSON.parse(raw);
    } catch (_) { /* corrupt data */ }
    return null;
  }

  /* ── Getters ───────────────────────────────────────────────── */

  function getOrderedQuestions() {
    if (!_order.length) return _pool;
    return _order.map(i => _pool[i]).filter(Boolean);
  }

  /* ── Panel render ──────────────────────────────────────────── */

  /**
   * renderPanel(questions, container)
   *
   * @param {object[]} questions  - QuestionDescriptor[] from QUESTION_POOL.generate()
   * @param {Element}  container  - DOM element to render into (e.g. #orderPoolGrid)
   */
  function renderPanel(questions, container) {
    if (!container) return;
    _pool  = questions;
    _order = _pool.map((_, i) => i);   // Default: original order

    // Restore persisted order if valid
    const saved = loadOrder();
    if (
      Array.isArray(saved) &&
      saved.length === _pool.length &&
      saved.every(i => Number.isInteger(i) && i >= 0 && i < _pool.length)
    ) {
      _order = saved;
    }

    _renderCards(container);
  }

  function _renderCards(container) {
    container.innerHTML = '';

    _order.forEach((poolIdx, position) => {
      const q  = _pool[poolIdx];
      if (!q) return;

      const card = document.createElement('div');
      card.className      = 'order-q-card';
      card.draggable      = true;
      card.dataset.pos    = position;
      card.dataset.idx    = poolIdx;

      let editorHtml = '';
      if (q.section === '3A3I') {
        editorHtml = `
          <input type="text" class="admin-edit-input js-edit-nums" value="${(q.data.nums || []).join(',')}" title="Numbers (comma separated)" style="width:140px; padding:2px 4px; border:1px solid #d1d5db; border-radius:4px; font-size:12px;"/>
        `;
      } else if (q.section === '2A8') {
        editorHtml = `
          <input type="text" class="admin-edit-input js-edit-bases" value="${(q.data.bases || []).join(',')}" title="Base Numbers" style="width:120px; padding:2px 4px; border:1px solid #d1d5db; border-radius:4px; font-size:12px;"/>
          ${q.variationId === 'E10' ? '' : `<input type="number" class="admin-edit-input js-edit-addend" value="${q.data.addend}" title="Addend" style="width:50px; padding:2px 4px; border:1px solid #d1d5db; border-radius:4px; font-size:12px;"/>`}
        `;
      } else if (q.section === '4A121') {
        editorHtml = `
          <input type="text" class="admin-edit-input js-edit-prompt" value="${_escapeHtml(q.data.prompt || '')}" title="Prompt Title" placeholder="Instruction" style="width:120px; padding:2px 4px; border:1px solid #d1d5db; border-radius:4px; font-size:12px;"/>
          <input type="url" class="admin-edit-input js-edit-image-url" value="${_escapeHtml(q.data.imageUrl || '')}" title="Image URL" placeholder="URL" style="width:100px; padding:2px 4px; border:1px solid #d1d5db; border-radius:4px; font-size:12px;"/>
          <input type="number" class="admin-edit-input js-edit-count" value="${q.data.imageCount}" title="Image Count" style="width:40px; padding:2px 4px; border:1px solid #d1d5db; border-radius:4px; font-size:12px;"/>
        `;
      } else {
        editorHtml = `<span class="order-q-label">${_escapeHtml(q.label)}</span>`;
      }

      card.innerHTML = `
        <span class="order-q-handle" aria-hidden="true" title="Drag to reorder">⠿</span>
        <span class="order-q-index" style="min-width: 24px;">${position + 1}</span>
        <div class="order-q-edit-fields" style="flex:1; display:flex; gap:6px; align-items:center;">${editorHtml}</div>
        <span class="order-q-type-badge">${_escapeHtml(q.variationId)}</span>
      `;

      // Attach edit listeners
      const numInput = card.querySelector('.js-edit-nums');
      if (numInput) {
        numInput.addEventListener('input', (e) => {
          const val = e.target.value;
          q.data.nums = val.split(',').map(n => parseInt(n.trim(), 10)).filter(Number.isFinite);
          if (window.renderCurrentPool) window.renderCurrentPool();
        });
      }

      const basesInput = card.querySelector('.js-edit-bases');
      if (basesInput) {
        basesInput.addEventListener('input', (e) => {
          const val = e.target.value;
          q.data.bases = val.split(',').map(n => parseInt(n.trim(), 10)).filter(Number.isFinite);
          if (window.renderCurrentPool) window.renderCurrentPool();
        });
      }

      const addendInput = card.querySelector('.js-edit-addend');
      if (addendInput) {
        addendInput.addEventListener('input', (e) => {
          q.data.addend = parseInt(e.target.value, 10) || 0;
          if (window.renderCurrentPool) window.renderCurrentPool();
        });
      }

      const promptInput = card.querySelector('.js-edit-prompt');
      if (promptInput) {
        promptInput.addEventListener('input', (e) => {
          q.data.prompt = e.target.value;
          if (window.renderCurrentPool) window.renderCurrentPool();
        });
      }

      const countInput = card.querySelector('.js-edit-count');
      if (countInput) {
        countInput.addEventListener('input', (e) => {
          q.data.imageCount = parseInt(e.target.value, 10) || 1;
          if (window.renderCurrentPool) window.renderCurrentPool();
        });
      }

      const imgUrlInput = card.querySelector('.js-edit-image-url');
      if (imgUrlInput) {
        imgUrlInput.addEventListener('input', (e) => {
          q.data.imageUrl = e.target.value;
          if (window.renderCurrentPool) window.renderCurrentPool();
        });
      }

      _attachDragHandlers(card, container);
      container.appendChild(card);
    });
  }

  /* ── Drag-and-drop ─────────────────────────────────────────── */

  let _dragSrcPos = null;

  function _attachDragHandlers(card, container) {
    card.addEventListener('dragstart', e => {
      _dragSrcPos = parseInt(card.dataset.pos, 10);
      card.classList.add('dragging');
      e.dataTransfer.effectAllowed = 'move';
    });

    card.addEventListener('dragend', () => {
      card.classList.remove('dragging');
      container.querySelectorAll('.order-q-card').forEach(c => c.classList.remove('drag-over'));
      _dragSrcPos = null;
    });

    card.addEventListener('dragover', e => {
      e.preventDefault();
      e.dataTransfer.dropEffect = 'move';
      container.querySelectorAll('.order-q-card').forEach(c => c.classList.remove('drag-over'));
      card.classList.add('drag-over');
    });

    card.addEventListener('drop', e => {
      e.preventDefault();
      const destPos = parseInt(card.dataset.pos, 10);
      if (_dragSrcPos === null || _dragSrcPos === destPos) return;

      // Reorder _order array
      const moved = _order.splice(_dragSrcPos, 1)[0];
      _order.splice(destPos, 0, moved);

      saveOrder(_order);
      _renderCards(container);   // re-render with new positions
    });
  }

  /* ── Admin actions ─────────────────────────────────────────── */

  function shufflePool(container) {
    // Fisher-Yates shuffle
    for (let i = _order.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [_order[i], _order[j]] = [_order[j], _order[i]];
    }
    saveOrder(_order);
    if (container) _renderCards(container);
  }

  function resetOrder(container) {
    _order = _pool.map((_, i) => i);
    saveOrder(_order);
    if (container) _renderCards(container);
  }

  /* ── Utility ───────────────────────────────────────────────── */

  function _escapeHtml(value) {
    return String(value || '')
      .replace(/&/g, '&amp;')
      .replace(/</g, '&lt;')
      .replace(/>/g, '&gt;');
  }

  /* ── Exports ───────────────────────────────────────────────── */

  window.ORDER_MANAGER = {
    renderPanel,
    getOrderedQuestions,
    shufflePool,
    resetOrder,
    saveOrder,
    loadOrder,
  };

})();
