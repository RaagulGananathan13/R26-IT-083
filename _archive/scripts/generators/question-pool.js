/**
 * scripts/generators/question-pool.js
 *
 * Generates a pool of unique question descriptors for any section + variation.
 * Each descriptor contains enough data to render one question without re-generating.
 *
 * The order-manager (admin/order-manager.js) can reorder the pool before rendering.
 *
 * Exports: window.QUESTION_POOL
 *   generate(sectionId, variationId, options) → QuestionDescriptor[]  (up to 50)
 *   generateMulti(sectionId, options)         → QuestionDescriptor[]  (100 for pool-print)
 *   applyOrder(pool, orderedIndices)          → QuestionDescriptor[]
 */
(function () {

  const POOL_SIZE = 50;
  const POOL_SIZE_MULTI = 100; // 2 questions per page × 50 pages

  /** All 3A3I variation IDs (A–K) */
  const ALL_3A3I_VARS = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K'];

  /* ── Seeded randomness ─────────────────────────────────────── */

  /**
   * Simple LCG seeded random — gives reproducible pools when the
   * admin wants to regenerate the same set.
   */
  function seededRandom(seed) {
    let s = seed || Date.now();
    return function () {
      s = (s * 1664525 + 1013904223) & 0xffffffff;
      return (s >>> 0) / 0xffffffff;
    };
  }

  function shuffle(arr, rng) {
    const a = [...arr];
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(rng() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  /* ── 3A3I pool generator (single variation) ────────────────── */

  function generate3A3I(variationId, options, poolSize) {
    const size     = poolSize || POOL_SIZE;
    const start    = Math.max(0, parseInt(options.rangeStart, 10) || 31);
    const end      = Math.max(start + 1, parseInt(options.rangeEnd, 10) || 70);
    const perRow   = Math.max(2, Math.min(20, parseInt(options.perRow, 10) || 10));
    const targetBlankCnt = Math.max(1, parseInt(options.blankCount, 10) || 4);
    const rng      = seededRandom(options.seed);

    const allNums  = [];
    for (let i = start; i <= end; i++) allNums.push(i);

    const pool = [];

    // Slide a window of `perRow` numbers across the range, progressively moving forward.
    const maxDistinct = Math.max(1, allNums.length - perRow + 1);

    while (pool.length < size) {
      // Progress over offsets linearly as page increases.
      const progress = pool.length / (size - 1 || 1); // 0.0 to 1.0
      
      // Map progress directly to the available distinct windows.
      const rawOffset = Math.floor(progress * (maxDistinct - 1));

      const safeOffset = rawOffset % Math.max(1, allNums.length);
      const nums = [];
      for (let i = 0; i < perRow; i++) {
        nums.push(allNums[(safeOffset + i) % allNums.length]);
      }

      // Progressively increase blank count up to targetBlankCnt
      const currentBlankCnt = Math.max(1, Math.ceil(((pool.length + 1) / size) * targetBlankCnt));

      pool.push({
        id:          pool.length + 1,
        section:     '3A3I',
        variationId,
        label:       `Q${pool.length + 1}: ${nums[0]}–${nums[nums.length - 1]} (${variationId})`,
        data:        { nums, blankCount: currentBlankCnt },
      });
    }

    return pool.slice(0, size);
  }

  /* ── 3A3I multi-variation pool (for 50-page pool-print) ────── */

  /**
   * Generates 100 questions cycling through all A–K variations.
   * Every 2 consecutive questions form one page (page 1 = q0,q1; page 2 = q2,q3; etc.)
   * Each page pair uses the same variation. 50 pages ÷ 11 variations ≈ ~4-5 pages each.
   * Each page gets a different number range (sliding window).
   */
  function generate3A3IMultiVariation(options) {
    const totalPages = 50;
    const totalQs    = totalPages * 2; // 100 questions
    const start    = Math.max(0, parseInt(options.rangeStart, 10) || 31);
    const end      = Math.max(start + 1, parseInt(options.rangeEnd, 10) || 70);
    const perRow   = Math.max(2, Math.min(20, parseInt(options.perRow, 10) || 10));
    const targetBlankCnt = Math.max(1, parseInt(options.blankCount, 10) || 4);
    const rng      = seededRandom(options.seed);

    const totalRange = end - start + 1;
    const pool = [];

    for (let pageIdx = 0; pageIdx < totalPages; pageIdx++) {
      // Assign variation: cycle through A-K
      const varIdx    = pageIdx % ALL_3A3I_VARS.length;
      const variation = ALL_3A3I_VARS[varIdx];

      // Progressive range: slide the window forward across the range
      const pageProgress = pageIdx / (totalPages - 1 || 1); // 0.0 to 1.0
      const maxDistinct  = Math.max(1, totalRange - perRow + 1);
      const baseOffset   = Math.floor(pageProgress * (maxDistinct - 1));

      // Generate 2 questions for this page — slightly different offsets
      for (let qOnPage = 0; qOnPage < 2; qOnPage++) {
        const offset = (baseOffset + qOnPage * Math.max(1, Math.floor(perRow / 2))) % totalRange;
        const nums = [];
        for (let i = 0; i < perRow; i++) {
          nums.push(start + ((offset + i) % totalRange));
        }

        const currentBlankCnt = Math.max(1, Math.ceil(((pageIdx + 1) / totalPages) * targetBlankCnt));

        pool.push({
          id:          pool.length + 1,
          section:     '3A3I',
          variationId: variation,
          pageIndex:   pageIdx,  // which page this belongs to (0-based)
          label:       `P${pageIdx + 1}-Q${qOnPage + 1}: ${nums[0]}–${nums[nums.length - 1]} (${variation})`,
          data:        { nums, blankCount: currentBlankCnt },
        });
      }
    }

    return pool;
  }

  /* ── 2A8 pool generator ────────────────────────────────────── */

  function generate2A8(variationId, options) {
    const targetAddend = Math.max(0, Math.min(99, parseInt(options.eqAddend, 10) || 3));
    let seedNumbers   = options.seedNumbers || [];
    const rng         = seededRandom(options.seed);
    const pool        = [];

    // Setup progressive base pool
    if (!seedNumbers.length) {
      seedNumbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11];
    } else {
      seedNumbers = [...seedNumbers].sort((a, b) => a - b);
    }

    // Determine the progression of addends (e.g., if target=3, build up +1, +2, +3)
    const addendsToCover = [];
    if (targetAddend === 0) {
      addendsToCover.push(0);
    } else {
      for (let i = 1; i <= targetAddend; i++) addendsToCover.push(i);
    }

    // Pre-calculate which pages get which addends to ensure smooth chunks
    const pagesAssignments = [];
    const chunkCounts = new Array(addendsToCover.length).fill(0);
    for (let i = 0; i < POOL_SIZE; i++) {
        const idx = Math.min(addendsToCover.length - 1, Math.floor(i / (POOL_SIZE / addendsToCover.length)));
        pagesAssignments.push(idx);
        chunkCounts[idx]++;
    }

    const currentIndexInChunk = new Array(addendsToCover.length).fill(0);

    while (pool.length < POOL_SIZE) {
      const addendIndex = pagesAssignments[pool.length];
      const currentAddend = addendsToCover[addendIndex];
      
      const chunkTotal = chunkCounts[addendIndex];
      const chunkPos = currentIndexInChunk[addendIndex]++;
      const chunkProgress = chunkTotal > 1 ? (chunkPos / (chunkTotal - 1)) : 1.0;

      // Slide a window over the available bases so smaller ones appear first in the chunk
      const windowSize = Math.max(3, Math.floor(seedNumbers.length * 0.4));
      const maxStartIndex = Math.max(0, seedNumbers.length - windowSize);
      
      const startIndex = Math.floor(chunkProgress * maxStartIndex);
      const allowedBases = seedNumbers.slice(startIndex, startIndex + windowSize);

      const bases = [];
      for (let i = 0; i < 16; i++) {
        const base = allowedBases[Math.floor(rng() * allowedBases.length)];
        bases.push(base);
      }

      pool.push({
        id:          pool.length + 1,
        section:     '2A8',
        variationId,
        label:       `Page ${pool.length + 1}: +${currentAddend} (${variationId})`,
        data:        { bases, addend: currentAddend, variationId },
      });
    }

    return pool.slice(0, POOL_SIZE);
  }

  /* ── 4A121 pool generator ──────────────────────────────────── */

  function generate4A121(variationId, options) {
    const imageUrl    = options.imageUrl  || '';
    const promptTitle = options.title || 'How many are there?';
    const imagePerRow = Math.max(1, Math.min(8, parseInt(options.imagePerRow, 10) || 5));
    const maxCount    = imagePerRow * 5;
    const rng         = seededRandom(options.seed);
    const pool        = [];

    while (pool.length < POOL_SIZE) {
      // Progressively increase the maximum allowed count based on the page number
      const progress = pool.length / (POOL_SIZE - 1 || 1); // 0.0 to 1.0
      
      const minCap = Math.min(3, maxCount);
      const currentMaxCap = Math.floor(minCap + progress * (maxCount - minCap));

      // Choose a random count up to the progressively allowed maximum
      const count = Math.max(1, Math.min(currentMaxCap, Math.floor(rng() * currentMaxCap) + 1));

      pool.push({
        id:          pool.length + 1,
        section:     '4A121',
        variationId,
        label:       `Q${pool.length + 1}: ${count} images (${variationId})`,
        data:        { prompt: promptTitle, imageUrl, imageCount: count, imagePerRow, variationId },
      });
    }

    return pool.slice(0, POOL_SIZE);
  }

  /* ── 4A pool generator ─────────────────────────────────────── */

  function generate4A(variationId, options) {
    // Each question in 4A is a full tracing page (2 sub-questions),
    // so we generate 50 descriptor entries all pointing to the same tracing type.
    const pool = [];
    const traceNumber = parseInt((variationId || '4A1').replace('4A', ''), 10) || 1;

    while (pool.length < POOL_SIZE) {
      pool.push({
        id:          pool.length + 1,
        section:     '4A',
        variationId,
        label:       `Q${pool.length + 1}: Trace ${traceNumber} (${variationId})`,
        data:        { traceNumber, imageValues: options.imageValues || [] },
      });
    }
    return pool.slice(0, POOL_SIZE);
  }

  /* ── Public API ────────────────────────────────────────────── */

  /**
   * generate(sectionId, variationId, options) → QuestionDescriptor[]
   *
   * options may include:
   *   rangeStart, rangeEnd, perRow, blankCount  — for 3A3I
   *   eqAddend, seedNumbers                     — for 2A8
   *   imageUrl, imageCount, imagePerRow         — for 4A121
   *   imageValues                               — for 4A
   *   seed                                      — for reproducibility
   */
  function generate(sectionId, variationId, options) {
    options = options || {};
    switch (sectionId) {
      case '3A3I':  return generate3A3I(variationId, options);
      case '2A8':   return generate2A8(variationId, options);
      case '4A121': return generate4A121(variationId, options);
      case '4A':    return generate4A(variationId, options);
      default:      return generate3A3I('A', options);
    }
  }

  /**
   * generateMulti(sectionId, options) → QuestionDescriptor[]
   *
   * For pool-print mode: generates 100 questions for 3A3I cycling all A–K variations,
   * designed to be paired into 50 pages (2 questions each).
   * For other sections, falls back to the normal single-variation generator.
   */
  function generateMulti(sectionId, options) {
    options = options || {};
    if (sectionId === '3A3I') {
      return generate3A3IMultiVariation(options);
    }
    // For non-3A3I sections, fall back to normal pool
    return generate(sectionId, options.variationId || 'A', options);
  }

  /**
   * applyOrder(pool, orderedIndices) → QuestionDescriptor[]
   *
   * Reorders the pool based on an array of 0-based indices
   * (produced by the admin order-manager panel).
   */
  function applyOrder(pool, orderedIndices) {
    if (!Array.isArray(orderedIndices) || !orderedIndices.length) return pool;
    return orderedIndices
      .map(i => pool[i])
      .filter(Boolean);
  }

  window.QUESTION_POOL = { generate, generateMulti, applyOrder };

})();

