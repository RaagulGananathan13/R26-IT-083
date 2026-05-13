/**
 * scripts/config/variations.js
 *
 * Defines all variation types for every worksheet section.
 * Each variation must have:
 *   id        — unique identifier used internally and in the UI
 *   label     — short display name shown on the type card
 *   desc      — one-line description shown below the label
 *   section   — the parent section this variation belongs to
 *
 * Admin: Add new variations here. The variation cards in the sidebar
 * are rendered automatically from this configuration.
 *
 * Question pool generators in scripts/generators/question-pool.js
 * use the `id` field to decide how to sample 50 unique questions
 * for each variation.
 */
window.WORKSHEET_VARIATIONS = {

  /* ── 3A3I: Number Sequence Worksheets ─────────────────────── */
  '3A3I': [
    {
      id:    'A',
      label: 'Type A',
      desc:  'Show all numbers, write them below',
    },
    {
      id:    'B',
      label: 'Type B',
      desc:  'Fill in the first N blanks from the left',
    },
    {
      id:    'C',
      label: 'Type C',
      desc:  'Circles with randomly missing values',
    },
    {
      id:    'D',
      label: 'Type D',
      desc:  'Fill in the last N blanks from the right',
    },
    {
      id:    'E',
      label: 'Type E',
      desc:  'Sequence in reverse (countdown) order',
    },
    {
      id:    'F',
      label: 'Type F',
      desc:  'Odd numbers only — even positions blank',
    },
    {
      id:    'G',
      label: 'Type G',
      desc:  'Even numbers only — odd positions blank',
    },
    {
      id:    'H',
      label: 'Type H',
      desc:  'Skip counting by 2 — every other blank',
    },
    {
      id:    'I',
      label: 'Type I',
      desc:  'Skip counting by 5 — every fifth blank',
    },
    {
      id:    'J',
      label: 'Type J',
      desc:  'Random blanks scattered throughout row',
    },
    {
      id:    'K',
      label: 'Type K',
      desc:  'Number-line — boxes connected by coloured lines',
    },
  ],

  /* ── 2A8: Addition Equation Worksheets ────────────────────── */
  '2A8': [
    {
      id:    'E',
      label: 'Type A',
      desc:  'Addition equations — write the sum answer',
    },
    {
      id:    'E2',
      label: 'Type B',
      desc:  'Addition with missing addend — find the addend',
    },
    {
      id:    'E3',
      label: 'Type C',
      desc:  'Subtraction equations — write the difference',
    },
    {
      id:    'E4',
      label: 'Type D',
      desc:  'Mixed addition and subtraction equations',
    },
    {
      id:    'E5',
      label: 'Type E',
      desc:  'Doubles — add same number to itself',
    },
    {
      id:    'E6',
      label: 'Type F',
      desc:  'Near doubles — doubles ± 1',
    },
    {
      id:    'E7',
      label: 'Type G',
      desc:  'Adding 10 — quick tens strategy',
    },
    {
      id:    'E8',
      label: 'Type H',
      desc:  'Adding 0 — zero property identity',
    },
    {
      id:    'E9',
      label: 'Type I',
      desc:  'Number bonds to 10 — fill the pair',
    },
    {
      id:    'E10',
      label: 'Type J',
      desc:  'Three-number addition — bracket groups',
    },
  ],

  /* ── 4A121: Image Counting Worksheets ─────────────────────── */
  '4A121': [
    {
      id:    'F',
      label: 'Type A',
      desc:  'Count repeated images — write in box',
    },
    {
      id:    'F2',
      label: 'Type B',
      desc:  'Count and circle the correct numeral',
    },
    {
      id:    'F3',
      label: 'Type C',
      desc:  'Count — write word and numeral',
    },
    {
      id:    'F4',
      label: 'Type D',
      desc:  'Two groups — count each and add',
    },
    {
      id:    'F5',
      label: 'Type E',
      desc:  'Count — match to number line position',
    },
    {
      id:    'F6',
      label: 'Type F',
      desc:  'Count — more/less comparison of two sets',
    },
    {
      id:    'F7',
      label: 'Type G',
      desc:  'Count groups of 2 — skip count by 2',
    },
    {
      id:    'F8',
      label: 'Type H',
      desc:  'Count groups of 5 — skip count by 5',
    },
    {
      id:    'F9',
      label: 'Type I',
      desc:  'Count total then subtract crossed out items',
    },
    {
      id:    'F10',
      label: 'Type J',
      desc:  'Order three image sets from fewest to most',
    },
  ],

  /* ── 4A: Number Tracing (10 numbers) ──────────────────────── */
  '4A': [
    { id: '4A1',  label: '4A1',  desc: 'Trace the number 1' },
    { id: '4A2',  label: '4A2',  desc: 'Trace the number 2' },
    { id: '4A3',  label: '4A3',  desc: 'Trace the number 3' },
    { id: '4A4',  label: '4A4',  desc: 'Trace the number 4' },
    { id: '4A5',  label: '4A5',  desc: 'Trace the number 5' },
    { id: '4A6',  label: '4A6',  desc: 'Trace the number 6' },
    { id: '4A7',  label: '4A7',  desc: 'Trace the number 7' },
    { id: '4A8',  label: '4A8',  desc: 'Trace the number 8' },
    { id: '4A9',  label: '4A9',  desc: 'Trace the number 9' },
    { id: '4A10', label: '4A10', desc: 'Trace the number 10' },
  ],
};
