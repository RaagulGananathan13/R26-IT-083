/**
 * scripts/config/levels.js
 *
 * Maps each worksheet section code to the English Level label
 * displayed in the worksheet header.
 *
 * Usage:
 *   window.WORKSHEET_LEVELS_CONFIG.sectionLevels['3A3I'] // → '3A3I'
 *
 * To change what prints as the "English Level:" in the worksheet header,
 * edit the values here. Each key is the section ID and the value is
 * the free-form label string that will appear on the printed sheet.
 */
window.WORKSHEET_LEVELS_CONFIG = {
  sectionLevels: {
    '3A3I':  '3A3I',
    '2A8':   '2A8',
    '4A121': '4A121',
    '4A':    '4A',
  }
};
