import type { Config } from "tailwindcss";

/**
 * Tokens mirror `globals.css`. Two families of colour, and they mean different
 * things:
 *
 *   accent   the product's own voice — teal, carried over from Component 01
 *   verdict  reserved entirely for the reliability verdict, and never used for
 *            decoration, so one amber element on a page always means the same
 *            thing wherever a reader sees it
 */
const config: Config = {
  content: ["./src/**/*.{ts,tsx}"],
  darkMode: "class",
  theme: {
    extend: {
      colors: {
        bg: "rgb(var(--bg) / <alpha-value>)",
        surface: {
          DEFAULT: "rgb(var(--surface) / <alpha-value>)",
          2: "rgb(var(--surface-2) / <alpha-value>)",
        },
        ink: {
          DEFAULT: "rgb(var(--ink) / <alpha-value>)",
          muted: "rgb(var(--ink-muted) / <alpha-value>)",
          faint: "rgb(var(--ink-faint) / <alpha-value>)",
        },
        line: "rgb(var(--line) / <alpha-value>)",
        accent: {
          DEFAULT: "rgb(var(--accent) / <alpha-value>)",
          strong: "rgb(var(--accent-strong) / <alpha-value>)",
        },
        verdict: {
          actionable: "rgb(var(--verdict-actionable) / <alpha-value>)",
          caution: "rgb(var(--verdict-caution) / <alpha-value>)",
          deferred: "rgb(var(--verdict-deferred) / <alpha-value>)",
          withheld: "rgb(var(--verdict-withheld) / <alpha-value>)",
          unavailable: "rgb(var(--verdict-unavailable) / <alpha-value>)",
        },
      },
      fontFamily: {
        sans: ["var(--font-sans)"],
        display: ["var(--font-display)"],
        mono: ["var(--font-mono)"],
      },
      fontSize: {
        "2xs": ["0.6875rem", { lineHeight: "1rem" }],
      },
      borderRadius: {
        xl: "0.875rem",
        "2xl": "1.125rem",
      },
      keyframes: {
        shimmer: { "100%": { transform: "translateX(100%)" } },
      },
      animation: {
        shimmer: "shimmer 1.6s infinite",
      },
    },
  },
  plugins: [],
};

export default config;
