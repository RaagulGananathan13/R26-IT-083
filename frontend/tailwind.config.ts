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
          /* What goes ON the accent fill. White in light mode; near-black in
             dark, where the accent is light enough that white would fail. */
          contrast: "rgb(var(--accent-contrast) / <alpha-value>)",
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
      /*
        Overriding Tailwind's defaults rather than adding new names: components
        across the console already reach for rounded / -md / -lg / -xl, so
        redefining what those mean re-shapes every one of them at once and keeps
        the decision in one place.

        Soft but not playful. 10px on cards reads as a considered product
        surface; 4px reads as a developer tool and 20px as a consumer app.
      */
      borderRadius: {
        DEFAULT: "0.375rem", // 6px   small controls, chips
        md: "0.5rem", //        8px   buttons, inputs
        lg: "0.625rem", //      10px  inner panels
        xl: "0.75rem", //       12px  cards
        "2xl": "1rem", //       16px  outermost containers
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
