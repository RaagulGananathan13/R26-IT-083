import type { Metadata, Viewport } from "next";

import { DisclaimerBar } from "@/components/layout/DisclaimerBar";
import { TopBar } from "@/components/layout/TopBar";

import "./globals.css";

export const metadata: Metadata = {
  title: {
    default: "Cardiovascular XAI — R26-IT-083",
    template: "%s · Cardiovascular XAI",
  },
  description:
    "Clinical console for the Explainable AI System for Cardiovascular Disease Detection and Diagnosis. Research prototype; not a medical device.",
  robots: { index: false, follow: false },
};

export const viewport: Viewport = {
  themeColor: [
    { media: "(prefers-color-scheme: light)", color: "#f9fafc" },
    { media: "(prefers-color-scheme: dark)", color: "#0a0e13" },
  ],
};

/** Applied before paint so a dark-theme reader never sees a white flash. */
const THEME_SCRIPT = `
try {
  var stored = localStorage.getItem('cvxai-theme');
  var dark = stored ? stored === 'dark'
    : window.matchMedia('(prefers-color-scheme: dark)').matches;
  if (dark) document.documentElement.classList.add('dark');
} catch (e) {}
`;

export default function RootLayout({ children }: { children: React.ReactNode }) {
  return (
    <html lang="en" suppressHydrationWarning>
      <head>
        <script dangerouslySetInnerHTML={{ __html: THEME_SCRIPT }} />
      </head>
      <body>
        {/* Document flow rather than a fixed-height flex shell: the rail is
            sticky, so the page scrolls as one surface and long findings tables
            are not trapped in an inner scroller with its own scrollbar. */}
        <TopBar />
        <DisclaimerBar />
        <main className="mx-auto w-full max-w-[88rem] px-5 py-8 sm:px-8 sm:py-10">
          {children}
        </main>
      </body>
    </html>
  );
}
