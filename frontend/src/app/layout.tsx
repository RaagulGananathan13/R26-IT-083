import type { Metadata, Viewport } from "next";

import { DisclaimerBar } from "@/components/layout/DisclaimerBar";
import { Sidebar } from "@/components/layout/Sidebar";

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
    { media: "(prefers-color-scheme: light)", color: "#f8fafc" },
    { media: "(prefers-color-scheme: dark)", color: "#020617" },
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
        <div className="flex h-screen overflow-hidden">
          <Sidebar />
          <div className="flex min-w-0 flex-1 flex-col">
            <DisclaimerBar />
            <main className="flex-1 overflow-y-auto scrollbar-thin">{children}</main>
          </div>
        </div>
      </body>
    </html>
  );
}
