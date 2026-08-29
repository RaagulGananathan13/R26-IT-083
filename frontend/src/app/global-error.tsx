"use client";

/**
 * Last-resort boundary, for a throw in the root layout itself.
 *
 * It must render its own <html> and <body>: at this point the layout that
 * normally provides them is the thing that failed. Tailwind classes are not
 * relied on here for the same reason, so the message survives a stylesheet
 * that never loaded.
 */
export default function GlobalError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  return (
    <html lang="en">
      <body
        style={{
          fontFamily: "system-ui, -apple-system, Segoe UI, sans-serif",
          margin: 0,
          padding: "3rem 1.5rem",
          background: "#f8fafc",
          color: "#0f172a",
        }}
      >
        <div style={{ maxWidth: "40rem", margin: "0 auto" }}>
          <h1 style={{ fontSize: "1.125rem", fontWeight: 600 }}>
            The console failed to start
          </h1>
          <p style={{ fontSize: "0.875rem", lineHeight: 1.6, color: "#475569" }}>
            This is a failure of the interface, not a clinical result. Reloading usually
            resolves it; if it does not, restart the front end.
          </p>
          <pre
            style={{
              background: "#f1f5f9",
              padding: "0.75rem",
              borderRadius: "0.5rem",
              fontSize: "0.6875rem",
              overflow: "auto",
            }}
          >
            {error.name}: {error.message}
          </pre>
          <button
            type="button"
            onClick={reset}
            style={{
              marginTop: "1rem",
              background: "#0f172a",
              color: "#fff",
              border: 0,
              borderRadius: "0.5rem",
              padding: "0.5rem 1rem",
              fontSize: "0.875rem",
              cursor: "pointer",
            }}
          >
            Try again
          </button>
        </div>
      </body>
    </html>
  );
}
