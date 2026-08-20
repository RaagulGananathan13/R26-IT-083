"use client";

import { useEffect } from "react";

/**
 * Route-level error boundary.
 *
 * Without one, any render-time throw shows Next's bare "Application error: a
 * client-side exception has occurred", which tells a reader nothing and offers
 * no way forward. A console someone is meant to work in should fail legibly.
 *
 * The common case in practice is ChunkLoadError: the server was rebuilt while
 * a tab was open, so the HTML references chunks that no longer exist. A reload
 * fetches the new build and fixes it, so that is offered first.
 */
export default function RouteError({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error("Route error:", error);
  }, [error]);

  const isStaleBuild =
    error.name === "ChunkLoadError" || /Loading chunk .* failed/i.test(error.message);

  return (
    <div className="mx-auto max-w-2xl px-6 py-16">
      <div className="rounded-xl border border-verdict-withheld/30 bg-verdict-withheld/10 px-6 py-6">
        <h1 className="text-lg font-semibold text-ink">
          {isStaleBuild ? "This page is from an older build" : "This page failed to render"}
        </h1>

        {isStaleBuild ? (
          <p className="mt-2 text-sm leading-relaxed text-ink-muted">
            The front end was rebuilt while this tab was open, so it is asking for files
            that no longer exist. Reloading fetches the current build.
          </p>
        ) : (
          <p className="mt-2 text-sm leading-relaxed text-ink-muted">
            No clinical result is shown, which is the correct outcome — a page that cannot
            render fully must not display a partial one.
          </p>
        )}

        <pre className="mt-4 max-h-40 overflow-auto scrollbar-thin rounded-lg bg-surface-2 p-3 text-2xs leading-relaxed text-ink-muted">
          {error.name}: {error.message}
          {error.digest ? `\ndigest: ${error.digest}` : ""}
        </pre>

        <div className="mt-5 flex flex-wrap gap-2">
          <button
            type="button"
            onClick={() => window.location.reload()}
            className="rounded-lg bg-ink px-4 py-2 text-sm font-medium text-white hover:bg-ink/90"
          >
            Reload page
          </button>
          <button
            type="button"
            onClick={reset}
            className="rounded-lg border border-line px-4 py-2 text-sm font-medium text-ink hover:bg-surface-2"
          >
            Try again
          </button>
        </div>
      </div>
    </div>
  );
}
