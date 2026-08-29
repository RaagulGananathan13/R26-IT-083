"use client";

import { Button, Card, CardBody } from "@/components/ui";

/**
 * What replaces the upload control once a study has been analysed.
 *
 * The file picker has done its job by then. Leaving it on screen costs a whole
 * card of vertical space above the result, and on a laptop that is the
 * difference between the verdict being visible on arrival and the reader
 * having to scroll to find out what the system said. The name of the study
 * still has to stay visible — a result with no visible provenance is worse
 * than the space it saves — so it collapses to one line rather than vanishing.
 */
export function LoadedStudy({
  name,
  detail,
  onReset,
  label = "New study",
}: {
  name: string;
  detail?: string;
  onReset: () => void;
  label?: string;
}) {
  return (
    <Card>
      <CardBody className="flex items-center gap-3 py-3">
        <span
          className="grid h-8 w-8 shrink-0 place-items-center rounded-full bg-verdict-actionable/15 text-verdict-actionable"
          aria-hidden
        >
          <svg viewBox="0 0 20 20" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth={2.2}>
            <path d="M4 10.5l4 4 8-8" strokeLinecap="round" strokeLinejoin="round" />
          </svg>
        </span>
        <span className="min-w-0 flex-1">
          <span className="eyebrow block">Analysed</span>
          <span className="block truncate text-sm font-medium text-ink" title={name}>
            {name}
          </span>
          {detail && (
            <span className="block truncate text-2xs text-ink-faint">{detail}</span>
          )}
        </span>
        <Button variant="ghost" size="sm" onClick={onReset} className="shrink-0">
          {label}
        </Button>
      </CardBody>
    </Card>
  );
}
