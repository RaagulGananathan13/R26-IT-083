"use client";

import { useState } from "react";

import { Card, CardBody, CardHeader } from "@/components/ui";

/**
 * The component's own payload, unmodified.
 *
 * Present on every result page so any figure shown in the interface can be
 * checked against what the component actually returned.
 */
export function RawPayload({ payload }: { payload: unknown }) {
  const [open, setOpen] = useState(false);

  return (
    <Card>
      <CardHeader
        title="Component payload"
        description="Returned verbatim. Every figure above can be checked against this."
        actions={
          <button
            type="button"
            onClick={() => setOpen((value) => !value)}
            className="rounded-lg border border-line px-3 py-1.5 text-xs font-medium text-ink hover:bg-surface-2"
          >
            {open ? "Hide" : "Show"}
          </button>
        }
      />
      {open && (
        <CardBody>
          <pre className="max-h-[28rem] overflow-auto scrollbar-thin rounded-lg bg-surface-2 p-3 text-2xs leading-relaxed text-ink-muted">
            {JSON.stringify(payload, null, 2)}
          </pre>
        </CardBody>
      )}
    </Card>
  );
}
