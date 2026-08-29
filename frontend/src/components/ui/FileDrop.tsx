"use client";

import { useCallback, useId, useRef, useState } from "react";

import { cn, fileSize } from "@/lib/format";

export function FileDrop({
  label,
  hint,
  accept,
  file,
  onFile,
  disabled,
  compact,
}: {
  label: string;
  hint?: string;
  /** Comma-separated extension list, e.g. ".png,.jpg" */
  accept?: string;
  file: File | null;
  onFile: (file: File | null) => void;
  disabled?: boolean;
  compact?: boolean;
}) {
  const inputId = useId();
  const inputRef = useRef<HTMLInputElement>(null);
  const [over, setOver] = useState(false);

  const accepts = useCallback(
    (candidate: File) => {
      if (!accept) return true;
      const allowed = accept.split(",").map((item) => item.trim().toLowerCase());
      return allowed.some((extension) => candidate.name.toLowerCase().endsWith(extension));
    },
    [accept],
  );

  const handleDrop = useCallback(
    (event: React.DragEvent) => {
      event.preventDefault();
      setOver(false);
      if (disabled) return;
      const dropped = event.dataTransfer.files?.[0];
      if (dropped && accepts(dropped)) onFile(dropped);
    },
    [accepts, disabled, onFile],
  );

  return (
    <div>
      <div
        onDragOver={(event) => {
          event.preventDefault();
          if (!disabled) setOver(true);
        }}
        onDragLeave={() => setOver(false)}
        onDrop={handleDrop}
        className={cn(
          "rounded-xl border border-dashed transition-colors",
          compact ? "px-3 py-3" : "px-4 py-6",
          over ? "border-ink-muted bg-surface-2" : "border-line bg-surface",
          disabled && "opacity-50",
        )}
      >
        <div className="flex items-center justify-between gap-3">
          <div className="min-w-0">
            <p className="text-sm font-medium text-ink">{label}</p>
            {file ? (
              <p className="mt-0.5 truncate text-xs text-ink-muted" title={file.name}>
                {file.name}
                <span className="text-ink-faint"> · {fileSize(file.size)}</span>
              </p>
            ) : (
              hint && <p className="mt-0.5 text-xs text-ink-faint">{hint}</p>
            )}
          </div>
          <div className="flex shrink-0 items-center gap-1.5">
            <label
              htmlFor={inputId}
              className={cn(
                "cursor-pointer rounded-lg border border-line px-3 py-1.5 text-xs font-medium",
                "text-ink transition-colors hover:bg-surface-2",
                disabled && "pointer-events-none",
              )}
            >
              {file ? "Replace" : "Choose"}
            </label>
            {file && (
              <button
                type="button"
                onClick={() => {
                  onFile(null);
                  if (inputRef.current) inputRef.current.value = "";
                }}
                className="rounded-lg px-2 py-1.5 text-xs text-ink-faint hover:text-ink"
              >
                Clear
              </button>
            )}
          </div>
        </div>
      </div>
      <input
        ref={inputRef}
        id={inputId}
        type="file"
        accept={accept}
        disabled={disabled}
        className="sr-only"
        onChange={(event) => onFile(event.target.files?.[0] ?? null)}
      />
    </div>
  );
}
