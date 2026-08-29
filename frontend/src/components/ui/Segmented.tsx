"use client";

import { cn } from "@/lib/format";

export interface SegmentedOption<T extends string> {
  value: T;
  label: string;
}

/** The segmented control from Component 01's console: projection, view mode. */
export function Segmented<T extends string>({
  options,
  value,
  onChange,
  disabled,
  className,
  ariaLabel,
}: {
  options: SegmentedOption<T>[];
  value: T;
  onChange: (value: T) => void;
  disabled?: boolean;
  className?: string;
  ariaLabel?: string;
}) {
  return (
    <div className={cn("segmented", className)} role="radiogroup" aria-label={ariaLabel}>
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          role="radio"
          aria-checked={value === option.value}
          disabled={disabled}
          onClick={() => onChange(option.value)}
          className={cn("segmented-btn", value === option.value && "active",
            disabled && "cursor-not-allowed opacity-50")}
        >
          {option.label}
        </button>
      ))}
    </div>
  );
}
