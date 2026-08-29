"use client";

import { forwardRef } from "react";

import { cn } from "@/lib/format";

/**
 * Actions.
 *
 * The primary variant carries the accent as a solid fill with a subtle shadow,
 * because on a page of white cards a bordered button does not read as the thing
 * to press. Secondary is a real surface rather than a ghost, so the two are
 * distinguishable at a glance without reading their labels.
 */
type Variant = "primary" | "secondary" | "ghost" | "danger";
type Size = "sm" | "md" | "lg";

const VARIANTS: Record<Variant, string> = {
  primary: cn(
    "bg-accent text-accent-contrast shadow-sm",
    "hover:bg-accent-strong hover:shadow",
    "active:translate-y-px",
    "disabled:bg-accent/35 disabled:text-accent-contrast/70 disabled:shadow-none",
  ),
  secondary: cn(
    "border border-line bg-surface text-ink shadow-sm",
    "hover:bg-surface-2 hover:border-ink-faint/40",
    "active:translate-y-px",
    "disabled:text-ink-faint disabled:shadow-none",
  ),
  ghost: "text-ink-muted hover:bg-surface-2 hover:text-ink",
  danger: cn(
    "border border-verdict-withheld/35 bg-verdict-withheld/[0.07] text-verdict-withheld",
    "hover:bg-verdict-withheld/[0.13]",
    "active:translate-y-px",
  ),
};

const SIZES: Record<Size, string> = {
  sm: "h-8 gap-1.5 px-3 text-[0.8125rem]",
  md: "h-10 gap-2 px-4 text-sm",
  lg: "h-11 gap-2 px-5 text-[0.9375rem]",
};

export interface ButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: Variant;
  size?: Size;
  loading?: boolean;
}

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(function Button(
  { className, variant = "primary", size = "md", loading, disabled, children, ...props },
  ref,
) {
  return (
    <button
      ref={ref}
      /* Explicit: <button> defaults to type="submit", which inside any form
         turns a click into a navigation instead of an onClick. */
      type="button"
      disabled={disabled || loading}
      className={cn(
        "inline-flex select-none items-center justify-center rounded-md font-semibold",
        "transition-all duration-150 disabled:cursor-not-allowed disabled:active:translate-y-0",
        VARIANTS[variant],
        SIZES[size],
        className,
      )}
      {...props}
    >
      {loading && (
        <span
          aria-hidden
          className="h-3.5 w-3.5 animate-spin rounded-full border-2 border-current border-t-transparent"
        />
      )}
      {children}
    </button>
  );
});
