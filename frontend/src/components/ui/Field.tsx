import { cn } from "@/lib/format";

export function Field({
  label,
  hint,
  children,
  className,
  group,
}: {
  label: string;
  hint?: string;
  children: React.ReactNode;
  className?: string;
  /**
   * Set when the control is a group rather than one input — a radiogroup, a
   * set of checkboxes. A <label> may only name a single form control, and
   * wrapping a group in one makes the label text the accessible name of
   * whichever child comes first: the projection picker announced its
   * "Not specified" option as "Projection Radiograph".
   */
  group?: boolean;
}) {
  const body = (
    <>
      <span className="eyebrow block">{label}</span>
      <div className="mt-1.5">{children}</div>
      {hint && <p className="mt-1 text-2xs leading-relaxed text-ink-faint">{hint}</p>}
    </>
  );

  if (group) {
    return (
      <div role="group" aria-label={label} className={cn("block", className)}>
        {body}
      </div>
    );
  }
  return <label className={cn("block", className)}>{body}</label>;
}

const CONTROL =
  "w-full rounded-lg border border-line bg-surface px-3 py-2 text-sm text-ink " +
  "placeholder:text-ink-faint focus:border-ink-muted focus:outline-none";

export function Input({ className, ...props }: React.InputHTMLAttributes<HTMLInputElement>) {
  return <input className={cn(CONTROL, "tabular", className)} {...props} />;
}

export function Textarea({
  className,
  ...props
}: React.TextareaHTMLAttributes<HTMLTextAreaElement>) {
  return <textarea className={cn(CONTROL, "resize-y leading-relaxed", className)} {...props} />;
}

export function Select({
  className,
  ...props
}: React.SelectHTMLAttributes<HTMLSelectElement>) {
  return <select className={cn(CONTROL, "cursor-pointer", className)} {...props} />;
}

export function Checkbox({
  label,
  className,
  ...props
}: React.InputHTMLAttributes<HTMLInputElement> & { label: string }) {
  return (
    <label className={cn("flex cursor-pointer items-center gap-2 text-sm text-ink", className)}>
      <input
        type="checkbox"
        className="h-4 w-4 rounded border-line accent-[rgb(var(--ink))]"
        {...props}
      />
      {label}
    </label>
  );
}
