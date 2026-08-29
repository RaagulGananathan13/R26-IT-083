import { cn } from "@/lib/format";

/**
 * Explainability on the left, the written analysis on the right.
 *
 * These two are read together or not at all. A heat map answers "where did it
 * look"; the report answers "what did it say"; and checking one against the
 * other is the entire point of an explainable system. Stacked vertically the
 * reader has to scroll between them and hold one in their head, which is
 * exactly the comparison the layout should be doing for them.
 *
 * Side by side also halves the height of the densest part of the page, which
 * matters on a six-stage walkthrough viewed on a laptop.
 *
 * LAYOUT ONLY, DELIBERATELY
 * -------------------------
 * This adds no border, background or padding of its own. Most of what gets
 * passed here is already a Card with its own header and controls, and wrapping
 * a card in another bordered box produces two frames around one thing. Where a
 * caller passes chrome-less content instead -- as the pathway does, since it is
 * already inside a stage card -- it supplies a label and the eyebrow carries
 * the heading that the missing card header would have.
 *
 * Falls back to a single column when only one side has content: an empty pane
 * beside a full one reads as something that failed to load. Stacks below `lg`,
 * where two columns would leave both too narrow to read.
 */
export function AnalysisSplit({
  left,
  right,
  leftLabel,
  rightLabel,
  className,
}: {
  left?: React.ReactNode;
  right?: React.ReactNode;
  /** Only needed when the content has no header of its own. */
  leftLabel?: string;
  rightLabel?: string;
  className?: string;
}) {
  const hasLeft = Boolean(left);
  const hasRight = Boolean(right);
  if (!hasLeft && !hasRight) return null;

  const pane = (node: React.ReactNode, label?: string) => (
    <section className="min-w-0 space-y-2.5">
      {label && <p className="eyebrow">{label}</p>}
      {node}
    </section>
  );

  if (!hasLeft || !hasRight) {
    return (
      <div className={className}>
        {hasLeft ? pane(left, leftLabel) : pane(right, rightLabel)}
      </div>
    );
  }

  return (
    <div className={cn("grid items-start gap-5 lg:grid-cols-2", className)}>
      {pane(left, leftLabel)}
      {pane(right, rightLabel)}
    </div>
  );
}
