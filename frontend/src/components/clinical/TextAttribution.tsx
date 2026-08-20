"use client";

import { Card, CardBody, CardHeader } from "@/components/ui";
import { cn, decimal } from "@/lib/format";

interface Token {
  term: string;
  category: string;
  weight: number;
  negated?: boolean;
}

/**
 * Which words in the chief complaint the model reacted to.
 *
 * Component 04 is the only one of the four whose explanation is linguistic, so
 * this is its equivalent of a saliency map: the complaint is shown verbatim
 * with matched terms marked in place, and the terms are listed with their
 * weight and category.
 *
 * Negated terms are marked distinctly rather than dropped. "Denies chest pain"
 * matched a chest-pain term and then negated it, and a reader who cannot see
 * that will wonder why a chest-pain term is listed for a patient without chest
 * pain.
 */
export function TextAttribution({
  complaint,
  tokens,
  modalityNote,
}: {
  complaint: string;
  tokens: Token[];
  modalityNote?: string;
}) {
  if (!complaint && tokens.length === 0) return null;

  return (
    <Card>
      <CardHeader
        title="Chief-complaint attribution"
        description="Clinical-lexicon matches in the triage free text. At the triage-desk horizon this channel carries 31.3 % of the model's attribution."
      />
      <CardBody className="space-y-4">
        {complaint && (
          <blockquote className="rounded-xl border border-line bg-surface-2 px-4 py-3 text-sm leading-relaxed text-ink">
            {highlight(complaint, tokens)}
          </blockquote>
        )}

        {tokens.length > 0 ? (
          <ul className="space-y-1.5">
            {tokens.map((token, index) => (
              <li
                key={`${token.term}-${index}`}
                className="flex flex-wrap items-center gap-2 text-xs"
              >
                <span
                  className={cn(
                    "rounded-md px-2 py-0.5 font-medium",
                    token.negated
                      ? "bg-surface-2 text-ink-faint line-through"
                      : "bg-accent/10 text-accent",
                  )}
                >
                  {token.term}
                </span>
                <span className="text-ink-muted">{token.category.replace(/_/g, " ")}</span>
                <span className="tabular ml-auto text-ink-faint">
                  weight {token.weight > 0 ? "+" : ""}
                  {decimal(token.weight, 1)}
                </span>
                {token.negated && (
                  <span className="text-2xs uppercase tracking-wide text-verdict-caution">
                    negated
                  </span>
                )}
              </li>
            ))}
          </ul>
        ) : (
          <p className="text-xs leading-relaxed text-ink-muted">
            No lexicon terms matched. The text channel contributes nothing for this
            record, which is itself informative rather than an error.
          </p>
        )}

        {modalityNote && (
          <p className="border-t border-line pt-3 text-2xs leading-relaxed text-ink-faint">
            {modalityNote}
          </p>
        )}
      </CardBody>
    </Card>
  );
}

/** Mark matched terms in place, longest first so overlaps resolve sensibly. */
function highlight(text: string, tokens: Token[]) {
  if (tokens.length === 0) return text;

  const terms = [...tokens]
    .filter((token) => token.term)
    .sort((a, b) => b.term.length - a.term.length);

  const pattern = new RegExp(
    `(${terms.map((token) => escapeRegExp(token.term)).join("|")})`,
    "gi",
  );
  const byTerm = new Map(terms.map((token) => [token.term.toLowerCase(), token]));

  return text.split(pattern).map((part, index) => {
    const token = byTerm.get(part.toLowerCase());
    if (!token) return <span key={index}>{part}</span>;
    return (
      <mark
        key={index}
        title={`${token.category.replace(/_/g, " ")} · weight ${token.weight}`}
        className={cn(
          "rounded px-0.5",
          token.negated
            ? "bg-transparent text-ink-faint line-through decoration-ink-faint"
            : "bg-accent/15 text-ink",
        )}
      >
        {part}
      </mark>
    );
  });
}

function escapeRegExp(value: string): string {
  return value.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}
