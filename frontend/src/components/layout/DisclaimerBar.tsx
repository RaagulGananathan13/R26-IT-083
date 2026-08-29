/** Non-dismissible. This is a research prototype and must always say so. */
export function DisclaimerBar() {
  return (
    <div className="no-print border-b border-verdict-caution/25 bg-verdict-caution/[0.07]">
      <p className="mx-auto max-w-[88rem] px-5 py-2 text-[0.8125rem] leading-relaxed text-ink-muted sm:px-8">
        <span className="font-bold text-verdict-caution">Research prototype.</span>{" "}
        Not a medical device, not clinically validated, and not for diagnosis or treatment
        decisions. Every output requires review by a qualified clinician.
      </p>
    </div>
  );
}
