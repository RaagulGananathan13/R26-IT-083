/** Non-dismissible. This is a research prototype and must always say so. */
export function DisclaimerBar() {
  return (
    <div className="border-b border-line bg-surface-2 px-6 py-2">
      <p className="text-2xs leading-relaxed text-ink-muted">
        <span className="font-semibold text-ink">Research prototype.</span> Not a medical
        device, not clinically validated, and not for diagnosis or treatment decisions.
        Every output requires review by a qualified clinician.
      </p>
    </div>
  );
}
