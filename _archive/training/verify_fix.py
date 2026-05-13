"""
Quick verification: apply V2 fix + final text cleanup
to EXISTING audit data (no model inference needed).
"""
import re, os

WORK_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIT_FILE = os.path.join(WORK_DIR, "audit_real_vs_generated.txt")


def clean_text(text):
    """Fix known BioBART text artifacts -- applied as final pass."""
    text = text.replace("ST-seal", "ST-segment")
    text = text.replace("STsegment", "ST-segment")
    text = text.replace("ST- segment", "ST-segment")
    text = text.replace(".Clinical", ". Clinical")
    text = text.replace(".Cl ", ". Clinical ")
    text = text.replace("Cl correlation", "Clinical correlation")
    text = text.replace("record keeping", "correlation")
    text = text.replace("  ", " ")
    text = re.sub(r'\.(?=[A-Z])', '. ', text)
    text = re.sub(r' {2,}', ' ', text)
    return text.strip()


def fix_smoothed(smoothed, template):
    """Fix truncation/garbled start, then clean typos."""
    smoothed = smoothed.strip()
    if not smoothed:
        result = template
    elif smoothed[0].isupper() and template.startswith(smoothed[:10]):
        result = smoothed
    else:
        tl = template.lower()
        sl = smoothed.lower()
        result = template  # default fallback
        for offset in range(1, min(30, len(template))):
            frag = tl[offset:offset + 15]
            if frag and sl.startswith(frag):
                result = template[:offset] + smoothed.lstrip(" -")
                break
    # Final text cleanup pass
    return clean_text(result)


def check_typos(text):
    issues = []
    if "ST-seal" in text:
        issues.append("ST-seal")
    if "STsegment" in text:
        issues.append("STsegment")
    if "ST- segment" in text:
        issues.append("ST-_segment")
    if re.search(r'\.[A-Z]', text):
        issues.append("period_no_space")
    if ".Cl " in text or "Cl correlation" in text:
        issues.append("Cl_truncated")
    if "record keeping" in text:
        issues.append("record_keeping")
    if "  " in text:
        issues.append("double_space")
    return issues


def is_bad(text):
    if not text:
        return False
    return (text[0] in " -" or text.startswith("G ") or text.startswith("Graphic"))


def parse_audit(path):
    records = []
    with open(path, "r", encoding="utf-8") as f:
        content = f.read()
    for block in content.split("=" * 80):
        block = block.strip()
        if not block or "ECG ID" not in block:
            continue
        rec = {}
        m = re.search(r"ECG ID:\s*(\d+)", block)
        if m:
            rec["ecg_id"] = int(m.group(1))
        m = re.search(r"\[GENERATED \(TIER 2 - TEMPLATE\)\]\s*\n(.+?)(?=\n\[GENERATED \(TIER 3)", block, re.DOTALL)
        if m:
            rec["tier2"] = m.group(1).strip()
        m = re.search(r"\[GENERATED \(TIER 3 - SMOOTHED\)\]\s*\n(.+?)(?=\n\[GENERATED \(LEGACY)", block, re.DOTALL)
        if m:
            rec["tier3"] = m.group(1).strip()
        if "ecg_id" in rec:
            records.append(rec)
    return records


def main():
    print("Parsing existing audit file...")
    records = parse_audit(AUDIT_FILE)
    print(f"  Loaded {len(records)} records")

    # BEFORE
    trunc_before = sum(1 for r in records if is_bad(r.get("tier3", "")))
    typo_before = sum(1 for r in records if check_typos(r.get("tier3", "")))

    # Apply fix
    trunc_after = 0
    typo_after = 0
    typo_details = []
    examples = []

    for rec in records:
        t2 = rec.get("tier2", "")
        t3 = rec.get("tier3", "")
        if not t2 or not t3:
            continue
        fixed = fix_smoothed(t3, t2)
        rec["fixed"] = fixed
        if is_bad(fixed):
            trunc_after += 1
        remaining = check_typos(fixed)
        if remaining:
            typo_after += 1
            if len(typo_details) < 10:
                typo_details.append((rec["ecg_id"], remaining, fixed[:120]))
        if fixed != t3 and len(examples) < 5:
            examples.append(rec)

    print()
    print("=" * 60)
    print("  V2 FIX + FINAL CLEANUP VERIFICATION")
    print("=" * 60)
    print(f"  Truncation: {trunc_before} => {trunc_after}")
    print(f"  Text typos: {typo_before} => {typo_after}")
    print()
    if examples:
        print("  SAMPLE FIXES:")
        print("-" * 60)
        for rec in examples:
            print(f"  ECG {rec['ecg_id']}:")
            print(f"    RAW:   {rec['tier3'][:90]}")
            print(f"    FIXED: {rec['fixed'][:90]}")
            print()
    if typo_details:
        print("  REMAINING TYPOS:")
        print("-" * 60)
        for eid, issues, preview in typo_details:
            print(f"  ECG {eid}: {issues}")
            print(f"    {preview}")
            print()

    print("=" * 60)
    if trunc_after == 0 and typo_after == 0:
        print("  ALL CLEAR! Zero truncation, zero typos.")
    else:
        print(f"  Remaining: {trunc_after} truncation, {typo_after} typos")
    print("=" * 60)


if __name__ == "__main__":
    main()
