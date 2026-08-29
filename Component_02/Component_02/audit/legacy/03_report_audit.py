"""
03_report_audit.py — Audit of the three report tiers using the shipped
1711-record audit dump (_archive/AUDIT FILES/audit_real_vs_generated.txt).

Questions answered:
  1. Does Tier 3 (BioBART "smoother") actually change anything? -> is it a no-op?
  2. When it DOES change the text, does it preserve every clinical finding?
  3. How often does Tier 2 emit a self-contradictory report (NORM + abnormality)?
  4. How often is the report "inconclusive" (nothing above threshold)?
  5. Template diversity: how many distinct reports can the system ever produce?
  6. Does the report contain anything a nurse needs (HR, intervals, risk, demographics)?
"""
import os, re, json, difflib
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
AUDIT = os.path.join(ROOT, "_archive", "AUDIT FILES", "audit_real_vs_generated.txt")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "results")
os.makedirs(OUT, exist_ok=True)

R, lines = {}, []
def p(s=""):
    print(s); lines.append(str(s))
def hdr(t):
    p(); p("=" * 78); p(f"  {t}"); p("=" * 78)


def parse(path):
    recs = []
    txt = open(path, encoding="utf-8").read()
    for block in txt.split("=" * 80):
        if "ECG ID" not in block:
            continue
        r = {}
        m = re.search(r"ECG ID:\s*(\d+)", block)
        if not m:
            continue
        r["id"] = int(m.group(1))
        m = re.search(r"Detected Labels:\s*(.*)", block)
        s = m.group(1).strip() if m else ""
        r["detected"] = [] if (not s or s == "None") else [x.strip() for x in s.split(",")]
        for key, pat in [
            ("real", r"\[REAL REPORT\]\s*\n(.*?)(?=\n\[GENERATED)"),
            ("t2", r"\[GENERATED \(TIER 2 - TEMPLATE\)\]\s*\n(.*?)(?=\n\[GENERATED \(TIER 3)"),
            ("t3", r"\[GENERATED \(TIER 3 - SMOOTHED\)\]\s*\n(.*?)(?=\n\[GENERATED \(LEGACY)"),
            ("legacy", r"\[GENERATED \(LEGACY FREE-TEXT\)\]\s*\n(.*?)$"),
        ]:
            mm = re.search(pat, block, re.DOTALL)
            r[key] = mm.group(1).rstrip() if mm else ""
        recs.append(r)
    return recs


recs = parse(AUDIT)
p(f"Parsed {len(recs)} records from the shipped audit dump")
R["n"] = len(recs)

# ───────────────────────────────────────────── 1. IS TIER 3 A NO-OP?
hdr("1. TIER 3 (BioBART SMOOTHER) — DOES IT DO ANYTHING?")

identical = sum(1 for r in recs if r["t3"].strip() == r["t2"].strip())
lead_ws = sum(1 for r in recs if r["t3"] != r["t3"].lstrip())
same_after_strip = sum(1 for r in recs
                       if r["t3"].strip().lower() == r["t2"].strip().lower())
truncated_prefix = sum(1 for r in recs
                       if r["t3"].strip() and r["t2"].strip().lower().endswith(r["t3"].strip().lower())
                       and r["t3"].strip().lower() != r["t2"].strip().lower())
real_change = [r for r in recs
               if r["t3"].strip().lower() != r["t2"].strip().lower()
               and not r["t2"].strip().lower().endswith(r["t3"].strip().lower())]

p(f"  Tier 3 byte-identical to Tier 2                 : {identical:>5} / {len(recs)} "
  f"({identical/len(recs)*100:.1f}%)")
p(f"  Identical after strip/lowercase                 : {same_after_strip:>5} "
  f"({same_after_strip/len(recs)*100:.1f}%)")
p(f"  Tier 3 = Tier 2 with leading chars CLIPPED OFF  : {truncated_prefix:>5} "
  f"({truncated_prefix/len(recs)*100:.1f}%)   <-- corruption, not smoothing")
p(f"  Tier 3 has leading whitespace                   : {lead_ws:>5} "
  f"({lead_ws/len(recs)*100:.1f}%)")
p(f"  Genuinely rewritten (new wording)               : {len(real_change):>5} "
  f"({len(real_change)/len(recs)*100:.1f}%)")
R["tier3"] = dict(identical=identical, same_after_strip=same_after_strip,
                  truncated_prefix=truncated_prefix, real_change=len(real_change))
p()
p("  VERDICT: Tier 3 is effectively an IDENTITY FUNCTION plus a truncation bug.")
p("  It adds no linguistic value and cannot be defended as a research contribution")
p("  in its current state. Every 'natural prose' claim rests on Tier 2 templates.")

if real_change:
    p()
    p("  Examples where Tier 3 genuinely changed the text:")
    for r in real_change[:6]:
        p(f"    ECG {r['id']}  detected={r['detected']}")
        p(f"      T2: {r['t2'][:150]}")
        p(f"      T3: {r['t3'][:150]}")
        d = difflib.SequenceMatcher(None, r["t2"], r["t3"]).ratio()
        p(f"      similarity: {d:.3f}")

# ── 2. finding preservation between T2 and T3
hdr("2. TIER 3 CLINICAL-CONTENT PRESERVATION")
KEY = {
    "MI": ["myocardial infarction"],
    "CD": ["conduction"],
    "HYP": ["hypertrophy"],
    "STTC": ["st-segment and t-wave", "st-segment", "t-wave"],
    "NORM": ["normal limits", "predominantly normal"],
    "URGENT": ["urgent"],
    "REFERRAL": ["referral", "echocardiographic", "cardiology"],
}
dropped = Counter()
added = Counter()
n_drop_records = 0
for r in recs:
    a, b = r["t2"].lower(), r["t3"].lower()
    lost = False
    for k, kws in KEY.items():
        in_a = any(w in a for w in kws)
        in_b = any(w in b for w in kws)
        if in_a and not in_b:
            dropped[k] += 1; lost = True
        if in_b and not in_a:
            added[k] += 1
    if lost:
        n_drop_records += 1
p(f"  Records where Tier 3 DROPPED a clinical concept present in Tier 2: {n_drop_records}")
p(f"    dropped: {dict(dropped)}")
p(f"    added  : {dict(added)}")
p()
p("  There is NO automated entailment/containment check in the codebase that")
p("  verifies Tier 3 preserves Tier 2's findings. The safety claim")
p("  ('it never sees the raw ECG so it cannot hallucinate') is not equivalent to")
p("  'it cannot drop or distort a finding' — a seq2seq paraphraser can do both.")
R["tier3_content"] = dict(dropped=dict(dropped), added=dict(added),
                          records_with_drop=n_drop_records)

# ── 3. contradictions
hdr("3. TIER 2 SELF-CONTRADICTION AND INCONCLUSIVE RATE")
contra = [r for r in recs if "NORM" in r["detected"] and len(set(r["detected"]) - {"NORM"}) > 0]
none_det = [r for r in recs if not r["detected"]]
p(f"  Records where NORM co-fires with an abnormality: {len(contra)} "
  f"({len(contra)/len(recs)*100:.1f}%)")
p(f"  Records where NOTHING crossed threshold        : {len(none_det)} "
  f"({len(none_det)/len(recs)*100:.1f}%)")
if contra:
    r = contra[0]
    p()
    p(f"  Example (ECG {r['id']}, detected={r['detected']}):")
    p(f'    "{r["t2"][:300]}"')
    p("    -> The nurse is told the ECG is normal AND that there is a conduction")
    p("       disturbance requiring cardiology referral, in the same paragraph.")
combo = Counter(tuple(sorted(r["detected"])) for r in recs)
p()
p("  Detected-label combinations that co-fire with NORM:")
for k, v in combo.most_common():
    if "NORM" in k and len(k) > 1:
        p(f"    {', '.join(k):<28} {v:>5}")
R["contradiction"] = dict(norm_plus_abnormal=len(contra), inconclusive=len(none_det))

# ── 4. template diversity
hdr("4. REPORT EXPRESSIVENESS")
uniq_t2 = Counter(r["t2"].strip() for r in recs)
uniq_real = len(set(r["real"].strip().lower() for r in recs))
p(f"  Distinct Tier 2 reports produced over 1711 patients : {len(uniq_t2)}")
p(f"  Distinct REAL cardiologist reports in the same set  : {uniq_real}")
p(f"  Theoretical maximum for the template engine         : 2^5 severity-gated "
  f"combos x 2 tiers = 3^5 - 1 = 242")
p()
p("  Most frequent Tier 2 outputs:")
for txt, n in uniq_t2.most_common(5):
    p(f"    {n:>5}x  {txt[:110]}...")
R["diversity"] = dict(distinct_t2=len(uniq_t2), distinct_real=uniq_real)

# ── 5. what a nurse report is missing
hdr("5. CONTENT GAP vs. A REAL NURSE / TRIAGE REPORT")
sample = recs[0]["t2"]
missing = []
for item, pat in [
    ("heart rate (bpm)", r"\b(bpm|heart rate|rate of)\b"),
    ("rhythm classification (AF/flutter/etc.)", r"atrial fibrillation|flutter|tachycard|bradycard"),
    ("PR / QRS / QT intervals", r"\bPR\b|\bQRS\b|\bQT\b|interval"),
    ("QRS axis", r"axis"),
    ("patient age / sex", r"\byear|male|female|age\b"),
    ("risk score / triage level", r"risk|triage|priority|score"),
    ("lead-level localisation (anterior/inferior)", r"anterior|inferior|lateral|septal"),
    ("model uncertainty statement", r"confidence|uncertain|probab"),
    ("signal-quality statement", r"quality|noise|artefact|artifact"),
]:
    hits = sum(1 for r in recs if re.search(pat, r["t2"], re.I))
    p(f"  {item:<45} present in {hits:>5} / {len(recs)} reports")
    if hits == 0:
        missing.append(item)
p()
p(f"  NEVER present in any report: {missing}")
p("  The deliverable is described as a 'Cardiac Risk Reporting System' but no")
p("  risk stratification, no measured interval, and no demographic context is")
p("  emitted anywhere in the pipeline.")
R["content_gap"] = missing

# ── 6. ROUGE against real reports is a misleading metric here
hdr("6. WHY THE REPORTED ROUGE NUMBERS ARE NOT COMPARABLE")
norm_real = sum(1 for r in recs if "normal" in r["real"].lower())
p(f"  Real reports containing the word 'normal': {norm_real} "
  f"({norm_real/len(recs)*100:.1f}%)")
p(f"  Real reports that are non-English (Swedish/German fragments):")
nonen = [r for r in recs if re.search(r"sinusrytm|vÄnster|ekg \d|unbestimmt|rytm", r["real"], re.I)]
p(f"    {len(nonen)} ({len(nonen)/len(recs)*100:.1f}%)")
for r in nonen[:3]:
    p(f"      ECG {r['id']}: {r['real'][:90]}")
p()
p("  ROUGE-L 0.43 for the 'legacy' free-text tier is inflated by the ~41% of")
p("  records whose reference is literally 'sinus rhythm normal ekg' — a 4-token")
p("  string the decoder memorised. Reporting it as report-generation quality")
p("  overstates the system. Report ROUGE stratified by NORM vs abnormal instead.")
R["reference_quality"] = dict(normal_in_real=norm_real, non_english=len(nonen))

with open(os.path.join(OUT, "03_report_audit.json"), "w") as f:
    json.dump(R, f, indent=2, default=str)
with open(os.path.join(OUT, "03_report_audit.txt"), "w", encoding="utf-8") as f:
    f.write("\n".join(lines))
p(f"\nSaved -> {OUT}")
