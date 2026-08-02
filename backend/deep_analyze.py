"""
Deep analysis to find perfect positive and negative cardiomegaly samples.
Criteria:
  POSITIVE: Cardiomegaly=1, report explicitly mentions cardiomegaly/enlarged heart (not negated)
  NEGATIVE: Cardiomegaly=0, Edema=0, Pleural_Effusion=0, report is clean (no effusion/edema mentions)
"""
import csv, os, re, json
csv.field_size_limit(10 * 1024 * 1024)

CSV_PATH = os.path.join(os.path.dirname(__file__), '..', 'cardiomegaly_dataset', 'cardio_test.csv')
POS_DIR = os.path.join(os.path.dirname(__file__), '..', 'cardio_image_384', 'test', 'positive')
NEG_DIR = os.path.join(os.path.dirname(__file__), '..', 'cardio_image_384', 'test', 'negative')

# Load CSV
rows = []
with open(CSV_PATH, 'r', encoding='utf-8') as f:
    for row in csv.DictReader(f):
        rows.append(row)

pos_imgs = set(os.listdir(POS_DIR))
neg_imgs = set(os.listdir(NEG_DIR))

def get_report(r):
    report = (r.get('report_text', '') or '').strip()
    if not report:
        report = (r.get('findings_text', '') or '').strip()
    if not report:
        report = (r.get('impression_text', '') or '').strip()
    return report

# ===== POSITIVE CANDIDATES =====
pos_candidates = []
for r in rows:
    fname = r['dicom_id'].strip() + '.png'
    if r['Cardiomegaly'] != '1' or fname not in pos_imgs:
        continue
    
    report = get_report(r)
    if not report or len(report) < 30:
        continue
    
    rl = report.lower()
    
    # MUST mention cardiomegaly or enlarged heart explicitly
    has_cardio = bool(re.search(
        r'cardiomegaly|enlarged.{0,5}(heart|cardiac)|cardiac enlargement|'
        r'heart.{0,15}(enlarged|enlarge)|heart size.{0,15}(enlarged|increased|large)',
        rl
    ))
    if not has_cardio:
        continue
    
    # Must NOT be negated (e.g., "no cardiomegaly", "without cardiomegaly")
    negated = bool(re.search(
        r'no\s+(evidence\s+of\s+)?cardiomegaly|'
        r'no\s+cardiac\s+enlargement|'
        r'without\s+cardiomegaly|'
        r'cardiomegaly.{0,10}(absent|resolved|not)',
        rl
    ))
    if negated:
        continue
    
    has_effusion = int(r.get('Pleural_Effusion', '0'))
    has_edema = int(r.get('Edema', '0'))
    report_len = len(report)
    
    # Quality score
    score = 0
    score += 10 if 50 < report_len < 600 else 0  # good readable length
    score += 5 if 'heart size is enlarged' in rl or 'the heart is enlarged' in rl else 0
    score += 5 if 'cardiomegaly' in rl else 0
    score += 3 if 'normal' not in rl or 'heart size is normal' not in rl else 0
    # Slight preference for samples without too many other pathologies
    score -= 2 if has_effusion else 0
    score -= 2 if has_edema else 0
    # Penalize very short or very long reports
    score -= 5 if report_len > 600 else 0
    score -= 5 if report_len < 50 else 0
    
    pos_candidates.append({
        'dicom_id': r['dicom_id'].strip(),
        'fname': fname,
        'report': report,
        'score': score,
        'report_len': report_len,
        'edema': has_edema,
        'effusion': has_effusion,
    })

pos_candidates.sort(key=lambda x: x['score'], reverse=True)
print(f"=== POSITIVE CANDIDATES: {len(pos_candidates)} total ===")
for i, c in enumerate(pos_candidates[:15]):
    print(f"  {i+1}. score={c['score']} edema={c['edema']} eff={c['effusion']} len={c['report_len']}")
    print(f"     ID: {c['dicom_id']}")
    print(f"     REPORT: {c['report'][:250]}")
    print()

# ===== NEGATIVE CANDIDATES =====
neg_candidates = []
for r in rows:
    fname = r['dicom_id'].strip() + '.png'
    if r['Cardiomegaly'] != '0' or fname not in neg_imgs:
        continue
    
    has_edema = int(r.get('Edema', '0'))
    has_effusion = int(r.get('Pleural_Effusion', '0'))
    has_no_finding = int(r.get('No_Finding', '0'))
    has_atelectasis = int(r.get('Atelectasis', '0'))
    has_consolidation = int(r.get('Consolidation', '0'))
    has_pneumonia = int(r.get('Pneumonia', '0'))
    has_pneumothorax = int(r.get('Pneumothorax', '0'))
    has_opacity = int(r.get('Lung_Opacity', '0'))
    
    # MUST NOT have edema or effusion (hard requirement)
    if has_edema or has_effusion:
        continue
    
    report = get_report(r)
    if not report or len(report) < 30:
        continue
    
    rl = report.lower()
    
    # Report must NOT mention effusion positively
    eff_pos = re.search(r'effusion|pleural fluid', rl)
    eff_neg = re.search(r'no.{0,25}effusion|without.{0,20}effusion', rl)
    if eff_pos and not eff_neg:
        continue  # mentions effusion without negating it
    
    # Report must NOT mention edema positively
    ede_pos = re.search(r'edema|vascular congestion|fluid overload|cephalization', rl)
    ede_neg = re.search(r'no.{0,25}edema|without.{0,20}edema|no.{0,25}congestion', rl)
    if ede_pos and not ede_neg:
        continue
    
    # Report must NOT say cardiomegaly positively
    card_pos = re.search(r'cardiomegaly|enlarged.{0,5}heart|cardiac enlargement|heart.{0,10}enlarged', rl)
    card_neg = re.search(r'no.{0,25}cardiomegaly|normal.{0,15}heart|heart size.{0,15}normal|without.{0,20}cardiomegaly', rl)
    if card_pos and not card_neg:
        continue
    
    report_len = len(report)
    
    score = 0
    score += 15 if has_no_finding else 0  # cleanest samples
    score += 10 if 50 < report_len < 500 else 0
    score += 5 if 'normal heart size' in rl or 'heart size is normal' in rl or 'heart size normal' in rl else 0
    score += 5 if 'no acute' in rl or 'lungs are clear' in rl else 0
    score += 3 if 'unremarkable' in rl or 'no acute cardiopulmonary' in rl else 0
    # Penalize other pathologies
    score -= 5 if has_atelectasis else 0
    score -= 5 if has_consolidation else 0
    score -= 5 if has_pneumonia else 0
    score -= 5 if has_pneumothorax else 0
    score -= 5 if has_opacity else 0
    # Penalize very long/short reports
    score -= 5 if report_len > 500 else 0
    score -= 5 if report_len < 50 else 0
    
    neg_candidates.append({
        'dicom_id': r['dicom_id'].strip(),
        'fname': fname,
        'report': report,
        'score': score,
        'report_len': report_len,
        'no_finding': has_no_finding,
        'atelectasis': has_atelectasis,
        'consolidation': has_consolidation,
        'opacity': has_opacity,
    })

neg_candidates.sort(key=lambda x: x['score'], reverse=True)
print(f"\n=== NEGATIVE CANDIDATES: {len(neg_candidates)} total ===")
for i, c in enumerate(neg_candidates[:15]):
    print(f"  {i+1}. score={c['score']} no_finding={c['no_finding']} atel={c['atelectasis']} cons={c['consolidation']} opac={c['opacity']} len={c['report_len']}")
    print(f"     ID: {c['dicom_id']}")
    print(f"     REPORT: {c['report'][:250]}")
    print()

# Save candidates for use by selector
with open(os.path.join(os.path.dirname(__file__), 'pos_candidates.json'), 'w') as f:
    json.dump(pos_candidates[:50], f, indent=2)
with open(os.path.join(os.path.dirname(__file__), 'neg_candidates.json'), 'w') as f:
    json.dump(neg_candidates[:50], f, indent=2)
    
print("Saved top 50 candidates each to pos_candidates.json and neg_candidates.json")
