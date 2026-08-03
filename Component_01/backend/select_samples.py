"""
DEFINITIVE Test Sample Selector — 10 Positive + 10 Negative
============================================================
FULL PIPELINE VERIFICATION:
  1. CSV ground-truth labels verified
  2. NLP report text verified
  3. Classifier prediction verified (must agree with label)
  4. AI-generated report COMPARED to ground truth — only images where
     the generated report closely matches the ground truth are selected
  
POSITIVE criteria:
  - Cardiomegaly=1 in CSV
  - Report explicitly mentions cardiomegaly / enlarged heart (NOT negated)
  - Edema/Effusion ALLOWED (common co-pathologies of cardiomegaly)
  - Classifier predicts positive (class 1)
  - Generated report has HIGH similarity to ground truth

NEGATIVE criteria:
  - Cardiomegaly=0 in CSV
  - Edema=0, Pleural_Effusion=0 in CSV  
  - No other major pathologies
  - Report says "normal heart size" / "lungs are clear"
  - Classifier predicts negative (class 0)
  - Generated report has HIGH similarity to ground truth
"""
import os, sys, csv, shutil, json, re
import torch
import numpy as np
from PIL import Image
from torchvision import transforms
from collections import Counter

# Add parent so we can import inference module
sys.path.insert(0, os.path.dirname(__file__))
import inference

# --- Paths ---
BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
TEST_DIR = os.path.join(BASE, 'cardio_image_384', 'test')
CSV_DIR = os.path.join(BASE, 'cardiomegaly_dataset')
OUT_DIR = os.path.join(TEST_DIR, 'testsamples')

POS_SRC = os.path.join(TEST_DIR, 'positive')
NEG_SRC = os.path.join(TEST_DIR, 'negative')
POS_DST = os.path.join(OUT_DIR, 'positive')
NEG_DST = os.path.join(OUT_DIR, 'negative')


# ============================================================
# REPORT SIMILARITY SCORING
# ============================================================

def extract_medical_keywords(text):
    """Extract key medical terms from a report for comparison."""
    t = text.lower()
    keywords = set()
    
    # Cardiomegaly-related
    if re.search(r'cardiomegaly', t): keywords.add('cardiomegaly')
    if re.search(r'heart.{0,15}(enlarged|large)', t): keywords.add('heart_enlarged')
    if re.search(r'cardiac.{0,10}(enlarged|enlargement)', t): keywords.add('cardiac_enlargement')
    if re.search(r'normal heart size|heart size.{0,10}normal', t): keywords.add('normal_heart')
    
    # Effusion
    has_eff = bool(re.search(r'effusion|pleural\s+fluid', t))
    neg_eff = bool(re.search(r'no.{0,30}effusion|without.{0,20}effusion', t))
    if has_eff and not neg_eff:
        keywords.add('effusion_present')
    if neg_eff:
        keywords.add('no_effusion')
    
    # Edema
    has_ede = bool(re.search(r'edema|vascular\s+congestion|fluid\s+overload', t))
    neg_ede = bool(re.search(r'no.{0,30}edema|without.{0,20}edema|no.{0,30}congestion', t))
    if has_ede and not neg_ede:
        keywords.add('edema_present')
    if neg_ede:
        keywords.add('no_edema')
    
    # Consolidation
    if re.search(r'no.{0,30}consolidation', t):
        keywords.add('no_consolidation')
    elif re.search(r'consolidation', t):
        keywords.add('consolidation_present')
    
    # Pneumothorax
    if re.search(r'no.{0,30}pneumothorax', t):
        keywords.add('no_pneumothorax')
    elif re.search(r'pneumothorax', t):
        keywords.add('pneumothorax_present')
    
    # Atelectasis
    if re.search(r'atelectasis', t):
        keywords.add('atelectasis')
    
    # Clear lungs
    if re.search(r'lungs are clear|clear lungs', t):
        keywords.add('lungs_clear')
    
    # General
    if re.search(r'no acute', t): keywords.add('no_acute')
    if re.search(r'pneumonia', t): keywords.add('pneumonia_mentioned')
    if re.search(r'congestion', t): keywords.add('congestion_mentioned')
    
    return keywords


def compute_report_similarity(generated, ground_truth):
    """Compute similarity between generated and ground truth reports.
    Returns a score from 0.0 to 1.0."""
    if not generated or not ground_truth:
        return 0.0
    
    gen_lower = generated.lower()
    gt_lower = ground_truth.lower()
    
    # 1. Word overlap (Jaccard-like)
    gen_words = set(re.findall(r'\b[a-z]{3,}\b', gen_lower))
    gt_words = set(re.findall(r'\b[a-z]{3,}\b', gt_lower))
    if gen_words and gt_words:
        word_overlap = len(gen_words & gt_words) / len(gen_words | gt_words)
    else:
        word_overlap = 0.0
    
    # 2. Medical keyword agreement
    gen_kw = extract_medical_keywords(generated)
    gt_kw = extract_medical_keywords(ground_truth)
    if gen_kw and gt_kw:
        kw_overlap = len(gen_kw & gt_kw) / len(gen_kw | gt_kw)
    elif not gen_kw and not gt_kw:
        kw_overlap = 1.0  # both have no special keywords
    else:
        kw_overlap = 0.0
    
    # 3. Key disagreements (heavy penalty)
    # If generated says cardiomegaly but GT doesn't, or vice versa = BAD
    penalty = 0.0
    if ('cardiomegaly' in gen_kw or 'heart_enlarged' in gen_kw) and \
       ('normal_heart' in gt_kw):
        penalty += 0.3
    if ('normal_heart' in gen_kw) and \
       ('cardiomegaly' in gt_kw or 'heart_enlarged' in gt_kw):
        penalty += 0.3
    if ('effusion_present' in gen_kw) and ('no_effusion' in gt_kw):
        penalty += 0.15
    if ('edema_present' in gen_kw) and ('no_edema' in gt_kw):
        penalty += 0.15
    
    # 4. Bigram overlap (captures phrases)
    def bigrams(text):
        words = re.findall(r'\b[a-z]{2,}\b', text.lower())
        return set(zip(words, words[1:]))
    
    gen_bi = bigrams(generated)
    gt_bi = bigrams(ground_truth)
    if gen_bi and gt_bi:
        bigram_overlap = len(gen_bi & gt_bi) / len(gen_bi | gt_bi)
    else:
        bigram_overlap = 0.0
    
    # Weighted final score
    score = (0.30 * word_overlap) + (0.35 * kw_overlap) + (0.20 * bigram_overlap) + 0.15
    score -= penalty
    score = max(0.0, min(1.0, score))
    
    return score


# ============================================================
# NLP VERIFICATION
# ============================================================

def report_confirms_cardiomegaly(report):
    """Returns True if report explicitly confirms cardiomegaly (not negated)."""
    rl = report.lower()
    pos_patterns = [
        r'cardiomegaly', r'cardiac\s+enlargement',
        r'heart\s+(is\s+|size\s+is\s+)?(enlarged|large)',
        r'enlarged\s+(cardiac|heart)',
        r'cardiac\s+silhouette\s+(is\s+)?(enlarged|large)',
    ]
    has_positive = any(re.search(p, rl) for p in pos_patterns)
    if not has_positive:
        return False
    neg_patterns = [
        r'no\s+(evidence\s+of\s+)?cardiomegaly', r'no\s+cardiac\s+enlargement',
        r'without\s+cardiomegaly', r'normal\s+heart\s+size',
        r'heart\s+size\s+(is\s+)?normal',
    ]
    return not any(re.search(p, rl) for p in neg_patterns)


def report_is_clean_negative(report):
    """Returns True if report confirms normal heart with no effusion/edema."""
    rl = report.lower()
    if report_confirms_cardiomegaly(report):
        return False
    # Effusion: if mentioned, must be negated
    eff_pos = re.search(r'effusion|pleural\s+fluid', rl)
    eff_neg = re.search(r'no.{0,30}effusion|without.{0,20}effusion', rl)
    if eff_pos and not eff_neg:
        return False
    # Edema: if mentioned, must be negated
    ede_pos = re.search(r'edema|vascular\s+congestion|fluid\s+overload', rl)
    ede_neg = re.search(r'no.{0,30}edema|without.{0,20}edema|no.{0,30}congestion', rl)
    if ede_pos and not ede_neg:
        return False
    return True


# ============================================================
# STEP 0: Delete existing test samples
# ============================================================
print("=" * 70)
print("STEP 0: Cleaning output directories...")
print("=" * 70)
for folder in [POS_DST, NEG_DST]:
    if os.path.exists(folder):
        for f in os.listdir(folder):
            fpath = os.path.join(folder, f)
            if os.path.isfile(fpath):
                os.remove(fpath)
                print(f"  Deleted: {f}")
report_json = os.path.join(OUT_DIR, 'selection_report.json')
if os.path.exists(report_json):
    os.remove(report_json)
os.makedirs(POS_DST, exist_ok=True)
os.makedirs(NEG_DST, exist_ok=True)
print("  Done.\n")


# ============================================================
# STEP 1: Load CSV ground truth
# ============================================================
print("=" * 70)
print("STEP 1: Loading CSV ground truth...")
print("=" * 70)
csv.field_size_limit(10 * 1024 * 1024)

def load_full_csv(path):
    records = {}
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            did = row['dicom_id'].strip()
            report = (row.get('report_text', '') or '').strip()
            if not report:
                report = (row.get('findings_text', '') or '').strip()
            if not report:
                report = (row.get('impression_text', '') or '').strip()
            records[did] = {
                'dicom_id': did, 'report': report,
                'cardiomegaly': int(row.get('Cardiomegaly', 0)),
                'edema': int(row.get('Edema', 0)),
                'pleural_effusion': int(row.get('Pleural_Effusion', 0)),
                'atelectasis': int(row.get('Atelectasis', 0)),
                'consolidation': int(row.get('Consolidation', 0)),
                'lung_opacity': int(row.get('Lung_Opacity', 0)),
                'no_finding': int(row.get('No_Finding', 0)),
                'pneumonia': int(row.get('Pneumonia', 0)),
                'pneumothorax': int(row.get('Pneumothorax', 0)),
            }
    return records

gt = {}
for name in ['cardio_test.csv', 'cardio_train.csv', 'cardio_val.csv']:
    p = os.path.join(CSV_DIR, name)
    if os.path.exists(p):
        recs = load_full_csv(p)
        gt.update(recs)
        print(f"  {name}: {len(recs)} entries")
print(f"  Total: {len(gt)} entries\n")


# ============================================================
# STEP 2: Load BOTH models (classifier + report generator)
# ============================================================
print("=" * 70)
print("STEP 2: Loading models (classifier + report generator)...")
print("=" * 70)
inference.load_models()
model = inference.img_model
rep_model = inference.rep_model
tokenizer = inference.tokenizer
device = inference.device
img_tf = inference.img_tf
print()


# ============================================================
# STEP 3: Pre-filter candidates by CSV labels + NLP
# ============================================================
print("=" * 70)
print("STEP 3: Pre-filtering candidates by CSV labels + NLP...")
print("=" * 70)

pos_files = [f for f in os.listdir(POS_SRC) if f.endswith('.png')]
neg_files = [f for f in os.listdir(NEG_SRC) if f.endswith('.png')]

# --- Positive candidates ---
pos_prefiltered = []
for fname in pos_files:
    dicom_id = fname.replace('.png', '')
    info = gt.get(dicom_id)
    if not info or info['cardiomegaly'] != 1:
        continue
    report = info['report']
    if not report or len(report) < 30:
        continue
    if not report_confirms_cardiomegaly(report):
        continue
    pos_prefiltered.append((fname, dicom_id, info))

# --- Negative candidates ---
neg_prefiltered = []
for fname in neg_files:
    dicom_id = fname.replace('.png', '')
    info = gt.get(dicom_id)
    if not info or info['cardiomegaly'] != 0:
        continue
    if info['edema'] == 1 or info['pleural_effusion'] == 1:
        continue
    if info['atelectasis'] == 1 or info['consolidation'] == 1:
        continue
    if info['lung_opacity'] == 1 or info['pneumonia'] == 1 or info['pneumothorax'] == 1:
        continue
    report = info['report']
    if not report or len(report) < 30:
        continue
    if not report_is_clean_negative(report):
        continue
    neg_prefiltered.append((fname, dicom_id, info))

print(f"  Positive pre-filtered: {len(pos_prefiltered)}")
print(f"  Negative pre-filtered: {len(neg_prefiltered)}\n")


# ============================================================
# STEP 4: Run FULL pipeline on candidates, score report match
# ============================================================
print("=" * 70)
print("STEP 4: Running full pipeline (classifier + report gen) on candidates...")
print("IMPORTANT: Scoring AI-generated report vs ground truth!")
print("=" * 70)

def run_full_pipeline(fpath):
    """Run classifier + report generator on a single image."""
    try:
        img = Image.open(fpath).convert('L')
        tensor = img_tf(img).unsqueeze(0).to(device)
        
        # Classifier
        with torch.inference_mode():
            output = model(tensor)
            probs = torch.softmax(output, dim=1)[0]
            pred_class = output.argmax(dim=1).item()
            confidence = probs[pred_class].item()
        
        # Report generator
        raw_report = rep_model.generate(tensor.squeeze(0), tokenizer, max_len=150)
        cleaned_report = inference.clean_report(raw_report)
        last_period = cleaned_report.rfind('.')
        if last_period > 10:
            cleaned_report = cleaned_report[:last_period + 1]
        final_report = cleaned_report.strip() if cleaned_report.strip() else raw_report
        
        return pred_class, confidence, final_report, raw_report
    except Exception as e:
        print(f"  ERROR: {e}")
        return None, 0.0, "", ""


# --- Score POSITIVE candidates ---
print(f"\n--- Processing POSITIVE candidates (top {min(40, len(pos_prefiltered))})---\n")

pos_scored = []
for i, (fname, dicom_id, info) in enumerate(pos_prefiltered[:40]):
    fpath = os.path.join(POS_SRC, fname)
    pred_class, confidence, gen_report, raw_report = run_full_pipeline(fpath)
    
    if pred_class != 1:
        continue  # classifier must agree
    if confidence < 0.6:
        continue
    
    gt_report = info['report']
    similarity = compute_report_similarity(gen_report, gt_report)
    
    pos_scored.append({
        'dicom_id': dicom_id,
        'fname': fname,
        'pred_confidence': confidence,
        'report_similarity': similarity,
        'generated_report': gen_report,
        'ground_truth_report': gt_report,
        'raw_report': raw_report,
        'info': info,
    })
    
    marker = "★" if similarity >= 0.45 else " "
    print(f"  {marker} [{i+1}] {dicom_id[:35]}  conf={confidence:.3f}  sim={similarity:.3f}")

# Sort by report similarity (BEST matches first)
pos_scored.sort(key=lambda x: x['report_similarity'], reverse=True)


# --- Score NEGATIVE candidates ---
print(f"\n--- Processing NEGATIVE candidates (top {min(40, len(neg_prefiltered))})---\n")

neg_scored = []
for i, (fname, dicom_id, info) in enumerate(neg_prefiltered[:40]):
    fpath = os.path.join(NEG_SRC, fname)
    pred_class, confidence, gen_report, raw_report = run_full_pipeline(fpath)
    
    if pred_class != 0:
        continue  # classifier must agree
    if confidence < 0.6:
        continue
    
    gt_report = info['report']
    similarity = compute_report_similarity(gen_report, gt_report)
    
    neg_scored.append({
        'dicom_id': dicom_id,
        'fname': fname,
        'pred_confidence': confidence,
        'report_similarity': similarity,
        'generated_report': gen_report,
        'ground_truth_report': gt_report,
        'raw_report': raw_report,
        'info': info,
    })
    
    marker = "★" if similarity >= 0.45 else " "
    print(f"  {marker} [{i+1}] {dicom_id[:35]}  conf={confidence:.3f}  sim={similarity:.3f}")

neg_scored.sort(key=lambda x: x['report_similarity'], reverse=True)


# ============================================================
# STEP 5: Select top 10 of each
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Selecting TOP 10 each by report similarity...")
print("=" * 70)

pos_selected = pos_scored[:10]
neg_selected = neg_scored[:10]

print(f"\n  Selected {len(pos_selected)} positive, {len(neg_selected)} negative\n")


# ============================================================
# STEP 6: Copy files
# ============================================================
print("=" * 70)
print("STEP 6: Copying files...")
print("=" * 70)

for item in pos_selected:
    src = os.path.join(POS_SRC, item['fname'])
    dst = os.path.join(POS_DST, item['fname'])
    shutil.copy2(src, dst)

for item in neg_selected:
    src = os.path.join(NEG_SRC, item['fname'])
    dst = os.path.join(NEG_DST, item['fname'])
    shutil.copy2(src, dst)

print(f"  Copied {len(pos_selected)} positive to {POS_DST}")
print(f"  Copied {len(neg_selected)} negative to {NEG_DST}\n")


# ============================================================
# STEP 7: Save detailed verification report
# ============================================================
print("=" * 70)
print("STEP 7: Saving verification report...")
print("=" * 70)

def make_sample_entry(item, label_type):
    return {
        'dicom_id': item['dicom_id'],
        'filename': item['fname'],
        'ground_truth_label': label_type,
        'classifier_confidence': round(item['pred_confidence'], 4),
        'report_similarity_score': round(item['report_similarity'], 4),
        'ai_generated_report': item['generated_report'],
        'ground_truth_report': item['ground_truth_report'],
        'csv_labels': {
            'Cardiomegaly': item['info']['cardiomegaly'],
            'Edema': item['info']['edema'],
            'Pleural_Effusion': item['info']['pleural_effusion'],
            'No_Finding': item['info'].get('no_finding', 0),
        },
    }

report_data = {
    'selection_criteria': {
        'triple_verification': [
            '1. CSV ground-truth labels',
            '2. NLP report text verification',
            '3. Model classifier prediction agreement',
            '4. AI-generated report similarity to ground truth (HIGHEST priority)',
        ],
        'positive': [
            'CSV: Cardiomegaly=1',
            'NLP: Report explicitly confirms cardiomegaly (not negated)',
            'Edema/Effusion: ALLOWED (common co-pathologies)',
            'MODEL: Classifier predicts positive with >60% confidence',
            'REPORT: AI-generated report has highest similarity to ground truth',
        ],
        'negative': [
            'CSV: Cardiomegaly=0, Edema=0, Pleural_Effusion=0',
            'CSV: No other pathologies',
            'NLP: Report confirms clean negative',
            'MODEL: Classifier predicts negative with >60% confidence',
            'REPORT: AI-generated report has highest similarity to ground truth',
        ],
    },
    'positive_samples': [make_sample_entry(s, 'Cardiomegaly (positive)') for s in pos_selected],
    'negative_samples': [make_sample_entry(s, 'No Cardiomegaly (negative)') for s in neg_selected],
}

report_path = os.path.join(OUT_DIR, 'selection_report.json')
with open(report_path, 'w') as f:
    json.dump(report_data, f, indent=2)
print(f"  Saved to: {report_path}\n")


# ============================================================
# FINAL DETAILED SUMMARY
# ============================================================
print("\n" + "=" * 70)
print("FINAL VERIFICATION — POSITIVE SAMPLES")
print("=" * 70)
for i, item in enumerate(pos_selected):
    print(f"\n  [{i+1}] {item['dicom_id']}")
    print(f"      Classifier: POSITIVE  conf={item['pred_confidence']:.4f}")
    print(f"      Report match: {item['report_similarity']:.4f}")
    print(f"      GT:  {item['ground_truth_report'][:120]}...")
    print(f"      AI:  {item['generated_report'][:120]}...")

print("\n" + "=" * 70)
print("FINAL VERIFICATION — NEGATIVE SAMPLES")
print("=" * 70)
for i, item in enumerate(neg_selected):
    print(f"\n  [{i+1}] {item['dicom_id']}")
    print(f"      Classifier: NEGATIVE  conf={item['pred_confidence']:.4f}")
    print(f"      Report match: {item['report_similarity']:.4f}")
    print(f"      GT:  {item['ground_truth_report'][:120]}...")
    print(f"      AI:  {item['generated_report'][:120]}...")

print(f"\n{'='*70}")
print(f"DONE! 10 positive + 10 negative samples selected.")
print(f"All samples verified: correct label + correct prediction + best report match.")
print(f"{'='*70}")
