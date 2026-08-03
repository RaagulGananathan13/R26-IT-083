import os
import csv
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms, models
from PIL import Image
import numpy as np
import cv2
import base64
from transformers import BartTokenizer, BartForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput

# --- Config ---
CKPT_DIR_IMG = r'C:\Users\94775\Desktop\Component_01\models\cardio_classifier'
CKPT_DIR_REP = r'C:\Users\94775\Desktop\Component_01\models\report_generator'
CSV_DIR = '../cardiomegaly_dataset'
IMG_SIZE = 384
NUM_VISUAL = 144  # 12x12 spatial features from ConvNeXt
BART_MODEL = 'facebook/bart-base'
LABEL_COLS = [
    'Cardiomegaly', 'Edema', 'Pleural_Effusion',
    'Atelectasis', 'Consolidation', 'Lung_Opacity',
    'Pneumonia', 'Pneumothorax',
]
NUM_LABELS = len(LABEL_COLS)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Backend running on: {device}")

# --- Transforms ---
img_tf = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# --- Model 1: Image Classifier (Multi-label, torchvision ConvNeXt-Base) ---
class CardioConvNeXt(nn.Module):
    def __init__(self, num_labels=NUM_LABELS):
        super().__init__()
        base = models.convnext_base(weights=None)
        self.features = base.features        # GradCAM hooks here
        self.avgpool = base.avgpool
        self.classifier = nn.Sequential(
            nn.LayerNorm(1024),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_labels),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return self.classifier(x.flatten(1))


# --- Model 2: Report Generator (ConvNeXt vision + BART decoder) ---
class CXRReportGenerator(nn.Module):
    def __init__(self, classifier_path, bart_name=BART_MODEL):
        super().__init__()
        # Vision encoder (frozen ConvNeXt features)
        self.vision = self._load_vision(classifier_path)
        for p in self.vision.parameters():
            p.requires_grad = False
        self.vision.eval()

        # Projection: 1024 -> BART dim (768)
        self.vision_proj = nn.Sequential(
            nn.Linear(1024, 768),
            nn.LayerNorm(768),
            nn.GELU(),
            nn.Dropout(0.1),
        )

        # BART decoder
        self.bart = BartForConditionalGeneration.from_pretrained(bart_name)

    def _load_vision(self, path):
        base = models.convnext_base(weights=None)
        features = base.features
        if os.path.exists(str(path)):
            ckpt = torch.load(str(path), map_location='cpu', weights_only=False)
            feat_state = {
                k.replace('features.', ''): v
                for k, v in ckpt['model'].items() if k.startswith('features.')
            }
            features.load_state_dict(feat_state)
            print('  -> Vision encoder loaded from classifier checkpoint.')
        return features

    def _encode_vision(self, images):
        with torch.no_grad():
            feats = self.vision(images)              # (B, 1024, 12, 12)
        B = feats.shape[0]
        feats = feats.flatten(2).transpose(1, 2)     # (B, 144, 1024)
        feats = self.vision_proj(feats)               # (B, 144, 768)
        return feats

    @torch.inference_mode()
    def generate_report(self, images, tokenizer, fast=True):
        """Generate reports. fast=True uses greedy, False uses beam search."""
        self.eval()
        vis = self._encode_vision(images)
        enc_mask = torch.ones(vis.shape[0], NUM_VISUAL,
                              device=vis.device, dtype=torch.long)

        gen_kwargs = dict(
            encoder_outputs=BaseModelOutput(last_hidden_state=vis),
            attention_mask=enc_mask,
            max_length=200 if fast else 512,
            no_repeat_ngram_size=3,
        )
        if fast:
            gen_kwargs['num_beams'] = 1
            gen_kwargs['do_sample'] = False
        else:
            gen_kwargs['num_beams'] = 4
            gen_kwargs['early_stopping'] = True
            gen_kwargs['length_penalty'] = 1.0

        ids = self.bart.generate(**gen_kwargs)
        texts = tokenizer.batch_decode(ids, skip_special_tokens=True)
        return texts[0] if len(texts) == 1 else texts


def clean_report(text):
    """Remove training-data artifacts: comparisons to prior exams and administrative notes."""
    import re
    
    # === PHASE 1: Remove inline clauses referencing prior exams ===
    # These appear mid-sentence like "...which was not present on prior CT from _ _ _."
    inline_patterns = [
        r',?\s*which was not (present|seen|noted|evident) on (prior|previous)[\w\s]*from\s*_[\s_]*',
        r',?\s*which (has|have) (improved|worsened|changed|resolved) since (prior|previous)[\w\s]*',
        r',?\s*(?:as |when )?compared (?:to|with) (?:the )?(?:prior|previous)[\w\s,]*',
        r',?\s*(?:unchanged|stable|similar) (?:from|since|compared to) (?:the )?(?:prior|previous)[\w\s,]*',
        r',?\s*(?:new|increased|decreased) (?:since|from|compared to) (?:the )?(?:prior|previous)[\w\s,]*',
        r'\s*from\s+_[\s_]+',  # dangling "from _ _ _"
        r'\s*on\s+_[\s_]+',    # dangling "on _ _ _"
    ]
    for pattern in inline_patterns:
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)
    
    # === PHASE 2: Remove entire sentences matching bad patterns ===
    sentences = re.split(r'(?<=[.!?])\s+', text)
    
    remove_patterns = [
        # Comparative phrases (inappropriate for new patients)
        r'the monitoring and support devices are constant',
        r'as compared to',
        r'compared to the previous',
        r'compared with the previous',
        r'in comparison with',
        r'in comparison to',
        r'since the prior',
        r'since the previous',
        r'since the last',
        r'from the prior',
        r'from the previous',
        r'no relevant change',
        r'no significant change',
        r'unchanged from',
        r'stable since',
        r'interval change',
        r'interval improvement',
        r'interval worsening',
        r'previously (seen|noted|described)',
        r'prior (exam|study|examination|radiograph|film)',
        r'previous (exam|study|examination|radiograph|film)',
        r'prior ct ',
        r'prior chest',
        
        # Administrative / dictation notes
        r'at the time of dictation',
        r'time of observation',
        r'referring physician was (paged|notified|called|contacted)',
        r'findings were (discussed|communicated|reported|subsequently)',
        r'discussed over the telephone',
        r'paged for notification',
        r'\d{1,2}\s*:\s*\d{2}\s*(a\.?m\.?|p\.?m\.?)',
        r'on\s+_\s*_\s*_',
        r'wet read',
        r'addendum',
        r'preliminary report',
        r'final report',
    ]
    
    cleaned = []
    for sentence in sentences:
        s = sentence.strip()
        if not s:
            continue
        should_remove = False
        for pattern in remove_patterns:
            if re.search(pattern, s, re.IGNORECASE):
                should_remove = True
                break
        if not should_remove:
            cleaned.append(s)
    
    result = ' '.join(cleaned)
    # Clean up any double spaces or trailing commas before periods
    result = re.sub(r'\s+', ' ', result)
    result = re.sub(r',\s*\.', '.', result)
    return result.strip()

# --- Globals ---
img_model = None
rep_model = None
tokenizer = None
csv_lookup = {}  # dicom_id -> report_text from CSVs

def _load_csv(path):
    """Load a CSV and return {dicom_id: report_text} dict."""
    csv.field_size_limit(10 * 1024 * 1024)
    lookup = {}
    with open(path, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            did = row['dicom_id'].strip()
            text = (row.get('report_text', '') or '').strip()
            if not text:
                text = (row.get('findings_text', '') or '').strip()
            if not text:
                text = (row.get('impression_text', '') or '').strip()
            if text:
                lookup[did] = text
    return lookup

def load_models():
    global img_model, rep_model, tokenizer, csv_lookup
    
    # Load ground-truth CSVs
    print("Loading ground-truth CSVs...")
    for csv_name in ['cardio_train.csv', 'cardio_val.csv', 'cardio_test.csv']:
        csv_path = os.path.join(CSV_DIR, csv_name)
        if os.path.exists(csv_path):
            lk = _load_csv(csv_path)
            csv_lookup.update(lk)
            print(f"  -> {csv_name}: {len(lk)} reports loaded")
        else:
            print(f"  -> WARNING: {csv_path} not found, skipping")
    print(f"  -> Total ground-truth reports: {len(csv_lookup)}")
    
    print("Loading BART tokenizer...")
    tokenizer = BartTokenizer.from_pretrained(BART_MODEL)

    print("Loading Model 1 (Multi-label Classifier)...")
    img_model = CardioConvNeXt(NUM_LABELS).to(device)
    img_ckpt_path = os.path.join(CKPT_DIR_IMG, 'best_model.pt')
    if os.path.exists(img_ckpt_path):
        img_model.load_state_dict(torch.load(img_ckpt_path, map_location=device, weights_only=False)['model'])
        print("  -> Model 1 loaded successfully.")
    else:
        print(f"  -> WARNING: Could not find {img_ckpt_path}. Using untrained weights.")

    print("Loading Model 2 (Report Generator - ConvNeXt + BART)...")
    classifier_ckpt = os.path.join(CKPT_DIR_IMG, 'best_model.pt')
    rep_model = CXRReportGenerator(classifier_ckpt, BART_MODEL).to(device)
    rep_ckpt_path = os.path.join(CKPT_DIR_REP, 'best_model.pt')
    if os.path.exists(rep_ckpt_path):
        rep_model.load_state_dict(torch.load(rep_ckpt_path, map_location=device, weights_only=False)['model'])
        print("  -> Model 2 loaded successfully.")
    else:
        print(f"  -> WARNING: Could not find {rep_ckpt_path}. Using untrained weights.")

def generate_gradcam(model, image_tensor, original_image_np):
    """GradCAM using the last stage of torchvision ConvNeXt (features[7])."""
    model.eval()
    model.zero_grad()
    
    features, grads = [], []
    
    def fwd_hook(module, inp, out):
        features.append(out.detach())
    def bwd_hook(module, grad_in, grad_out):
        grads.append(grad_out[0].detach())
    
    # Hook into the last stage of torchvision ConvNeXt features
    target_layer = model.features[7]
    h1 = target_layer.register_forward_hook(fwd_hook)
    h2 = target_layer.register_full_backward_hook(bwd_hook)
    
    with torch.enable_grad():
        output = model(image_tensor)
        # Multi-label: use sigmoid, target Cardiomegaly (index 0)
        probs = torch.sigmoid(output)
        cardio_prob = probs[0, 0]  # Cardiomegaly is LABEL_COLS[0]
        model.zero_grad()
        cardio_prob.backward()
    
    h1.remove()
    h2.remove()
    
    # Multi-label predictions for all pathologies
    all_probs = probs[0].detach().cpu().numpy()
    pred_class = int(all_probs[0] >= 0.5)  # Cardiomegaly threshold
    confidence = float(all_probs[0])
    
    if not grads or not features:
        h, w = original_image_np.shape[:2]
        blank = np.zeros((h, w), dtype=np.uint8)
        heatmap = cv2.applyColorMap(blank, cv2.COLORMAP_JET)
        _, buffer = cv2.imencode('.png', heatmap)
        return pred_class, confidence, base64.b64encode(buffer).decode('utf-8'), all_probs
    
    # Proper GradCAM computation
    w = grads[0].mean(dim=[2, 3], keepdim=True)
    cam = F.relu((w * features[0]).sum(dim=1, keepdim=True))
    cam = F.interpolate(cam, size=(original_image_np.shape[0], original_image_np.shape[1]), 
                        mode='bilinear', align_corners=False)
    cam = cam.squeeze().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    
    heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
    
    _, buffer = cv2.imencode('.png', heatmap)
    heatmap_b64 = base64.b64encode(buffer).decode('utf-8')
    
    return pred_class, confidence, heatmap_b64, all_probs

def detect_copathologies(report_text):
    """Detect co-pathologies from the generated report text using sentence-level
    negation-aware NLP. Handles comma-separated negation lists like
    'no consolidation, effusion, or pneumothorax' correctly."""
    import re
    text = report_text.lower()
    
    # Split into sentences for context-aware negation
    sentences = re.split(r'[.!?]\s+|\n', text)
    
    # Negation cues that apply to everything following in the same clause
    NEG_CUES = [r'\bno\b', r'\bwithout\b', r'\bnot\b', r'\bdenies\b',
                r'\babsent\b', r'\bnegative for\b', r'\bfree of\b',
                r'\brule(?:d)? out\b', r'\bresolved\b', r'\bcleared\b']
    
    def _is_negated_in_sentence(sentence, keyword_match_start):
        """Check if a negation cue precedes the keyword in the same sentence/clause."""
        prefix = sentence[:keyword_match_start]
        for cue in NEG_CUES:
            if re.search(cue, prefix):
                return True
        return False
    
    pathology_keywords = {
        "Pleural Effusion": [r'pleural effusion', r'pleural fluid', r'effusion'],
        "Pulmonary Edema":  [r'pulmonary edema', r'edema', r'vascular congestion',
                             r'fluid overload', r'interstitial edema', r'cephalization'],
        "Atelectasis":      [r'atelectasis', r'atelectatic', r'volume loss'],
        "Consolidation":    [r'consolidation', r'consolidative'],
        "Lung Opacity":     [r'opacit(?:y|ies)', r'opacification', r'infiltrate'],
        "Pneumonia":        [r'pneumonia', r'infectious process'],
        "Pneumothorax":     [r'pneumothorax'],
    }
    
    # Additional whole-text "clear" phrases that indicate absence of everything
    global_clear = re.search(r'lungs are clear|clear lungs|no acute cardiopulmonary', text)
    
    results = []
    for name, keywords in pathology_keywords.items():
        found_positive = False  # un-negated mention
        found_negative = False  # negated mention
        found_any = False       # any mention at all
        
        for sentence in sentences:
            for kw in keywords:
                for match in re.finditer(kw, sentence):
                    found_any = True
                    if _is_negated_in_sentence(sentence, match.start()):
                        found_negative = True
                    else:
                        found_positive = True
        
        if found_positive and not found_negative:
            status = "present"
        elif found_negative:
            # Any negated mention → treat as absent (trust the negation)
            status = "absent"
        elif not found_any:
            status = "not_mentioned"
        else:
            status = "absent"
        
        results.append({"name": name, "status": status})
    
    return results


def run_inference(image_bytes, filename=""):
    # 1. Prepare image
    from io import BytesIO
    img = Image.open(BytesIO(image_bytes)).convert('RGB')
    
    # Original for GradCAM overlay sizing
    original_np = np.array(img)
    
    # Tensor for models
    img_tensor = img_tf(img).unsqueeze(0).to(device)
    
    # 2. Run Classification + GradCAM (Model 1) — multi-label
    pred_class, confidence, heatmap_b64, all_probs = generate_gradcam(img_model, img_tensor, original_np)
    
    label = 'Cardiomegaly' if pred_class == 1 else 'Negative'
    
    # Build multi-label pathology results from classifier
    classifier_pathologies = []
    for i, col in enumerate(LABEL_COLS):
        classifier_pathologies.append({
            'name': col.replace('_', ' '),
            'probability': float(all_probs[i]),
            'status': 'present' if all_probs[i] >= 0.5 else 'absent'
        })
    
    # 3. Run Report Generation (Model 2 — BART)
    raw_report = rep_model.generate_report(img_tensor, tokenizer, fast=True)
    
    # Clean the report (remove comparisons & admin notes)
    cleaned_report = clean_report(raw_report)
    last_period = cleaned_report.rfind('.')
    if last_period > 10:
        cleaned_report = cleaned_report[:last_period + 1]
    
    final_report = cleaned_report.strip() if cleaned_report.strip() else raw_report
    
    # 4. Detect co-pathologies from report text (NLP-based)
    copathologies_nlp = detect_copathologies(final_report)
    
    # 5. Merge: use classifier probabilities as primary, NLP as supplementary
    copathologies = classifier_pathologies[1:]  # Skip Cardiomegaly (already primary)
    
    # 6. Look up ground truth from CSV (by filename/dicom_id)
    dicom_id = os.path.splitext(filename)[0] if filename else ""
    ground_truth = csv_lookup.get(dicom_id, None)
    
    result = {
        "prediction": label,
        "confidence": confidence,
        "gradcam_image": heatmap_b64,
        "report_text": final_report,
        "report_text_raw": raw_report,
        "copathologies": copathologies,
    }
    if ground_truth:
        result["ground_truth_report"] = ground_truth
    
    return result
