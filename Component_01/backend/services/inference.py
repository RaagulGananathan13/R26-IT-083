"""
Inference service — loads both models once, exposes predict().

Preserves the original API response shape so the existing frontend works
unchanged, and adds three fields: `view`, `reliability`, `threshold_source`.
"""
from __future__ import annotations

import io
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image

from backend import config as C
from backend.models.classifier import load_classifier
from backend.models.report_generator import load_report_generator
from backend.services.gradcam import GradCAM, overlay_heatmap, image_to_base64
from backend.services.thresholds import ThresholdPolicy
from backend.services.deferral import DeferralPolicy

# cxr_transforms lives in the project root. Importing it rather than
# redefining the transform here is deliberate: the original backend defined its
# own ImageNet normalisation, which does not match how these models were
# trained. A silent mismatch degrades every prediction with no error.
sys.path.insert(0, str(C.PROJECT_ROOT))
from cxr_transforms import build_transform            # noqa: E402


class InferenceService:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print("[service] device: %s" % self.device)

        self.transform = build_transform("test")       # per-image z-score
        self.policy = ThresholdPolicy(C.THRESHOLDS_JSON, C.PROJECTION_AUROC,
                                      C.PROJECTION_GAP)
        self.deferral = DeferralPolicy(C.DEFERRAL_POLICY_JSON)

        print("[service] loading classifier ...")
        self.classifier = load_classifier(C.CLASSIFIER_WEIGHTS, self.device,
                                          C.NUM_LABELS)
        self.gradcam = GradCAM(self.classifier, C.IMG_SIZE)

        if C.REPORTGEN_STAGE11.exists():
            path, self.reportgen_stage = C.REPORTGEN_STAGE11, "stage11"
        elif C.REPORTGEN_STAGE4.exists():
            path, self.reportgen_stage = C.REPORTGEN_STAGE4, "stage4"
            print("[service] !! Stage 11 checkpoint not found -- serving Stage 4. "
                  "Download checkpoints/stage11/best.pt from Drive.")
        else:
            raise FileNotFoundError(
                "No report generator found. Expected one of:\n  %s\n  %s"
                % (C.REPORTGEN_STAGE11, C.REPORTGEN_STAGE4))

        print("[service] loading report generator (%s) ..." % self.reportgen_stage)
        self.reportgen, self.tokenizer, self.rg_info = load_report_generator(
            path, C.DECODER_NAME, self.device)

        self.test_df = pd.read_csv(str(C.TEST_MANIFEST), low_memory=False)
        self.test_index = {str(d): i for i, d in enumerate(self.test_df.dicom_id)}

        # Original radiologist text, read-only.
        self.truth = {}
        if C.ORIGINAL_TEST_CSV.exists():
            od = pd.read_csv(str(C.ORIGINAL_TEST_CSV), low_memory=False)
            for _, r in od.iterrows():
                txt = str(r.get("report_text") or "").strip()
                if txt in ("", "nan", "None"):
                    txt = str(r.get("findings_text") or "").strip()
                if txt not in ("", "nan", "None"):
                    self.truth[str(r["dicom_id"])] = txt
            print("[service] ground truth indexed: %d original reports" % len(self.truth))
        else:
            print("[service] !! original dataset not found at %s -- "
                  "Ground Truth will be unavailable" % C.ORIGINAL_TEST_CSV)
        print("[service] ready -- %d test samples indexed" % len(self.test_index))

    def lookup_ground_truth(self, filename: str | None) -> str | None:
        """Match an uploaded file to the original radiologist report.

        Images exported by extract_review_cases.py are named <dicom_id>.png, so
        the stem is the key. An arbitrary upload will not match, and returning
        None is correct -- the UI then says no ground truth exists rather than
        showing someone else's report.
        """
        if not filename:
            return None
        stem = Path(filename).stem.strip()
        return self.truth.get(stem)

    # ------------------------------------------------------------------
    def _prepare(self, image_bytes: bytes):
        pil = Image.open(io.BytesIO(image_bytes))
        x = self.transform(pil).unsqueeze(0).to(self.device)
        # Display copy comes from the ORIGINAL pixels, never the z-scored
        # tensor -- z-scoring destroys absolute intensity by design, and an
        # overlay built from it would not look like the radiograph.
        disp = np.array(pil.convert("L").resize((C.IMG_SIZE, C.IMG_SIZE)))
        import cv2
        return x, cv2.cvtColor(disp, cv2.COLOR_GRAY2BGR)

    def predict(self, image_bytes: bytes, view: str | None = None,
                filename: str | None = None) -> dict:
        """Response shape matches the Component_1 UI contract, plus the
        projection fields this deployment adds."""
        x, img_bgr = self._prepare(image_bytes)

        with torch.no_grad():
            probs = torch.sigmoid(self.classifier(x)).cpu().numpy()[0]

        # --- cardiomegaly: the primary target -------------------------------
        ci = C.LABEL_COLS.index("Cardiomegaly")
        p_card = float(probs[ci])
        thr, thr_src = self.policy.get("Cardiomegaly", view)
        detected = p_card >= thr

        # Stage 13. The prediction is still computed and returned in full -- the
        # deferral flag sits alongside it rather than replacing it, so the UI can
        # show what the model would have said while marking it as not actionable.
        defer = self.deferral.assess(p_card, thr, view)

        # --- co-pathologies -------------------------------------------------
        # The UI filters on display names with spaces ("Pleural Effusion"),
        # not the underscored column names.
        copath = []
        for i, name in enumerate(C.LABEL_COLS):
            if name == "Cardiomegaly":
                continue
            t, _ = self.policy.get(name, view)
            copath.append(dict(
                name=name.replace("_", " "),
                status="present" if probs[i] >= t else "absent",
                probability=round(float(probs[i]), 4),
                threshold=round(t, 4)))

        # --- Grad-CAM on the cardiomegaly logit -----------------------------
        cam = self.gradcam.generate(x, ci)
        gradcam_b64 = overlay_heatmap(img_bgr, cam)

        # --- report ---------------------------------------------------------
        text, raw, prompt = self._generate_report(x, probs, view)

        return dict(
            prediction="Cardiomegaly" if detected else "No Cardiomegaly",
            confidence=round(p_card if detected else 1.0 - p_card, 4),
            probability=round(p_card, 4),
            gradcam_image=gradcam_b64,
            report_text=text,
            report_text_raw=raw,
            classifier_prompt=prompt,
            ground_truth_report=self.lookup_ground_truth(filename),
            copathologies=copath,
            # ---- added by this deployment ----
            view=(view or None),
            threshold=round(thr, 4),
            threshold_source=thr_src,
            reliability=self.policy.reliability(view),
            deferral=defer,
            model_info=dict(report_generator_stage=self.reportgen_stage,
                            **C.MODEL_STATS),
        )

    def _generate_report(self, x, probs, view):
        """Returns (clean_text, raw_decode, prompt).

        `raw_decode` keeps BART's special tokens so the "Raw Output" view shows
        exactly what the model emitted. In this system the two are otherwise
        near-identical -- and that is the point. The original deployment stripped
        artefacts like "compared to the prior study" from the OUTPUT with regex.
        Here those phrases were removed from the training TARGETS (Stage 1), so
        the model never learned to produce them: prior-study hallucination is
        0.0000 across all 4,722 test reports with no post-processing at all.
        """
        prompt = ""
        prompt_ids = prompt_mask = None
        if self.reportgen_stage == "stage11":
            try:
                import stage11_conditioned as s11
                thr = [self.policy.get(k, view)[0] for k in C.LABEL_COLS]
                prompt = s11.build_prompt(probs, thr, C.LABEL_COLS)
                prompt_ids, prompt_mask = s11.encode_prompts(
                    [prompt], self.tokenizer, device=self.device)
            except Exception as e:
                # Stage 11 without its prompt still produces sensible text -- the
                # ablation measured the prompt's own contribution at +0.0023.
                print("[service] prompt construction failed (%s); "
                      "generating without it" % e)
                prompt = ""

        with torch.no_grad():
            ids = self.reportgen.generate(
                x, prompt_ids, prompt_mask,
                num_beams=C.GEN_NUM_BEAMS, max_length=C.GEN_MAX_TOKENS,
                min_length=C.GEN_MIN_TOKENS,
                no_repeat_ngram_size=C.GEN_NO_REPEAT_NGRAM)

        clean = " ".join(self.tokenizer.batch_decode(
            ids, skip_special_tokens=True)[0].split())
        raw = self.tokenizer.batch_decode(ids, skip_special_tokens=False)[0]
        return clean, raw, prompt

    # ------------------------------------------------------------------
    def get_test_samples(self, limit: int = 200) -> list[dict]:
        out = []
        for _, r in self.test_df.head(limit).iterrows():
            out.append(dict(dicom_id=str(r["dicom_id"]), view=str(r["view"]),
                            cardiomegaly=int(r["Cardiomegaly"]),
                            edema=int(r.get("Edema", 0)),
                            pleural_effusion=int(r.get("Pleural_Effusion", 0))))
        return out

    def get_test_sample(self, dicom_id: str) -> dict | None:
        i = self.test_index.get(dicom_id)
        if i is None:
            return None
        r = self.test_df.iloc[i]
        import cv2
        p = C.TEST_IMAGE_DIR / str(r["image_path"])
        img_b64 = None
        if p.exists():
            im = cv2.imread(str(p))
            if im is not None:
                img_b64 = image_to_base64(im)
        report = " ".join(str(r.get("report", "")).split())
        return dict(
            dicom_id=dicom_id, image=img_b64, view=str(r["view"]),
            impression="", findings=report, report=report,
            labels={k: int(r.get(k, 0)) for k in C.LABEL_COLS})
