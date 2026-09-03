# -*- coding: utf-8 -*-
"""Final pass. Narrative and restatement only; every result, control arm,
citation and threat-to-validity item is kept."""
import io
import sys

sys.path.insert(0, '.')
import paper_content as C
import edit_body as EB

NEW = {
'In a single day': (
    "In a single day, a patient with chest pain who arrives at an emergency "
    "department produces four distinct kinds of data: a triage record of "
    "vital signs and a free-text complaint, an electrocardiogram within ten "
    "minutes [[1]], a chest radiograph taken early to rule out causes that "
    "are not cardiac [[2]], and an ultrasound scan of the beating heart. Each "
    "has become a machine-learning problem in its own right, and in each case "
    "the published result is typically one accuracy figure on one dataset."),

'Each modality has an established': (
    "Each modality has an established model family: convolutional "
    "classification over MIMIC-CXR [[3]] with a ConvNeXt backbone [[4]] and a "
    "BioBART decoder [[5]]; one-dimensional residual networks over PTB-XL "
    "[[6]]; video regression on EchoNet-Dynamic [[7]] with an R(2+1)D "
    "backbone [[8]], with CAMUS [[9]] as a smaller cohort richer in severe "
    "cases; and gradient-boosted trees [[10]], [[11]] with post-hoc "
    "attribution [[12]] over emergency-department tables [[13]]. Saliency is "
    "typically Grad-CAM [[14]]."),

'The reliability side has': (
    "The reliability side has its own literature, which we use rather than "
    "reinvent. Subgroup performance gaps are established [[15]], equal "
    "opportunity [[16]] is the standard metric, and fitting one threshold per "
    "group is the post-processing method proposed alongside it, so on this "
    "axis we claim the measurement and not the technique; the common "
    "alternative trains the offending factor out of the representation "
    "[[17]]. A separate line lets the model abstain, from the reject option "
    "[[18]] to conformal prediction, which converts a score into a decision "
    "carrying a finite-sample bound [[19]], [[20]], typically after "
    "calibration [[21]]. Closest to our deferral finding is [[22]], which "
    "shows abstention can *widen* group disparities; that is what our "
    "uniform-deferral control does, and the gap to the group-conditional "
    "version is our result. Ordinal targets employ rank-consistent heads "
    "[[23]], long tails deferred re-weighting [[24]], and leakage was "
    "formalised generally [[25]]."),

'Research and serving code drift': (
    "Research and serving code drift apart, so the radiograph endpoint was "
    "also scored on 200 stratified real studies posted through the live HTTP "
    "path beside the 132 automated tests the service carries. Served accuracy "
    "was 0.790 [0.728, 0.841] at 14.0 % deferred, and 0.766 on bedside images "
    "against 0.833 on standing ones: the covariate shift reappears on real "
    "inputs through the deployed path."),

'*Threats to validity.*': (
    "*Threats to validity.* All four cohorts are retrospective and public, so "
    "distribution, labelling convention and case mix reflect the institutions "
    "that released them: MIMIC-CXR and MIMIC-IV-ED from one US hospital, "
    "PTB-XL from a German cohort of the 1990s, EchoNet-Dynamic from one US "
    "centre and CAMUS from one French centre. No result here transfers to "
    "another site without being re-measured. Every reliability threshold is "
    "fitted on validation and frozen, so it inherits that cohort's case mix; "
    "the coverage targets and the recall floor *rho* = 0.75 are chosen by us, "
    "not derived from a clinical standard. There is no external validation, "
    "no prospective evaluation, no clinician-in-the-loop study and no patient "
    "outcome measured, so we can report that a decision was withheld but not "
    "whether withholding it benefited anyone. Without fully paired multimodal "
    "data, the aggregation rule of Section III is tested per component and "
    "not end to end."),

'*Limitations.*': (
    "*Limitations.* The C1 split is custom, its positive class enriched to "
    "50.4 %, and 98.3 % of its test images fall inside the official MIMIC-CXR "
    "training split; we therefore treat it as an internal operating point, "
    "make no comparison with published MIMIC-CXR benchmarks anywhere, and put "
    "a strict patient-level holdout first in further work. The same "
    "restriction applies to C2, which drops 21 % of PTB-XL. Training variance "
    "is measured for C2 and C3 only, and subgroup coverage is partial: C1 has "
    "the acquisition field, C2 sex and three age bands, C3 none, C4 has them "
    "but no breakdown yet. The electrode audit rests on 200 recordings and "
    "the infarct-wall head on 104 test cases. C3's worst-class recall "
    "improved but fell short of the 0.75 target we set, on test and on "
    "validation. Grad-CAM is used as a sanity check rather than proof of "
    "localisation, its repeatability on chest radiographs assessed at a "
    "structural similarity of 0.12 [[28]], so we make no explainability claim "
    "beyond that. This is a retrospective research prototype, not a "
    "clinically validated system and not a medical device."),
}

body = list(C.BODY)
n0 = sum(len(v) for k, v in body if k == 'P')
hit = []
for i, (k, v) in enumerate(body):
    if k != 'P':
        continue
    for key, new in NEW.items():
        if v.startswith(key):
            body[i] = (k, new)
            hit.append(key)
            break
missing = [k for k in NEW if k not in hit]
io.open('paper_content.py', 'w', encoding='utf-8').write(EB.dump(body, C.REFERENCES))
n1 = sum(len(v) for k, v in body if k == 'P')
print("condensed %d of %d (unmatched: %s)" % (len(hit), len(NEW), missing or "none"))
print("prose %d -> %d" % (n0, n1))
