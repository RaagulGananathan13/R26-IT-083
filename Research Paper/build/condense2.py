# -*- coding: utf-8 -*-
"""Second condensation: the methods carry the most compressible text.

The paper is about the contract, not about four standard architectures, so the
model descriptions shrink to whatever the reliability argument needs.
"""
import io
import sys

sys.path.insert(0, '.')
import paper_content as C
import edit_body as EB

NEW = {
'A ConvNeXt-Base backbone': (
    "A ConvNeXt-Base backbone [[4]] produces eight sigmoid outputs from a 384 "
    "by 384 image; Grad-CAM [[14]] is taken at the final feature block, and a "
    "BioBART decoder [[5]] drafts report text we do not evaluate here. The "
    "network is not the interesting part. Each radiograph carries a metadata "
    "field *g* in {AP, PA} recording how it was taken; the two groups are not "
    "the same distribution, so a single operating point for both is a "
    "modelling error. Following [[16]], we fit one per group on validation "
    "only,"),

'with *q*(AP)': (
    "with *q*(AP) and *q*(PA) fitted on validation under one shared coverage "
    "budget, choosing the pair that minimises the absolute accuracy "
    "difference between groups, then frozen. At the deployed 85 % coverage "
    "target they are 0.2247 and 0.0029, so the system is far more reluctant "
    "to commit on the weaker group. An image with no projection field falls "
    "back to the global threshold."),

'A quality gate checking': (
    "A quality gate checking shape, duration, dead leads, units, amplitude, "
    "noise and rhythm runs before the classifier, so a rejected recording "
    "never produces a probability. Accepted recordings are filtered, "
    "resampled and normalised per lead, then classified by a one-dimensional "
    "residual network with squeeze-and-excitation and attention pooling, "
    "followed by per-class temperature scaling [[21]]. A triage layer "
    "converts each calibrated probability into one of three decisions under a "
    "training-conditional, or PAC, conformal bound [[19]]. For a class with "
    "*n* calibration positives, miss-rate budget *alpha* and confidence "
    "*delta*, the order statistic is"),

'since the coverage of the': (
    "since the coverage of the *k*-th order statistic follows Beta(*k*, "
    "*n*-*k*+1). The rule-out threshold is that order statistic among the "
    "positive calibration scores, the rule-in threshold its mirror over the "
    "negatives, and anything between is referred. A class with too few "
    "calibration positives is reported unattainable rather than approximated. "
    "Two models are served side by side, each with its own calibrator, and a "
    "class is ruled out only when both rule it out, so the merged miss rate "
    "is bounded by the tighter single-model bound at the expense of more "
    "referrals."),

'A bound like this': (
    "A bound like this depends on assumptions the input can break without "
    "appearing broken, so two checks withdraw the guarantee while leaving the "
    "prediction in place: one flags a swapped pair of limb electrodes from "
    "the polarity of one lead and the inversion of another, the other asks "
    "whether the rhythm is inside the label space at all, from an "
    "irregularity score thresholded on validation at a 5 % false-positive "
    "budget. Either way the probabilities are still returned and only the "
    "bounded-miss-rate claim is withdrawn, which is the *caution* state of "
    "(1) rather than *withheld*."),

'An R(2+1)D-18 backbone': (
    "An R(2+1)D-18 backbone [[8]] takes 32-frame clips and four heads: "
    "regression, ordinal cumulative, auxiliary class and log-variance. The "
    "boundaries at 30, 40 and 55 are clinical conventions, and the label they "
    "cut is itself noisy, since two readers typically disagree by about 4 "
    "points. Treating it as exact throws information away, so the ordinal "
    "targets are soft,"),

'where *e* is the recorded': (
    "where *e* is the recorded value, *t_k* the *k*-th boundary and *sigma* = "
    "4, so *s_k* is the probability that the true value lies above *t_k*. "
    "Unlike [[23]], rank consistency is structural rather than repaired after "
    "the fact: one severity score is compared against cut-points that "
    "increase by construction, because each gap is a softplus,"),

'so the cumulative probabilities': (
    "so the cumulative probabilities can never cross. Training uses a "
    "class-balanced sampler with deferred re-weighting [[24]] from epoch 15. "
    "The second cohort is intensity-matched before being blended in, since a "
    "balanced sampler over-draws from it and would let the network use "
    "scanner brightness as a shortcut for severity. A regressor on a skewed "
    "target also shrinks predictions toward the mean, pushing the severe tail "
    "over the boundary at 30, so an expansion is fitted on validation and "
    "applied without changing the weights,"),

'after which the boundaries': (
    "after which the boundaries are re-optimised on validation "
    "lexicographically: worst-class recall, then balanced accuracy, then "
    "macro-F1."),

'and values are clipped': (
    "and values are clipped to those recorded within *H* hours. The same "
    "cohort, split and code are featurised at *H* = 0, 6 and 24 hours, making "
    "accuracy against time a reported axis rather than an unstated "
    "assumption. Detection is a mean blend of a LightGBM and an XGBoost "
    "ensemble [[10]], [[11]]; subtyping uses a single four-class model rather "
    "than a cascade, since a cascade compounds error. The operating point is "
    "a stated optimisation rather than a hand-tuned multiplier,"),

'with the recall floor': (
    "with the recall floor *rho* = 0.75, solved on validation over bootstrap "
    "resamples and frozen. A case whose top-two margin falls below the (1 - "
    "*C*) quantile is referred to a clinician rather than subtyped. Results "
    "are reported on the intended-use population, visits with a cardiac "
    "complaint or an early electrocardiogram order, both observable at "
    "triage, so this is selection, not leakage. A separate head asks which "
    "wall of the heart the infarct involves."),

'Each subsection below fills': (
    "Each subsection fills the same four slots for one modality: the quality "
    "gate *q*, the uncertainty statistic *u* and its frozen threshold, the "
    "validity condition *v* whose failure downgrades a result to caution, and "
    "the control arm the mechanism is tested against. The models are "
    "standard; the reliability machinery around them is the contribution."),

'A component that cannot say': (
    "A component that cannot say when to disbelieve it forces the caller to "
    "guess, and every caller guesses differently. The contract makes that "
    "judgement an output. Each component keeps its own weights, uncertainty "
    "statistic and frozen decision rule; what it must additionally emit is a "
    "single field from five ordered states:"),

'One check of the same kind': (
    "Research and serving code drift apart, so beside the 132 automated tests "
    "the service carries, the radiograph endpoint was scored on 200 "
    "stratified real studies posted through the live HTTP path. Served "
    "accuracy was 0.790 [0.728, 0.841] at 14.0 % deferred, consistent with "
    "the offline figures, and 0.766 on bedside images against 0.833 on "
    "standing ones: the covariate shift reappears on real inputs through the "
    "deployed path."),

'The rebuild of this component': (
    "The rebuild of this component started from the leak described in Section "
    "I, so every feature declares an availability time *a(f)* relative to "
    "arrival. At a disclosure horizon *H* the admitted feature set is"),
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
print("prose %d -> %d  (-%.1f%%)" % (n0, n1, 100.0 * (n0 - n1) / n0))
