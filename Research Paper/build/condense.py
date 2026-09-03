# -*- coding: utf-8 -*-
"""Condense to fit six rendered pages. No fact, number or citation is dropped."""
import io
import sys

sys.path.insert(0, '.')
import paper_content as C
import edit_body as EB

NEW = {}

NEW['Although the literature'] = (
    "Although the literature for each of these four issues is separate, the "
    "engineering response is the same each time: the model must expose when "
    "it should not be believed, in a form the calling code can branch on. We "
    "therefore treat reliability as the object of study rather than a "
    "property of any one network, and ask three questions. *RQ1*, can "
    "reliability-aware mechanisms reduce clinically important failure modes "
    "without materially reducing predictive performance? *RQ2*, can "
    "mechanisms as different as a group threshold, a conformal zone, a "
    "prediction interval and a disclosure horizon be represented by one "
    "contract? *RQ3*, does reliability-aware abstention produce safer "
    "decisions than answering everything, or than deferring uniformly? The "
    "contributions are:")

NEW['The reliability side has'] = (
    "The reliability side has its own literature, which we use rather than "
    "reinvent. Subgroup performance gaps are established [[15]], equal "
    "opportunity [[16]] is the standard metric, and fitting one threshold per "
    "group is the post-processing method proposed alongside it, so on this "
    "axis we claim the measurement and not the technique. The common "
    "alternative trains the offending factor out of the representation, for "
    "instance with label-conditional gradient reversal [[17]]. A separate "
    "line lets the model abstain, from the reject option [[18]] to conformal "
    "prediction, which converts a score into a decision carrying a "
    "finite-sample bound [[19]], [[20]], typically after calibration [[21]]. "
    "Closest to our deferral finding is [[22]], which shows abstention can "
    "*widen* group disparities; that is what our uniform-deferral control "
    "does, and the gap to the group-conditional version is our result. "
    "Ordinal targets employ rank-consistent heads [[23]], long tails deferred "
    "re-weighting [[24]], and leakage was formalised generally [[25]].")

NEW['That the endpoint aggregates'] = (
    "That the endpoint aggregates rather than fuses is a limitation we "
    "measured, not a preference. Of the six cohort pairs only radiograph and "
    "triage are linkable, both deriving from MIMIC-IV and sharing 19,979 "
    "patients, 81.6 % of the radiograph cohort; the other five share zero by "
    "construction. No patient in any cohort carries all four studies, so this "
    "is a cross-modality framework evaluated on four distinct cohorts, not "
    "patient-level multimodal fusion; we train no fusion model and report no "
    "joint accuracy.")

NEW['A ConvNeXt-Base backbone'] = (
    "A ConvNeXt-Base backbone [[4]] with a two-layer head generates eight "
    "sigmoid outputs from a 384 by 384 image, standardised per image rather "
    "than with ImageNet statistics. Grad-CAM [[14]] is taken at the final "
    "feature block, and a BioBART decoder [[5]] drafts report text we do not "
    "evaluate here. The network is not the interesting part. Each radiograph "
    "carries a metadata field *g* in {AP, PA} recording how it was taken; the "
    "two groups are not the same distribution, so a single operating point "
    "for both is a modelling error. Following [[16]], we fit one per group on "
    "validation data only,")

NEW['and apply the threshold'] = (
    "and apply the threshold belonging to the image's own group at inference. "
    "Ranking cannot change under (3), since AUROC is computed over the whole "
    "ordering and cutting each group at a different point reorders nothing. "
    "On top of it a selective rule refers the case when the prediction sits "
    "too close to the operating point,")

NEW['with *q*(AP)'] = (
    "with *q*(AP) and *q*(PA) fitted on validation under one shared coverage "
    "budget, choosing the pair that minimises the absolute accuracy "
    "difference between groups, and then frozen. At the deployed 85 % "
    "coverage target they are 0.2247 and 0.0029, so the system is far more "
    "reluctant to commit on the weaker group. An image with no projection "
    "field falls back to the global threshold.")

NEW['A quality gate checking'] = (
    "A quality gate checking shape, duration, dead leads, units, amplitude, "
    "noise and rhythm runs before the classifier, so a rejected recording "
    "never produces a probability. Accepted recordings are band-pass filtered "
    "from 0.5 to 40 Hz, resampled to 500 Hz and normalised per lead, then "
    "classified by a one-dimensional residual network with "
    "squeeze-and-excitation and attention pooling, followed by per-class "
    "temperature scaling [[21]]. A triage layer converts each calibrated "
    "probability into one of three decisions under a training-conditional, or "
    "PAC, conformal bound [[19]]. For a class with *n* calibration positives, "
    "miss-rate budget *alpha* and confidence *delta*, the order statistic is")

NEW['since the coverage of the'] = (
    "since the coverage of the *k*-th order statistic follows Beta(*k*, "
    "*n*-*k*+1). The rule-out threshold is that order statistic among the "
    "positive calibration scores, the rule-in threshold its mirror over the "
    "negatives, and anything between them is referred. A class with too few "
    "calibration positives is reported unattainable rather than approximated. "
    "Two models are served side by side, each with its own calibrator and "
    "thresholds, and a class is ruled out only when both rule it out, so the "
    "merged miss rate is bounded by the tighter single-model bound at the "
    "expense of more referrals.")

NEW['A bound like this'] = (
    "A bound like this depends on assumptions the input can break without "
    "appearing broken, so two checks withdraw the guarantee while leaving the "
    "prediction in place. The first flags a swapped pair of limb electrodes "
    "from the polarity of one lead and the inversion of another; the second "
    "asks whether the rhythm is inside the label space at all, from a "
    "beat-interval irregularity score thresholded on validation at a 5 % "
    "false-positive budget. In either case the probabilities are still "
    "returned and only the bounded-miss-rate claim is withdrawn, which is the "
    "*caution* state of (1) rather than *withheld*.")

NEW['An R(2+1)D-18 backbone'] = (
    "An R(2+1)D-18 backbone [[8]] pretrained on Kinetics-400 takes 32-frame "
    "clips of 112 by 112 pixels in two channels and four heads: regression, "
    "ordinal cumulative, auxiliary class and log-variance. The boundaries at "
    "30, 40 and 55 are clinical conventions, and the label they cut is itself "
    "noisy, since two readers typically disagree by about 4 points. Treating "
    "it as exact throws information away, so the ordinal targets are soft,")

NEW['where *e* is the recorded'] = (
    "where *e* is the recorded value, *t_k* the *k*-th boundary, *sigma* = 4 "
    "and *Phi* the standard normal distribution function, so *s_k* is the "
    "probability that the true value lies above *t_k*. Unlike [[23]], rank "
    "consistency is structural rather than repaired after the fact: one "
    "severity score is compared against cut-points that increase by "
    "construction, because each gap is a softplus,")

NEW['so the cumulative probabilities'] = (
    "so the cumulative probabilities can never cross. Training uses a "
    "class-balanced sampler with deferred re-weighting [[24]] from epoch 15 "
    "and an exponential moving average of the weights. The second cohort is "
    "intensity-matched before being blended in, since a balanced sampler "
    "over-draws from it and would let the network use scanner brightness as a "
    "shortcut for severity. A regressor on a skewed target also shrinks "
    "predictions toward the mean, pushing the severe tail over the boundary "
    "at 30, so an expansion is fitted on validation and applied without "
    "changing the weights,")

NEW['after which the boundaries'] = (
    "after which the boundaries are re-optimised on validation "
    "lexicographically: worst-class recall, then balanced accuracy, then "
    "macro-F1. At inference a study is sampled into ten clips and averaged "
    "across three seeds, and the interval is split-conformal.")

NEW['and values are clipped'] = (
    "and values are clipped to those recorded within *H* hours. The same "
    "cohort, split and code are featurised at *H* = 0, 6 and 24 hours, making "
    "accuracy against time a reported axis rather than an unstated "
    "assumption. Detection is a mean blend of a LightGBM and an XGBoost "
    "ensemble [[10]], [[11]]; subtyping uses a single four-class model rather "
    "than a cascade, since a cascade compounds error. The operating point is "
    "a stated optimisation rather than a hand-tuned multiplier,")

NEW['with the recall floor'] = (
    "with the recall floor *rho* = 0.75, solved on validation over bootstrap "
    "resamples and then frozen. A case whose top-two margin falls below the "
    "(1 - *C*) quantile of the validation margins is referred to a clinician "
    "rather than being subtyped. Results are reported on the intended-use "
    "population, visits with a cardiac complaint or an early "
    "electrocardiogram order; both are observable at triage, so this is "
    "selection, not leakage. A separate head asks which wall of the heart the "
    "infarct involves, on a label rebuilt from ICD-9 and ICD-10 codes.")

NEW['Four public cohorts'] = (
    "Four public cohorts are used, every split patient-disjoint and quoted as "
    "train / validation / test. C1: MIMIC-CXR-JPG [[3]], 36,362 / 4,474 / "
    "4,722 images, the test fold containing 2,891 bedside and 1,831 standing "
    "films, positive class enriched to 50.4 %. C2: the official PTB-XL [[6]] "
    "folds, 13,801 / 1,709 / 1,711 recordings, keeping only codes at full "
    "likelihood. C3: EchoNet-Dynamic [[7]], 7,465 / 1,288 / 1,277 studies, "
    "severity classes at 5.9 / 7.2 / 18.0 / 68.9 %, plus 1,000 CAMUS [[9]] "
    "clips in training only. C4: MIMIC-IV-ED [[13]], 142,111 / 30,453 / "
    "30,452 stays grouped by patient, 2.65 % positive. Each test split was "
    "evaluated once, and every decision rule in Section IV was fitted on "
    "validation and frozen before that split was opened. Metrics follow the "
    "task: AUROC, sensitivity, specificity and true-positive-rate disparity "
    "between acquisition groups [[16]] for C1; per-class recall and NPV with "
    "the empirical miss rate against the promised bound for C2; mean absolute "
    "error, R^{2} and minimum per-class recall for C3; AUROC, NPV and minimum "
    "per-class recall for C4. Coverage is quoted beside every selective "
    "number, since accuracy on the answered subset is not the accuracy of the "
    "system.")

NEW['Differences are tested'] = (
    "Differences are tested rather than eyeballed. Two rules scoring the same "
    "items are compared using McNemar's test [[26]] in mid-*p* form with Holm "
    "correction [[27]] within each family; gaps and aggregate metrics use a "
    "paired bootstrap of 10,000 resamples on identical indices. Subgroup miss "
    "rates use exact binomial tests with Wilson intervals across all 23 "
    "cells; triage intervals are cluster bootstraps resampled by patient.")

NEW['Accuracy on the answered subset'] = (
    "Accuracy on the answered subset flatters any system that abstains, so we "
    "report one measure that means the same thing in every modality. The "
    "*unsafe answer rate* is the probability that the system both answers and "
    "is wrong, U = *c* (1 - *A*) for coverage *c* and accuracy *A* among "
    "answered cases; a component that never abstains has U equal to its error "
    "rate. Reported with coverage, it prices abstention instead of concealing "
    "it. Fig. 2 gives all three components. C1 falls from U = 16.81 % at full "
    "coverage to 9.64 % at 80.6 % coverage under group-conditional deferral; "
    "uniform deferral of the same budget reaches a marginally lower 8.89 % "
    "but leaves the acquisition gap intact, which is the trade the contract "
    "is meant to make visible. C3 falls from 27.02 % to 20.36 % at 88.4 % "
    "coverage, and the 148 deferred studies are genuinely the hard ones, "
    "scoring 42.6 % against 73.0 % overall. C4 falls from 21.89 % to 6.68 %, "
    "at the price of deferring a third of subtyping decisions. Answering "
    "RQ3, abstention lowered the unsafe answer rate against the "
    "answer-everything arm in all three, and in C1 the group-conditional arm "
    "was the only one that also removed the failure mode.")

NEW['*Acquisition shift.*'] = (
    "*Acquisition shift.* C1 scores AUROC 0.8224 on bedside images against "
    "0.8864 on standing ones, a gap of 0.0639 [0.0491, 0.0790] in the same "
    "direction for all eight labels. Fitting the operating point per group "
    "reduced the reported true-positive-rate disparity by 73.3 % with an "
    "AUROC spread of exactly zero and no discernible accuracy cost (+0.02 "
    "points, [-0.26, +0.31], McNemar mid-*p* 0.885). We also reimplemented "
    "the representation-side alternative [[17]] on our own data, backbone and "
    "split: it achieved complete invariance, projection-detection AUC 0.5000, "
    "and still made the disparity 25.4 % *worse* at a cost of 0.0789 AUROC. "
    "Deferral behaves the same way (Fig. 3a). Deferring the same fraction of "
    "both groups leaves the 6.68-point gap at 6.28, the behaviour [[22]] "
    "describes; deferring per group at matched coverage closes it to -0.62 "
    "[-2.78, 1.37], a difference of 5.83 points with paired bootstrap *p* = "
    "0.0004.")

NEW['*Conditional validity.*'] = (
    "*Conditional validity.* Fitted marginally, the conformal bound held in "
    "only 14 of 23 class-by-subgroup cells, and two violations survive Holm "
    "correction: one class at a miss rate of 0.333 against a promised 0.10 "
    "under age 50, another at 0.330 against 0.20 at age 70 and over. "
    "Refitting one threshold per subgroup [[20]] restored the bound in 22 of "
    "23 cells, at the expense that every cell now needs its own positives: "
    "one cell with 42 positives cannot support a finite threshold and is "
    "reported as unattainable.")

NEW['*Silent corruption'] = (
    "*Silent corruption and open-set inputs.* Simulating each of the three "
    "limb-electrode swaps on 200 test recordings, the corrupted signal passes "
    "the quality gate in 197 to 198 of them, because the recording is clean "
    "but wired wrongly; up to 87 % of diagnoses change and 7 guarantees are "
    "voided. The physiology check detects 65.5 % and 60.5 % of two swaps at "
    "4.5 % false positives, and 4.0 % of the third. Separately, 114 "
    "recordings carry a rhythm the label space cannot represent and 113 "
    "received a bounded rule-out for a disease the model has no output unit "
    "for; the irregularity gate withholds the claim on 48.9 % of them.")

NEW['*Temporal leakage.*'] = (
    "*Temporal leakage.* One comorbidity column equals 1 for every positive "
    "stay and reaches AUROC 0.9200 alone; adding it back to an otherwise safe "
    "feature set moves the screen from AUROC 0.9665 to 0.9889, which is how "
    "an apparently outstanding result gets manufactured. A random split "
    "places 5,804 patients on both sides and contaminates 7,627 test rows; "
    "the patient-grouped split shares none. Under the availability contract, "
    "performance becomes a function of time (Fig. 3b): screen AUROC 0.8763, "
    "0.9121 and 0.9560, and recall on the hardest subtype 37.3 %, 58.2 % and "
    "80.0 % at *H* = 0, 6 and 24. At *H* = 0 the laboratory channel carries "
    "exactly 0.0 % of the attribution mass, rising to 4.6 % and 29.6 %; a "
    "leaking pipeline cannot produce that pattern.")

NEW["C4's infarct-wall head"] = (
    "C4's infarct-wall head is the second. With every feature it reaches "
    "AUROC 0.9074 and no per-class metric below 0.7551 on 104 test cases; "
    "removing three features parsed from the printed interpretation of the "
    "recording device costs 0.133 AUROC and reduces the weakest metric to "
    "0.6038, and those three alone reach AUROC 0.841. This is not temporal "
    "leakage, since they exist at triage, but the person who assigned the "
    "diagnosis code read the same printout. Feature and label therefore share "
    "a source: the label is partly defined by an input the model is given, a "
    "circularity no timestamp check can detect. Widening the head beyond two "
    "territories was measured too: a third class is recalled in 1 case of 12 "
    "and drags the other two down. C3's backbone was tested against the "
    "un-factorised alternative at three matched seeds and is worth keeping, "
    "though only on the classification metrics. Text generation exists in two "
    "components but is not evaluated here, and neither it nor the wall head "
    "is served.")

NEW['One check of the same kind'] = (
    "One check of the same kind closes the loop. Research and serving code "
    "drift apart, so beside the service's 132 automated tests the radiograph "
    "endpoint was scored on 200 stratified real studies posted through the "
    "live HTTP path. Served accuracy was 0.790 [0.728, 0.841], sensitivity "
    "0.880 and specificity 0.700, consistent with the offline figures at 14.0 "
    "% deferred, and accuracy was 0.766 on bedside images against 0.833 on "
    "standing ones. The covariate shift reappears on real inputs through the "
    "deployed path.")

NEW['Answering RQ1 and RQ3'] = (
    "Answering RQ1 and RQ3 together: in these experiments the effective "
    "intervention was located in the decision layer rather than the "
    "representation. Three attempts to close the acquisition gap by altering "
    "the model failed against a null arm, while a threshold and a deferral "
    "budget conditioned on the same variable worked, and the unsafe answer "
    "rate fell in every component where coverage is measurable. We do not "
    "conclude that clinical reliability is generally post-processing. We draw "
    "a narrower conclusion: on these four tasks, several reliability problems "
    "were addressed effectively by decision-layer controls that are "
    "inexpensive to fit and auditable, and conditioning each control on the "
    "variable that truly causes the failure mattered more than the strength "
    "of the control. RQ2 is answered by construction and then tested through "
    "use: four mechanisms with nothing in common were reduced to a single "
    "five-state field without any of them losing information, since the "
    "component-native payload is returned unaltered alongside it.")

NEW['*Threats to validity.*'] = (
    "*Threats to validity.* All four cohorts are retrospective and public, so "
    "their distribution, labelling convention and case mix reflect the "
    "institutions that released them; MIMIC-CXR and MIMIC-IV-ED come from one "
    "US hospital, PTB-XL from a German cohort of the 1990s, EchoNet-Dynamic "
    "from one US centre and CAMUS from one French centre, and no result here "
    "transfers to another site without being re-measured. Every reliability "
    "threshold is fitted on validation and frozen, so it inherits that "
    "cohort's case mix; the coverage targets and the recall floor *rho* = "
    "0.75 are chosen by us, not derived from a clinical standard. There is no "
    "external validation, no prospective evaluation, no clinician-in-the-loop "
    "study and no patient outcome measured, so we can report that a decision "
    "was withheld but not whether withholding it benefited anyone. The "
    "absence of fully paired multimodal data means the aggregation rule of "
    "Section III is tested per component and not end to end.")

NEW['*Limitations.*'] = (
    "*Limitations.* The C1 split is custom, its positive class enriched to "
    "50.4 %, and 98.3 % of its test images fall inside the official MIMIC-CXR "
    "training split; we therefore treat it as an internal operating point, "
    "make no comparison with published MIMIC-CXR benchmarks anywhere in this "
    "paper, and place a strict patient-level holdout first in further work. "
    "The same restriction applies to C2, which drops 21 % of PTB-XL. Training "
    "variance is measured for C2 and C3 only. Subgroup coverage is partial: "
    "C1 has the acquisition field, C2 sex and three age bands, C3 no "
    "demographic fields, and C4 has them but no breakdown yet. The electrode "
    "audit rests on 200 recordings, and regular out-of-scope rhythms remain "
    "silent failures. The infarct-wall head has 104 test cases. C3's "
    "worst-class recall improved but fell short of the 0.75 target we set, on "
    "test and on validation. Grad-CAM is used as a sanity check rather than "
    "proof of localisation: its repeatability on chest radiographs has been "
    "assessed at a structural similarity of 0.12 [[28]], so we make no "
    "explainability claim beyond that. This is a retrospective research "
    "prototype, not a clinically validated system and not a medical device.")

NEW['A deployed clinical model'] = (
    "A deployed clinical model should not only produce a prediction; it "
    "should state whether that prediction is reliable enough to act on. We "
    "made that statement a first-class output through a reliability contract, "
    "five ordered states assigned by a precedence rule over a component's own "
    "quality, uncertainty and validity signals, and instantiated it in four "
    "cardiovascular modalities that share no patients and no features. "
    "Conditioning abstention on the variable that actually causes the failure "
    "beat altering the model in every comparison we ran, and lowered the "
    "unsafe answer rate wherever coverage could be measured. The same "
    "standard applied to our own design choices returned three negative "
    "results out of four. What the contract buys is that a caller applies one "
    "rule everywhere. Next are external validation, a strict patient-level "
    "holdout for C1, and a paired study on the 19,979 patients our radiograph "
    "and triage cohorts share.")

# ---------------------------------------------------------------- apply ---
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

# insert the new figure just after the cost-of-abstaining paragraph
out = []
for k, v in body:
    out.append((k, v))
    if k == 'P' and isinstance(v, str) and v.startswith('Accuracy on the answered'):
        out.append(("FIG", ("fig3_uar.png",
                            "Unsafe answer rate before and after abstention, "
                            "with the coverage each arm retains printed above "
                            "the bar. Lower is safer; the percentage is the "
                            "price paid for it.",
                            False)))
body = out

# the results figure is now Fig. 3
for i, (k, v) in enumerate(body):
    if k == 'P' and isinstance(v, str):
        body[i] = (k, v.replace('Fig. 2a', 'Fig. 3a').replace('Fig. 2b', 'Fig. 3b'))

io.open('paper_content.py', 'w', encoding='utf-8').write(EB.dump(body, C.REFERENCES))
n1 = sum(len(v) for k, v in body if k == 'P')
print("paragraphs condensed : %d   (unmatched: %s)" % (len(hit), missing or "none"))
print("prose chars          : %d -> %d  (-%d, %.1f%%)"
      % (n0, n1, n0 - n1, 100.0 * (n0 - n1) / n0))
