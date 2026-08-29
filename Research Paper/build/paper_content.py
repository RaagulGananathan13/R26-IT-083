# -*- coding: utf-8 -*-
"""Content of the conference paper (double-blind submission version).

Single source of truth. build_docx.py and build_tex.py both render this, so the
Word file and the LaTeX source cannot drift apart.

ANONYMOUS = True renders the double-blind author block required by the
compliance guidelines. Set it to False and fill in AUTHORS for camera-ready.

Block grammar
-------------
("H1", title)                      numbered I, II, III ...
("H2", title)                      numbered A, B, C ... within the section
("P",  text)                       body paragraph
("EQ", (unicode_form, latex_form)) numbered equation
("LIST", [item, ...])              tight bullet list
("FIG", (file, caption, span))     span=True -> figure across both columns
("TABLE", dict)                    span=False -> table inside one column

Inline markup
-------------
  *word*    italic
  ^{x}      superscript
  [[n]]     citation n
"""

ANONYMOUS = True

TITLE = ("A Reliability-Aware Explainable AI System for "
         "Cardiovascular Disease Detection and Diagnosis")

# Used only when ANONYMOUS is False (camera-ready).
AUTHORS = [
    ("Author One", "Department", "Institution", "City, Country", "email"),
]

ABSTRACT = (
    "A machine-learning model placed in a clinical workflow returns a "
    "probability, and the software around it usually treats every probability "
    "as equally trustworthy. Our measurements say otherwise. We present a "
    "decision-support system over chest radiographs, 12-lead "
    "electrocardiograms, echocardiogram video and emergency-department triage "
    "records, in which each component carries an explicit rule for declining "
    "to answer and each rule is tested against a control. On radiographs, a "
    "covariate stored in the image metadata costs "
    "0.0639 AUROC; group-wise thresholds cut the reported true-positive-rate "
    "disparity by 73.3 % at no cost in AUROC, and group-conditional deferral "
    "closed a 6.68 point accuracy gap to within sampling noise of zero, where "
    "uniform deferral did not. On electrocardiograms, a conformal layer gives "
    "rule-in and rule-out decisions under a stated miss-rate bound that holds "
    "for the population and fails inside 9 of 23 sex and age subgroups, which "
    "group-conditional calibration repairs in 22 of 23. On echocardiograms, "
    "ordinal supervision derived from the label's own measurement noise "
    "reaches 3.979 points of mean absolute error and 0.723 worst-class recall "
    "under an 11 to 1 imbalance. On triage records, admitting a feature only "
    "when it existed at the decision time gives AUROC 0.9560 at 99.41 % "
    "negative predictive value. Four further ablations turn the same standard "
    "on our own design choices, and three come back negative, including a "
    "network component of ours that costs 0.0042 macro-AUROC."
)

INDEX_TERMS = ("abstention, ablation, chest radiography, class imbalance, "
               "clinical decision support, conformal prediction, covariate "
               "shift, data leakage, echocardiography, electrocardiography")

BODY = [

    # ------------------------------------------------------------------ I
    ("H1", "Introduction"),

    ("P",
     "A patient who arrives at an emergency department with chest pain "
     "generates four very different kinds of data within a day. A triage nurse "
     "records vital signs and a free-text complaint. An electrocardiogram is "
     "recorded within ten minutes [[1]], because a blocked artery has to be "
     "found before any blood test can come back. A chest radiograph is taken "
     "early, mainly to rule out causes that are not cardiac at all [[2]]. "
     "Blood is drawn and repeated, and an ultrasound scan of the beating heart "
     "is booked. Each has become a machine-learning problem in its own right, "
     "and in each case the published result is normally one accuracy figure on "
     "one dataset."),

    ("P",
     "One figure is a poor description of a model that is about to sit behind "
     "an API. It says nothing about the inputs on which the model is weakest, "
     "and here those inputs are not rare corner cases. We measured four. Our "
     "chest-radiograph classifier is 0.0639 AUROC worse on images taken at the "
     "bedside than on images taken standing, a covariate shift driven by a "
     "field in the image metadata; the bedside view is used when the patient "
     "is too ill to stand, so the model is weakest on the sickest patients. A "
     "conformal guarantee on our electrocardiogram classifier holds over the "
     "test set as a whole and fails inside 9 of 23 sex and age subgroups. Our "
     "echocardiogram model is weakest on the most severe cases, 5.9 % of the "
     "training data. And a triage model can look excellent because it read a "
     "feature that did not exist yet: in the raw records that blood test was "
     "taken a median of 21.75 hours after the decision it claims to support. "
     "That one is not hypothetical. An audit of an earlier version of our own "
     "triage component found it reading its label out of a comorbidity "
     "column, and we rebuilt it from the raw tables."),

    ("P",
     "These four problems have separate literatures, but the engineering "
     "response is the same each time: the model has to expose when it should "
     "not be believed, in a form the calling code can branch on. So "
     "we built each component around a rule that declines to commit, tested "
     "that rule against a control which keeps the procedure and removes the "
     "signal, and reduced the four rules to one field at the service boundary. "
     "We then turned the same discipline on our own design choices. The "
     "contributions are:"),

    ("LIST", [
        "four detection models, one per modality, each carrying an abstention "
        "rule fitted on validation data and frozen before the test split was "
        "opened;",

        "one measured failure mode per modality, each with a control arm: "
        "acquisition shift, subgroup validity of a conformal guarantee, silent "
        "input corruption, inputs outside the label space, tail shrinkage "
        "under imbalance, and temporal leakage;",

        "four ablations of our own design decisions, three of them negative "
        "results we report rather than bury, including one network "
        "component that measurably costs accuracy;",

        "a reliability contract mapping four incompatible vocabularies onto "
        "five actionability levels, and a gated six-stage traversal over it.",
    ]),

    # ----------------------------------------------------------------- II
    ("H1", "Related Work"),

    ("P",
     "Each modality has an established model family: convolutional "
     "classification over MIMIC-CXR [[3]] with a ConvNeXt backbone [[4]] and a "
     "BioBART decoder [[5]]; one-dimensional residual networks over PTB-XL "
     "[[6]]; video regression on EchoNet-Dynamic [[7]] with an R(2+1)D "
     "backbone [[8]], with CAMUS [[9]] as a smaller cohort richer in severe "
     "cases; and gradient-boosted trees [[10]] with post-hoc attribution "
     "[[11]] over emergency-department tables [[12]]. Saliency is usually "
     "Grad-CAM [[13]]."),

    ("P",
     "The reliability side has its own literature, which we use rather than "
     "reinvent. Subgroup performance gaps are well documented [[14]]; equal "
     "opportunity [[15]] is the usual metric, and fitting one threshold per "
     "group is the post-processing method proposed alongside it, so on this "
     "axis we claim the measurement and not the technique. The common alternative is to train the offending factor "
     "out of the representation, for instance with label-conditional gradient "
     "reversal [[16]]. A separate line lets the model abstain, from the reject "
     "option [[17]] to conformal prediction, which turns a score into a "
     "decision carrying a finite-sample bound [[18]], [[19]], usually after "
     "calibration [[20]]. Closest to our deferral result is [[21]], "
     "which shows abstention can *widen* group disparities; that is what our "
     "uniform-deferral control does, and the gap to the group-conditional "
     "version is our result. Ordinal targets use "
     "rank-consistent heads [[22]], and long tails with deferred re-weighting "
     "[[23]]. Leakage, using information unavailable at prediction time, was "
     "formalised for data mining in general [[24]]."),

    ("P",
     "What we did not find was these ideas used together, across more than one "
     "modality, with every mechanism checked against a null arm and the "
     "guarantee reported other than marginally."),

    # ---------------------------------------------------------------- III
    ("H1", "System Architecture"),

    ("P",
     "The system is four models behind one FastAPI process and one web console "
     "(Fig. 1). The four share no "
     "predictions and no features, since a cardiomegaly probability and an "
     "ejection fraction have nothing in common. What they share is the "
     "contract in Section IV-E."),

    ("FIG", ("fig1_architecture.png",
             "System architecture. Each component keeps its own weights and "
             "its own frozen decision rule; the service applies that rule "
             "using the component's own code and normalises the outcome.",
             False)),

    ("P",
     "Getting four independently written codebases into one process was not "
     "free. Their top-level module names collide, so a plain import hands one "
     "component another's configuration object and the first symptom is a "
     "wrong number rather than an exception. Each therefore runs inside a "
     "module sandbox that installs its own search path and then lifts its "
     "modules back out, asserting that every name resolved to the intended "
     "owner. No component logic is reimplemented in the service: thresholds, "
     "calibration maps, conformal bounds and decision rules are read from each "
     "component's frozen artefacts and applied by its own code."),

    ("P",
     "Above the four adapters sits a traversal engine that walks one patient "
     "through six stages in the order the tests are actually done: triage on "
     "arrival, electrocardiogram within ten minutes [[1]], radiograph as the "
     "rule-out step [[2]], triage again when the blood test returns, the "
     "ultrasound, and triage once more at the end. Three results end the walk "
     "early because they make the remaining tests pointless, and every routing "
     "decision records the values it rests on. A stage with no study is marked "
     "not supplied, never read as a negative finding."),

    ("P",
     "The multi-modal endpoint aggregates rather than fuses: it reduces the "
     "four verdicts to their worst case and claims no joint performance. That "
     "restraint is measured. Of the six cohort pairs only radiograph and "
     "triage are linkable, both deriving from MIMIC-IV, sharing 19,979 "
     "patients or 81.6 % of the radiograph cohort; the other five share zero "
     "by construction, since PTB-XL, EchoNet-Dynamic and CAMUS come from "
     "different hospitals, countries and decades. A patient-level "
     "four-modality dataset cannot be built from these sources, so no fusion "
     "model was trained."),

    # ----------------------------------------------------------------- IV
    ("H1", "Methods"),

    ("H2", "Covariate Shift from Acquisition Metadata"),
    ("P",
     "A ConvNeXt-Base backbone [[4]] with a two-layer head produces eight "
     "sigmoid outputs from a 384 by 384 image, standardised per image rather "
     "than with ImageNet statistics, which are wrong for a single-channel "
     "radiograph. Grad-CAM [[13]] is taken at the last feature block, and a "
     "BioBART decoder [[5]] over 144 visual tokens writes a draft report. The "
     "interesting part is not the network. Every radiograph carries a metadata "
     "field *g* in {AP, PA} recording how it was taken; the two groups are not "
     "the same distribution, so one operating point for both is a modelling "
     "error. Following [[15]], we fit one per group on validation data only,"),

    ("EQ", ("τ_g = arg max_τ F1( y_g , [ p_g ≥ τ ] ) ,",
            r"\tau_g \;=\; \arg\max_{\tau}\; F_1\!\left(y_g,\; "
            r"\mathbf{1}[\,p_g \ge \tau\,]\right),")),

    ("P",
     "and apply the threshold belonging to the image's own group at inference. "
     "Ranking quality cannot change under (1), because AUROC is computed over "
     "the whole ordering and cutting each group at a different point reorders "
     "nothing; we checked numerically, and the AUROC difference was zero to "
     "twelve decimal places. On top of the threshold, a selective rule hands "
     "the case to a radiologist when the prediction sits too close to the "
     "operating point,"),

    ("EQ", ("m = | p − τ_g | ,   answer if m ≥ q_g ,   otherwise refer,",
            r"m \;=\; \lvert p - \tau_g \rvert, \qquad "
            r"\text{answer if } m \ge q_g, \ \text{else refer,}")),

    ("P",
     "with *q*(AP) and *q*(PA) fitted on validation under one shared coverage "
     "budget, choosing the pair that minimises the absolute accuracy "
     "difference between groups, and then frozen. At the deployed 85 % "
     "coverage target they are 0.2247 and 0.0029, so the system is far more "
     "reluctant to commit on the weaker group. An image with no projection "
     "field falls back to the global threshold."),

    ("H2", "A Guarantee, and Two Ways It Silently Stops Holding"),
    ("P",
     "A quality gate checking shape, duration, dead leads, units, amplitude, "
     "noise and rhythm runs before the classifier, so a rejected recording "
     "never produces a probability. Accepted recordings are band-pass filtered "
     "from 0.5 to 40 Hz with a 50 Hz notch, resampled to 500 Hz and normalised "
     "per lead, then classified by a one-dimensional residual network with "
     "squeeze-and-excitation and attention pooling, followed by per-class "
     "temperature scaling [[20]]. A triage layer turns each calibrated "
     "probability into one of three decisions under a training-conditional, or "
     "PAC, conformal bound [[18]]. For a class with *n* calibration positives, "
     "miss-rate budget α and confidence δ, the order statistic is"),

    ("EQ", ("m* = max { k ≤ n :  F_Beta( α ; k , n − k + 1 ) ≥ 1 − δ } ,",
            r"m^{*} \;=\; \max\left\{\,k \le n \;:\; "
            r"F_{\mathrm{Beta}(k,\,n-k+1)}(\alpha) \ge 1-\delta \,\right\},")),

    ("P",
     "since the coverage of the *k*-th order statistic follows Beta(*k*, *n*−"
     "*k*+1). The rule-out threshold is the *m**-th smallest calibration score "
     "among positives, the rule-in threshold its mirror over the negatives at "
     "a false-alarm budget β, and anything between them is referred. A class "
     "with too few calibration positives is reported as unattainable rather "
     "than approximated. Two models are served side by side, each with its own "
     "calibrator and thresholds, and a class is ruled out only when both rule "
     "it out; the merged rule-out set is their intersection, so the merged "
     "miss rate is bounded by the tighter single-model bound, at the cost of "
     "more referrals."),

    ("P",
     "A bound like this is conditional on assumptions the input can break "
     "without looking broken, so two checks withdraw the guarantee while "
     "leaving the prediction in place. The first flags a swapped pair of limb "
     "electrodes, an exact linear map of the standard lead definitions, from "
     "the polarity of one lead and the inversion of another; the second asks "
     "whether the rhythm is inside the label space at all, from a "
     "beat-interval irregularity score thresholded on validation at a 5 % "
     "false-positive budget. Either way the probabilities are still returned "
     "and only the sentences promising a bounded miss rate are suppressed. A "
     "final gate checks each generated sentence against the numbers that "
     "produced it."),

    ("H2", "Ordinal Targets, Noisy Labels and a Long Tail"),
    ("P",
     "An R(2+1)D-18 backbone [[8]] pretrained on Kinetics-400 takes 32-frame "
     "clips of 112 by 112 pixels in two channels, grey level and a temporal "
     "difference, with four heads: regression, an ordinal cumulative head, an "
     "auxiliary class head and a log-variance head. The class boundaries at "
     "30, 40 and 55 are clinical conventions, and the label they cut is itself "
     "noisy, since two human readers typically disagree by about 4 points. "
     "Treating it as exact throws information away, so the ordinal targets are "
     "soft,"),

    ("EQ", ("s_k = 1 − Φ( ( t_k − e ) / σ ) ,",
            r"s_k \;=\; 1 - \Phi\!\left(\frac{t_k - e}{\sigma}\right),")),

    ("P",
     "where *e* is the recorded value, *t_k* the *k*-th boundary, σ = 4 and "
     "Φ the standard normal distribution function, so *s_k* is the "
     "probability that the true value lies above *t_k* given a noisy "
     "measurement. Rank consistency is structural instead of repaired "
     "afterwards, unlike [[22]]: one severity score is compared against "
     "cut-points that increase by construction, because each gap is a "
     "softplus,"),

    ("EQ", ("c_1 = a ,   c_k = a + Σ_{j<k} softplus( g_j ) ,   z_k = f(x) − c_k ,",
            r"c_1 = a,\quad c_k = a + \sum_{j<k}\mathrm{softplus}(g_j),"
            r"\quad z_k = f(x) - c_k,")),

    ("P",
     "so the cumulative probabilities can never cross. Training uses a "
     "class-balanced sampler with deferred re-weighting [[23]] from epoch 15 "
     "and an exponential moving average of the weights. The second cohort is "
     "intensity-matched before being mixed in, because a balanced sampler "
     "over-draws from it and would let the network use scanner brightness as a "
     "shortcut for severity. A regressor on a skewed target also shrinks "
     "predictions towards the mean, pushing the severe tail over the boundary "
     "at 30, so an expansion is fitted on validation and applied without "
     "touching the weights,"),

    ("EQ", ("ê' = ȳ + κ ( ê − x̄ ) ,   κ = sd(y) / sd(ê) , clipped to [1.0, 1.7],",
            r"\hat{e}\,' \;=\; \bar{y} + \kappa\,(\hat{e}-\bar{x}),\qquad "
            r"\kappa = \frac{\mathrm{sd}(y)}{\mathrm{sd}(\hat{e})}"
            r"\ \text{clipped to } [1.0,\,1.7],")),

    ("P",
     "after which the boundaries are re-optimised on validation with a "
     "lexicographic objective: worst-class recall, then balanced accuracy, "
     "then macro-F1, then closeness to the clinical value. At inference a "
     "study is sampled into ten clips and averaged over three seeds, and the "
     "interval is split-conformal, widened by the learned aleatoric term and "
     "by disagreement between clips."),

    ("H2", "An Information-Availability Contract"),
    ("P",
     "The rebuild of this component started from the leak described in Section "
     "I, so every feature now declares an availability time *a(f)* relative to "
     "arrival. At a disclosure horizon *H* the admitted feature set is"),

    ("EQ", ("F_H = { f : a(f) ≤ H } ,",
            r"\mathcal{F}_H \;=\; \{\, f \;:\; a(f) \le H \,\},")),

    ("P",
     "and values are clipped to those recorded within *H* hours. The same "
     "cohort, split and code are featurised at *H* = 0, 6 and 24 hours, making "
     "accuracy against time a reported axis rather than an unstated "
     "assumption. Detection is a gradient-boosted tree ensemble [[10]]; "
     "subtyping uses one four-class model instead of a cascade, since a "
     "cascade compounds error, a patient the screen misses never being "
     "recoverable later. The operating point is a stated optimisation rather "
     "than a hand-tuned multiplier,"),

    ("EQ", ("w* = arg max_w  macroF1( arg max_k w_k p_k )   s.t.   "
            "min_k recall_k(w) ≥ ρ ,",
            r"w^{*} = \arg\max_{w}\ \mathrm{macroF_1}"
            r"\!\left(\arg\max_k w_k p_k\right)\ \ \text{s.t.}\ \ "
            r"\min_k \mathrm{recall}_k(w) \ge \rho,")),

    ("P",
     "with the recall floor ρ = 0.75, solved on validation over bootstrap "
     "resamples and then frozen. A case whose top-two margin falls below the "
     "(1 − *C*) quantile of the validation margins is referred to a "
     "clinician instead of being subtyped. Results are reported on the "
     "intended-use population, meaning visits with a cardiac complaint or an "
     "early electrocardiogram order; both are observable at triage, so this is "
     "selection, not leakage. A separate head asks which wall of the heart the "
     "infarct involves. That label is absent from the processed tables and was "
     "rebuilt from diagnosis codes, which needs both the ICD-9 and ICD-10 "
     "vocabularies since the cohort straddles the transition almost evenly."),

    ("H2", "One Reliability Contract"),
    ("P",
     "The four mechanisms speak four vocabularies: an acquisition group and a "
     "deferral margin; a conformal zone and a withdrawn guarantee; a "
     "prediction interval and two kinds of variance; a disclosure horizon and "
     "a referral. The service maps them onto one enumerated field: "
     "*actionable* when the component stands behind the result, *caution* when "
     "it stands but measured reliability is lower for this input, *deferred* "
     "when it declines to commit, *withheld* when output was suppressed after "
     "a quality or verification failure, and *unavailable* when it could not "
     "run. A client then applies one rule, do not act on a result that is not "
     "actionable, without knowing anything about projections, conformal zones "
     "or horizons. A component that declines to answer returns a normal "
     "response, not an error, because turning a safety mechanism into an error "
     "status pushes callers into retry loops."),

    # ------------------------------------------------------------------ V
    ("H1", "Experimental Setup"),

    ("P",
     "Splits are patient-disjoint throughout: MIMIC-CXR-JPG [[3]] 36,362 / "
     "4,474 / 4,722 images, with 2,891 bedside and 1,831 standing in the test "
     "fold and the positive class enriched to 50.4 %; the official PTB-XL "
     "[[6]] folds, 13,801 / 1,709 / 1,711 recordings, keeping only codes at "
     "full likelihood; EchoNet-Dynamic [[7]] 7,465 / 1,288 / 1,277 studies, "
     "severity classes at 5.9 / 7.2 / 18.0 / 68.9 %, with 1,000 CAMUS [[9]] "
     "clips added to training only; and MIMIC-IV-ED [[12]] 142,111 / 30,453 / "
     "30,452 stays grouped by patient, 2.65 % positive. Each test split was "
     "evaluated once, and every decision rule in Section IV was fitted on "
     "validation and frozen before that split was opened. Two components were "
     "trained on an NVIDIA L4 and two on an RTX 4060 laptop GPU. Metrics "
     "follow the task: AUROC, sensitivity, "
     "specificity and true-positive-rate disparity between acquisition groups "
     "[[15]] for C1; per-class recall and NPV, the precision of the negative "
     "class, with the empirical miss rate against the promised bound for C2; "
     "mean absolute error, R^{2} and minimum per-class recall for C3; AUROC, "
     "NPV and minimum per-class recall for C4. "
     "Coverage, the fraction of cases answered rather than deferred, is quoted "
     "beside every selective number, because accuracy on the answered subset "
     "is not the accuracy of the system."),

    ("P",
     "Differences are tested, not eyeballed. Two decision rules scoring the "
     "same items are compared with McNemar's test [[25]] in its mid-*p* form "
     "with Holm correction [[26]] inside each family, and differences of gaps "
     "or of aggregate metrics with a paired bootstrap of 10,000 resamples in "
     "which both systems use identical indices. Subgroup miss rates use exact "
     "binomial tests with Wilson intervals and Holm correction across all 23 "
     "cells; triage intervals are cluster bootstraps resampled by patient."),

    # ----------------------------------------------------------------- VI
    ("H1", "Results"),

    ("H2", "Detection Performance"),
    ("P",
     "Table I gives the headline test-set result per component; these are four "
     "tasks on four cohorts, so the rows are not comparable. Three numbers are "
     "worth adding. C1's report generator reaches a clinical-efficacy F1 of "
     "0.5937, and invented references to earlier studies, present in 70.7 % of "
     "the raw training reports, occur in 0 of 4,722 generated reports once the "
     "targets are cleaned. C3 places 99.7 % of studies within one severity "
     "class of the truth, with no severe case graded normal. C4 misses 66 of "
     "the 763 positive cases in its intended-use population, at 18.09 alerts "
     "per 100 patients."),

    ("TABLE", {
        "caption": "Headline test-set performance. Four tasks on four "
                   "cohorts; rows are not comparable with each other.",
        "span": False,
        "cols": ["Comp.", "Task", "n", "Result"],
        "widths": [0.09, 0.26, 0.11, 0.54],
        "rows": [
            ["C1", "Cardiomegaly, 7 co-findings", "4,722",
             "AUROC 0.9189, sensitivity 92.3 %, specificity 74.0 %; mean "
             "AUROC 0.8554 over 8 labels"],
            ["C2", "5 ECG superclasses", "1,711",
             "macro accuracy 0.864, recall 0.810, NPV 0.933; every class "
             "above 0.75"],
            ["C3", "Ejection fraction, 4 grades", "1,277",
             "MAE 3.979 points, R^{2} 0.818, worst-class recall 0.723"],
            ["C4", "ACS screen and subtype", "30,452",
             "screen AUROC 0.9560, NPV 99.41 %; subtyping macro-F1 0.7448; "
             "wall head AUROC 0.9074"],
        ],
    }),

    ("P",
     "Fig. 2 breaks C1 down by finding, and the two curves do not track each "
     "other. Pneumothorax reaches AUROC 0.9141 at 54.0 % sensitivity, because "
     "at 3.7 % prevalence the F1-optimal cut sits high; pneumonia and "
     "consolidation are worse on both axes. Cardiomegaly, the finding this "
     "component is built around, is the second best of the eight rather than "
     "the best."),

    ("FIG", ("fig2_per_class.png",
             "C1 per-finding discrimination and sensitivity on the 4,722-image "
             "test split, ordered by AUROC. The dotted line marks 0.75.",
             False)),

    ("P",
     "Fig. 3 shows what the training schedule bought C2. Validation "
     "macro-AUROC rises for roughly ten epochs, peaks near epoch 19 and then "
     "drifts down by about 0.005 while the training loss keeps falling, so "
     "checkpoints are selected on validation discrimination rather than at the "
     "end of the schedule. The three seeds differ by less than the "
     "epoch-to-epoch noise, which is why we report seed variance for this "
     "component and not run-to-run stability for the others."),

    ("FIG", ("fig3_training.png",
             "C2 validation macro-AUROC and training loss over 40 epochs, "
             "three seeds. Grey lines are individual seeds, the solid line "
             "their mean.",
             False)),

    ("H2", "What the Abstention Mechanisms Buy"),
    ("P",
     "Each result below is stated against a control, meaning an arm that keeps "
     "the procedure and removes the signal. Fig. 4 shows one result per "
     "component."),

    ("FIG", ("fig4_results.png",
             "(a) Accuracy gap between acquisition groups under three deferral "
             "policies, with 95 % bootstrap intervals. (b) Screen AUROC, "
             "unstable-angina recall and the share of attribution carried by "
             "laboratory features, against the disclosure horizon.",
             False)),

    ("P",
     "*Acquisition shift.* C1 scores AUROC 0.8224 on bedside images against "
     "0.8864 on standing ones, a gap of 0.0639 [0.0491, 0.0790] in the same "
     "direction for all eight labels. Fitting the operating point per group "
     "cut the reported true-positive-rate disparity by 73.3 % with an AUROC "
     "spread of exactly zero, at no significant accuracy cost (+0.02 points, "
     "[−0.26, +0.31], McNemar mid-*p* 0.885). We also reimplemented the "
     "representation-side alternative [[16]] on our own data, backbone and "
     "split: it reached complete invariance, projection-detection AUC 0.5000, "
     "and still made the disparity 25.4 % *worse* at a cost of 0.0789 AUROC. "
     "Deferral behaves the same way (Fig. 4a). Deferring the same fraction of "
     "both groups leaves the 6.68-point gap at 6.28, the behaviour [[21]] "
     "describes; deferring per group at matched coverage closes it to −0.62 "
     "[−2.78, 1.37], a difference of 5.83 points with paired bootstrap "
     "*p* = 0.0004, at 85.8 % coverage overall."),

    ("P",
     "*Conditional validity.* Fitted marginally, the conformal bound held in "
     "only 14 of 23 class-by-subgroup cells, and two violations survive Holm "
     "correction: one class at a miss rate of 0.333 against a promised 0.10 "
     "under age 50, another at 0.330 against 0.20 at age 70 and over, with "
     "adjusted *p* of 5.1 × 10^{−6} and 0.029. Refitting one threshold per "
     "subgroup [[19]] restored the bound in 22 of 23 cells, at the "
     "cost that every cell now needs its own positives: one with 42 cannot "
     "support a finite threshold and is reported as unattainable."),

    ("P",
     "*Silent corruption and open-set inputs.* Simulating each of the three "
     "limb-electrode swaps on 200 test recordings, the corrupted signal passes "
     "the quality gate in 197 to 198 of them, because it is a clean recording "
     "wired wrongly; up to 87 % of diagnoses change and 7 guarantees are "
     "voided. The physiology check catches 65.5 % and 60.5 % of two swaps at "
     "4.5 % false positives, and 4.0 % of the third, which leaves the "
     "diagnostic lead untouched. Separately, 114 recordings carry a rhythm the "
     "label space cannot express, and 113 received a bounded rule-out for a "
     "disease the model has no output unit for; the irregularity gate "
     "withholds it on 48.9 % of them. The claim is withdrawn, not the "
     "diagnosis."),

    ("P",
     "*Shrinkage under imbalance.* On identical weights, the expansion in (6) "
     "lifted recall on the rarest class from 0.590 to 0.687, and seed "
     "averaging carried the worst class to 0.723. Selective "
     "prediction, which helped C1, failed here: at 88.4 % coverage worst-class "
     "recall fell to 0.706 while overall accuracy rose to 0.770. The "
     "uncertainty signal is fine, since accuracy on deferred studies is 0.426 "
     "against 0.770 on answered ones; the problem is geometric, in that one "
     "class occupies a 10-point interior band and abstention removes its "
     "members first."),

    ("P",
     "*Temporal leakage.* One comorbidity column equals 1 for every positive "
     "stay and reaches AUROC 0.9200 alone; adding it back to an otherwise safe "
     "feature set moves the screen from AUROC 0.9665 to 0.9889, which is how "
     "an apparently excellent result gets manufactured. A random split puts "
     "5,804 patients on both sides and contaminates 7,627 test rows; the "
     "patient-grouped split shares none. Under the availability contract, "
     "performance becomes a function of time (Fig. 4b): screen AUROC 0.8763, "
     "0.9121 and 0.9560, and recall on the hardest subtype 37.3 %, 58.2 % and "
     "80.0 % at *H* = 0, 6 and 24. That subtype is defined by a normal blood "
     "test, so it cannot be separated from its neighbour until the test "
     "returns. At *H* = 0 the laboratory channel carries exactly 0.0 % of the "
     "attribution mass, rising to 4.6 % and 29.6 %; a leaking pipeline cannot "
     "produce that pattern."),

    ("H2", "Turning the Same Standard on Our Own Designs"),
    ("P",
     "The mechanisms above were tested against controls. So were four of our "
     "own design decisions, and three came back negative. The sharpest is "
     "C2's architecture, which adds three things to a plain one-dimensional "
     "residual network. At three seeds each, compared by paired bootstrap on "
     "the untouched fold, they do not earn their 566 k parameters: the stem "
     "and attention pooling change nothing (*p* = 0.741), and "
     "squeeze-and-excitation on top of them costs 0.0042 macro-AUROC "
     "(*p* = 0.0040). Almost the whole loss sits on one class, +0.0147 AUROC "
     "without it and three times the next largest effect, and there is a "
     "mechanism rather than a coincidence: that diagnosis is read from QRS "
     "amplitude, and squeeze-and-excitation recalibrates channels by learned "
     "importance, an operation on relative amplitude across leads. The shipped "
     "model is still the worse one, because swapping it means refitting the "
     "calibrator and the conformal thresholds and re-verifying every figure."),

    ("P",
     "C4's infarct-wall head is the second. With every feature it reaches "
     "AUROC 0.9074 and no per-class metric below 0.7551 on 104 test cases; "
     "removing three features parsed from the recording device's own printed "
     "interpretation costs 0.133 AUROC and drops the weakest metric to 0.6038, "
     "and those three alone reach AUROC 0.841. This is not temporal leakage, "
     "since they exist at triage, but the person who assigned the diagnosis "
     "code read the same printout, so feature and label share a source and the "
     "model largely transcribes an interpretation already in the record. Both "
     "numbers belong in any report of it. Widening the head beyond two "
     "territories was measured too: a third class is recalled in 1 case of 12 "
     "and drags the other two down with it. The remaining two ablations are "
     "quieter. C3's backbone, inherited from a benchmark, was tested against "
     "the un-factorised alternative at three matched seeds and is worth "
     "keeping, though only on the classification metrics; the first version of "
     "that test was single-seed and confounded, which we found and re-ran. "
     "C2's neural report generator beats a constant sentence and a "
     "five-string lookup on BLEU-4 by 0.054, a real gain and a small one, and "
     "preserves the classifier's findings exactly in 73.9 % of reports. "
     "Neither it nor the wall head is served."),

    ("P",
     "One check of the same kind closes the loop. Research and serving code "
     "drift apart, so beside the service's 132 automated tests the radiograph "
     "endpoint was scored on 200 stratified real studies posted through the "
     "live HTTP path. Served accuracy was 0.790 "
     "[0.728, 0.841], sensitivity 0.880 [0.802, 0.930] and specificity 0.700 "
     "[0.604, 0.781], all consistent with the offline figures at 14.0 % "
     "deferred, and accuracy was 0.766 on bedside images against 0.833 on "
     "standing ones. The covariate shift reappears on real inputs through the "
     "deployed path, and the horizon contract is enforced per patient: for a "
     "single record the laboratory channel reads 0.000 % at triage."),

    # ---------------------------------------------------------------- VII
    ("H1", "Discussion and Limitations"),

    ("P",
     "Across all four modalities the useful intervention sat in the decision "
     "rule, not the representation: three attempts to close the acquisition "
     "gap by changing the model failed against a null arm, while a threshold "
     "and a deferral budget conditioned on the same variable worked at no cost "
     "in ranking quality. Much of clinical reliability is therefore "
     "post-processing, which is cheap to fit, auditable and changeable "
     "without retraining."),

    ("P",
     "The limitations are real. No component has external validation; each "
     "uses one dataset, or one pair, and every evaluation is retrospective. "
     "The C1 split is custom, its positive class enriched to 50.4 %, and "
     "98.3 % of its test images fall inside the official MIMIC-CXR training "
     "split, so those numbers are not comparable with published MIMIC-CXR "
     "results; C2 keeps only codes at full likelihood and drops 21 % of "
     "PTB-XL, so the same applies there. Training variance is measured for C2 "
     "and C3 and not for the other two. "
     "The four cohorts are disjoint, so the ordering in Section III is how the "
     "components would compose clinically and not a validated end-to-end "
     "study. Subgroup coverage is partial: C1 covers the acquisition field "
     "only, C2 sex and three age bands, the C3 cohort has no demographic "
     "fields, and C4 has them but no breakdown yet. The electrode audit rests "
     "on 200 recordings, and regular out-of-scope rhythms remain silent "
     "failures. The infarct-wall head has 104 test cases, and two either way "
     "move its weakest metric across 0.75. C3 does not reach its 0.75 "
     "worst-class target. Grad-CAM is a sanity "
     "check, not evidence of localisation, its repeatability on chest "
     "radiographs measured at a structural similarity of 0.12 [[27]]. This "
     "is not a medical device."),

    # --------------------------------------------------------------- VIII
    ("H1", "Conclusion"),
    ("P",
     "We built a four-modality clinical decision-support system in which every "
     "component has an explicit rule for declining to answer, and every rule "
     "is measured against a control. Conditioning that rule on whatever "
     "variable actually causes the failure, an acquisition setting, a patient "
     "subgroup, a rhythm outside the label space, or the time a feature became "
     "available, beat changing the model in every case we tested and usually "
     "cost nothing in ranking quality. The same standard applied to our own "
     "design decisions returned three negative results out of four. Reducing "
     "four refusal vocabularies to one five-level field then lets client code "
     "apply a single rule everywhere. Next are external validation, multi-seed "
     "training for the two components that lack it, subgroup analysis on "
     "whatever axes each cohort supports, and a paired study on the 19,979 "
     "patients two of our cohorts share."),
]

REFERENCES = [
    # 1
    "R. A. Byrne et al., “2023 ESC guidelines for the management of acute coronary syndromes,” Eur. Heart J., vol. 44, no. 38, pp. 3720–3826, 2023.",
    # 2
    "M. Gulati et al., “2021 AHA/ACC/ASE/CHEST/SAEM/SCCT/SCMR guideline for the evaluation and diagnosis of chest pain,” Circulation, vol. 144, no. 22, pp. e368–e454, 2021.",
    # 3
    "A. E. W. Johnson et al., “MIMIC-CXR-JPG, a large publicly available database of labeled chest radiographs,” arXiv:1901.07042, 2019.",
    # 4
    "Z. Liu, H. Mao, C.-Y. Wu, C. Feichtenhofer, T. Darrell, and S. Xie, “A ConvNet for the 2020s,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2022, pp. 11976–11986.",
    # 5
    "H. Yuan, Z. Yuan, R. Gan, J. Zhang, Y. Xie, and S. Yu, “BioBART: Pretraining and evaluation of a biomedical generative language model,” in Proc. BioNLP Workshop, 2022, pp. 97–109.",
    # 6
    "P. Wagner et al., “PTB-XL, a large publicly available electrocardiography dataset,” Sci. Data, vol. 7, art. 154, 2020.",
    # 7
    "D. Ouyang et al., “Video-based AI for beat-to-beat assessment of cardiac function,” Nature, vol. 580, pp. 252–256, 2020.",
    # 8
    "D. Tran, H. Wang, L. Torresani, J. Ray, Y. LeCun, and M. Paluri, “A closer look at spatiotemporal convolutions for action recognition,” in Proc. IEEE/CVF Conf. Comput. Vis. Pattern Recognit. (CVPR), 2018, pp. 6450–6459.",
    # 9
    "S. Leclerc et al., “Deep learning for segmentation using an open large-scale dataset in 2D echocardiography,” IEEE Trans. Med. Imag., vol. 38, no. 9, pp. 2198–2210, 2019.",
    # 10
    "G. Ke et al., “LightGBM: A highly efficient gradient boosting decision tree,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2017, pp. 3146–3154.",
    # 11
    "S. M. Lundberg and S.-I. Lee, “A unified approach to interpreting model predictions,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2017, pp. 4765–4774.",
    # 12
    "A. E. W. Johnson et al., “MIMIC-IV-ED, a large, publicly available database of emergency department electronic health records,” Sci. Data, vol. 10, art. 1, 2023.",
    # 13
    "R. R. Selvaraju, M. Cogswell, A. Das, R. Vedantam, D. Parikh, and D. Batra, “Grad-CAM: Visual explanations from deep networks via gradient-based localization,” in Proc. IEEE Int. Conf. Comput. Vis. (ICCV), 2017, pp. 618–626.",
    # 14
    "L. Seyyed-Kalantari, G. Liu, M. McDermott, I. Y. Chen, and M. Ghassemi, “CheXclusion: Fairness gaps in deep chest X-ray classifiers,” in Proc. Pacific Symp. Biocomputing (PSB), 2021, pp. 232–243.",
    # 15
    "M. Hardt, E. Price, and N. Srebro, “Equality of opportunity in supervised learning,” in Proc. Adv. Neural Inf. Process. Syst. (NIPS), 2016, pp. 3315–3323.",
    # 16
    "S. C. Pereira, J. Rocha, A. Gaudio, A. Smailagic, A. Campilho, and A. M. Mendonça, “Addressing chest radiograph projection bias in deep classification models,” in Proc. Med. Imag. Deep Learn. (MIDL), PMLR, vol. 227, 2023, pp. 1199–1210.",
    # 17
    "C. K. Chow, “On optimum recognition error and reject tradeoff,” IEEE Trans. Inf. Theory, vol. 16, no. 1, pp. 41–46, 1970.",
    # 18
    "V. Vovk, “Conditional validity of inductive conformal predictors,” in Proc. Asian Conf. Mach. Learn. (ACML), PMLR, vol. 25, 2012, pp. 475–490.",
    # 19
    "V. Vovk, D. Lindsay, I. Nouretdinov, and A. Gammerman, “Mondrian confidence machine,” Tech. Rep., 2003.",
    # 20
    "C. Guo, G. Pleiss, Y. Sun, and K. Q. Weinberger, “On calibration of modern neural networks,” in Proc. Int. Conf. Mach. Learn. (ICML), 2017, pp. 1321–1330.",
    # 21
    "E. Jones, S. Sagawa, P. W. Koh, A. Kumar, and P. Liang, “Selective classification can magnify disparities across groups,” in Proc. Int. Conf. Learn. Represent. (ICLR), 2021.",
    # 22
    "W. Cao, V. Mirjalili, and S. Raschka, “Rank consistent ordinal regression for neural networks with application to age estimation,” Pattern Recognit. Lett., vol. 140, pp. 325–331, 2020.",
    # 23
    "K. Cao, C. Wei, A. Gaidon, N. Arechiga, and T. Ma, “Learning imbalanced datasets with label-distribution-aware margin loss,” in Proc. Adv. Neural Inf. Process. Syst. (NeurIPS), 2019, pp. 1567–1578.",
    # 24
    "S. Kaufman, S. Rosset, C. Perlich, and O. Stitelman, “Leakage in data mining: Formulation, detection, and avoidance,” ACM Trans. Knowl. Discovery Data, vol. 6, no. 4, art. 15, 2012.",
    # 25
    "Q. McNemar, “Note on the sampling error of the difference between correlated proportions or percentages,” Psychometrika, vol. 12, no. 2, pp. 153–157, 1947.",
    # 26
    "S. Holm, “A simple sequentially rejective multiple test procedure,” Scand. J. Statist., vol. 6, no. 2, pp. 65–70, 1979.",
    # 27
    "N. Arun et al., “Assessing the trustworthiness of saliency maps for localizing abnormalities in medical imaging,” Radiol. Artif. Intell., vol. 3, no. 6, art. e200267, 2021.",
]
