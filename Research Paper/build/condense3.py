# -*- coding: utf-8 -*-
"""Third pass: remove restatement.

Table I and Figs. 2 and 3 now carry the headline deltas, so the Results prose
stops repeating them and keeps only what a table cannot hold: the control arm,
the statistic, and the mechanism.
"""
import io
import sys

sys.path.insert(0, '.')
import paper_content as C
import edit_body as EB

NEW = {
'*Acquisition shift.*': (
    "*Acquisition shift.* C1 scores AUROC 0.8224 on bedside images against "
    "0.8864 on standing ones, a gap of 0.0639 [0.0491, 0.0790] in the same "
    "direction for all eight labels. Fitting the operating point per group "
    "cut the reported true-positive-rate disparity by 73.3 % with an AUROC "
    "spread of exactly zero and no discernible accuracy cost (+0.02 points, "
    "McNemar mid-*p* 0.885). We also reimplemented the representation-side "
    "alternative [[17]] on our own data, backbone and split: it reached "
    "complete invariance, projection-detection AUC 0.5000, and still made the "
    "disparity 25.4 % *worse* at a cost of 0.0789 AUROC. Deferral behaves the "
    "same way (Fig. 3a): deferring uniformly leaves the gap at 6.28, the "
    "behaviour [[22]] describes, while deferring per group at matched "
    "coverage closes it, a difference of 5.83 points with paired bootstrap "
    "*p* = 0.0004."),

'*Conditional validity.*': (
    "*Conditional validity.* Fitted marginally, the conformal bound held in "
    "only 14 of 23 class-by-subgroup cells, and two violations survive Holm "
    "correction: miss rates of 0.333 against a promised 0.10 under age 50, "
    "and 0.330 against 0.20 at age 70 and over. Refitting one threshold per "
    "subgroup [[20]] restored the bound in 22 of 23, at the expense that "
    "every cell now needs its own positives: one cell with 42 cannot support "
    "a finite threshold and is reported unattainable."),

'*Silent corruption': (
    "*Silent corruption and open-set inputs.* Simulating each of the three "
    "limb-electrode swaps on 200 test recordings, the corrupted signal passes "
    "the quality gate in 197 to 198 of them, because the recording is clean "
    "but wired wrongly; up to 87 % of diagnoses change and 7 guarantees are "
    "voided. The physiology check detects 65.5 % and 60.5 % of two swaps at "
    "4.5 % false positives, and 4.0 % of the third. Separately, 114 "
    "recordings carry a rhythm the label space cannot represent, and the "
    "irregularity gate withholds the bounded claim on 48.9 % of them."),

'*Shrinkage under imbalance.*': (
    "*Shrinkage under imbalance.* On identical weights, the expansion in (8) "
    "lifted recall on the rarest class from 0.590 to 0.687, and seed "
    "averaging carried the worst class to 0.723. Selective prediction, which "
    "helped C1, failed here: at 88.4 % coverage worst-class recall fell to "
    "0.706 while overall accuracy rose. The uncertainty signal is sound, "
    "since accuracy on deferred studies is 0.426 against 0.770 on answered "
    "ones; the problem is geometric, in that one class occupies a 10-point "
    "interior band and abstention removes its members first."),

'*Temporal leakage.*': (
    "*Temporal leakage.* One comorbidity column equals 1 for every positive "
    "stay and reaches AUROC 0.9200 alone; adding it back to an otherwise safe "
    "feature set moves the screen from 0.9665 to 0.9889, which is how an "
    "apparently outstanding result gets manufactured. A random split places "
    "5,804 patients on both sides and contaminates 7,627 test rows; the "
    "patient-grouped split shares none. Under the availability contract "
    "performance becomes a function of time (Fig. 3b), and recall on the "
    "hardest subtype moves 37.3 %, 58.2 %, 80.0 % at *H* = 0, 6, 24. At *H* = "
    "0 the laboratory channel carries exactly 0.0 % of the attribution mass, "
    "rising to 4.6 % and 29.6 %; a leaking pipeline cannot produce that "
    "pattern."),

'Accuracy on the answered subset': (
    "Accuracy on the answered subset flatters any system that abstains, so we "
    "report one measure that means the same thing in every modality. The "
    "*unsafe answer rate* is the probability that the system both answers and "
    "is wrong, U = *c* (1 - *A*) for coverage *c* and accuracy *A* among "
    "answered cases; a component that never abstains has U equal to its error "
    "rate. Reported with coverage it prices abstention instead of concealing "
    "it. Fig. 2 gives all three components. The nuance is C1: uniform "
    "deferral reaches a marginally lower U than the group-conditional arm, "
    "8.89 % against 9.64 % at the same budget, yet leaves the acquisition gap "
    "intact, which is exactly the trade the contract is meant to make "
    "visible. C3's 148 deferred studies are genuinely the hard ones, scoring "
    "42.6 % against 73.0 % overall, and C4 buys its drop to 6.68 % by "
    "deferring a third of subtyping decisions. Answering RQ3, abstention "
    "lowered the unsafe answer rate against the answer-everything arm in all "
    "three components, and in C1 only the group-conditional arm also removed "
    "the failure mode."),

'The mechanisms above were tested': (
    "The mechanisms above were tested against controls. So were four of our "
    "own design choices, and three came back negative. The sharpest is C2's "
    "architecture, which adds three things to a plain one-dimensional "
    "residual network. At three seeds each, compared by paired bootstrap on "
    "the untouched fold, they do not earn their 566 k parameters: the stem "
    "and attention pooling change nothing (*p* = 0.741), and "
    "squeeze-and-excitation costs 0.0042 macro-AUROC (*p* = 0.0040). Almost "
    "the whole loss sits on one class, +0.0147 AUROC without it, and there is "
    "a mechanism rather than a coincidence: that diagnosis is read from QRS "
    "amplitude, and squeeze-and-excitation recalibrates channels by learned "
    "importance, an operation on relative amplitude across leads."),

"C4's infarct-wall head": (
    "C4's infarct-wall head is the second. With every feature it reaches "
    "AUROC 0.9074 on 104 test cases; removing three features parsed from the "
    "printed interpretation of the recording device costs 0.133 AUROC, and "
    "those three alone reach AUROC 0.841. This is not temporal leakage, since "
    "they exist at triage, but the person who assigned the diagnosis code "
    "read the same printout. Feature and label therefore share a source: the "
    "label is partly defined by an input the model is given, a circularity no "
    "timestamp check can detect. Widening the head beyond two territories was "
    "measured too, and a third class is recalled in 1 case of 12. C3's "
    "backbone was tested against the un-factorised alternative at three "
    "matched seeds and is worth keeping, though only on the classification "
    "metrics. Text generation exists in two components but is not evaluated "
    "here, and neither it nor the wall head is served."),

'Detection accuracy is a precondition': (
    "Detection accuracy is a precondition rather than the contribution, so we "
    "state it once. On its own test fold each component reaches: C1 "
    "cardiomegaly AUROC 0.9189 at 92.3 % sensitivity, n = 4,722; C2 macro "
    "accuracy 0.864 and recall 0.810, n = 1,711; C3 mean absolute error 3.979 "
    "ejection-fraction points and worst-class recall 0.723, n = 1,277; C4 "
    "screening AUROC 0.9560 at 99.41 % negative predictive value and "
    "subtyping macro-F1 0.7448, n = 30,452. These are four tasks on four "
    "cohorts: the rows are not comparable with one another, and none is "
    "compared with a published benchmark, for the split reasons in Section "
    "VII."),

'Answering RQ1 and RQ3': (
    "Answering RQ1 and RQ3 together: in these experiments the effective "
    "intervention was in the decision layer rather than the representation. "
    "Three attempts to close the acquisition gap by altering the model failed "
    "against a null arm, while a threshold and a deferral budget conditioned "
    "on the same variable worked, and the unsafe answer rate fell in every "
    "component where coverage is measurable. We do not conclude that clinical "
    "reliability is generally post-processing. We draw a narrower conclusion: "
    "on these four tasks, several reliability problems were addressed "
    "effectively by decision-layer controls that are inexpensive to fit and "
    "auditable, and conditioning each control on the variable that truly "
    "causes the failure mattered more than the strength of the control. RQ2 "
    "is answered by construction and tested through use: four mechanisms with "
    "nothing in common reduced to a single five-state field without any of "
    "them losing information, since the component-native payload is returned "
    "unaltered alongside it."),

'The states are ordered': (
    "The states are ordered from most to least usable, and that ordering is "
    "the whole interface: a caller applies one rule, do not act on a result "
    "that is not actionable, without knowing what a projection, a conformal "
    "zone or a disclosure horizon is. Assignment is a precedence cascade over "
    "four signals a component already computes: whether it ran, *r*; whether "
    "the input passed its quality and verification gates, *q*; an uncertainty "
    "statistic *u* against a threshold *tau* fitted on validation and frozen; "
    "and whether the validity conditions of the reported reliability hold "
    "here, *v*. Numbering the states 0 to 4 in the order above, component *m* "
    "returns"),
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
