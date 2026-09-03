# -*- coding: utf-8 -*-
"""Edit BODY by index and rewrite paper_content.py from the structure itself.

String surgery on the wrapped source kept missing; this loads the module,
changes the entries, and serialises the whole thing back. Formatting becomes
machine-made, which is fine for a build script.
"""
import io
import sys

sys.path.insert(0, '.')
import paper_content as C


def wrap(text, indent, width=74):
    """Emit a python string literal broken into short quoted lines."""
    words, lines, cur = text.split(), [], ''
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = (cur + ' ' + w).strip()
    if cur:
        lines.append(cur)
    pad = ' ' * indent
    out = []
    for i, l in enumerate(lines):
        l = l.replace('\\', '\\\\').replace('"', '\\"')
        tail = ' ' if i < len(lines) - 1 else ''
        out.append('%s"%s%s"' % (pad if i else '', l, tail))
    return '\n'.join(out)


def dump(body, refs):
    o = ['# -*- coding: utf-8 -*-',
         '"""Single source of truth for the paper. Rendered by build_docx.py '
         'and build_tex.py."""',
         '',
         'ANONYMOUS = True',
         '',
         'TITLE = %r' % C.TITLE,
         '',
         'AUTHORS = %r' % getattr(C, 'AUTHORS', []),
         '',
         'ABSTRACT = (',
         wrap(C.ABSTRACT, 4),
         ')',
         '',
         'INDEX_TERMS = (',
         wrap(C.INDEX_TERMS, 4),
         ')',
         '',
         'BODY = [']
    for kind, payload in body:
        if kind in ('H1', 'H2'):
            o.append('    (%r, %r),' % (kind, payload))
        elif kind == 'P':
            o.append('    ("P",')
            o.append(wrap(payload, 5))
            o.append('     ),')
        elif kind == 'LIST':
            o.append('    ("LIST", [')
            for it in payload:
                o.append(wrap(it, 8))
                o.append('        ,')
            o.append('    ]),')
        elif kind == 'EQ':
            o.append('    ("EQ", (%r,\n            %r)),' % payload)
        elif kind == 'FIG':
            o.append('    ("FIG", (%r,\n             %r,\n             %r)),'
                     % payload)
        elif kind == 'TABLE':
            o.append('    ("TABLE", %r),' % (payload,))
        o.append('')
    o.append(']')
    o.append('')
    o.append('REFERENCES = [')
    for i, r in enumerate(refs, 1):
        o.append('    # %d' % i)
        o.append('    %r,' % r)
    o.append(']')
    return '\n'.join(o) + '\n'


# ---------------------------------------------------------------- edits ---
body = list(C.BODY)
EDITS = {}

for i, (k, v) in enumerate(body):
    if k != 'P':
        continue
    t = v
    if t.startswith('*Silent corruption'):
        EDITS[i] = (
            "*Silent corruption and open-set inputs.* Simulating each of the "
            "three limb-electrode swaps on 200 test recordings, the corrupted "
            "signal passes the quality gate in 197 to 198 of them, because it "
            "is a clean recording wired wrongly; up to 87 % of diagnoses "
            "change and 7 guarantees are voided. The physiology check catches "
            "65.5 % and 60.5 % of two swaps at 4.5 % false positives, and 4.0 "
            "% of the third. Separately, 114 recordings carry a rhythm the "
            "label space cannot express and 113 received a bounded rule-out "
            "for a disease the model has no output unit for; the irregularity "
            "gate withholds the claim on 48.9 % of them.")
    elif t.startswith('A bound like this is conditional'):
        EDITS[i] = (
            "A bound like this is conditional on assumptions the input can "
            "break without looking broken, so two checks withdraw the "
            "guarantee while leaving the prediction in place. The first flags "
            "a swapped pair of limb electrodes, an exact linear map of the "
            "standard lead definitions, from the polarity of one lead and the "
            "inversion of another; the second asks whether the rhythm is "
            "inside the label space at all, from a beat-interval irregularity "
            "score thresholded on validation at a 5 % false-positive budget. "
            "Either way the probabilities are still returned and only the "
            "bounded-miss-rate claim is withdrawn, which is the *caution* "
            "state of (1) rather than *withheld*.")
    elif t.startswith('since the coverage of the'):
        EDITS[i] = (
            "since the coverage of the *k*-th order statistic follows "
            "Beta(*k*, *n*-*k*+1). The rule-out threshold is the *m**-th "
            "smallest calibration score among positives, the rule-in "
            "threshold its mirror over the negatives at a false-alarm budget "
            "*beta*, and anything between them is referred. A class with too "
            "few calibration positives is reported unattainable rather than "
            "approximated. Two models are served side by side, each with its "
            "own calibrator and thresholds, and a class is ruled out only "
            "when both rule it out, so the merged miss rate is bounded by the "
            "tighter single-model bound at the cost of more referrals.")
    elif t.startswith('so the cumulative probabilities can never cross'):
        EDITS[i] = (
            "so the cumulative probabilities can never cross. Training uses a "
            "class-balanced sampler with deferred re-weighting [[24]] from "
            "epoch 15 and an exponential moving average of the weights. The "
            "second cohort is intensity-matched before being mixed in, since "
            "a balanced sampler over-draws from it and would let the network "
            "use scanner brightness as a shortcut for severity. A regressor "
            "on a skewed target also shrinks predictions towards the mean, "
            "pushing the severe tail over the boundary at 30, so an expansion "
            "is fitted on validation and applied without touching the "
            "weights,")
    elif t.startswith('after which the boundaries are re-optimised'):
        EDITS[i] = (
            "after which the boundaries are re-optimised on validation "
            "lexicographically: worst-class recall, then balanced accuracy, "
            "then macro-F1. At inference a study is sampled into ten clips "
            "and averaged over three seeds, and the interval is "
            "split-conformal, widened by the learned aleatoric term and by "
            "disagreement between clips.")
    elif t.startswith('with the recall floor'):
        EDITS[i] = (
            "with the recall floor *rho* = 0.75, solved on validation over "
            "bootstrap resamples and then frozen. A case whose top-two margin "
            "falls below the (1 - *C*) quantile of the validation margins is "
            "referred to a clinician instead of being subtyped. Results are "
            "reported on the intended-use population, meaning visits with a "
            "cardiac complaint or an early electrocardiogram order; both are "
            "observable at triage, so this is selection, not leakage. A "
            "separate head asks which wall of the heart the infarct involves, "
            "on a label rebuilt from ICD-9 and ICD-10 diagnosis codes.")

for i, new in EDITS.items():
    body[i] = (body[i][0], new)

io.open('paper_content.py', 'w', encoding='utf-8').write(dump(body, C.REFERENCES))
print("rewrote %d paragraphs" % len(EDITS))
