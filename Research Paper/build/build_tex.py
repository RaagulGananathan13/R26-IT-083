# -*- coding: utf-8 -*-
"""Render paper_content.py into IEEEtran LaTeX (A4, conference)."""
import os
import re
import shutil

import paper_content as C

PAPER_DIR = r"c:\Users\94775\Desktop\box\R26-IT-083\Research Paper"
TEX_DIR = os.path.join(PAPER_DIR, "latex")
os.makedirs(os.path.join(TEX_DIR, "figures"), exist_ok=True)
for f in ("fig1_architecture.png", "fig2_per_class.png",
          "fig3_training.png", "fig4_results.png"):
    shutil.copyfile(os.path.join(PAPER_DIR, "figures", f),
                    os.path.join(TEX_DIR, "figures", f))

# ------------------------------------------------------------ text mapping
SPECIALS = [("\\", r"\textbackslash{}"), ("&", r"\&"), ("%", r"\%"),
            ("#", r"\#"), ("_", r"\_"), ("{", r"\{"), ("}", r"\}")]

UNICODE = [
    ("\u2212", "$-$"), ("\u2265", "$\\ge$"), ("\u2264", "$\\le$"),
    ("\u00d7", "$\\times$"), ("\u2248", "$\\approx$"),
    ("\u03b1", "$\\alpha$"), ("\u03b2", "$\\beta$"), ("\u03b4", "$\\delta$"),
    ("\u03c3", "$\\sigma$"), ("\u03c4", "$\\tau$"), ("\u03ba", "$\\kappa$"),
    ("\u03c1", "$\\rho$"), ("\u03a6", "$\\Phi$"), ("\u03a3", "$\\Sigma$"),
    ("\u2014", "---"), ("\u2013", "--"),
    ("\u201c", "``"), ("\u201d", "''"), ("\u2019", "'"), ("\u2018", "`"),
    ("\u00e7", "\\c{c}"), ("\u00e9", "\\'e"), ("\u00f1", "\\~n"),
    ("\u00a0", "~"),
]


def esc(t):
    for a, b in SPECIALS:
        t = t.replace(a, b)
    for a, b in UNICODE:
        t = t.replace(a, b)
    return t


SUPER = re.compile(r"\^\{[^}]*\}")


def inline(t):
    """Escape, then apply *italic*, [[n]] citations and ^{...} superscripts."""
    parts = []
    for piece in SUPER.split(t):
        piece = esc(piece)
        piece = re.sub(r"\[\[(\d+)\]\]",
                       lambda m: "\\cite{ref%s}" % m.group(1), piece)
        piece = re.sub(r"\*([^*]+)\*",
                       lambda m: "\\textit{%s}" % m.group(1), piece)
        parts.append(piece)
    sups = [m.group(0)[2:-1].replace("\u2212", "-")
            for m in SUPER.finditer(t)]
    out = parts[0]
    for sup, nxt in zip(sups, parts[1:]):
        out += "$^{%s}$" % sup + nxt
    return out


# ------------------------------------------------------------------ header
L = []
add = L.append

add(r"% Conference paper, IEEEtran, A4, double-blind version.")
add(r"% Generated from paper_content.py; the Word file is the same text.")
add(r"\documentclass[conference,a4paper]{IEEEtran}")
add(r"\IEEEoverridecommandlockouts")
add(r"\usepackage{cite}")
add(r"\usepackage{amsmath,amssymb,amsfonts}")
add(r"\usepackage{graphicx}")
add(r"\usepackage{array}")
add(r"\usepackage{textcomp}")
add(r"\usepackage[T1]{fontenc}")
add(r"\usepackage[utf8]{inputenc}")
add(r"\newlength{\tblw}")
add(r"\def\BibTeX{{\rm B\kern-.05em{\sc i\kern-.025em b}\kern-.08em"
    r"T\kern-.1667em\lower.7ex\hbox{E}\kern-.125emX}}")
add(r"\begin{document}")
add("")
add(r"\title{%s}" % esc(C.TITLE))
add("")

# authors
if getattr(C, "ANONYMOUS", False):
    add(r"\author{\IEEEauthorblockN{Anonymous Author(s)}")
    add(r"\IEEEauthorblockA{\textit{Affiliation withheld for "
        r"double-blind review}}}")
else:
    add(r"\author{")
    blocks = []
    for name, dept, org, city, sid in C.AUTHORS:
        blocks.append(
            "\\IEEEauthorblockN{%s}\n"
            "\\IEEEauthorblockA{\\textit{%s} \\\\\n"
            "\\textit{%s} \\\\\n"
            "%s \\\\\n"
            "{%s}}" % (esc(name), esc(dept), esc(org), esc(city), esc(sid)))
    add("\n\\and\n".join(blocks))
    add(r"}")
add("")
add(r"\maketitle")
add("")
add(r"\begin{abstract}")
add(inline(C.ABSTRACT))
add(r"\end{abstract}")
add("")
add(r"\begin{IEEEkeywords}")
add(esc(C.INDEX_TERMS))
add(r"\end{IEEEkeywords}")
add("")

# ------------------------------------------------------------------- body
fig_n = 0
tab_n = 0
for kind, payload in C.BODY:
    if kind == "H1":
        add(r"\section{%s}" % esc(payload))
    elif kind == "H2":
        add(r"\subsection{%s}" % esc(payload))
    elif kind == "P":
        add(inline(payload))
        add("")
    elif kind == "LIST":
        add(r"\begin{itemize}")
        for it in payload:
            add(r"\item %s" % inline(it))
        add(r"\end{itemize}")
        add("")
    elif kind == "EQ":
        add(r"\begin{equation}")
        add(payload[1])
        add(r"\end{equation}")
        add("")
    elif kind == "FIG":
        fname, cap, span = payload
        fig_n += 1
        env = "figure*" if span else "figure"
        w = r"\textwidth" if span else r"\columnwidth"
        add(r"\begin{%s}[!t]" % env)
        add(r"\centering")
        add(r"\includegraphics[width=%s]{figures/%s}" % (w, fname))
        add(r"\caption{%s}" % inline(cap))
        add(r"\label{fig:%d}" % fig_n)
        add(r"\end{%s}" % env)
        add("")
    elif kind == "TABLE":
        spec = payload
        tab_n += 1
        span = spec.get("span", False)
        env = "table*" if span else "table"
        n = len(spec["cols"])
        add(r"\begin{%s}[!t]" % env)
        add(r"\caption{%s}" % esc(spec["caption"]))
        add(r"\label{tab:%d}" % tab_n)
        add(r"\centering")
        add(r"\setlength{\tabcolsep}{3pt}")
        add(r"\setlength{\tblw}{\dimexpr%s-%d\tabcolsep\relax}"
            % (r"\textwidth" if span else r"\columnwidth", 2 * n))
        colspec = "|" + "|".join(
            ">{\\raggedright\\arraybackslash}p{%.3f\\tblw}" % w
            for w in spec["widths"]) + "|"
        add(r"\footnotesize")
        add(r"\begin{tabular}{%s}" % colspec)
        add(r"\hline")
        add(" & ".join(r"\textbf{%s}" % esc(c) for c in spec["cols"]) + r" \\")
        add(r"\hline")
        for row in spec["rows"]:
            add(" & ".join(inline(c) for c in row) + r" \\")
            add(r"\hline")
        add(r"\end{tabular}")
        add(r"\end{%s}" % env)
        add("")

# ------------------------------------------------------------- references
add(r"\begin{thebibliography}{00}")
for i, r in enumerate(C.REFERENCES, 1):
    add(r"\bibitem{ref%d} %s" % (i, esc(r)))
add(r"\end{thebibliography}")
add("")
add(r"\end{document}")

out = os.path.join(TEX_DIR, "main.tex")
with open(out, "w", encoding="utf-8") as fh:
    fh.write("\n".join(L) + "\n")
print("saved", out)

# quick sanity checks
txt = "\n".join(L)
bad = set(re.findall(r"[^\x00-\x7F]", txt))
print("non-ASCII characters left:", sorted(bad) if bad else "none")
print("figures:", fig_n, " tables:", tab_n,
      " citations:", len(set(re.findall(r"\\cite\{ref(\d+)\}", txt))))
