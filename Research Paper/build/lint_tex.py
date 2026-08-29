# -*- coding: utf-8 -*-
"""Static checks on main.tex, since there is no LaTeX engine on this machine."""
import io
import os
import re

TEX = r"c:\Users\94775\Desktop\box\R26-IT-083\Research Paper\latex\main.tex"
ROOT = os.path.dirname(TEX)

def strip_comment(ln):
    """Remove a LaTeX comment, respecting escaped percent signs."""
    out = []
    i = 0
    while i < len(ln):
        if ln[i] == "\\" and i + 1 < len(ln):
            out.append(ln[i:i + 2])
            i += 2
            continue
        if ln[i] == "%":
            break
        out.append(ln[i])
        i += 1
    return "".join(out)


src = io.open(TEX, encoding="utf-8").read()
lines = src.split("\n")
problems = []

# 1. environments balance
stack = []
for i, ln in enumerate(lines, 1):
    body = strip_comment(ln)
    for m in re.finditer(r"\\begin\{([^}]+)\}", body):
        stack.append((m.group(1), i))
    for m in re.finditer(r"\\end\{([^}]+)\}", body):
        if not stack:
            problems.append("line %d: \\end{%s} with nothing open" % (i, m.group(1)))
        else:
            name, ln0 = stack.pop()
            if name != m.group(1):
                problems.append("line %d: \\end{%s} closes \\begin{%s} (line %d)"
                                % (i, m.group(1), name, ln0))
if stack:
    problems.append("unclosed: " + ", ".join("%s (line %d)" % s for s in stack))

# 2. brace balance, ignoring escaped braces
depth = 0
for i, ln in enumerate(lines, 1):
    body = strip_comment(ln)
    body = body.replace(r"\{", "").replace(r"\}", "")
    depth += body.count("{") - body.count("}")
    if depth < 0:
        problems.append("line %d: closing brace with none open" % i)
        depth = 0
if depth:
    problems.append("unbalanced braces at end of file: %+d" % depth)

# 3. math-mode dollars balance per line
for i, ln in enumerate(lines, 1):
    body = strip_comment(ln)
    if body.replace(r"\$", "").count("$") % 2:
        problems.append("line %d: odd number of $" % i)

# 4. unescaped percent signs in body text
for i, ln in enumerate(lines, 1):
    if ln.startswith("%"):
        continue
    for m in re.finditer(r"(?<!\\)%", ln):
        problems.append("line %d: unescaped %% -> '%s'" % (i, ln[max(0, m.start() - 25):m.start() + 5]))

# 5. graphics exist
for m in re.finditer(r"\\includegraphics\[[^\]]*\]\{([^}]+)\}", src):
    p = os.path.join(ROOT, m.group(1))
    if not os.path.exists(p):
        problems.append("missing graphic: " + m.group(1))

# 6. every \cite has a \bibitem
cited = set(re.findall(r"\\cite\{([^}]+)\}", src))
cited = {c for group in cited for c in group.split(",")}
items = set(re.findall(r"\\bibitem\{([^}]+)\}", src))
for c in sorted(cited - items):
    problems.append("cite without bibitem: " + c)
for b in sorted(items - cited):
    problems.append("bibitem never cited: " + b)

# 7. required packages for commands used
need = {r"\rowcolor": "xcolor", r"\arraybackslash": "array",
        r"\mathbf": "amsmath", r"\mathcal": "amsmath",
        r"\includegraphics": "graphicx"}
for cmd, pkg in need.items():
    if cmd in src and pkg not in src:
        problems.append("uses %s but does not load %s" % (cmd, pkg))

print("environments checked, %d cite keys, %d bibitems" % (len(cited), len(items)))
print("PROBLEMS:" if problems else "NO PROBLEMS FOUND")
for p in problems:
    print("  -", p)
