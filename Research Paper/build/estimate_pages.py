# -*- coding: utf-8 -*-
"""Estimate the rendered page count of the paper.

Wraps every paragraph with real Times New Roman glyph widths, then adds up
column-points and divides by the capacity of an IEEE A4 two-column page.
"""
import re
from PIL import ImageFont, Image, ImageDraw

import paper_content as C

PAGE_H = 841.90
M_TOP, M_BOT, M_SIDE = 54.0, 72.0, 45.35
PAGE_W = 595.30
GUTTER = 18.0
COL_W = (PAGE_W - 2 * M_SIDE - GUTTER) / 2.0
FULL_W = PAGE_W - 2 * M_SIDE
TEXT_H = PAGE_H - M_TOP - M_BOT                      # 715.9 pt per column

REG = ImageFont.truetype(r"C:\Windows\Fonts\times.ttf", 100)
BOLD = ImageFont.truetype(r"C:\Windows\Fonts\timesbd.ttf", 100)
ITAL = ImageFont.truetype(r"C:\Windows\Fonts\timesi.ttf", 100)
_img = Image.new("L", (10, 10))
_d = ImageDraw.Draw(_img)


def width_pt(text, size, font=REG):
    return _d.textlength(text, font=font) * size / 100.0


def line_h(size):
    """Word single line spacing for Times: about 1.15 em."""
    return size * 1.15


def wrap_lines(text, size, width, font=REG, first_indent=0.0):
    words = text.split()
    lines, cur, avail = 0, "", width - first_indent
    for w in words:
        trial = (cur + " " + w) if cur else w
        if width_pt(trial, size, font) <= avail:
            cur = trial
        else:
            lines += 1
            cur = w
            avail = width
    if cur:
        lines += 1
    return max(lines, 1)


CLEAN = re.compile(r"\[\[(\d+)\]\]")


def plain(t):
    return CLEAN.sub(r"[\1]", t).replace("*", "")


total = 0.0          # column-points consumed
detail = {}


def add(section, pts):
    global total
    total += pts
    detail[section] = detail.get(section, 0) + pts


# ---- abstract + index terms (9 pt bold, single column)
sec = "front"
add(sec, 10 + wrap_lines("Abstract-" + plain(C.ABSTRACT), 9, COL_W,
                         BOLD, 13.7) * line_h(9))
add(sec, 6 + 8 + wrap_lines("Index Terms-" + C.INDEX_TERMS, 9, COL_W,
                            BOLD, 13.7) * line_h(9))

section = "I"
for kind, payload in C.BODY:
    if kind == "H1":
        section = payload
        add(section, 8 + 4 + line_h(10))
    elif kind == "H2":
        add(section, 6 + 3 + line_h(10))
    elif kind == "P":
        add(section, wrap_lines(plain(payload), 10, COL_W, REG, 14.4)
            * line_h(10))
    elif kind == "LIST":
        for it in payload:
            add(section, wrap_lines("- " + plain(it), 10, COL_W - 11.5)
                * line_h(10))
    elif kind == "EQ":
        add(section, 4 + 4 + line_h(10))
    elif kind == "FIG":
        fname, cap, span = payload
        from PIL import Image as I
        im = I.open(r"c:\Users\94775\Desktop\box\R26-IT-083\Research Paper"
                    r"\figures\\" + fname)
        w = FULL_W if span else COL_W
        h = im.height * (w / im.width)
        cap_h = wrap_lines("Fig. 1. " + plain(cap), 8, w) * line_h(8)
        block = 6 + h + 2 + cap_h + 8
        add(section, block * (2 if span else 1))
    elif kind == "TABLE":
        spec = payload
        span = spec.get("span", False)
        w = FULL_W if span else COL_W
        h = 8 + line_h(8) + 2 + line_h(8) + 3        # two caption lines
        cells = [spec["cols"]] + spec["rows"]
        for r in cells:
            rh = 0
            for j, val in enumerate(r):
                cw = w * spec["widths"][j] - 5
                rh = max(rh, wrap_lines(plain(val), 8, cw) * line_h(8))
            h += rh + 1.5
        h += 4
        add(section, h * (2 if span else 1))

# ---- references (8 pt)
add("refs", 8 + 4 + line_h(10))
for i, r in enumerate(C.REFERENCES, 1):
    add("refs", wrap_lines("[%d] %s" % (i, r), 8, COL_W - 17) * line_h(8) + 1)

# ---- capacity
title_block = (30 + 6 + line_h(11) + line_h(10) + 10
               if getattr(C, "ANONYMOUS", False)
               else 30 + 6 + (line_h(11) + 4 * line_h(10)) + 10)
cap_page1 = (TEXT_H - title_block) * 2
cap_rest = TEXT_H * 2
pages = 1 + (total - cap_page1) / cap_rest

print("column-points used: %.0f" % total)
print("page-1 capacity %.0f, later pages %.0f each" % (cap_page1, cap_rest))
print("ESTIMATED PAGES: %.2f" % pages)
print()
for k, v in detail.items():
    print("  %-28s %7.0f pt  (%.2f col)" % (k, v, v / TEXT_H))
