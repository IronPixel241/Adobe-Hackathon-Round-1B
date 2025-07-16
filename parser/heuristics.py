"""
Heuristics for narrowing raw extracted lines down to heading candidates.

Signals used:
- Repeating header/footer removal (appears on >= threshold fraction of pages in top/bottom band).
- Length sanity (drop very long body lines unless strong heading cue).
- Global font prominence (>= PCT threshold).
- Per-page font prominence (top K font sizes on that page).
- Bold boost.
- Trailing-colon boost (common in principle / label headings).
"""

import numpy as np
from collections import Counter, defaultdict
import re

# ---- Tunables ----
GLOBAL_FONT_PCT = 85          # percentile cut for "large font" globally
PER_PAGE_TOPK = 3             # top K font sizes per page always kept
MAX_WORDS = 20                # hard drop if over (unless colon & short chars)
MAX_CHARS = 150               # guardrail
HEADER_FOOTER_FRAC = 0.5      # drop text seen on >= 50% of pages in header/footer band
HEADER_FOOTER_BAND = 0.07     # top/bottom % of page height considered header/footer
COLON_LEN_GRACE = 28          # if endswith ":" allow up to 28 words

# mild normalization for repetition tests
_norm_ws_re = re.compile(r"\s+")
_norm_digits_re = re.compile(r"\d+")

def _norm_repetition_text(t: str) -> str:
    t = t.strip()
    t = _norm_digits_re.sub("#", t)          # mask numbers (page numbers/dates)
    t = _norm_ws_re.sub(" ", t)
    return t.lower()

def _compute_page_heights(blocks):
    """Infer page height from max y1 in bbox per page; fallback to 1000 if missing."""
    heights = defaultdict(lambda: 1000.0)
    tmp = defaultdict(list)
    for b in blocks:
        if "bbox" in b:
            tmp[b["page"]].append(b["bbox"][3])  # y1
    for p, ys in tmp.items():
        if ys:
            heights[p] = max(ys)
    return heights

def _drop_repeating_lines(blocks, frac=HEADER_FOOTER_FRAC, band=HEADER_FOOTER_BAND):
    """Drop lines that repeat on many pages within header/footer bands."""
    if not blocks:
        return blocks

    heights = _compute_page_heights(blocks)
    page_total = len({b["page"] for b in blocks})

    # group by normalized text
    seen = defaultdict(set)  # norm_text -> set(pages)
    pos_by_text = defaultdict(list)  # norm_text -> list of vertical centers

    for b in blocks:
        ntext = _norm_repetition_text(b["text"])
        seen[ntext].add(b["page"])
        if "bbox" in b:
            y0, y1 = b["bbox"][1], b["bbox"][3]
            pos_by_text[ntext].append((b["page"], (y0 + y1) / 2.0))

    # decide which normalized texts are repeating header/footer
    drop_norms = set()
    for ntext, pages in seen.items():
        if len(pages) / page_total < frac:
            continue
        # if we have positions, confirm they live in header/footer bands
        if pos_by_text[ntext]:
            in_band = True
            for p, ymid in pos_by_text[ntext]:
                h = heights[p]
                top_cut = h * band
                bot_cut = h * (1 - band)
                if not (ymid <= top_cut or ymid >= bot_cut):
                    in_band = False
                    break
            if in_band:
                drop_norms.add(ntext)

    return [b for b in blocks if _norm_repetition_text(b["text"]) not in drop_norms]


def filter_heading_candidates(blocks):
    """
    Return a pruned list of candidate blocks likely to be headings.
    Non-destructive: returns shallow copies of original dicts.
    """
    if not blocks:
        return []

    # 1. Remove repeating header/footer noise
    blocks_nf = _drop_repeating_lines(blocks)

    if not blocks_nf:
        return []

    # 2. Basic length sanity
    kept_len = []
    for b in blocks_nf:
        words = b["text"].split()
        if len(words) > MAX_WORDS and not (b["text"].endswith(":") and len(words) <= COLON_LEN_GRACE):
            # drop long body lines
            continue
        if len(b["text"]) > MAX_CHARS:
            continue
        kept_len.append(b)

    if not kept_len:
        return []

    # 3. Font prominence global
    sizes = [b["font_size"] for b in kept_len if b["font_size"] is not None]
    if sizes:
        global_thr = np.percentile(sizes, GLOBAL_FONT_PCT)
    else:
        global_thr = 0

    # 4. Per-page top-K font sizes
    page_fonts = defaultdict(list)
    for b in kept_len:
        page_fonts[b["page"]].append(b["font_size"])

    page_top_sizes = {}
    for p, szs in page_fonts.items():
        unique_sorted = sorted(set(szs), reverse=True)
        page_top_sizes[p] = set(unique_sorted[:PER_PAGE_TOPK])

    # 5. Candidate selection
    candidates = []
    for b in kept_len:
        fs = b["font_size"]
        is_global_big = fs >= global_thr
        is_page_big = fs in page_top_sizes.get(b["page"], set())
        is_bold = b.get("bold", False)
        has_colon = b["text"].endswith(":")
        # keep if any strong signal
        if is_global_big or is_page_big or (is_bold and len(b["text"].split()) <= 6) or has_colon:
            candidates.append(dict(b))  # shallow copy

    return candidates
