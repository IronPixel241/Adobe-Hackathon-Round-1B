import re

def classify_with_local_context(blocks):
    """
    Assign heading levels based on local font size contrast and boldness.
    Modifies blocks in-place.
    """
    for i, block in enumerate(blocks):
        text = block.get("text", "").strip()
        font_size = block.get("font_size")
        if not text or font_size is None:
            block["level"] = None
            continue

        # Heuristic skips
        lower = text.lower()
        if (
            re.match(r'^\d+[.]?$', text) or
            any(x in lower for x in ["copyright", "page", "version", "remarks", "date"]) or
            re.match(r'^\d{1,2}\s?[a-z]{3,9}\s?\d{4}$', lower) or
            len(text.split()) > 20 or len(text) > 150
        ):
            block["level"] = None
            continue

        prev = blocks[i - 1] if i > 0 else None
        next_ = blocks[i + 1] if i + 1 < len(blocks) else None
        prev_size = prev["font_size"] if prev else 0
        next_size = next_["font_size"] if next_ else 0

        is_larger_than_prev = font_size > prev_size
        is_larger_than_next = font_size > next_size
        is_bold = block.get("bold", False)
        all_caps = text.isupper()

        heading_candidate = (
            (is_larger_than_prev and is_larger_than_next) or
            (is_bold and len(text.split()) <= 6) or
            all_caps or
            text.endswith(":")
        )

        if not heading_candidate:
            block["level"] = None
            continue

        # Estimate level based on size gap
        avg_neighbor_size = (prev_size + next_size) / 2 if (prev_size and next_size) else min(prev_size, next_size)
        size_diff = font_size - avg_neighbor_size

        if size_diff >= 4:
            block["level"] = "H1"
        elif size_diff >= 2:
            block["level"] = "H2"
        else:
            block["level"] = "H3"

    return blocks
