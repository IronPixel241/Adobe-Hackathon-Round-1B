import re

def classify_heading(text):
    """
    Balanced heading classifier:
    - Keeps real headings (sections/subsections)
    - Filters out boilerplate/legal/footer text
    """
    t = text.strip()
    if not t:
        return None

    # Normalize
    lower = t.lower()

    # Exclude obvious non-headings
    if any(x in lower for x in ["copyright", "page", "version", "remarks", "date"]):
        return None
    if re.match(r'^\d{1,2}\s?[a-z]{3,9}\s?\d{4}$', lower):  # date like "18 JUNE 2013"
        return None

    # Drop overly long candidates (likely body text)
    if len(t.split()) > 20 or len(t) > 150:
        return None

    # 1. Numeric structure-based classification
    if re.match(r'^\d+\.\s', t):
        return "H1"
    if re.match(r'^\d+\.\d+\s', t):
        return "H2"
    if re.match(r'^\d+\.\d+\.\d+\s', t):
        return "H3"
    if re.match(r'^\d+\.\d+\.\d+\.\d+\s', t):
        return "H4"

    # 2. Lines ending with colon (principles, labels)
    if t.endswith(":"):
        if len(t.split()) <= 4:
            return "H3"
        else:
            return "H2"

    # 3. Keyword-based signals
    if lower in ["overview", "introduction", "revision history"]:
        return "H2"
    if "summary" in lower:
        return "H2"
    if "background" in lower:
        return "H2"
    if "timeline" in lower or "milestone" in lower:
        return "H3"

    # 4. Special Ontario-style
    if re.match(r'^for each .* it could mean:', lower):
        return "H4"

    # 5. Fallback for likely headings: short & NOT boilerplate
    if 2 <= len(t.split()) <= 6 and not any(word in lower for word in ["board", "international", "tester"]):
        return "H2"

    return None
