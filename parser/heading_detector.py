from typing import List, Dict
from dataclasses import dataclass
from .pdf_parser import PdfElement
import statistics

def extract_outline(elements: List[PdfElement]) -> Dict:
    """Extract title and hierarchical outline from PDF elements"""
    if not elements:
        return {"title": "", "outline": []}
    
    # Step 1: Detect title (first prominent element)
    title = detect_title(elements)
    
    # Step 2: Detect heading candidates
    heading_candidates = detect_heading_candidates(elements)
    
    # Step 3: Determine hierarchy
    outline = determine_hierarchy(heading_candidates)
    
    return {
        "title": title,
        "outline": outline
    }

def detect_title(elements: List[PdfElement]) -> str:
    """Identify document title"""
    if not elements:
        return ""
    
    # First element with largest font size in first 5 elements
    first_page_elements = [e for e in elements if e.page == 1]
    if not first_page_elements:
        return ""
    
    # Get element with largest font size that's not numbered
    title_candidates = sorted(
        [e for e in first_page_elements[:5] if not e.text[0].isdigit()],
        key=lambda x: x.font_size,
        reverse=True
    )
    
    return title_candidates[0].text if title_candidates else first_page_elements[0].text

def detect_heading_candidates(elements: List[PdfElement]) -> List[Dict]:
    """Identify potential headings"""
    candidates = []
    
    for elem in elements:
        # Heading characteristics
        is_numbered = elem.text.strip() and elem.text.strip()[0].isdigit()
        is_bold_large = elem.is_bold and elem.font_size > 9.5
        
        if is_numbered or is_bold_large:
            candidates.append({
                "text": elem.text,
                "page": elem.page,
                "font_size": elem.font_size,
                "bbox": elem.bbox
            })
    
    return candidates

def determine_hierarchy(candidates: List[Dict]) -> List[Dict]:
    """Assign H1/H2/H3 levels based on font size clustering"""
    if not candidates:
        return []
    
    # Simple approach for forms (all H1)
    if all(c['text'].strip()[0].isdigit() for c in candidates):
        return [{"level": "H1", "text": c["text"], "page": c["page"]} for c in candidates]
    
    # Advanced clustering for more complex documents
    font_sizes = [c['font_size'] for c in candidates]
    mean_size = statistics.mean(font_sizes)
    std_dev = statistics.stdev(font_sizes) if len(font_sizes) > 1 else 0
    
    outline = []
    for c in candidates:
        if c['font_size'] > mean_size + std_dev:
            level = "H1"
        elif c['font_size'] > mean_size - std_dev:
            level = "H2"
        else:
            level = "H3"
        
        outline.append({
            "level": level,
            "text": c["text"],
            "page": c["page"]
        })
    
    return outline