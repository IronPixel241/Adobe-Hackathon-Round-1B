from pdfminer.high_level import extract_pages
from pdfminer.layout import LTTextBoxHorizontal, LTTextLineHorizontal
from typing import List
from dataclasses import dataclass

@dataclass
class PdfElement:
    text: str
    page: int
    bbox: tuple  # (x0, y0, x1, y1)
    font_size: float
    font_name: str
    is_bold: bool

def process_pdf(pdf_path: str) -> List[PdfElement]:
    """Extract structured elements from PDF using current pdfminer.six API"""
    elements = []
    
    for page_num, page_layout in enumerate(extract_pages(pdf_path), start=1):
        for element in page_layout:
            if isinstance(element, LTTextBoxHorizontal):
                for text_line in element:
                    if isinstance(text_line, LTTextLineHorizontal):
                        # Get font information from the first character
                        font_name = "Unknown"
                        is_bold = False
                        if text_line._objs:  # Check if there are characters
                            first_char = text_line._objs[0]
                            if hasattr(first_char, 'fontname'):
                                font_name = first_char.fontname
                                is_bold = 'Bold' in font_name
                        
                        elements.append(PdfElement(
                            text=text_line.get_text().strip(),
                            page=page_num,
                            bbox=(text_line.x0, text_line.y0, text_line.x1, text_line.y1),
                            font_size=text_line.height,
                            font_name=font_name,
                            is_bold=is_bold
                        ))
    
    return elements