import fitz  # PyMuPDF
from typing import List
from dataclasses import dataclass
import os

@dataclass
class PdfElement:
    text: str
    page: int
    bbox: tuple  # (x0, y0, x1, y1)
    font_size: float
    font_name: str
    is_bold: bool

def process_pdf(pdf_path: str) -> List[PdfElement]:
    elements = []
    dump_dir = "dump"
    os.makedirs(dump_dir, exist_ok=True)
    debug_path = os.path.join(dump_dir, "debug.txt")

    with open(debug_path, "w", encoding="utf-8") as debug_file:
        doc = fitz.open(pdf_path)

        for page_num, page in enumerate(doc, start=1):
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                for line in block.get("lines", []):
                    for span in line.get("spans", []):
                        text = span["text"].strip()
                        if not text:
                            continue

                        bbox = (span["bbox"][0], span["bbox"][1], span["bbox"][2], span["bbox"][3])
                        font_name = span.get("font", "Unknown")
                        font_size = span.get("size", 0.0)
                        is_bold = "bold" in font_name.lower()

                        element = PdfElement(
                            text=text,
                            page=page_num,
                            bbox=bbox,
                            font_size=font_size,
                            font_name=font_name,
                            is_bold=is_bold
                        )

                        elements.append(element)

                        debug_file.write(
                            f"[Page {page_num}] '{text}'\n"
                            f"  Font: {font_name}, Size: {font_size}, Bold: {is_bold}\n"
                            f"  BBox: {bbox}\n\n"
                        )

    return elements
