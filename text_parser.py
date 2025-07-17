import fitz  # PyMuPDF
import os

INPUT_DIR = "input"
OUTPUT_DIR = "text_dump"

os.makedirs(OUTPUT_DIR, exist_ok=True)

def extract_text_with_metadata(pdf_path, txt_path):
    doc = fitz.open(pdf_path)

    with open(txt_path, "w", encoding="utf-8") as out:
        for page_num, page in enumerate(doc, start=1):
            out.write(f"\n--- Page {page_num} ---\n")
            blocks = page.get_text("dict")["blocks"]
            for block in blocks:
                for line in block.get("lines", []):
                    spans = line.get("spans", [])
                    for span in spans:
                        text = span.get("text", "").strip()
                        if not text:
                            continue
                        font_size = span.get("size")
                        font_name = span.get("font")
                        bbox = span.get("bbox")
                        is_bold = "Bold" in font_name
                        is_italic = "Italic" in font_name or "Oblique" in font_name

                        out.write(f"[Page {page_num}] ({font_size:.1f}pt, {font_name}, {'Bold' if is_bold else 'Regular'}, {'Italic' if is_italic else 'Normal'})\n")
                        out.write(f"Text : {text}\n")
                        out.write(f"BBox : {bbox}\n\n")

def main():
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(INPUT_DIR, filename)
            txt_path = os.path.join(OUTPUT_DIR, os.path.splitext(filename)[0] + ".txt")
            extract_text_with_metadata(pdf_path, txt_path)
            print(f"[✓] Dumped text from {filename} → {os.path.basename(txt_path)}")

if __name__ == "__main__":
    main()
