import fitz  # PyMuPDF

def extract_text_blocks(pdf_path):
    blocks = []
    doc = fitz.open(pdf_path)
    
    for page_num, page in enumerate(doc, start=1):
        for block in page.get_text("dict")["blocks"]:
            lines = []
            for line in block.get("lines", []):
                text = " ".join([span["text"] for span in line["spans"]]).strip()
                if not text:
                    continue
                font_size = line["spans"][0]["size"]
                font_name = line["spans"][0]["font"]
                is_bold = "Bold" in font_name
                lines.append({
                    "text": text,
                    "font_size": font_size,
                    "bold": is_bold,
                    "page": page_num,
                    "font": font_name
                })

            # ✅ Merge lines with same style
            if lines:
                merged = [lines[0]]
                for current in lines[1:]:
                    prev = merged[-1]
                    same_style = (
                        prev["font_size"] == current["font_size"] and
                        prev["font"] == current["font"] and
                        prev["page"] == current["page"]
                    )
                    if same_style:
                        prev["text"] += " " + current["text"]
                    else:
                        
                        merged.append(current)
                blocks.extend(merged)  # Add merged lines to result

    # ✅ Print all extracted blocks
    print("\n📄 Extracted Blocks from PDF:")
    for block in blocks:
        print(f"[Page {block['page']}] ({'Bold' if block['bold'] else 'Regular'}, {block['font_size']:.1f}pt): {block['text']}")

    return blocks
