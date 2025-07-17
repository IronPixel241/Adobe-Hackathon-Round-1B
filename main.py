# main.py

import os
import json
from parser.extract_text_blocks import extract_text_blocks
from parser.heuristics import filter_heading_candidates,compute_font_thresholds
from parser.heading_classifier import classify_with_local_context
from parser.hierarchy_builder import build_hierarchy

INPUT_DIR = "input"
OUTPUT_DIR = "output"

def process_pdf(pdf_path):
    # Step 1: Extract all text blocks
    blocks = extract_text_blocks(pdf_path)

    # Step 2: Apply heuristics to filter potential headings
    candidate_blocks = filter_heading_candidates(blocks)

# Compute font thresholds
    global_thresholds = compute_font_thresholds(candidate_blocks)

    # Classify each block
    classify_with_local_context(candidate_blocks)

   
    # Step 4: Build structured JSON output
    hierarchy = build_hierarchy(candidate_blocks)
    return hierarchy

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    for filename in os.listdir(INPUT_DIR):
        if filename.lower().endswith(".pdf"):
            pdf_path = os.path.join(INPUT_DIR, filename)
            result = process_pdf(pdf_path)

            output_filename = os.path.splitext(filename)[0] + ".json"
            output_path = os.path.join(OUTPUT_DIR, output_filename)

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(result, f, indent=4, ensure_ascii=False)

            print(f"[✓] Processed: {filename} → {output_filename}")

if __name__ == "__main__":
    main()
