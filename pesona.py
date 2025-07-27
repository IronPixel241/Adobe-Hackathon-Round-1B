import os
import json
import fitz  # PyMuPDF
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
import datetime

# --- Configuration ---
# Set the input directory. For the hackathon's Docker environment, this should be '/app/input'.
INPUT_DIR = 'input'
OUTPUT_DIR = 'output'

# --- Input JSON (remains the same) ---
input_data = {
    "challenge_info": {
        "challenge_id": "round_1b_003",
        "test_case_name": "create_manageable_forms",
        "description": "Creating manageable forms"
    },
    "documents": [
        {"filename": "Breakfast Ideas.pdf"},
        {"filename": "Dinner Ideas - Mains_1.pdf"},
        {"filename": "Dinner Ideas - Mains_2.pdf"},
        {"filename": "Dinner Ideas - Mains_3.pdf"},
        {"filename": "Dinner Ideas - Sides_1.pdf"},
        {"filename": "Dinner Ideas - Sides_2.pdf"},
        {"filename": "Dinner Ideas - Sides_3.pdf"},
        {"filename": "Dinner Ideas - Sides_4.pdf"},
        {"filename": "Lunch Ideas.pdf"}
    ],
    "persona": {
        "role": "Food Contractor"
    },
    "job_to_be_done": {
        "task": "Prepare a vegetarian buffet-style dinner menu for a corporate gathering, including gluten-free items."
    }
}

# --- Utility Functions ---
def extract_pdf_text_by_page(filename):
    # ... (function is identical to Version 1)
    filepath = os.path.join(INPUT_DIR, filename)
    if not os.path.isfile(filepath): return []
    try:
        with fitz.open(filepath) as doc:
            return [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
    except Exception: return []

def is_heading(line):
    # ... (function is identical to Version 1)
    if not line or line.strip().endswith('.'): return False
    words = line.strip().split()
    if len(words) > 10: return False
    return line.istitle() or line.isupper()

def extract_sections(pages_text, doc_name):
    # ... (function is identical to Version 1)
    sections = []
    current_heading = "Introduction"
    current_content = ""
    start_page = 1
    for page_num, text in pages_text:
        lines = text.split('\n')
        for line in lines:
            if is_heading(line):
                if current_content.strip():
                    sections.append({"document": doc_name, "section_title": current_heading, "content": current_content.strip(), "page_number": start_page})
                start_page, current_heading, current_content = page_num, line.strip(), ""
            else:
                current_content += " " + line.strip()
    if current_content.strip():
        sections.append({"document": doc_name, "section_title": current_heading, "content": current_content.strip(), "page_number": start_page})
    return sections

def get_refined_summary(section_content, model, job_embedding, top_k=3):
    """
    Extracts the most relevant sentences from a section to form a summary.
    """
    # Split content into sentences. This regex handles various sentence endings.
    sentences = re.split(r'(?<=[.!?])\s+', section_content)
    if not sentences or all(s.isspace() for s in sentences):
        return ""

    # Score each sentence against the job description
    sentence_embeddings = model.encode(sentences, convert_to_tensor=True)
    similarities = util.cos_sim(job_embedding, sentence_embeddings)
    
    # Get the indices of the top_k most relevant sentences
    ranked_indices = np.argsort(-similarities.flatten())[:top_k]
    
    # Sort the top sentences by their original order in the text
    ranked_indices.sort()
    
    # Join the top sentences to form the refined summary
    summary = " ".join([sentences[idx].strip() for idx in ranked_indices])
    return summary

# --- Main Processing Logic ---
def run_refined_extraction():
    print("\n--- Running Version 2: Refined Extraction ---")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    model = SentenceTransformer('all-MiniLM-L6-v2')
    job_text = input_data["job_to_be_done"]["task"]
    job_embedding = model.encode(job_text, convert_to_tensor=True)

    all_sections = []
    for doc in input_data["documents"]:
        pages_text = extract_pdf_text_by_page(doc["filename"])
        if pages_text:
            all_sections.extend(extract_sections(pages_text, doc["filename"]))

    if not all_sections:
        print("No sections found.")
        return
        
    section_contents = [sec["content"] for sec in all_sections]
    section_embeddings = model.encode(section_contents, convert_to_tensor=True)
    similarities = util.cos_sim(job_embedding, section_embeddings)
    ranked_indices = np.argsort(-similarities.flatten())

    output = { "metadata": { "version": "2 - Refined" }, "extracted_sections": [], "subsection_analysis": [] }

    for i, idx in enumerate(ranked_indices[:5]):
        section = all_sections[idx]
        rank = i + 1
        output["extracted_sections"].append({ "document": section["document"], "section_title": section["section_title"], "importance_rank": rank, "page_number": section["page_number"] })
        
        # *** The key difference is here: we call get_refined_summary ***
        refined_text = get_refined_summary(section["content"], model, job_embedding)
        output["subsection_analysis"].append({ "document": section["document"], "refined_text": refined_text, "page_number": section["page_number"] })

    output_filepath = os.path.join(OUTPUT_DIR, "output_refined.json")
    with open(output_filepath, "w") as f:
        json.dump(output, f, indent=4)
    print(f"✅ Version 2 output saved to {output_filepath}")

run_refined_extraction()