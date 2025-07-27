import os
import json
import fitz  # PyMuPDF
import numpy as np
import re
from sentence_transformers import SentenceTransformer
import datetime

# --- Input JSON (as provided) ---
input_data = {
    "challenge_info": {
        "challenge_id": "round_1b_003",
        "test_case_name": "create_manageable_forms",
        "description": "Creating manageable forms"
    },
    "documents": [
        {"filename": "Breakfast Ideas.pdf", "title": "Breakfast Ideas"},
        {"filename": "Dinner Ideas - Mains_1.pdf", "title": "Dinner Ideas - Mains_1"},
        {"filename": "Dinner Ideas - Mains_2.pdf", "title": "Dinner Ideas - Mains_2"},
        {"filename": "Dinner Ideas - Mains_3.pdf", "title": "Dinner Ideas - Mains_3"},
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
    """Extracts text from each page of a PDF."""
    if not os.path.isfile(filename):
        print(f"⚠️ File not found: {filename}")
        return []
    try:
        with fitz.open(filename) as doc:
            return [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
    except Exception as e:
        print(f"❌ Failed to extract {filename}: {e}")
        return []

def extract_sections(pages_text, doc_name):
    """
    (Simulated Round 1A logic)
    Extracts structured sections from document text.
    This heuristic identifies potential headings based on capitalization and length.
    A true solution would use font size, style, and layout analysis.
    """
    sections = []
    current_heading = "General Content"
    current_content = ""
    current_page = 1

    for page_num, text in pages_text:
        # Split page into lines and identify potential headings
        lines = text.split('\n')
        for line in lines:
            stripped_line = line.strip()
            # Heuristic: A heading is a short, title-cased line.
            if 1 < len(stripped_line.split()) < 7 and stripped_line.istitle() and not stripped_line.endswith('.'):
                # Save the previous section
                if current_content.strip():
                    sections.append({
                        "document": doc_name,
                        "section_title": current_heading,
                        "content": current_content.strip(),
                        "page_number": current_page
                    })
                # Start a new section
                current_heading = stripped_line
                current_content = ""
                current_page = page_num
            else:
                current_content += " " + stripped_line
    
    # Add the last section
    if current_content.strip():
        sections.append({
            "document": doc_name,
            "section_title": current_heading,
            "content": current_content.strip(),
            "page_number": current_page
        })
    return sections

def get_relevant_snippets(section_content, query_embedding, model, top_k=3):
    """Extracts the most relevant sentences from a section."""
    sentences = re.split(r'(?<=[.!?])\s+', section_content)
    if not sentences or all(s.isspace() for s in sentences):
        return []

    sentence_embeddings = model.encode(sentences)
    similarities = sentence_embeddings @ query_embedding.T
    
    # Get top_k unique sentences
    ranked_indices = np.argsort(-similarities.flatten())
    snippets = []
    seen_sentences = set()
    for idx in ranked_indices:
        if len(snippets) < top_k:
            sentence = sentences[idx].strip()
            if sentence and sentence not in seen_sentences:
                snippets.append(sentence)
                seen_sentences.add(sentence)
    return snippets


# --- Main Processing Logic ---

# 1. Load Model and Define Query
# The 'all-MiniLM-L6-v2' model is small (~80MB) and efficient.
# It fits within the <1GB model size and offline constraints. [cite: 59, 60, 152]
model = SentenceTransformer('all-MiniLM-L6-v2')
job_text = input_data["job_to_be_done"]["task"]
job_embedding = model.encode(job_text)

# 2. Extract Sections from All Documents
all_sections = []
doc_list = input_data["documents"]
for doc in doc_list:
    filename = doc["filename"]
    pages_text = extract_pdf_text_by_page(filename)
    if pages_text:
        all_sections.extend(extract_sections(pages_text, filename))

if not all_sections:
    print("❌ No sections could be extracted from any documents. Exiting.")
    exit()

# 3. Score and Rank Sections Semantically
section_contents = [sec["content"] for sec in all_sections]
section_embeddings = model.encode(section_contents)
similarities = section_embeddings @ job_embedding.T
ranked_indices = np.argsort(-similarities.flatten())

# 4. Build Output JSON
output = {
    "metadata": {
        "persona": input_data["persona"]["role"],
        "job_to_be_done": job_text,
        "input_documents": [doc["filename"] for doc in doc_list],
        "processing_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    },
    "extracted_sections": [],
    "subsection_analysis": []
}

# 5. Populate Output with Ranked Sections and Snippets
seen_sections = set()
rank = 1
for idx in ranked_indices:
    section = all_sections[idx]
    # Avoid duplicate sections if content is identical
    if section["content"] in seen_sections:
        continue
    
    # Add to Extracted Sections
    output["extracted_sections"].append({
        "document": section["document"],
        "section_title": section["section_title"],
        "importance_rank": rank,
        "page_number": section["page_number"]
    })
    
    # Add to Subsection Analysis
    snippets = get_relevant_snippets(section["content"], job_embedding, model)
    for snip in snippets:
        output["subsection_analysis"].append({
            "document": section["document"],
            "refined_text": snip,
            "page_number": section["page_number"]
        })
    
    seen_sections.add(section["content"])
    rank += 1

# --- Save output JSON ---
output_file = f"output_{input_data['challenge_info']['test_case_name']}.json"
with open(output_file, "w") as f:
    json.dump(output, f, indent=4)

print(f"✅ Done: {output_file} generated.")