import os
import json
import fitz  # PyMuPDF
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
import datetime

# --- Input JSON ---
input_data = {
    "challenge_info": {
        "challenge_id": "round_1b_recipes_final",
        "test_case_name": "extract_recipes_refined",
    },
    "documents": [
        {"filename": "Breakfast Ideas.pdf"}, {"filename": "Dinner Ideas - Mains_1.pdf"},
        {"filename": "Dinner Ideas - Mains_2.pdf"}, {"filename": "Dinner Ideas - Mains_3.pdf"},
        {"filename": "Dinner Ideas - Sides_1.pdf"}, {"filename": "Dinner Ideas - Sides_2.pdf"},
        {"filename": "Dinner Ideas - Sides_3.pdf"}, {"filename": "Dinner Ideas - Sides_4.pdf"},
        {"filename": "Lunch Ideas.pdf"}
    ],
    "persona": { "role": "Food Contractor" },
    "job_to_be_done": { "task": "Prepare a vegetarian buffet-style dinner menu for a corporate gathering, including gluten-free items." }
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

def extract_recipe_sections(pages_text, doc_name):
    """A stricter parser that identifies a recipe title and its full content."""
    sections = []
    full_text = "\n".join([text for _, text in pages_text])
    # This regex is a positive lookbehind, splitting the text *after* a potential title.
    # It assumes a title is followed by a newline and then "Ingredients:".
    recipe_blocks = re.split(r'(?<=\n)\n(?=[A-Z][a-z]+(?: [A-Z][a-z]+)*\nIngredients:)', full_text)

    for block in recipe_blocks:
        lines = block.strip().split('\n')
        if not lines or "Ingredients" not in block:
            continue

        title_candidate = lines[0].strip()
        page_num = 1
        for p_num, p_text in pages_text:
            if title_candidate in p_text:
                page_num = p_num
                break
        
        sections.append({
            "document": doc_name, "section_title": title_candidate,
            "content": block, "page_number": page_num
        })
    return sections

def get_refined_ingredients(section_content):
    """
    Extracts only the ingredients list from a recipe section.
    This is a more targeted refinement for the "Food Contractor" persona.
    """
    try:
        # Find the text between "Ingredients:" and "Instructions:"
        match = re.search(r'Ingredients:(.*?)Instructions:', section_content, re.DOTALL | re.IGNORECASE)
        if match:
            # Clean up the extracted text
            ingredients_text = match.group(1).strip()
            # Replace bullet points and clean up newlines
            cleaned_ingredients = re.sub(r'[\n\uf0b7\u2022]', ' ', ingredients_text).strip()
            return re.sub(r'\s+', ' ', cleaned_ingredients) # Normalize whitespace
    except Exception:
        pass
    return "Could not isolate ingredients." # Fallback


# --- Main Processing Logic ---

# 1. Load Model and Define Query
model = SentenceTransformer('all-MiniLM-L6-v2')
job_text = input_data["job_to_be_done"]["task"]
job_embedding = model.encode(job_text, convert_to_tensor=True)

# 2. Extract Sections
all_sections = []
for doc in input_data["documents"]:
    pages_text = extract_pdf_text_by_page(doc["filename"])
    if pages_text:
        all_sections.extend(extract_recipe_sections(pages_text, doc["filename"]))

# 3. Calculate Relevance and Rank
titles = [sec["section_title"] for sec in all_sections]
contents = [sec["content"] for sec in all_sections]
title_embeddings = model.encode(titles, convert_to_tensor=True)
content_embeddings = model.encode(contents, convert_to_tensor=True)
title_similarities = util.cos_sim(job_embedding, title_embeddings)[0]
content_similarities = util.cos_sim(job_embedding, content_embeddings)[0]
combined_scores = (0.4 * title_similarities) + (0.6 * content_similarities)
ranked_indices = np.argsort(-combined_scores)

# 4. Build Final Output (with 1-to-1 correspondence)
output = {
    "metadata": {
        "persona": input_data["persona"]["role"],
        "job_to_be_done": input_data["job_to_be_done"]["task"],
        "input_documents": [doc["filename"] for doc in input_data["documents"]],
        "processing_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    },
    "extracted_sections": [],
    "subsection_analysis": []
}

final_rank = 1
MIN_RELEVANCE_SCORE = 0.15

for idx in ranked_indices:
    if final_rank > 15: break # Limit output to top 15 relevant sections

    if combined_scores[idx] > MIN_RELEVANCE_SCORE:
        section = all_sections[idx]
        
        # Add to extracted_sections
        output["extracted_sections"].append({
            "document": section["document"],
            "section_title": section["section_title"],
            "importance_rank": final_rank,
            "page_number": section["page_number"]
        })
        
        # Generate refined text (ingredients only) for this specific section
        refined_text = get_refined_ingredients(section["content"])
        
        # Add to subsection_analysis (one entry per extracted section)
        output["subsection_analysis"].append({
            "document": section["document"],
            "page_number": section["page_number"],
            "refined_text": refined_text
        })
        
        final_rank += 1

# --- Save output JSON ---
output_file = f"output_{input_data['challenge_info']['test_case_name']}.json"
with open(output_file, "w", encoding='utf-8') as f:
    json.dump(output, f, indent=4)

print(f"\n✅ Done: {output_file} generated with refined, 1-to-1 analysis.")