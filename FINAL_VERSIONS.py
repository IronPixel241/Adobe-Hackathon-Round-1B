import os
import json
import fitz  # PyMuPDF
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
import datetime

# --- INPUT TEST CASE ---
# This hybrid code is designed to work across all test cases.
input_data = {
    "challenge_info": {
        "challenge_id": "round_1b_hybrid",
        "test_case_name": "hybrid_extraction",
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
    "persona": { "role": "Food Contractor" },
    "job_to_be_done": { "task": "Prepare a vegetarian buffet-style dinner menu for a corporate gathering, including gluten-free items." }
}

# --- CONFIGURATION ---
TITLE_WEIGHT = 0.3
CONTENT_WEIGHT = 0.7
MIN_RELEVANCE_SCORE = 0.2
MIN_WORDS_FOR_SUMMARY_SENTENCE = 5
MAX_HEADING_PERCENTAGE = 0.05

# --- HYBRID EXTRACTION LOGIC ---

def extract_recipe_sections(pages_text, doc_name):
    """Specialized extractor for documents with a clear 'Title' -> 'Ingredients' structure."""
    sections = []
    full_text = "\n".join([text for _, text in pages_text])
    # Regex to split text by a Title followed by "Ingredients:" on a new line
    recipe_blocks = re.split(r'\n(?=[A-Z][a-z]+(?: [A-Z][a-z]+)*\nIngredients:)', full_text)

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
        sections.append({"document": doc_name, "section_title": title_candidate, "content": block, "page_number": page_num})
    return sections

def score_line_as_heading(line):
    """Scores a line based on how likely it is to be a main heading in a general document."""
    stripped = line.strip()
    words = stripped.split()
    if not 2 <= len(words) <= 12 or stripped.endswith(('.', ',', ':', ';')) or ',' in stripped:
        return 0
    score = 0
    if stripped.isupper(): score += 3
    if stripped.istitle(): score += 2
    if re.match(r'^([IVXLCDM]+\.|[A-Z]\.)', stripped): score += 2
    return max(0, score)

def extract_dynamic_h1_sections(pages_text, doc_name):
    """General-purpose extractor using dynamic scoring for formal documents."""
    all_lines = []
    total_line_count = 0
    for page_num, text in pages_text:
        lines = text.split('\n')
        for line_num, line_content in enumerate(lines):
            if line_content.strip():
                all_lines.append({"text": line_content.strip(), "page": page_num, "line_num": line_num, "score": score_line_as_heading(line_content)})
        total_line_count += len(lines)

    all_lines.sort(key=lambda x: x['score'], reverse=True)
    num_headings_to_keep = int(total_line_count * MAX_HEADING_PERCENTAGE)
    top_headings = [line for line in all_lines if line['score'] > 2][:num_headings_to_keep]
    top_headings.sort(key=lambda x: (x['page'], x['line_num']))

    if not top_headings:
        full_content = " ".join([text for _, text in pages_text])
        return [{"document": doc_name, "section_title": "Full Document", "content": full_content, "page_number": 1}]

    sections = []
    content_accumulator = ""
    # Simplified section building for dynamic headings
    full_content_string = " ".join(text for _, text in pages_text)
    for i, heading in enumerate(top_headings):
        start_index = full_content_string.find(heading['text'])
        end_index = -1
        if i + 1 < len(top_headings):
            next_heading = top_headings[i+1]
            end_index = full_content_string.find(next_heading['text'], start_index)
        
        section_content = full_content_string[start_index:end_index].strip()
        sections.append({"document": doc_name, "section_title": heading['text'], "content": section_content, "page_number": heading['page']})
    return sections

def extract_sections_hybrid(pages_text, doc_name):
    """(NEW) Intelligently chooses the best extraction method."""
    recipe_sections = extract_recipe_sections(pages_text, doc_name)
    # If the recipe extractor finds at least 2 recipes, trust it.
    if len(recipe_sections) > 1:
        print(f"INFO: Detected recipe format for '{doc_name}'. Using specialized parser.")
        return recipe_sections
    else:
        # Otherwise, fall back to the general-purpose dynamic H1 parser.
        print(f"INFO: Recipe format not detected for '{doc_name}'. Using dynamic H1 parser.")
        return extract_dynamic_h1_sections(pages_text, doc_name)


# --- OTHER UTILITY FUNCTIONS ---

def get_refined_summary(section_content, model, job_embedding, top_k=3):
    """General-purpose summarizer."""
    sentences = re.split(r'(?<=[.!?])\s+', section_content)
    meaningful_sentences = [s for s in sentences if len(s.split()) >= MIN_WORDS_FOR_SUMMARY_SENTENCE]
    if not meaningful_sentences: return section_content
    sentence_embeddings = model.encode(meaningful_sentences, convert_to_tensor=True)
    similarities = util.cos_sim(job_embedding, sentence_embeddings)[0]
    top_indices = np.argsort(-similarities)[:top_k]
    top_indices.sort()
    summary = " ".join([meaningful_sentences[idx].strip() for idx in top_indices])
    return summary if summary else section_content
    
def extract_pdf_text_by_page(filename):
    """Extracts text from a PDF."""
    if not os.path.isfile(filename): return []
    try:
        with fitz.open(filename) as doc:
            return [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
    except Exception as e: return []


# --- MAIN PROCESSING LOGIC ---
model = SentenceTransformer('all-MiniLM-L6-v2')
job_text = input_data["job_to_be_done"]["task"]
job_embedding = model.encode(job_text, convert_to_tensor=True)

# (MODIFIED) Using the new HYBRID section extractor
all_sections = []
for doc in input_data["documents"]:
    pages_text = extract_pdf_text_by_page(doc["filename"])
    if pages_text:
        all_sections.extend(extract_sections_hybrid(pages_text, doc["filename"]))

# The rest of the pipeline remains the same
if not all_sections:
    print("❌ No sections could be extracted. Exiting.")
    exit()

titles = [sec["section_title"] for sec in all_sections]
contents = [sec["content"] for sec in all_sections]
title_embeddings = model.encode(titles, convert_to_tensor=True)
content_embeddings = model.encode(contents, convert_to_tensor=True)
title_similarities = util.cos_sim(job_embedding, title_embeddings)[0]
content_similarities = util.cos_sim(job_embedding, content_embeddings)[0]

combined_scores = (TITLE_WEIGHT * title_similarities) + (CONTENT_WEIGHT * content_similarities)
ranked_indices = np.argsort(-combined_scores)

output = {
    "metadata": {
        "persona": input_data["persona"]["role"],
        "job_to_be_done": input_data["job_to_be_done"]["task"],
        "input_documents": [doc["filename"] for doc in input_data["documents"]],
        "processing_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    },
    "extracted_sections": [], "subsection_analysis": []
}

final_rank = 1
for idx in ranked_indices:
    if final_rank > 15: break
    if combined_scores[idx] > MIN_RELEVANCE_SCORE:
        section = all_sections[idx]
        output["extracted_sections"].append({
            "document": section["document"], "section_title": section["section_title"],
            "importance_rank": final_rank, "page_number": section["page_number"]
        })
        refined_text = get_refined_summary(section["content"], model, job_embedding)
        output["subsection_analysis"].append({
            "document": section["document"], "page_number": section["page_number"],
            "refined_text": refined_text
        })
        final_rank += 1

output_file = f"output_{input_data['challenge_info']['test_case_name']}.json"
with open(output_file, "w", encoding='utf-8') as f:
    json.dump(output, f, indent=4)

print(f"\n✅ Done: Hybrid extraction output saved to {output_file}")