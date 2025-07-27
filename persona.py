import os
import json
import fitz  # PyMuPDF
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
import datetime
import time # CHANGE: Imported for timing

# --- DICTIONARY TO STORE TIMING RESULTS ---
processing_times = {}

# --- START OF TIMED OPERATIONS ---
main_start_time = time.perf_counter()

# --- CHANGE 1: LOAD INPUT FROM A JSON FILE ---
start_time = time.perf_counter()
try:
    with open("input.json", "r", encoding='utf-8') as f:
        input_data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"❌ Error reading input.json: {e}")
    exit()
processing_times["input_loading_s"] = time.perf_counter() - start_time


# --- CONFIGURATION ---
TITLE_WEIGHT = 0.3
CONTENT_WEIGHT = 0.7
MIN_RELEVANCE_SCORE = 0.2
MIN_WORDS_FOR_SUMMARY_SENTENCE = 5
# The percentile of scores to use as the dynamic threshold for heading detection
HEADING_SCORE_PERCENTILE = 95

# --- UTILITY FUNCTIONS ---

def extract_pdf_text_by_page(filename):
    if not os.path.isfile(filename): return []
    try:
        with fitz.open(filename) as doc:
            return [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
    except Exception as e: return []

def score_line_as_heading(line):
    """Assigns a 'heading potential' score to a line of text."""
    stripped = line.strip()
    words = stripped.split()
    word_count = len(words)

    if not stripped or word_count > 10: return 0
    if stripped.endswith(('.', ',', ':', ';')) or ',' in stripped: return 0
    if words and words[0].islower(): return 0
    
    score = 0
    if 1 <= word_count <= 5: score += 2 # Short lines are good candidates
    if stripped.istitle(): score += 3
    if stripped.isupper() and word_count > 1: score += 4 # ALL CAPS is a strong signal
    if re.match(r'^([IVXLCDM]+\.|[A-Z]\.)', stripped): score += 5 # Academic style is very strong

    return score

def extract_dynamic_sections(pages_text, doc_name):
    """Dynamically identifies main headings by analyzing all lines in the document."""
    line_profiles = []
    for page_num, text in pages_text:
        for line_num, line in enumerate(text.split('\n')):
            stripped_line = line.strip()
            if stripped_line:
                line_profiles.append({
                    "text": stripped_line, "page": page_num, "line_num": line_num,
                    "score": score_line_as_heading(stripped_line)
                })

    if not line_profiles: return []

    scores = [p['score'] for p in line_profiles if p['score'] > 0]
    if not scores: # If no line scores above 0, treat as a single section
        full_content = " ".join(p['text'] for p in line_profiles)
        return [{"document": doc_name, "section_title": "Full Document", "content": full_content, "page_number": 1}]

    # Calculate the dynamic threshold based on the score distribution
    dynamic_threshold = np.percentile(scores, HEADING_SCORE_PERCENTILE)
    
    # Select only the lines that meet our dynamic criteria for a heading
    headings = [p for p in line_profiles if p['score'] >= dynamic_threshold and p['score'] > 0]
    headings.sort(key=lambda x: (x['page'], x['line_num'])) # Ensure document order

    if not headings:
        full_content = " ".join(p['text'] for p in line_profiles)
        return [{"document": doc_name, "section_title": "Full Document", "content": full_content, "page_number": 1}]

    # Reconstruct the document content around the confirmed headings
    sections = []
    for i, heading in enumerate(headings):
        start_page = heading['page']
        start_line = heading['line_num']
        
        # Find the end of the section (where the next heading starts)
        end_page = len(pages_text) + 1
        end_line = float('inf')
        if i + 1 < len(headings):
            next_heading = headings[i+1]
            end_page = next_heading['page']
            end_line = next_heading['line_num']
        
        content = []
        for p_num, p_text in pages_text:
            if start_page <= p_num <= end_page:
                lines = p_text.split('\n')
                for l_num, line in enumerate(lines):
                    is_after_start = p_num > start_page or (p_num == start_page and l_num > start_line)
                    is_before_end = p_num < end_page or (p_num == end_page and l_num < end_line)
                    if is_after_start and is_before_end:
                        content.append(line)
        
        sections.append({
            "document": doc_name, "section_title": heading['text'],
            "content": " ".join(content).strip(), "page_number": start_page
        })
        
    return sections

def get_refined_summary(section_content, model, job_embedding, top_k=3):
    """General-purpose summarizer."""
    sentences = re.split(r'(?<=[.!?])\s+', section_content)
    meaningful_sentences = [s.strip() for s in sentences if len(s.split()) >= MIN_WORDS_FOR_SUMMARY_SENTENCE]
    if not meaningful_sentences: return section_content
    sentence_embeddings = model.encode(meaningful_sentences, convert_to_tensor=True)
    similarities = util.cos_sim(job_embedding, sentence_embeddings)[0]
    top_indices = np.argsort(-similarities)[:top_k]
    top_indices.sort()
    summary = " ".join([meaningful_sentences[idx] for idx in top_indices])
    return summary if summary else section_content


# --- MAIN PROCESSING LOGIC ---

# Timing: Model loading
start_time = time.perf_counter()
model = SentenceTransformer('all-MiniLM-L6-v2')
processing_times["model_loading_s"] = time.perf_counter() - start_time

job_text = input_data["job_to_be_done"]["task"]

# Timing: Job embedding
start_time = time.perf_counter()
job_embedding = model.encode(job_text, convert_to_tensor=True)
processing_times["job_embedding_s"] = time.perf_counter() - start_time

all_sections = []
# Timing: Document parsing and sectioning
start_time = time.perf_counter()
for doc in input_data["documents"]:
    pages_text = extract_pdf_text_by_page(doc["filename"])
    if pages_text:
        all_sections.extend(extract_dynamic_sections(pages_text, doc["filename"]))
processing_times["pdf_parsing_and_sectioning_s"] = time.perf_counter() - start_time


if not all_sections:
    print("❌ No valid sections could be extracted. Exiting.")
    exit()

# Timing: Content embedding and similarity scoring
start_time = time.perf_counter()
titles = [sec["section_title"] for sec in all_sections]
contents = [sec["content"] for sec in all_sections]

title_embeddings = model.encode(titles, convert_to_tensor=True)
content_embeddings = model.encode(contents, convert_to_tensor=True)
title_similarities = util.cos_sim(job_embedding, title_embeddings)[0]
content_similarities = util.cos_sim(job_embedding, content_embeddings)[0]

combined_scores = (TITLE_WEIGHT * title_similarities) + (CONTENT_WEIGHT * content_similarities)
ranked_indices = np.argsort(-combined_scores)
processing_times["content_embedding_and_scoring_s"] = time.perf_counter() - start_time

output = {
    "metadata": {
        "persona": input_data["persona"]["role"],
        "job_to_be_done": input_data["job_to_be_done"]["task"],
        "input_documents": [doc["filename"] for doc in input_data["documents"]],
        "processing_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat()
    },
    "extracted_sections": [], "subsection_analysis": []
}

# Timing: Summarization of relevant sections
start_time = time.perf_counter()
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
processing_times["summarization_s"] = time.perf_counter() - start_time

# --- END OF TIMED OPERATIONS ---
processing_times["total_runtime_s"] = time.perf_counter() - main_start_time


# --- CHANGE 2: WRITE PROCESSING TIMES TO A SEPARATE JSON FILE ---
# This is done before writing the main output, so its own I/O time is not counted.
with open("processing_results.json", "w", encoding='utf-8') as f:
    json.dump(processing_times, f, indent=4)
print(f"📊 Performance metrics saved to processing_results.json")


# --- FINAL OUTPUT ---
output_file = f"output_{input_data['challenge_info']['test_case_name']}.json"
with open(output_file, "w", encoding='utf-8') as f:
    json.dump(output, f, indent=4)

print(f"\n✅ Done: Main output saved to {output_file}")