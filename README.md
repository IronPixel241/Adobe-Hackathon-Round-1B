# Adobe-Hackathon-Round-1

import os
import json
import fitz  # PyMuPDF
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
import datetime

# --- CONFIGURATION BASED ON HACKATHON STANDARDS ---
# These paths match the Docker volume mounts: -v $(pwd)/input:/app/input
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"

# Based on the specified project structure (e.g., "Collection 1/PDFs/")
PDF_SUBDIR = "PDFs" 
INPUT_JSON_NAME = "challenge1b_input.json"
OUTPUT_JSON_NAME = "challenge1b_output.json"

# --- CONFIGURATION FOR ROBUSTNESS (Tunable Parameters) ---
TITLE_WEIGHT = 0.3
CONTENT_WEIGHT = 0.7
MIN_RELEVANCE_SCORE = 0.2
MIN_WORDS_FOR_SUMMARY_SENTENCE = 5
HEADING_BLACKLIST_STARTERS = {'The', 'A', 'An', 'In', 'On', 'For', 'With', 'This'}


# --- GENERALIZED UTILITY FUNCTIONS (Unchanged) ---

def clean_text(text):
    """Normalizes whitespace and removes common unwanted characters."""
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'[\uf0b7\u2022]', '', text)
    return text.strip()

def extract_pdf_text_by_page(filepath):
    """Extracts and cleans text from a PDF given its full filepath."""
    if not os.path.isfile(filepath):
        print(f"⚠️ File not found: {filepath}")
        return []
    try:
        with fitz.open(filepath) as doc:
            return [(i + 1, clean_text(page.get_text("text"))) for i, page in enumerate(doc)]
    except Exception as e:
        print(f"❌ Failed to extract {filepath}: {e}")
        return []

def is_heading(line):
    """A robust, universal heuristic to determine if a line is a heading."""
    stripped = line.strip()
    if not stripped or stripped.endswith(('.', ',', ':')):
        return False
    words = stripped.split()
    if words and words[0] in HEADING_BLACKLIST_STARTERS:
        return False
    if ',' in stripped:
        return False
    if re.search(r'[@=+\-@\[\](){}]', stripped) or len(re.findall(r'\d', stripped)) > 5:
        return False
    if 1 <= len(words) <= 10:
        if stripped.isupper() or stripped.istitle() or re.match(r'^([IVXLCDM]+\.|[A-Z]\.)', stripped):
            return True
    return False

def extract_sections(pages_text, doc_name):
    """A general-purpose function to extract sections using the is_heading heuristic."""
    sections, current_heading, current_content, current_page = [], "Introduction", "", 1
    for page_num, text in pages_text:
        for line in text.split('\n'):
            if is_heading(line):
                if current_content.strip():
                    sections.append({"document": doc_name, "section_title": current_heading, "content": current_content.strip(), "page_number": current_page})
                current_heading, current_content, current_page = line.strip(), "", page_num
            else:
                current_content += " " + line.strip()
    if current_content.strip():
        sections.append({"document": doc_name, "section_title": current_heading, "content": current_content.strip(), "page_number": current_page})
    return sections

def get_refined_summary(section_content, model, job_embedding, top_k=3):
    """A general-purpose summarizer with a minimum sentence length."""
    sentences = re.split(r'(?<=[.!?])\s+', section_content)
    meaningful_sentences = [s for s in sentences if len(s.split()) >= MIN_WORDS_FOR_SUMMARY_SENTENCE]
    if not meaningful_sentences:
        return section_content
    sentence_embeddings = model.encode(meaningful_sentences, convert_to_tensor=True)
    similarities = util.cos_sim(job_embedding, sentence_embeddings)[0]
    top_indices = np.argsort(-similarities)[:top_k]
    top_indices.sort()
    summary = " ".join([meaningful_sentences[idx].strip() for idx in top_indices])
    return summary if summary else section_content


# --- MAIN PROCESSING LOGIC ---

def process_document_collection(input_data):
    """
    Takes the loaded input JSON and runs the full analysis pipeline.
    """
    model = SentenceTransformer('all-MiniLM-L6-v2')
    job_text = input_data["job_to_be_done"]["task"]
    job_embedding = model.encode(job_text, convert_to_tensor=True)

    all_sections = []
    for doc in input_data["documents"]:
        # Construct the full path to the PDF inside the input directory
        pdf_path = os.path.join(INPUT_DIR, PDF_SUBDIR, doc["filename"])
        pages_text = extract_pdf_text_by_page(pdf_path)
        if pages_text:
            all_sections.extend(extract_sections(pages_text, doc["filename"]))
    
    if not all_sections:
        print("Could not extract any sections from the provided documents.")
        return None

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
            "input_documents": [doc["filename"] for doc in input_data["documents"]],
            "persona": input_data["persona"]["role"],
            "job_to_be_done": input_data["job_to_be_done"]["task"]
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
            
    return output

if __name__ == "__main__":
    # This is the main execution block that will run in the Docker container.
    
    # 1. Construct the path to the input JSON file.
    input_json_path = os.path.join(INPUT_DIR, INPUT_JSON_NAME)
    
    try:
        # 2. Read the input JSON file.
        with open(input_json_path, 'r', encoding='utf-8') as f:
            input_data = json.load(f)
        print(f"Successfully loaded input file: {input_json_path}")
        
        # 3. Run the core analysis logic.
        output_data = process_document_collection(input_data)
        
        if output_data:
            # 4. Construct the path for the output JSON file.
            output_json_path = os.path.join(OUTPUT_DIR, OUTPUT_JSON_NAME)
            os.makedirs(OUTPUT_DIR, exist_ok=True) # Ensure output directory exists

            # 5. Write the final output.
            with open(output_json_path, "w", encoding='utf-8') as f:
                json.dump(output_data, f, indent=4)
            print(f"✅ Success! Output saved to {output_json_path}")
            
    except FileNotFoundError:
        print(f"❌ ERROR: Input file not found at {input_json_path}. Please check the input volume.")
    except Exception as e:
        print(f"❌ An unexpected error occurred: {e}")