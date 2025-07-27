import os
import json
import fitz  # PyMuPDF
import numpy as np
import re
from sentence_transformers import SentenceTransformer, util
import datetime
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline

# --- Define Input/Output Paths for Docker Compliance ---
INPUT_DIR = "/app/input"
OUTPUT_DIR = "/app/output"

# --- LOAD INPUT FROM A JSON FILE ---
input_json_path = os.path.join(INPUT_DIR, "input.json")
try:
    with open(input_json_path, "r", encoding='utf-8') as f:
        input_data = json.load(f)
except (FileNotFoundError, json.JSONDecodeError) as e:
    print(f"❌ Error reading {input_json_path}: {e}")
    exit()

# --- CONFIGURATION ---
TITLE_WEIGHT = 0.3
CONTENT_WEIGHT = 0.7
MIN_RELEVANCE_SCORE = 0.2
MIN_WORDS_FOR_SUMMARY_SENTENCE = 5
HEADING_SCORE_PERCENTILE = 95
LLM_VERIFICATION_TOP_K = 25 

# --- HACKATHON-COMPLIANT MODEL NAMES ---
RETRIEVAL_MODEL_NAME = 'all-MiniLM-L6-v2'
LLM_MODEL_NAME = "google/flan-t5-small"


# --- UTILITY FUNCTIONS ---
def extract_pdf_text_by_page(filename):
    if not os.path.isfile(filename): 
        print(f"❌ PDF file not found at: {filename}")
        return []
    try:
        with fitz.open(filename) as doc:
            return [(i + 1, page.get_text("text")) for i, page in enumerate(doc)]
    except Exception as e: 
        print(f"❌ Could not process PDF {filename}: {e}")
        return []

def score_line_as_heading(line):
    stripped = line.strip()
    words = stripped.split()
    word_count = len(words)
    if not stripped or word_count > 10 or stripped.endswith(('.', ',', ':', ';')) or ',' in stripped or (words and words[0].islower()):
        return 0
    score = 0
    if 1 <= word_count <= 5: score += 2
    if stripped.istitle(): score += 3
    if stripped.isupper() and word_count > 1: score += 4
    if re.match(r'^([IVXLCDM]+\.|[A-Z]\.)', stripped): score += 5
    return score

def extract_dynamic_sections(pages_text, doc_name):
    line_profiles = [{"text": line.strip(), "page": p_num, "line_num": l_num, "score": score_line_as_heading(line)}
                     for p_num, text in pages_text for l_num, line in enumerate(text.split('\n')) if line.strip()]
    if not line_profiles: return []
    scores = [p['score'] for p in line_profiles if p['score'] > 0]
    if not scores:
        full_content = " ".join(p['text'] for p in line_profiles)
        return [{"document": doc_name, "section_title": "Full Document", "content": full_content, "page_number": 1}]
    
    dynamic_threshold = np.percentile(scores, HEADING_SCORE_PERCENTILE)
    headings = sorted([p for p in line_profiles if p['score'] >= dynamic_threshold], key=lambda x: (x['page'], x['line_num']))
    if not headings:
        full_content = " ".join(p['text'] for p in line_profiles)
        return [{"document": doc_name, "section_title": "Full Document", "content": full_content, "page_number": 1}]

    sections = []
    for i, heading in enumerate(headings):
        start_page, start_line = heading['page'], heading['line_num']
        end_page, end_line = (headings[i+1]['page'], headings[i+1]['line_num']) if i + 1 < len(headings) else (len(pages_text) + 1, float('inf'))
        content = [line for p_num, p_text in pages_text if start_page <= p_num <= end_page 
                   for l_num, line in enumerate(p_text.split('\n')) 
                   if (p_num > start_page or l_num > start_line) and (p_num < end_page or l_num < end_line)]
        sections.append({"document": doc_name, "section_title": heading['text'], "content": "\n".join(content).strip(), "page_number": start_page})
    return sections

def get_refined_summary(section_content, model, job_embedding, top_k=3):
    cleaned_content = re.sub(r'\s+', ' ', section_content)
    cleaned_content = re.sub(r'[\u2022\uf0b7\uf06f]', '', cleaned_content)

    sentences = re.split(r'(?<=[.!?])\s+', cleaned_content)
    meaningful_sentences = [s.strip() for s in sentences if len(s.split()) >= MIN_WORDS_FOR_SUMMARY_SENTENCE]
    if not meaningful_sentences: return cleaned_content
    
    sentence_embeddings = model.encode(meaningful_sentences, convert_to_tensor=True)
    similarities = util.cos_sim(job_embedding, sentence_embeddings)[0]
    top_indices = sorted(np.argsort(-similarities)[:top_k])
    return " ".join([meaningful_sentences[idx] for idx in top_indices]) or cleaned_content

# --- MAIN PROCESSING LOGIC ---
with torch.no_grad():
    print("INFO: Loading models...")
    retrieval_model = SentenceTransformer(RETRIEVAL_MODEL_NAME)
    llm_pipeline = pipeline("text2text-generation", model=LLM_MODEL_NAME, tokenizer=LLM_MODEL_NAME, device=-1)

    job_text = input_data["job_to_be_done"]["task"]
    job_embedding = retrieval_model.encode(job_text, convert_to_tensor=True)
    all_sections = []
    for doc in input_data["documents"]:
        pdf_path = os.path.join(INPUT_DIR, doc["filename"])
        pages_text = extract_pdf_text_by_page(pdf_path)
        if pages_text:
            all_sections.extend(extract_dynamic_sections(pages_text, doc["filename"]))

    if not all_sections:
        print("❌ No valid sections could be extracted. Exiting.")
        exit()

    print(f"INFO: Performing initial semantic ranking on {len(all_sections)} sections...")
    titles = [sec["section_title"] for sec in all_sections]
    contents = [sec["content"] for sec in all_sections]
    title_embeddings = retrieval_model.encode(titles, convert_to_tensor=True)
    content_embeddings = retrieval_model.encode(contents, convert_to_tensor=True)
    title_similarities = util.cos_sim(job_embedding, title_embeddings)[0]
    content_similarities = util.cos_sim(job_embedding, content_embeddings)[0]
    combined_scores = (TITLE_WEIGHT * title_similarities) + (CONTENT_WEIGHT * content_similarities)
    ranked_indices = np.argsort(-combined_scores)

    print(f"INFO: Applying smart verification to the Top {LLM_VERIFICATION_TOP_K} sections...")
    output = {
        "metadata": { "persona": input_data["persona"]["role"], "job_to_be_done": job_text,
                      "input_documents": [doc["filename"] for doc in input_data["documents"]],
                      "processing_timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat() },
        "extracted_sections": [], "subsection_analysis": []
    }

    final_rank = 1
    for rank, idx in enumerate(ranked_indices):
        if final_rank > 15: break 
        section = all_sections[idx]
        if combined_scores[idx] < MIN_RELEVANCE_SCORE: continue

        is_compliant = True
        if rank < LLM_VERIFICATION_TOP_K:
            content_snippet = " ".join(section['content'].split()[:250])
            # (MODIFIED) This is the truly generic, universal prompt.
            prompt = f"Analyze if the following text section conflicts with or violates the user's primary goal. User Goal: '{job_text}'. Text Section: '{section['section_title']}. {content_snippet}'. Does the text section violate the user's goal? Answer only with 'Yes' or 'No'."
            
            try:
                outputs = llm_pipeline(prompt, max_new_tokens=3)
                answer = outputs[0]['generated_text'].strip().lower()
                # If the LLM says 'Yes' it violates the goal, then it is NOT compliant.
                if 'yes' in answer:
                    is_compliant = False
            except Exception as e:
                print(f"⚠️ LLM verification failed, assuming compliance. Error: {e}")
                pass
        
        if not is_compliant:
            continue

        clean_title = re.sub(r'^[\W_]+', '', section["section_title"]).strip()
        if not clean_title: continue

        output["extracted_sections"].append({
            "document": section["document"], "section_title": clean_title,
            "importance_rank": final_rank, "page_number": section["page_number"]
        })
        
        refined_text = get_refined_summary(section["content"], retrieval_model, job_embedding)
        
        output["subsection_analysis"].append({
            "document": section["document"], "page_number": section["page_number"],
            "refined_text": refined_text
        })
        final_rank += 1
            
    main_output_path = os.path.join(OUTPUT_DIR, "result.json")
    with open(main_output_path, "w", encoding='utf-8') as f:
        json.dump(output, f, indent=4)
    print(f"✅ Done: Main output saved to {main_output_path}")