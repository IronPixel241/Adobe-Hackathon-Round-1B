import os
import json
import fitz  # PyMuPDF
import numpy as np
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

# --- Input JSON (can replace with external file read) ---
input_data = {
    "challenge_info": {
        "challenge_id": "round_1b_003",
        "test_case_name": "create_manageable_forms",
        "description": "Creating manageable forms"
    },
    "documents": [
        {"filename": "Learn Acrobat - Fill and Sign.pdf", "title": "Learn Acrobat - Fill and Sign"},
        {"filename": "Learn Acrobat - Request e-signatures_1.pdf", "title": "Learn Acrobat - Request e-signatures_1"}
    ],
    "persona": {"role": "HR professional"},
    "job_to_be_done": {"task": "Create and manage fillable forms for onboarding and compliance."}
}

# --- Keyword extraction ---
def extract_keywords_from_task(task_text, top_k=6):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=top_k)
    tfidf = vectorizer.fit_transform([task_text])
    return vectorizer.get_feature_names_out()

# --- Extract PDF text by page ---
def extract_pdf_text_by_page(filename):
    try:
        page_texts = []
        with fitz.open(filename) as doc:
            for i in range(len(doc)):  # Access len(doc) inside 'with'
                page = doc[i]
                page_texts.append((i, page.get_text()))
            page_count = len(doc)
        return page_texts, page_count
    except Exception as e:
        print(f"❌ Failed to extract {filename}: {e}")
        return [], 0


# --- Sentence-aware, page-aware snippet extraction ---
def extract_snippets_by_page(text_by_page, keywords, max_snippets=5):
    results = []
    for page_num, text in text_by_page:
        sentences = re.split(r'(?<=[.!?])\s+', text)
        for sentence in sentences:
            lowered = sentence.lower()
            if any(kw in lowered for kw in keywords):
                results.append({
                    "sentence": sentence.strip(),
                    "page": page_num + 1  # Pages are 1-indexed
                })
                if len(results) >= max_snippets:
                    return results
    return results

# --- Load and process PDFs ---
mock_pdfs = {}
doc_list = input_data["documents"][:2]  # Limit to 2 for this test
for doc in doc_list:
    filename = doc["filename"]
    if os.path.isfile(filename):
        page_texts, page_count = extract_pdf_text_by_page(filename)
        mock_pdfs[filename] = {"pages": page_count, "page_texts": page_texts}
    else:
        print(f"⚠️ File not found: {filename}")
        mock_pdfs[filename] = {"pages": 0, "page_texts": []}

# --- Process job and similarity scoring ---
job_text = input_data["job_to_be_done"]["task"]
persona = input_data["persona"]["role"]
keywords = extract_keywords_from_task(job_text)

texts = ["\n".join([text for _, text in mock_pdfs[doc["filename"]]["page_texts"]]) for doc in doc_list]
doc_names = [doc["filename"] for doc in doc_list]

# TF-IDF
vectorizer = TfidfVectorizer(vocabulary=keywords)
tfidf_scores = np.sum(vectorizer.fit_transform(texts).toarray(), axis=1)

# Semantic similarity
model = SentenceTransformer('all-MiniLM-L6-v2')  # ~80MB
job_embedding = model.encode(job_text)
doc_embeddings = model.encode(texts)
semantic_scores = cosine_similarity([job_embedding], doc_embeddings)[0]

# Combine scores
combined_scores = 0.6 * tfidf_scores + 0.4 * semantic_scores
ranked_indices = np.argsort(-combined_scores)

# --- Build output JSON ---
output = {
    "metadata": {
        "persona": persona,
        "job_to_be_done": job_text,
        "input_documents": doc_names,
        "extracted_keywords": list(keywords),
        "processing_timestamp": "2025-07-21T10:00:00Z"
    },
    "extracted_sections": [],
    "subsection_analysis": []
}

for rank, idx in enumerate(ranked_indices, 1):
    doc_name = doc_names[idx]
    text_by_page = mock_pdfs[doc_name]["page_texts"]

    # Top-level section summary
    first_page_text = text_by_page[0][1] if text_by_page else ""
    section_lines = [line.strip() for line in first_page_text.split("\n") if line.strip()]
    main_section = section_lines[0] if section_lines else "No section found"

    output["extracted_sections"].append({
        "document": doc_name,
        "section_title": main_section,
        "importance_rank": rank,
        "page_number": 1
    })

    # Extract snippets with page numbers
    snippets = extract_snippets_by_page(text_by_page, keywords)
    for snip in snippets:
        output["subsection_analysis"].append({
            "document": doc_name,
            "refined_text": snip["sentence"],
            "page_number": snip["page"]
        })

# --- Save output JSON ---
output_file = f"output_{input_data['challenge_info']['test_case_name']}.json"
with open(output_file, "w") as f:
    json.dump(output, f, indent=4)

print(f"✅ Done: {output_file} generated.")
