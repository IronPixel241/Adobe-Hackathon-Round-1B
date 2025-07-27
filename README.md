# 🚀 Adobe Hackathon 2025 – Persona-Driven Document Intelligence

This project was developed for **Round 1B** of the [Adobe “Connecting the Dots” Hackathon](https://github.com/jhaaj08/Adobe-India-Hackathon25). It transforms static PDF documents into an intelligent assistant tailored to specific personas and tasks.

---

## 📌 Problem Statement

Given a **collection of related PDFs**, a **persona** (e.g., Researcher, Student, Analyst), and a **specific job-to-be-done**, the system intelligently:
- Extracts and segments the documents into meaningful sections
- Ranks sections based on semantic relevance
- Applies lightweight LLM verification to ensure alignment with the task
- Outputs structured JSON with ranked sections and refined content

---

## ✨ Key Features

### 🧠 Hybrid AI Architecture
- **Vector Embedding (Fast Retrieval)**: Uses `all-MiniLM-L6-v2` (≤90MB) to perform semantic search across all document sections.
- **LLM Verification (Deep Understanding)**: Uses `google/flan-t5-small` (≈100MB) as a fast, local verifier for top 20 candidates only.

### ⚡ Performance-Aware Pipeline
- Smartly balances speed and intelligence within the strict **60-second** offline runtime limit.
- Applies LLM only to most promising sections to save time and compute.

### 📄 Robust Section Segmentation
- Dynamically segments PDFs into logical sections using:
  - Title case detection
  - Heuristics on line length, indentation, and structure
  - No reliance on font size or brittle PDF rules

---

## 🛠️ Tech Stack

| Component              | Description                                  |
|------------------------|----------------------------------------------|
| 📚 Libraries           | `PyMuPDF`, `sentence-transformers`, `NumPy`, `torch`, `transformers` |
| 🔍 Embedding Model     | `all-MiniLM-L6-v2` (via sentence-transformers) |
| 🤖 Lightweight LLM     | `google/flan-t5-small` (via HuggingFace Transformers) |
| 🐳 Containerization    | Docker (CPU-only, offline, amd64-compatible)  |

---

## 📂 Directory Structure

```
├── local_input/ 
│   └── input.json          # place the input.json file here
├── local_output/           # Results (JSON) will be generated here
├── src/
│   └── init.py             # intialization
│   └── main.py             # Pipeline orchestration
├── Dockerfile              # Dockerfile to containerize the solution
├── requirements.txt
├── download_models.py      # locally caches the model while running docker build (no network usage during docker run)
└── README.md               # (You're reading it!)
```

---

## 🧪 How It Works

### Step-by-Step Approach:

1. **Section Extraction**
   - All PDFs in the `input/` folder are parsed using `PyMuPDF`.
   - A custom heuristic algorithm splits text into titles, sections, and subsections.

2. **Semantic Ranking**
   - Persona and job-to-be-done are embedded into vectors.
   - Each section’s title and content is vectorized and compared using **cosine similarity**.

3. **LLM Verification**
   - Top 20 semantically relevant sections are passed through `flan-t5-small`.
   - LLM ensures they logically fit the persona’s needs (e.g., "Is this relevant for a Chemistry student preparing for kinetics?").

4. **Output Generation**
   - A structured JSON is written to the `output/` directory in the required format (matching `challenge1b_output.json`).

---

## 🚀 Quick Start

### 🐳 Build Docker Image

```bash
docker build -t persona-doc-intel:1b .
```

### ▶️ Run the Container

```bash
docker run --rm   -v "$(pwd)/input:/app/input"   -v "$(pwd)/output:/app/output"   --network none   persona-doc-intel:1b
```

> 📌 This will generate `result.json` or `{filename}.json` inside the `output/` folder.

---

## 📥 Input Format

**input/input.json** (example):

```json
{
  "persona": "Undergraduate Chemistry Student",
  "job": "Identify key concepts and mechanisms for exam preparation on reaction kinetics"
}
```

**input/** also contains PDFs:
```
input/
├── input.json
├── Organic_Chemistry_Chapter1.pdf
├── Organic_Chemistry_Chapter2.pdf
└── ...
```

---

## 📤 Output Format

```json
{
  "metadata": {
    "documents": ["Organic_Chemistry_Chapter1.pdf", ...],
    "persona": "Undergraduate Chemistry Student",
    "job": "Identify key concepts and mechanisms for exam preparation on reaction kinetics",
    "timestamp": "2025-07-27T17:35:10Z"
  },
  "extracted_sections": [
    {
      "document": "Organic_Chemistry_Chapter1.pdf",
      "page_number": 5,
      "section_title": "Reaction Kinetics: Overview",
      "importance_rank": 1
    }
    ...
  ],
  "sub_section_analysis": [
    {
      "document": "Organic_Chemistry_Chapter1.pdf",
      "refined_text": "This section explains the rate-determining steps...",
      "page_number": 5
    }
  ]
}
```

---

## 🧾 Constraints Met

| Constraint               | Status       |
|--------------------------|--------------|
| ⏱️ Execution Time        | ≤ 60 sec     |
| 📦 Model Size            | ≤ 1GB        |
| ❌ No Internet Required  | ✅ Fully offline |
| 💻 CPU Only              | ✅ Supported |
| 🔄 Multidocument Support | ✅ 3–10 PDFs |

---

## 📚 Reference

- [Sentence-Transformers](https://www.sbert.net/)
- [Flan-T5 Small on HuggingFace](https://huggingface.co/google/flan-t5-small)
- Adobe Challenge Docs: [PDF Brief](https://github.com/jhaaj08/Adobe-India-Hackathon25)

---

## 🏁 Final Notes

> This solution is designed to generalize well across different domains (academic, financial, educational, etc.).  
> All models are local and efficient, making the system fully compliant with Adobe’s hackathon constraints.
