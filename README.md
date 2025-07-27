# Adobe Hackathon 2025 - Round 1B Submission

[cite_start]This project is a persona-driven document intelligence system built for the "Connecting the Dots" Hackathon[cite: 5, 27].

## [cite_start]Approach Explanation [cite: 89, 158]

Our system analyzes a collection of PDF documents to extract the most relevant sections based on a given user persona and their specific "job-to-be-done."

1.  **Section Extraction**: We use the PyMuPDF library to parse text from each PDF. A custom scoring algorithm identifies potential headings based on features like title case, capitalization, and structure. [cite_start]This allows us to dynamically segment documents without relying on fixed font sizes[cite: 94].
2.  **Semantic Ranking**: We employ a `all-MiniLM-L6-v2` sentence-transformer model to generate embeddings for the job description, section titles, and section content. A weighted combination of title and content similarity scores is used to rank all extracted sections.
3.  **Intelligent Verification & Refinement**: A lightweight LLM (`google/flan-t5-small`) acts as a final filter. It verifies if a top-ranked section is genuinely aligned with the user's goal. For the final output, a summary of the most relevant sentences from each top-ranked section is generated to provide a concise `refined_text`.

[cite_start]This solution is designed to run completely offline [cite: 60] [cite_start]and respects all model size ($\le1GB$) and performance constraints[cite: 152, 153].

## [cite_start]Libraries and Models Used [cite: 90]

* **Libraries**: PyMuPDF, NumPy, Sentence-Transformers, PyTorch (CPU), Transformers
* **Retrieval Model**: `all-MiniLM-L6-v2`
* **LLM Model**: `google/flan-t5-small`

## [cite_start]How to Build and Run [cite: 93]

### Build the Docker Image

From the root directory, run the following command:

```bash
docker build -t my-adobe-solution:1b .
```

### Run the Container

Place your input PDFs and `input.json` file into a local directory (e.g., `./local_input`). Create an empty local output directory (e.g., `./local_output`). Then, run the container using the specified command format:

```bash
docker run --rm \
  -v "$(pwd)/local_input:/app/input" \
  -v "$(pwd)/local_output:/app/output" \
  --network none \
  my-adobe-solution:1b
```

The results will be generated in `local_output/result.json`.