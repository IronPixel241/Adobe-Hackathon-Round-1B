Adobe Hackathon 2025 - Persona-Driven Document Intelligence
This project is a persona-driven document intelligence system built for the "Connecting the Dots" Hackathon. It analyzes a collection of PDF documents to extract and rank the most relevant sections based on a user's role and specific task.


Key Features
Hybrid AI Approach: Combines the speed and breadth of semantic vector search with the deep comprehension of a lightweight LLM for state-of-the-art relevance ranking.


Performance-Aware Logic: Intelligently applies the most computationally expensive LLM checks only to the top-ranked candidates, ensuring the solution stays well within the strict 60-second execution limit.


Dynamic Document Parsing: Utilizes a custom scoring algorithm to identify document structure, making the solution robust and adaptable to PDFs without relying on brittle rules like font size.

Approach Explanation 


Our system follows a multi-stage process to deliver highly relevant, persona-driven insights:

Section Extraction: We first parse text from each PDF using the PyMuPDF library. A custom scoring algorithm then analyzes text characteristics (e.g., title case, length, structure) to dynamically segment the documents into logical sections and subsections.

Semantic Ranking: We employ the all-MiniLM-L6-v2 sentence-transformer model to generate vector embeddings for the persona's job description, as well as for the title and content of every extracted section. By calculating the cosine similarity, we produce a ranked list of all sections based on their semantic relevance to the user's task.

Intelligent Verification: To refine the top results without sacrificing performance, a lightweight LLM (google/flan-t5-small) performs a final verification step. To remain fast, this check is only applied to the Top 5 semantically-ranked sections. It acts as an expert reviewer, confirming their alignment with the user's goal and filtering out any nuanced mismatches.

This hybrid strategy ensures our solution is not only accurate and intelligent but also fully compliant with the hackathon's offline and performance constraints.


Libraries and Models Used 

Libraries: PyMuPDF, NumPy, Sentence-Transformers, PyTorch (CPU), Transformers

Retrieval Model: all-MiniLM-L6-v2

LLM Model: google/flan-t5-small

How to Build and Run 

Build the Docker Image
From the root directory, run the following command:

Bash

docker build -t my-adobe-solution:1b .
Run the Container
Place your input PDFs and input.json file into a local directory (e.g., ./local_input). Create an empty local output directory (e.g., ./local_output). Then, run the container using the specified command format:

Bash

docker run --rm \
  -v "$(pwd)/local_input:/app/input" \
  -v "$(pwd)/local_output:/app/output" \
  --network none \
  my-adobe-solution:1b
The results will be generated in local_output/result.json.