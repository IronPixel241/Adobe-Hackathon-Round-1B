# download_model.py
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sentence_transformers import SentenceTransformer

# The small, compliant model for intelligent verification (< 1GB)
LLM_MODEL_NAME = "google/flan-t5-small"
# The model for initial retrieval
RETRIEVAL_MODEL_NAME = 'all-MiniLM-L6-v2'

print(f"Downloading verification model: {LLM_MODEL_NAME}...")
# This line downloads and caches the tokenizer and model files
tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(LLM_MODEL_NAME)
print("Verification model downloaded successfully.")

print(f"Downloading retrieval model: {RETRIEVAL_MODEL_NAME}...")
# This line downloads and caches the sentence transformer model
ret_model = SentenceTransformer(RETRIEVAL_MODEL_NAME)
print("Retrieval model downloaded successfully.")

print("\nAll models are downloaded and cached for offline use.")