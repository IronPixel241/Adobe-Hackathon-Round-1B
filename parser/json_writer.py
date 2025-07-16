import json
import os

def write_output_json(title, outline, output_path):
    output = {
        "title": title,
        "outline": outline
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)
