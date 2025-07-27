#!/usr/bin/env python3
import os
import json
import traceback
from parser.pdf_parser import process_pdf
from parser.heading_detector import extract_outline

def main():
    input_dir = 'input'
    output_dir = 'output'
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    processed_count = 0
    error_count = 0
    
    for filename in sorted(os.listdir(input_dir)):
        if filename.lower().endswith('.pdf'):
            input_path = os.path.join(input_dir, filename)
            output_name = os.path.splitext(filename)[0]
            output_path = os.path.join(output_dir, f"{output_name}.json")
            
            print(f"\nProcessing {filename}...")
            
            try:
                # Step 1: Parse PDF into structured elements
                pdf_elements = process_pdf(input_path)
                
                # Step 2: Extract title and headings
                result = extract_outline(pdf_elements)
                
                # Step 3: Save JSON output
                with open(output_path, 'w') as f:
                    json.dump(result, f, indent=2)
                
                print(f"Successfully processed {filename}")
                print(f"Title: {result['title']}")
                print(f"Found {len(result['outline'])} headings")
                processed_count += 1
                
            except Exception as e:
                error_count += 1
                print(f"Error processing {filename}")
                print(f"Error type: {type(e).__name__}")
                print(f"Error message: {str(e)}")
                print("Skipping this file...")
                # traceback.print_exc()  # Uncomment for debugging
    
    print(f"\nProcessing complete. {processed_count} files processed successfully, {error_count} errors.")

if __name__ == "__main__":
    main()