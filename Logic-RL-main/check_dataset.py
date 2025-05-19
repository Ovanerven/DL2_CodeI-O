#!/usr/bin/env python3
"""
Script to verify the dataset format for reason_io training.
"""

import pandas as pd
import json
from transformers import AutoTokenizer

# Path to the dataset
DATASET_PATH = "data/reason_io/reason_io_dataset_preview.parquet"

def main():
    print(f"Checking dataset at {DATASET_PATH}")
    
    # Load the dataset
    df = pd.read_parquet(DATASET_PATH)
    print(f"Dataset loaded: {len(df)} rows")
    print(f"Columns: {df.columns}")
    
    # Check prompt format
    if 'prompt' in df.columns:
        print("\nChecking prompt format...")
        
        # Get a sample row
        row = df.iloc[0]
        prompt = row['prompt']
        
        print(f"Prompt type: {type(prompt)}")
        print(f"Prompt content: {prompt}")
        
        # Check if it's the expected format (array of dicts with 'role' and 'content')
        if isinstance(prompt, list):
            for message in prompt:
                print(f"- Message role: {message.get('role')}")
                if message.get('role') == 'system':
                    print(f"Found system message:\n{message.get('content')}")
        
        # Load tokenizer and check how the prompt is processed
        print("\nChecking tokenized prompt...")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        
        # Apply chat template
        chat_output = tokenizer.apply_chat_template(prompt, add_generation_prompt=True, tokenize=False)
        print(f"Chat template output:")
        print(chat_output)
        
        # Tokenize and show token count
        tokens = tokenizer.encode(chat_output)
        print(f"Token count: {len(tokens)}")
    
    else:
        print("Warning: No 'prompt' column found in dataset")
    
    # If there are expected_solution or solution fields, check them
    solution_fields = [col for col in df.columns if 'solution' in col]
    if solution_fields:
        print("\nChecking solution format...")
        for field in solution_fields:
            print(f"Solution field: {field}")
            solution = df.iloc[0][field]
            print(f"Type: {type(solution)}")
            print(f"Value: {solution}")
            
            # If it's a string that looks like JSON, try to parse it
            if isinstance(solution, str) and solution.strip().startswith('{'):
                try:
                    parsed = json.loads(solution)
                    print(f"Parsed JSON: {parsed}")
                except json.JSONDecodeError:
                    print("Failed to parse as JSON")

if __name__ == "__main__":
    main() 