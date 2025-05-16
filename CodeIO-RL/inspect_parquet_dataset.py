import pandas as pd
import json
import argparse
import os
from pprint import pprint

def inspect_parquet(file_path, num_samples=2, detailed=False):
    """
    Inspect a parquet file's structure and content.
    
    Args:
        file_path: Path to the parquet file
        num_samples: Number of samples to display
        detailed: Whether to show detailed content of each record
    """
    print(f"\n=== Inspecting Parquet File: {file_path} ===\n")
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"Error: File {file_path} not found")
        return
    
    try:
        # Read the parquet file
        df = pd.read_parquet(file_path)
        
        # Display basic information
        print(f"Number of records: {len(df)}")
        print(f"Columns: {df.columns.tolist()}")
        print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024 / 1024:.2f} MB")
        print(f"Data types:\n{df.dtypes}\n")
        
        # Display sample records
        if detailed:
            for i in range(min(num_samples, len(df))):
                print(f"\n=== Sample {i+1} ===")
                record = df.iloc[i].to_dict()
                
                # Handle the prompt field specifically
                if 'prompt' in record:
                    print("\n--- Prompt ---")
                    try:
                        if isinstance(record['prompt'], str):
                            prompt_data = json.loads(record['prompt'])
                        else:
                            prompt_data = record['prompt']
                        
                        # Extract the user prompt (usually the second message in chat format)
                        if isinstance(prompt_data, list) and len(prompt_data) > 1:
                            user_message = prompt_data[1].get('content', '') if 'content' in prompt_data[1] else prompt_data[1]
                            print(f"User message preview: {user_message[:100]}...")
                        else:
                            print(f"Prompt preview: {str(prompt_data)[:100]}...")
                    except Exception as e:
                        print(f"Error parsing prompt: {e}")
                        print(f"Raw prompt: {str(record['prompt'])[:100]}...")
                
                # Handle ground truth data
                print("\n--- Ground Truth ---")
                for key in record:
                    if 'reward_model' in key or 'ground_truth' in key:
                        try:
                            if isinstance(record[key], str) and record[key].startswith('{'):
                                parsed = json.loads(record[key])
                                print(f"{key}: {json.dumps(parsed, indent=2)}")
                            else:
                                print(f"{key}: {record[key]}")
                        except:
                            print(f"{key}: {record[key]}")
                
                # Display reference code length if available
                if 'reference_code_length' in record:
                    print(f"\nReference code length: {record['reference_code_length']}")
                
                print("="*50)
        else:
            # Just display column names and structure
            print("\nSample record keys:")
            pprint(df.iloc[0].to_dict().keys())
            
            # Try to parse and display some critical fields to verify structure
            print("\nSample data_source value:", df['data_source'].iloc[0])
            
            # Check if prompt is structured correctly
            if 'prompt' in df.columns:
                print("\nPrompt structure check:")
                sample_prompt = df['prompt'].iloc[0]
                if isinstance(sample_prompt, list) and len(sample_prompt) > 0:
                    print(f"- Prompt is a list with {len(sample_prompt)} items")
                    if len(sample_prompt) > 1 and 'content' in sample_prompt[1]:
                        print("- Contains expected chat format with 'content' field")
                else:
                    print(f"- Warning: Prompt is not in expected list format")
            
            # Check ground truth fields
            for col in df.columns:
                if 'solution' in col:
                    print(f"\nSolution field found: {col}")
                    try:
                        if isinstance(df[col].iloc[0], str):
                            solution_data = json.loads(df[col].iloc[0])
                            print(f"- Parsed solution data: {json.dumps(solution_data, indent=2)[:100]}...")
                    except:
                        print(f"- Raw solution data: {str(df[col].iloc[0])[:100]}...")
    
    except Exception as e:
        print(f"Error reading parquet file: {str(e)}")

def main():
    parser = argparse.ArgumentParser(description='Inspect a parquet dataset')
    parser.add_argument('file_path', type=str, help='Path to the parquet file')
    parser.add_argument('--samples', type=int, default=2, help='Number of sample records to display')
    parser.add_argument('--detailed', action='store_true', help='Show detailed content of records')
    
    args = parser.parse_args()
    inspect_parquet(args.file_path, args.samples, args.detailed)

if __name__ == '__main__':
    main() 