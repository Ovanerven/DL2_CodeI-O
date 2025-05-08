import json
import re
import os
import time
from tqdm import tqdm
import pandas as pd
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import csv

def read_jsonl(filename):
    """Read jsonl files and return list of dictionaries."""
    data = []
    with open(filename, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Error decoding line in {filename}")
    return data

def extract_context(prompt):
    """Extract just the problem context from the prompt, without code snippets."""
    # Split at "Given the following output/input:"
    parts = re.split(r"Given the following (output|input):", prompt)
    if len(parts) > 1:
        raw_context = parts[0]
        
        # Remove template beginning
        template_start = "You are given a question that requires some input and output variables as follows:\n\n"
        if raw_context.startswith(template_start):
            raw_context = raw_context[len(template_start):]
        
        # Remove any reference code section by looking for the marker
        code_marker = "Tip: Here is a reference code snippet for this question."
        if code_marker in raw_context:
            # Only keep the part before the code marker
            raw_context = raw_context.split(code_marker)[0].strip()
        
        return raw_context.strip()
    return None

def extract_reference_code(prompt):
    """Extract only the runnable Python code from the reference section."""
    marker = "Tip: Here is a reference code snippet for this question."
    if marker in prompt:
        code_part = prompt.split(marker)[1].strip()
        lines = code_part.split('\n')
        cleaned_lines = []
        code_started = False
        
        for line in lines:
            if not code_started and (line.startswith('import ') or 
                                   line.startswith('from ') or 
                                   line.startswith('def ') or 
                                   line.startswith('class ') or 
                                   line.startswith('# ')):
                code_started = True
            if code_started:
                cleaned_lines.append(line)
        clean_code = '\n'.join(cleaned_lines)
        return clean_code
    return None

def process_data(input_file, max_rows=None):
    """Process the dataset and return a DataFrame with the extracted components."""
    data = read_jsonl(input_file)
    
    if max_rows is not None:
        data = data[:max_rows]
    
    records = []
    for item in tqdm(data):
        if 'prompt' not in item:
            continue
        
        context = extract_context(item['prompt'])
        reference_code = extract_reference_code(item['prompt'])
        
        if context and reference_code:
            records.append({
                'context': context,
                'reference_code': reference_code
            })
    
    return pd.DataFrame(records)

def create_input_generator_prompt(row):
    """Create a prompt for generating input generators."""
    inputgen_prompt_template = """
You are an expert programmer tasked with creating an input generator function for a given code snippet. This function will be used to generate test inputs for the code.

I'll provide you with a reference code implementation. Your job is to create a Python function called `input_generator()` that:

- You need to provide a function named `input_generator` that generates the input arguments for the `main_solution` function.
- The `input_generator` function should not require any input arguments, and each time it is called, it should return a set of input arguments that meet the requirements of the `main_solution` function.
- The output of `input_generator` should always be a dictionary because we always call by `**kwargs` in the `main_solution` function.
- Add some randomness in the `input_generator` function to ensure the input arguments are different each time it is called.
- Please try to make the generated input arguments as reasonable as possible, try to avoid generating too complex or too trivial input variables, also the size of the variables should be reasonable, like less than 1KB.

Here is the reference code:    
```python	
{reference_code}
```
Please respond with ONLY the input_generator() function definition. Your response should start with "import" statements if needed, followed by the function definition. Do not include any explanations or other text.
"""
    return inputgen_prompt_template.format(reference_code=row['reference_code'])

def call_deepseek_api_with_retry(client, prompt, temperature, max_retries=3, backoff_factor=2, timeout=30):
    """Call the Deepseek API with retry logic and exponential backoff."""
    if pd.isna(prompt) or not prompt.strip():
        return None
        
    retries = 0
    while retries <= max_retries:
        try:
            time.sleep(0.5)
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=[
                    {"role": "system", "content": "You are an expert Python programmer. Provide only valid, runnable Python code."},
                    {"role": "user", "content": prompt}
                ],
                temperature=temperature,
                stream=False,
                timeout=timeout
            )
            return response.choices[0].message.content
            
        except Exception as e:
            retries += 1
            if retries > max_retries:
                return f"# ERROR: {str(e)}"
            wait_time = backoff_factor ** retries
            print(f"API call failed, retrying in {wait_time:.1f}s... ({str(e)})")
            time.sleep(wait_time)
    
    return "# ERROR: Maximum retries exceeded"

def generate_input_generators_parallel(df, api_key, max_rows=None, temperature=0.3, max_workers=5, 
                                     save_interval=25, output_dir='../generated_data'):
    """Generate input generators in parallel with improved robustness."""
    os.makedirs(output_dir, exist_ok=True)
    
    result_df = df.copy()
    if max_rows is not None:
        result_df = result_df.iloc[:max_rows].copy()

    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompts = result_df['input_generator_prompt'].tolist()

    results = [None] * len(prompts)
    completed_count = 0
    
    start_time = time.time()
    
    print(f"Starting parallel API calls with {max_workers} workers, temperature={temperature}")
    
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {
            executor.submit(
                call_deepseek_api_with_retry, 
                client, 
                prompt, 
                temperature
            ): idx for idx, prompt in enumerate(prompts)
        }
        
        for future in tqdm(as_completed(futures), total=len(futures), desc="Parallel API calls"):
            idx = futures[future]
            results[idx] = future.result()
            
            completed_count += 1
            
            if completed_count % save_interval == 0:
                # Save interim results as JSONL
                timestamp = int(time.time())
                interim_file = os.path.join(
                    output_dir, 
                    f"input_generators_interim_{completed_count}_{timestamp}.jsonl"
                )
                
                with open(interim_file, 'w', encoding='utf-8') as f:
                    for i, result in enumerate(results):
                        if result is not None:
                            record = result_df.iloc[i].to_dict()
                            record['input_generator'] = result
                            f.write(json.dumps(record) + '\n')
                
                print(f"\nInterim progress saved ({completed_count}/{len(prompts)}) to {interim_file}")
                
                elapsed = time.time() - start_time
                rate = completed_count / elapsed
                remaining = (len(prompts) - completed_count) / rate if rate > 0 else 0
                
                print(f"Elapsed: {elapsed:.1f}s | Rate: {rate:.2f} items/s | Est. remaining: {remaining:.1f}s")
    
    # Save final results as JSONL
    final_file = os.path.join(output_dir, "input_generators.jsonl")
    with open(final_file, 'w', encoding='utf-8') as f:
        for i, result in enumerate(results):
            if result is not None:
                record = result_df.iloc[i].to_dict()
                record['input_generator'] = result
                f.write(json.dumps(record) + '\n')
    
    print(f"Saved results with {len(results)} rows to {final_file}")
    
    success_count = sum(1 for r in results if r is not None and not str(r).startswith("# ERROR:"))
    error_count = sum(1 for r in results if r is not None and str(r).startswith("# ERROR:"))
    empty_count = sum(1 for r in results if r is None)
    
    print(f"Results summary:")
    print(f"  Success: {success_count} ({success_count/len(results)*100:.1f}%)")
    print(f"  Errors: {error_count} ({error_count/len(results)*100:.1f}%)")
    print(f"  Empty: {empty_count} ({empty_count/len(results)*100:.1f}%)")
    
    return result_df

def clean_code_block(code_text):
    """Clean a code block by removing markdown formatting."""
    if pd.isna(code_text):
        return code_text
        
    if code_text.strip().startswith("```python"):
        code_text = code_text.replace("```python", "", 1)
    
    code_text = re.sub(r"```$", "", code_text.strip())
    
    return code_text.strip()

def process_and_generate(input_file, output_file, api_key, max_rows=None, temperature=0.3, max_workers=5):
    """Process the dataset and generate input generators line by line."""
    records = []
    processed_count = 0
    
    with open(input_file, 'r', encoding='utf-8') as f:
        for line in tqdm(f):
            if max_rows and processed_count >= max_rows:
                break
                
            try:
                # Parse the JSONL line
                item = json.loads(line)
                
                # Extract code from the item if it exists, otherwise from prompt
                if 'code' in item:
                    reference_code = item['code']
                else:
                    reference_code = extract_reference_code(item['prompt'])
                
                # Extract context from prompt
                context = extract_context(item['prompt'])
                
                if not context or not reference_code:
                    continue
                
                # Generate input generator
                prompt = create_input_generator_prompt({'reference_code': reference_code})
                input_generator = call_deepseek_api_with_retry(
                    OpenAI(api_key=api_key, base_url="https://api.deepseek.com"),
                    prompt,
                    temperature
                )
                
                if input_generator and not str(input_generator).startswith("# ERROR:"):
                    # Clean the input generator code
                    input_generator = clean_code_block(input_generator)
                    
                    # Create a single record
                    record = {
                        'context': context,
                        'reference_code': reference_code,
                        'input_generator': input_generator
                    }
                    
                    # Write record directly to file to avoid memory issues
                    if processed_count == 0:
                        # Write header for first record
                        with open(output_file, 'w', encoding='utf-8') as out_f:
                            out_f.write(json.dumps(record) + '\n')
                    else:
                        # Append subsequent records
                        with open(output_file, 'a', encoding='utf-8') as out_f:
                            out_f.write(json.dumps(record) + '\n')
                    
                    processed_count += 1
                    
                    if processed_count % 10 == 0:
                        print(f"\nProcessed {processed_count} items...")
                
            except Exception as e:
                print(f"Error processing line: {str(e)}")
                continue
    
    return processed_count

def jsonl_to_csv(jsonl_file, csv_file):
    """Convert JSONL to CSV, properly handling commas and newlines in the data."""
    # Read all records
    records = []
    with open(jsonl_file, 'r', encoding='utf-8') as f:
        for line in f:
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError:
                print(f"Error decoding line in {jsonl_file}")
                continue
    
    if not records:
        print(f"No valid records found in {jsonl_file}")
        return
    
    # Define field order and process fields
    fieldnames = ["context", "input_generator", "reference_code"]
    
    # Write to CSV with proper handling
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, quoting=csv.QUOTE_ALL)
        writer.writeheader()
        
        for record in records:
            processed = {}
            for field in fieldnames:
                # Get value or empty string if missing
                value = record.get(field, '')  
                
                # Clean the value
                if isinstance(value, str):
                    # Replace newlines with spaces and strip extra whitespace
                    value = value.replace('\n', ' ').strip()
                    
                processed[field] = value
            
            writer.writerow(processed)
    
    print(f"Successfully converted {jsonl_file} to {csv_file}")

def main():
    """Main function to generate input generators."""
    parser = argparse.ArgumentParser(description='Generate input generators for code problems')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Path to input JSONL file')
    parser.add_argument('--output_dir', type=str, default='generated_data',
                      help='Directory to save output files')
    parser.add_argument('--api_key', type=str, required=True,
                      help='Deepseek API key')
    parser.add_argument('--test_mode', action='store_true',
                      help='Run in test mode with only 5 samples')
    parser.add_argument('--temperature', type=float, default=0.3,
                      help='Temperature for API calls')
    parser.add_argument('--max_workers', type=int, default=5,
                      help='Number of parallel workers')
    
    args = parser.parse_args()
    
    # Configuration
    input_file = args.input_file
    output_dir = os.path.abspath(args.output_dir)
    api_key = args.api_key
    max_rows = 1 if args.test_mode else None
    temperature = args.temperature
    max_workers = args.max_workers
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running with configuration:")
    print(f"  Input file: {os.path.abspath(input_file)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Test mode: {args.test_mode}")
    print(f"  Max rows: {max_rows}")
    print(f"  Temperature: {temperature}")
    print(f"  Max workers: {max_workers}")
    
    # Generate input generators and save to file
    output_name = 'test.jsonl' if args.test_mode else 'input_generators.jsonl'
    output_file = os.path.join(output_dir, output_name)
    
    print("\nGenerating input generators...")
    processed_count = process_and_generate(
        input_file=input_file,
        output_file=output_file,
        api_key=api_key,
        max_rows=max_rows,
        temperature=temperature,
        max_workers=max_workers
    )
    
    print(f"\nPipeline completed. Generated input generators for {processed_count} problems")
    print(f"Output saved to {output_file}")
    
    # Convert JSONL to CSV for easier viewing
    csv_file = output_file.replace('.jsonl', '.csv')
    print("\nConverting to CSV for easier viewing...")
    jsonl_to_csv(output_file, csv_file)
    
    # Verify files exist and print their sizes
    if os.path.exists(output_file):
        jsonl_size = os.path.getsize(output_file) / 1024  # Convert to KB
        print(f"JSONL file successfully created! Size: {jsonl_size:.2f} KB")
    else:
        print("Warning: JSONL file was not created successfully!")
        
    if os.path.exists(csv_file):
        csv_size = os.path.getsize(csv_file) / 1024  # Convert to KB
        print(f"CSV file successfully created! Size: {csv_size:.2f} KB")
    else:
        print("Warning: CSV file was not created successfully!")

if __name__ == "__main__":
    main() 