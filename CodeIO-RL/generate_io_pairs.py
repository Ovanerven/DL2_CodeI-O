import json
import os
import sys
import subprocess
import signal
from tqdm import tqdm
import pandas as pd
from pympler import asizeof
import argparse
import csv
import tempfile
import random
import numpy as np
import traceback
import math

def generate_io_pairs(input_generator_code, main_solution_code, num_pairs=10, timeout=60):
    """Generate I/O pairs using the input generator and main solution with robust subprocess handling."""
    # Create a temporary file with both functions
    temp_code = f"""
import json
from pympler import asizeof

def strict_check_size(obj):
    # Check if object size is less than 1024 bytes
    if asizeof.asizeof(obj) >= 1024: 
        return False

    # Check for dict type
    if isinstance(obj, dict):
        if len(obj) >= 20:  # Check dict has fewer than 20 key-value pairs
            return False
        # Recursively check keys and values
        for k, v in obj.items():
            if not strict_check_size(k) or not strict_check_size(v):
                return False

    # Check for list, tuple, or set
    elif isinstance(obj, (list, tuple, set)):
        if len(obj) >= 20:  # Check if the length is less than 20
            return False
        # Recursively check each element
        for item in obj:
            if not strict_check_size(item):
                return False

    # Check for string
    elif isinstance(obj, str):
        if len(obj) >= 100:  # Check if string length is less than 100 characters
            return False

    # Other objects - check size in bytes
    else:
        if asizeof.asizeof(obj) >= 128:  # Check if object size is less than 128 bytes
            return False

    # If all checks are passed, return True
    return True

# Define the input generator and main solution functions
{input_generator_code}

{main_solution_code}

# Generate I/O pairs
diff_inputs = []
corr_outputs = []
for i in range(1000):
    try:
        cand_input = input_generator()
        if cand_input not in diff_inputs and strict_check_size(cand_input):
            cand_output = main_solution(**cand_input)
            if strict_check_size(cand_output) and cand_output is not None:
                diff_inputs.append(cand_input)
                corr_outputs.append(cand_output)
        if len(diff_inputs) >= {num_pairs}:
            break
    except Exception as e:
        continue
        
assert len(diff_inputs) == len(corr_outputs)

iolist = [{{"input": diff_inputs[i], "output": corr_outputs[i]}} for i in range(len(diff_inputs))]

jsoniolist = json.dumps(iolist)
    
print("[JSON IOS START]" + jsoniolist + "[JSON IOS END]")
"""
    
    # Execute the code and capture output with better handling
    try:
        # Create a temporary directory to run the code
        temp_dir = tempfile.mkdtemp(prefix="io_gen_")
        
        # Windows doesn't support killpg, so we need a different approach
        is_windows = os.name == 'nt'
        
        if is_windows:
            # On Windows, use regular subprocess without process groups
            proc = subprocess.Popen(
                [sys.executable, '-c', temp_code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=temp_dir
            )
        else:
            # On Unix, use process groups for proper cleanup
            proc = subprocess.Popen(
                [sys.executable, '-c', temp_code],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                cwd=temp_dir,
                start_new_session=True
            )
            
        try:
            stdout, stderr = proc.communicate(timeout=timeout)
            
            # Look for special markers in the output
            start_marker = "[JSON IOS START]"
            end_marker = "[JSON IOS END]"
            
            if start_marker in stdout and end_marker in stdout:
                start_index = stdout.index(start_marker) + len(start_marker)
                end_index = stdout.index(end_marker)
                json_str = stdout[start_index:end_index].strip()
                
                try:
                    # Parse the result
                    result = json.loads(json_str)
                    return {"io_pairs": result, "error_log": []}
                except json.JSONDecodeError:
                    return {"io_pairs": [], "error_log": [f"JSON decode error: {json_str[:100]}..."]}
            else:
                # Capture stderr for error diagnosis
                error_messages = []
                if stderr:
                    error_messages.append(f"STDERR: {stderr}")
                
                if stdout:
                    error_messages.append(f"STDOUT: {stdout}")
                
                return {"io_pairs": [], "error_log": error_messages}
                
        except subprocess.TimeoutExpired:
            # Handle timeout
            if is_windows:
                proc.kill()
            else:
                os.killpg(os.getpgid(proc.pid), signal.SIGTERM)
            
            return {"io_pairs": [], "error_log": ["Timeout expired after {timeout} seconds"]}
            
        finally:
            # Ensure the subprocess is terminated
            try:
                proc.kill()
                proc.wait()
            except:
                pass
            
            # Clean up the temporary directory
            try:
                import shutil
                shutil.rmtree(temp_dir)
            except:
                pass
            
    except Exception as e:
        return {"io_pairs": [], "error_log": [f"Error in IO generation subprocess: {str(e)}"]}
    
    return {"io_pairs": [], "error_log": ["Unknown error occurred"]}

def process_and_generate_io_pairs(input_file, output_file, num_pairs=10, timeout=60):
    """Process input file line by line and generate I/O pairs."""
    processed_count = 0
    success_count = 0
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Input file '{input_file}' not found!")
        return 0, 0
    
    # Create output directory if it doesn't exist
    output_dir = os.path.dirname(output_file)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
    
    # Read all records first to ensure file format is correct
    records = []
    try:
        with open(input_file, 'r', encoding='utf-8') as in_f:
            for line in in_f:
                try:
                    record = json.loads(line)
                    records.append(record)
                except json.JSONDecodeError:
                    print(f"Warning: Invalid JSON in line: {line[:100]}...")
    except Exception as e:
        print(f"Error reading input file: {str(e)}")
        return 0, 0
    
    print(f"Found {len(records)} records in input file")
    
    # Create temp directory for temporary files
    temp_dir = os.path.join(os.getcwd(), "temp_scripts")
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Process each record
        with open(output_file, 'w', encoding='utf-8') as out_f:
            for idx, record in enumerate(tqdm(records)):
                try:
                    # Check if required fields exist
                    if 'input_generator' not in record or 'reference_code' not in record:
                        print(f"Skipping record {idx}: missing required fields")
                        record['io_pairs'] = []
                        record['error_log'] = ["Missing required fields: input_generator or reference_code"]
                        out_f.write(json.dumps(record) + '\n')
                        processed_count += 1
                        continue
                    
                    # Skip records with error markers
                    if pd.isna(record['input_generator']) or str(record['input_generator']).startswith("# ERROR:"):
                        print(f"Skipping record {idx}: input generator has errors")
                        record['io_pairs'] = []
                        record['error_log'] = ["Input generator contains error markers"]
                        out_f.write(json.dumps(record) + '\n')
                        processed_count += 1
                        continue
                    
                    # Generate I/O pairs
                    result = generate_io_pairs(
                        record['input_generator'],
                        record['reference_code'],
                        num_pairs=num_pairs,
                        timeout=timeout
                    )
                    
                    # Extract IO pairs and error log from result
                    io_pairs = result.get("io_pairs", [])
                    error_log = result.get("error_log", [])
                    
                    if io_pairs and len(io_pairs) > 0:
                        success_count += 1
                        print(f"Record {idx}: Successfully generated {len(io_pairs)} I/O pairs")
                    else:
                        print(f"Record {idx}: Failed to generate I/O pairs.")
                        if error_log:
                            # For very long error logs, truncate to just the first error
                            if len(error_log) > 1:
                                print(f"First error of {len(error_log)}: {error_log[0][:300]}...")
                            else:
                                print(f"Errors: {error_log[0][:300] if error_log else 'Unknown error'}")
                    
                    # Add io_pairs and error_log to record
                    record['io_pairs'] = io_pairs
                    record['error_log'] = error_log
                    
                    # Write record to output file
                    out_f.write(json.dumps(record) + '\n')
                    
                    processed_count += 1
                    
                    if processed_count % 10 == 0:
                        print(f"\nProcessed {processed_count} items, {success_count} successful...")
                        
                except Exception as e:
                    error_message = f"Error processing record {idx}: {str(e)}"
                    print(error_message)
                    # Add error to record and write it
                    record['io_pairs'] = []
                    record['error_log'] = [error_message]
                    out_f.write(json.dumps(record) + '\n')
                    processed_count += 1
                    continue
    finally:
        # Clean up temp directory
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except:
            pass
    
    return processed_count, success_count

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
    fieldnames = ["context", "reference_code", "input_generator", "io_pairs"]
    
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
    """Main function to generate I/O pairs."""
    parser = argparse.ArgumentParser(description='Generate I/O pairs from input generators')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Path to input JSONL file with generators')
    parser.add_argument('--output_file', type=str, required=True,
                      help='Path to output JSONL file for I/O pairs')
    parser.add_argument('--num_pairs', type=int, default=10,
                      help='Number of I/O pairs to generate per problem (default: 10)')
    parser.add_argument('--timeout', type=int, default=60,
                      help='Timeout in seconds for each problem (default: 60)')
    
    args = parser.parse_args()
    
    # Configuration
    input_file = args.input_file
    output_file = args.output_file
    num_io_pairs = args.num_pairs
    timeout = args.timeout
    
    print(f"Running with configuration:")
    print(f"  Input file: {os.path.abspath(input_file)}")
    print(f"  Output file: {os.path.abspath(output_file)}")
    print(f"  Number of I/O pairs per problem: {num_io_pairs}")
    print(f"  Timeout per problem: {timeout} seconds")
    
    # Verify input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {os.path.abspath(input_file)}")
        return
    
    # Generate I/O pairs
    print("\nGenerating I/O pairs...")
    processed_count, success_count = process_and_generate_io_pairs(
        input_file=input_file,
        output_file=output_file,
        num_pairs=num_io_pairs,
        timeout=timeout
    )
    
    print(f"\nPipeline completed. Generated I/O pairs for {processed_count} problems")
    print(f"Output saved to {output_file}")
    
    # Convert JSONL to CSV for easier viewing
    csv_file = output_file.replace('.jsonl', '.csv')
    print("\nConverting to CSV for easier viewing...")
    jsonl_to_csv(output_file, csv_file)
    
    # Print success rate
    success_rate = success_count / processed_count * 100 if processed_count > 0 else 0
    print(f"Success rate: {success_rate:.1f}% ({success_count} out of {processed_count} problems)")
    
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