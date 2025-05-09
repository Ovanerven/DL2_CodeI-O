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

def generate_io_pairs(input_generator_code, main_solution_code, num_pairs=10, timeout=60):
    """Generate I/O pairs using the input generator and main solution with robust subprocess handling."""
    # Create a temporary file with both functions
    temp_code = f"""
import json
import random
import numpy as np
import sys
import traceback
from pympler import asizeof

# Helper function to make complex numbers JSON serializable
def make_json_serializable(obj):
    if isinstance(obj, complex):
        return {{"real": obj.real, "imag": obj.imag, "__complex__": True}}
    elif isinstance(obj, (list, tuple)):
        return [make_json_serializable(item) for item in obj]
    elif isinstance(obj, dict):
        return {{k: make_json_serializable(v) for k, v in obj.items()}}
    elif isinstance(obj, np.ndarray):
        # Special handling for numpy arrays and matrices
        try:
            return make_json_serializable(obj.tolist())
        except:
            # If we can't convert to list, try as a string representation
            return str(obj)
    elif hasattr(obj, 'tolist'):
        # Handle any numpy-like objects with tolist method
        try:
            return make_json_serializable(obj.tolist())
        except:
            return str(obj)
    elif isinstance(obj, (int, float, str, bool, type(None))):
        return obj
    else:
        try:
            # Try to convert to a basic type
            return float(obj)
        except:
            return str(obj)

# Custom JSON encoder to handle complex numbers and numpy types
class ComplexEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, complex):
            return {{"real": obj.real, "imag": obj.imag, "__complex__": True}}
        if isinstance(obj, np.ndarray):
            try:
                return obj.tolist()
            except:
                return str(obj)
        if hasattr(obj, 'tolist'):
            try:
                return obj.tolist()
            except:
                return str(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        return json.JSONEncoder.default(self, obj)

def strict_check_size(obj):
    # Check if object size is less than 2048 bytes (increased from 1024)
    if asizeof.asizeof(obj) >= 2048: 
        return False

    # Check for dict type
    if isinstance(obj, dict):
        if len(obj) >= 50:  # Check dict has fewer than 50 key-value pairs
            return False
        # Recursively check keys and values
        for k, v in obj.items():
            if not strict_check_size(k) or not strict_check_size(v):
                return False

    # Check for list, tuple, or set
    elif isinstance(obj, (list, tuple, set)):
        if len(obj) >= 100:  # Check if the length is less than 100 (increased from 20)
            return False
        # Recursively check each element
        for item in obj:
            if not strict_check_size(item):
                return False

    # Check for string
    elif isinstance(obj, str):
        if len(obj) >= 200:  # Check if string length is less than 200 characters
            return False

    # Other objects - check size in bytes
    else:
        if asizeof.asizeof(obj) >= 256:  # Check if object size is less than 256 bytes
            return False

    # If all checks are passed, return True
    return True

# Keep track of errors for diagnosis
error_log = []

# Define the input generator and main solution functions
{input_generator_code}

{main_solution_code}

# Generate I/O pairs using systematic seeds
diff_inputs = []
corr_outputs = []
base_seed = 42  # Starting seed

# Try to generate with different seeds
for i in range(1000):
    try:
        # Set a deterministic seed based on the iteration
        current_seed = base_seed + i
        
        # Set the seed for both random and numpy if they're used
        if 'random' in globals():
            random.seed(current_seed)
        if 'np' in globals() or 'numpy' in globals():
            np.random.seed(current_seed)
            
        # Generate input with the current seed
        try:
            cand_input = input_generator()
        except Exception as e:
            error_msg = f"ERROR IN INPUT_GENERATOR (seed={{current_seed}}): {{str(e)}}\\n{{traceback.format_exc()}}"
            print(error_msg)
            error_log.append(error_msg)
            continue
        
        # Only accept if it meets our criteria and is unique
        if cand_input not in diff_inputs and strict_check_size(cand_input):
            # Try to generate output
            try:
                cand_output = main_solution(**cand_input)
                
                # Make the output JSON-serializable (handle complex numbers)
                cand_output = make_json_serializable(cand_output)
                
            except Exception as e:
                error_msg = f"ERROR IN MAIN_SOLUTION (seed={{current_seed}}): {{str(e)}}\\n{{traceback.format_exc()}}"
                print(error_msg)
                error_log.append(error_msg)
                continue
            
            # Accept if output meets criteria
            if strict_check_size(cand_output) and cand_output is not None:
                diff_inputs.append(cand_input)
                corr_outputs.append(cand_output)
                print(f"Generated example {{len(diff_inputs)}}/{{{num_pairs}}} with seed {{current_seed}}")
                
        # Stop when we have enough examples
        if len(diff_inputs) >= {num_pairs}:
            break
    except Exception as e:
        error_msg = f"ERROR IN ITERATION {{i}} (seed={{current_seed}}): {{str(e)}}\\n{{traceback.format_exc()}}"
        print(error_msg)
        error_log.append(error_msg)
        continue
        
assert len(diff_inputs) == len(corr_outputs)

# Create result object with both IO pairs and error logs
result = {{
    "io_pairs": [{{"input": diff_inputs[i], "output": corr_outputs[i]}} for i in range(len(diff_inputs))],
    "error_log": error_log
}}

# Use the custom encoder to handle complex numbers
jsoniolist = json.dumps(result, cls=ComplexEncoder)
    
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
                    # Parse the result with IO pairs and error log
                    result = json.loads(json_str)
                    
                    # If running within our diagnostic pipeline, include error logs
                    if isinstance(result, dict) and "io_pairs" in result:
                        io_pairs = result["io_pairs"]
                        error_log = result.get("error_log", [])
                        return {"io_pairs": io_pairs, "error_log": error_log}
                    else:
                        # For backward compatibility
                        return {"io_pairs": result, "error_log": []}
                        
                except json.JSONDecodeError:
                    print(f"Error parsing JSON: {json_str[:100]}...")
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
            
            print(f"Timeout expired for IO generation")
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
        print(f"Error in IO generation: {str(e)}")
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
                    print(f"Error processing record {idx}: {str(e)}")
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