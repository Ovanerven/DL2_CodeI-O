import json
import os
import sys
import subprocess
from tqdm import tqdm
import pandas as pd
from pympler import asizeof
import argparse

def strict_check_size(obj):
    """Check if object size is reasonable (less than 1KB)."""
    if asizeof.asizeof(obj) >= 1024: 
        return False

    if isinstance(obj, dict):
        if len(obj) >= 20:  # Check dict has fewer than 20 key-value pairs
            return False
        for k, v in obj.items():
            if not strict_check_size(k) or not strict_check_size(v):
                return False

    elif isinstance(obj, (list, tuple, set)):
        if len(obj) >= 20:  # Check if the length is less than 20
            return False
        for item in obj:
            if not strict_check_size(item):
                return False

    elif isinstance(obj, str):
        if len(obj) >= 100:  # Check if string length is less than 100 characters
            return False

    else:
        if asizeof.asizeof(obj) >= 128:  # Check if object size is less than 128 bytes
            return False

    return True

def generate_io_pairs(input_generator_code, main_solution_code, num_pairs=10):
    """Generate I/O pairs using the input generator and main solution."""
    # Create a temporary file with both functions
    temp_code = f"""
{input_generator_code}

{main_solution_code}

# Generate I/O pairs
io_pairs = []
for _ in range({num_pairs}):
    try:
        input_args = input_generator()
        if not strict_check_size(input_args):
            continue
        output = main_solution(**input_args)
        if not strict_check_size(output) or output is None:
            continue
        io_pairs.append({{"input": input_args, "output": output}})
    except Exception as e:
        continue

print(json.dumps(io_pairs))
"""
    
    # Execute the code and capture output
    try:
        result = subprocess.run(
            [sys.executable, '-c', temp_code],
            capture_output=True,
            text=True,
            timeout=30
        )
        if result.returncode == 0:
            return json.loads(result.stdout)
    except Exception as e:
        print(f"Error generating I/O pairs: {str(e)}")
    
    return []

def process_and_generate_io_pairs(input_file, output_file, num_pairs=10, timeout=30):
    """Process input file line by line and generate I/O pairs."""
    # Create output file and write header
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('context,reference_code,input_generator,io_pairs\n')
    
    # Process input file line by line
    processed_count = 0
    success_count = 0
    
    # Use csv module for proper handling of quoted fields
    import csv
    with open(input_file, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader):
            try:
                if pd.isna(row['input_generator']) or str(row['input_generator']).startswith("# ERROR:"):
                    # Write row with empty io_pairs
                    row['io_pairs'] = '[]'
                else:
                    # Generate I/O pairs
                    pairs = generate_io_pairs(
                        row['input_generator'],
                        row['reference_code'],
                        num_pairs=num_pairs
                    )
                    
                    if pairs:
                        success_count += 1
                    
                    # Add I/O pairs to row
                    row['io_pairs'] = json.dumps(pairs)
                
                # Write row to output file
                with open(output_file, 'a', encoding='utf-8', newline='') as out_f:
                    writer = csv.DictWriter(out_f, fieldnames=['context', 'reference_code', 'input_generator', 'io_pairs'])
                    writer.writerow(row)
                
                processed_count += 1
                
                if processed_count % 10 == 0:
                    print(f"\nProcessed {processed_count} items, {success_count} successful...")
                    
            except Exception as e:
                print(f"Error processing row: {str(e)}")
                continue
    
    return processed_count, success_count

def main():
    """Main function to generate I/O pairs."""
    parser = argparse.ArgumentParser(description='Generate I/O pairs from input generators')
    parser.add_argument('--input_file', type=str, required=True,
                      help='Path to input CSV file with generators')
    parser.add_argument('--output_dir', type=str, default='generated_data',
                      help='Directory to save output files')
    parser.add_argument('--test_mode', action='store_true',
                      help='Run in test mode with only 5 samples')
    parser.add_argument('--num_pairs', type=int, default=5,
                      help='Number of I/O pairs to generate per problem')
    parser.add_argument('--timeout', type=int, default=30,
                      help='Timeout in seconds for each problem')
    
    args = parser.parse_args()
    
    # Configuration
    input_file = args.input_file
    output_dir = os.path.abspath(args.output_dir)
    num_io_pairs = args.num_pairs
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Running with configuration:")
    print(f"  Input file: {os.path.abspath(input_file)}")
    print(f"  Output directory: {output_dir}")
    print(f"  Test mode: {args.test_mode}")
    print(f"  Number of I/O pairs per problem: {num_io_pairs}")
    print(f"  Timeout per problem: {args.timeout} seconds")
    
    # Verify input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file not found at {os.path.abspath(input_file)}")
        return
    
    # Generate I/O pairs
    output_name = 'test.csv' if args.test_mode else 'final_dataset.csv'
    output_file = os.path.join(output_dir, output_name)
    
    print("\nGenerating I/O pairs...")
    processed_count, success_count = process_and_generate_io_pairs(
        input_file=input_file,
        output_file=output_file,
        num_pairs=num_io_pairs,
        timeout=args.timeout
    )
    
    print(f"\nPipeline completed. Generated I/O pairs for {processed_count} problems")
    print(f"Output saved to {output_file}")
    
    # Print success rate
    success_rate = success_count / processed_count * 100 if processed_count > 0 else 0
    print(f"Success rate: {success_rate:.1f}% ({success_count} out of {processed_count} problems)")
    
    # Verify file exists and print its size
    if os.path.exists(output_file):
        file_size = os.path.getsize(output_file) / 1024  # Convert to KB
        print(f"File successfully created! Size: {file_size:.2f} KB")
    else:
        print("Warning: Output file was not created successfully!")

if __name__ == "__main__":
    main() 