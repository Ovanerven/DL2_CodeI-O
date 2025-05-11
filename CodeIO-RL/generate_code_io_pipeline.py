import os
import sys
import argparse
import time
import json
from datetime import datetime

# Import functions from our scripts
from generate_input_generators import process_and_generate, jsonl_to_csv as gen_to_csv
from generate_io_pairs import process_and_generate_io_pairs, jsonl_to_csv as io_to_csv

def run_pipeline(args):
    """Run the full pipeline to generate input generators and IO pairs."""
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directories
    run_dir = args.output_dir
    if args.timestamp:
        run_dir = f"{run_dir}_{timestamp}"
    
    os.makedirs(run_dir, exist_ok=True)
    
    try:
        # Determine if we're using existing input generators or generating new ones
        using_existing_input_generators = args.skip_input_gen
        
        # Setup output files
        if using_existing_input_generators:
            # When using existing generators, the input_file is already the input_gen_file
            input_gen_file = args.input_file
            input_gen_csv = input_gen_file.replace('.jsonl', '.csv')
        else:
            # When generating new ones, create paths for output files
            input_gen_file = os.path.join(run_dir, "test.jsonl" if args.test_mode else "input_generators.jsonl")
            input_gen_csv = input_gen_file.replace('.jsonl', '.csv')
        
        io_pairs_dir = os.path.join(run_dir, "io") 
        os.makedirs(io_pairs_dir, exist_ok=True)
        io_pairs_file = os.path.join(io_pairs_dir, "test_with_io.jsonl" if args.test_mode else "final_dataset.jsonl")
        io_pairs_csv = io_pairs_file.replace('.jsonl', '.csv')
        
        # Save configuration to file
        config = {
            "timestamp": timestamp,
            "input_file": os.path.abspath(args.input_file),
            "output_dir": os.path.abspath(run_dir),
            "test_mode": args.test_mode,
            "skip_input_gen": args.skip_input_gen,
            "api_key": args.api_key[:5] + "..." if args.api_key else None,
            "max_rows": 20 if args.test_mode else None,
            "num_io_pairs": args.num_io_pairs,
            "timeout": args.timeout,
            "temperature": args.temperature,
            "max_workers": args.max_workers
        }
        
        with open(os.path.join(run_dir, "config.json"), 'w') as f:
            json.dump(config, f, indent=2)
        
        # === STEP 1: Generate Input Generators (if not skipped) ===
        if not using_existing_input_generators:
            if not args.api_key:
                print("ERROR: API key is required when generating input generators.")
                return False
                
            print(f"\n{'='*80}")
            print(f"STEP: Generating Input Generators")
            print(f"{'='*80}")
            print(f"Input file: {args.input_file}")
            print(f"Output file: {input_gen_file}")
            print(f"Test mode: {args.test_mode}")
            print(f"{'-'*80}")
            
            processed_count = process_and_generate(
                input_file=args.input_file,
                output_file=input_gen_file,
                api_key=args.api_key,
                max_rows=20 if args.test_mode else None,
                temperature=args.temperature,
                max_workers=args.max_workers
            )
            
            if processed_count == 0:
                print("ERROR: No input generators were generated. Stopping pipeline.")
                return False
            
            # Convert to CSV for easier viewing
            print("\nConverting to CSV for easier viewing...")
            gen_to_csv(input_gen_file, input_gen_csv)
        else:
            print(f"\n{'='*80}")
            print(f"STEP: Using Existing Input Generators")
            print(f"{'='*80}")
            print(f"Input generators file: {input_gen_file}")
            print(f"{'-'*80}")
            
            # Verify the file exists
            if not os.path.exists(input_gen_file):
                print(f"ERROR: Input generators file not found: {input_gen_file}")
                return False
                
            # Check if the file has the expected format
            try:
                with open(input_gen_file, 'r', encoding='utf-8') as f:
                    first_line = f.readline().strip()
                    record = json.loads(first_line)
                    
                    if 'input_generator' not in record or 'reference_code' not in record:
                        print("ERROR: Input file does not appear to contain input generators and reference code.")
                        print("Make sure it's a JSONL file with 'input_generator' and 'reference_code' fields.")
                        return False
                        
                    print(f"Found valid input generators file with required fields.")
            except Exception as e:
                print(f"ERROR: Failed to validate input generators file: {str(e)}")
                return False
        
        # === STEP 2: Generate I/O Pairs ===
        print(f"\n{'='*80}")
        print(f"STEP: Generating I/O Pairs")
        print(f"{'='*80}")
        print(f"Input file: {input_gen_file}")
        print(f"Output file: {io_pairs_file}")
        print(f"Number of pairs: {args.num_io_pairs}")
        print(f"Timeout: {args.timeout} seconds")
        print(f"{'-'*80}")
        
        processed_count, success_count = process_and_generate_io_pairs(
            input_file=input_gen_file,
            output_file=io_pairs_file,
            num_pairs=args.num_io_pairs,
            timeout=args.timeout
        )
        
        if processed_count == 0:
            print("ERROR: No I/O pairs were generated.")
            return False
        
        # Convert to CSV for easier viewing
        print("\nConverting to CSV for easier viewing...")
        io_to_csv(io_pairs_file, io_pairs_csv)
        
        # === PIPELINE COMPLETE ===
        end_time = time.time()
        duration = end_time - start_time
        
        print(f"\n{'='*80}")
        print(f"PIPELINE COMPLETED SUCCESSFULLY")
        print(f"{'='*80}")
        print(f"Total runtime: {duration:.1f} seconds ({duration/60:.1f} minutes)")
        
        if not using_existing_input_generators:
            print(f"Input generators: {input_gen_file}")
            
        print(f"I/O pairs: {io_pairs_file}")
        print(f"Success rate: {success_count}/{processed_count} ({success_count/processed_count*100:.1f}%)")
        print(f"Configuration saved to: {os.path.join(run_dir, 'config.json')}")
        print(f"{'='*80}")
        
        return True
        
    except Exception as e:
        print(f"ERROR: Pipeline failed with exception: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main entry point for the pipeline."""
    parser = argparse.ArgumentParser(
        description='Run the full Code-IO generation pipeline: input generators and I/O pairs',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Required parameters
    parser.add_argument('--input_file', type=str, required=True,
                      help='Path to input JSONL file with code problems (or input generators if --skip_input_gen is used)')
    
    # Options to skip steps
    parser.add_argument('--skip_input_gen', action='store_true',
                      help='Skip input generator creation and use existing ones (input_file should point to file with generators)')
    
    # API key (required only if not skipping input generator step)
    parser.add_argument('--api_key', type=str,
                      help='Deepseek API key for generating input generators (required unless --skip_input_gen is used)')
    
    # Directory configuration
    parser.add_argument('--output_dir', type=str, default='generated_data/run',
                      help='Directory to save all output files')
    parser.add_argument('--timestamp', action='store_true',
                      help='Add timestamp to output directory to prevent overwriting')
    
    # Run configuration
    parser.add_argument('--test_mode', action='store_true',
                      help='Run in test mode with only a few samples')
    
    # Input generator configuration
    parser.add_argument('--temperature', type=float, default=0.3,
                      help='Temperature for API calls')
    parser.add_argument('--max_workers', type=int, default=5,
                      help='Number of parallel workers for API calls')
    
    # IO pair configuration
    parser.add_argument('--num_io_pairs', type=int, default=10,
                      help='Number of I/O pairs to generate per problem')
    parser.add_argument('--timeout', type=int, default=60,
                      help='Timeout in seconds for each problem')
    
    args = parser.parse_args()
    
    # Validate arguments
    if not args.skip_input_gen and not args.api_key:
        parser.error("--api_key is required unless --skip_input_gen is specified")
    
    print(f"\n{'='*80}")
    print(f"CODE-IO GENERATION PIPELINE")
    print(f"{'='*80}")
    print(f"Input file: {os.path.abspath(args.input_file)}")
    print(f"Output directory: {os.path.abspath(args.output_dir)}")
    print(f"Skip input generator creation: {args.skip_input_gen}")
    print(f"Test mode: {args.test_mode}")
    
    if not args.skip_input_gen:
        print(f"API key: {args.api_key[:5]}...")
        print(f"Temperature: {args.temperature}")
        print(f"Max workers: {args.max_workers}")
        
    print(f"Number of I/O pairs: {args.num_io_pairs}")
    print(f"Timeout: {args.timeout} seconds")
    print(f"{'='*80}\n")
    
    # Run the pipeline
    success = run_pipeline(args)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()