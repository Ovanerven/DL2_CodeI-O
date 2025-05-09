import os
import sys
import argparse
import time
import json
from datetime import datetime

# Import functions from our scripts
from generate_input_generators import process_and_generate, jsonl_to_csv as gen_to_csv
from generate_io_pairs import process_and_generate_io_pairs, jsonl_to_csv as io_to_csv

def run_diagnostic_checks(io_pairs_file, output_dir, show_solutions=False):
    """Run diagnostic checks on the generated IO pairs to identify patterns in failures."""
    print(f"\n{'='*80}")
    print(f"RUNNING DIAGNOSTIC CHECKS ON FAILED IO GENERATIONS")
    print(f"{'='*80}")
    
    try:
        # Read the generated IO pairs file
        with open(io_pairs_file, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f]
            
        # Count records with and without IO pairs
        success_count = sum(1 for r in records if r.get('io_pairs') and len(r.get('io_pairs', [])) > 0)
        failed_count = len(records) - success_count
        
        if failed_count == 0:
            print("No failures detected! All records have IO pairs.")
            return True
            
        # Create diagnostic report
        diagnostic_file = os.path.join(output_dir, "diagnostic_report.txt")
        with open(diagnostic_file, 'w', encoding='utf-8') as f:
            f.write(f"IO GENERATION DIAGNOSTIC REPORT\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Total records: {len(records)}\n")
            f.write(f"Successful: {success_count} ({success_count/len(records)*100:.1f}%)\n")
            f.write(f"Failed: {failed_count} ({failed_count/len(records)*100:.1f}%)\n\n")
            
            # Analyze failed records
            f.write(f"ANALYSIS OF FAILED RECORDS\n")
            f.write(f"{'='*80}\n\n")
            
            # Error type counters for summary
            error_types = {}
            consolidated_errors = {}
            
            for idx, record in enumerate(records):
                if not record.get('io_pairs') or len(record.get('io_pairs', [])) == 0:
                    f.write(f"FAILED RECORD #{idx}\n")
                    f.write(f"{'-'*80}\n")
                    
                    # Get the actual error messages from the record
                    error_logs = record.get('error_log', [])
                    
                    # Report the captured errors from execution - PRIORITIZE THESE
                    if error_logs:
                        f.write("ACTUAL ERRORS ENCOUNTERED:\n")
                        for error in error_logs[:5]:  # Show the first 5 errors
                            # Extract just the first line if it's a multiline error
                            error_summary = error.split('\n')[0] if '\n' in error else error
                            f.write(f"- {error_summary}\n")
                            
                            # Add to error type counter
                            error_type = "Unknown error"
                            
                            # Extract error type
                            if "TypeError:" in error:
                                error_type = "TypeError"
                            elif "NameError:" in error:
                                error_type = "NameError"
                            elif "AttributeError:" in error:
                                error_type = "AttributeError"
                            elif "SyntaxError:" in error:
                                error_type = "SyntaxError"
                            elif "ValueError:" in error:
                                error_type = "ValueError"
                            elif "IndexError:" in error:
                                error_type = "IndexError"
                            elif "KeyError:" in error:
                                error_type = "KeyError"
                            elif "ImportError:" in error:
                                error_type = "ImportError"
                            elif "ZeroDivisionError:" in error:
                                error_type = "ZeroDivisionError"
                            elif "Timeout expired" in error:
                                error_type = "Timeout"
                            elif "complex" in error:
                                error_type = "ComplexError"
                            
                            # Increment counter and collect examples
                            if error_type in error_types:
                                error_types[error_type] += 1
                                # Only store up to 3 examples per error type
                                if error_type in consolidated_errors and len(consolidated_errors[error_type]) < 3:
                                    consolidated_errors[error_type].append((idx, error_summary))
                            else:
                                error_types[error_type] = 1
                                consolidated_errors[error_type] = [(idx, error_summary)]
                        
                        # Write full error details
                        f.write("\nDETAILED ERROR LOGS:\n")
                        for i, error in enumerate(error_logs[:3]):  # Show the first 3 full error logs
                            f.write(f"Error {i+1}:\n{error}\n\n")
                    else:
                        f.write("NO SPECIFIC ERROR MESSAGES CAPTURED\n")
                    
                    # Check common failure patterns only if no specific errors found
                    if not error_logs:
                        input_gen = record.get('input_generator', '')
                        reference_code = record.get('reference_code', '')
                        
                        # Run checks
                        issues = []
                        
                        # Check 1: Missing input_generator function
                        if 'def input_generator' not in input_gen:
                            issues.append("Missing input_generator function definition")
                        
                        # Check 2: Missing main_solution function
                        if 'def main_solution' not in reference_code:
                            issues.append("Missing main_solution function definition")
                        
                        # Check 3: Random functions that might not work reliably
                        random_check = False
                        seed_check = False
                        
                        if 'random.' in input_gen or 'np.random' in input_gen or 'numpy.random' in input_gen:
                            random_check = True
                            
                        if 'random.seed' in input_gen or 'np.random.seed' in input_gen or 'numpy.random.seed' in input_gen:
                            seed_check = True
                            
                        if random_check and not seed_check:
                            issues.append("Uses random functions without setting a seed")
                        
                        # Report the issues only if we found no concrete errors
                        if issues:
                            f.write("\nPOTENTIAL ISSUES IDENTIFIED (speculative):\n")
                            for issue in issues:
                                f.write(f"- {issue}\n")
                    
                    # Show truncated code
                    f.write("\nTRUNCATED INPUT GENERATOR:\n")
                    f.write(f"{record.get('input_generator', '')[:300]}...\n\n")
                    
                    f.write("\nTRUNCATED REFERENCE CODE:\n")
                    f.write(f"{record.get('reference_code', '')[:300]}...\n\n")
                    
                    f.write(f"{'='*80}\n\n")
            
            # Summary of error types - PRIORITIZE THIS SECTION
            if error_types:
                f.write("\nERROR TYPE SUMMARY\n")
                f.write(f"{'-'*80}\n")
                f.write("The following error types were encountered:\n\n")
                
                for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True):
                    f.write(f"{error_type}: {count} occurrences ({count/failed_count*100:.1f}%)\n")
                    
                    # Show examples of each error type
                    if error_type in consolidated_errors:
                        f.write("Examples:\n")
                        for ex_idx, (record_idx, error_text) in enumerate(consolidated_errors[error_type]):
                            f.write(f"  {ex_idx+1}. Record #{record_idx}: {error_text}\n")
                        f.write("\n")
                f.write("\n")
            
            # Solutions section - move below the error summary
            if error_types:
                f.write("\nSUGGESTED SOLUTIONS BY ERROR TYPE\n")
                f.write(f"{'-'*80}\n")
                
                if "TypeError" in error_types:
                    f.write("For TypeError issues:\n")
                    f.write("- Check if the input generator provides the exact parameter types expected by main_solution\n")
                    f.write("- Verify that complex numbers or special objects are properly handled\n\n")
                
                if "NameError" in error_types:
                    f.write("For NameError issues:\n")
                    f.write("- Check for undefined variables in input_generator or main_solution\n")
                    f.write("- Make sure all required imports are present\n\n")
                
                if "SyntaxError" in error_types:
                    f.write("For SyntaxError issues:\n")
                    f.write("- Look for markdown code markers (```) or other non-Python syntax\n")
                    f.write("- Check for missing parentheses, brackets, or indentation issues\n\n")
                
                if "ComplexError" in error_types:
                    f.write("For Complex Number errors:\n")
                    f.write("- Our updated code should handle these automatically now\n\n")
                    
                if "Timeout" in error_types:
                    f.write("For Timeout issues:\n")
                    f.write("- The code is taking too long to execute (>60 seconds)\n")
                    f.write("- Check for infinite loops or very inefficient algorithms\n")
                    f.write("- Consider reducing the complexity of generated inputs\n\n")
            
            # Only include common solutions if there are no specific error types identified
            if not error_types:
                f.write("\nCOMMON SOLUTIONS\n")
                f.write(f"{'-'*80}\n")
                f.write("Here are common solutions to try:\n\n")
                
                # 1. Random seeds
                f.write("1. Random seed issues: Add these lines to the beginning of your input_generator:\n")
                f.write("   ```python\n")
                f.write("   import random\n")
                f.write("   import numpy as np\n\n")
                f.write("   def input_generator():\n")
                f.write("       # Set a fixed seed for consistent results\n")
                f.write("       random.seed(42)\n")
                f.write("       if 'np' in globals() or 'numpy' in globals():\n")
                f.write("           np.random.seed(42)\n")
                f.write("   ```\n\n")
                
                # 2. Dictionary returns
                f.write("2. Dictionary return issues: Make sure your input_generator returns a dictionary:\n")
                f.write("   ```python\n")
                f.write("   return {\"param1\": value1, \"param2\": value2}\n")
                f.write("   ```\n\n")
            
            # New section focused on rerunning with fixed seeds
            f.write("\nHOW TO RERUN FAILED CASES\n")
            f.write(f"{'-'*80}\n")
            f.write("To retry just the failed cases with fixed seeds, run:\n\n")
            f.write("python generate_code_io_pipeline.py --input_file failed_cases.jsonl --skip_input_gen --output_dir retry_output\n\n")
            f.write("Where failed_cases.jsonl contains only the records that failed previously.\n")
            
        print(f"Diagnostic report generated: {diagnostic_file}")
        
        # Print a summary of the most common errors
        print("\nMOST COMMON ERROR TYPES:")
        print(f"{'-'*80}")
        
        # Show the most common error types
        if error_types:
            for error_type, count in sorted(error_types.items(), key=lambda x: x[1], reverse=True)[:5]:
                print(f"- {error_type}: {count} occurrences ({count/failed_count*100:.1f}%)")
                
                # Show an example of this error type
                if error_type in consolidated_errors and consolidated_errors[error_type]:
                    print(f"  Example: {consolidated_errors[error_type][0][1]}")
            print()
        else:
            print("No specific error types detected. Check the diagnostic report for speculative issues.")
        
        print(f"\nSee detailed analysis in: {diagnostic_file}")
        
        return True
    except Exception as e:
        print(f"Error running diagnostics: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

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
        
        # === STEP 3: Run Diagnostic Checks if failures occurred ===
        if success_count < processed_count and args.diagnostics:
            run_diagnostic_checks(io_pairs_file, io_pairs_dir, show_solutions=args.show_solutions)
        
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
        
        # Reminder about diagnostics if failures occurred and diagnostics were not run
        if success_count < processed_count and not args.diagnostics:
            print(f"\nNOTE: Some IO generations failed. Run with '--diagnostics' for detailed error analysis.")
            
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
    parser.add_argument('--diagnostics', action='store_true',
                      help='Run diagnostic checks on failed IO generations')
    parser.add_argument('--show_solutions', action='store_true',
                      help='Show suggested solutions for common errors in the diagnostic report')
    
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
    print(f"Run diagnostics: {args.diagnostics}")
    print(f"Show solutions: {args.show_solutions}")
    print(f"{'='*80}\n")
    
    # Run the pipeline
    success = run_pipeline(args)
    
    # Exit with appropriate status code
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main() 