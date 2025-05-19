import json
import random
import os
import argparse
import pandas as pd
from typing import List, Dict, Any

output_pred_template = """You are given a question that requires some input and output variables as follows:

<<<<query>>>>

The input and output requirements are as follows:

<<<<io_req>>>>

Given the following input:

<<<<input>>>>

Can you predict the output without writing any code?

First, think about how to solve this problem. Then, provide your final answer in the following format:

<answer>{"output": <your output>}</answer>

Your <your output> should strictly match the output requirement as specified."""

# IMPORTANT: After reasoning, provide ONLY the JSON answer block with your final answer. Do NOT include any other text after the JSON block.

input_pred_template = """You are given a question that requires some input and output variables as follows:

<<<<query>>>>

The input and output requirements are as follows:

<<<<io_req>>>>

Given the following output:

<<<<output>>>>

Can you predict a feasible input without writing any code?

First, think about how to solve this problem. Then, provide your final answer in the following format:

<answer>{"input": <your input>}</answer>

Your <your input> should be a dictionary, even if the there is only one input variable, with keys strictly match the input variables' names as specified."""

#IMPORTANT: After reasoning, provide ONLY the JSON answer block with your final answer. Do NOT include any other text after the JSON block.

inductive_template = """You are given a question that requires some input and output variables as follows:

<<<<query>>>>

The input and output requirements are as follows:

<<<<io_req>>>>

Given the following input and output pairs:

<<<<examples>>>>

Now, can you predict the output for the following input?

<<<<input>>>>

First, think about how to solve this problem. Then, provide your final answer in the following format:

<answer>{"output": <your output>}</answer>

Your <your output> should strictly match the output requirement as specified."""

# IMPORTANT: After reasoning, provide ONLY the JSON answer block with your final answer. Do NOT include any other text after the JSON block.

refcode_template = """Tip: Here is a reference code snippet for this question. You can refer to this code to guide your reasoning but not copy spans of code directly.

<<<<refcode>>>>"""

# Extract I/O requirements from context
def extract_io_requirements(context: str) -> str:
    """Extract the input and output requirements section from the context."""
    io_req = ""
    if "Input:" in context and "Output:" in context:
        start_idx = context.find("Input:")
        io_req = context[start_idx:]
    return io_req

def add_task_type(records: List[Dict[str, Any]], task_types: List[str]) -> List[Dict[str, Any]]:
    """Add task type to each record in a round-robin fashion using only the specified task types."""
    if not task_types:
        task_types = ['deductive', 'abductive', 'inductive']
    
    for i, rec in enumerate(records):
        rec['task_type'] = task_types[i % len(task_types)]
    return records

def format_codeio_prompt(record: Dict[str, Any], task_type: str) -> str:
    """Format the prompt based on the task type following the consistent approach."""
    problem_statement = record['context']
    reference_code = record['reference_code']
    io_req = extract_io_requirements(problem_statement)
    all_io_pairs = record.get('io_pairs', [])
    
    # Always use the first IO pair as the test case
    if not all_io_pairs:
        raise ValueError("Record has no IO pairs")
    
    io_pair = all_io_pairs[0]
    input_data = io_pair['input']
    output_data = io_pair['output']
    
    # For inductive tasks, use additional pairs as examples
    example_pairs = all_io_pairs[1:5] if len(all_io_pairs) > 1 else []
    
    # Select appropriate template based on task type
    if task_type == 'abductive':
        template = input_pred_template
    elif task_type == 'inductive':
        template = inductive_template
    else:  # Default to deductive
        template = output_pred_template
    
    # Format the prompt based on the task type
    if task_type == 'abductive':
        prompt = template.replace("<<<<query>>>>", problem_statement).replace("<<<<io_req>>>>", io_req)
        output_str = json.dumps(output_data, indent=2)
        prompt = prompt.replace("<<<<output>>>>", output_str)
        
        # Add reference code for abductive tasks
        refcode_part = refcode_template.replace("<<<<refcode>>>>", reference_code)
        prompt += "\n\n" + refcode_part
        
    elif task_type == 'inductive':
        prompt = template.replace("<<<<query>>>>", problem_statement).replace("<<<<io_req>>>>", io_req)
        
        # Format example pairs
        examples_str = ""
        for i, example in enumerate(example_pairs):
            examples_str += f"Example {i+1}:\nInput: {json.dumps(example['input'], indent=2)}\nOutput: {json.dumps(example['output'], indent=2)}\n\n"
        
        prompt = prompt.replace("<<<<examples>>>>", examples_str)
        
        # Add the test input
        input_str = json.dumps(input_data, indent=2)
        prompt = prompt.replace("<<<<input>>>>", input_str)
        
        # For inductive tasks, we deliberately don't add reference code to force pattern recognition
        
    else:  # Default to deductive
        prompt = template.replace("<<<<query>>>>", problem_statement).replace("<<<<io_req>>>>", io_req)
        input_str = json.dumps(input_data, indent=2)
        prompt = prompt.replace("<<<<input>>>>", input_str)
        
        # Add reference code for deductive tasks
        refcode_part = refcode_template.replace("<<<<refcode>>>>", reference_code)
        prompt += "\n\n" + refcode_part
    
    return prompt, io_pair

def get_expected_field_and_value(task_type: str, io_pair: Dict[str, Any]) -> tuple:
    """Get the expected field and value based on task type."""
    if task_type == 'abductive':
        return 'input', io_pair.get('input')
    else:  # deductive or inductive
        return 'output', io_pair.get('output')

def process_dataset(input_file: str, output_file: str, task_types: List[str], preview_mode: bool = False, 
                   logic_rl_format: bool = False, split_dataset: bool = False, 
                   train_ratio: float = 0.85, seed: int = 42, max_train_examples: int = None,
                   shuffle_train: bool = False) -> None:
    """Process the dataset to create a balanced set of reasoning tasks.
    
    Args:
        input_file: Path to the input JSONL file
        output_file: Path to the output JSONL or Parquet file
        task_types: List of task types to include
        preview_mode: If True, only process 5 records (for quick previewing)
        logic_rl_format: If True, output in Logic-RL compatible format (Parquet)
        split_dataset: If True, split the dataset into train and validation sets
        train_ratio: Ratio of training data (0.85 means 85% train, 15% validation)
        seed: Random seed for reproducibility
        max_train_examples: Maximum number of examples in the training set (validation examples will be added on top of this)
        shuffle_train: If True, shuffle training data; otherwise maintain order
    """
    try:
        # Set random seed for reproducibility
        random.seed(seed)
        
        # Load the input dataset
        with open(input_file, 'r', encoding='utf-8') as f:
            records = [json.loads(line) for line in f if line.strip()]
        
        print(f"Loaded {len(records)} records from {input_file}")
        
        # Ensure records have io_pairs and filter out those without
        valid_records = [r for r in records if r.get('io_pairs') and len(r.get('io_pairs', [])) >= 1]
        print(f"Found {len(valid_records)} records with valid IO pairs")
        
        # Sort valid records by the length of their reference code (number of lines)
        valid_records.sort(key=lambda r: len(r.get('reference_code', '').split('\n')))
        print(f"Sorted records by reference code length")
        
        # For preview mode, only take a small representative sample
        if preview_mode:
            # Get 5 records total, with at least one of each task type if possible
            sample_records = []
            remaining_types = task_types.copy()
            
            # First try to get one of each type
            for record in valid_records:
                if not remaining_types:
                    break
                
                # Temporarily assign a task type for sampling purposes
                record_type = task_types[len(sample_records) % len(task_types)]
                if record_type in remaining_types:
                    record['task_type'] = record_type
                    sample_records.append(record)
                    remaining_types.remove(record_type)
            
            # Fill the rest with any valid records
            remaining = 100 - len(sample_records)
            if remaining > 0:
                # Get the next few records
                for record in valid_records[len(sample_records):len(sample_records) + remaining]:
                    record['task_type'] = task_types[len(sample_records) % len(task_types)]
                    sample_records.append(record)
            
            valid_records = sample_records
            print(f"Preview mode: selected {len(valid_records)} records")
        else:
            # Add task type to each record in a round-robin fashion
            valid_records = add_task_type(valid_records, task_types)
            
            # Shuffle records if requested
            if shuffle_train:
                random.shuffle(valid_records)
                print(f"Shuffled records as requested")
        
        # Create reasoning tasks
        tasks = []
        for record in valid_records:
            task_type = record['task_type']
            
            try:
                # Format the prompt using consistent IO pair selection
                prompt, io_pair = format_codeio_prompt(record, task_type)
                
                # Get expected field and value
                expected_field, expected_value = get_expected_field_and_value(task_type, io_pair)
                
                # Create task
                task = {
                    'prompt': prompt,
                    # 'context': record['context'],
                    'solution': {expected_field: expected_value},
                    'reference_code_length': len(record['reference_code'].split('\n')),
                    'task_type': task_type,
                    'io_pair': io_pair
                }
                
                tasks.append(task)
            except Exception as e:
                print(f"Error processing record: {str(e)}")
                continue
        
        # Split into train and validation sets if requested
        if split_dataset and not preview_mode:
            # Shuffle tasks to ensure randomization
            random.shuffle(tasks)
            
            # If max_train_examples is set, calculate total examples needed
            if max_train_examples is not None:
                # Ensure we have enough examples
                if len(tasks) < max_train_examples:
                    print(f"Warning: Only {len(tasks)} valid examples available, using all for training")
                    train_tasks = tasks
                    val_tasks = []
                else:
                    # Calculate validation size based on train_ratio
                    val_ratio = 1 - train_ratio
                    
                    total_size = int(max_train_examples / train_ratio)
                    val_size = total_size - max_train_examples
                    
                    # Total examples needed
                    total_needed = max_train_examples + val_size
                    
                    # Ensure we don't try to use more examples than we have
                    total_examples = min(total_needed, len(tasks))
                    
                    # If we don't have enough for the ideal split, recalculate
                    if total_examples < total_needed:
                        train_size = int(total_examples * train_ratio)
                    else:
                        train_size = max_train_examples
                    
                    # Split the tasks
                    train_tasks = tasks[:train_size]
                    val_tasks = tasks[train_size:total_examples]
                    
                    # Always shuffle validation
                    random.shuffle(val_tasks)
                    
                    # Shuffle train if requested
                    if shuffle_train:
                        random.shuffle(train_tasks)
                    
                    print(f"Using {len(train_tasks)} examples for training and {len(val_tasks)} for validation")
            else:
                # Use the original train_ratio on all examples
                train_size = int(len(tasks) * train_ratio)
                train_tasks = tasks[:train_size]
                val_tasks = tasks[train_size:]
                
                # Always shuffle validation
                random.shuffle(val_tasks)
                
                # Shuffle train if requested
                if shuffle_train:
                    random.shuffle(train_tasks)
            
            print(f"Split dataset into {len(train_tasks)} training samples and {len(val_tasks)} validation samples")
            
            # Prepare output filenames for train and validation sets
            base_name, ext = os.path.splitext(output_file)
            train_output = f"{base_name}_train{ext}"
            val_output = f"{base_name}_val{ext}"
            
            # Save train and validation sets
            if logic_rl_format:
                # Process and save train set
                train_logic_rl_records = create_logic_rl_records(train_tasks)
                train_df = pd.DataFrame(train_logic_rl_records)
                train_df.to_parquet(train_output, index=False)
                
                # Process and save validation set if there are validation samples
                if val_tasks:
                    val_logic_rl_records = create_logic_rl_records(val_tasks)
                    val_df = pd.DataFrame(val_logic_rl_records)
                    val_df.to_parquet(val_output, index=False)
                    print(f"Processed validation dataset saved to {val_output} in Logic-RL format (Parquet)")
                
                print(f"Processed train dataset saved to {train_output} in Logic-RL format (Parquet)")
            else:
                # Save train set as JSONL
                with open(train_output, 'w', encoding='utf-8') as f:
                    for task in train_tasks:
                        f.write(json.dumps(task) + '\n')
                
                # Save validation set as JSONL if there are validation samples
                if val_tasks:
                    with open(val_output, 'w', encoding='utf-8') as f:
                        for task in val_tasks:
                            f.write(json.dumps(task) + '\n')
                    print(f"Processed validation dataset saved to {val_output} in JSONL format")
                
                print(f"Processed train dataset saved to {train_output} in JSONL format")
                
        else:
            # Limit the dataset if max_train_examples is specified and no split is requested
            if max_train_examples is not None and not preview_mode:
                # Shuffle before limiting to ensure a random subset
                random.shuffle(tasks)
                tasks = tasks[:max_train_examples]
                print(f"Limited dataset to {len(tasks)} examples as specified")
                
            # Shuffle if requested and no split
            if shuffle_train and not preview_mode:
                random.shuffle(tasks)
                print("Shuffled final task list as requested")
            
            # Save the processed dataset in the appropriate format (no split)
            if logic_rl_format:
                # Convert to Logic-RL format
                logic_rl_records = create_logic_rl_records(tasks)
                
                # Convert to a pandas DataFrame
                df = pd.DataFrame(logic_rl_records)
                
                # Save as a Parquet file
                df.to_parquet(output_file, index=False)
                print(f"Processed dataset saved to {output_file} in Logic-RL format (Parquet)")
            else:
                # Save as JSONL (original format)
                with open(output_file, 'w', encoding='utf-8') as f:
                    for task in tasks:
                        f.write(json.dumps(task) + '\n')
                
                print(f"Processed dataset saved to {output_file} in JSONL format")
        
        # Print statistics
        task_types_count = {t: 0 for t in ['deductive', 'abductive', 'inductive']}
        for task in tasks:
            task_types_count[task['task_type']] += 1
        
        print(f"Task type distribution:")
        for task_type, count in task_types_count.items():
            if count > 0:
                percentage = count/len(tasks)*100 if tasks else 0
                print(f"  {task_type.capitalize()}: {count} ({percentage:.1f}%)")
        
    except Exception as e:
        print(f"Error processing dataset: {str(e)}")

def create_logic_rl_records(tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Convert tasks to Logic-RL format records."""
    logic_rl_records = []
    for task in tasks:
        prompt = task.get('prompt', '')
        task_type = task.get('task_type', 'unknown')
        solution = task.get('solution', {})
        io_pair = task.get('io_pair', {})
        ref_code_length = task.get('reference_code_length', 0)
        
        # Create the system prompt that matches KK dataset format
        system_message = (
            "You are a helpful assistant. The assistant first thinks about the reasoning process "
            "and then provides the user with the answer. The reasoning process should be "
            "enclosed within <think> </think> tags, i.e., <think> reasoning process here </think>. "
            "For your final answer, you must format it as a JSON object, exactly as specified in the prompt, "
            "and enclose it within <answer> </answer> tags. "
            "For example: <answer>{\"output\": value}</answer> or <answer>{\"input\": value}</answer> depending on what's requested. "
            "Now the user asks you to solve a complex problem. After thinking through your reasoning, "
            "clearly state your answer as a properly formatted JSON object within answer tags."
        )

        # Format exactly like KK dataset with system prompt second
        combined = (
            "<|im_start|>user\n"
            f"{prompt}\n"
            "<|im_end|>\n"
            "<|im_start|>system\n"
            f"{system_message}\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n<think>"
        )

        # Convert nested dictionaries to strings to avoid dataframe issues
        solution_str = json.dumps(solution)
        io_pair_str = json.dumps(io_pair)

        logic_rl_record = {
            "prompt": [
                {
                    "role": "user",
                    "content": combined
                }
            ],
            "data_source": "reason_io",
            "reward_model": {
                "ground_truth": {
                    "solution": solution_str,
                    "task_type": task_type,
                    "io_pair": io_pair_str,
                    "reference_code_length": ref_code_length
                }
            },
            "reference_code_length": ref_code_length
        }
        
        logic_rl_records.append(logic_rl_record)
    
    return logic_rl_records

def main():
    parser = argparse.ArgumentParser(description='Process dataset to create reasoning tasks')
    parser.add_argument('--input', type=str, required=False, default='CodeIO-RL/full_run_20250509_020645/io/final_dataset.jsonl', help='Input jsonl file path')
    parser.add_argument('--output', type=str, required=False, default='final_dataset/3_reasoning_types_dataset_sorted.jsonl', help='Output file path (JSONL or Parquet)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed for reproducibility')
    parser.add_argument('--deductive', action='store_true', help='Include deductive reasoning tasks')
    parser.add_argument('--abductive', action='store_true', help='Include abductive reasoning tasks')
    parser.add_argument('--inductive', action='store_true', help='Include inductive reasoning tasks')
    parser.add_argument('--preview', action='store_true', help='Create a small preview dataset with only 5 records')
    parser.add_argument('--logic_rl_format', action='store_true', help='Output in Logic-RL compatible format (Parquet)')
    parser.add_argument('--split', action='store_true', help='Split the dataset into train and validation sets')
    parser.add_argument('--train_ratio', type=float, default=0.85, help='Ratio of training data (0.85 means 85% train, 15% validation)')
    parser.add_argument('--max_train_examples', type=int, default=None, help='Maximum number of examples in the training set (validation examples will be added on top of this)')
    parser.add_argument('--shuffle_train', action='store_true', help='Shuffle training data instead of ordering by code length')
    
    args = parser.parse_args()
    
    # Set random seed for reproducibility
    random.seed(args.seed)
    
    # Determine which task types to include
    task_types = []
    if args.deductive:
        task_types.append('deductive')
    if args.abductive:
        task_types.append('abductive')
    if args.inductive:
        task_types.append('inductive')
    
    # If no task types are specified, include all types
    if not task_types:
        task_types = ['deductive', 'abductive', 'inductive']
    
    print(f"Including task types: {', '.join(task_types)}")
    
    # If preview mode is enabled, modify the output filename
    if args.preview:
        base_name, ext = os.path.splitext(args.output)
        args.output = f"{base_name}_preview{ext}"
        print(f"Preview mode enabled, using output file: {args.output}")
    
    # If logic_rl_format is enabled, ensure the output file has .parquet extension
    if args.logic_rl_format:
        base_name, ext = os.path.splitext(args.output)
        if ext.lower() != '.parquet':
            args.output = f"{base_name}.parquet"
            print(f"Logic-RL format enabled, changing output file extension to: {args.output}")
    
    # Process the dataset
    process_dataset(
        args.input, 
        args.output, 
        task_types, 
        preview_mode=args.preview, 
        logic_rl_format=args.logic_rl_format,
        split_dataset=args.split,
        train_ratio=args.train_ratio,
        seed=args.seed,
        max_train_examples=args.max_train_examples,
        shuffle_train=args.shuffle_train
    )

if __name__ == "__main__":
    main()