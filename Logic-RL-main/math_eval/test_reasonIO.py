#!/usr/bin/python python3
import warnings
warnings.filterwarnings("ignore")
import re
import os
import csv
import json
import time
import types
import random
import textwrap
from tqdm import tqdm
from datetime import datetime
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
from vllm import LLM, SamplingParams
from datetime import datetime
from collections import defaultdict

# Import the official ReasonIO evaluation functions
import sys
import re
import json
import ast
from typing import Dict, Any, Optional, Tuple

def normalize_literal(s):
    """
    Try to parse `s` as JSON first, then as a Python literal.
    If it's a dict or scalar, return it directly.
    If it's a list/tuple/set, return it as a list.
    Otherwise fall back to returning the raw string.
    """
    # fast-path for non-strings
    if not isinstance(s, str):
        return s

    # attempt JSON, then Python literal
    for parser in (json.loads, ast.literal_eval):
        try:
            val = parser(s)
            break
        except Exception:
            continue
    else:
        # couldn't parse—return original string
        return s

    # normalize container types
    if isinstance(val, dict):
        return val
    if isinstance(val, (list, tuple, set)):
        return list(val)
    # scalar (int, float, bool, str, etc.)
    return val

def extract_json_answer(s: str) -> Optional[Dict[str, Any]]:
    """
    Pulls out the JSON-like blob inside <answer>…</answer>
    """
    patterns = [
        r'<answer>\s*(\{[\s\S]*?\})\s*</answer>',
        r'<answer>\s*(\{[\s\S]*?\})\s*<answer>',
        r'<answer>\s*(\{[\s\S]*?\})\s*$'
    ]
    for pattern in patterns:
        matches = re.findall(pattern, s, re.IGNORECASE)
        if matches:
            valid = [m for m in matches if m.strip() != '{}']
            blob = valid[-1] if valid else matches[-1]
            try:
                return json.loads(blob)
            except json.JSONDecodeError:
                try:
                    return ast.literal_eval(blob)
                except Exception:
                    continue
    return None

def extract_prompt_content(prompt_data):
    """Extract the actual prompt text from the complex prompt structure"""
    if isinstance(prompt_data, np.ndarray) and len(prompt_data) > 0:
        prompt_obj = prompt_data[0]
        if isinstance(prompt_obj, dict) and 'content' in prompt_obj:
            content = prompt_obj['content']
            # Extract the user's question part before the system message
            if '<|im_end|>' in content:
                user_part = content.split('<|im_end|>')[0]
                # Remove the initial <|im_start|>user\n
                if user_part.startswith('<|im_start|>user\n'):
                    return user_part[len('<|im_start|>user\n'):]
            return content
    return str(prompt_data)

def parse_ground_truth(reward_model_data):
    """Parse ground truth data from reward_model field"""
    try:
        ground_truth = reward_model_data.get('ground_truth', {})
        
        # Parse solution (JSON string to dict)
        solution_str = ground_truth.get('solution', '{}')
        if isinstance(solution_str, str):
            solution = json.loads(solution_str)
        else:
            solution = solution_str
            
        # Get task type
        task_type = ground_truth.get('task_type', 'unknown')
        
        # Get reference code length
        ref_code_length = ground_truth.get('reference_code_length', 0)
        
        return solution, task_type, ref_code_length
    except Exception as e:
        print(f"Error parsing ground truth: {e}")
        return {}, 'unknown', 0

def check_answer_correctness(model_answer, expected_solution):
    """Check if model answer matches expected solution using normalize_literal from reason_io.py"""
    if not model_answer or not expected_solution:
        return False
    
    # Get the expected field (either 'input' or 'output')
    expected_field = next(iter(expected_solution), None) if expected_solution else None
    
    if not expected_field or expected_field not in model_answer:
        return False
    
    # Get values and normalize them using the official function
    expected_value = expected_solution[expected_field]
    model_value = model_answer[expected_field]
    
    norm_expected = normalize_literal(expected_value)
    norm_model = normalize_literal(model_value)
    
    return norm_expected == norm_model

def create_difficulty_bins(ref_code_lengths):
    """Create difficulty bins based on reference code length"""
    lengths = [x for x in ref_code_lengths if x > 0]
    if not lengths:
        return [(0, 100, 'Very Easy')]
    
    # Create quintile-based bins (5 bins)
    quantiles = np.percentile(lengths, [0, 20, 40, 60, 80, 100])
    bins = [
        (quantiles[0], quantiles[1], 'Very Easy'),
        (quantiles[1], quantiles[2], 'Easy'),
        (quantiles[2], quantiles[3], 'Medium'), 
        (quantiles[3], quantiles[4], 'Hard'),
        (quantiles[4], quantiles[5], 'Very Hard')
    ]
    return bins

def get_difficulty_bin(ref_code_length, bins):
    """Get difficulty bin for a given reference code length"""
    for min_len, max_len, label in bins:
        if min_len <= ref_code_length <= max_len:
            return label
    return 'Unknown'

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--data_path', type=str, default='Logic-RL-main/data/reason_io/reason_io_dataset_val.parquet')
    parser.add_argument('--print_responses', action='store_true', help='Print model responses for manual verification')
    parser.add_argument('--max_examples', type=int, default=None, help='Limit number of examples to process')
    args = parser.parse_args()
    
    # Extract step number from model path for output filename, following test_aime.py
    step_match = re.search(r'(\d+)$', args.model_path)
    step = step_match.group(1) if step_match else 'unknown'

    # Initialize VLLM model with identical settings as test_aime.py
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        max_num_seqs=4,
        max_model_len=20000
    )
    
    # Use identical sampling params as test_aime.py
    sampling_params = SamplingParams(
        max_tokens=10000,
        temperature=0.8,
        top_p=0.95,
    )

    # Load ReasonIO dataset
    print(f"Loading dataset from {args.data_path}")
    df = pd.read_parquet(args.data_path)
    
    # Limit examples if requested
    if args.max_examples:
        df = df.head(args.max_examples)
        print(f"Limited to {len(df)} examples")
    
    print(f"Processing {len(df)} examples")
    
    # Create difficulty bins
    ref_lengths = df['reference_code_length'].tolist()
    difficulty_bins = create_difficulty_bins(ref_lengths)
    print(f"Created difficulty bins: {difficulty_bins}")
    
    # Initialize tracking variables following test_aime.py pattern
    cnt = 0
    total_time = 0
    results = []
    task_type_stats = defaultdict(lambda: {'correct': 0, 'total': 0})
    difficulty_stats = defaultdict(lambda: {'correct': 0, 'total': 0})

    # Process each example
    for idx, row in enumerate(tqdm(df.iterrows(), total=len(df))):
        d = row[1]  # Get the row data
        
        # Extract prompt content
        prompt_content = extract_prompt_content(d['prompt'])
        
        # Parse ground truth
        expected_solution, task_type, ref_code_length = parse_ground_truth(d['reward_model'])
        
        # Get difficulty bin
        difficulty = get_difficulty_bin(ref_code_length, difficulty_bins)
        
        # Create messages using IDENTICAL format as test_aime.py
        messages = [
            {"role": "system", "content": "You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a math problem. After thinking, when you finally reach a conclusion, clearly state the answer within <answer> </answer> tags. i.e., <answer> (\\boxed{}\\) </answer>."},
            {"role": "user", "content": prompt_content}
        ]
        
        # Apply chat template and generate response - IDENTICAL to test_aime.py
        tokenizer = llm.get_tokenizer()
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        start_time = time.time()
        outputs = llm.generate([text], sampling_params)
        time_taken = time.time() - start_time
        response = outputs[0].outputs[0].text.strip()

        # Extract model answer using the official reason_io function
        model_answer = extract_json_answer(response)
        
        # For fallback, use test_aime.py style extraction if no JSON found
        if model_answer is None:
            if '<answer>' in response:
                result = re.split(r'<answer>', response)[1]
            else:
                result = response[len(response) - 30:]
            # Try to find JSON in the result
            json_pattern = r'\{[^{}]*\}'
            matches = re.findall(json_pattern, result)
            if matches:
                try:
                    model_answer = json.loads(matches[-1])
                except:
                    pass
        
        # Check correctness
        correct = check_answer_correctness(model_answer, expected_solution)
        
        # Update statistics following test_aime.py pattern
        if correct:
            cnt += 1
            task_type_stats[task_type]['correct'] += 1
            difficulty_stats[difficulty]['correct'] += 1
        
        task_type_stats[task_type]['total'] += 1
        difficulty_stats[difficulty]['total'] += 1
        total_time += time_taken
        
        # Store detailed result following test_aime.py structure
        result = {
            "question": prompt_content[:200] + "..." if len(prompt_content) > 200 else prompt_content,
            "generated_output": response,
            "model_answer": model_answer,
            "expected_solution": expected_solution,
            "task_type": task_type,
            "reference_code_length": ref_code_length,
            "difficulty": difficulty,
            "correct": correct,
            "time_taken": time_taken
        }
        results.append(result)
        
        # Print responses for manual verification if requested
        if args.print_responses:
            print(f"\n{'='*80}")
            print(f"Example {idx+1}/{len(df)} - {'CORRECT' if correct else 'INCORRECT'}")
            print(f"Task Type: {task_type}, Difficulty: {difficulty}")
            print(f"Expected: {expected_solution}")
            print(f"Model Answer: {model_answer}")
            print(f"Response:\n{response}")
            print(f"{'='*80}")

    # Calculate accuracy following test_aime.py pattern
    acc = cnt / len(df) if len(df) > 0 else 0
    print(f"ACC: {acc}")
    
    # Print detailed analysis
    print(f"\nOverall Accuracy: {acc:.4f} ({cnt}/{len(df)})")
    print(f"Average time per example: {total_time/len(df):.2f}s")
    
    # Print task type analysis
    print("\n" + "="*50)
    print("ANALYSIS BY TASK TYPE")
    print("="*50)
    for task_type, stats in task_type_stats.items():
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{task_type.capitalize():12} | {acc:.4f} ({stats['correct']:3d}/{stats['total']:3d})")
    
    # Print difficulty analysis  
    print("\n" + "="*50)
    print("ANALYSIS BY DIFFICULTY (Reference Code Length)")
    print("="*50)
    for difficulty, stats in difficulty_stats.items():
        acc = stats['correct'] / stats['total'] if stats['total'] > 0 else 0
        print(f"{difficulty:12} | {acc:.4f} ({stats['correct']:3d}/{stats['total']:3d})")
    
    # Print difficulty bin ranges for reference
    print("\nDifficulty Bin Ranges:")
    for min_len, max_len, label in difficulty_bins:
        print(f"  {label}: {min_len:.1f} - {max_len:.1f} lines")
    
    print(f"\nEvaluation completed successfully!")

if __name__ == "__main__":
    main()