import json
import os
import random
import argparse
import time
import ast
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

def load_jsonl(file_path: str, max_records: int = None) -> List[Dict[str, Any]]:
    """Load records from a JSONL file"""
    records = []
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if max_records is not None and i >= max_records:
                    break
                if line.strip():
                    record = json.loads(line)
                    records.append(record)
        print(f"Loaded {len(records)} records from {file_path}")
        return records
    except Exception as e:
        print(f"Error loading {file_path}: {str(e)}")
        return []

def normalize_literal(s: str):
    """
    Turn a string like '{"x":1,"y":1}', "{'x':2,'y':1}", '"False"', 'False', etc.
    into the corresponding Python object: dict, bool, int, etc.
    """
    # If it's not a string, return as is
    if not isinstance(s, str):
        return s
        
    # 1. Try JSON first (handles double-quoted JSON)
    try:
        val = json.loads(s)
    except (ValueError, json.JSONDecodeError):
        # 2. Fall back to Python literal_eval (handles single-quoted dicts, ints, booleans)
        try:
            val = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            # 3. Nothing to parse, leave as raw string
            val = s

    # 4. If we still have a string that is exactly "True"/"False", turn it into a bool
    if isinstance(val, str) and val.lower() in ('true', 'false'):
        return val.lower() == 'true'

    return val

def extract_json_answer(s: str):
    """
    Pulls out the JSON-like blob inside <answer>…</answer> (or <answer>…<answer>)
    and returns it as a Python object, using json.loads first, then ast.literal_eval.
    """
    # 1) allow a closing </answer> or a stray <answer>
    pattern = r'<answer>\s*(\{[\s\S]*?\})\s*(?:</answer>|<answer>)'
    m = re.search(pattern, s, re.IGNORECASE)
    if not m:
        return None

    blob = m.group(1)
    # 2) try strict JSON
    try:
        return json.loads(blob)
    except json.JSONDecodeError:
        # 3) fall back to Python literal
        try:
            return ast.literal_eval(blob)
        except (ValueError, SyntaxError):
            return None

def run_reasoning_evaluation(
    model_name: str,
    dataset_path: str,
    num_samples: int = 5,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    use_chat_format: bool = True,
    max_records: int = None
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Evaluate a language model's ability to reason through coding problems step-by-step.
    
    Args:
        model_name: HuggingFace model name or path
        dataset_path: Path to the JSONL dataset (pre-formatted with prompts and solutions)
        num_samples: Number of samples to evaluate
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for generation
        use_chat_format: Whether to use a chat format for the model
        max_records: Maximum number of records to load from the dataset
        
    Returns:
        accuracy: Percentage of correct answers
        results: List of evaluation results
    """
    print(f"Loading model {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, 
        device_map="auto",
        torch_dtype=torch.float16,
        trust_remote_code=True
    )
    
    # Check if the model supports chat format
    supports_chat = hasattr(tokenizer, "apply_chat_template") and callable(getattr(tokenizer, "apply_chat_template"))
    use_chat_format = use_chat_format and supports_chat
    
    if use_chat_format:
        print("Using chat format for this model")
    else:
        print("Using regular text format for this model")
    
    print(f"Loading dataset from {dataset_path}...")
    records = load_jsonl(dataset_path, max_records)
    
    if not records:
        print("No records found in the dataset.")
        return 0.0, []
    
    # Dataset should be pre-formatted with prompts and expected solutions
    valid_records = [r for r in records if r.get('prompt') and r.get('solution')]
    if not valid_records:
        print("No valid pre-formatted records found in the dataset. Make sure to preprocess with reasoning_questions.py first.")
        return 0.0, []
    
    # Sample records if needed
    sampled_records = valid_records
    if num_samples and num_samples < len(valid_records):
        # Ensure we have a balanced sample across task types if they exist
        if all('task_type' in r for r in valid_records):
            task_types = list(set(r['task_type'] for r in valid_records))
            records_by_type = {t: [] for t in task_types}
            for r in valid_records:
                records_by_type[r['task_type']].append(r)
            
            # Determine how many of each type to sample
            n = num_samples
            base = n // len(task_types)
            remainder = n % len(task_types)
            type_counts = {t: base for t in task_types}
            for i, t in enumerate(task_types):
                if i < remainder:
                    type_counts[t] += 1
            
            # Sample from each type
            sampled_records = []
            for t in task_types:
                available = records_by_type[t]
                k = min(type_counts[t], len(available))
                if k > 0:
                    sampled_records.extend(random.sample(available, k))
            random.shuffle(sampled_records)
        else:
            # If task_type is not available, sample randomly
            sampled_records = random.sample(valid_records, num_samples)
    
    results = []
    correct_count = 0
    
    # Create a dictionary to track correct counts by task type
    task_counts = {}
    task_correct = {}
    format_errors = 0
    
    for i, record in enumerate(sampled_records):
        # Get the prompt and expected solution directly from the record
        prompt = record['prompt']
        task_type = record.get('task_type', 'unknown')
        expected_field = record.get('expected_field', list(record['solution'].keys())[0])
        expected_value = record['solution'].get(expected_field)
        
        # Update task counts
        task_counts[task_type] = task_counts.get(task_type, 0) + 1
        
        print(f"\nEvaluating problem {i+1}/{len(sampled_records)}...")
        print(f"Task type: {task_type}")
        print(f"Problem: {record.get('context', '')[:100]}...")
        
        # Print the full prompt for debugging
        print("=== PROMPT TO MODEL ===")
        print(prompt)
        print("======================")
        
        # Generate the reasoning
        start_time = time.time()
        
        if use_chat_format:
            # Create a chat message format
            messages = [
                {
                    "role": "system", 
                    "content": (
                        "You are a helpful assistant. The assistant first thinks about the reasoning process "
                        "and then provides the user with the answer. The reasoning process should be "
                        "enclosed within <think> </think> tags, i.e., <think> reasoning process here </think>. "
                        "For your final answer, you must format it as a JSON object, exactly as specified in the prompt, "
                        "and enclose it within <answer> </answer> tags. "
                        "For example: <answer>{\"output\": value}</answer> or <answer>{\"input\": value}</answer> depending on what's requested. "
                        "Now the user asks you to solve a complex problem. After thinking through your reasoning, "
                        "clearly state your answer as a properly formatted JSON object within answer tags. "
                    )
                },
                {"role": "user", "content": prompt}
            ]
            chat_text = tokenizer.apply_chat_template(messages, tokenize=False)
            inputs = tokenizer(chat_text, return_tensors="pt").to(model.device)
        else:
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=(temperature > 0)
            )
        
        # Get only the model's response, not including the original prompt
        if use_chat_format:
            generated_full = tokenizer.decode(outputs[0], skip_special_tokens=True)
            # Extract the model's response after the prompt
            if prompt in generated_full:
                generated_text = generated_full.split(prompt, 1)[1].strip()
            else:
                generated_text = generated_full
        else:
            full_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
            if prompt in full_text:
                generated_text = full_text.split(prompt, 1)[1].strip()
            else:
                generated_text = full_text
        
        # Print the full model output for debugging
        print("=== FULL MODEL OUTPUT ===")
        print(generated_text)
        print("=========================")
        
        # Extract the last complete JSON object from the output
        model_json = extract_json_answer(generated_text)
        
        end_time = time.time()
        duration = end_time - start_time
        print(f"Expected {expected_field}: {expected_value}")
        
        # Try to evaluate if the answer is correct
        is_correct = False
        if model_json is not None:
            model_value = model_json.get(expected_field) if isinstance(model_json, dict) else model_json
            
            # Normalize both expected and model values
            normalized_expected = normalize_literal(expected_value)
            normalized_model = normalize_literal(model_value)
            
            # Compare the normalized values
            is_correct = normalized_model == normalized_expected
            
            model_answer_str = json.dumps(model_value)
        else:
            model_answer_str = "No valid JSON answer found"
        
        if is_correct:
            correct_count += 1
            task_correct[task_type] = task_correct.get(task_type, 0) + 1
            print("✓ CORRECT")
        else:
            print("✗ INCORRECT")
        
        if model_json is None:
            print("! FORMAT ERROR: Answer not provided in the required JSON format")
            format_errors += 1
        
        print(f"Model's answer: {model_answer_str}")
        print(f"Time taken: {duration:.2f} seconds")
        
        # Save the result
        result = {
            "problem_id": i+1,
            "task_type": task_type,
            "expected_field": expected_field,
            "expected_value": expected_value,
            "model_reasoning": generated_text,
            "model_answer": model_answer_str,
            "is_correct": is_correct,
            "format_error": model_json is None,
            "time_taken": duration
        }
        
        # Add additional fields from the record if available
        if 'context' in record:
            result["context"] = record['context']
        if 'io_pair' in record:
            result["io_pair"] = record['io_pair']
        
        results.append(result)
    
    # Calculate accuracy
    accuracy = correct_count / len(results) if results else 0.0
    print(f"\nAccuracy: {accuracy * 100:.2f}% ({correct_count}/{len(results)} correct)")
    
    # Calculate accuracy by task type
    accuracy_by_type = {}
    for task_type in task_counts.keys():
        type_accuracy = task_correct.get(task_type, 0) / task_counts[task_type] if task_counts[task_type] > 0 else 0.0
        accuracy_by_type[task_type] = type_accuracy
        print(f"{task_type.capitalize()} accuracy: {type_accuracy * 100:.2f}% ({task_correct.get(task_type, 0)}/{task_counts[task_type]} correct)")
    
    # Print format error statistics
    format_error_rate = format_errors / len(results) if results else 0.0
    print(f"Format errors: {format_errors}/{len(results)} ({format_error_rate * 100:.2f}%)")
    
    return accuracy, results

def main():
    parser = argparse.ArgumentParser(description='Test a language model\'s reasoning abilities on code problems')
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-3B-Instruct", help='HuggingFace model name or path')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the pre-formatted JSONL dataset')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples to evaluate')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for generation')
    parser.add_argument('--output', type=str, default="reasoning_evaluation_results.json", help='Output file for evaluation results')
    parser.add_argument('--no_chat', action='store_true', help='Disable chat format even for models that support it')
    parser.add_argument('--max_records', type=int, help='Maximum number of records to load from the dataset')
    
    args = parser.parse_args()
    
    # Evaluate the model
    accuracy, results = run_reasoning_evaluation(
        model_name=args.model,
        dataset_path=args.dataset,
        num_samples=args.samples,
        max_tokens=args.max_tokens,
        temperature=args.temperature,
        use_chat_format=not args.no_chat,
        max_records=args.max_records
    )
    
    # Save the results
    output_dir = os.path.dirname(args.output)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    with open(args.output, 'w', encoding='utf-8') as f:
        json.dump({
            "model": args.model,
            "dataset": args.dataset,
            "accuracy": accuracy,
            "num_samples": len(results),
            "results": results
        }, f, indent=2)
    
    print(f"Results saved to {args.output}")

if __name__ == "__main__":
    main()