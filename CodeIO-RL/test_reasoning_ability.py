import json
import os
import random
import argparse
import time
from typing import List, Dict, Any, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import re

# Define distinct templates for each reasoning type
output_pred_template = """You are given a question that requires some input and output variables as follows:

<<<<query>>>>

The input and output requirements are as follows:

<<<<io_req>>>>

Given the following input:

<<<<input>>>>

**Required Deductive Reasoning Steps:**
1. Analyze what the input means in the context of the problem
2. Identify the expected transformation from input to output
3. Step through the transformation process systematically
4. Verify your reasoning against the expected output format
5. Double-check your solution

After completing your step-by-step reasoning, provide your final answer in the following JSON format within a code block:

```json
{"output": <your output>}
```

Your <your output> should strictly match the output requirement as specified.

IMPORTANT: After reasoning, provide ONLY the JSON answer block with your final answer. Do NOT include any other text after the JSON block."""

input_pred_template = """You are given a question that requires some input and output variables as follows:

<<<<query>>>>

The input and output requirements are as follows:

<<<<io_req>>>>

Given the following output:

<<<<output>>>>

**Required Abductive Reasoning Steps:**
1. Analyze what the output represents in the context of the problem
2. Work backwards to determine possible inputs that would produce this output
3. Consider constraints on the input format
4. Determine the most likely input values
5. Verify that your proposed input would produce the given output

After completing your step-by-step reasoning, provide your final answer in the following JSON format within a code block:

```json
{"input": <your input>}
```

Your <your input> should be a dictionary with keys that strictly match the input variables' names as specified.

IMPORTANT: After reasoning, provide ONLY the JSON answer block with your final answer. Do NOT include any other text after the JSON block."""

inductive_template = """You are given a question that requires some input and output variables as follows:

<<<<query>>>>

The input and output requirements are as follows:

<<<<io_req>>>>

Given the following input and output pairs:

<<<<examples>>>>

**Required Inductive Reasoning Steps:**
1. Study the example pairs carefully to identify patterns
2. Look for relationships between inputs and outputs
3. Formulate a general rule or pattern based on the examples
4. Apply your inferred rule to the new input
5. Explain why your pattern explains the examples and leads to your answer

Now, can you predict the output for the following input?

<<<<input>>>>

After completing your step-by-step reasoning, provide your final answer in the following JSON format within a code block:

```json
{"output": <your output>}
```

Your <your output> should strictly match the output requirement as specified.

IMPORTANT: After reasoning, provide ONLY the JSON answer block with your final answer. Do NOT include any other text after the JSON block."""

# Define basic templates from original CodeIO (without structured reasoning steps)
basic_output_pred_template = """You are given a question that requires some input and output variables as follows:

<<<<query>>>>

The input and output requirements are as follows:

<<<<io_req>>>>

Given the following input:

<<<<input>>>>

Can you predict the output without writing any code? Please reason and put your final answer in the following json format: {"output": <your output>}, where <your output> should strictly match the output requirement as specified.

IMPORTANT: After reasoning, provide ONLY the JSON answer block with your final answer. Do NOT include any other text after the JSON block."""

basic_input_pred_template = """You are given a question that requires some input and output variables as follows:

<<<<query>>>>

The input and output requirements are as follows:

<<<<io_req>>>>

Given the following output:

<<<<output>>>>

Can you predict a feasible input without writing any code? Please reason and put your final answer in the following json format: {"input": <your input>}, where <your input> should be a dictionary, even if the there is only one input variable, with keys strictly match the input variables' names as specified.

IMPORTANT: After reasoning, provide ONLY the JSON answer block with your final answer. Do NOT include any other text after the JSON block."""

basic_inductive_template = """You are given a question that requires some input and output variables as follows:

<<<<query>>>>

The input and output requirements are as follows:

<<<<io_req>>>>

Given the following input and output pairs:

<<<<examples>>>>

Can you predict the output for the following input? Please reason and put your final answer in the following json format: {"output": <your output>}, where <your output> should strictly match the output requirement as specified.

<<<<input>>>>

IMPORTANT: After reasoning, provide ONLY the JSON answer block with your final answer. Do NOT include any other text after the JSON block."""

refcode_template = """Tip: Here is a reference code snippet for this question. You can refer to this code to guide your reasoning but not copy spans of code directly.

<<<<refcode>>>>"""

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

def extract_reasoning_and_answer(generated_text: str) -> Tuple[str, str]:
    """
    Extract both the reasoning process and the final answer from the generated text.
    Attempts to identify the final answer at the end of the reasoning.
    """
    text = generated_text.strip()
    
    # Identify final answer (typically at the end)
    lines = text.split('\n')
    
    # Look for markers that might indicate the final answer
    answer_markers = [
        "The final output is", "The answer is", "Therefore, the output is",
        "So the result is", "The output is", "Final output:", "Output:",
        "Result:", "Final result:", "Therefore, the result is", "Final answer:"
    ]
    
    answer = ""
    reasoning = text
    
    # First try to find answer markers in the text
    for marker in answer_markers:
        if marker.lower() in text.lower():
            parts = text.lower().split(marker.lower(), 1)
            if len(parts) > 1:
                answer_section = parts[1].strip()
                # If the answer section is multiple lines, take just the first line
                answer = answer_section.split('\n')[0].strip()
                break
    
    # If no markers found, try to extract from the last few lines
    if not answer:
        # Consider the last few lines as potential answers
        for i in range(min(5, len(lines))):
            candidate = lines[-(i+1)].strip()
            # If the line looks like it contains just data (no explanatory text)
            if (candidate.startswith('{') or 
                candidate.startswith('[') or 
                candidate.startswith('"') or
                candidate.isdigit() or
                candidate == "True" or 
                candidate == "False" or
                candidate == "None"):
                answer = candidate
                break
    
    return reasoning, answer

def run_reasoning_evaluation(
    model_name: str,
    dataset_path: str,
    num_samples: int = 5,
    max_tokens: int = 2048,
    temperature: float = 0.7,
    use_chat_format: bool = True,
    use_basic_prompts: bool = False,
    max_records: int = None
) -> Tuple[float, List[Dict[str, Any]]]:
    """
    Evaluate a language model's ability to reason through coding problems step-by-step.
    
    Args:
        model_name: HuggingFace model name or path
        dataset_path: Path to the JSONL dataset
        num_samples: Number of samples to evaluate
        max_tokens: Maximum number of tokens to generate
        temperature: Temperature for generation
        use_chat_format: Whether to use a chat format for the model
        use_basic_prompts: Whether to use basic prompts from original CodeIO
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
    
    if use_basic_prompts:
        print("Using basic prompts from original CodeIO (without structured reasoning steps)")
    else:
        print("Using structured reasoning prompts with step-by-step guidance")
    
    print(f"Loading dataset from {dataset_path}...")
    records = load_jsonl(dataset_path, max_records)
    records = add_task_type(records)
    
    if not records:
        print("No records found in the dataset.")
        return 0.0, []
    
    # Filter records to those that have io_pairs
    valid_records = [r for r in records if r.get('io_pairs') and r['io_pairs'] is not None and r['io_pairs']]
    if not valid_records:
        print("No records with valid IO pairs found.")
        return 0.0, []

    # Group records by task type
    task_types = ['deductive', 'abductive', 'inductive']
    records_by_type = {t: [] for t in task_types}
    for r in valid_records:
        records_by_type[r['task_type']].append(r)

    # Determine how many of each type to sample
    n = num_samples
    base = n // 3
    remainder = n % 3
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
    if not sampled_records:
        print("No records available for sampling.")
        return 0.0, []
    
    results = []
    correct_count = 0
    correct_by_type = {t: 0 for t in task_types}
    total_by_type = {t: 0 for t in task_types}
    
    for i, record in enumerate(sampled_records):
        # For each record, pick a random IO pair
        io_pair = None
        if record.get('io_pairs'):
            io_pair = random.choice(record['io_pairs'])
        else:
            print("No io_pairs for record", i)
            continue
        task_type = record.get('task_type', 'deductive')
        total_by_type[task_type] += 1
        
        prompt = format_codeio_prompt(record, io_pair, task_type, use_basic_prompts)
        print(f"\nEvaluating problem {i+1}/{len(sampled_records)}...")
        print(f"Task type: {task_type}")
        print(f"Problem: {record['context'][:100]}...")
        
        # Print the full prompt for debugging
        print("=== PROMPT TO MODEL ===")
        print(prompt)
        print("======================")
        
        # Generate the reasoning
        start_time = time.time()
        
        if use_chat_format:
            # Create a chat message format
            messages = [
                {"role": "system", "content": "You are a helpful AI assistant trained to solve coding problems through step-by-step reasoning. When approaching a problem, first analyze it thoroughly, then apply logical reasoning to reach a solution."},
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
            # Try to extract only the assistant's response from the chat template
            if "assistant" in generated_full.lower():
                try:
                    # Extract everything after the last occurrence of "assistant" or "assistant:"
                    parts = re.split(r'assistant:?', generated_full.lower(), flags=re.IGNORECASE)
                    if len(parts) > 1:
                        generated_text = parts[-1].strip()
                    else:
                        generated_text = generated_full
                except:
                    generated_text = generated_full
            else:
                # If we can't extract the assistant's response, use the output minus the prompt
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
        
        # Remove any system message-like content at the beginning
        generated_text = re.sub(
            r'^.*?(To approach this problem|To solve this problem|Let\'s analyze|Given the|Looking at|I\'ll solve|You are|system\n|user\n|assistant\n)', 
            'To approach this problem', 
            generated_text, 
            flags=re.IGNORECASE|re.DOTALL
        ).strip()
        
        # Print the full model output for debugging
        print("=== FULL MODEL OUTPUT ===")
        print(generated_text)
        print("=========================")
        
        # Extract the last complete JSON object from the output
        model_json = extract_last_complete_json(generated_text)
        
        # Get the expected output or input depending on task type
        if task_type == 'abductive':
            expected_field = 'input'
        else:
            expected_field = 'output'
        expected_value = io_pair.get(expected_field)
        end_time = time.time()
        duration = end_time - start_time
        print(f"Input: {json.dumps(io_pair, indent=2)}")
        print(f"Expected {expected_field}: {expected_value}")
        # Try to evaluate if the answer is correct
        is_correct = False
        if model_json is not None:
            model_value = model_json.get(expected_field) if isinstance(model_json, dict) else model_json
            is_correct = model_value == expected_value
            model_answer_str = json.dumps(model_value)
        else:
            model_answer_str = generated_text
        if is_correct:
            correct_count += 1
            correct_by_type[task_type] += 1
            print("✓ CORRECT")
        else:
            print("✗ INCORRECT")
        print(f"Model's answer: {model_answer_str}")
        print(f"Time taken: {duration:.2f} seconds")
        # Save the result
        results.append({
            "problem_id": i+1,
            "context": record['context'],
            "task_type": task_type,
            "input": io_pair,
            "expected_output": expected_value,
            "model_reasoning": generated_text,
            "model_answer": model_answer_str,
            "is_correct": is_correct,
            "time_taken": duration
        })
    
    # Calculate accuracy
    accuracy = correct_count / len(results) if results else 0.0
    print(f"\nAccuracy: {accuracy * 100:.2f}% ({correct_count}/{len(results)} correct)")
    
    # Calculate accuracy by task type
    accuracy_by_type = {}
    for task_type in task_types:
        type_results = [r for r in results if r['task_type'] == task_type]
        type_correct = sum(1 for r in type_results if r['is_correct'])
        type_accuracy = type_correct / len(type_results) if type_results else 0.0
        accuracy_by_type[task_type] = type_accuracy
        print(f"{task_type.capitalize()} accuracy: {type_accuracy * 100:.2f}% ({type_correct}/{len(type_results)} correct)")
    
    return accuracy, results

def add_task_type(records):
    task_types = ['deductive', 'abductive', 'inductive']
    for i, rec in enumerate(records):
        rec['task_type'] = task_types[i % len(task_types)]
    return records

def extract_last_complete_json(s):
    # Try to extract the last complete JSON object from a string
    def sub_extract_last_complete_json(s):
        # First look for JSON format with markdown code blocks
        if '```json' in s:
            # Extract all code blocks
            pattern = r'```json\s*(.*?)\s*```'
            matches = re.findall(pattern, s, re.DOTALL)
            if matches:
                for match in matches[-1:]:  # Try the last match first
                    match = match.strip()
                    try:
                        return json.loads(match)
                    except json.JSONDecodeError:
                        try:
                            # Try with some cleaning
                            cleaned = match.replace("False", "false").replace("True", "true").replace("None", "null")
                            return json.loads(cleaned)
                        except json.JSONDecodeError:
                            continue
        
        # If no valid JSON in code blocks, look for JSON objects in the text
        stack = []
        last_json_start = None
        last_json_end = None
        
        for i, char in enumerate(s):
            if char == '{':
                if not stack:  # Start of a potential JSON object
                    last_json_start = i
                stack.append(i)
            elif char == '}':
                if stack:  # Possible end of a JSON object
                    stack.pop()
                    if not stack:  # Complete JSON object found
                        last_json_end = i + 1
        
        if last_json_start is not None and last_json_end is not None:
            last_json_str = s[last_json_start:last_json_end]
            try:
                return json.loads(last_json_str)
            except json.JSONDecodeError:
                # Try with some common replacements for JavaScript vs JSON
                cleaned = last_json_str.replace("False", "false").replace("True", "true").replace("None", "null")
                try:
                    return json.loads(cleaned)
                except json.JSONDecodeError:
                    pass
        
        return None
    
    # First try normal extraction
    res = sub_extract_last_complete_json(s)
    
    # If that fails, try with some common string replacements
    if res is None:
        s = s.replace("\{","{").replace("\}","}").replace('(','[').replace(')',']')
        res = sub_extract_last_complete_json(s)
    
    # If that also fails, look for LaTex boxed answers (common in math problems)
    if res is None and "\\boxed{" in s:
        boxstart = s.rfind("\\boxed{")+len("\\boxed{")
        boxend = s.rfind("}",boxstart)
        boxcontent = s[boxstart:boxend]
        processed_box_content = boxcontent.replace("\\\\","\\").replace("\{","{").replace("\}","}").replace('\\left','').replace('\\right','')
        res = sub_extract_last_complete_json(processed_box_content)
    
    return res

def format_codeio_prompt(record, io_pair, task_type, use_basic_prompts=False):
    problem_statement = record['context']
    reference_code = record['reference_code']
    io_req = record.get('io_requirements', "Please follow the input/output format described in the problem.")
    
    # Select appropriate templates based on use_basic_prompts flag
    if use_basic_prompts:
        # Use original CodeIO templates
        if task_type == 'abductive':
            template = basic_input_pred_template
        elif task_type == 'inductive':
            template = basic_inductive_template
        else:  # Default to deductive
            template = basic_output_pred_template
    else:
        # Use structured reasoning templates
        if task_type == 'abductive':
            template = input_pred_template
        elif task_type == 'inductive':
            template = inductive_template
        else:  # Default to deductive
            template = output_pred_template
    
    # Format the prompt based on the task type
    if task_type == 'abductive':
        prompt = template.replace("<<<<query>>>>", problem_statement).replace("<<<<io_req>>>>", io_req)
        output_str = json.dumps(io_pair['output'], indent=2)
        # Handle both tag formats for compatibility
        if "<<<<o>>>>" in template:
            prompt = prompt.replace("<<<<o>>>>", output_str)
        else:
            prompt = prompt.replace("<<<<output>>>>", output_str)
        
        # Add reference code for abductive tasks
        refcode_part = refcode_template.replace("<<<<refcode>>>>", reference_code)
        prompt += "\n\n" + refcode_part
        
    elif task_type == 'inductive':
        prompt = template.replace("<<<<query>>>>", problem_statement).replace("<<<<io_req>>>>", io_req)
        
        # Get example IO pairs (up to 4 pairs, excluding the current one)
        all_io_pairs = record.get('io_pairs', [])
        example_pairs = []
        example_count = 0
        
        for example in all_io_pairs:
            if example != io_pair and example_count < 4:
                example_pairs.append(example)
                example_count += 1
        
        # Format example pairs
        examples_str = ""
        for i, example in enumerate(example_pairs):
            examples_str += f"Example {i+1}:\nInput: {json.dumps(example['input'], indent=2)}\nOutput: {json.dumps(example['output'], indent=2)}\n\n"
        
        prompt = prompt.replace("<<<<examples>>>>", examples_str)
        
        # Add the test input
        input_str = json.dumps(io_pair['input'], indent=2)
        prompt = prompt.replace("<<<<input>>>>", input_str)
        
        # For inductive tasks, we deliberately don't add reference code to force pattern recognition
        
    else:  # Default to deductive
        prompt = template.replace("<<<<query>>>>", problem_statement).replace("<<<<io_req>>>>", io_req)
        input_str = json.dumps(io_pair['input'], indent=2)
        prompt = prompt.replace("<<<<input>>>>", input_str)
        
        # Add reference code for deductive tasks
        refcode_part = refcode_template.replace("<<<<refcode>>>>", reference_code)
        prompt += "\n\n" + refcode_part
    
    return prompt

def main():
    parser = argparse.ArgumentParser(description='Test a language model\'s reasoning abilities on code problems')
    parser.add_argument('--model', type=str, default="Qwen/Qwen2.5-3B-Instruct", help='HuggingFace model name or path')
    parser.add_argument('--dataset', type=str, required=True, help='Path to the JSONL dataset')
    parser.add_argument('--samples', type=int, default=5, help='Number of samples to evaluate')
    parser.add_argument('--max_tokens', type=int, default=2048, help='Maximum number of tokens to generate')
    parser.add_argument('--temperature', type=float, default=0.7, help='Temperature for generation')
    parser.add_argument('--output', type=str, default="reasoning_evaluation_results.json", help='Output file for evaluation results')
    parser.add_argument('--no_chat', action='store_true', help='Disable chat format even for models that support it')
    parser.add_argument('--basic_prompts', action='store_true', help='Use basic prompts from original CodeIO (without structured reasoning steps)')
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
        use_basic_prompts=args.basic_prompts,
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