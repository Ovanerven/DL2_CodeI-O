import re
import json
import ast
from typing import Dict, Any, Optional, Tuple

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

def extract_json_answer(s: str) -> Optional[Dict[str, Any]]:
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

def extract_solution(solution_str: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer_dict, processed_string)
    """
    # Split response to isolate assistant output
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        print("[Error] Failed to locate model response header")
        return None, solution_str

    # Extract the JSON answer using the extract_json_answer function
    answer_dict = extract_json_answer(processed_str)
    
    return answer_dict, processed_str

def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    tags = {
        'think_start': ('<think>', 1),
        'think_end': ('</think>', 1),
        'answer_start': ('<answer>', 1),
        'answer_end': ('</answer>', 1)
    }

    positions = {}
    for tag_name, (tag_str, expected_count) in tags.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        
        print(f"  {tag_str}: count={count}, position={pos}")
        
        if count != expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected {expected_count})")
            validation_passed = False

    # Verify tag order
    if (positions['think_start'] > positions['think_end'] or
        positions['think_end'] > positions['answer_start'] or
        positions['answer_start'] > positions['answer_end']):
        print("  [Error] Incorrect tag order: Expected <think>...</think><answer>...</answer>")
        validation_passed = False
    else:
        print("  Tag sequence validation passed")

    return validation_passed

def compute_score(solution_str: str, ground_truth: Dict[str, Any], 
                 format_reward: int = 1,
                 answer_reward: float = 2.0) -> float:
    """
    Compute the reward score for the model's response to Reason-IO dataset.
    
    Args:
        solution_str: The model's full response string
        ground_truth: Dictionary containing ground truth data including expected solution
        format_reward: Points awarded/deducted for format correctness
        answer_reward: Points awarded/deducted for answer correctness
        
    Returns:
        Total score (sum of format and answer rewards)
    """
    print("\n" + "="*80)
    print(" Evaluating Reason-IO Response ".center(80, '='))
    
    # Extract solution data from the flattened parquet structure
    # Check for flattened keys first (from parquet)
    solution_str_field = ground_truth.get('solution', None)
    
    # Try to get flattened fields from the ground truth dictionary
    for key in ground_truth:
        if 'reward_model.ground_truth.solution' in key:
            solution_str_field = ground_truth[key]
        elif 'solution' in key and isinstance(ground_truth[key], str):
            solution_str_field = ground_truth[key]
    
    # Parse solution if it's a string
    if isinstance(solution_str_field, str):
        try:
            solution_data = json.loads(solution_str_field)
        except (json.JSONDecodeError, TypeError):
            solution_data = {}
    else:
        solution_data = solution_str_field if solution_str_field is not None else {}
    
    # Extract task type
    task_type = None
    for key in ground_truth:
        if 'task_type' in key:
            task_type = ground_truth[key]
            break
    
    if task_type is None:
        task_type = 'unknown'
        
    # Extract the expected field and value
    expected_solution = solution_data
    expected_field = list(expected_solution.keys())[0] if expected_solution else None  # 'input' or 'output'
    expected_value = expected_solution.get(expected_field) if expected_field else None
    
    print(f"[Ground Truth]")
    print(f"  Task Type: {task_type}")
    print(f"  Expected Field: {expected_field}")
    print(f"  Expected Value: {expected_value}")
    
    # Extract model's answer
    answer_dict, processed_str = extract_solution(solution_str)
    
    # Print the full model response
    print(f"\n[Full Model Response]")
    print(processed_str)
    
    # Validate response structure
    format_correct = validate_response_structure(processed_str)
    format_score = format_reward if format_correct else -abs(format_reward)
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")
    
    # Validate answer content
    answer_score = 0
    if format_correct and answer_dict:
        print(f"\n[Content Validation]")
        
        # Check if the expected field exists in the model's answer
        model_value = answer_dict.get(expected_field) if answer_dict else None
        
        if model_value is not None:
            print(f"  Model answer: {model_value}")
            
            # Normalize both expected and model values for comparison
            normalized_expected = normalize_literal(expected_value)
            normalized_model = normalize_literal(model_value)
            
            # Compare the normalized values
            if normalized_model == normalized_expected:
                answer_score = answer_reward
                print("  Content validation: CORRECT")
            else:
                answer_score = -1.5
                print("  Content validation: INCORRECT")
                print(f"  Expected: {normalized_expected}")
                print(f"  Got: {normalized_model}")
        else:
            answer_score = -2.0
            print(f"  [Error] Field '{expected_field}' missing from answer")
    else:
        answer_score = -2.0
        print("\n[Content Validation] Skipped due to format errors or missing answer")
    
    total_score = format_score + answer_score
    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {answer_score}")
    print(f"  Total: {total_score}")
    print("="*80 + "\n")
    
    return total_score 