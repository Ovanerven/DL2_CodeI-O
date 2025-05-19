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
    # Find all answer blocks - try different pattern variations
    patterns = [
        r'<answer>\s*(\{[\s\S]*?\})\s*</answer>',  # Standard closing tag
        r'<answer>\s*(\{[\s\S]*?\})\s*<answer>',   # Repeated opening tag (typo)
        r'<answer>\s*(\{[\s\S]*?\})\s*$'           # Tag at end without closing
    ]
    
    for pattern in patterns:
        matches = re.findall(pattern, s, re.IGNORECASE)
        if matches:
            # Use the last match (most likely the final answer after reasoning)
            # Also skip empty answers like {}
            valid_matches = [m for m in matches if m.strip() != '{}']
            blob = valid_matches[-1] if valid_matches else matches[-1]
            
            # Try to parse the answer
            try:
                # 1) try strict JSON
                return json.loads(blob)
            except json.JSONDecodeError:
                # 2) fall back to Python literal
                try:
                    return ast.literal_eval(blob)
                except (ValueError, SyntaxError):
                    # Continue to the next pattern if parsing fails
                    continue
    
    # If we couldn't find or parse an answer with any pattern
    print("[Error] Failed to find or parse a valid answer in the response")
    return None

def extract_solution(solution_str: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """Extracts the final answer from the model's response string.
    
    Args:
        solution_str: Raw response string from the language model
        
    Returns:
        Tuple containing (extracted_answer_dict, processed_string)
    """
    # Clean up repeated EOS tokens (appears as <|endoftext|>)
    if "<|endoftext|>" in solution_str:
        # Keep only the content before the first EOS token
        solution_str = solution_str.split("<|endoftext|>")[0]
    
    # The model might repeat the system prompt, so we need to be more flexible
    # Check if the response contains any typical response markers
    if "Assistant:" in solution_str:
        processed_str = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed_str = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        # If we can't find the typical markers, just use the whole string
        # This allows us to at least try to find the answer within it
        print("[Warning] Could not locate model response header, using full response")
        processed_str = solution_str

    # Extract the JSON answer using the extract_json_answer function
    answer_dict = extract_json_answer(processed_str)
    
    return answer_dict, processed_str

def validate_response_structure(processed_str: str) -> bool:
    """Performs comprehensive validation of response structure.
    
    Args:
        processed_str: Processed response string from the model
        
    Returns:
        Boolean indicating whether minimum formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags
    # Now we only require <answer> tags as essential
    essential_tags = {
        'answer_start': ('<answer>', 1),  # Must have at least one answer tag
    }
    
    optional_tags = {
        'answer_end': ('</answer>', 0),  # Optional closing tag (some models might miss it)
        'think_start': ('<think>', 0),   # Optional thinking section
        'think_end': ('</think>', 0)     # Optional closing thinking tag
    }
    
    # Count tag occurrences
    tags_to_check = {**essential_tags, **optional_tags}
    positions = {}
    for tag_name, (tag_str, _) in tags_to_check.items():
        count = processed_str.count(tag_str)
        positions[tag_name] = pos = processed_str.find(tag_str)
        print(f"  {tag_str}: count={count}, position={pos}")
    
    # Check essential tags are present
    for tag_name, (tag_str, expected_count) in essential_tags.items():
        count = processed_str.count(tag_str)
        if count < expected_count:
            print(f"  [Error] {tag_str} appears {count} times (expected at least {expected_count})")
            validation_passed = False

    # Check if <answer> tag contains valid content
    if validation_passed:
        # If we found answer but couldn't extract JSON, it's likely invalid
        answer_json = extract_json_answer(processed_str)
        if answer_json is None:
            print("  [Error] Found <answer> tag but couldn't extract valid JSON content")
            validation_passed = False
        else:
            print("  Found valid JSON content inside <answer> tags")
    
    # Provide a final summary
    if validation_passed:
        print("  Overall structure validation passed")
    else:
        print("  Overall structure validation failed")

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