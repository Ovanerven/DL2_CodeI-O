import re
import json
import ast
from typing import Dict, Any, Optional, Tuple, List, Union

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

def extract_solution(solution_str: str) -> Tuple[Dict[str, Any], str]:
    """
    Extract the solution from the model's response.
    
    Args:
        solution_str: Model's response to the prompt
        
    Returns:
        Tuple of (extracted solution as dict, processed string)
    """
    processed_str = solution_str.strip()
    
    # Try to find content within <answer> tags
    answer_content = None
    
    # Look for <answer> and </answer> tags
    answer_start = processed_str.find("<answer>")
    answer_end = processed_str.find("</answer>")
    
    if answer_start != -1 and answer_end != -1 and answer_start < answer_end:
        answer_content = processed_str[answer_start + len("<answer>"):answer_end].strip()
        
        # Try to parse JSON content
        try:
            # First try standard JSON parsing
            result = json.loads(answer_content)
        except json.JSONDecodeError:
            try:
                # If JSON parsing fails, try to replace Python expressions with valid JSON
                # Convert Python's 10**10 syntax to numeric value
                answer_content = re.sub(r'(\d+)\s*\*\*\s*(\d+)', 
                                       lambda m: str(int(m.group(1)) ** int(m.group(2))), 
                                       answer_content)
                # Try parsing again
                result = json.loads(answer_content)
            except:
                # If still failing, try ast.literal_eval
                try:
                    result = ast.literal_eval(answer_content)
                except:
                    # Return empty solution if all parsing attempts fail
                    print(f"Failed to parse answer content: {answer_content}")
                    return {}, processed_str
    else:
        # Could not find answer tags
        return {}, processed_str
        
    return result, processed_str

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

def compare_solutions(extracted_solution: Dict[str, Any], expected_solution: Any) -> Tuple[bool, str]:
    """
    Compare the extracted solution with the expected solution.
    
    Args:
        extracted_solution: The solution extracted from the model's response
        expected_solution: The expected solution (can be dict or string)
        
    Returns:
        Tuple of (is_correct, explanation)
    """
    # Handle case when expected_solution is a string
    if isinstance(expected_solution, str):
        try:
            # Try to parse it as JSON
            expected_solution = json.loads(expected_solution)
        except json.JSONDecodeError:
            try:
                # Try as Python literal
                expected_solution = ast.literal_eval(expected_solution)
            except:
                # Use as is if parsing fails
                pass
    
    # Special handling for numeric comparisons
    def compare_values(model_value, expected_value):
        # Convert strings to numbers for accurate comparison
        if isinstance(model_value, str) and model_value.isdigit():
            model_value = int(model_value)
        if isinstance(expected_value, str) and expected_value.isdigit():
            expected_value = int(expected_value)
            
        # Handle numeric comparison with tolerance for floats
        if isinstance(model_value, (int, float)) and isinstance(expected_value, (int, float)):
            if isinstance(model_value, float) or isinstance(expected_value, float):
                # Use small tolerance for float comparison
                return abs(model_value - expected_value) < 1e-6
            else:
                return model_value == expected_value
        
        # Handle dict/list comparison
        if isinstance(model_value, (dict, list)) and isinstance(expected_value, (dict, list)):
            return json.dumps(model_value, sort_keys=True) == json.dumps(expected_value, sort_keys=True)
            
        # Fallback to string comparison
        return str(model_value).strip() == str(expected_value).strip()
    
    # If both are dictionaries
    if isinstance(expected_solution, dict) and isinstance(extracted_solution, dict):
        # First check if the key exists
        for key in expected_solution:
            if key not in extracted_solution:
                return False, f"Missing expected key: {key}"
                
            # Check if values match
            if not compare_values(extracted_solution[key], expected_solution[key]):
                return False, f"Values don't match for key {key}: Expected {expected_solution[key]}, got {extracted_solution[key]}"
                
        return True, "Solutions match"
        
    # Direct comparison for other cases
    return compare_values(extracted_solution, expected_solution), "Direct comparison"

def compute_score(
    solution_str: str, ground_truth: dict, **kwargs
) -> float:
    """
    Compute correctness score for a prediction against a reference.
    
    Args:
        solution_str: Model's prediction string
        ground_truth: Reference data including expected solution
        
    Returns:
        Score as a float
    """
    print("\n" + "="*80)
    print(" Evaluating Reason-IO Response ".center(80, '='))
    
    # Print ground truth information for debugging
    task_type = ground_truth.get('task_type', 'unknown')
    expected_solution = None
    expected_field = None
    expected_value = None
    
    # Extract solution data
    for key in ground_truth:
        if 'solution' in key:
            expected_solution = ground_truth[key]
            
            # Try to parse the solution if it's a string
            if isinstance(expected_solution, str):
                try:
                    expected_solution = json.loads(expected_solution)
                except json.JSONDecodeError:
                    print(f"[Warning] Failed to parse ground truth solution: {expected_solution}")
                    try:
                        expected_solution = ast.literal_eval(expected_solution)
                    except:
                        print(f"[Error] Failed to parse solution string using ast.literal_eval")
            
            if isinstance(expected_solution, dict) and len(expected_solution) > 0:
                expected_field = list(expected_solution.keys())[0]
                expected_value = expected_solution[expected_field]
                break
    
    # Print ground truth information
    print(f"[Ground Truth]")
    print(f"  Task Type: {task_type}")
    if expected_field:
        print(f"  Expected Field: {expected_field}")
        print(f"  Expected Value: {expected_value}")
    
    # Extract solution from the model's prediction
    extracted_solution, processed_str = extract_solution(solution_str)
    
    # Print the full model response
    print(f"\n[Full Model Response]")
    print(processed_str)
    
    # Validate the response structure
    format_correct = validate_response_structure(processed_str)
    
    if format_correct:
        format_score = kwargs.get('format_reward', 1)
    else:
        format_score = -kwargs.get('format_reward', 1)
    
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")
    
    # If format is incorrect, assign a penalty for content
    if not format_correct or not extracted_solution:
        print("\n[Content Validation] Skipped due to format errors or missing answer")
        content_score = -kwargs.get('answer_reward', 2.0)
    else:
        print("\n[Content Validation]")
        
        # Print the extracted answer for debugging
        extracted_value = None
        if expected_field and expected_field in extracted_solution:
            extracted_value = extracted_solution[expected_field]
            print(f"  Model answer: {extracted_value}")
        else:
            print(f"  Model answer: {extracted_solution}")
        
        # Compare extracted solution with expected solution
        is_correct, explanation = compare_solutions(extracted_solution, expected_solution)
        
        if is_correct:
            content_score = kwargs.get('answer_reward', 2.0)
            print(f"  Content validation: CORRECT")
        else:
            content_score = -kwargs.get('answer_reward', 2.0) * 0.75  # Partial penalty for wrong answer
            print(f"  Content validation: INCORRECT")
            print(f"  Expected: {expected_value}")
            print(f"  Got: {extracted_value}")
    
    # Calculate total score
    total_score = format_score + content_score
    
    print("\n" + "-"*80)
    print(" Final Score ".center(80, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {content_score}")
    print(f"  Total: {total_score}")
    print("="*80)
    print("")
    
    return total_score 