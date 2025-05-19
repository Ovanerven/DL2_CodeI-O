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

def extract_solution(solution_str: str) -> Tuple[Dict[str, Any], str]:
    """
    Extract the solution from the model's response.
    
    Args:
        solution_str: Model's response to the prompt
        
    Returns:
        Tuple of (extracted solution as dict, processed string)
    """
    # Limit processing to a reasonable length to avoid memory issues
    max_length = 10000
    if len(solution_str) > max_length:
        solution_str = solution_str[:max_length]
        
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
                # Replace Python expressions with valid JSON
                answer_content = re.sub(r'(\d+)\s*\*\*\s*(\d+)', 
                                       lambda m: str(int(m.group(1)) ** int(m.group(2))), 
                                       answer_content)
                # Remove any trailing commas in objects or arrays
                answer_content = re.sub(r',\s*}', '}', answer_content)
                answer_content = re.sub(r',\s*]', ']', answer_content)
                # Fix unquoted keys
                answer_content = re.sub(r'([{,])\s*(\w+)\s*:', r'\1"\2":', answer_content)
                # Try parsing again
                result = json.loads(answer_content)
            except:
                # If still failing, try ast.literal_eval with safety checks
                try:
                    # Only process if content is reasonably sized to avoid memory issues
                    if len(answer_content) < 1000:
                        result = ast.literal_eval(answer_content)
                    else:
                        print(f"Answer content too large for ast.literal_eval: {len(answer_content)} chars")
                        return {}, processed_str
                except:
                    # Return empty solution if all parsing attempts fail
                    print(f"Failed to parse answer content: {answer_content[:100]}...")
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
            except (ValueError, SyntaxError):
                # Leave as is if can't be parsed
                pass
    
    # If expected_solution is still a string, we need a different comparison strategy
    if isinstance(expected_solution, str):
        if "output" in extracted_solution:
            # Compare string directly with output value
            model_value = normalize_literal(extracted_solution.get("output"))
            normalized_expected = normalize_literal(expected_solution)
            if model_value == normalized_expected:
                return True, "Model's answer matches expected answer"
            else:
                return False, f"Model's answer '{model_value}' does not match expected '{normalized_expected}'"
        else:
            return False, f"Expected output field missing from model's answer"
    
    # Check if keys match (input vs output) for dictionary case
    expected_field = list(expected_solution.keys())[0] if expected_solution else None
    if expected_field not in extracted_solution:
        return False, f"Expected field '{expected_field}' missing from model's answer"
    
    # Get values for comparison
    expected_value = expected_solution.get(expected_field)
    model_value = extracted_solution.get(expected_field)
    
    # Normalize both values for comparison
    normalized_expected = normalize_literal(expected_value)
    normalized_model = normalize_literal(model_value)
    
    # Do a direct equality check
    if normalized_model == normalized_expected:
        return True, "Model's answer matches expected answer"
    else:
        return False, f"Model's answer '{normalized_model}' does not match expected '{normalized_expected}'"

def compute_score(
    solution_str: str, ground_truth: dict, 
    format_reward: int = 1, 
    answer_reward: float = 2.0
) -> float:
    """
    Compute correctness score for a prediction against a reference.
    
    Args:
        solution_str: Model's prediction string
        ground_truth: Reference data including expected solution
        format_reward: Points for format correctness
        answer_reward: Points for answer correctness
        
    Returns:
        Score as a float
    """
    print("\n" + "="*40)
    print(" Evaluating Reason-IO Response ".center(40, '='))
    
    # Extract expected solution and task type with memory efficiency in mind
    task_type = ground_truth.get('task_type', 'unknown')
    expected_solution = None
    expected_field = None
    expected_value = None
    
    # Find solution in a memory-efficient way
    for key in ground_truth:
        if 'solution' in key:
            temp_solution = ground_truth[key]
            
            # Try to parse the solution if it's a string
            if isinstance(temp_solution, str):
                try:
                    expected_solution = json.loads(temp_solution)
                    break
                except json.JSONDecodeError:
                    # Continue trying other fields
                    continue
            elif isinstance(temp_solution, dict):
                expected_solution = temp_solution
                break
    
    # Extract field and value
    if expected_solution and isinstance(expected_solution, dict) and len(expected_solution) > 0:
        expected_field = next(iter(expected_solution.keys()))
        expected_value = expected_solution[expected_field]
    
    # Print ground truth information (shortened for memory efficiency)
    print(f"[Ground Truth]")
    print(f"  Task Type: {task_type}")
    if expected_field:
        print(f"  Expected Field: {expected_field}")
        # Limit output size for large values
        if isinstance(expected_value, str) and len(expected_value) > 100:
            print(f"  Expected Value: {expected_value[:100]}...")
        else:
            print(f"  Expected Value: {expected_value}")
    
    # Extract solution from the model's prediction (already memory-optimized)
    extracted_solution, processed_str = extract_solution(solution_str)
    
    # Print shortened model response for memory efficiency
    print(f"\n[Model Response Preview]")
    response_preview = processed_str[:300] + "..." if len(processed_str) > 300 else processed_str
    print(response_preview)
    
    # Validate the response structure
    format_correct = validate_response_structure(processed_str)
    
    # Calculate format score - use the exact same names as in kk.py for WandB compatibility
    format_score = format_reward if format_correct else -format_reward
    
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    print(f"  Format score: {format_score}")
    
    # Content validation and scoring - match kk.py naming structure
    answer_score = 0
    if not format_correct or not extracted_solution:
        print("\n[Content Validation] Skipped due to format errors or missing answer")
        answer_score = -answer_reward
    else:
        print("\n[Content Validation]")
        
        # Compare extracted solution with expected solution
        is_correct, explanation = compare_solutions(extracted_solution, expected_solution)
        
        if is_correct:
            answer_score = answer_reward
            print(f"  Content validation: CORRECT")
        else:
            answer_score = -answer_reward * 0.75
            print(f"  Content validation: INCORRECT")
            print(f"  Expected: {expected_value}")
            extracted_value = extracted_solution.get(expected_field, "field not found")
            print(f"  Got: {extracted_value}")
    
    # Calculate total score
    total_score = format_score + answer_score
    
    print("\n" + "-"*40)
    print(" Final Score ".center(40, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {answer_score}")
    print(f"  Total: {total_score}")
    print("="*40)
    
    return total_score 