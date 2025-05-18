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
        Boolean indicating whether all formatting requirements are met
    """
    print("\n[Structure Validation]")
    validation_passed = True

    # Check required tags - all tags are now required like in kk.py
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

def compare_solutions(extracted_solution: Dict[str, Any], expected_solution: Dict[str, Any]) -> Tuple[bool, str]:
    """
    Compare the extracted solution with the expected solution.
    
    Args:
        extracted_solution: The solution extracted from the model's response
        expected_solution: The expected solution
        
    Returns:
        Tuple of (is_correct, explanation)
    """
    # Check if keys match (input vs output)
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
    
    # Extract solution from the model's prediction
    extracted_solution, processed_str = extract_solution(solution_str)
    
    # Print the full model response
    print(f"\n[Full Model Response]")
    print(processed_str)
    
    # Validate the response structure (required tags)
    format_correct = validate_response_structure(processed_str)
    format_score = 1.0 if format_correct else 0.0
    print(f"\n  Format validation: {'PASS' if format_correct else 'FAIL'}")
    
    # Check if we got a structured solution from the model
    if extracted_solution is None:
        print("  [Error] Failed to extract solution structure")
        return 0.0
    
    if not format_correct:
        print("  [Error] Response format validation failed")
        return 0.0
    
    # Get expected solution from the reference
    expected_solution = ground_truth.get("solution", {})
    
    if not expected_solution:
        print("[Error] Expected solution not found in reference data")
        return 0.0
    
    # Compare extracted solution with expected
    is_correct, explanation = compare_solutions(extracted_solution, expected_solution)
    answer_score = 2.0 if is_correct else -1.5
    
    print(f"\n[Content Validation]")
    print(f"  Expected: {expected_solution}")
    print(f"  Extracted: {extracted_solution}")
    print(f"  {'CORRECT' if is_correct else 'INCORRECT'}: {explanation}")
    
    # Calculate total score (format + answer)
    total_score = format_score + (answer_score if format_correct else 0)
    
    print("\n" + "-"*80)
    print(f" Final Score ".center(80, '-'))
    print(f"  Format: {format_score}")
    print(f"  Answer: {answer_score if format_correct else 0}")
    print(f"  Total: {total_score}")
    print("="*80 + "\n")
    
    # Return score
    return total_score
