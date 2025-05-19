import re
import json
import ast
from typing import Dict, Any, Optional, Tuple


def normalize_literal(s: str):
    """
    Turn a string like '{"x":1,"y":1}', "{'x':2,'y':1}", '"False"', 'False', etc.
    into the corresponding Python object: dict, bool, int, etc.
    """
    if not isinstance(s, str):
        return s
    # Try JSON first
    try:
        val = json.loads(s)
    except (ValueError, json.JSONDecodeError):
        try:
            val = ast.literal_eval(s)
        except (ValueError, SyntaxError):
            val = s
    if isinstance(val, str) and val.lower() in ('true', 'false'):
        return val.lower() == 'true'
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


def extract_solution(solution_str: str) -> Tuple[Optional[Dict[str, Any]], str]:
    """
    Extracts answer dict and cleaned assistant output
    """
    if "<|endoftext|>" in solution_str:
        solution_str = solution_str.split("<|endoftext|>")[0]
    if "Assistant:" in solution_str:
        processed = solution_str.split("Assistant:", 1)[1]
    elif "<|im_start|>assistant" in solution_str:
        processed = solution_str.split("<|im_start|>assistant", 1)[1]
    else:
        processed = solution_str
    answer_dict = extract_json_answer(processed)
    return answer_dict, processed


def validate_response_structure(processed_str: str) -> bool:
    """
    Check for <think>...</think><answer>...</answer> tags in correct order
    """
    print("\n[Structure Validation]")
    validation = True
    tags = {
        'think_start': ('<think>', 1),
        'think_end':   ('</think>', 1),
        'answer_start':('<answer>', 1),
        'answer_end':  ('</answer>', 1)
    }
    pos = {}
    for name, (tag, expected) in tags.items():
        cnt = processed_str.count(tag)
        pos[name] = processed_str.find(tag)
        print(f"  {tag}: count={cnt}, pos={pos[name]}")
        if cnt != expected:
            print(f"  [Error] {tag} appears {cnt} times (expected {expected})")
            validation = False
    if pos['think_start'] > pos['think_end'] or \
       pos['think_end'] > pos['answer_start'] or \
       pos['answer_start'] > pos['answer_end']:
        print("  [Error] Incorrect tag order")
        validation = False
    return validation


def compute_score(solution_str: str,
                  ground_truth: Dict[str, Any],
                  format_reward: int = 1,
                  answer_reward: float = 2.0) -> float:
    """
    Unified scoring: format + content
    """
    print("\n" + "="*80)
    print(" Evaluating Reason-IO Response ".center(80, '='))
    # --- Ground truth parsing ---
    # solution is JSON string or dict
    sol_field = ground_truth.get('solution')
    if isinstance(sol_field, str):
        try:
            expected = json.loads(sol_field)
        except:
            expected = {}
    else:
        expected = sol_field or {}
    task_type = ground_truth.get('task_type', 'unknown')
    # expected field and value
    expected_field = next(iter(expected), None)
    expected_value = expected.get(expected_field) if expected_field else None
    print(f"[Ground Truth] field={expected_field}, value={expected_value}")
    # --- Model response ---
    answer_dict, processed = extract_solution(solution_str)
    print(f"\n[Model Response]\n{processed}")
    # --- Structure validation ---
    fmt_ok = validate_response_structure(processed)
    fmt_score = format_reward if fmt_ok else -abs(format_reward)
    print(f"\n  Format: {'PASS' if fmt_ok else 'FAIL'} => {fmt_score}")
    # --- Content validation ---
    ans_score = 0
    if fmt_ok and answer_dict is not None and expected_field:
        model_val = answer_dict.get(expected_field)
        print(f"\n[Content Validation] Model {expected_field}={model_val}")
        norm_exp = normalize_literal(expected_value)
        norm_mod = normalize_literal(model_val)
        if norm_mod == norm_exp:
            ans_score = answer_reward
            print("  Content: CORRECT")
        else:
            ans_score = -1.5
            print(f"  Content: INCORRECT (expected {norm_exp})")
    else:
        ans_score = -2
        print("  Content: SKIPPED or MISSING")
    total = fmt_score + ans_score
    print("\n" + "-"*80)
    print(f" Final Score: format={fmt_score}, answer={ans_score}, total={total}")
    print("="*80 + "\n")
    return total