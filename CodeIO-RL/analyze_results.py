import json
import os
import sys
from typing import Dict, List, Any

def load_json_file(file_path: str) -> Dict[str, Any]:
    """Load a JSON file and return its contents"""
    with open(file_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def calculate_stats(results_data: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate statistics from the results data"""
    total_correct = 0
    total_problems = 0
    
    # Count by type
    type_correct = {'deductive': 0, 'abductive': 0, 'inductive': 0}
    type_total = {'deductive': 0, 'abductive': 0, 'inductive': 0}
    
    for result in results_data['results']:
        task_type = result['task_type']
        is_correct = result['is_correct']
        
        total_problems += 1
        type_total[task_type] += 1
        
        if is_correct:
            total_correct += 1
            type_correct[task_type] += 1
    
    # Calculate accuracies
    overall_accuracy = total_correct / total_problems if total_problems > 0 else 0
    type_accuracy = {}
    for t in type_total:
        type_accuracy[t] = type_correct[t] / type_total[t] if type_total[t] > 0 else 0
    
    return {
        'overall_accuracy': overall_accuracy,
        'total_correct': total_correct,
        'total_problems': total_problems,
        'type_accuracy': type_accuracy,
        'type_correct': type_correct,
        'type_total': type_total
    }

def display_stats(stats: Dict[str, Any], title: str):
    """Display the calculated statistics"""
    print(f"\n{title}")
    print("-" * len(title))
    print(f"Overall accuracy: {stats['overall_accuracy'] * 100:.2f}% ({stats['total_correct']}/{stats['total_problems']})")
    
    for task_type in ['deductive', 'abductive', 'inductive']:
        acc = stats['type_accuracy'][task_type]
        correct = stats['type_correct'][task_type]
        total = stats['type_total'][task_type]
        print(f"{task_type.capitalize()} accuracy: {acc * 100:.2f}% ({correct}/{total})")

def compare_results(file1: str, file2: str, label1: str = "Basic Prompts", label2: str = "Structured Prompts"):
    """Compare results from two different evaluation runs"""
    # Load both result files
    data1 = load_json_file(file1)
    data2 = load_json_file(file2)
    
    # Calculate stats for both
    stats1 = calculate_stats(data1)
    stats2 = calculate_stats(data2)
    
    # Display individual stats
    display_stats(stats1, f"Results with {label1}")
    display_stats(stats2, f"Results with {label2}")
    
    # Display comparison
    print("\nImprovement Analysis")
    print("-" * 20)
    
    # Overall improvement
    overall_diff = stats2['overall_accuracy'] - stats1['overall_accuracy']
    print(f"Overall accuracy change: {overall_diff * 100:+.2f}% points")
    
    # Improvement by type
    for task_type in ['deductive', 'abductive', 'inductive']:
        type_diff = stats2['type_accuracy'][task_type] - stats1['type_accuracy'][task_type]
        print(f"{task_type.capitalize()} accuracy change: {type_diff * 100:+.2f}% points")

if __name__ == "__main__":
    # Check if file paths are provided as command-line arguments
    if len(sys.argv) >= 3:
        basic_results_file = sys.argv[1]
        structured_results_file = sys.argv[2]
    else:
        # Default file paths
        basic_results_file = os.path.join("codeIO_difficulty", "basic_prompts_results.json")
        structured_results_file = os.path.join("codeIO_difficulty", "structured_prompts_results.json")
    
    # Ensure files exist
    if not os.path.exists(basic_results_file):
        print(f"Error: {basic_results_file} not found.")
        sys.exit(1)
    
    if not os.path.exists(structured_results_file):
        print(f"Error: {structured_results_file} not found.")
        sys.exit(1)
    
    # Compare the results
    compare_results(basic_results_file, structured_results_file) 