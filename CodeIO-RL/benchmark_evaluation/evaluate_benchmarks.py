import argparse
import json
import os
import re
import time
from tqdm import tqdm
from datasets import load_dataset
from vllm import LLM, SamplingParams


def load_benchmark(benchmark):
    if benchmark == "gsm8k":
        return load_dataset("gsm8k", "main", split="test")
    elif benchmark == "winogrande":
        return load_dataset("winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
    elif benchmark == "humaneval":
        return load_dataset("openai_humaneval", split="test")
    else:
        raise ValueError("Unsupported benchmark")


def build_messages(benchmark, sample):
    if benchmark == "gsm8k":
        return [
            {"role": "system", "content": (
                "You are a helpful assistant. Solve the math problem step-by-step. "
                "Put all intermediate reasoning inside <think> tags. "
                "Place the final numerical answer only (no units, no punctuation) inside <answer> tags. "
                "The <answer> tag should contain only the number, like <answer>540</answer>."
            )},
            {"role": "user", "content": f"Answer the following math question step by step:\n{sample['question']}"}
        ]

    # elif benchmark == "winogrande":
    #     return [
    #         {"role": "system", "content": (
    #             "You are a helpful assistant. Think step-by-step inside the <think> </think> tags. "
    #             "Then provide your final choice as either '1' or '2' inside the <answer> </answer> tags. "
    #             "The <answer> tag should contain only the number, like <answer>1</answer> or <answer>2</answer>."
    #         )},
    #         {"role": "user", "content": (
    #             f"Choose the correct option to complete the sentence:\n\n"
    #             f"{sample['sentence']}\n\n"
    #             f"1. {sample['option1']}\n"
    #             f"2. {sample['option2']}\n\n"
    #             f"Answer with 1 (for option 1) or 2 (for option 2):"
    #         )}
    #     ]

    elif benchmark == "winogrande":
        system_content = (
            "You are a helpful assistant. The assistant first thinks about the reasoning process "
            "in the mind and then provides the user with the answer. The reasoning process and answer "
            "are enclosed within <think> </think> and <answer> </answer> tags, respectively, "
            "i.e., <think> reasoning process here </think><answer> answer here </answer>. "
            "Now the user asks you to solve a logical reasoning problem. After thinking, when you "
            "finally reach a conclusion, clearly state your choice as either 1 or 2 within "
            "<answer> </answer> tags. i.e., <answer>1</answer> or <answer>2</answer>."
        )
        
        user_content = (
            f"You are given a sentence completion task. Choose the correct option that best "
            f"completes the sentence.\n\n"
            f"Sentence: {sample['sentence']}\n\n"
            f"Options:\n"
            f"Option 1. {sample['option1']}\n"
            f"Option 2. {sample['option2']}\n\n"
            f"So which option correctly completes the sentence? Think, and then answer with <answer>1</answer> (for option 1) or <answer>2</answer> (for option 2):"
        )
        
        # Match exact training format with <think> priming
        combined_content = (
            f"<|im_start|>system\n{system_content}\n<|im_end|>\n"
            f"<|im_start|>user\n{user_content}\n<|im_end|>\n"
            f"<|im_start|>assistant\n<think>"
        )
        
        return [{"role": "user", "content": combined_content}]
    
    # elif benchmark == "humaneval":
    #     return [
    #         {"role": "system", "content": (
    #             "You are a helpful coding assistant. Think through the solution inside the <think> </think> tags, "
    #             "and place the full function code inside the <answer> </answer> tags. Do not include explanations outside the tags."
    #         )},
    #         {"role": "user", "content": f"Complete the following Python function:\n\n{sample['prompt']}"}
    #     ]
    
    else:
        raise ValueError("Unsupported benchmark")
    

def get_answer(benchmark, sample):
    if benchmark in ["gsm8k", "winogrande"]:
        return sample["answer"]
    elif benchmark == "humaneval":
        return sample["entry_point"]
    else:
        raise ValueError("Unknown benchmark")
    

def extract_gsm8k_answer(response):
    match = re.search(r"<answer>\s*([$]?\s*\d[\d,]*)\s*</answer>", response, re.IGNORECASE)
    if match:
        raw = match.group(1)
        clean = re.sub(r"[^\d]", "", raw)  # remove everything except digits
        return clean
    return ""


def extract_winogrande_answer(response):
    """Extract 1 or 2 from <answer> tags"""
    match = re.search(r"<answer>\s*([12])\s*</answer>", response, re.IGNORECASE)
    if match:
        return match.group(1)
    
    # Fallback: look for just 1 or 2 in the response
    match = re.search(r"<answer>\s*.?([12]).?\s*</answer>", response, re.IGNORECASE)
    if match:
        return match.group(1)
    
    return ""


def extract_humaneval_answer(response):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.DOTALL | re.IGNORECASE)
    return match.group(1).strip() if match else ""


def function_defined(code, expected_func):
    pattern = rf"(?<![a-zA-Z0-9_])def\s+{re.escape(expected_func)}\s*\("
    return bool(re.search(pattern, code))


def main():
    parser = argparse.ArgumentParser(description="Evaluate benchmarks")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--benchmark", type=str, required=True, choices=["gsm8k", "winogrande", "humaneval"],
                        help="Benchmark to evaluate")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of runs with different seeds")
    args = parser.parse_args()
    
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        max_num_seqs=4,
        max_model_len=20000
    )

    data = load_benchmark(args.benchmark)
    tokenizer = llm.get_tokenizer()
    
    # Store results across all runs
    all_accuracies = []
    all_logs = []
    
    for run_idx in range(args.num_runs):
        print(f"\n=== Run {run_idx + 1}/{args.num_runs} ===")
        
        # Different seed for each run
        seed = 42 + run_idx
        sampling_params = SamplingParams(
            max_tokens=10000,
            temperature=0.8,
            top_p=0.95,
            seed=seed
        )

        correct_cnt = 0
        curr = 0
        total_time = 0
        logs = []

        for sample in tqdm(data, desc=f"Run {run_idx + 1}"):
            messages = build_messages(args.benchmark, sample)
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            expected_answer = get_answer(args.benchmark, sample)
            
            start = time.time()
            output = llm.generate([text], sampling_params=sampling_params)
            time_taken = time.time() - start
            response = output[0].outputs[0].text.strip()

            # benchmark specific extraction logic
            if args.benchmark == "gsm8k":
                prediction = extract_gsm8k_answer(response)
                match = re.search(r"####\s*(\d+)", str(expected_answer))
                expected_clean = match.group(1) if match else ""
                correct = prediction == expected_clean

            elif args.benchmark == "winogrande":
                prediction = extract_winogrande_answer(response)
                expected_clean = str(expected_answer).strip()
                correct = prediction == expected_clean

            elif args.benchmark == "humaneval":
                # For humaneval, we assume the response is a Python function
                # and we will execute it to check if it matches the expected answer
                expected_clean = str(expected_answer).strip()
                predicted_code = extract_humaneval_answer(response)

                try:
                    compile(predicted_code, "<string>", "exec")
                    correct = function_defined(predicted_code, expected_clean)
                except SyntaxError as e:
                    print(f"Error executing generated code: {e}")
                    correct = False

                prediction = predicted_code
            
            else:
                correct = False
                prediction = ""

            logs.append({
                "run": run_idx + 1,
                "seed": seed,
                "prompt": messages[-1]["content"],
                "expected_answer": expected_answer,
                "expected_clean": expected_clean,
                "response": response,
                "predicted_answer": prediction,
                "correct": correct
            })

            curr += 1
            if correct:
                correct_cnt += 1

            if curr % 100 == 0:
                print(f"Run {run_idx + 1} - Accuracy after {curr} samples: {correct_cnt / curr:.4f}")

            total_time += time_taken
        
        accuracy = correct_cnt / len(data)
        all_accuracies.append(accuracy)
        all_logs.extend(logs)
        
        print(f"Run {run_idx + 1} - {args.benchmark} accuracy: {accuracy:.4f}")
        print(f"Run {run_idx + 1} - Average time taken: {total_time / len(data):.4f} seconds")
        print(f"Run {run_idx + 1} - Total time taken: {total_time:.4f} seconds")
    
    # Calculate and report final statistics
    mean_accuracy = sum(all_accuracies) / len(all_accuracies)
    std_accuracy = (sum((acc - mean_accuracy) ** 2 for acc in all_accuracies) / len(all_accuracies)) ** 0.5
    
    print(f"\n=== Final Results ===")
    print(f"{args.benchmark} - Individual accuracies: {[f'{acc:.4f}' for acc in all_accuracies]}")
    print(f"{args.benchmark} - Mean accuracy: {mean_accuracy:.4f} ± {std_accuracy:.4f}")
    print(f"{args.benchmark} - Min accuracy: {min(all_accuracies):.4f}")
    print(f"{args.benchmark} - Max accuracy: {max(all_accuracies):.4f}")

    model = args.model_path.split("/")[-3]
    
    # Create benchmark_logs directory if it doesn't exist
    os.makedirs("benchmark_logs", exist_ok=True)
    
    # Save individual run logs
    with open(f"benchmark_logs/{model}_{args.benchmark}_all_runs_log.json", "w") as f:
        json.dump(all_logs, f, indent=2)
    
    # Save summary statistics
    summary = {
        "model": model,
        "benchmark": args.benchmark,
        "num_runs": args.num_runs,
        "individual_accuracies": all_accuracies,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "min_accuracy": min(all_accuracies),
        "max_accuracy": max(all_accuracies)
    }
    
    with open(f"benchmark_logs/{model}_{args.benchmark}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

if __name__ == "__main__":
    main()
