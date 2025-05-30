import argparse
import json
import os
import re
import time
import warnings
from tqdm import tqdm
import torch
from datasets import load_dataset

def is_base_model(model_path):
    """Determine if this is a base model or fine-tuned model based on path structure"""
    # Base models typically don't have 'global_step' in their path
    return 'global_step' not in model_path

def load_model_and_tokenizer(model_path, use_vllm=True):
    """Load model and tokenizer, choosing between vLLM and transformers based on model type"""
    if use_vllm:
        from vllm import LLM, SamplingParams
        llm = LLM(
            model=model_path,
            tensor_parallel_size=1,
            dtype="bfloat16",
            trust_remote_code=True,
            max_num_seqs=4,
            max_model_len=20000
        )
        return llm, llm.get_tokenizer(), True
    else:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True
        )
        return model, tokenizer, False

def generate_response(model, tokenizer, text, sampling_params=None, is_vllm=True, seed=42):
    """Generate response using either vLLM or transformers"""
    if is_vllm:
        from vllm import SamplingParams
        if sampling_params is None:
            sampling_params = SamplingParams(
                max_tokens=10000,
                temperature=0.8,
                top_p=0.95,
                seed=seed
            )
        outputs = model.generate([text], sampling_params)
        return outputs[0].outputs[0].text.strip()
    else:
        # Set seed for transformers
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            
        inputs = tokenizer(text, return_tensors="pt", return_attention_mask=True).to(model.device)
        generation_output = model.generate(
            **inputs,
            max_new_tokens=10000,
            temperature=0.8,
            top_p=0.95,
            do_sample=True
        )
        response = tokenizer.decode(generation_output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        return response.strip()

def load_benchmark(benchmark):
    if benchmark == "gsm8k":
        return load_dataset("gsm8k", "main", split="test")
    elif benchmark == "winogrande":
        return load_dataset("winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
    elif benchmark == "humaneval":
        return load_dataset("openai_humaneval", split="test")
    elif benchmark == "aime":
        with open("aime_2021_2024.jsonl", encoding="utf-8") as file:
            data = [json.loads(line) for line in file.readlines() if line]
        return data
    elif benchmark == "amc":
        with open("amc.jsonl", encoding="utf-8") as file:
            data = [json.loads(line) for line in file.readlines() if line]
        return data
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
    
    elif benchmark in ["aime", "amc"]:
        system_content = (
            "You are a helpful assistant. The assistant first thinks about the reasoning process "
            "in the mind and then provides the user with the answer. The reasoning process and answer "
            "are enclosed within <think> </think> and<answer> </answer> tags, respectively, "
            "i.e., <think> reasoning process here </think><answer> answer here </answer>. "
            "Now the user asks you to solve a math problem. After thinking, when you "
            "finally reach a conclusion, clearly state the answer within <answer> </answer> tags. "
            "i.e., <answer> (\\boxed{}\\) </answer>."
        )
        
        if benchmark == "aime":
            user_content = sample["question"]
        elif benchmark == "amc":
            user_content = sample["problem"]
        
        return [
            {"role": "system", "content": system_content},
            {"role": "user", "content": user_content}
        ]
    
    else:
        raise ValueError("Unsupported benchmark")
    

def get_answer(benchmark, sample):
    if benchmark == "gsm8k":
        return sample["answer"]
    elif benchmark == "winogrande":
        return sample["answer"]
    elif benchmark == "humaneval":
        return sample["entry_point"]
    elif benchmark in ["aime", "amc"]:
        return sample["answer"]
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


def extract_math_answer(response):
    """Extract answer from math benchmark responses"""
    if '<answer>' in response:
        result = re.split(r'<answer>', response)[1]
    else:
        result = response[len(response) - 30:]
    return result


def check_math_correctness(benchmark, extracted_answer, expected_answer):
    """Check if the extracted answer is correct for math benchmarks"""
    if benchmark == "aime":
        return expected_answer in extracted_answer
    elif benchmark == "amc":
        return str(int(float(expected_answer))) in extracted_answer


def function_defined(code, expected_func):
    pattern = rf"(?<![a-zA-Z0-9_])def\s+{re.escape(expected_func)}\s*\("
    return bool(re.search(pattern, code))


def get_model_name(model_path):
    """Extract model name for file naming"""
    if 'global_step' in model_path:
        # Fine-tuned model - use timestamp folder
        return model_path.split("/")[-3]
    else:
        # Base model - use last folder name
        return model_path.split("/")[-1]


def main():
    parser = argparse.ArgumentParser(description="Evaluate benchmarks")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--benchmark", type=str, required=True, 
                        choices=["gsm8k", "winogrande", "humaneval", "aime", "amc"],
                        help="Benchmark to evaluate")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of runs with different seeds")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    args = parser.parse_args()
    
    # Determine if we should use vLLM or transformers
    use_vllm = not is_base_model(args.model_path)
    print(f"Using {'vLLM' if use_vllm else 'transformers'} for model loading")
    
    # Load model and tokenizer
    model, tokenizer, is_vllm = load_model_and_tokenizer(args.model_path, use_vllm)
    
    data = load_benchmark(args.benchmark)
    if args.limit and args.benchmark in ["aime", "amc"]:
        data = data[:args.limit]
        print(f"Limited to {args.limit} samples for testing")
    
    # Store results across all runs
    all_accuracies = []
    all_logs = []
    
    for run_idx in range(args.num_runs):
        print(f"\n=== Run {run_idx + 1}/{args.num_runs} ===")
        
        # Different seed for each run
        seed = 42 + run_idx
        
        correct_cnt = 0
        curr = 0
        total_time = 0
        logs = []

        for sample in tqdm(data, desc=f"Run {run_idx + 1}"):
            messages = build_messages(args.benchmark, sample)
            text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            expected_answer = get_answer(args.benchmark, sample)
            
            start = time.time()
            if is_vllm:
                # For vLLM, use the original method
                from vllm import SamplingParams
                sampling_params = SamplingParams(
                    max_tokens=10000,
                    temperature=0.8,
                    top_p=0.95,
                    seed=seed
                )
                outputs = model.generate([text], sampling_params)
                response = outputs[0].outputs[0].text.strip()
            else:
                # For transformers, use the generate_response function
                response = generate_response(model, tokenizer, text, is_vllm=False, seed=seed)
            time_taken = time.time() - start

            # Benchmark specific extraction and correctness logic
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
                expected_clean = str(expected_answer).strip()
                predicted_code = extract_humaneval_answer(response)

                try:
                    compile(predicted_code, "<string>", "exec")
                    correct = function_defined(predicted_code, expected_clean)
                except SyntaxError as e:
                    print(f"Error executing generated code: {e}")
                    correct = False

                prediction = predicted_code
                
            elif args.benchmark in ["aime", "amc"]:
                prediction = extract_math_answer(response)
                expected_clean = str(expected_answer).strip()
                correct = check_math_correctness(args.benchmark, prediction, expected_clean)
            
            else:
                correct = False
                prediction = ""
                expected_clean = ""

            logs.append({
                "run": run_idx + 1,
                "seed": seed,
                "prompt": messages[-1]["content"],
                "expected_answer": expected_answer,
                "expected_clean": expected_clean,
                "response": response,
                "predicted_answer": prediction,
                "correct": correct,
                "time_taken": time_taken
            })

            curr += 1
            if correct:
                correct_cnt += 1

            if curr % 50 == 0:
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

    # Get model name for file naming
    model_name = get_model_name(args.model_path)
    
    # Create logs directory if it doesn't exist
    os.makedirs("benchmark_evaluation/logs", exist_ok=True)
    
    # Save individual run logs
    with open(f"benchmark_evaluation/logs/{model_name}_{args.benchmark}_all_runs_log.json", "w") as f:
        json.dump(all_logs, f, indent=2)
    
    # Save summary statistics
    summary = {
        "model": model_name,
        "model_path": args.model_path,
        "benchmark": args.benchmark,
        "num_runs": args.num_runs,
        "num_samples": len(data),
        "individual_accuracies": all_accuracies,
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "min_accuracy": min(all_accuracies),
        "max_accuracy": max(all_accuracies),
        "used_vllm": is_vllm
    }
    
    with open(f"benchmark_evaluation/logs/{model_name}_{args.benchmark}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to benchmark_evaluation/logs/{model_name}_{args.benchmark}_*")

if __name__ == "__main__":
    main()
