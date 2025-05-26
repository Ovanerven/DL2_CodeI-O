#!/usr/bin/python python3
import warnings
warnings.filterwarnings("ignore")
import re
import os
import json
import time
import argparse
from tqdm import tqdm
import torch

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

def load_dataset(benchmark):
    """Load the appropriate dataset"""
    if benchmark == "aime":
        with open("aime_2021_2024.jsonl", encoding="utf-8") as file:
            data = [json.loads(line) for line in file.readlines() if line]
        return data
    elif benchmark == "amc":
        with open("amc.jsonl", encoding="utf-8") as file:
            data = [json.loads(line) for line in file.readlines() if line]
        return data
    else:
        raise ValueError(f"Unsupported benchmark: {benchmark}")

def build_messages(benchmark, sample):
    """Build messages for the given benchmark and sample"""
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

def extract_answer(response):
    """Extract answer from response"""
    if '<answer>' in response:
        result = re.split(r'<answer>', response)[1]
    else:
        result = response[len(response) - 30:]
    return result

def check_correctness(benchmark, extracted_answer, expected_answer):
    """Check if the extracted answer is correct"""
    if benchmark == "aime":
        return expected_answer in extracted_answer
    elif benchmark == "amc":
        return str(int(expected_answer)) in extracted_answer

def get_expected_answer(benchmark, sample):
    """Get expected answer from sample"""
    return sample["answer"]

def get_question(benchmark, sample):
    """Get question from sample"""
    if benchmark == "aime":
        return sample["question"]
    elif benchmark == "amc":
        return sample["problem"]

def main():
    parser = argparse.ArgumentParser(description="Evaluate math benchmarks (AIME/AMC)")
    parser.add_argument("--model_path", type=str, required=True, help="Path to the model")
    parser.add_argument("--benchmark", type=str, required=True, choices=["aime", "amc"],
                        help="Benchmark to evaluate")
    parser.add_argument("--num_runs", type=int, default=3, help="Number of runs with different seeds")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of samples for testing")
    args = parser.parse_args()
    
    # Determine if we should use vLLM or transformers
    use_vllm = not is_base_model(args.model_path)
    print(f"Using {'vLLM' if use_vllm else 'transformers'} for model loading")
    
    # Load model and tokenizer
    model, tokenizer, is_vllm = load_model_and_tokenizer(args.model_path, use_vllm)
    
    # Load dataset
    data = load_dataset(args.benchmark)
    if args.limit:
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
            text = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            
            expected_answer = get_expected_answer(args.benchmark, sample)
            
            start_time = time.time()
            response = generate_response(model, tokenizer, text, is_vllm=is_vllm, seed=seed)
            time_taken = time.time() - start_time
            
            extracted_answer = extract_answer(response)
            correct = check_correctness(args.benchmark, extracted_answer, expected_answer)
            
            logs.append({
                "run": run_idx + 1,
                "seed": seed,
                "question": get_question(args.benchmark, sample),
                "generated_output": response,
                "extracted_answer": extracted_answer,
                "expected_answer": expected_answer,
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

    # Extract model name for file naming
    if 'global_step' in args.model_path:
        # Fine-tuned model - use timestamp folder
        model_name = args.model_path.split("/")[-3]
    else:
        # Base model - use last folder name
        model_name = args.model_path.split("/")[-1]
    
    # Create output directory if it doesn't exist
    os.makedirs("math_eval_logs", exist_ok=True)
    
    # Save individual run logs
    with open(f"math_eval_logs/{model_name}_{args.benchmark}_all_runs_log.json", "w") as f:
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
    
    with open(f"math_eval_logs/{model_name}_{args.benchmark}_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nResults saved to math_eval_logs/{model_name}_{args.benchmark}_*")

if __name__ == "__main__":
    main() 