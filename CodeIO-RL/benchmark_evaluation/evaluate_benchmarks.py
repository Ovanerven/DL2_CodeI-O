import argparse
import json
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

    elif benchmark == "winogrande":
        return [
            {"role": "system", "content": (
                "You are a helpful assistant. Think step-by-step inside the <think> </think> tags. "
                "Then provide your final choice word-for-word inside the <answer> </answer> tags. "
                "The <answer> tag should contain only the answer, like <answer>Sarah</answer>."
            )},
            {"role": "user", "content": (
                f"Choose the correct option to complete the sentence: {sample['sentence']}\n"
                f"Option 1: {sample['option1']}\nOption 2: {sample['option2']}"
            )}
        ]
    
    elif benchmark == "humaneval":
        return [
            {"role": "system", "content": (
                "You are a helpful coding assistant. Think through the solution inside the <think> </think> tags, "
                "and place the full function code inside the <answer> </answer> tags. Do not include explanations outside the tags."
            )},
            {"role": "user", "content": f"Complete the following Python function:\n\n{sample['prompt']}"}
        ]
    
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


def extract_winogrande_answer(response, option1, option2):
    match = re.search(r"<answer>\s*(.*?)\s*</answer>", response, re.IGNORECASE | re.DOTALL)
    if not match:
        return ""
    answer = match.group(1).strip().lower()
    answer = re.sub(r"option\s*\d:\s*", "", answer).strip()
    if answer.lower() == option1.lower():
        return "1"
    elif answer.lower() == option2.lower():
        return "2"
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
    args = parser.parse_args()
    
    llm = LLM(
        model=args.model_path,
        tensor_parallel_size=1,
        dtype="bfloat16",
        trust_remote_code=True,
        max_num_seqs=4,
        max_model_len=20000
    )

    sampling_params = SamplingParams(
        max_tokens=10000,
        temperature=0.8,
        top_p=0.95
    )

    data = load_benchmark(args.benchmark)
    tokenizer = llm.get_tokenizer()

    correct_cnt = 0
    curr = 0
    total_time = 0
    logs = []

    for sample in tqdm(data):
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
            option1 = sample["option1"]
            option2 = sample["option2"]
            prediction = extract_winogrande_answer(response, option1, option2)
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
            print(f"Accuracy after {curr} samples: {correct_cnt / curr:.4f}")

        total_time += time_taken
    
    accuracy = correct_cnt / len(data)
    print(f"{args.benchmark} accuracy: {accuracy:.4f}")
    print(f"Average time taken: {total_time / len(data):.4f} seconds")
    print(f"Total time taken: {total_time:.4f} seconds")

    model = args.model_path.split("/")[-1]
    with open(f"benchmark_logs/{model}_{args.benchmark}_log.json", "w") as f:
        json.dump(logs, f, indent=2)

if __name__ == "__main__":
    main()
