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


def build_prompt(benchmark, sample):
    if benchmark == "gsm8k":
        return f"Answer the following math question step by step:\n{sample['question']}"
    elif benchmark == "winogrande":
        return (
            f"Choose the correct option to complete the sentence: {sample['sentence']}\n"
            f"Option 1: {sample['option1']}\nOption 2: {sample['option2']}"
        )
    elif benchmark == "humaneval":
        return f"Complete the following Python function:\n\n{sample['prompt']}"
    else:
        raise ValueError("Unknown benchmark")
    

def get_answer(benchmark, sample):
    if benchmark in ["gsm8k", "winogrande"]:
        return sample["answer"]
    elif benchmark == "humaneval":
        return sample["entry_point"]
    else:
        raise ValueError("Unknown benchmark")
    

def extract_numeric_answer(text):
    numbers = re.findall(r"[-+]?\d*\.\d+|\d+", text)
    return numbers[-1] if numbers else ""


def extract_function_def(response):
    match = re.search(r"(def [\s\S]+)", response)
    return match.group(1).strip() if match else ""


def function_defined(predicted_code, expected_func):
    pattern = rf"def\s+{re.escape(expected_func)}\s*\("
    return bool(re.search(pattern, predicted_code))


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
    total_time = 0

    for sample in tqdm(data):
        prompt = build_prompt(args.benchmark, sample)
        expected_answer = get_answer(args.benchmark, sample)

        messages = [
            {"role": "system", "content": "You are a helpful assistant. Think step-by-step before answering. Enclose your reasoning within the <think> tags and your final answer within the <answer> tags."},
            {"role": "user", "content": prompt}
        ]

        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        start = time.time()
        output = llm.generate([text], sampling_params=sampling_params)
        time_taken = time.time() - start
        response = output[0].outputs[0].text.strip()

        # benchmark specific extraction logic
        if args.benchmark == "gsm8k":
            prediction = extract_numeric_answer(response)
            expected_clean = extract_numeric_answer(str(expected_answer))
            correct = prediction == expected_clean

        elif args.benchmark == "winogrande":
            prediction = "1" if "option 1" in response.lower() else "2" if "option 2" in response.lower() else ""
            expected_clean = str(expected_answer).strip()
            correct = prediction == expected_clean

        elif args.benchmark == "humaneval":
            # For humaneval, we assume the response is a Python function
            # and we will execute it to check if it matches the expected answer
            expected_clean = str(expected_answer).strip()
            cleaned_response = re.sub(r"<.*?>", "", response)
            predicted_code = extract_function_def(cleaned_response)
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

        if correct:
            correct_cnt += 1
        total_time += time_taken
    
    accuracy = correct_cnt / len(data)
    print(f"{args.benchmark} accuracy: {accuracy:.4f}")
    print(f"Average time taken: {total_time / len(data):.4f} seconds")
    print(f"Total time taken: {total_time:.4f} seconds")

if __name__ == "__main__":
    main()