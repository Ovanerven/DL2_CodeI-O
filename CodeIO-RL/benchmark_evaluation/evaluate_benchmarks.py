import argparse
import torch
import re
import os
import tempfile
from tqdm import tqdm
import subprocess
from transformers import AutoTokenizer, AutoModelForCausalLM
from datasets import load_dataset

def load_model(model_path):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=torch.float16).to("cuda")
    model.eval()
    return tokenizer, model


def evaluate_gsm8k(model, tokenizer):
    print("Evaluating GSM8K...")
    dataset = load_dataset("gsm8k", "main", split="test")
    correct = 0

    for item in tqdm(dataset):
        question = item["question"]
        gt_answer = item["answer"]
        prompt = f"Q: {question}\nA: Let's think step-by-step.\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=256)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)

        match = re.search(  
            r"(?:the\s+)?(?:answer|result|output)\s+is\s+(-?\d+)", 
            decoded, 
            re.IGNORECASE
        )

        if match:
            pred = match.group(1)
        else:
            nums = re.findall(f"[-+]?\d+", decoded)
            pred = nums[-1] if nums else None # taking the last number as the answer
        
        if pred and pred in gt_answer:
            correct += 1

    acc = correct / len(dataset)
    print(f"[GSM8K] Accuracy = {acc:.3%}")


def evaluate_winogrande(model, tokenizer):
    print("Evaluating WinoGrande...")
    dataset = load_dataset("winogrande", "winogrande_xl", split="validation", trust_remote_code=True)
    correct = 0

    for item in tqdm(dataset):
        sentence = item["sentence"]
        opt1 = item["option1"]
        opt2 = item["option2"]
        gt_answer = item["answer"] # "1" or "2"

        prompt = (
            f"{sentence}\n"
            f"Options:\nA: {opt1}\nB: {opt2}"
            f"Which option best fits the blank? Answer with A or B:\nAnswer:"
        )

        # # generate code from prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=16, temperature=0.3)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True).lower()

        pred = None
        if "a" in decoded:
            pred = "1"
        elif "b" in decoded:
            pred = "2"

        if pred == gt_answer:
            correct += 1
    
    acc = correct / len(dataset)
    print(f"[WinoGrande] Accuracy = {acc:.3%}")


def sanitizeTripleQuotes(code):
    triple_quotes = ["'''", '"""']
    for quote in triple_quotes:
        parts = code.split(quote)
        if len(parts) % 2 == 1:
            return quote.join(parts[:-1])
    return code


def safe_execute_test(candidate_code, test_code):
    try:
        namespace = {}
        compiled_code = compile(candidate_code + "\n" + test_code, "<string>", "exec")
        exec(compiled_code, namespace)
        return True
    except Exception as e:
        print(f"Execution error: {e}")
        print("Generated code:\n", candidate_code)
        print("=" * 40)
        return False


def evaluate_humaneval(model, tokenizer):
    print("Evaluating HumanEval...")
    dataset = load_dataset("openai_humaneval", split="test")
    correct = 0

    for item in tqdm(dataset):
        prompt = item["prompt"]
        test_case = item["test"]

        # generate code from prompt
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=256, temperature=0.2)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)

        decoded = decoded.strip().split("\n\n")[0].strip()
        decoded = sanitizeTripleQuotes(decoded)

        success = safe_execute_test(decoded, test_case)
        if success:
            correct += 1

    acc = correct / len(dataset)
    print(f"[HumanEval] Accuracy = {acc:.3%}")


def evaluate_aime(model, tokenizer):
    """
    Evaluate on the AIME subset (AIME I & II) of the E2H-AMC benchmark.
    """
    print("Evaluating AIME...")
    # load the full AMC/AIME collection and pick out AIME problems
    ds = load_dataset(
        "furonghuang-lab/Easy2Hard-Bench",
        "E2H-AMC",
        split="eval"
    )

    aime_ds = ds.filter(lambda x: "AIME" in x["contest"])
    correct = 0

    for item in tqdm(aime_ds):
        problem = item["problem"]
        gt_answer = str(item["answer"])

        # prompt with chain-of-thought
        prompt = f"Q: {problem}\nA: Let's think step-by-step.\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=4096, temperature=0.3)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)

        # extract the first integer seen
        match = re.search(r"[-+]?\d+", decoded)
        pred = match.group(0) if match else None

        print("\n" + "="*80)
        print(f"PROBLEM: {problem}")
        print("-"*80)
        print(f"MODEL RESPONSE: {decoded}")
        print("-"*80)
        print(f"EXTRACTED ANSWER: {pred}")
        print(f"GROUND TRUTH: {gt_answer}")
        print(f"CORRECT: {pred == gt_answer}")
        print("="*80)

        if pred == gt_answer:
            correct += 1

    acc = correct / len(aime_ds)
    print(f"[AIME] Accuracy = {acc:.3%}")


def evaluate_amc(model, tokenizer):
    """
    Evaluate on the AMC subset (all contests starting with 'AMC') of the E2H-AMC benchmark.
    """
    print("Evaluating AMC...")
    ds = load_dataset(
        "furonghuang-lab/Easy2Hard-Bench",
        "E2H-AMC",
        split="eval"
    )

    amc_ds = ds.filter(lambda x: x["contest"].startswith("AMC"))
    correct = 0

    for item in tqdm(amc_ds):
        problem = item["problem"]
        gt_answer = str(item["answer"])

        prompt = f"Q: {problem}\nA: Let's think step-by-step.\n"
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
        output = model.generate(**inputs, max_new_tokens=4096, temperature=0.3)
        decoded = tokenizer.decode(output[0], skip_special_tokens=True)

        match = re.search(r"[-+]?\d+", decoded)
        pred = match.group(0) if match else None

        if pred == gt_answer:
            correct += 1

    acc = correct / len(amc_ds)
    print(f"[AMC] Accuracy = {acc:.3%}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--datasets", nargs="+", choices=["gsm8k", "winogrande", "humaneval", "amc", "aime"], required=True)
    args = parser.parse_args()

    tokenizer, model = load_model(args.model_path)

    if "gsm8k" in args.datasets:
        evaluate_gsm8k(model, tokenizer)

    if "winogrande" in args.datasets:
        evaluate_winogrande(model, tokenizer)

    if "humaneval" in args.datasets:
        evaluate_humaneval(model, tokenizer)

    if "amc" in args.datasets:
        evaluate_amc(model, tokenizer)

    if "aime" in args.datasets:
        evaluate_aime(model, tokenizer)


if __name__ == "__main__":
    main()