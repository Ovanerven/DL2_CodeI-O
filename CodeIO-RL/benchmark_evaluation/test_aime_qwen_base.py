#!/usr/bin/python python3
import warnings
warnings.filterwarnings("ignore")
import re
import os
import csv
import json
import time
import types
import random
import textwrap
from tqdm import tqdm
from datetime import datetime
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import argparse
# from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
from datetime import datetime

def main():
    parser = argparse.ArgumentParser()
    # parser.add_argument('model', type=str)
    parser.add_argument('--model_path', type=str, required=True)
    # parser.add_argument('--json_path', type=str, required=True)
    # parser.add_argument('--step', type=int, required=True)
    args = parser.parse_args()
    # print(args.model_path)
    # step = re.search(r'(\d+)$', args.model_path).group(1)
    # # # model_path = f"/volume/ailab4sci/ztgao/Logic-RL/checkpoints/GRPO_logic_KK/RF++-xppl-stage3-len8192-step1800-t0_7-001/actor/global_step_{args.step}"
    # # # model_path = f"/volume/ailab4sci/ztgao/Logic-RL/checkpoints/GRPO_logic_KK/RF++-Qwen-7B-1M-xppl-test-01/actor/global_step_{args.step}"
    
    # # if args.stage == 0:
    # #     model_path = "/volume/ailab4sci/models/Qwen2.5-7B-Instruct-1M"
    # # elif args.stage == 1:
    # #     model_path = f"/volume/ailab4sci/ztgao/Logic-RL/checkpoints/GRPO_logic_KK/RF++-Qwen-7B-1M-xppl-002/actor/global_step_{args.step}"
    # # elif args.stage == 2:
    # #     model_path = f"/volume/ailab4sci/ztgao/Logic-RL/checkpoints/GRPO_logic_KK/RF++-xppl-step1320-t0_7-001/actor/global_step_{args.step}"

    # model_path = "/volume/ailab4sci/models/Qwen2.5-7B-Instruct"
    # model_path = "/volume/ailab4sci/models/CodeR1-Zero-Qwen2.5-7B-12k-832"
    # model_path = "/volume/ailab4sci/models/CodeR1-Zero-Qwen2.5-7B-LC2k-1088"
    # model_name = args.model
    # model_path = f"/volume/ailab4sci/models/{model_name}"
    # model_path = "/volume/ailab4sci/ztgao/checkpoints/GRPO_logic_KK/rpp_qwen32b_5ppl_2e-6_16gpu/actor/global_step_120"

    # Load model and tokenizer from Hugging Face transformers
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )
    
    # Read test data
    with open("aime_2021_2024.jsonl", encoding="utf-8") as file:
        data = [json.loads(line) for line in file.readlines() if line]
    
    cnt = 0
    total_time = 0
    results = []
    # step = args.model_path.split('_')[-1] if '_' in args.model_path else 'base'
    data = data[:5]
    for d in tqdm(data):
        prompt = d["question"]
        messages = [
            {"role": "system", "content": "You are a helpful assistant. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and<answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> answer here </answer>.  Now the user asks you to solve a math problem. After thinking, when you finally reach a conclusion, clearly state the answer within <answer> </answer> tags. i.e., <answer> (\\boxed{}\\) </answer>."},
            {"role": "user", "content": prompt}
        ]
        
        # Apply chat template to format messages
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        
        expected_answer = d['answer']
        
        # Generate response
        start_time = time.time()
        
        # Tokenize input text
        inputs = tokenizer(text, return_tensors="pt",  return_attention_mask=True).to(model.device)
        
        # Generate output
        generation_output = model.generate(
            **inputs,
            max_new_tokens=10000,
            temperature=0.8,
            top_p=0.95,
            do_sample=True
        )
        
        # Decode output tokens to text
        response = tokenizer.decode(generation_output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        time_taken = time.time() - start_time

        if '<answer>' in response:
            result = re.split(r'<answer>', response)[1]
        else:
            result = response[len(response) - 30:]
        print(result)
        correct = expected_answer in result
        
        result = {
            "question": d['question'],
            "generated_output": response,
            "expected_expected_answer": expected_answer,
            "correct": correct,
            "time_taken": time_taken
        }

        results.append(result)

        if correct:
            cnt += 1

        total_time += time_taken
    
    acc = cnt / len(data)
    print(f"ACC: {acc}")
    # with open(f"aime.json", 'w') as outfile:
    #     json.dump(results, outfile, indent=4)

if __name__ == "__main__":
    main()