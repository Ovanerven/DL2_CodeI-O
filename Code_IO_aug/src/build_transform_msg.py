

import argparse
from codeio_utils import build_testcases_prompt_advanced
from utils import *

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw_code_file', type=str, default='data/rawcode_1k.jsonl')
    parser.add_argument('--raw_code_msg_file', type=str, default='data/rawcode_1k_msg.jsonl')
    args = parser.parse_args()
    dt = read_jsonl(args.raw_code_file)
    messages = []
    for item in dt:
        messages.append(
                        {
                            "messages":[
                                {
                                "role":"user",
                                "content":build_testcases_prompt_advanced.replace("<<<<code>>>>", item["content"])
                                }       
                            ]
                        }
                        )
    write_jsonl(messages, args.raw_code_msg_file)
    






