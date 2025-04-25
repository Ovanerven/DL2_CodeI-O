
from utils import *
from tqdm import tqdm

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--result_file_turn1", type=str, default=None)
    parser.add_argument("--result_file_turn2", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()

    fn1 = args.result_file_turn1
    fn2 = args.result_file_turn2
    ofn = args.output_file
    dt1 = load_jsonl_yield(fn1)
    dt2 = load_jsonl_yield(fn2)
    ndt = []
    for item in tqdm(dt1):
        status = item['res']['status']
        if status == 'success':
            sample = {"prompt":item['messages'][0]['content'],
                      "turn_1":item['output'],
                      "feedback_1":item['res']['message'],
                      "turn_2":None,
                      "feedback_2":None}
            ndt.append(sample)
        if len(ndt)==1000:
            write_jsonl(ndt,ofn,"a")
            ndt = []
    for item in tqdm(dt2):
        # elegant_show(item,full=True)
        # raise ValueError
        sample = {"prompt":item['messages'][0]['content'],
                      "turn_1":item['messages'][1]['content'],
                      "feedback_1":item['messages'][2]['content'],
                      "turn_2":item['output'],
                      "feedback_2":item['res']['message']}
        ndt.append(sample)
        if len(ndt)==1000:
            write_jsonl(ndt,ofn,"a")
            ndt = []
    
    write_jsonl(ndt,ofn,"a")