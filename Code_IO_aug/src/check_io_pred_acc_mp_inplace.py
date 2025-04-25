from os import write
from codeio_utils import *
from utils import *
import copy
import multiprocessing as mp
from tqdm import tqdm
from itertools import islice
import json
from multiprocessing.pool import ThreadPool
from collections import defaultdict

# global vars
python_path = "python"
run_path = "./temp/temp/temp"

# Your check_io_pred_acc function uses source_2_data
def check_io_pred_acc(item):
    output = item["output"]
    last_json = extract_last_complete_json(output)
    if last_json is None:
        return {"status":"no answer","message":"Fail to extract a complete and valid json from the output!"}
    needed_oriitem = parsed_ios[item["itemid"]]
    needed_oriio = needed_oriitem["ios"][item["ioid"]]
    if item['io_pred'] == "o":
        if "output" not in last_json:
            return {"status":"no answer","message":"No field 'output' in the last json!"}
        pred_output = last_json["output"]
        acc = is_close(pred_output, needed_oriio["output"])
        if acc:
            return {"status":"success","message":"Correct output!"}
        else:
            return {"status":"failed","message":f"[Mismatch] Your output is not correct! Given the input {json.dumps(needed_oriio['input'])}, your predicted output is {json.dumps(pred_output)} which is wrong!"}
    elif item['io_pred'] == "i":
        if "input" not in last_json:
            return {"status":"no answer","message":"No field 'input' in the last json!"}
        pred_input = last_json["input"]
        candio = {'input': pred_input, 'output': needed_oriio['output']}
        res = check_input(needed_oriitem['refcode'], candio, needed_oriitem['funcname'], solution_prefix=solution_prefix, used_python_path = python_path, run_path=run_path)
        if "exception_type" in res:
            res['messages'] = f"[{res['exception_type']}] {res['message']}"
        return res

# Function to batch items from an iterator
def batcher(iterable, batch_size):
    """Batch an iterator into lists of length batch_size"""
    it = iter(iterable)
    while True:
        chunk = list(islice(it, batch_size))
        if not chunk:
            break
        yield chunk

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed_file_name",type=str,help="The path to the parsed io file.")
    parser.add_argument("--pred_file_name", type=str, help="The path to the prediction file.")
    parser.add_argument("--batchsize", type=int, help="The batch size.", default=1024)
    parser.add_argument("--write_batchsize", type=int, help="The batch size of batch size", default=8)
    parser.add_argument("--num_processes", type=int, help="The number of processes.", default=16)
    parser.add_argument('--python_path', type=str, default="python")
    parser.add_argument('--run_path', type=str, default="./temp/temp/temp")   
    args = parser.parse_args()

    pred_file_name = args.pred_file_name
    batchsize = args.batchsize
    write_batchsize = args.write_batchsize
    global parsed_ios
    parsed_ios = read_jsonl(args.parsed_file_name)

    global python_path, run_path
    python_path = args.python_path
    run_path = args.run_path

    if not os.path.exists(run_path):
        os.makedirs(run_path,exist_ok=True)

    print(f"Reading data from {pred_file_name}...")
    dt = read_jsonl(pred_file_name)
    total_num_items = len(dt)

    # Get list of items to process: items without 'res' or items with 'res','status' == "timeout"
    items_to_process = [(idx, item) for idx, item in enumerate(dt) if 'res' not in item or item['res']['status'] == "timeout"]
    total_items_to_process = len(items_to_process)

    print(f"Total items: {total_num_items}")
    print(f"Items to process: {total_items_to_process}")

    # Use fork method to avoid duplicating source_2_data
    ctx = mp.get_context('fork')
    num_processes = args.num_processes

    with ctx.Pool(processes=num_processes) as pool:
        # Create batches of items to process
        batches = batcher(items_to_process, batchsize)

        # Initialize the progress bar
        pbar = tqdm(total=total_items_to_process)

        bid=0
        fullbatchstat = defaultdict(int)

        for batch in batches:
            # batch is a list of (idx, item)
            batch_indices = [idx for idx, item in batch]
            batch_items_o = [item for idx, item in batch if item['io_pred'] == 'o']
            batch_items_i = [item for idx, item in batch if item['io_pred'] == 'i']
            assert len(batch) == len(batch_items_i) + len(batch_items_o)
            print(f"I: {len(batch_items_i)}, O: {len(batch_items_o)}")

            batchstat = defaultdict(int)

            # Process the batch in parallel - o
            results = pool.map(check_io_pred_acc, batch_items_o)
            # Update items in dt
            for idx, res in zip(batch_indices, results):
                dt[idx]['res'] = res
                batchstat[res['status']] += 1
                fullbatchstat[res['status']] += 1

            # Process the batch in parallel - i
            results = pool.map(check_io_pred_acc, batch_items_i)
            for idx, res in zip(batch_indices, results):
                dt[idx]['res'] = res
                batchstat[res['status']] += 1
                fullbatchstat[res['status']] += 1

            print(batchstat)

            if (bid+1) % write_batchsize == 0:
                # Write the batch to the output file
                write_jsonl(dt, pred_file_name, mode='w')
                print(f"[bid {bid} finished] Wrote several batches of items.")

            # # Write the entire dt back to the same file
            # write_jsonl(dt, pred_file_name, mode='w')

            # Update the progress bar
            pbar.update(len(batch))
            bid+=1

        # Write the remaining items to the output file
        write_jsonl(dt, pred_file_name, mode='w')

        print(fullbatchstat)

        pbar.close()

if __name__ == "__main__":
    import multiprocessing as mp
    import time
    # 获取当前时间的结构化格式
    current_time = time.localtime()

    # 格式化输出（秒、分钟、小时、日、月、年）
    print("Current time: ", time.strftime("%H:%M:%S %d-%m-%Y", current_time))

    # Set the start method to 'fork' to avoid duplicating source_2_data
    mp.set_start_method('fork')

    main()
