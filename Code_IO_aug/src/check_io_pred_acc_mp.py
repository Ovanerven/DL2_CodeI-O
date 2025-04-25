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
        if not isinstance(last_json, dict):
            return {"status":"no answer","message":"The last json is not a dict!"}
        if "output" not in last_json:
            return {"status":"no answer","message":"No field 'output' in the last json!"}
        pred_output = last_json["output"]
        acc = is_close(pred_output, needed_oriio["output"])
        if acc:
            return {"status":"success","message":"Correct output!"}
        else:
            return {"status":"failed","message":f"[Mismatch] Your output is not correct! Given the input {json.dumps(needed_oriio['input'])}, your predicted output is {json.dumps(pred_output)} which is wrong!"}
    elif item['io_pred'] == "i":
        if not isinstance(last_json, dict):
            return {"status":"no answer","message":"The last json is not a dict!"}
        if "input" not in last_json:
            return {"status":"no answer","message":"No field 'input' in the last json!"}
        pred_input = last_json["input"]
        candio = {'input': pred_input, 'output': needed_oriio['output']}
        res = check_input(needed_oriitem['refcode'], candio, needed_oriitem['funcname'], solution_prefix=solution_prefix, runtime_limit=5, used_python_path = python_path, run_path=run_path)
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

def get_total_items_with_wc(filename):
    result = subprocess.run(['wc', '-l', filename], stdout=subprocess.PIPE, text=True)
    total_lines = int(result.stdout.split()[0])  # wc输出的形式是: 行数 文件名, 所以只取第一部分
    return total_lines

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--parsed_file_name",type=str,help="The path to the parsed io file.")
    parser.add_argument("--pred_file_name", type=str, help="The path to the prediction file.")
    parser.add_argument("--res_file_name", type=str, help="The path to the result file.")
    parser.add_argument("--batchsize", type=int, help="The batch size.")
    parser.add_argument("--num_processes", type=int, help="The number of processes.")
    parser.add_argument('--python_path', type=str, default="python")
    parser.add_argument('--run_path', type=str, default="./temp/temp/temp")
    args = parser.parse_args()

    pred_file_name = args.pred_file_name
    res_file_name = args.res_file_name
    global parsed_ios
    parsed_ios = read_jsonl(args.parsed_file_name)

    global python_path, run_path
    python_path = args.python_path
    run_path = args.run_path

    if not os.path.exists(run_path):
        os.makedirs(run_path,exist_ok=True)

    # Load dt as a generator
    dt = load_jsonl_yield(pred_file_name)  # This is a generator object

    if os.path.exists(res_file_name):
        existing = get_total_items_with_wc(res_file_name)
    else:
        existing = 0
    
    print(f"Existing items: {existing}, skipping them.")
    dt = islice(dt, existing, None)

    total_num_items = get_total_items_with_wc(pred_file_name)-existing

    batchsize = args.batchsize

    # Use fork method to avoid duplicating source_2_data
    ctx = mp.get_context('fork')

    num_processes = args.num_processes

    with ctx.Pool(processes = num_processes) as pool:
        # Create batches of items
        batches = batcher(dt, batchsize)

        # Initialize the progress bar
        pbar = tqdm(total=total_num_items)
        
        for batch_idx, batch in enumerate(batches):
            
            batch_i = [item for item in batch if item['io_pred'] == 'i']
            batch_o = [item for item in batch if item['io_pred'] == 'o']
            assert len(batch_i) + len(batch_o) == len(batch)
            print("I:", len(batch_i), "O:", len(batch_o))

            batchstat = defaultdict(int)
            # Process the batch in parallel

            results = pool.map(check_io_pred_acc, batch_i)
            # Combine items with their results
            for item, res in zip(batch_i, results):
                item['res'] = res
                batchstat[res['status']] += 1
            results = pool.map(check_io_pred_acc, batch_o)
            # Combine items with their results
            for item, res in zip(batch_o, results):
                item['res'] = res
                batchstat[res['status']] += 1

            # Write the batch to the output file
            write_jsonl(batch_i, res_file_name, mode='a')
            write_jsonl(batch_o, res_file_name, mode='a')

            print(f"Wrote a batch of {len(batch)} items.")
            print(f"Batch {batch_idx} status: {batchstat}")

            # Update the progress bar
            pbar.update(len(batch))

        pbar.close()

if __name__ == "__main__":
    import multiprocessing as mp

    # Set the start method to 'fork' to avoid duplicating source_2_data
    mp.set_start_method('fork')

    main()
