
from utils import *
from codeio_utils import *
import subprocess
import tempfile
import os
import json
import signal
from time import sleep

from multiprocessing import Pool
from collections import defaultdict
from tqdm import tqdm
import gc
import shutil

def resolve_output(output):
    # extract_last_python
    pos1 = output.find("## Main Function")
    if pos1 == -1: 
        return None
    pos11 = pos1+len("## Main Function")
    pos2 = output.find("## Input Output Description", pos11)
    if pos2 == -1: 
        return None
    maincode = extract_last_python(output[pos11:pos2].strip())
    if maincode is None: 
        return None

    pos21 = pos2+len("## Input Output Description")
    pos3 = output.find("## Input Generator", pos21)
    if pos3 == -1: 
        pos3 = output.find("## Input generator", pos21)
        if pos3 == -1:
            return None
    io_desc = output[pos21:pos3].strip()

    pos31 = pos3+len("## Input Generator")
    pos4 = output.find("## Problem Statement", pos31)
    if pos4 == -1: 
        return None
    inputgencode = extract_last_python(output[pos31:pos4].strip())
    if inputgencode is None: 
        return None

    pos41 = pos4+len("## Problem Statement")
    pos5 = len(output)
    problem_statement = output[pos41:pos5].strip()

    if "main_solution" not in maincode:
        return None
    if "input_generator" not in inputgencode:
        return None

    return {
        "maincode": maincode,
        "io_desc": io_desc,
        "inputgencode": inputgencode,
        "problem_statement": problem_statement
    }

templatexx = """
import json
from pympler import asizeof

def strict_check_size(obj):
    # Check if object size is less than 1024 bytes
    if asizeof.asizeof(obj) >= 1024: 
        return False

    # Check for dict type
    if isinstance(obj, dict):
        if len(obj) >= 20:  # Check dict has fewer than 20 key-value pairs
            return False
        # Recursively check keys and values
        for k, v in obj.items():
            if not strict_check_size(k) or not strict_check_size(v):
                return False

    # Check for list, tuple, or set
    elif isinstance(obj, (list, tuple, set)):
        if len(obj) >= 20:  # Check if the length is less than 20
            return False
        # Recursively check each element
        for item in obj:
            if not strict_check_size(item):
                return False

    # Check for string
    elif isinstance(obj, str):
        if len(obj) >= 100:  # Check if string length is less than 100 characters
            return False

    # elif isinstance(obj, float):
    #     d = Decimal(str(obj))
    #     if d.as_tuple().exponent < -3:
    #         return False

    # Other objects - check size in bytes
    else:
        if asizeof.asizeof(obj) >= 128:  # Check if object size is less than 128 bytes
            return False

    # If all checks are passed, return True
    return True

<<<|!!|!!|maincode|!!|!!|>>>

<<<|!!|!!|inputgencode|!!|!!|>>>

diff_inputs = []
corr_outputs = []
for i in range(1000):
    cand_input = input_generator()
    if cand_input not in diff_inputs and strict_check_size(cand_input):
        cand_output = main_solution(**cand_input)
        if strict_check_size(cand_output) and cand_output is not None:
            diff_inputs.append(cand_input)
            corr_outputs.append(cand_output)
    if len(diff_inputs) >= 10:
        break
        
assert len(diff_inputs) == len(corr_outputs)

iolist = [{"input": diff_inputs[i], "output": corr_outputs[i]} for i in range(len(diff_inputs))]

jsoniolist = json.dumps(iolist)
    
print("[JSON IOS START]" + jsoniolist + "[JSON IOS END]")  
"""

def process_one_item(res):
    
    maincode = res['maincode']
    io_desc = res['io_desc']
    inputgencode = res['inputgencode']
    problem_statement = res['problem_statement']
    pyruncode = "import json\nfrom pympler import asizeof\n\n"+templatexx.replace("<<<|!!|!!|maincode|!!|!!|>>>", maincode).replace("<<<|!!|!!|inputgencode|!!|!!|>>>", inputgencode)
    
    runtime_limit = 60  # seconds

    try:
        # Start the subprocess in a new session (process group)
        proc = subprocess.Popen(
            [used_python_path, '-'],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=run_path,
            text=True,
            start_new_session=True
        )

        try:
            # Communicate with the subprocess
            stdout, stderr = proc.communicate(input=pyruncode, timeout=runtime_limit)
        except subprocess.TimeoutExpired:
            # Timeout expired; kill the process group
            os.killpg(proc.pid, signal.SIGTERM)  # Send SIGTERM to the process group
            stdout, stderr = proc.communicate()
            return None
        except Exception as e:
            # Other exception occurred; kill the process group
            os.killpg(proc.pid, signal.SIGTERM)
            stdout, stderr = proc.communicate()
            return None
        finally:
            # Ensure the subprocess is terminated
            proc.kill()
            proc.wait()

        # Process stdout as before
        start_marker = "[JSON IOS START]"
        end_marker = "[JSON IOS END]"

        if start_marker in stdout and end_marker in stdout:
            start_index = stdout.index(start_marker) + len(start_marker)
            end_index = stdout.index(end_marker)
            json_str = stdout[start_index:end_index].strip()

            # Load the JSON string
            json_data = json.loads(json_str)
            return json_data
        else:
            # If markers are not found, return None
            return None

    except Exception as e:
        # Handle any exceptions that might occur while setting up the subprocess
        return None

def process_item(item):
    res = resolve_output(item['output'])
    if res is None:
        return None
    ios = process_one_item(res)
    if ios is None or len(ios)<=1:
        return None
    sample = {
        "problem_description": res['problem_statement'],
        "io_requirements": res['io_desc'],
        "refcode": res['maincode'],
        "funcname": "main_solution",
        "ios": ios, # can be empty, to indicate all of them are too large
        # "source": "z3examples",
        "category": item.get('category',None),
        "meta": {
            # "repo_name": item['repo_name'],
            # "path": item['path'],
            # "original_sourece": item['source'],
            # "eid": item['eid'],
            "msgidx": item['index'],
        }
    }
    # Clean up
    del res
    del ios
    gc.collect()
    sleep(0.05)
    return sample

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default="data/rawcode_1k_unified.jsonl")
    parser.add_argument('--output_file', type=str, default="data/rawcode_1k_parsed_2.jsonl")
    parser.add_argument('--python_path', type=str, default="python")
    parser.add_argument('--run_path', type=str, default="./temp/temp/temp")
    args = parser.parse_args()

    if not os.path.exists(args.run_path):
        os.makedirs(args.run_path,exist_ok=True)
    
    used_python_path = args.python_path
    run_path = args.run_path

    fn = args.input_file

    ofn = args.output_file
    dt = read_jsonl(fn)

    if os.path.exists(ofn):
        exdt = read_jsonl(ofn)
        exindex = set([x['meta']['msgidx'] for x in exdt])
    else:
        exindex = set()

    dt = [x for x in dt if x['index'] not in exindex]

    print("Skip existing:", len(exindex))

    adt = []
    goodcount=0
    prevgoodcount=0
    totalcount=0
    prevtotalcount=0
    with Pool(processes=64, maxtasksperchild=10) as pool:
        for result in tqdm(pool.imap_unordered(process_item, dt), total=len(dt)):
            totalcount+=1
            if result is not None:
                adt.append(result)
                goodcount+=1
            if len(adt) >= 100:
                write_jsonl(adt, ofn,"a")
                adt = []
                print(f"{goodcount}/{totalcount}")
                detlagoodcount = goodcount - prevgoodcount
                detlatotalcount = totalcount - prevtotalcount
                print(f"Delta: {detlagoodcount}/{detlatotalcount} = {detlagoodcount/detlatotalcount}")
                prevgoodcount = goodcount
                prevtotalcount = totalcount

    if len(adt) > 0:
        write_jsonl(adt, ofn,"a")
        print(f"Final - {goodcount}/{totalcount}")

    try:
        shutil.rmtree(args.run_path)
    except Exception as e:
        print(f"Error: {e}")