import json
import os

def load_jsonl_yield(path):
    with open(path) as f:
        for row, line in enumerate(f):
            try:
                line = json.loads(line)
                yield line
            except:
                pass

def read_jsonl(jsonl_file_path):
    s = []
    with open(jsonl_file_path, "r") as f:
        lines = f.readlines()
    for line in lines:
        linex = line.strip()
        if linex == "":
            continue
        s.append(json.loads(linex))
    return s

def write_jsonl(data, jsonl_file_path, mode="w"):
    # data is a list, each of the item is json-serilizable
    assert isinstance(data, list)
    if len(data) == 0:
        return
    if not os.path.exists(os.path.dirname(jsonl_file_path)):
        os.makedirs(os.path.dirname(jsonl_file_path))
    with open(jsonl_file_path, mode) as f:
        for item in data:
            f.write(json.dumps(item) + "\n")

def elegant_show(something, level=0, sid=0, full=False):
    # str,float,int
    # all print in this call should add level*4 spaces
    prefix = "\t" * level

    if isinstance(something, (str, float, int)) or something is None:
        if isinstance(something, str):
            # if '\n' in something:
            #     something = '\n'+something
            # add prefix whenever go to a new line in this string
            something = something.replace("\n", f"\n{prefix}")
        print(prefix, f"\033[1;35mElement: \033[0m", something)
    elif isinstance(something, list) or isinstance(something, tuple):
        # take a random example, and length
        # sid = 0
        if len(something) == 0:
            print(
                prefix,
                f"\033[1;33mLen: \033[0m{len(something)} \t\033[1;33m& No elements! \033[0m",
            )
        elif not full or len(something) == 1:
            print(
                prefix,
                f"\033[1;33mLen: \033[0m{len(something)} \t\033[1;33m& first element ...\033[0m",
            )
            elegant_show(something[sid], level + 1, sid, full)
        else:
            print(
                prefix,
                f"\033[1;33mLen: \033[0m{len(something)} \t\033[1;33m& Elements ...\033[0m",
            )
            for i in range(len(something) - 1):
                elegant_show(something[i], level + 1, sid, full)
                print(
                    prefix + "\t", f"\033[1;33m-------------------------------\033[0m"
                )
            elegant_show(something[-1], level + 1, sid, full)

    elif isinstance(something, dict):
        for k, v in something.items():
            print(prefix, f"\033[1;34mKey: \033[0m{k} \033[1;34m...\033[0m")
            elegant_show(v, level + 1, sid, full)
    else:
        print(prefix, f"\033[1;31mError @ Type: \033[0m{type(something)}")
        # raise NotImplementedError

def build_messages(prompt, response = None, system_message = None):
    messages = []
    if system_message is not None:
        messages.append({"role":"system","content":system_message})
    messages.append({"role":"user","content":prompt})
    if response is not None:
        messages.append({"role":"assistant","content":response})
    return messages