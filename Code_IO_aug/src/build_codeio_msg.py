from utils import *
from tqdm import tqdm
from codeio_utils import *

def build_io_pred(problem_statement, io_req, refcode, inputx, outputx, io = "i", w_refcode=True):
    template = input_pred_template if io=="i" else output_pred_template
    prompt = template.replace("<<<<query>>>>", problem_statement).replace("<<<<io_req>>>>", io_req)
    tag = "<<<<output>>>>" if io=="i" else "<<<<input>>>>"
    inputxx = f"{inputx}"
    outputxx = f"{outputx}"
    prompt = prompt.replace(tag, outputxx if io=="i" else inputxx)
    if w_refcode:
        refcodepart = refcode_template.replace("<<<<refcode>>>>", refcode)
        prompt+="\n\n"+refcodepart
    return prompt

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_file', type=str, default='data/rawcode_1k_parsed.jsonl')
    parser.add_argument('--output_file', type=str, default='data/codeio_1k_msg.jsonl')
    args = parser.parse_args()

    fn = args.input_file
    ofn = args.output_file
    adt = []
    dt = load_jsonl_yield(fn)
    for iid,item in enumerate(tqdm(dt)):
        problem_description = item['problem_description']
        io_req = item['io_requirements']
        refcode = item['refcode']
        for ioid,io in enumerate(item['ios']):
            uplimit = 3
            if ioid>=uplimit:break # we now first only use the first 3 io
            input_xx = io['input']
            output_xx = io['output']
            oprompt = build_io_pred(problem_description, io_req, refcode, input_xx, output_xx, io="o")
            iprompt = build_io_pred(problem_description, io_req, refcode, input_xx, output_xx, io="i")
            imsg = build_messages(iprompt)
            omsg = build_messages(oprompt)
            isample = {"messages":imsg, "itemid":iid,"ioid":ioid,"io_pred":"i"}
            osample = {"messages":omsg, "itemid":iid,"ioid":ioid,"io_pred":"o"}
            adt.append(isample)
            adt.append(osample)

    write_jsonl(adt,ofn)
