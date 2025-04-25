

from utils import *
from tqdm import tqdm


def wrap_feedback(errormessage,xtype=0):
    if xtype==0:
        return errormessage
    elif xtype==1:
        return errormessage+'\n\nPlease redo it, and your prediction should no longer be any of the wrong ones you have made before!'
    else:
        raise ValueError

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", type=str, default=None)
    parser.add_argument("--output_file", type=str, default=None)
    args = parser.parse_args()
    revision_type = 1

    ifn = args.input_file
    ofn = args.output_file

    dt = load_jsonl_yield(ifn)
    batch = []
    for iid, item in enumerate(tqdm(dt)):
        if item['res']['status']=='success':
            continue
        item['messages'].append({"role":"assistant","content": item['output']} )
        item.pop('output')
        item['messages'].append({"role":"user","content":wrap_feedback(item['res']['message'],revision_type)})
        item['retry_count'] = item.get('retry_count',0)+1
        item['history_errors'] = item.get('history_errors',[])
        item['history_errors'].append(item['res'])
        item.pop('res')

        item['original_id'] = item.get('original_id',iid)

        batch.append(item)
        if len(batch)==1000:
            write_jsonl(batch,ofn,"a")
            batch = []
    write_jsonl(batch,ofn,"a")