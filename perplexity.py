from datasets import Dataset
from transformers import AutoTokenizer, AutoModelForCasualLM, DataCollectorForSeq2Seq

device = "cuda"
model = AutoModelForCasualLM.from_pretrained("xxx", cache_dir='xxxxxx').to(device)
tokenizer = AutoTokenizer.from_pretrained("xxx", cache_dir='xxxxxx')

from datasets import load_dataset

test = load_dataset("xxx", "xxx", split="test")
# concatenate all items to one sequence
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

import torch
from tqdm import tqdm

max_length = model.config.seq_length # model's max length
stride = 512 # stride of sliding window
seq_len = encodings.input_ids.size(1) # length of concatenated sequence

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len) # slice out sequence of length=stride or to the end of the seq
    tgt_loc = end_loc - prev_end_loc # just calculate the new part of seq
    
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()[:, :-tgt_loc] = -100 # set positions which aren't target as -100
    # Note that -100 is a default value for pytorch CrossEntropyLoss to ignore the loss of these positions
    
    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)
        neg_log_likelihood = outputs.loss # cross_entropy_loss whose exp() is just perplexity
        
    nlls.append(neg_log_likelihood)
    
    # prepare for next iteration
    prev_end_loc = end_loc
    if end_loc == seq_len:
        break
    
ppl = torch.exp(torch.stack(nlls).mean())

print(ppl)
    