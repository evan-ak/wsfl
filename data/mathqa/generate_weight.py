
import os
import json
import torch

local_path = os.path.abspath(".")
path_raw = local_path + "/glove.42B.300d.txt"
path_vocab = local_path + "/mathqa_vocab.json"
path_out_mask = local_path + "/mathqa_vocab_mask"
path_out_weight = local_path + "/mathqa_vocab_weight"

with open(path_vocab, 'r') as f :
  math_vocab = json.load(f)
math_vocab = math_vocab["t2v"]

collected = {v:None for v in math_vocab}
cc = 0
lc = 0
with open(path_raw, 'r') as f :
  for line in f :
    lc += 1
    line = line.split()
    if line[0] in collected :
      try :
        collected[line[0]] = [eval(v) for v in line[1:]]
      except :
        print()
        print("file to load line", lc)
        print(line)
      cc += 1
    elif lc % 100 != 0 :
      continue
    print(f"\r {cc}/{lc} words loaded from raw data", end='')
print()

mask = [collected[v] is not None for v in math_vocab]
weight = [collected[v] or [0]*300 for v in math_vocab]
weight = torch.FloatTensor(weight)
emb_m = torch.nn.Embedding.from_pretrained(weight)

with open(path_out_mask, 'w+') as f :
  f.write(str(mask))
torch.save(emb_m.state_dict(), path_out_weight)
