
import os
import json
import string

local_path = os.path.abspath(".")
file_in = local_path + "/Math_23K.json"
file_out = local_path + "/math23k_reg.json"
file_vocab = local_path + "/math23k_vocab.json"
file_ops = local_path + "/math23k_ops.json"

op_func = {"+": lambda x,y: x+y,
           "-" : lambda x,y: x-y,
           "*" : lambda x,y: x*y,
           "/" : lambda x,y: x/y}
ops = ['+','-','*','/']
mks = ['(', ')']
token_2_op = ["n"+str(i).zfill(2) for i in range(6)] + ["c"+str(i) for i in (1,100)] + ['+','-','*','/']
op_2_token = {n:i for i,n in enumerate(token_2_op)}
op_2_token["end"] = -1
alphabet_list = list(string.ascii_lowercase)
def rm_alp(s) :
  for c in alphabet_list :
    s = s.replace(c,'')
  return s

with open(file_in, 'r') as f :
  raw = f.read()
raw = (raw + '\n').split("}\n")[:-1]
raw = [json.loads(line + '}') for line in raw]

def _eval(n) :
  if not '(' in n or n[0] == '(':
    if n[0] == '(' :
      while n[-1] != ')' :
        n = n[:-1]
    if '%' in n :
      r = eval(n.replace('%', ''))/100
    else :
      r = eval(n)
  else :
    si = n.index('(')
    r = eval(n[:si]) + eval(n[si:])
  return r

vocab = set()
vocab_count = {}
collection = []

for idx, line in enumerate(raw) :
  unit = {"idx":idx, "text":line["original_text"]}
  vocs = []
  nums_raw = []
  for word in line["segmented_text"].split() :
    if word[0].isdigit() or (word[0] == '(' and ')' in word):
      vocs.append("N"+str(len(nums_raw)))
      nums_raw.append(word)
    else :
      vocs.append(word)
  unit["vocs"] = vocs
  unit["nums_raw"] = nums_raw
  nums = [_eval("".join(filter(lambda x:(x.isdigit() or x in ('/', '.', '(', ')', '%')), nr))) for nr in nums_raw]
  unit["nums"] = nums

  vocab.update(set(vocs))
  for v in vocs :
    if not v in vocab_count :
      vocab_count[v] = [0, 0]
    vocab_count[v][0] += 1
  for v in set(vocs) :
    vocab_count[v][1] += 1

  ar = line["ans"]
  unit["ans_raw"] = ar
  try :
    unit["ans"] = _eval(ar)
  except :
    unit["ans"] = None
    print("fail to decode token", ar)
    print()

  col = {k:[v] for k,v in unit.items()}
  col["vocs"] = unit["vocs"]
  col["pgm"] = None
  col["pgm_rate"] = 0
  collection.append(col)

outs = sorted(collection, key = lambda x: len(x["vocs"]))

print()
print("from", file_in)
print(len(raw), "data processed.")
print(len(outs), "data after combined")
print()

vocab_sort = sorted(vocab_count.items(), key = lambda x:-x[1][0])
token_2_vocab = ["<None>", "<START>", "<END>", "<UNKNOWN>"] + ["N"+str(i) for i in range(16)]
for v,_ in vocab_sort :
  if not v in token_2_vocab :
    token_2_vocab.append(v)
vocab_2_token = {v:i for i,v in enumerate(token_2_vocab)}
len_max = max([len(l["vocs"]) for l in outs])
for line in outs :
  line["toks"] = [1] + [vocab_2_token[v] for v in line["vocs"]] + [2] + [0]*(len_max - len(line["vocs"]))

with open(file_out, "w") as f :
  json.dump(outs, f)
with open(file_vocab, "w") as f :
  json.dump({"t2v":token_2_vocab, "v2k":vocab_2_token}, f)
with open(file_ops, "w") as f :
  json.dump({"t2v":token_2_op, "v2k":op_2_token}, f)
