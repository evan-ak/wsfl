
import os
import json

local_path = os.path.abspath(".")
files = {"train" : {"in": local_path + "/train.json",
                    "out": local_path + "/mathqa_train_reg.json",
                    "vocab": local_path + "/mathqa_vocab.json",
                    "ops": local_path + "/mathqa_ops.json"},
          "test" : {"in": local_path + "/test.json",
                    "out": local_path + "/mathqa_test_reg.json",
                    "vocab": local_path + "/mathqa_vocab.json"}}

op_accept = ["add", "subtract", "multiply", "divide"]
op_reject = ["choose", "circle_area", "circumface", "cosine", "cube_edge_by_volume",
             "diagonal", "factorial", "floor", "gcd", "inverse", "lcm", "log", "max", "min", "negate", "negate_prob",
             "original_price_before_gain", "original_price_before_loss", "p_after_gain", "permutation",
             "quadrilateral_area", "rectangle_area", "rectangle_perimeter", "reminder", "rhombus_area", "rhombus_perimeter",
             "sine", "speed", "speed_in_still_water", "square_area", "square_edge_by_area", "square_edge_by_perimeter",
             "square_perimeter", "stream_speed", "surface_cube", "surface_cylinder", "surface_rectangular_prism",
             "surface_sphere", "tangent", "triangle_area", "triangle_area_three_edges", "triangle_perimeter",
             "volume_cone", "volume_cube", "volume_cylinder", "volume_rectangular_prism", "volume_sphere"] + ["power", "sqrt"]
op_func = {"add": (lambda x,y: x+y),
           "subtract": (lambda x,y: x-y),
           "multiply": (lambda x,y: x*y),
           "divide": (lambda x,y: x/y)}
eng_num = ("zero", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten",
           "eleven", "twelve", "thirteen", "fourteen", "fifteen", "sixteen", "seventeen", "eighteen", "nineteen")

token_accept = ["n"+str(i) for i in range(20)] + ["const_"+str(i) for i in (1,2,3,100)]
token_2_op = token_accept + op_accept
op_2_token = {n:i for i,n in enumerate(token_2_op)}
op_2_token[-1] = -1

def formula_traverse(raw) :
  l = raw.split('|')
  l = [t for t in l if len(t) > 0]
  l = [t.replace('(', " ").replace(',', " ").replace(')', " ").split() for t in l]
  for t in l :
    if len(t) != 3 :
      print("traverse fail, number of child not valid")
      print(raw)
      print(t)
      print()
      return None
  root = [l[-1][0],-1,]
  stack = [root]
  output = [l[-1][0]]
  while len(stack) > 0 :
    cur = stack[-1]
    idx = len(cur) - 2
    if idx < len(l[cur[1]]) - 1:
      token = l[cur[1]][idx + 1]
      if token[0] == "#" :
        nidx = int(token[1:])
        next = [l[nidx][0], nidx]
        cur.append(next)
        stack.append(next)
        output.append(l[nidx][0])
        continue
      elif token in token_accept or (token[0] == 'n' and token[1].isdigit()) :
        cur.append(token)
        output.append(token)
        output.append(-1)
      else :
        print("traverse fail, unknown token", token)
        print(raw)
        print()
        return None
    else :
      stack.pop(-1)
  return output

def formula_calculate(form, inputs) :
  stack = [[form[0],],]
  for op in form[1:] :
    if op in op_accept :
      stack.append([op,])
    elif op in token_accept :
      if op[0] == 'c' :
        num = int(op.split('_')[-1])
      if op[0] == 'n' :
        num = int(op[1:])
        num = inputs[num]
      stack[-1].append(num)
      while len(stack[-1][1:]) == 2 :
        try :
          out = op_func[stack[-1][0]](*stack[-1][1:])
        except ZeroDivisionError :
          return None
        stack.pop(-1)
        if len(stack) == 0 :
          return out
        else :
          stack[-1].append(out)
    elif op == -1 :
      pass
  return None

# =====================================================================================================================

for mode,fp in files.items() :
  vocab = set()
  vocab_count = {}
  vocab_unseen = set()
  with open(fp["in"]) as f :
    data = json.load(f)
  if mode == "test" :
    with open(fp["vocab"]) as f :
      vocab_load = json.load(f)
    token_2_vocab = vocab_load["t2v"]
    vocab_2_token = vocab_load["v2k"]
    vocab = set(token_2_vocab)

  collection = {} if mode == "train" else []

  for idx, line in enumerate(data) :
    txt = line["Problem"]
    txt_split = txt.split()
    if (len(txt_split)) > 100 :
      if mode == "train" :
        continue
      else :
        txt_split = txt_split[:100]
    if mode == "train" :
      oprej = False
      for op in op_reject :
        if op in line["linear_formula"] :
          oprej = True
          break

    unit = {"idx":idx, "text":txt}
    vocs = []
    nums_raw = []
    for word in txt_split :
      if word[0].isdigit() and not word in ('¹', '²', '³'):
        vocs.append("N"+str(len(nums_raw)))
        nums_raw.append(word.replace(',', ''))
      elif mode == "test" and not word in vocab :
        continue
      else :
        vocs.append(word)

    unit["vocs"] = vocs
    unit["nums_raw"] = nums_raw
    nums = [eval(n) if '.' in n else int(n) for n in nums_raw]
    unit["nums"] = nums
    if mode == "train" :
      vocab.update(set(vocs))
      for v in vocs :
        if not v in vocab_count :
          vocab_count[v] = [0, 0]
        vocab_count[v][0] += 1
      for v in set(vocs) :
        vocab_count[v][1] += 1

    opt_cor = line["correct"]
    opt_raw = line["options"]
    if "'" in opt_raw :
      opt_raw = opt_raw.replace("'", "")[1:-1]
    opts = []
    # for t in (", e )",", d )",", c )",", b )","a )") :
    for t in (" e )"," d )"," c )"," b )","a )") :
      s = opt_raw.split(t)
      opts.append(s[-1])
      opt_raw = s[0]
    opts = opts[::-1]
    ans_opt = []
    for opt in opts :
      ans_tok = opt.split()
      ans = ""
      for t in ans_tok :
        if t[0].isdigit() or t in (":", "%", "/", "^", "-", "√") or t.lower() in eng_num :
          ans += t
      try :
        ans = ans.replace("%", "").replace(",", "")
        if ans in eng_num :
          ans = eng_num.index(ans)
        elif ":" in ans :
          ans = (lambda x,y:eval(x)/eval(y))(*ans.split(':')[:2])
        else :
          ans = eval(ans)
        assert type(ans) in (int, float,)
      except :
        ans = None
      ans_opt.append(ans)
    unit["opt_raw"] = opts
    unit["opt_cor"] = ord(opt_cor) - ord('a')
    unit["ans_opt"] = ans_opt
    unit["ans"] = ans_opt[unit["opt_cor"]]

    if mode == "train" and not oprej:
      pgm_raw = formula_traverse(line["linear_formula"])
      if pgm_raw is None :
        unit["pgm_raw"] = None
        unit["pgm"] = None
        unit["pgm_valid"] = 0
      else :
        unit["pgm_raw"] = pgm_raw
        unit["pgm"] = [op_2_token[t] for t in pgm_raw]
        res = formula_calculate(pgm_raw, nums)
        try :
          pgm_valid = int(abs(res - unit["ans"]) < 0.01)
        except :
          pgm_valid = 0
        unit["pgm_valid"] = pgm_valid
    else :
      unit["pgm_raw"] = None
      unit["pgm"] = None
      unit["pgm_valid"] = 0

    if mode == "train" :
      key = tuple(unit["vocs"])
      if not key in collection :
        collection[key] = {k:[] for k in unit.keys()}
        collection[key]["vocs"] = key
      for k,v in unit.items() :
        if type(container := collection[key][k]) is list :
          container.append(v)
    else :
      col = {k:[v] for k,v in unit.items()}
      col["vocs"] = unit["vocs"]
      col["pgm_each"] = unit["pgm"]
      col["pgm"] = None
      col["pgm_rate"] = 0
      collection.append(col)

  if mode == "train" :
    for key,group in collection.items() :
      group["pgm_each"] = group["pgm"]
      if len(group["idx"]) <= 1 :
        group["pgm"] = group["pgm"][0]
        group["pgm_rate"] = group["pgm_valid"][0]
        continue
      if sum(group["pgm_valid"]) < 1 :
        group["pgm"] = None
        group["pgm_rate"] = 0
        continue
      pgm_set = set()
      need_reeval = False
      for pgm_raw in group["pgm_raw"] :
        if pgm_raw is None :
          need_reeval = True
        else :
          pgm_set.add(tuple(pgm_raw))
      if len(pgm_set) <= 1 and not need_reeval :
        group["pgm"] = group["pgm"][0]
        group["pgm_rate"] = sum(group["pgm_valid"]) / len(group["pgm_valid"])
      else :
        pgm_reeval = {}
        for pgm in pgm_set :
          pgm_reeval[pgm] = 0
          for nums,ans in zip(group["nums"], group["ans"]) :
            if ans is None :
              continue
            res = formula_calculate(pgm, nums)
            try :
              pgm_reeval[pgm] += int(abs(res - ans) < 0.01)
            except :
              pass
        pgm, rate = max(pgm_reeval.items(), key=lambda x:x[1])
        group["pgm"] = [op_2_token[t] for t in pgm]
        group["pgm_rate"] = rate / len(group["idx"])

    for key,group in collection.items() :
      if group["pgm_rate"] > 0 :
        assert group["pgm"] is not None
    collection = collection.values()

  outs = sorted(collection, key = lambda x: len(x["vocs"]))

  print("from", fp["in"])
  print(len(data), "data processed.")
  print(len(outs), "data after combined")
  print()

  if mode == "train" :
    vocab_sort = sorted(vocab_count.items(), key = lambda x:-x[1][0])
    token_2_vocab = ["<None>", "<START>", "<END>", "<UNKNOWN>"] + ["N"+str(i) for i in range(16)]
    for v,_ in vocab_sort :
      if not v in token_2_vocab :
        token_2_vocab.append(v)
    vocab_2_token = {v:i for i,v in enumerate(token_2_vocab)}
    len_max = max([len(l["vocs"]) for l in outs])
    for line in outs :
      line["toks"] = [1] + [vocab_2_token[v] for v in line["vocs"]] + [2] + [0]*(len_max - len(line["vocs"]))

    with open(fp["out"], "w") as f :
      json.dump(outs, f)
    with open(fp["vocab"], "w") as f :
      json.dump({"t2v":token_2_vocab, "v2k":vocab_2_token}, f)
    with open(fp["ops"], "w") as f :
      json.dump({"t2v":token_2_op, "v2k":op_2_token}, f)

  else :
    len_max = max([len(l["vocs"]) for l in outs])
    for line in outs :
      line["toks"] = [1] + [vocab_2_token[v] for v in line["vocs"]] + [2] + [0]*(len_max - len(line["vocs"]))

    with open(fp["out"], "w") as f :
      json.dump(outs, f)
