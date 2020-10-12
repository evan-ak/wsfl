
import json
import torch
import random

from config import *
from dataset import *
from mcgs import *
from policynet import PolicyNet
from actionnet import ActionNet, Math23Modules, MathQAModules

def test(cfg) :
  print("PolicyNet: initializing")
  _PolicyNet_ = PolicyNet(cfg).cuda()
  cfg.device = next(_PolicyNet_.parameters()).device
  print("PolicyNet: initialized")

  print("ActionNet: initializing")
  if cfg.on_dataset == "math23" :
    _ActionNet_ = ActionNet(cfg, Math23Modules())
  elif cfg.on_dataset == "mathqa" :
    _ActionNet_ = ActionNet(cfg, MathQAModules())
  cfg.current_module_size = len(_ActionNet_.modulelist)
  cfg.current_decode_size = cfg.current_module_size + 4
  ActionOps.modules_meta_info = cfg.modules_meta_info
  print("ActionNet: initialized")

  print("Dataset: loading")
  if cfg.on_dataset == "math23" :
    if cfg.test_metric == "5flod" :
      random.seed(cfg.test_split_seed)
      mask_train = set(random.sample(range(23162), int(23162*0.8)))
      mask_test = set(range(23162)) - mask_train
      dataset_train = DatasetMath(cfg.path_data, mode="train", mask=mask_train)
      dataset_test = DatasetMath(cfg.path_data, mode="test", mask=mask_test)
    elif cfg.test_metric == "public" :
      mask_test = set(math23_public_test)
      mask_train = set(range(23162)) - mask_test
      dataset_train = DatasetMath(cfg.path_data, mode="train", mask=mask_train)
      dataset_test = DatasetMath(cfg.path_data, mode="test", mask=mask_test)
  elif cfg.on_dataset == "mathqa" :
    dataset_train = DatasetMath(cfg.path_data_train, mode="train")
    dataset_test = DatasetMath(cfg.path_data_test, mode="test")
  dataset_sampled = DatasetSample(dataset_train)
  print("Dataset: loaded")
  print()

  dataloader = torch.utils.data.DataLoader(dataset_test, batch_size = 64, collate_fn = DatasetMath.collate_fn, shuffle = False, num_workers = 0)
  if cfg.policynet_load_weight is None :
    print("Must assign save point for testing.")
    return
  else :
    print("Save point: loading")
    sd_loaded = torch.load(cfg.policynet_load_weight)
    sd_new = _PolicyNet_.state_dict()
    for k,p in sd_loaded.items() :
      if k in sd_new :
        sd_new[k].copy_(p)
    print("Save point loaded :", cfg.policynet_load_weight)
    print()

  if cfg.on_dataset == "math23" :
    _eval = None
  if cfg.on_dataset == "mathqa" :
    opt_iter = iter([(data["ans_opt"], data["opt_cor"]) for data in dataset_test.data_valid])
    def _eval(raw) :
      res = []
      for ans, ans_opt, opt_cor in zip(raw, *next(opt_iter)) :
        diff = [(i, abs(ans - opt)) for i,opt in enumerate(ans_opt) if ans is not None and type(opt) in (int, float,)]
        if len(diff) == 0 :
          opt = int(random.random()*5)
        else :
          opt = min(diff, key=lambda x:x[1])[0]
        res.append(opt == opt_cor)
      return res

  with torch.no_grad() :
    infer(dataloader, _PolicyNet_, _ActionNet_, _eval)
  return

def infer(dataloader, _PolicyNet_, _ActionNet_, _eval=None) :
  _PolicyNet_.eval()
  out_raw = []
  accu = [0, 0]
  for que, num ,ans, _, _ in dataloader :
    q = torch.tensor(que, dtype=torch.long, device=cfg.device)
    _PolicyNet_.set_batch_data({"x" : q, "y" : None})
    with torch.no_grad() :
      pgm = _PolicyNet_.program_predictor().tolist()
    for pgm, num, ans in zip(pgm, num, ans) :
      if 2 in pgm :
        pgm = pgm[:pgm.index(2)]
      pgm = MathPgmFunc.program_r2p(pgm)
      pgm = MathPgmFunc.program_regularizing(pgm, cfg)
      if pgm is None :
        accu[1] += len(ans)
        out_raw += [None] * len(ans)
      else :
        act = ActionOps.list_to_tree(pgm)
        res, raw = _ActionNet_.validate(act[0], {"num":num, "ans":ans})
        if _eval is not None :
          res = _eval(raw)
        accu[0] += sum(res)
        accu[1] += len(res)
        out_raw += raw
      print(f"\rTesting on {accu[1]:>5d}", end = "")
  print(f"\nAccuracy : {accu[0]:>5d}/{accu[1]:>5d}  {accu[0]/accu[1]:>.8f}")
  return accu, out_raw

if __name__ == '__main__' :
  print("\n================================")
  cfg = TotalConfig()
  # torch.cuda.set_device(0)
  test(cfg)
  print("================================\n")
