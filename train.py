
import json
import math
import numpy
import random
import multiprocessing
from multiprocessing import Process, shared_memory

import torch
import torch.utils.data

import os
import datetime

from config import *
from dataset import *
from mcgs import *
from policynet import PolicyNet
from actionnet import ActionNet, Math23Modules, MathQAModules
from test import infer

def train(cfg) :
  print("PolicyNet: initializing")
  _PolicyNet_ = PolicyNet(cfg).cuda()
  cfg.device = next(_PolicyNet_.parameters()).device
  print("PolicyNet: initialized")

  print("ActionNet: initializing")
  if cfg.on_dataset == "math23" :
    _ActionNet_ = ActionNet(cfg, Math23Modules())
    cfg.n_candidate = (4, 10)
  elif cfg.on_dataset == "mathqa" :
    _ActionNet_ = ActionNet(cfg, MathQAModules())
    cfg.n_candidate = (5, 20)
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

  optimizer = torch.optim.Adam(_PolicyNet_.parameters(), lr = 0.001)
  _PolicyNet_.train()

  if cfg.on_dataset == "mathqa" and cfg.use_annotation :
    solved = set()
    unsolved = {}
    unmet = []
    pre_encode = None

    cfg.module_used = [False] * cfg.decoder_word_size
    for i, d in enumerate(dataset_train.data_valid) :
      if d["pgm_rate"] > 0 and d["pgm"] is not None and len(d["pgm"]) < 48 :
        dataset_sampled.add(i)
        for a in d["pgm"] :
          if a >= 0 :
            cfg.module_used[a] = True
        if d["pgm_rate"] > cfg.actions_accept_boundary :
          solved.add(i)
        else :
          unsolved[i] = [1.0, 1, 0.0]
      else :
        unmet.append(i)
    solved_count = len(solved)
  else :
    solved = set()
    solved_count = 0
    unsolved = {}
    unmet = list(range(len(dataset_train)))
    pre_encode = None
    cfg.module_used = [False] * cfg.decoder_word_size

  if cfg.load_search_log is not None :
    print("Search log: loading")
    log_c = 0
    with open(cfg.load_search_log, "r") as f :
      try :
        while True :
          succ = f.readline()
          if len(succ) == 0 :
            break
          succ = eval(succ)
          fail = eval(f.readline())
          pgms = eval(f.readline())
          flag = f.readline()

          solved.update(succ)
          for idx in succ :
            if idx in unsolved :
              unsolved.pop(idx)
            if idx in unmet :
              unmet.remove(idx)
          for idx in fail :
            if idx in unsolved :
              unsolved[idx][1] += 1
              unsolved[idx][0] = None
            else :
              unsolved[idx] = [None, 1, 0]
          for idx, (rate, pgm) in pgms.items() :
            idx = int(idx)
            dataset_train.programs[idx] = tuple(pgm)
            dataset_train.programs_raw[idx] = MathPgmFunc.program_p2r(pgm, pad = cfg.decoder_max_len)
            dataset_train.programs_rate[idx] = rate
            dataset_train.weight[idx] = len(dataset_train.data_valid[idx]["idx"])**0.5 * rate
            dataset_sampled.add(idx)

            for a in pgm :
              if a >= 0 :
                cfg.module_used[a] = True
          log_c += 1
      except Exception as e :
        print("loading interrupted at line", log_c*4+1)
        print(e)
    for idx, rec in unsolved.items() :
      if idx in unmet :
        unmet.remove(idx)
      if rec[0] is None :
        txt = dataset_train.data_valid[idx]["vocs"]
        l = len(txt)
        t = rec[1]
        weight = (5 / l ** 0.5) * (1.0 / t)
        rec[0] = weight
    solved_count = len(solved)
    print("Search log loaded :", cfg.load_search_log)
    print(log_c, "logs,", log_c*4, "lines loaded.")
  print(f"solved :{solved_count:>4d},   unsolved :{len(unsolved):>4d},   unmet :{len(unmet):>4d}")
  print()

  policynet_trained = False
  if cfg.policynet_load_weight is not None :
    print("Save point: loading")
    policynet_trained = True
    sd_loaded = torch.load(cfg.policynet_load_weight)
    sd_new = _PolicyNet_.state_dict()
    for k,p in sd_loaded.items() :
      if k in sd_new :
        sd_new[k].copy_(p)
    print("Save point loaded :", cfg.policynet_load_weight)
    print()

  if cfg.policynet_retrain_after_load :
    print("PolicyNet: training on",len(dataset_sampled),"samples")
    policynet_trained = True
    dataloader = torch.utils.data.DataLoader(dataset_sampled, batch_size=64, collate_fn=DatasetMath.collate_fn, shuffle=True, num_workers=8)
    c_batch = 0
    try :
      while True :
        for q, _, _, p, w in dataloader :
          q = torch.tensor(q, dtype=torch.long, device=cfg.device)
          p = torch.tensor(p, dtype=torch.long, device=cfg.device)
          w = torch.tensor(w, dtype=torch.float, device=cfg.device)
          _PolicyNet_.set_batch_data({"x":q, "y":p, "w":w})
          optimizer.zero_grad()

          _ = _PolicyNet_.program_predictor()
          _, accu_p = _PolicyNet_.program_predictor.get_accuracy()
          accu_p = accu_p[-1]
          loss_p = _PolicyNet_.program_predictor.backward()

          _ = _PolicyNet_.necessity_predictor()
          accu_n = _PolicyNet_.necessity_predictor.get_accuracy(mask=cfg.module_used)
          loss_n = _PolicyNet_.necessity_predictor.backward(mask=cfg.module_used)

          optimizer.step()
          print(f"\r {c_batch:>6d}  {accu_p:>6.4f}  {loss_p:>8.6f}  {accu_n:>6.4f}  {loss_n:>8.6f}", end="")

          c_batch += 1
          if c_batch >= cfg.policynet_pretrain_batch :
            raise StopIteration
    except StopIteration :
      pass
    except KeyboardInterrupt :
      print("\nTraining interrupted.")
      pass
    finally :
      timestr = datetime.datetime.now().strftime("%Y%m%d")+"_"+datetime.datetime.now().strftime("%H%M%S")
      torch.save(_PolicyNet_.state_dict_ex(), cfg.saves_path + "/savepoint_policynet_" + timestr)
    print()

  if policynet_trained and len(dataset_sampled) > 0 :
    _PolicyNet_.eval()
    dataloader = torch.utils.data.DataLoader(dataset_sampled, batch_size=256, collate_fn=DatasetMath.collate_fn, shuffle=False, num_workers=0)
    encoded_solved = []
    done_c = 0
    for que, _, _, _, _ in dataloader :
      q = torch.tensor(que, dtype=torch.long, device=cfg.device)
      _PolicyNet_.set_batch_data({"x" : q, "y" : None})
      with torch.no_grad() :
        _PolicyNet_.program_predictor()
      hidden = _PolicyNet_.program_predictor.batch_data["encoded_hidden"]
      encoded = hidden[0].permute(1,0,2).reshape(-1, 1024)
      encoded_solved.append(encoded)
      done_c += len(que)
      print(f"\rEncoding ... {done_c:>5d}/{len(dataset_sampled)}", end = "")
    encoded_solved = torch.cat(encoded_solved, 0)
    dataset_train.pre_encode = (dataset_sampled.samples, encoded_solved)
    print("\nEncoding done.")
    torch.cuda.empty_cache()
  else :
    dataset_train.pre_encode = None

  protected_memory = []
  memory_pool = []
  for _ in range(cfg.MAX_PROCESSES_NUMBER) :
    m = numpy.empty((101, cfg.max_program_length+4))
    sm = multiprocessing.shared_memory.SharedMemory(create=True, size=m.nbytes)
    sna = numpy.ndarray(m.shape, dtype=m.dtype, buffer=sm.buf)
    protected_memory.append(sm)
    memory_pool.append(sna)
  try :
    for loop in range(10000000) :
      print("="*64)
      print(f"loop :{loop:>4d},   solved :{solved_count:>4d},   unsolved :{len(unsolved):>4d},   unmet :{len(unmet):>4d}")

      unmet_idx = 0
      process_logs = []
      process_finished = []
      process_pool = {}

      for memory in memory_pool :
        _try_new = (unmet_idx < len(unmet)) and (random.random() <= max(math.exp(-len(unsolved) / (solved_count + 1)), 0.1))
        if _try_new :
          _que_idx = unmet[unmet_idx]
          unmet_idx += 1
        else :
          _que_idx = random.choices(*zip(*[(k, v[0]) for k, v in unsolved.items()]))[0]
        print("On:", _que_idx, "new" if _try_new else "retry")

        pid = len(process_logs)
        param = search_preprocess(cfg, dataset_train, _que_idx, _PolicyNet_)
        _que_T = param.que_T
        del param.que_T
        param.process_id = pid
        param.actionnet = _ActionNet_
        param.shared_memory = memory
        memory[0][0] = pid
        memory[0][1] = 0

        process = multiprocessing.Process(target=search_main, args=(cfg, param))
        process_logs.append([process, _try_new, _que_idx, _que_T, param, 0, None, 0.0])
        process_finished.append(False)
        process_pool[pid] = process
        process.start()

      try :
        with torch.no_grad() :
          while True :
            for memory in memory_pool :
              if memory[0][1] == 1 :
                pid = int(memory[0][0])
                yn = int(memory[0][2])
                yl = int(memory[0][3])
                x = process_logs[pid][3].expand(yn, -1)
                y = torch.tensor(memory[1:1+yn, :yl], dtype=torch.long, device=cfg.device)
                score_pgm, score_per_act = _PolicyNet_.program_predictor.score(x, y, '+')
                scores = []
                memory[1:1+yn, 0] = score_pgm
                memory[1:1+yn, 1:yl-1] = score_per_act
                memory[0][1] = 0
              elif memory[0][1] == 2 :
                pid = int(memory[0][0])
                step = int(memory[0][2])
                accu = float(memory[0][3])
                if accu > 0 :
                  succ = 2 if accu >= cfg.actions_accept_boundary else 1
                  len_act = int(memory[1][0])
                  best_act = memory[1][1:1+len_act].astype(int).tolist()
                else :
                  succ = 0
                  best_act = None
                process = process_pool.pop(pid)
                process.join()
                process.close()

                print()
                print(f"process {pid} finished,", "new," if process_logs[pid][1] else "retry,",
                      f"idx {process_logs[pid][2]}, success {succ}")
                print(process_logs[pid][4].num, process_logs[pid][4].ans)
                print(step, best_act, accu)
                print()
                process_finished[pid] = True
                process_logs[pid][5:] = [succ, best_act, accu]

                if all(process_finished[:cfg.MAX_PROCESSES_NUMBER]) :
                  for pid, process in process_pool.items() :
                    process.terminate()
                    process.join()
                    process.close()
                  print("shut all down")
                  raise StopIteration

                _try_new = (unmet_idx < len(unmet))
                if _try_new :
                  _que_idx = unmet[unmet_idx]
                  unmet_idx += 1
                else :
                  _que_idx = random.choices(*zip(*[(k, v[0]) for k, v in unsolved.items()]))[0]
                print("On:", _que_idx, "new" if _try_new else "retry")

                pid = len(process_logs)
                param = search_preprocess(cfg, dataset_train, _que_idx, _PolicyNet_)
                _que_T = param.que_T
                del param.que_T
                param.process_id = pid
                param.actionnet = _ActionNet_
                param.shared_memory = memory
                memory[0][0] = pid
                memory[0][1] = 0

                process = multiprocessing.Process(target=search_main, args=(cfg, param))
                process_logs.append([process, _try_new, _que_idx, _que_T, param, 0, None, 0.0])
                process_finished.append(False)
                process_pool[pid] = process
                process.start()
      except StopIteration :
        pass

      def iter_results() :
        for finished, (_process_id, _try_new, _idx, _que_T, _param, _succ, _pgm, _rate) in zip(process_finished, process_logs) :
          if finished and not _idx in solved_new :
            yield (_try_new, _idx, _param.que, _succ, _pgm, _rate, _que_T)

      solved_new = []
      unsolved_new = []
      update_with = []
      pgm_log = {}
      q0_raw = []
      p0_raw = []
      w0_raw = []
      for _try_new, _idx, _que, _succ, _pgm, _rate, _que_T in iter_results() :
        if _succ > 0 and _rate > dataset_train.programs_rate[_idx] :
          update_with.append(_idx)
          dataset_sampled.add(_idx)
          for a in _pgm :
            if a >= 0 :
              cfg.module_used[a] = True
          pgm_raw = MathPgmFunc.program_p2r(_pgm, pad=cfg.decoder_max_len)
          weight = len(dataset_train.data_valid[_idx]["idx"])**0.5 * _rate
          dataset_train.programs[_idx] = tuple(_pgm)
          dataset_train.programs_raw[_idx] = tuple(pgm_raw)
          dataset_train.programs_rate[_idx] = _rate
          dataset_train.weight[_idx] = weight
          pgm_log[_idx] = (_rate, _pgm)
          q0_raw.append(_que_T)
          p0_raw.append(pgm_raw)
          w0_raw.append(weight)

        if _succ >= 2 :
          solved.add(_idx)
          solved_count += 1
          if _try_new :
            unmet.remove(_idx)
          else :
            unsolved.pop(_idx)

          solved_new.append(_idx)
          while _idx in unsolved_new :
            unsolved_new.pop(unsolved_new.index(_idx))
        else :
          if _idx in solved_new :
            continue
          unsolved_new.append(_idx)
          if _try_new :
            if _idx in unmet :
              unmet.remove(_idx)

            l = _que.index(2) - 1
            weight = (5 / l ** 0.5) * 1.0
            unsolved[_idx] = [weight, 1, _rate]
          else :
            l = _que.index(2) - 1
            t = unsolved[_idx][1] + 1
            weight = (5 / l ** 0.5) * (1.0 / t)
            unsolved[_idx] = [weight, t, _rate]

      if loop == 0 and not os.path.isfile(log_path := cfg.saves_path + "/search_log_" + cfg.DATE) :
        if cfg.load_search_log is not None :
          with open(cfg.load_search_log, 'r') as fi :
            with open(log_path, "a+") as fo :
              fo.write(fi.read())
      with open(log_path, "a+") as f :
        f.write(str(solved_new) + "\n")
        f.write(str(unsolved_new) + "\n")
        f.write(json.dumps(pgm_log) + "\n")
        f.write("\n")

      if len(update_with) > 0 :
        print("update with", update_with)
        q0 = torch.cat(q0_raw, 0)
        p0 = torch.tensor(p0_raw, dtype=torch.long, device=cfg.device)
        w0 = torch.tensor(w0_raw, dtype=torch.float, device=cfg.device)
        dataloader = torch.utils.data.DataLoader(dataset_sampled, batch_size=64, collate_fn=DatasetMath.collate_fn, shuffle=True, num_workers=0)

        _PolicyNet_.train()
        c_batch = 0
        try :
          while True :
            for q, _, _, p, w in dataloader :
              q = torch.cat((q0, torch.tensor(q, dtype=torch.long, device=cfg.device)))
              p = torch.cat((p0, torch.tensor(p, dtype=torch.long, device=cfg.device)))
              w = torch.cat((w0, torch.tensor(w, dtype=torch.float, device=cfg.device)))
              _PolicyNet_.set_batch_data({"x":q, "y":p, "w":w})
              optimizer.zero_grad()

              _ = _PolicyNet_.program_predictor()
              _, accu_p = _PolicyNet_.program_predictor.get_accuracy()
              accu_p = accu_p[-1]
              loss_p = _PolicyNet_.program_predictor.backward()

              _ = _PolicyNet_.necessity_predictor()
              accu_n = _PolicyNet_.necessity_predictor.get_accuracy(mask=cfg.module_used)
              loss_n = _PolicyNet_.necessity_predictor.backward(mask=cfg.module_used)

              optimizer.step()
              print(f"\r {c_batch:>3d}  {accu_p:>6.4f}  {loss_p:>8.6f}  {accu_n:>6.4f}  {loss_n:>8.6f}", end="")

              c_batch += 1
              if c_batch >= cfg.policynet_train_batch_perloop:
                raise StopIteration
        except StopIteration :
          pass
        print()

        _PolicyNet_.eval()
        dataloader = torch.utils.data.DataLoader(dataset_sampled, batch_size=256, collate_fn=DatasetMath.collate_fn, shuffle=False, num_workers=0)
        encoded_solved = []
        done_c = 0
        for que, _, _, _, _ in dataloader :
          q = torch.tensor(que, dtype=torch.long, device=cfg.device)
          _PolicyNet_.set_batch_data({"x" : q, "y" : None})
          with torch.no_grad() :
            _PolicyNet_.program_predictor()
          hidden = _PolicyNet_.program_predictor.batch_data["encoded_hidden"]
          encoded = hidden[0].permute(1,0,2).reshape(-1, 1024)
          encoded_solved.append(encoded)
          done_c += len(que)
          print(f"\rEncoding ... {done_c:>5d}/{len(dataset_sampled)}", end="")
        encoded_solved = torch.cat(encoded_solved, 0)
        dataset_train.pre_encode = (dataset_sampled.samples, encoded_solved)
        print("\nEncoding done.")
        torch.cuda.empty_cache()

  except :
    timestr = datetime.datetime.now().strftime("%Y%m%d")+"_"+datetime.datetime.now().strftime("%H%M%S")
    torch.save(_PolicyNet_.state_dict_ex(), cfg.saves_path + "/savepoint_policynet_" + timestr)
    raise

  return

def search_preprocess(cfg, dataset, idx, _PolicyNet_) :
  param = CustomData()

  que, num, ans, _, _ = dataset[idx]
  que_T = torch.tensor([que], dtype=torch.long, device=cfg.device)
  param.que = que
  param.que_T = que_T
  param.num = num
  param.ans = ans
  print(f"\tnumbers: {num}  answer: {ans}")
  param.graph = MCGSGraph(config = cfg, param = param)

  _PolicyNet_.set_batch_data({"x" : que_T, "y" : None})
  _PolicyNet_.eval()
  with torch.no_grad() :
    pgm_predicted = _PolicyNet_.program_predictor()[0].tolist()
  pgm_predicted = MathPgmFunc.program_r2p(pgm_predicted)
  pgm_predicted = MathPgmFunc.program_regularizing(pgm_predicted, cfg)

  if cfg.activate_candidate :
    with torch.no_grad() :
      candidate_scores = _PolicyNet_.necessity_predictor()[0][:cfg.current_module_size]

    topv, topi = candidate_scores.topk(cfg.n_candidate[1] - cfg.n_candidate[0])
    sampled_idxs = random.sample(set(range(cfg.current_module_size)) - set(topi.tolist()), cfg.n_candidate[0])
    candidate_scores[sampled_idxs] += 0.5
    candidate_scores = candidate_scores.clamp(topv.min(), topv.max())
    candidates = sorted(sampled_idxs + topi.tolist())
    selected_scores = candidate_scores[candidates]

    param.candidate_scores = candidate_scores.tolist()
    param.available_modules = candidates
    print("\tcandidates:", candidates)
  else :
    param.candidate_scores = [0] * cfg.current_module_size
    param.available_modules = list(range(cfg.current_module_size))

  score_ranking = []
  pgm_default = (0, -1)
  prior_programs = {}
  prior_programs[pgm_default] = ActionOps.find_illegality(pgm_default, cfg.type_match_table)
  if pgm_predicted is not None :
    pgm_predicted = tuple(pgm_predicted)
    prior_programs[pgm_predicted] = ActionOps.find_illegality(pgm_predicted, cfg.type_match_table)
  if dataset.pre_encode is not None :
    idx_solved, encoded_solved = dataset.pre_encode
    hidden = _PolicyNet_.program_predictor.batch_data["encoded_hidden"]
    encoded_cur = hidden[0].permute(1,0,2).reshape(-1, 1024)
    nearest_idx = ((encoded_solved - encoded_cur.expand(len(idx_solved), -1)) ** 2).sum(-1).min(0)[1].item()
    nearest_idx = idx_solved[nearest_idx]
    pgm_nearest = dataset.programs[nearest_idx]
    prior_programs[pgm_nearest] = 0
  else :
    pass

  for pgm, ill in prior_programs.items() :
    print(pgm, ill)
    raw = MathPgmFunc.program_p2r(pgm)
    with torch.no_grad() :
      score_pgm, score_per_act = _PolicyNet_.program_predictor.score(que_T, torch.tensor([raw], dtype=torch.long, device=cfg.device), "+")
    node = MCGSNode(config = cfg, param = param, action_list = pgm, actions_illegality = ill,
                    prior_score = score_pgm[0], action_program_score = numpy.array(score_per_act[0]))
    print(score_pgm[0], score_per_act[0])
    print(node.explore_estimation)
    print()
    score_ranking.append((node.search_score, node))
  for n in param.graph.all_nodes.values() :
    n.update_score_self()
  param.graph.score_ranking = sorted(score_ranking, key = lambda x: x[0], reverse = True)

  return param


def search_main(cfg, param) :
  print(f"process {param.process_id} start on pid {os.getpid()}")

  ActionOps.modules_meta_info = cfg.modules_meta_info
  graph = param.graph
  score_ranking = graph.score_ranking
  best = (None, 0)

  for step in range(cfg.max_search_steps) :
    if cfg.SEARCH_LOG_LEVEL > 0 :
      print("Step :", step)
    with_random = random.random() < 0.5
    if with_random :
      idxe = int(random.random() * len(score_ranking))
      while score_ranking[0][0] - score_ranking[idxe][0] > 0.02 :
        idxe //= 2
    else :
      idxe = 0
    node_to_explore = score_ranking[idxe][1]

    eval_accuracy, new_nodes = graph.explore(node_to_explore)

    if eval_accuracy > best[1] :
      best = (node_to_explore, eval_accuracy)
    if eval_accuracy >= cfg.actions_accept_boundary :
      break

    ranking_op = []
    node_to_explore.recalculate_score()
    ranking_op.append((idxe, -1, 0, None))
    if node_to_explore.visit_flag < 2 :
      ranking_op.append((score_locating(score_ranking, node_to_explore.search_score), 1, node_to_explore.search_score, node_to_explore))

    node_to_explore.update_score_explored([(n, 0) for n in new_nodes])
    for n in new_nodes :
      ranking_op.append((score_locating(score_ranking, n.search_score), 1, n.search_score, n))

    ranking_op = sorted(ranking_op, key = lambda x: (x[0], -x[2]))
    sp = 0
    score_ranking_new = []
    for op in ranking_op :
      score_ranking_new += score_ranking[sp:op[0]]
      if op[1] > 0 :
        score_ranking_new.append((op[2], op[3]))
        sp = op[0]
      else :
        sp = op[0] + 1
    score_ranking = score_ranking_new + score_ranking[sp:]
    if len(score_ranking) == 0 :
      break

  memory = param.shared_memory
  memory[0][2] = step
  if best[0] is None :
    memory[0][3] = -1
  else :
    best_act = best[0].action_list
    len_act = len(best_act)
    memory[0][3] = best[1]
    memory[1][0] = len_act
    memory[1][1:1+len_act] = best_act
  memory[0][1] = 2
  return

def score_locating(l, s) :
  i = 0
  j = len(l)
  while j - i > 1 :
    if l[(i + j) // 2][0] >= s :
      i = (i + j) // 2
    else :
      j = (i + j) // 2
  return j

if __name__ == "__main__" :
  print("\n================================")
  cfg = TotalConfig()
  # torch.cuda.set_device(0)
  multiprocessing.set_start_method("fork")
  train(cfg)
  print("================================\n")
