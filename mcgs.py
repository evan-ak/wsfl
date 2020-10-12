
import time
import numpy
import torch

from dataset import MathPgmFunc

class MCGSGraph() :
  def __init__(self, config, param) :
    self.cfg = config
    self.param = param
    self.all_nodes = {}
    self.score_ranking = []

  def explore(self, at_node) :
    assert len(at_node.explore_estimation) > 0

    idx_e, _ = at_node.explore_estimation.pop(0)
    at_node.visit_count += 1
    if at_node.visit_count == len(at_node.action_list) :
      at_node.visit_flag = 2
    else :
      at_node.visit_flag = 1

    if at_node.eval_accuracy is None :
      at_node.evaluate()

    if self.cfg.SEARCH_LOG_LEVEL > 0 :
      print("exploring at", at_node.action_list, ":", idx_e)

    node_e = at_node.action_tree[idx_e]
    for i, c in enumerate(node_e.parent.children) :
      if c is node_e :
        node_e_child_idx = i
        break
    new_attempts = {}
    modules_meta_info = self.cfg.modules_meta_info
    type_match_table = self.cfg.type_match_table

    # Insert
    if type_match_table[node_e.action][node_e.parent.action][node_e_child_idx] :
      illegality_raw = at_node.actions_illegality
    else :
      illegality_raw = at_node.actions_illegality - 1
    if len(at_node.action_list) < at_node.cfg.actions_length_limit :
      for i in at_node.param.available_modules :
        if len(at_node.action_list) + modules_meta_info[i]["shape"][0] > self.cfg.actions_length_limit :
          continue
        if modules_meta_info[i]["level"] > 0 and len(at_node.action_list) + len(advanced_actions[i]) > self.cfg.actions_length_limit :
          continue
        illegality_up = illegality_raw
        if not type_match_table[i][node_e.parent.action][node_e_child_idx] :
          illegality_up += 1
          if illegality_up > self.cfg.TYPE_ILLEGALITY_TOLERANCE :
            continue
        for j in range(modules_meta_info[i]["shape"][0]) :
          illegality = illegality_up
          for k in range(modules_meta_info[i]["shape"][0]) :
            if not type_match_table[node_e.action if k==j else -1][i][k] :
              illegality += 1
              if illegality > self.cfg.TYPE_ILLEGALITY_TOLERANCE :
                break
          if illegality > self.cfg.TYPE_ILLEGALITY_TOLERANCE :
            continue

          nodes_new_attempt = ActionOps.list_to_tree(at_node.action_list, need_root=True)
          vroot = nodes_new_attempt[0].parent
          node_add_before = nodes_new_attempt[idx_e]
          node_add_after = node_add_before.parent
          node_add = ActionNode(i, node_add_after)
          node_add_before.parent = node_add
          for k, c in enumerate(node_add_after.children) :
            if c is node_add_before :
              node_add_after.children[k] = node_add
              break
          for k in range(modules_meta_info[i]["shape"][0]) :
            node_add.children.append(node_add_before if k == j else ActionNode(-1, node_add))

          new_attempts[ActionOps.tree_to_list(vroot.children[0])] = illegality

    if illegality_raw > 0 :
      for k, c in enumerate(node_e.children) :
        if not type_match_table[c.action][node_e.action][k] :
          illegality_raw -= 1
          if illegality_raw <= 0 :
            break

    # Delete
    if len(at_node.action_list) > 2 and node_e.action >= 0 :
      for i in range(modules_meta_info[node_e.action]["shape"][0]) :
        if node_e.parent.action < 0 and node_e.children[i].action < 0 :
          continue
        illegality = illegality_raw
        if not type_match_table[node_e.children[i].action][node_e.parent.action][node_e_child_idx] :
          illegality += 1
          if illegality >= self.cfg.TYPE_ILLEGALITY_TOLERANCE :
            continue

        nodes_new_attempt = ActionOps.list_to_tree(at_node.action_list, need_root=True)
        vroot = nodes_new_attempt[0].parent
        node_delete = nodes_new_attempt[idx_e]
        for k, c in enumerate(node_delete.parent.children) :
          if c is node_delete :
            node_delete.parent.children[k] = node_delete.children[i]
            node_delete.children[i].parent = node_delete.parent
            break

        new_attempts[ActionOps.tree_to_list(vroot.children[0])] = illegality

    # Substitute
    if node_e.action >= 0 :
      for i in self.param.available_modules :
        if i == node_e.action :
          continue
        if modules_meta_info[i]["level"] > 0 and len(at_node.action_list) + len(advanced_actions[i]) - 1 > self.cfg.actions_length_limit :
          continue
        if modules_meta_info[node_e.action]["shape"] != modules_meta_info[i]["shape"] :
          if not self.cfg.ADD_STRUCT_ILLEGAL_ACTIONS :
            continue

        illegality = illegality_raw
        if not type_match_table[i][node_e.parent.action][node_e_child_idx] :
          illegality += 1
          if illegality > self.cfg.TYPE_ILLEGALITY_TOLERANCE :
            continue
        for j in range(modules_meta_info[node_e.action]["shape"][0]) :
          if not type_match_table[node_e.children[j].action][i][j] :
            illegality += 1
            if illegality > self.cfg.TYPE_ILLEGALITY_TOLERANCE :
              break
        if illegality > self.cfg.TYPE_ILLEGALITY_TOLERANCE :
          continue

        nodes_new_attempt = ActionOps.list_to_tree(at_node.action_list, need_root=True)
        vroot = nodes_new_attempt[0].parent
        node_change = nodes_new_attempt[idx_e]
        node_change.action = i

        new_attempts[ActionOps.tree_to_list(vroot.children[0])] = illegality

    i = 0
    new_nodes = []
    if len(new_attempts) > 0 :
      existing_nodes_info = {}
      new_nodes_info = {}
      infers = []
      for k, v in new_attempts.items() :
        if k in self.all_nodes :
          existing_nodes_info[k] = v
        else :
          new_nodes_info[k] = v
          infers.append(MathPgmFunc.program_p2r(k))
      if (infn := len(infers)) > 0 :
        infl = max([len(p) for p in infers])
        infers = [p + [0]*(infl-len(p)) for p in infers]

        memory = self.param.shared_memory
        memory[0][2] = infn
        memory[0][3] = infl
        memory[1:1+infn, :infl] = infers
        memory[0][1] = 1

        while memory[0][1] != 0 :
          pass
        scores = memory[1:1+infn]

        for (k, v), score in zip(new_nodes_info.items(), scores) :
          score_pgm = score[0].item()
          score_per_act = score[1:1+len(k)]
          if self.cfg.SEARCH_LOG_LEVEL > 1 :
            print(k, v, score_pgm)
            print(score_per_act)
          new_node = MCGSNode(config=self.cfg, param=self.param, action_list=k, actions_illegality=v,
                              prior_score=score_pgm, action_program_score=score_per_act)
          new_node.neighbors.append(at_node)
          at_node.neighbors.append(new_node)
          new_nodes.append(new_node)

      for k, v in existing_nodes_info.items() :
        existing_node = self.all_nodes[k]
        if not at_node in existing_node.neighbors :
          existing_node.neighbors.append(at_node)
        if not existing_node in at_node.neighbors :
          at_node.neighbors.append(existing_node)

    return at_node.eval_accuracy, new_nodes


class MCGSNode() :
  def __init__(self, config, param, action_list, action_tree=None, actions_illegality=0,
               prior_score=0, std_score=0, action_program_score=None, action_necessity_score=None) :
    self.cfg = config
    self.param = param
    self.action_list = action_list
    self.action_tree = ActionOps.list_to_tree(self.action_list, need_root=True)
    self.actions_illegality = actions_illegality
    self.prior_score = prior_score
    self.prior_score += ActionOps.find_prior_score_len_bias(self.action_list, alpha = 1.0)
    self.eval_accuracy = None
    self.exprt_score = std_score
    self.total_score = self.prior_score + self.exprt_score

    self.visit_flag = 0
    self.visit_count = 0
    self.neighbors = []
    self.neighbor_score = None
    self.neighbor_score_ref = [self] * len(self.cfg.scoring_weight_decay)
    self.neighbor_score_src = {self.action_list: [0,1,2,3]}
    self.search_score = None
    self.recalculate_score()

    self.param.graph.all_nodes[self.action_list] = self

    if action_program_score is None :
      action_program_score = numpy.random.rand(len(self.action_list))
    if action_necessity_score is None :
      action_necessity_score = numpy.array([(self.param.candidate_scores[a] if a >= 0 else 1) for a in self.action_list])
    self.action_confidence = action_program_score * self.cfg.pgm_nec_confidence_scale[0] + action_necessity_score * self.cfg.pgm_nec_confidence_scale[1]
    self.explore_estimation = sorted(enumerate(self.action_confidence), key=(lambda x: x[1]), reverse=False)

  def evaluate(self) :
    if self.actions_illegality > 0 :
      self.eval_accuracy = 0
      self.exprt_score = 0
      self.total_score = self.prior_score + self.exprt_score
    else :
      val_param = {"num" : self.param.num, "ans" : self.param.ans}
      res, _ = self.param.actionnet.validate(self.action_tree[0], val_param)
      self.eval_accuracy = sum(res)/len(res)
      self.exprt_score = self.eval_accuracy
      self.total_score = self.prior_score + self.exprt_score
    return

  def recalculate_score(self) :
    self.search_score = 0.05 * 1.0 / (self.visit_count + 1)
    for i, w in enumerate(self.cfg.scoring_weight_decay) :
      self.search_score += w * self.neighbor_score_ref[i].total_score

  def update_score_self(self, d0=0, dm=None) :
    if dm is None :
      dm = len(self.cfg.scoring_weight_decay)
    nodes_in_d = [[self]]
    for d in range(1, dm) :
      nodes_in_d.append(nodes_in_d[-1].copy())
      for n in nodes_in_d[-2] :
        for nn in n.neighbors :
          if not nn in nodes_in_d[-1] :
            nodes_in_d[-1].append(nn)
      if d >= d0 :
        scores = [(self.total_score, self)] + [(n.total_score, n) for n in nodes_in_d[-1]]
        score_max = max(scores, key = lambda x: x[0])
        self.neighbor_score_ref[d] = score_max[1]
    self.recalculate_score()

  def update_score_explored(self, new_nodes) :
    cfg = self.cfg
    dm = len(cfg.scoring_weight_decay)
    new_best = (self.total_score, self)
    for d in range(0, dm) :
      for n, l in new_nodes :
        if d == 0 :
          if d >= l :
            n.neighbor_score_ref[0] = n
          if n.total_score > new_best[0] :
            new_best = (n.total_score, n)
        elif d >= l:
          scores = [(n.total_score, n)] + [(self.neighbor_score_ref[d - 1].total_score, self.neighbor_score_ref[d - 1])]
          if d >= 2 :
            scores.append(new_best)
          score_max = max(scores, key = lambda x : x[0])
          n.neighbor_score_ref[d] = score_max[1]
    for n, _ in new_nodes :
      n.recalculate_score()

class ActionNode() :
  def __init__(self, action = None, parent = None) :
    self.action = action
    self.parent = parent
    self.children = []

  def __str__(self) :
    return "action: " + str(self.action) + " parent: " + ("None" if self.parent is None else str(self.parent.action)) + " children: " + str([c.action for c in self.children])

  def __repr__(self) :
    return ("\t" if self.parent is None else "\n\t") + self.__str__()

class ActionOps() :
  modules_meta_info = None

  @staticmethod
  def list_to_tree(action_list, need_root = False) :
    if need_root :
      root = ActionNode(-2, None)
      visit = root
      nodes = []
    else :
      root = None
      visit = ActionNode(action_list[0], None)
      nodes = [visit]
      action_list = action_list[1:]
    for a in action_list :
      nodes.append(ActionNode(a, visit))
      visit.children.append(nodes[-1])
      if a >= 0 :
        visit = visit.children[-1]
      while len(visit.children) == ActionOps.modules_meta_info[visit.action]["shape"][0] :
        visit = visit.parent
        if visit is root :
          break
      else :
        continue
      break
    return nodes

  @staticmethod
  def tree_to_list(input) :
    if type(input) is ActionNode :
      input = ActionOps.node_traversal(input, need_leaf=True)
    return tuple([n.action for n in input])

  @staticmethod
  def node_traversal(root, need_leaf = True) :
    seek = root
    suspended = []
    nodes = []
    while True :
      nodes.append(seek)
      if need_leaf :
        cs = seek.children
      else :
        cs = list(filter(lambda n: n.action >= 0, seek.children))
      if len(cs) > 0 :
        seek = cs[0]
        if len(cs) > 1 :
          suspended += cs[:0:-1]
      else :
        if len(suspended) > 0 :
          seek = suspended.pop(-1)
        else :
          return nodes

  @staticmethod
  def find_illegality(actions, type_match_table) :
    if type(actions) in (list, tuple) :
      nodes = ActionOps.list_to_tree(actions)
    else :
      nodes = ActionOps.node_traversal(root, need_leaf=False)
    illegality = 0
    for n in nodes :
      for i,c in enumerate(n.children) :
        if not type_match_table[c.action][n.action][i] :
          illegality += 1
    return illegality

  @staticmethod
  def find_prior_score_len_bias(action_list, alpha=0.5) :
    return alpha / len(action_list)
