
import torch

def Math23Modules() :
  modulelist = []
  for i in range(6) :
    modulelist.append(Module_Math_Key(i))
  for i in (1,100) :
    modulelist.append(Module_Math_Const(i))
  modulelist.append(Module_Math_Plus())
  modulelist.append(Module_Math_Minus())
  modulelist.append(Module_Math_Multiply())
  modulelist.append(Module_Math_Divide())
  return modulelist

def MathQAModules() :
  modulelist = []
  for i in range(20) :
    modulelist.append(Module_Math_Key(i))
  for i in (1,2,3,100) :
    modulelist.append(Module_Math_Const(i))
  modulelist.append(Module_Math_Plus())
  modulelist.append(Module_Math_Minus())
  modulelist.append(Module_Math_Multiply())
  modulelist.append(Module_Math_Divide())
  return modulelist

class ActionNet(torch.nn.Module) :
  def __init__(self, cfg, modulelist) :
    super().__init__()
    self.cfg = cfg
    self.modulelist = modulelist
    self.update_type_info()

  def update_type_info(self) :
    for i, m in enumerate(self.modulelist) :
      self.cfg.modules_meta_info[i] = {"level":0, "shape":m.module_shape, "data_type_in":m.data_type_in, "data_type_out":m.data_type_out}
    self.cfg.modules_meta_info[-2] = {"level" : 0, "shape" : (1, 0), "data_type_in" : (("none", "number"), ), "data_type_out": None}
    self.cfg.modules_meta_info[-1] = {"level" : 0, "shape" : (0, 1), "data_type_in" : None, "data_type_out": ("none", )}
    self.cfg.modules_meta_info[-3] = {"level" : 0, "shape" : (0, 1), "data_type_in" : None, "data_type_out": ("none", )}

    ds = list(enumerate(self.cfg.all_datatype))
    for v in self.cfg.modules_meta_info.values() :
      if (dti := v["data_type_in"]) is None :
        v["data_type_in_b"] = None
      else :
        v["data_type_in_b"] = tuple([sum([1<<i for i,t in ds if t in c]) for c in dti])
      if (dto := v["data_type_out"]) is None :
        v["data_type_out_b"] = None
      else :
        v["data_type_out_b"] = sum([1<<i for i,t in ds if t in dto])
    type_match_table = {}
    for io,mo in self.cfg.modules_meta_info.items() :
      if (mob := mo["data_type_out_b"]) is None :
        pass
      else :
        type_match_table[io] = {}
        for ii,mi in self.cfg.modules_meta_info.items() :
          if (mib := mi["data_type_in_b"]) is None :
            pass
          else :
            type_match_table[io][ii] = [(mob & mibi > 0) for mibi in mib]
    self.cfg.type_match_table = type_match_table

  def forward_step(self, action_node, param) :
    if action_node.action == -1 :
      return [None] * len(param)
    else :
      inputs = []
      for i, c in enumerate(action_node.children) :
        c_out = self.forward_step(c, param)
        inputs.append(c_out)
      inputs.append(param)
    m = self.modulelist[action_node.action]
    return [(None if input[-1]["failed"] else try_module(m, input)) for input in zip(*inputs)]

  def validate(self, action_root, param) :
    ans = param["ans"]
    param = [{"num":num, "failed":(a is None)} for num,a in zip(param["num"], ans)]
    res = self.forward_step(action_root, param)
    return ([(a is not None and r is not None and abs(r - a) <= 1e-6) for r,a in zip(res, ans)], res)


def try_module(module, input) :
  try :
    return module(*input)
  except Exception as e:
    input[-1]["failed"] = True
    return None

class Module_Math_Key(torch.nn.Module) :
  def __init__(self, idx = 0) :
    super().__init__()
    self.module_name = "Key_" + str(idx)
    self.idx = idx
    self.module_shape = (1, 1)
    self.data_type_in = (("none",),)
    self.data_type_out = ("number",)

  def forward(self, v, param) :
    assert self.idx < len(param["num"])
    return param["num"][self.idx]

class Module_Math_Plus(torch.nn.Module) :
  def __init__(self) :
    super().__init__()
    self.module_name = "Plus"
    self.module_shape = (2, 1)
    self.data_type_in = (("number",), ("number",))
    self.data_type_out = ("number",)

  def forward(self, a, b, param) :
    return a + b

class Module_Math_Minus(torch.nn.Module) :
  def __init__(self) :
    super().__init__()
    self.module_name = "Minus"
    self.module_shape = (2, 1)
    self.data_type_in = (("number",), ("number",))
    self.data_type_out = ("number",)

  def forward(self, a, b, param) :
    return a - b

class Module_Math_Multiply(torch.nn.Module) :
  def __init__(self) :
    super().__init__()
    self.module_name = "Multiply"
    self.module_shape = (2, 1)
    self.data_type_in = (("number",), ("number",))
    self.data_type_out = ("number",)

  def forward(self, a, b, param) :
    return a * b

class Module_Math_Divide(torch.nn.Module) :
  def __init__(self) :
    super().__init__()
    self.module_name = "Divide"
    self.module_shape = (2, 1)
    self.data_type_in = (("number",), ("number",))
    self.data_type_out = ("number",)

  def forward(self, a, b, param) :
    return a / b

class Module_Math_Const(torch.nn.Module) :
  def __init__(self, value = 0) :
    super().__init__()
    self.module_name = "Const_" + str(value)
    self.const_value = value
    self.module_shape = (1, 1)
    self.data_type_in = (("none",),)
    self.data_type_out = ("number",)

  def forward(self, v, param) :
    return self.const_value
