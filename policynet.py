
import torch

from dataset import MathPgmFunc

class PolicyNet(torch.nn.Module) :
  def __init__(self, cfg) :
    super().__init__()
    self.cfg = cfg
    self.program_predictor = PN_ProgramPredictor(cfg)
    self.necessity_predictor = PN_NecessityPredictor(cfg)
    self.loss_scale = (1.0, 1.0)

  def set_batch_data(self, batch_data) :
    if not "w" in batch_data :
      batch_data["w"] = None
    self.program_predictor.batch_data = batch_data

    if not "x_bool" in batch_data :
      batch_data["x_bool"] = MathPgmFunc.seq_to_bool(batch_data["x"], self.cfg.encoder_word_size)
    if not "y_bool" in batch_data and batch_data["y"] is not None :
      batch_data["y_bool"] = MathPgmFunc.seq_to_bool(batch_data["y"], self.cfg.decoder_word_size)
    self.necessity_predictor.batch_data = batch_data

  def forward(self, x=None, y=None, reinforce=False) :
    pred_program = self.program_predictor(x, y, reinforce = reinforce)

    x_bool = MathPgmFunc.seq_to_bool(x, self.cfg.encoder_word_size) if x is not None else None
    pred_necessity = self.necessity_predictor(x_bool)

    return pred_program, pred_necessity

  def backward(self, pred=None, y=None) :
    if type(pred) in (tuple, list) :
      pred_program = pred[0]
      pred_necessity = pred[1]
    elif type(pred) in (torch.Tensor, ) :
      pred_program = pred
      pred_necessity = None
    else :
      pred_program = None
      pred_necessity = None

    loss_program = self.program_predictor.backward(pred_program, y)

    y_bool = MathPgmFunc.seq_to_bool(y, self.cfg.decoder_word_size) if y is not None else None
    loss_necessity = self.necessity_predictor.backward(pred_necessity, y_bool)

    loss = (loss_program * self.loss_scale[0] + loss_necessity * self.loss_scale[1]) / (self.loss_scale[0] + self.loss_scale[1])

    return(loss, loss_full, loss_relation)

  def state_dict_ex(self) :
    sd = self.state_dict()
    for k in filter(lambda k:"trans_cache" in k, list(sd.keys())) :
      sd.pop(k)
    return sd

class PN_ProgramPredictor(torch.nn.Module) :
  def __init__(self, cfg) :
    super().__init__()
    self.cfg = cfg
    self.encoder_word_size = self.cfg.encoder_word_size
    self.encoder_embed_dim = 300
    self.encoder_hidden_dim = 256
    self.use_pretrained_embedding = self.cfg.policynet_load_word_embedding
    if self.use_pretrained_embedding :
      with open(self.cfg.policynet_embedding_mask, 'r') as f :
        encoder_embed_mask = eval(f.read())
      self.encoder_embed_size = (sum(encoder_embed_mask), len(encoder_embed_mask)-sum(encoder_embed_mask))
      self.encoder_embedding_loaded = torch.nn.Embedding(len(encoder_embed_mask), self.encoder_embed_dim)
      self.encoder_embedding_loaded.load_state_dict(torch.load(self.cfg.policynet_embedding_weight))
      self.encoder_embedding_loaded.need_grad = False
      self.encoder_embedding_trainable = torch.nn.Embedding(self.encoder_embed_size[1], self.encoder_embed_dim)

      encoder_embed_trans = (lambda x,y:([y.append(min((not m)+y[-1], self.encoder_embed_size[1]-1)) for m in x],y))(encoder_embed_mask, [0])[-1][:-1]
      self.encoder_embed_trans = torch.nn.Parameter(torch.tensor(encoder_embed_trans, dtype=torch.float, requires_grad=False).unsqueeze(0).unsqueeze(0), requires_grad=False)
      self.encoder_embed_mask = torch.nn.Parameter(torch.tensor(encoder_embed_mask, dtype=torch.float, requires_grad=False).unsqueeze(0).unsqueeze(0), requires_grad=False)
    else :
      self.encoder_embedding = torch.nn.Embedding(self.encoder_word_size, self.encoder_embed_dim)
    self.encoder_rnn = torch.nn.LSTM(input_size = self.encoder_embed_dim,
                                     hidden_size = self.encoder_hidden_dim,
                                     num_layers = 2,
                                     batch_first = True,
                                     dropout = 0,
                                     bidirectional = True)

    self.decoder_word_size = self.cfg.decoder_word_size
    self.decoder_max_len = cfg.decoder_max_len
    self.decoder_embed_dim = 300
    self.decoder_hidden_dim = 2 * self.encoder_hidden_dim
    self.decoder_embeding = torch.nn.Embedding(self.decoder_word_size, self.decoder_embed_dim)
    self.decoder_rnn = torch.nn.LSTM(input_size = self.decoder_embed_dim,
                                     hidden_size = self.decoder_hidden_dim,
                                     num_layers = 2,
                                     batch_first = True,
                                     dropout = 0)
    self.decoder_linear = torch.nn.Linear(self.decoder_hidden_dim, self.decoder_word_size)
    self.decoder_with_attention = True
    self.attention_input_dim = 2 * self.decoder_hidden_dim
    self.attention_mixing = torch.nn.Linear(self.attention_input_dim, self.decoder_hidden_dim)

    self.scoring = False
    self.loss_mask = True
    self.loss_F = torch.nn.CrossEntropyLoss(reduction="none")
    self.batch_data = {}

  def decoder_meta(self, decoder_embeded, decoder_initial, encoder_output) :
    decoder_output, decoder_hidden = self.decoder_rnn(decoder_embeded, decoder_initial)
    if self.decoder_with_attention :
      attention_raw = torch.nn.functional.softmax(torch.bmm(decoder_output, encoder_output.transpose(1, 2)), dim=2)
      attention_combined = torch.cat((decoder_output, torch.bmm(attention_raw, encoder_output)), dim = 2)
      attention_mixxed = self.attention_mixing(attention_combined.view(-1, 2 * self.decoder_hidden_dim))
      linear_input = torch.tanh(attention_mixxed)
    else :
      linear_input = decoder_output.contiguous().view(-1, self.decoder_hidden_dim)
    linear_output = self.decoder_linear(linear_input.contiguous()).view(self.batch_size, -1, self.decoder_word_size)
    return linear_output, decoder_hidden

  def forward(self, x=None, y=None, reinforce=False) :
    if x is None :
      x = self.batch_data["x"]
    self.batch_size = x.shape[0]
    if y is None :
      y = self.batch_data["y"]
    if self.use_pretrained_embedding :
      x_t = torch.gather(self.encoder_embed_trans.long().expand(x.shape+(-1,)), -1, x.unsqueeze(-1)).squeeze(-1)
      encoder_embedded = torch.stack((self.encoder_embedding_trainable(x_t), self.encoder_embedding_loaded(x)), 0)
      mask = torch.gather(self.encoder_embed_mask.long().expand(x.shape+(-1,)), -1, x.unsqueeze(-1)).unsqueeze(0).expand(-1,-1,-1,self.encoder_embed_dim)
      encoder_embedded = torch.gather(encoder_embedded, 0, mask).squeeze(0)
    else :
      encoder_embedded = self.encoder_embedding(x)
    encoder_output, encoder_hidden = self.encoder_rnn(encoder_embedded)
    self.batch_data["encoded_hidden"] = encoder_hidden

    decoder_initial = tuple([torch.cat((h[0::2], h[1::2]), -1) for h in encoder_hidden])
    if self.training or self.scoring:
      decoder_embeded = self.decoder_embeding(y)
      linear_output, decoder_hidden = self.decoder_meta(decoder_embeded, decoder_initial, encoder_output)
      program_fin = linear_output.argmax(-1)
    else :
      decoder_embeded_step = self.decoder_embeding(torch.ones(self.batch_size, 1, dtype=torch.long, device=x.device))
      decoder_hidden_step = decoder_initial
      linear_output = []
      program_fin = []
      for t in range(self.decoder_max_len) :
        decoder_output_step, decoder_hidden_step = self.decoder_meta(decoder_embeded_step, decoder_hidden_step, encoder_output)
        linear_output.append(decoder_output_step)
        if reinforce:
          decoder_output_step = torch.nn.functional.softmax(decoder_output_step[:,:,:self.cfg.current_decode_size].view(-1, self.cfg.current_decode_size), -1)
          pred_step = decoder_output_step.multinomial(1).view(self.batch_size, -1)
        else :
          pred_step = decoder_output_step[:,:,:self.cfg.current_decode_size].argmax(-1)
        program_fin.append(pred_step)
        decoder_embeded_step = self.decoder_embeding(pred_step)
      linear_output = torch.cat(linear_output, 1)
      program_fin = torch.cat(program_fin, 1)
    self.batch_data["pred_program_per"] = linear_output
    self.batch_data["pred_program_fin"] = program_fin

    return self.batch_data["pred_program_fin"]

  def backward(self, pred=None, y=None) :
    if pred is None :
      pred = self.batch_data["pred_program_per"]
    if y is None :
      y = self.batch_data["y"]
    y_r = y[:,1:].contiguous().view(-1)
    loss_raw = self.loss_F(pred[:,:-1,:].contiguous().view(-1, self.decoder_word_size), y_r)
    if self.loss_mask :
      loss_raw = (loss_raw * (y_r > 0).to(loss_raw.dtype))
    w = self.batch_data["w"]
    if w is not None :
      loss = loss_raw.view(len(w), -1).mean(-1)
      loss = (loss * w).mean()
    else :
      loss = loss_raw.mean()
    if self.training :
      loss.backward()
    return loss.item()

  def get_accuracy(self, pred=None, y=None) :
    if pred is None :
      pred = self.batch_data["pred_program_per"]
    if len(pred.shape) > 2 :
      pred = pred.argmax(-1)
    if y is None :
      y = self.batch_data["y"]
    accu_action = [0, 0]
    accu_program = [0, y.shape[0]]
    l_min = min(pred.shape[1], y.shape[1])
    for p_e, y_e in zip(pred[:,:l_min-1], y[:,1:l_min]) :
      l = (y_e == 0).nonzero(as_tuple=False)[0].item() if 0 in y_e else len(y_e)
      m = (p_e[:l] == y_e[:l]).sum().item()
      accu_action[0] += m
      accu_action[1] += l
      accu_program[0] += m == l
    accu_action.append(accu_action[0] / accu_action[1] if accu_action[1] > 0 else None)
    accu_program.append(accu_program[0] / accu_program[1] if accu_program[1] > 0 else None)
    return accu_action, accu_program

  def score(self, x=None, y=None, method="+") :
    self.eval()
    self.scoring = True
    self(x, y)
    linear_output = self.batch_data["pred_program_per"][:,:-1]
    self.scoring = False

    if y is None :
      y = self.batch_data["y"]
    y = y[:,1:]
    action_score = linear_output.softmax(-1).gather(-1, y.unsqueeze(-1)).view(y.shape[0], -1)
    y_mask = (y > 2).to(action_score.dtype)

    if method == "+" :
      program_score = (action_score * y_mask).sum(-1) / y_mask.sum(-1)
    elif method == "*" :
      program_score = ((action_score.log() * y_mask).sum(-1) / y_mask.sum(-1)).exp()

    return (program_score.tolist(), action_score[:,:-1].tolist())

class PN_NecessityPredictor(torch.nn.Module) :
  def __init__(self, cfg) :
    super().__init__()
    self.cfg = cfg
    self.actf = torch.nn.Tanh()
    self.fc = torch.nn.Sequential(torch.nn.Linear(self.cfg.encoder_word_size, 256),
                                  self.actf,
                                  torch.nn.Linear(256, 256),
                                  self.actf,
                                  torch.nn.Linear(256, self.cfg.decoder_word_size),
                                  torch.nn.Sigmoid())
    self.loss_F = torch.nn.MSELoss(reduction = "none")
    self.batch_data = {}

  def forward(self, x=None) :
    if x is None :
      x = self.batch_data["x_bool"]
    self.batch_data["pred_necessity"] = self.fc(x)
    return self.batch_data["pred_necessity"]

  def backward(self, pred=None, y=None, mask=None) :
    if pred is None :
      pred = self.batch_data["pred_necessity"]
    if y is None :
      y = self.batch_data["y_bool"]
    loss_raw = self.loss_F(pred[:,:y.shape[-1]], y)
    if mask is not None :
      loss = (loss_raw * torch.tensor(mask, dtype=pred.dtype, device=pred.device)).mean(-1)
    else :
      loss = loss_raw.mean(-1)
    w = self.batch_data["w"]
    if w is not None :
      loss = (loss * w).mean()
    else :
      loss = loss.mean()
    if self.training :
      loss.backward()
    return loss.item()

  def get_accuracy(self, pred=None, y=None, mask=None) :
    if pred is None :
      pred = self.batch_data["pred_necessity"]
    if y is None :
      y = self.batch_data["y_bool"]
    if mask is None :
      accuracy = ((pred[:,:y.shape[-1]] - y).abs() < 0.5).sum().item() / y.view(-1).shape[0]
    else :
      accuracy = (((pred[:,:y.shape[-1]] - y).abs() < 0.5).sum(0) * torch.tensor(mask, dtype=torch.long, device=pred.device)).sum().item() / (y.shape[0] * sum(mask))
    return accuracy
