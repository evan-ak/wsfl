
import os
import datetime

class TotalConfig() :
  def __init__(self) :
    self.encoder_word_size = 12000
    self.decoder_word_size = 32
    self.max_program_length = 48
    self.decoder_max_len = self.max_program_length + 2

    self.all_datatype = ("none", "number", )
    self.modules_meta_info = {}
    self.scoring_weight_decay = (0.5, 0.25, 0.15, 0.1)
    self.pgm_nec_confidence_scale = (0.5, 0.5)
    self.max_search_steps = 10000
    self.actions_length_limit = self.max_program_length
    self.actions_accept_boundary = 0.9
    self.TYPE_ILLEGALITY_TOLERANCE = 2
    self.ADD_STRUCT_ILLEGAL_ACTIONS = False

    self.activate_candidate = True
    self.MAX_PROCESSES_NUMBER = 8
    self.SEARCH_LOG_LEVEL = 0   # 0 none, 1 only exploring, 2 full

    self.DATE = datetime.datetime.now().strftime("%m%d")
    self.local_path = os.path.abspath(".")

    self.on_dataset = "mathqa"
    self.saves_path = self.local_path + "/saves/" + self.on_dataset

    if self.on_dataset == "math23" :
      self.path_data = self.local_path + "/data/math23/math23k_reg.json"
      self.policynet_load_word_embedding = False
      self.policynet_embedding_mask = self.local_path + "/data/math23/math23_vocab_mask"
      self.policynet_embedding_weight = self.local_path + "/data/math23/math23_vocab_weight"
      self.test_metric = ("5flod", "public")[0]
      self.test_split_seed = "standard seed 0"
    if self.on_dataset == "mathqa" :
      self.path_data_train = self.local_path + "/data/mathqa/mathqa_train_reg.json"
      self.path_data_test = self.local_path + "/data/mathqa/mathqa_test_reg.json"
      self.use_annotation = True
      self.policynet_load_word_embedding = False
      self.policynet_embedding_mask = self.local_path + "/data/mathqa/mathqa_vocab_mask"
      self.policynet_embedding_weight = self.local_path + "/data/mathqa/mathqa_vocab_weight"

    self.policynet_train_batch_perloop = 500
    self.policynet_pretrain_batch = 20000

    self.load_search_log = None
    # self.load_search_log = self.local_path + "/saves/mathqa/search_log_fin"

    self.policynet_load_weight = None
    # self.policynet_load_weight = self.local_path + "/saves/mathqa/savepoint_policynet_"

    ## if you loaded only the search log but not the model weight, a retraining is recommended.
    ## if both search log and model weight are loaded, retraining is not necessary.
    self.policynet_retrain_after_load = (False, True)[1]


class CustomData() :
  def __init__(self, dic = {}):
    for k, v in dic.items() :
      setattr(self, k, v)
