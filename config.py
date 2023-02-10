import logging
import time
import configparser

class _Config:
    def __init__(self):
        self._init_logging_handler()
        self.cuda_device = 0        
        self.eos_m_token = 'EOS_M'       
        self.beam_len_bonus = 0.6

        self.mode = 'unknown'
        self.m = 'TSD'
        self.prev_z_method = 'none'
        self.dataset = 'unknown'
        self.exp_setting = 'unknown'

        self.seed = 0
  
    def init_handler(self, m):
        init_method = {
            'tsdf-camrest':self._camrest_tsdf_init,
            'tsdf-kvret':self._kvret_tsdf_init
        }
        init_method[m]()

    def _camrest_tsdf_init(self):
        self.beam_len_bonus = 0.5
        self.prev_z_method = 'separate'
        self.vocab_size = 800 # maybe bs diubah2
        self.embedding_size = 300 # 50 for glove, 300 for fasttext
        self.hidden_size = 50
        self.split = (3, 1, 1)
        self.lr = 0.003
        self.lr_decay = 0.5
        self.vocab_path = './vocab/vocab-camrest.pkl'
        # indo
        # self.data = './data/CamRest676/IndoCamRest/IndoCamRest676.json'
        # self.entity = './data/CamRest676/IndoCamRest/ontology_indo.json'
        # self.db = './data/CamRest676/IndoCamRest/KB_indo.json'
        # self.fasttext_path = './data/fasttext/cc.id.300.vec'

        # eng
        # self.data = './data/CamRest676/CamRest/CamRest676.json'
        # self.entity = './data/CamRest676/CamRest/CamRestOTGY.json'
        # self.db = './data/CamRest676/CamRest/CamRestDB.json'
        # self.fasttext_path = './data/fasttext/cc.en.300.vec'

        # x-lang
        self.fasttext_path = './data/fasttext/wiki.multi.id.vec'
        self.vocab_data_path = './data/CamRest676/bi/IndoCamRest676_bi.json'
        self.vocab_db_path = './data/CamRest676/bi/kb_indo_bi.json'
        # # train
        self.data = './data/CamRest676/CamRest/CamRest676.json'
        self.entity = './data/CamRest676/CamRest/CamRestOTGY.json'
        self.db = './data/CamRest676/CamRest/CamRestDB.json'
        # # test
        # self.data = './data/CamRest676/IndoCamRest/IndoCamRest676.json'
        # self.entity = './data/CamRest676/IndoCamRest/ontology_indo.json'
        # self.db = './data/CamRest676/IndoCamRest/KB_indo.json'

        # bi
        # self.fasttext_path = './data/fasttext/wiki.multi.id.vec'
        # self.data = './data/CamRest676/bi/IndoCamRest676_bi.json'
        # self.entity = './data/CamRest676/bi/ontology_indo_bi.json'
        # self.db = './data/CamRest676/bi/KB_indo.json'

        self.batch_size = 32
        self.z_length = 8
        self.degree_size = 5
        self.layer_num = 1
        self.dropout_rate = 0.5
        self.epoch_num = 100 # triggered by early stop
        self.rl_epoch_num = 1
        self.cuda = False
        self.spv_proportion = 100
        self.max_ts = 40
        self.early_stop_count = 3
        self.new_vocab = True
        self.model_path = './models/camrest.pkl'
        self.result_path = './results/camrest-rl.csv'
        self.teacher_force = 100
        self.beam_search = False
        self.beam_size = 10
        self.sampling = False
        self.use_positional_embedding = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = False
        self.pretrain = False

    def _kvret_tsdf_init(self):
        self.prev_z_method = 'separate'
        self.intent = 'all'
        self.vocab_size = 1400
        self.embedding_size = 300
        self.hidden_size = 50
        self.split = None
        self.lr = 0.003
        self.lr_decay = 0.5
        self.vocab_path = './vocab/vocab-kvret.pkl'
        # indo
        # self.fasttext_path = './data/fasttext/cc.id.300.vec'
        # self.train = './data/kvret/indosmd/IndoSMD_train.json'
        # self.dev = './data/kvret/indosmd/IndoSMD_dev.json'
        # self.test = './data/kvret/indosmd/IndoSMD_test.json'
        # self.entity = './data/kvret/indosmd/kvret_indo_entities.json'

        # eng
        # self.fasttext_path = './data/fasttext/cc.en.300.vec'
        # self.train = './data/kvret/smd/kvret_train_public.json'
        # self.dev = './data/kvret/smd/kvret_dev_public.json'
        # self.test = './data/kvret/smd/kvret_test_public.json'
        # self.entity = './data/kvret/smd/kvret_entities.json'

        # x-lang
        # self.fasttext_path = './data/fasttext/wiki.multi.id.vec'
        # self.vocab_data_path = ''
        # self.vocab_entity_path = ''
        # self.train = './data/kvret/smd/kvret_train_public.json'
        # self.dev = './data/kvret/smd/kvret_dev_public.json'
        # self.test = './data/kvret/indosmd/IndoSMD_test.json'
        # # train
        # self.entity = './data/kvret/smd/kvret_entities.json'
        # # test
        # self.entity = './data/kvret/indosmd/kvret_indo_entities.json'

        # bi
        # self.fasttext_path = './data/fasttext/wiki.multi.id.vec'
        # self.train = './data/kvret/bi/IndoSMD_train_bi.json'
        # self.dev = './data/kvret/bi/IndoSMD_dev_bi.json'
        # self.test = './data/kvret/bi/IndoSMD_test_bi.json'
        # self.entity = './data/kvret/bi/kvret_indo_entities_bi.json'

        self.batch_size = 32
        self.degree_size = 5
        self.z_length = 8
        self.layer_num = 1
        self.dropout_rate = 0.5
        self.epoch_num = 100
        self.rl_epoch_num = 2
        self.cuda = False
        self.spv_proportion = 100
        self.alpha = 0.0
        self.max_ts = 40
        self.early_stop_count = 3
        self.new_vocab = True
        self.model_path = './models/kvret.pkl'
        self.result_path = './results/kvret.csv'
        self.teacher_force = 100
        self.beam_search = False
        self.beam_size = 10
        self.sampling = False
        self.use_positional_embedding = False
        self.unfrz_attn_epoch = 0
        self.skip_unsup = False
        self.truncated = False
        self.pretrain = False

    def __str__(self):
        s = ''
        for k,v in self.__dict__.items():
            s += '{} : {}\n'.format(k,v)
        return s

    def _init_logging_handler(self):
        current_time = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())

        stderr_handler = logging.StreamHandler()
        file_handler = logging.FileHandler('./log/log_{}.txt'.format(current_time))
        logging.basicConfig(handlers=[stderr_handler, file_handler])
        logger = logging.getLogger()
        logger.setLevel(logging.INFO)

global_config = _Config()

