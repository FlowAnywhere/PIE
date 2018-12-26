from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import logging
import os

import tensorflow as tf

from bert import tokenization, modeling


class Config(object):
    def __get_logger(log_filename):
        os.makedirs(os.path.dirname(log_filename), exist_ok=True)

        logger = logging.getLogger('logger')
        logger.setLevel(logging.DEBUG)
        logging.basicConfig(format='%(message)s', level=logging.DEBUG)
        handler = logging.FileHandler(log_filename)
        handler.setLevel(logging.DEBUG)
        handler.setFormatter(logging.Formatter('%(asctime)s:%(levelname)s: %(message)s'))
        logging.getLogger().addHandler(handler)

        return logger

    data_dir_root = '../data/'
    data_dir_raw = data_dir_root + 'raw/'

    dataset_dir_root = '../dataset/'
    dataset_dir_train = dataset_dir_root + 'train/'
    dataset_dir_valid = dataset_dir_root + 'valid/'

    lr = 2e-5
    dropout = 0.9
    batch_size = 128 if tf.test.is_gpu_available() else 64
    patience = 6  # early stop

    hidden_size_char = 100  # lstm on chars
    hidden_size_lstm = 100  # lstm on word embeddings

    output_dir_root = '../output/'
    log_filename = output_dir_root + 'logs/log.txt'
    logger = __get_logger(log_filename)

    exporter_name = 'BestExport'
    output_dir_savedmodel = output_dir_root + 'export/' + exporter_name

    bert_checkpoint_dir = data_dir_root + 'bert/'
    bert_checkpoint_file = bert_checkpoint_dir + 'bert_model.ckpt'
    bert_config_file = bert_checkpoint_dir + 'bert_config.json'
    bert_config = modeling.BertConfig.from_json_file(bert_config_file)

    vocab_file = bert_checkpoint_dir + 'vocab.txt'
    do_lower_case = True
    label_list = ["B-PERSON", "I-PERSON", "E-PERSON", "S-PERSON", "B-ADDRESS", "I-ADDRESS", "E-ADDRESS", "S-ADDRESS",
                  "B-ORG", "I-ORG", "E-ORG", "S-ORG", "O", "X", "[CLS]", "[SEP]"]
    max_seq_length = 128

    if max_seq_length > bert_config.max_position_embeddings:
        raise ValueError(
            "Cannot use sequence length %d because the BERT model was only trained up to sequence length %d" %
            (max_seq_length, bert_config.max_position_embeddings))

    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file, do_lower_case=do_lower_case)
