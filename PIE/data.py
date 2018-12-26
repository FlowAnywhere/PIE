from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import multiprocessing
import os

import collections
import re
import tensorflow as tf

from PIE.config import Config
from BERT import tokenization


class Data(object):
    def __init__(self, config):
        self.config = config

    def __transfer_raw(self, file_suffix='train.txt', tfrecord=None):
        train_files = []
        for root, dirs, files in os.walk(self.config.data_dir_raw):
            for file in files:
                if file.endswith(file_suffix):
                    train_files.append(os.path.join(root, file))

        for train_file in train_files:
            with open(train_file) as f:
                lines = []
                words = []
                labels = []
                for line in f:
                    contends = line.strip()

                    if len(contends) == 0 or contends.startswith("-DOCSTART-"):
                        if len(words) != 0:
                            l = ' '.join([label for label in labels if len(label) > 0])
                            w = ' '.join([word for word in words if len(word) > 0])
                            lines.append([tokenization.convert_to_unicode(l), tokenization.convert_to_unicode(w)])
                            words = []
                            labels = []
                    else:
                        ls = contends.split(' ')
                        word, label = ls[0], ls[-1]

                        words.append(word)
                        labels.append(label)

                if tfrecord is not None:
                    tfrecord_filename = re.sub(r'[\/\\\.]', '_', train_file)
                    tfrecord.write(tfrecord_filename, lines)

        # return lines_list

    def generate_train_tfrecords(self):
        self.__transfer_raw('train.txt', TFRecordManager(self.config, True))

    def generate_valid_tfrecords(self):
        self.__transfer_raw('valid.txt', TFRecordManager(self.config, False))


class TFRecordManager(object):
    def __init__(self, config, train=True):
        self.config = config
        self.train = train

    def _convert_single_example(self, example):
        label_map = {}
        for (i, label) in enumerate(self.config.label_list, 1):
            label_map[label] = i

        textlist = example[1].split(' ')
        labellist = example[0].split(' ')
        tokens = []
        labels = []
        for i, word in enumerate(textlist):
            token = self.config.tokenizer.tokenize(word)
            tokens.extend(token)
            label_1 = labellist[i]
            for m in range(len(token)):
                if m == 0:
                    labels.append(label_1)
                else:
                    labels.append("X")
        # tokens = tokenizer.tokenize(example.text)
        if len(tokens) >= self.config.max_seq_length - 1:
            tokens = tokens[0:(self.config.max_seq_length - 2)]
            labels = labels[0:(self.config.max_seq_length - 2)]
        ntokens = []
        segment_ids = []
        label_ids = []
        ntokens.append("[CLS]")
        segment_ids.append(0)
        # append("O") or append("[CLS]") not sure!
        label_ids.append(label_map["[CLS]"])
        for i, token in enumerate(tokens):
            ntokens.append(token)
            segment_ids.append(0)
            label_ids.append(label_map[labels[i]])
        ntokens.append("[SEP]")
        segment_ids.append(0)
        # append("O") or append("[SEP]") not sure!
        label_ids.append(label_map["[SEP]"])
        input_ids = self.config.tokenizer.convert_tokens_to_ids(ntokens)
        input_mask = [1] * len(input_ids)
        # label_mask = [1] * len(input_ids)
        while len(input_ids) < self.config.max_seq_length:
            input_ids.append(0)
            input_mask.append(0)
            segment_ids.append(0)
            # we don't concerned about it!
            label_ids.append(0)
            ntokens.append("**NULL**")
            # label_mask.append(0)
        # print(len(input_ids))
        assert len(input_ids) == self.config.max_seq_length
        assert len(input_mask) == self.config.max_seq_length
        assert len(segment_ids) == self.config.max_seq_length
        assert len(label_ids) == self.config.max_seq_length
        # assert len(label_mask) == max_seq_length

        return input_ids, input_mask, segment_ids, label_ids

    def write(self, filename, lines):
        file = (
                   self.config.dataset_dir_train if self.train else self.config.dataset_dir_valid) + filename + ".tfrecords"
        os.makedirs(os.path.dirname(file), exist_ok=True)
        writer = tf.python_io.TFRecordWriter(file)

        for (ex_index, example) in enumerate(lines):
            input_ids, input_mask, segment_ids, label_ids = self._convert_single_example(example)

            def create_int_feature(values):
                f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
                return f

            features = collections.OrderedDict()
            features["input_ids"] = create_int_feature(input_ids)
            features["input_mask"] = create_int_feature(input_mask)
            features["segment_ids"] = create_int_feature(segment_ids)
            features["label_ids"] = create_int_feature(label_ids)
            # features["label_mask"] = create_int_feature(feature.label_mask)
            tf_example = tf.train.Example(features=tf.train.Features(feature=features))
            writer.write(tf_example.SerializeToString())

        writer.close()


class DataSet(object):

    def __init__(self, config):
        self.config = config

        self.name_to_features = {
            "input_ids": tf.FixedLenFeature([self.config.max_seq_length], tf.int64),
            "input_mask": tf.FixedLenFeature([self.config.max_seq_length], tf.int64),
            "segment_ids": tf.FixedLenFeature([self.config.max_seq_length], tf.int64),
            "label_ids": tf.FixedLenFeature([self.config.max_seq_length], tf.int64),
            # "label_ids":tf.VarLenFeature(tf.int64),
            # "label_mask": tf.FixedLenFeature([seq_length], tf.int64),
        }

    def _decode_record(self, record, name_to_features):
        example = tf.parse_single_example(record, name_to_features)
        for name in list(example.keys()):
            t = example[name]
            if t.dtype == tf.int64:
                t = tf.to_int32(t)
            example[name] = t
        return example

    def train(self):
        train_tfrecord_files = []
        for root, dirs, files in os.walk(self.config.dataset_dir_train):
            for file in files:
                if file.endswith(".tfrecords"):
                    train_tfrecord_files.append(os.path.join(root, file))

        # TODO shuffle
        return tf.data.TFRecordDataset(train_tfrecord_files, buffer_size=64000,
                                       num_parallel_reads=multiprocessing.cpu_count()).apply(
            tf.contrib.data.map_and_batch(
                lambda record: self._decode_record(record, self.name_to_features),
                batch_size=self.config.batch_size,
                drop_remainder=True
            ))

    def valid(self):
        valid_tfrecord_files = []
        for root, dirs, files in os.walk(self.config.dataset_dir_valid):
            for file in files:
                if file.endswith(".tfrecords"):
                    valid_tfrecord_files.append(os.path.join(root, file))

        return tf.data.TFRecordDataset(valid_tfrecord_files, buffer_size=64000,
                                       num_parallel_reads=multiprocessing.cpu_count()).apply(
            tf.contrib.data.map_and_batch(
                lambda record: self._decode_record(record, self.name_to_features),
                batch_size=self.config.batch_size,
                drop_remainder=False
            ))


if __name__ == '__main__':
    data = Data(Config())
    data.generate_train_tfrecords()
    data.generate_valid_tfrecords()
