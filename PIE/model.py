from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
import tensorflow as tf
from tensorflow.python.training import session_run_hook

from PIE.config import Config
from PIE.data import Data, DataSet
from bert import modeling, optimization


class Model(object):
    def __init__(self, config):
        self.config = config
        self.logger = config.logger
        self.data = Data(self.config)
        self.dataset = DataSet(self.config)

        self.training_hook = None
        self.eval_hook = None

    def _train_input_fn(self):
        return self.dataset.train()

    def _valid_input_fn(self):
        return self.dataset.valid()

    def _create_serving_input_receiver(self):
        inputs = {'input_ids': tf.placeholder(dtype=tf.int64, shape=[None, None], name="input_ids"),
                  'input_mask': tf.placeholder(dtype=tf.int64, shape=[None, None], name="input_mask"),
                  'segment_ids': tf.placeholder(dtype=tf.int64, shape=[None, None], name="segment_ids")}
        return tf.estimator.export.ServingInputReceiver(inputs, inputs)

    def _model_fn(self, features, labels, mode, params):
        self._create_model(features, mode)

        tvars = tf.trainable_variables()
        (assignment_map, initialized_variable_names) = modeling.get_assignment_map_from_checkpoint(tvars,
                                                                                                   self.config.bert_checkpoint_file)
        tf.train.init_from_checkpoint(self.config.bert_checkpoint_file, assignment_map)

        if mode == tf.estimator.ModeKeys.TRAIN:
            if self.training_hook is None:
                self.training_hook = _TrainingHook(model=self)

            return tf.estimator.EstimatorSpec(mode, loss=self.loss, train_op=self.train_op,
                                              training_chief_hooks=[_CPSaverHook(
                                                  checkpoint_dir=self.config.output_dir_root,
                                                  save_steps=sys.maxsize // 2), self.training_hook])
        if mode == tf.estimator.ModeKeys.EVAL:
            if self.eval_hook is None:
                self.eval_hook = _EvaluationHook(model=self)

            return tf.estimator.EstimatorSpec(mode, loss=self.loss, eval_metric_ops={
                'accuracy': self.accuracy,
                'f1': self.f1
            }, evaluation_hooks=[self.eval_hook])

        if mode == tf.estimator.ModeKeys.PREDICT:
            predictions = {
                'viterbi_sequence': self.viterbi_sequence,
                'logits': self.logits
            }
            export_outputs = {
                'prediction': tf.estimator.export.PredictOutput(predictions)
            }
            return tf.estimator.EstimatorSpec(
                mode,
                predictions=predictions,
                export_outputs=export_outputs)

    def _create_model(self, features, mode):
        self._add_variables(features, mode)
        self._add_bert(features, mode)

        self._add_logits_op()

        if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
            self._add_loss_op()

        if mode in [tf.estimator.ModeKeys.TRAIN]:
            self._add_train_op()

        if mode in [tf.estimator.ModeKeys.EVAL, tf.estimator.ModeKeys.PREDICT]:
            self._add_transition_parameter()

        self._add_prediction_op()

        if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
            self._add_accuracy_op()

    def _add_bert(self, features, mode):
        model = modeling.BertModel(
            config=self.config.bert_config,
            is_training=True if mode == tf.estimator.ModeKeys.TRAIN else False,
            input_ids=features['input_ids'],
            input_mask=features['input_mask'],
            token_type_ids=features['segment_ids'],
            use_one_hot_embeddings=False
        )

        self.word_embeddings = model.get_sequence_output()
        # TODO dropout

    def _add_variables(self, features, mode):
        with tf.variable_scope("variable"):
            used = tf.sign(tf.abs(features['input_ids']))
            self.sequence_lengths = tf.reduce_sum(used, reduction_indices=1)

            # shape = (batch size, max length of sentence in batch)
            if mode in [tf.estimator.ModeKeys.TRAIN, tf.estimator.ModeKeys.EVAL]:
                self.labels = tf.cast(features['label_ids'], dtype=tf.int32, name='labels')

            if mode in [tf.estimator.ModeKeys.TRAIN]:
                self.dropout = tf.constant(self.config.dropout, dtype=tf.float32, name='dropout')
            else:
                self.dropout = tf.constant(1.0, dtype=tf.float32, name='dropout')

    def _add_logits_op(self):
        """
        For each word in each sentence of the batch, it corresponds to a vector
        of scores, of dimension equal to the number of tags.
        """
        with tf.variable_scope("bi-lstm"):
            cell_fw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            cell_bw = tf.contrib.rnn.LSTMCell(self.config.hidden_size_lstm)
            (output_fw, output_bw), _ = tf.nn.bidirectional_dynamic_rnn(cell_fw, cell_bw, self.word_embeddings,
                                                                        sequence_length=self.sequence_lengths,
                                                                        dtype=tf.float32)
            output = tf.concat([output_fw, output_bw], axis=-1)
            output = tf.nn.dropout(output, self.dropout)

        with tf.variable_scope("proj"):
            W = tf.get_variable("W", dtype=tf.float32,
                                shape=[2 * self.config.hidden_size_lstm, len(self.config.label_list)])

            b = tf.get_variable("b", shape=[len(self.config.label_list)],
                                dtype=tf.float32, initializer=tf.zeros_initializer())

            nsteps = tf.shape(output)[1]
            output = tf.reshape(output, [-1, 2 * self.config.hidden_size_lstm])
            pred = tf.matmul(output, W) + b
            self.logits = tf.reshape(pred, [-1, nsteps, len(self.config.label_list)])

    def _add_loss_op(self):
        with tf.variable_scope("loss_op"):
            log_likelihood, self.trans_params = tf.contrib.crf.crf_log_likelihood(
                self.logits, self.labels, self.sequence_lengths)
            self.loss = tf.reduce_mean(-log_likelihood, name='loss')

    def _add_transition_parameter(self):
        with tf.variable_scope("loss_op", reuse=tf.AUTO_REUSE):
            self.trans_params = tf.get_variable('transitions',
                                                [len(self.config.label_list), len(self.config.label_list)])

    def _add_train_op(self):
        with tf.variable_scope("train_op"):
            self.train_op = optimization.create_optimizer(self.loss, self.config.lr, 100000, 0, False)

            # optimizer = tf.train.AdamOptimizer(self.lr)
            # if self.config.clip > 0:  # gradient clipping if clip is positive
            #     grads, vs = zip(*optimizer.compute_gradients(self.loss))
            #     grads, gnorm = tf.clip_by_global_norm(grads, self.config.clip)
            #     self.train_op = optimizer.apply_gradients(zip(grads, vs),
            #                                               global_step=tf.train.get_or_create_global_step())
            # else:
            #     self.train_op = optimizer.minimize(loss=self.loss, global_step=tf.train.get_or_create_global_step())

    def _add_prediction_op(self):
        with tf.variable_scope("prediction_op"):
            self.viterbi_sequence, self.viterbi_score = tf.contrib.crf.crf_decode(self.logits, self.trans_params,
                                                                                  self.sequence_lengths)

    def _add_accuracy_op(self):
        with tf.variable_scope('accuracy_op'):
            self.accuracy = tf.metrics.accuracy(self.labels, self.viterbi_sequence)
            self.precision = tf.metrics.precision_at_top_k(tf.cast(self.labels, tf.int64), self.viterbi_sequence,
                                                           len(self.config.label_list))
            self.recall = tf.metrics.recall_at_top_k(tf.cast(self.labels, tf.int64), self.viterbi_sequence,
                                                     len(self.config.label_list))

            self.f1 = (2.0 * self.precision[0] * self.recall[0] / (self.precision[0] + self.recall[0]),
                       2.0 * self.precision[1] * self.recall[1] / (self.precision[1] + self.recall[1]))

            tf.summary.scalar('accuracy', self.accuracy[1])
            tf.summary.scalar('f1', self.f1[1])

    def run(self, _):
        self.run_config = tf.estimator.RunConfig(keep_checkpoint_max=3)

        self.predictor = tf.estimator.Estimator(
            model_fn=self._model_fn,
            model_dir=self.config.output_dir_root,
            config=self.run_config
        )

        tf.gfile.MakeDirs(self.config.output_dir_savedmodel)

        def _f1_bigger(best_eval_result, current_eval_result):
            return best_eval_result['f1'] < current_eval_result['f1']

        try:
            tf.estimator.train_and_evaluate(estimator=self.predictor,
                                            train_spec=tf.estimator.TrainSpec(input_fn=self._train_input_fn),
                                            eval_spec=tf.estimator.EvalSpec(input_fn=self._valid_input_fn, steps=None,
                                                                            start_delay_secs=0,
                                                                            exporters=tf.estimator.BestExporter(
                                                                                name=self.config.exporter_name,
                                                                                serving_input_receiver_fn=self._create_serving_input_receiver,
                                                                                exports_to_keep=2,
                                                                                compare_fn=_f1_bigger)))
        except RuntimeError:
            # workaround to exit training loop when no evaluation performance improvement after long epochs.
            pass
        # estimator.train does not work in distributed training
        # predictor.train(input_fn=self._train_input_fn)
        # predictor.evaluate(input_fn=self._valid_input_fn)


class _TrainingHook(session_run_hook.SessionRunHook):
    def __init__(self, model):
        self.model = model
        self.f1 = 0
        self.accuracy = 0
        self.epoch = 0

    def begin(self):
        self.f1 = 0
        self.accuracy = 0

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return session_run_hook.SessionRunArgs([self.model.accuracy[1], self.model.f1[1]])

    def after_run(self, run_context, run_values):
        self.accuracy, self.f1 = run_values.results

    def end(self, session):
        self.epoch += 1

        self.model.logger.info('*******************************Training Result******************************')
        self.model.logger.info(
            'F1: {} \tAccuracy: {} \tEpoch: {}'.format(100 * self.f1, 100 * self.accuracy, self.epoch))
        self.model.logger.info('*******************************Training Result******************************')


class _EvaluationHook(session_run_hook.SessionRunHook):
    def __init__(self, model):
        self.wait = 0
        self.best = -np.Inf

        self.model = model

        self.f1 = 0
        self.accuracy = 0

        self.epoch = 0

        self.word_ids = []
        self.labels = []
        self.predictions = []

    def begin(self):
        self.f1 = 0
        self.accuracy = 0
        self.word_ids = []
        self.labels = []
        self.predictions = []

    def before_run(self, run_context):  # pylint: disable=unused-argument
        return session_run_hook.SessionRunArgs(
            [self.model.accuracy[1], self.model.f1[1], self.model.word_ids, self.model.labels,
             self.model.viterbi_sequence])

    def after_run(self, run_context, run_values):
        self.accuracy, self.f1, w, l, p = run_values.results
        self.word_ids.extend(w)
        self.labels.extend(l)
        self.predictions.extend(p)

    def end(self, session):
        self.epoch += 1

        self.model.logger.info('======================Evaluation Result===========================')
        self.model.logger.info(
            'F1: {} \tAccuracy: {} \tEpoch: {}'.format(100 * self.f1, 100 * self.accuracy, self.epoch))

        if self.f1 > self.best:
            self.best = self.f1
            self.wait = 0
            self.model.logger.info('New Best F1 Score!')
            self.model.logger.info('======================Evaluation Result===========================')

            # output prediction
            corrected_pred, total_pred, corrected_pred_wo_o, total_pred_wo_o = 0, 0, 0, 0
            with open(self.model.config.output_dir_root + 'eval_result_' + str(self.epoch % 3) + '.txt', mode='w',
                      encoding='UTF-8') as f:
                for word, label, pred in zip(self.word_ids, self.labels, self.predictions):
                    for w, l, p in zip(word, label, pred):
                        if l != 0 and p != 0:
                            if l == p:
                                corrected_pred += 1
                                if self.model.data.idx_tag_vocab[l] != 'O':
                                    corrected_pred_wo_o += 1
                            total_pred += 1
                            if self.model.data.idx_tag_vocab[l] != 'O':
                                total_pred_wo_o += 1

                            f.write(
                                '{:20}{:20}{:20}\n'.format(self.model.data.idx_word_vocab[w],
                                                           self.model.data.idx_tag_vocab[l],
                                                           self.model.data.idx_tag_vocab[p]))
                    f.write('\n')
                f.write('\n\nAccuracy: {}\tTotal: {}\tCorrect: {}\n'.format((100 * corrected_pred) / total_pred,
                                                                            total_pred, corrected_pred))
                f.write('Accuracy w/o O: {}\tTotal: {}\tCorrect: {}\n'.format(
                    (100 * corrected_pred_wo_o) / total_pred_wo_o, total_pred_wo_o, corrected_pred_wo_o))
        else:
            self.wait += 1
            self.model.logger.info('# epochs with no improvement: {}'.format(self.wait))
            self.model.logger.info('======================Evaluation Result===========================')
            if self.wait >= self.model.config.patience:
                raise RuntimeError('Can not make progress!')


class _CPSaverHook(tf.train.CheckpointSaverHook):
    def after_create_session(self, session, coord):
        # override parent class to disable checkpoint file writing
        pass

    def after_run(self, run_context, run_values):
        # override parent class to disable checkpoint file writing
        pass


if __name__ == '__main__':
    # tf.logging.set_verbosity(tf.logging.INFO)
    # tf.enable_eager_execution()

    model = Model(Config())

    tf.app.run(main=model.run)
