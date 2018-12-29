from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import tensorflow as tf
import numpy as np
from grpc.beta import implementations
from grpc.framework.interfaces.face.face import AbortionError
from tensorflow_serving.apis import prediction_service_pb2, predict_pb2

from PIE.config import Config
from PIE.data import DataProcessor


class Predictor(object):
    def __init__(self, config, host='localhost', port=9000):
        self.config = config
        self.dataProcessor = DataProcessor(self.config)
        self.stub = prediction_service_pb2.beta_create_PredictionService_stub(
            implementations.insecure_channel(host, port))

    def predict_json(self, json_instance):
        # TODO support depth > 1, no inner structure, array, etc is supported yet
        if isinstance(json_instance, dict):  # is json object?
            json_instance = [json_instance]

        doc_field_value_list = []
        doc_field_name_list = []
        for list_item in json_instance:
            doc_field_value_list.append([str(list_item[x]).strip() for x in list_item])
            doc_field_name_list.extend([x.strip() for x in list_item])

        return self._predict(json.dumps(json_instance), doc_field_value_list, doc_field_name_list)

    def predict_json_string(self, json_string):
        json_instance = json.loads(json_string, encoding='UTF-8')
        return self.predict_json(json_instance)

    def predict_sentence(self, sentence):
        return self._predict(sentence, [[sentence]])

    def _predict(self, json_string, documents, headers=None):
        """

        :param header:
        :param document: 2-D list is expected
        :return:
        """
        documents_feature = [self.dataProcessor.convert_for_prediction(field)
                             for document in documents for field in document]
        input = []
        input_ids = []
        input_mask = []
        segment_ids = []
        new_headers = []

        for i, feature in enumerate(documents_feature):
            if headers:
                new_headers.extend([headers[i]] * len(feature['input_ids']))
            input.extend(feature['input'])
            input_ids.extend(feature['input_ids'])
            input_mask.extend(feature['input_mask'])
            segment_ids.extend(feature['segment_ids'])

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'pie'
        request.inputs['input_ids'].CopyFrom(tf.make_tensor_proto(input_ids, dtype=tf.int64))
        request.inputs['input_mask'].CopyFrom(tf.make_tensor_proto(input_mask, dtype=tf.int64))
        request.inputs['segment_ids'].CopyFrom(tf.make_tensor_proto(segment_ids, dtype=tf.int64))

        try:
            response = self.stub.Predict(request, 10.0)  # 10 seconds timeout
        except AbortionError as ae:
            if ae.code.name == 'INVALID_ARGUMENT':
                raise EnvironmentError('Invalid argument: %s' % json_string)
            else:
                raise EnvironmentError('Backend service is unavailable for now. Try later.')

        def softmax(x):
            """Compute softmax values for each sets of scores in x."""
            e_x = np.exp(x - np.max(x))
            return e_x / e_x.sum()

        labels_pred = tf.contrib.util.make_ndarray(response.outputs['viterbi_sequence'])
        logits = tf.contrib.util.make_ndarray(response.outputs['logits'])
        # tp = tf.contrib.util.make_ndarray(response.outputs['tp'])

        label_map = {}
        for (i, label) in enumerate(self.config.label_list, 1):
            label_map[i] = label

        if headers is None:
            labels = [label_map[x] for labels in labels_pred for x in labels if x > 0][1:-1]
            labels = [x for x in labels if x != 'X']
            return labels
        else:
            label_texts = [[label_map[x] for x in labels if x > 0][1:-1] for labels in labels_pred]
            header_dict = {}

            for i, label_text in enumerate(label_texts):
                field_tags = []
                pred = None
                for j, l in enumerate(label_text):
                    if l not in ['O', 'X']:
                        split_label = l.split('-')
                        pred = {split_label[1]: {"token": "", "confidence": 0}}
                        pred[split_label[1]]["token"] = input[i][j + 1]
                        k = 1
                        while j + k < len(label_text) and label_text[j + k] == 'X':
                            pred[split_label[1]]["token"] += input[i][j + 1 + k][2:]
                            k += 1
                        pred[split_label[1]]["confidence"] = round(float(np.max(softmax(logits[i][j + 1]))), 4)
                        field_tags.append(pred)

                if len(field_tags) > 0:
                    if new_headers[i] in header_dict:
                        field_tags.extend(header_dict[new_headers[i]])

                    header_dict[new_headers[i]] = field_tags

            return json.dumps(header_dict)


if __name__ == '__main__':
    prediction = Predictor(Config(),
                           '192.168.99.100',  # IP address is retrieved by "docker-machine ip" on windows
                           9000)

    while True:
        sentence = input("input>")

        if sentence == 'exit':
            break
        try:
            result = prediction.predict_json_string(sentence)
        except ValueError:
            result = prediction.predict_sentence(sentence)
        except EnvironmentError as ee:
            print(ee)
            continue

        print(result)
