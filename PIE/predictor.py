from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json

import tensorflow as tf
from grpc.beta import implementations
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

        return self._predict(doc_field_value_list, doc_field_name_list)

    def predict_json_string(self, json_string):
        json_instance = json.loads(json_string, encoding='UTF-8')
        return self.predict_json(json_instance)

    def predict_sentence(self, sentence):
        return self._predict([[sentence]])

    def _predict(self, documents, headers=None):
        """

        :param header:
        :param document: 2-D list is expected
        :return:
        """
        documents_feature = [self.dataProcessor.convert_for_prediction(field) for document in documents for field in
                             document]
        input_ids = []
        input_mask = []
        segment_ids = []
        new_headers = []

        for i, feature in enumerate(documents_feature):
            new_headers.extend([headers[i]] * len(feature['input_ids']))
            input_ids.extend(feature['input_ids'])
            input_mask.extend(feature['input_mask'])
            segment_ids.extend(feature['segment_ids'])

        request = predict_pb2.PredictRequest()
        request.model_spec.name = 'pie'
        request.inputs['input_ids'].CopyFrom(tf.make_tensor_proto(input_ids, dtype=tf.int64))
        request.inputs['input_mask'].CopyFrom(tf.make_tensor_proto(input_mask, dtype=tf.int64))
        request.inputs['segment_ids'].CopyFrom(tf.make_tensor_proto(segment_ids, dtype=tf.int64))

        response = self.stub.Predict(request, 10.0)  # 10 seconds timeout
        # TODO raise error message in case of timeout

        labels_pred = tf.contrib.util.make_ndarray(response.outputs['viterbi_sequence'])
        logits = tf.contrib.util.make_ndarray(response.outputs['logits'])
        # tp = tf.contrib.util.make_ndarray(response.outputs['tp'])

        label_map = {}
        for (i, label) in enumerate(self.config.label_list, 1):
            label_map[i] = label

        if headers is None:
            return [label_map[x] for labels in labels_pred for x in labels if x > 0]
        else:
            label_texts = [[label_map[x] for x in labels if x > 0] for labels in labels_pred]
            header_dict = {}

            for i, label_text in enumerate(label_texts):
                field_tag = set()
                for l in label_text:
                    if l not in ['O', 'X', '[CLS]', '[SEP]']:
                        field_tag.add(l.split('-')[1])
                if len(field_tag) > 0:
                    if new_headers[i] in header_dict:
                        field_tag.update(header_dict[new_headers[i]])

                    header_dict[new_headers[i]] = list(field_tag)

            return json.dumps(header_dict)


if __name__ == '__main__':
    prediction = Predictor(Config(), '192.168.99.100', 9000) # IP address is retrieved by "docker-machine ip" on windows

    while True:
        sentence = input("input> ")

        if sentence == 'exit':
            break
        try:
            result = prediction.predict_json_string(sentence)
        except ValueError:
            result = prediction.predict_sentence(sentence)

        print(result)
