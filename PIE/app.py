from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

import numpy as np
from flask import Flask
from flask import jsonify
from flask import request, Response

from PIE.config import Config
from PIE.predictor import Predictor

application = Flask(__name__)

prediction = Predictor(Config(), 'localhost', 9000)


class HttpException(Exception):
    status_code = 400

    def __init__(self, message, status_code=None, payload=None):
        Exception.__init__(self)
        self.message = message
        if status_code is not None:
            self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        rv = dict(self.payload or ())
        rv['message'] = self.message
        rv['code'] = self.status_code

        return rv


@application.route("/c3api_ai/v1/privacy", methods=['GET', 'POST'])
def privacy_predict():
    if request.method == 'POST':
        content_type = request.headers.get('Content-Type')

        if content_type.lower() == 'application/json':
            # if header Content-Type is not application/json, json_instance = None
            json_instance = request.get_json()
            try:
                resp = Response(prediction.predict_json(json_instance))
                resp.headers['Content-Type'] = 'application/json'
                return resp
            except EnvironmentError as ee:
                raise HttpException(ee.args[0], status_code=500)
        elif content_type.lower() == 'audio/wave':
            audio = request.get_data()
            try:
                text = stt(audio)
                print('decoded text from audio: %s' % text, file=sys.stdout)
                resp = Response(prediction.predict_json_string('{"audio": "%s"}' % text))
                resp.headers['Content-Type'] = 'application/json'
                return resp
            except EnvironmentError as ee:
                raise HttpException(ee.args[0], status_code=500)
    else:
        if 'json' in request.args:
            json_string = request.args['json']
            try:
                resp = Response(prediction.predict_json_string(json_string))
                resp.headers['Content-Type'] = 'application/json'
                return resp
            except EnvironmentError as ee:
                raise HttpException(ee.args[0], status_code=500)
        else:
            return 'URL query parameter "json" must be specified.'


def stt(audio):
    # These constants control the beam search decoder

    # Beam width used in the CTC decoder when building candidate transcriptions
    BEAM_WIDTH = 500

    # The alpha hyperparameter of the CTC decoder. Language Model weight
    LM_WEIGHT = 1.50

    # Valid word insertion weight. This is used to lessen the word insertion penalty
    # when the inserted word is part of the vocabulary
    VALID_WORD_COUNT_WEIGHT = 2.10

    # These constants are tied to the shape of the graph used (changing them changes
    # the geometry of the first layer), so make sure you use the same constants that
    # were used during training

    # Number of MFCC features to use
    N_FEATURES = 26

    # Size of the context window used for producing timesteps in the input vector
    N_CONTEXT = 9

    model_path = '../models/output_graph.pbmm'
    alphabet_path = '../models/alphabet.txt'
    lm_path = '../models/lm.binary'
    trie_path = '../models/trie'

    from deepspeech import Model
    ds = Model(model_path, N_FEATURES, N_CONTEXT, alphabet_path, BEAM_WIDTH)
    # ds.enableDecoderWithLM(alphabet_path, lm_path, trie_path, LM_WEIGHT, VALID_WORD_COUNT_WEIGHT)
    return ds.stt(np.frombuffer(audio, np.int16), 16000)


@application.errorhandler(HttpException)
def handle_http_exception(exception):
    response = jsonify(exception.to_dict())
    response.status_code = exception.status_code
    return response


if __name__ == '__main__':
    application.run(debug=False)
