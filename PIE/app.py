from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask
from flask import request
from flask import jsonify

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
        # if header Content-Type is not application/json, json_object = None
        json_instance = request.get_json()
        try:
            return prediction.predict_json(json_instance)
        except EnvironmentError as ee:
            raise HttpException(ee.args[0], status_code=500)
    else:
        if 'json' in request.args:
            json_string = request.args['json']
            try:
                return prediction.predict_json_string(json_string)
            except EnvironmentError as ee:
                raise HttpException(ee.args[0], status_code=500)
        else:
            return 'URL query parameter "json" must be specified.'


@application.errorhandler(HttpException)
def handle_http_exception(exception):
    response = jsonify(exception.to_dict())
    response.status_code = exception.status_code
    return response


if __name__ == '__main__':
    application.run(debug=False)
