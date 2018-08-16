from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from flask import Flask
from flask import request

from PIE.config import Config
from PIE.predict import Prediction

application = Flask(__name__)

prediction = Prediction(Config(), 'localhost', 9000)


@application.route("/c3api_ai/v1/privacy", methods=['GET', 'POST'])
def privacy_predict():
    if request.method == 'POST':
        # if header Content-Type is not application/json, json_object = None
        json_object = request.get_json()
        return prediction.predict_json_object(json_object)
    else:
        if 'json' in request.args:
            json_string = request.args['json']
            return prediction.predict_json_string(json_string)
        else:
            return 'URL query parameter "json" must be specified.'

if __name__ == '__main__':
    application.run(debug=False)