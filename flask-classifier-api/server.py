###########################################
# Copyright (c) Javier Cabero Guerra 2018 #
# Licensed under MIT                      #
###########################################


import os
from flask import Flask
from flask_restful import Resource, Api, reqparse
import numpy as np
from mnist_classifier import MNISTClassifier

app = Flask(__name__)
api = Api(app)

@app.before_first_request
def setup():
  app.config.update({"MNIST_CLASSIFIER": MNISTClassifier()})


port = int(os.getenv('PORT', 8000)) 

class MNISTClassifierResource(Resource):

  def get(self):
    parser = reqparse.RequestParser()
    parser.add_argument('img_data', type=str)
    img_data_raw = parser.parse_args()['img_data']
    if img_data_raw is None:
      return -1

    img_data_flat = np.fromstring(img_data_raw, sep=',', dtype=np.float32, count=784)
    img_data_flat = img_data_flat.clip(min=0)
    img_data = img_data_flat.reshape(28, 28, 1)
    predicted_digit_label = app.config["MNIST_CLASSIFIER"].predict(img_data)
    return predicted_digit_label



api.add_resource(MNISTClassifierResource, '/') 

app.run(debug=False, host='0.0.0.0', port=port)