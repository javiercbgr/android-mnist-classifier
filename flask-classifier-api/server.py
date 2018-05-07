###########################################
# Copyright (c) Javier Cabero Guerra 2018 #
# Licensed under MIT                      #
###########################################


import os
from flask import Flask
from flask_restful import Resource, Api, reqparse
import numpy as np

app = Flask(__name__)
api = Api(app)

port = int(os.getenv('PORT', 8000)) 

class MNISTClassifierResource(Resource):
  def get(self):
    parser = reqparse.RequestParser()
    parser.add_argument('img_data', type=str)
    img_data_raw = parser.parse_args()['img_data']
    if img_data_raw is None:
      return -1
    img_data = list(img_data_raw)
    return img_data

api.add_resource(MNISTClassifierResource, '/') 

app.run(host='0.0.0.0', port=port)