###########################################
# Copyright (c) Javier Cabero Guerra 2018 #
# Licensed under MIT                      #
###########################################


import os
import random
from flask import Flask
from flask_restful import Resource, Api, reqparse

app = Flask(__name__)
api = Api(app)

port = int(os.getenv('PORT', 8000)) 

class APITest(Resource):
  def get(self):
  	#return {'test':  random.randint(1, 10)}
    parser = reqparse.RequestParser()
    parser.add_argument('foo', type=str)

    return parser.parse_args()

api.add_resource(APITest, '/') 

app.run(host='0.0.0.0', port=port)