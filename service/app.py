from flask import Flask
from flask.ext.restplus import Api

app = Flask("aspide-ede")
api = Api(app, version='0.0.1', title='Aspide Event Detection Engine',
          description="RESTful API for the Aspide Event Detection Engine",
          )

adp = api.namespace('ede', description='ede operations')