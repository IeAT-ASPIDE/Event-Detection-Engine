"""

Copyright 2022, West University of Timisoara, Timisoara, Romania
Developers:
 * Gabriel Iuhasz, iuhasz.gabriel@info.uvt.ro

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at:
    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

from flask import send_file
from flask import request
import os
import jinja2
import sys
import subprocess
import platform
import logging
from logging.handlers import RotatingFileHandler
from flask import send_from_directory
from flask_restful import Resource, Api
from flask_apispec.views import MethodResource
from flask_apispec.annotations import doc
from app import *


#directory locations
tmpDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'templates')
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')

#file location
log_file = os.path.join(os.path.dirname(os.path.abspath(__file__)), '../ede.log')


@doc(description='EDE Status descriptor', tags=['status'])
class EDEStatus(Resource, MethodResource):
    def get(self):
        return "Current status of EDE workload and dask details"


@doc(description='Global config for EDE', tags=['config'])
class Config(Resource, MethodResource):
    def get(self):
        return "Global Config"

    def put(self):
        return "Modify Global Config (unsafe)"


@doc(description='Get descriptor for specific config field', tags=['config'])
class ConfigDescriptors(Resource, MethodResource):
    def get(self, param):
        return f"Get descriptor for specific resource defined by {param}"


@doc(description='Connector config for specific parameter', tags=['connector'])
class ConnectorParamConfig(Resource, MethodResource):
    def get(self, param):
        return f"Connector {param} config"

    def put(self, param):
        return f"Modify Connect {param} config"


@doc(description='Connector config for EDE', tags=['connector'])
class ConnectorConfig(Resource, MethodResource):
    def get(self):
        return "Connector Config "

    def put(self):
        return "Modify Connector Config"


@doc(description='Augmentation config for EDE', tags=['preprocessing'])
class AugmentationConfig(Resource, MethodResource):
    def get(self):
        return "Augmentation Config"

    def put(self):
        return "Modify Augmentation Config"


@doc(description='Augmentation config for specific parameter', tags=['preprocessing'])
class AugmentationParamConfig(Resource, MethodResource):
    def get(self, param):
        return f"Augmentation {param} config"

    def put(self, param):
        return f"Modify Augmentation {param} config"


@doc(description='Analysis config for EDE', tags=['preprocessing'])
class AnalysisConfig(Resource, MethodResource):
    def get(self):
        return "Analysis config"

    def put(self):
        return "Modify Analysis Config"


@doc(description='Analysis config for specific param', tags=['preprocessing'])
class AnalysisParamConfig(Resource, MethodResource):
    def get(self, param):
        return f"Analysis {param} config"

    def put(self, param):
        return f"Modify Analysis {param} config"


@doc(description='Mode config EDE', tags=['mode'])
class ModeConfig(Resource, MethodResource):
    def get(self):
        return "Mode config"

    def put(self):
        return "Modify Mode Config"


@doc(description='Filter config for EDE', tags=['preprocessing'])
class FilterConfig(Resource, MethodResource):
    def get(self):
        return "Filter Config"

    def put(self):
        return "Modify Filter Config"


@doc(description='Training config for EDE', tags=['training'])
class TrainingConfig(Resource, MethodResource):
    def get(self):
        return 'Training config'

    def put(self):
        return "Modify training config"


@doc(description='Training config for param', tags=['training'])
class TrainingParamConfig(Resource, MethodResource):
    def get(self, param):
        return f"Training {param}"

    def put(self, param):
        return f"Modify Training {param}"


@doc(description='Prediction config for EDE', tags=['inference'])
class PredictionConfig(Resource, MethodResource):
    def get(self):
        return "Prediction Config"

    def put(self):
        return "Modify prediction Config"


@doc(description='Point config for EDE', tags=['inference'])
class PointConfig(Resource, MethodResource):
    def get(self):
        return "Point config"

    def put(self):
        return "Modify point config"


@doc(description='Misc config for EDE', tags=['misc'])
class MiscConfig(Resource, MethodResource):
    def get(self):
        return "Misc config"

    def put(self):
        return "Modify misc config"


# ############## Control #################
@doc(description='Fetch EDE Logs', tags=['logs'])
class EDELogs(Resource, MethodResource):
    def get(self):
        return send_file(log_file, mimetype='text/plain')


@doc(description='Local Data sources', tags=['data'])
class LocalDataSource(Resource, MethodResource):
    def get(self):
        return f"List available local data"

    def post(self):
        return f"Push training data in csv format"

    def delete(self):
        return f"Delete Local data source"


@doc(description='Execute Training', tags=['training'])
class ExecuteTraining(Resource, MethodResource):
    def get(self):
        return "Status of current training session"

    def post(self):
        return "Start training of specific model"


@doc(description='Fetch trained models ids', tags=['training'])
class TrainingModelList(Resource, MethodResource):
    def get(self):
        return "List all trained model ids"


@doc(description='Fetch Training details of specific model', tags=['training'])
class TrainingDetails(Resource, MethodResource):
    def get(self, model_id):
        return f"Fetch training details of model indentified by {model_id}"


@doc(description='Model Handling', tags=['training'])
class TrainingModel(Resource, MethodResource):
    def get(self, model_id):
        return f"Fetch model with id {model_id}"

    def post(self, model_id):
        return f"Upload model with id {model_id}"

    def delete(self, model_id):
        return f"Delete model with id {model_id}"


@doc(description='Execute prediction', tags=['inference'])
class ExecuteInference(Resource, MethodResource):
    def get(self):
        return "List of currently active predictive models used for inference"

    def post(self):
        return "Execute currently configured model prediction"


@doc(description='Get all predictions stored', tags=['inference'])
class GetInference(Resource, MethodResource):
    def get(self):
        return f"Fetch all inference"


@doc(description='Get predictions of specific model', tags=['inference'])
class GetInferenceModel(Resource, MethodResource):
    def get(self, model_id):
        return f"Return a list of inference ids made by model identified by {model_id}"

    def post(self, model_id):
        return f"Push data to model identified by {model_id} and get prediction"


@doc(description='Get specific predictions of specific model', tags=['inference'])
class GetInferenceModelReference(Resource, MethodResource):
    def get(self, model_id, inference_id):
        return f"Fetch predictions of model identified by {model_id} identified by uuid {inference_id}"


@doc(description='Get details about a specific predictions', tags=['inference'])
class GetInferenceDetails(Resource, MethodResource):
    def get(self, model_id, inference_id):
        return f"Fetch inference details of inference {inference_id} made by {model_id}"


@doc(description='List all scalers and details', tags=['preprocessing'])
class NormalizersModels(Resource, MethodResource):
    def get(self):
        return "List all scalers description"


@doc(description='Scaler handler', tags=['preprocessing'])
class NormalizersModelsHandler(Resource, MethodResource):
    def get(self, scaler_id):
        return f"Return specific scaler by {scaler_id}"

    def put(self, scaler_id):
        return f"Push scaler with id {scaler_id}"

    def delete(self, scaler_id):
        return f"Delete scaler with id {scaler_id}"


# Rest API routing
api.add_resource(Config, '/v1/config')
api.add_resource(EDELogs, '/v1/logs')
api.add_resource(LocalDataSource, '/v1/data/local')
api.add_resource(ConfigDescriptors, '/v1/config/descriptor/<string:param>')
api.add_resource(ConnectorConfig, '/v1/config/connector')
api.add_resource(ConnectorParamConfig, '/v1/config/connector/<string:param>')
api.add_resource(AugmentationConfig, '/v1/config/augmentation')
api.add_resource(AugmentationParamConfig, '/v1/config/augmentation/<string:param>')
api.add_resource(AnalysisConfig, '/v1/config/analysis')
api.add_resource(AnalysisParamConfig, '/v1/config/analysis/<string:param>')
api.add_resource(ModeConfig, '/v1/config/mode')
api.add_resource(FilterConfig, '/v1/config/filter')
api.add_resource(NormalizersModels, '/v1/preprocessing/scalers')
api.add_resource(NormalizersModelsHandler, '/v1/preprocessing/scalers/<string:scaler_id>')
api.add_resource(TrainingConfig, '/v1/config/training')
api.add_resource(TrainingParamConfig, '/v1/config/training/<string:param>')
api.add_resource(ExecuteTraining, '/v1/training')
api.add_resource(TrainingModelList, '/v1/training/models')
api.add_resource(TrainingDetails, '/v1/training/models/<string:model_id>')
api.add_resource(TrainingModel, '/v1/training/models/<string:model_id>/artifact')
api.add_resource(PredictionConfig, '/v1/config/detect')
api.add_resource(ExecuteInference, '/v1/detect')
api.add_resource(GetInference, '/v1/detect/all/events')
api.add_resource(GetInferenceModel, '/v1/detect/models/<string:model_id>')
api.add_resource(GetInferenceModelReference, '/v1/detect/models/<string:model_id>/events/<string:inference_id>')
api.add_resource(GetInferenceDetails, '/v1/detect/models/<string:model_id>/events/<string:inference_id>/detailed')
api.add_resource(PointConfig, '/v1/config/point')
api.add_resource(MiscConfig, '/v1/config/misc')


# Rest API docs, Swagger
docs.register(Config)
docs.register(EDELogs)
docs.register(LocalDataSource)
docs.register(ConfigDescriptors)
docs.register(ConnectorConfig)
docs.register(ConnectorParamConfig)
docs.register(AugmentationConfig)
docs.register(AugmentationParamConfig)
docs.register(AnalysisConfig)
docs.register(AnalysisParamConfig)
docs.register(ModeConfig)
docs.register(FilterConfig)
docs.register(NormalizersModels)
docs.register(NormalizersModelsHandler)
docs.register(TrainingConfig)
docs.register(TrainingParamConfig)
docs.register(ExecuteTraining)
docs.register(TrainingModelList)
docs.register(TrainingDetails)
docs.register(TrainingModel)
docs.register(PredictionConfig)
docs.register(ExecuteInference)
docs.register(GetInference)
docs.register(GetInferenceModel)
docs.register(GetInferenceModelReference)
docs.register(GetInferenceDetails)
docs.register(PointConfig)
docs.register(MiscConfig)