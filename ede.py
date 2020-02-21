"""
Copyright 2019, Institute e-Austria, Timisoara, Romania
    http://www.ieat.ro/
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

import sys, getopt
import os.path
from addict import Dict
from edeconfig import readConf
from edelogger import logger
from datetime import datetime
from edengine import aspideedengine
from util import getModelList, check_dask_settings
import time
from dask.distributed import Client, LocalCluster
from signal import signal, SIGINT
import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


def main(argv,
         cluster,
         client):
    dataDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
    modelsDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    queryDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'queries')

    settings = Dict()
    settings.esendpoint = None
    settings.prendpoint = None
    settings.Dask.SchedulerEndpoint = None  # "local"
    settings.Dask.SchedulerPort = 8787
    settings.Dask.EnforceCheck = False
    settings.prkafkaendpoint = None
    settings.prkafkaport = 9092
    settings.prkafkatopic = "edetopic"
    settings.augmentation = None  # augmentation including scaler and user defined methods
    settings.detectionscaler = None
    settings.MPort = 9090
    settings.dmonPort = 5001
    settings.index = "logstash-*"
    settings["from"] = None
    settings.to = None
    settings.query = None
    settings.nodes = None
    settings.qsize = None
    settings.qinterval = None
    settings.fillna = None
    settings.dropna = None
    settings.local = None
    settings.train = None
    settings.hpomethod = None
    settings.ParamDistribution = None
    settings.detecttype = None # TODO
    settings.traintype = None
    settings.validationtype = None # Todo
    settings.target = None
    settings.load = None
    settings.file = None
    settings.method = None
    settings.detectMethod = None
    settings.trainMethod = None
    settings.cv = None
    settings.trainscore = None
    settings.scorer = None
    settings.returnestimators = None
    settings.analysis = None
    settings.validate = None
    settings.export = None
    settings.trainexport = None
    settings.detect = None  # Bool default None
    settings.cfilter = None
    settings.rfilter = None
    settings.dfilter = None
    settings.sload = None
    settings.smemory = None
    settings.snetwork = None
    settings.heap = None
    settings.checkpoint = None
    settings.delay = None
    settings.interval = None
    settings.resetindex = None
    settings.training = None
    settings.validation = None
    settings.validratio = 0.2
    settings.compare = False
    settings.anomalyOnly = False
    settings.categorical = None
    settings.point = False


    # Only for testing
    settings['validate'] = False
    dask_backend = False

    try:
        opts, args = getopt.getopt(argv, "he:tf:m:vx:d:lq:", ["endpoint=", "file=", "method=", "export=", "detect=", "query="])  # todo:expand comand line options
    except getopt.GetoptError:
        logger.warning('[%s] : [WARN] Invalid argument received exiting', datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
        print("ede.py -f <filelocation>, -t -m <method> -v -x <modelname>")
        sys.exit(0)
    for opt, arg in opts:
        if opt == '-h':
            print("#" * 100)
            print('Event Detection Engine')
            print('Utilisation:')
            print('-f -> condifuration file location')
            print('-t -> activate training mode')
            print('-m -> methods')
            print('   -> allowed methods: skm, em, dbscan, sdbscan, isoforest')
            print('-x -> export model name')
            print('-v -> validation')
            print('-q -> query string for anomaly detection')
            print("#" * 100)
            sys.exit(0)
        elif opt in ("-e", "--endpoint"):
            settings['esendpoint'] = arg
        elif opt in ("-t"):
            settings["train"] = True
        elif opt in ("-f", "--file"):
            settings["file"] = arg
        elif opt in ("-m", "--method"):
            settings["method"] = arg
        elif opt in ("-v"):
            settings["validate"] = True
        elif opt in ("-x", "--export"):
            settings["export"] = arg
        elif opt in ("-d", "--detect"):
            settings["detect"] = arg
        elif opt in ("-l", "--list-models"):
            print ("Current saved models are:\n")
            print((getModelList()))
            sys.exit(0)
        elif opt in ("-q", "--query"):
            settings["query"] = arg

    # print("#" * 100)
    # print(queryDir)
    logger.info('[{}] : [INFO] Starting EDE framework ...'.format(
        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
    logger.info('[{}] : [INFO] Trying to read configuration file ...'.format(
        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))

    if settings["file"] is None:
        file_conf = 'ede_config.yaml'
        logger.info('[%s] : [INFO] Settings file set to %s',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), file_conf)
    else:
        if os.path.isfile(settings["file"]):
            file_conf = settings["file"]
            logger.info('[%s] : [INFO] Settings file set to %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), file_conf)
        else:
            logger.error('[%s] : [ERROR] Settings file not found at locations %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings["file"])
            sys.exit(1)

    readCnf = readConf(file_conf)
    logger.info('[{}] : [INFO] Reading configuration file ...'.format(
        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))

    # TODO: create def dls(file_conf)
    # Connector
    try:
        logger.info('[{}] : [INFO] Index Name set to : {}'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
            readCnf['Connector']['indexname']))
    except:
        logger.warning('[%s] : [WARN] Index not set in conf setting to default value %s',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['index'])

    if settings['esendpoint'] is None:
        try:
            logger.info('[{}] : [INFO] Monitoring ES Backend endpoint in config {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                readCnf['Connector']['ESEndpoint']))
            settings['esendpoint'] = readCnf['Connector']['ESEndpoint']
        except:
            if readCnf['Connector']['PREndpoint'] is None:  # todo; now only available in config file not in commandline
                logger.error('[%s] : [ERROR] ES and PR backend Enpoints not set in conf or commandline!',
                                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                sys.exit(1)
            else:
                settings['prendpoint'] = readCnf['Connector']['PREndpoint']
                logger.info('[{}] : [INFO] Monitoring PR Endpoint set to {}'.format(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                            settings["prendpoint"]))
    else:
        logger.info('[%s] : [INFO] ES Backend Enpoint set to %s',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['esendpoint'])
    if settings["from"] is None:
        try:
            settings["from"] = readCnf['Connector']['From']
            logger.info('[%s] : [INFO] From timestamp set to %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                        settings["from"])
        except:
            logger.info('[{}] : [INFO] PR Backend endpoint set to {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['prendpoint']))
            if settings['prendpoint'] is not None:
                logger.info('[{}] : [INFO] PR Backedn endpoint set to {}'.format(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['prendpoint']))
            else:
                logger.error('[%s] : [ERROR] From timestamp not set in conf or commandline!',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                sys.exit(1)
    else:
        logger.info('[%s] : [INFO] From timestamp set to %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['from'])

    if settings["to"] is None:
        try:
            settings["to"] = readCnf['Connector']['to']
            logger.info('[%s] : [INFO] To timestamp set to %s',
                                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                                settings["to"])
        except:
            if settings['prendpoint'] is not None:
                pass
            else:
                logger.error('[%s] : [ERROR] To timestamp not set in conf or commandline!',
                                     datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                sys.exit(1)
    else:
        logger.info('[%s] : [INFO] To timestamp set to %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['to'])

    if settings['query'] is None:
        try:
            settings['query'] = readCnf['Connector']['Query']
            logger.info('[%s] : [INFO] Query set to %s',
                                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                                settings['query'])
        except:
            if settings['prendpoint'] is not None:
                pass
            logger.error('[%s] : [ERROR] Query not set in conf or commandline!',
                                 datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
            sys.exit(1)
    else:
        logger.info('[%s] : [INFO] Query set to %s',
                           datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['query'])

    if settings.prkafkaendpoint is None:
        try:
            settings.prkafkaendpoint = readCnf['Connector']['KafkaEndpoint']
            if settings.prkafkaendpoint == 'None':
                settings.prkafkaendpoint = None
            else:
                settings.prkafkatopic = readCnf['Connector']['KafkaTopic']
                settings.prkafkaport = readCnf['Connector']['KafkaPort']
            logger.info('[{}] : [INFO] Kafka Endpoint set to  {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings.prkafkaendpoint))
        except:
            logger.warning('[{}] : [WARN] Kafka Endpoint not set.'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings.prkafkaendpoint))

    if settings["nodes"] is None:
        try:
            if not readCnf['Connector']['nodes']:
                readCnf['Connector']['nodes'] = 0
            settings["nodes"] = readCnf['Connector']['nodes']
            logger.info('[%s] : [INFO] Desired nodes set to %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                    settings['nodes'])
        except:
            logger.warning('[%s] : [WARN] No nodes selected from config file or comandline querying all',
                           datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
            settings["nodes"] = 0
    else:
        logger.info('[%s] : [INFO] Desired nodes set to %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings["nodes"])

    if settings["qsize"] is None:
        try:
            settings["qsize"] = readCnf['Connector']['QSize']
            logger.info('[%s] : [INFO] Query size set to %s',
                                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                                settings['qsize'])
        except:
            logger.warning('[%s] : [WARN] Query size not set in conf or commandline setting to default',
                                 datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
            settings["qsize"] = 'default'
    else:
        logger.info('[%s] : [INFO] Query size set to %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings["qsize"])

    if settings["qinterval"] is None:
        try:
            settings["qinterval"] = readCnf['Connector']['MetricsInterval']
            logger.info('[%s] : [INFO] Metric Interval set to %s',
                                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                                settings['qinterval'])
        except:
            logger.warning('[%s] : [WARN] Metric Interval not set in conf or commandline setting to default',
                                 datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
            settings["qsize"] = "default"
    else:
        logger.info('[%s] : [INFO] Metric interval set to %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings["qinterval"])
    if readCnf['Connector']['Dask']:
        try:
            settings['Dask']['SchedulerEndpoint'] = readCnf['Connector']['Dask']['SchedulerEndpoint']
            settings['Dask']['SchedulerPort'] = readCnf['Connector']['Dask']['SchedulerPort']
            settings['Dask']['EnforceCheck'] = readCnf['Connector']['Dask']['EnforceCheck']
            logger.info('[{}] : [INFO] Dask scheduler  set to: endpoint {}, port {}, check {}'.format(
        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['Dask']['SchedulerEndpoint'],
                settings['Dask']['SchedulerPort'], settings['Dask']['EnforceCheck']))
            dask_backend = True
        except:
            logger.warning('[{}] : [WARN] Dask scheduler  set to default values'.format(datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
            dask_backend = False
    if settings['local'] is None:
        try:
            settings['local'] = readCnf['Connector']['Local']
            logger.info('[{}] : [INFO] Local datasource set to {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['local']))
        except:
            logger.info('[{}] : [INFO] Local datasource set to default'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
            settings['local'] = None
    else:
        logger.info('[{}] : [INFO] Local datasource set to {}'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['local']))
    # Mode
    if settings["train"] is None:
        try:
            settings["train"] = readCnf['Mode']['Training']
            logger.info('[%s] : [INFO] Train is set to %s from conf',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['train'])
        except:
            logger.error('[%s] : [ERROR] Train is not set in conf or comandline!',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
            sys.exit(1)
    else:
        logger.info('[%s] : [INFO] Train is set to %s from comandline',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['train'])

    # Analysis
    if settings.analysis is None:
        try:
            logger.info('[{}] : [INFO] Loading user defined analysis'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
            settings.analysis = readCnf['Analysis']
        except:
            logger.info('[{}] : [INFO] No user defined analysis detected'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))

    # Validate
    if settings["validate"] is None:
        try:
            settings["validate"] = readCnf['Mode']['Validate']
            logger.info('[%s] : [INFO] Validate is set to %s from conf',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['validate'])
        except:
            logger.error('[%s] : [ERROR] Validate is not set in conf or comandline!',
                                 datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
            sys.exit(1)
    else:
        logger.info('[%s] : [INFO] Validate is set to %s from comandline',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['validate'])

    # Detect
    if settings["detect"] is None:
        try:
            settings["detect"] = readCnf['Mode']['Detect']
            logger.info('[%s] : [INFO] Detect is set to %s from conf',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['detect'])
        except:
            logger.error('[%s] : [ERROR] Detect is not set in conf or comandline!',
                                 datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
            sys.exit(1)
    else:
        logger.info('[%s] : [INFO] Detect is set to %s from comandline',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['detect'])

    if settings["detectMethod"] is None:
        try:
            settings["detectMethod"] = readCnf['Detect']['Method']
            logger.info('[%s] : [INFO] Detect Method is set to %s from conf',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings["detectMethod"])
        except:
            logger.error('[%s] : [ERROR] Detect Method is not set in conf or comandline!',
                                 datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
            sys.exit(1)
    else:
        logger.info('[%s] : [INFO] Detect Method is set to %s from comandline',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings["detectMethod"])

    if settings["detecttype"] is None:
        try:
            settings["detecttype"] = readCnf['Detect']['Type']
            logger.info('[{}] : [INFO] Detect Type is set to {} from conf'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings["detecttype"]))
        except:
            logger.error('[%s] : [ERROR] Detect Type is not set in conf or command line!',
                                 datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
            sys.exit(1)
    else:
        logger.info('[%s] : [INFO] Detect Type is set to %s from command line',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings["detecttype"])

    if settings["trainMethod"] is None:
        try:
            settings["trainMethod"] = readCnf['Training']['Method']
            logger.info('[%s] : [INFO] Train Method is set to %s from conf',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings["trainMethod"])
        except:
            logger.error('[%s] : [ERROR] Train Method is not set in conf or comandline!',
                                 datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
            sys.exit(1)
    else:
        logger.info('[%s] : [INFO] Train Method is set to %s from comandline',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings["trainMethod"])

    if settings["traintype"] is None:
        try:
            settings["traintype"] = readCnf['Training']['Type']
            logger.info('[%s] : [INFO] Train Type is set to %s from conf',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings["traintype"])
        except:
            logger.error('[%s] : [ERROR] Train Type is not set in conf or command line!',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
            sys.exit(1)
    else:
        logger.info('[%s] : [INFO] Train Type is set to %s from command line',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings["traintype"])
    if settings.target is None:
        try:
            settings.target = readCnf['Training']['Target']
            logger.info('[{}] : [INFO] Classification Target set to {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings.target))
        except:
            if settings['traintype'] == 'classification':
                logger.warning('[{}] : [WARN] Classification Target not set in config'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings.target))
            else:
                pass

    if settings.hpomethod is None:
        try:
            settings.hpomethod = readCnf['Training']['HPOMethod']
            logger.info('[{}] : [INFO] HPO method set to {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings.hpomethod))
            try:
                settings.hpoparam = readCnf['Training']['HPOParam']
                for k, v in readCnf['Training']['HPOParam'].items():
                    logger.info('[{}] : [INFO] HPO Method {}  Param {} set to {}'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings.hpomethod, k, v))
            except:
                logger.warn('[{}] : [WARN] HPO Method Params set to default!'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
                settings.hpoparam = {}
        except:
            if readCnf['Training']['Type'] == 'hpo':
                logger.error('[{}] : [ERROR] HPO invoked without method! Exiting'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings.hpomethod))
                sys.exit(1)
            else:
                pass

    if settings.ParamDistribution is None:
        try:
            settings.ParamDistribution = readCnf['Training']['ParamDistribution']
            logger.info('[{}] : [INFO] HPO Parameter Distribution found.'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
        except:
            if readCnf['Training']['Type'] == 'hpo':
                logger.error('[{}] : [ERROR] HPO invoked without Parameter distribution! Exiting'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings.hpomethod))
                sys.exit(1)
            else:
                pass

    if settings["export"] is None:
        try:
            settings["export"] = readCnf['Training']['Export']
            logger.info('[%s] : [INFO] Export is set to %s from conf',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings["export"])
        except:
            logger.error('[%s] : [ERROR] Export is not set in conf or comandline!',
                                 datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
            sys.exit(1)
    else:
        logger.info('[%s] : [INFO] Model is set to %s from comandline',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings["export"])

    if settings.cv is None:
        try:
            settings.cv = readCnf['Training']['CV']
            try:
                logger.info('[{}] : [INFO] Cross Validation set to {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['cv']['Type']))
            except:
                logger.info('[{}] : [INFO] Cross Validation set to {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['cv']))
                try:
                    settings['cv'] = int(settings['cv'])
                except:
                    logger.error('[{}] : [ERROR] Issues with CV definition in Training!'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
                    sys.exit(1)
        except:
            logger.info('[{}] : [INFO] Cross Validation not defined'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))

    if settings.trainscore is None:
        try:
            settings.trainscore = readCnf['Training']['TrainScore']
            logger.info('[{}] : [INFO] Cross Validation set to include training scores'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
        except:
            settings.trainscore = False

    if settings.scorer is None:
        try:
            settings.scorer = readCnf['Training']['Scorers']
            logger.info('[{}] : [INFO] Training scorers defined'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
        except:
            logger.info('[{}] : [INFO] No Training scorers defined'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))

    if settings.returnestimators is None:
        try:
            settings.returnestimators = readCnf['Training']['ReturnEstimators']
            logger.info('[{}] : [INFO] CV Estimators will be saved'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
        except:
            settings.returnestimators = False

    if settings["load"] is None:
        try:
            settings["load"] = readCnf['Detect']['Load']
            logger.info('[%s] : [INFO] Load is set to %s from conf',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings["load"])
        except:
            logger.error('[%s] : [ERROR] Load is not set in conf or comandline!',
                                 datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
            sys.exit(1)
    else:
        logger.info('[%s] : [INFO] Load is set to %s from comandline',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings["load"])

    if settings.detectionscaler is None:
        try:
            settings.detectionscaler = readCnf['Detect']['Scaler']
            logger.info('[{}] : [INFO] Detection Scaler set to {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings.detectionscaler))
        except:
            settings.detectionscaler = None
            logger.warning('[{}] : [WARN] Detection scaler not specified'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))

    try:
        settings['MethodSettings'] = {}   #todo read settings from commandline ?
        for name, value in readCnf['Training']['MethodSettings'].items():
            # print("%s -> %s" % (name, value))
            settings['MethodSettings'][name] = value
    except:
        settings['MethodSettings'] = None
        logger.warning('[%s] : [WARN] No Method settings detected, using defaults for %s!',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings["method"])

    # Augmentation
    try:
        settings['augmentation'] = readCnf['Augmentation']
        logger.info('[%s] : [INFO] Augmentations loaded',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    except:
        settings['augmentation'] = None
        logger.info('[%s] : [INFO] Augmentations not defined',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

    # Point anomaly settings
    try:
        settings["smemory"] = readCnf['Point']['memory']
        logger.info('[%s] : [INFO] System memory is set to %s',
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings["smemory"])
    except:
        settings["smemory"] = "default"
        logger.warning('[%s] : [WARN] System memory is not set, using default!',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

    try:
        settings["sload"] = readCnf['Point']['load']
        logger.info('[%s] : [INFO] System load is  set to %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings["sload"])
    except:
        settings["sload"] = "default"
        logger.warning('[%s] : [WARN] System load is not set, using default!',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

    try:
        settings["snetwork"] = readCnf['Point']['network']
        logger.info('[%s] : [INFO] System netowrk is  set to %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings["snetwork"])
    except:
        settings["snetwork"] = "default"
        logger.warning('[%s] : [WARN] System network is not set, using default!',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

    try:
        settings['heap'] = readCnf['Misc']['heap']
        logger.info('[%s] : [INFO] Heap size set to %s',
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['heap'])
    except:
        settings['heap'] = '512m'
        logger.info('[%s] : [INFO] Heap size set to default %s',
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['heap'])

    # Filter
    try:
        if readCnf['Filter']['Columns']:
            logger.info('[{}] : [INFO] Filter columns set in config as {}.'.format(
        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), readCnf['Filter']['Columns']))
            settings["cfilter"] = readCnf['Filter']['columns']
        else:
            logger.info('[{}] : [INFO] Filter columns set in config as {}.'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings["cfilter"]))
    except:
        pass
    finally:
        logger.info('[%s] : [INFO] Filter column set to %s',
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['cfilter'])

    try:
        # logger.info('[%s] : [INFO] Filter rows set to %s',
        #             datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), readCnf['Filter']['Rows'])
        settings["rfilter"] = readCnf['Filter']['Rows']
    except:
        pass
        # logger.info('[%s] : [INFO] Filter rows  %s',
        #             datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings["rfilter"])
    finally:
        logger.info('[%s] : [INFO] Filter rows set to %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['rfilter'])

    try:
        if readCnf['Filter']['DColumns']:
            # print("Filter drop columns -> %s" % readCnf['Filter']['DColumns'])
            settings["dfilter"] = readCnf['Filter']['DColumns']
        else:
            # print("Filter drop columns -> %s" % settings["dfilter"])
            pass
    except:
        # print("Filter drop columns -> %s" % settings["dfilter"])
        pass
    finally:
        logger.info('[%s] : [INFO] Filter drop column set to %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['dfilter'])

    try:
        if readCnf['Filter']['Fillna']:
            settings['fillna'] = readCnf['Filter']['Fillna']
        else:
            settings['fillna'] = False
        logger.info('[{}] : [INFO] Fill None values set to {}'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), readCnf['Filter']['Fillna']))
    except:
        logger.info('[{}] : [INFO] Fill None not set, skipping ...'.format(
        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
        settings['fillna'] = False

    try:
        if readCnf['Filter']['Dropna']:
            settings['dropna'] = readCnf['Filter']['Dropna']
        else:
            settings['dropna'] = False
        logger.info('[{}] : [INFO] Drop None values set to {}'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), readCnf['Filter']['Dropna']))
    except:
        logger.info('[{}] : [INFO] Drop None not set, skipping ...'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
        settings['dropna'] = False

    if settings["checkpoint"] is None:
        try:

            settings["checkpoint"] = readCnf['Misc']['checkpoint']
            logger.info('[%s] : [INFO] Checkpointing is  set to %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['checkpoint'])
        except:
            settings["checkpoint"] = "True"
            logger.info('[%s] : [INFO] Checkpointing is  set to True',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    else:
        logger.info('[%s] : [INFO] Checkpointing is  set to %s',
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['checkpoint'])

    if settings["delay"] is None:
        try:

            settings["delay"] = readCnf['Misc']['delay']
            # logger.info('[%s] : [INFO] Delay is  set to %s',
            #         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['delay'])
        except:
            settings["delay"] = "2m"
        logger.info('[%s] : [INFO] Delay is  set to %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['delay'])
    else:
        logger.info('[%s] : [INFO] Delay is  set to %s',
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['delay'])

    if settings["interval"] is None:
        try:

            settings["interval"] = readCnf['Misc']['interval']
            logger.info('[%s] : [INFO] Interval is  set to %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['interval'])
        except:

            settings["interval"] = "15m"
            logger.info('[%s] : [INFO] Interval is  set to %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['interval'])
    else:
        logger.info('[%s] : [INFO] Interval is  set to %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['interval'])

    if settings["resetindex"] is None:
        try:

            settings["resetindex"] = readCnf['Misc']['resetindex']
        except:

            settings["resetindex"] = False
    else:
        logger.info('[%s] : [INFO] Reset index set to %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['resetindex'])

    try:
        settings['dmonPort'] = readCnf['Connector']['dmonport']
        logger.info('[{}] : [INFO] DMon Port is set to {}'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
            settings['dmonPort']))
    except:
        logger.info('[%s] : [INFO] DMon Port is set to %s"',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(settings['dmonPort']))

    try:
        settings['training'] = readCnf['Detect']['training']
        logger.info('[{}] : [INFO] Classification Training set is {}'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
            readCnf['Detect']['training']))
    except:
        logger.info('[%s] : [INFO] Classification Training set is %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(settings['training']))

    # try:
    #     print("Classification Validation set is %s" % readCnf['Detect']['validation'])
    #     settings['validation'] = readCnf['Detect']['validation']
    # except:
    #     print("Classification Validation set is default")
    # logger.info('[%s] : [INFO] Classification Validation set is %s',
    #             datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(settings['validation']))


    try:
        # print("Classification validation ratio is set to %d" % int(readCnf['Training']['ValidRatio']))
        logger.info('[{}] : [INFO] Classification validation ratio is set to {}'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), readCnf['Training']['ValidRatio']))
        if float(readCnf['Training']['ValidRatio']) > 1.0:
            # print("Validation ratio is out of range, must be between 1.0 and 0.1")
            settings['validratio'] = 0.0
            logger.warning('[{}] : [WARN] Validation ratio is out of range, must be between 1.0 and 0.1, overwritting'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), readCnf['Training']['ValidRatio']))
        settings['validratio'] = float(readCnf['Detect']['validratio'])
    except:
        logger.warning('[{}] : [WARN] Validation ratio is set to default'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
    logger.info('[%s] : [INFO] Classification Validation ratio is %s',
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(settings['validratio']))

    # try:
    #     print("Classification comparison is set to %s" % readCnf['Detect']['compare'])
    #     settings['compare'] = readCnf['Detect']['compare']
    # except:
    #     print("Classification comparison is default")
    # logger.info('[%s] : [INFO] Classification comparison is %s',
    #             datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['compare'])

    try:
        # print("Classification data generation using only anomalies set to %s" % readCnf['Detect']['anomalyOnly'])
        settings['anomalyOnly'] = readCnf['Detect']['anomalyOnly']
    except:
        # print("Classification data generation using only anomalies set to False")
        pass
    logger.info('[%s] : [INFO] Classification data generation using only anomalies set to %s',
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(settings['anomalyOnly']))

    if settings["categorical"] is None:
        try:
            if not readCnf['Augmentation']['Categorical']:
                readCnf['Augmentation']['Categorical'] = None
                logger.info('[{}] : [INFO] Categorical columns defined as: {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                    readCnf['Augmentation']['Categorical']))
            if readCnf['Augmentation']['Categorical'] == '0':
                settings["categorical"] = None
            else:
                settings["categorical"] = readCnf['Augmentation']['Categorical']
            logger.info('[%s] : [INFO] Categorical Features ->  %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                    settings['categorical'])
        except:
            logger.warning('[%s] : [WARN] No Categorical Features selected from config file or comandline! Skipping encoding',
                           datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
            settings["categorical"] = None
    else:
        logger.info('[%s] : [INFO] Categorical Features ->  %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings["categorical"])

    if not settings["point"]:
        try:
            settings['point'] = readCnf['Misc']['point']
            logger.info('[%s] : [INFO] Point  set to %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['point'])
        except:
            settings['point'] = 'False'
            logger.info('[%s] : [INFO] Point detection set to default %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), settings['point'])

    #print dmonC
    # sys.exit()
    # print("Conf file -> %s" %readCnf)
    # print("Settings  -> %s" %settings)

    engine = aspideedengine.EDEngine(settings,
                                     dataDir=dataDir,
                                     modelsDir=modelsDir,
                                     queryDir=queryDir)
    #engine.printTest()
    engine.initConnector()
    if dask_backend:
        engine.runDask(engine)
    else:
        try:
            engine.runProcess(engine)
        except Exception as inst:
            logger.error('[{}] : [ERROR] Failed Process backend initialization with {} and {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
            logger.warning('[{}] : [WARN] Initializing default threaded engine, limited performance to be expected!'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
            engine.run(engine)

    logger.info('[{}] : [INFO] Exiting EDE framework'.format(
        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))


if __name__ == "__main__":
    def handler(singal_received, frame):
        logger.info('[{}] : [INFO] User break detected. Exiting EDE framework'.format(
        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
        sys.exit(0)
    signal(SIGINT, handler)
    SchedulerEndpoint, Scale, SchedulerPort, EnforceCheck = check_dask_settings()  # Todo Better solution
    if SchedulerEndpoint:
        if SchedulerEndpoint == "local":
            cluster = LocalCluster(n_workers=int(Scale))
            logger.info('[{}] : [INFO] Starting Dask local Cluster Backend with: {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), cluster))
            client = Client(cluster)
            logger.info('[{}] : [INFO] Dask Client started with: {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), client))
        else:
            scheduler_address = "{}:{}".format(SchedulerEndpoint, SchedulerPort)
            client = Client(address=scheduler_address)
            client.get_versions(check=EnforceCheck)
    else:
        cluster = 0
        client = 0
    main(sys.argv[1:],
         cluster,
         client)

