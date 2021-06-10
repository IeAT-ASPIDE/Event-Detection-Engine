"""
Copyright 2021, Institute e-Austria, Timisoara, Romania
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

from datetime import datetime
from elasticsearch import Elasticsearch
from kafka import KafkaProducer
import pandas as pd
import requests
import os
import sys, getopt
from edelogger import logger
import json
import time


class Connector:
    def __init__(self,
                 prEndpoint=None,
                 esEndpoint=None,
                 dmonPort=5001,
                 MInstancePort=9200,
                 index="logstash-*",
                 prKafkaEndpoint=None,
                 prKafkaPort=9092,
                 prKafkaTopic='edetopic'):
        if esEndpoint is None:
            self.esInstance = None
        else:
            self.esInstance = Elasticsearch(esEndpoint)
            self.esEndpoint = esEndpoint
            self.dmonPort = dmonPort
            self.esInstanceEndpoint = MInstancePort
            self.myIndex = index
            logger.info('[{}] : [INFO] EDE ES backend Defined at: {} with port {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), esEndpoint, MInstancePort))
        if prEndpoint is None:
            pass
        else:
            self.prEndpoint = prEndpoint
            self.MInstancePort = MInstancePort
            logger.info('[{}] : [INFO] EDE PR backend Defined at: {} with port {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), prEndpoint, MInstancePort))
            self.dataDir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
        if prKafkaEndpoint is None:
            self.producer = None
            logger.warning('[{}] : [WARN] EDE Kafka reporter not set'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
        else:
            self.prKafkaTopic = prKafkaTopic
            try:
                self.producer = KafkaProducer(value_serializer=lambda v: json.dumps(v).encode('utf-8'),
                                              bootstrap_servers=["{}:{}".format(prKafkaEndpoint, prKafkaPort)],
                                              retries=5)
                logger.info('[{}] : [INFO] EDE Kafka reporter initialized to server {}:{}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), prKafkaEndpoint, prKafkaPort))
            except Exception as inst:
                logger.error('[{}] : [ERROR] EDE Kafka reporter failed with {} and {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
                self.producer = None

    def pr_health_check(self):
        pr_target_health = '/-/healthy'
        pr_target_ready = '/-/ready'
        try:
            resp_h = requests.get("http://{}:{}{}".format(self.prEndpoint, self.MInstancePort, pr_target_health))
            resp_r = requests.get("http://{}:{}{}".format(self.prEndpoint, self.MInstancePort, pr_target_ready))
        except Exception as inst:
            logger.error(
                '[{}] : [ERROR] Exception has occured while connecting to PR endpoint with type {} at arguments {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
            sys.exit(2)
        if resp_h.status_code != 200:
            logger.error(
                '[{}] : [ERROR] PR endpoint health is bad, exiting'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
            sys.exit(2)
        if resp_r.status_code != 200:
            logger.error(
                '[{}] : [ERROR] PR endpoint not ready to serve traffic'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
            sys.exit(2)
        logger.info(
            '[{}] : [INFO] PR endpoint healthcheck pass'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
        return resp_h.status_code, resp_r.status_code

    def pr_status(self, type=None):
        """
        Get status of prometheus

        TODO: check runtimeinfo and flags
        :param type: suported types
        :return:
        """
        suported = ['runtimeinfo', 'config', 'flags']
        if type is None:
            pr_target_string = '/api/v1/status/config'
        elif type in suported:
            pr_target_string = '/api/v1/status/{}'.format(type)
        else:
            logger.error('[{}] : [ERROR] unsupported status type {}, supported types are {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type, suported))
            sys.exit(1)
        try:
            resp = requests.get("http://{}:{}{}".format(self.prEndpoint, self.MInstancePort, pr_target_string))
        except Exception as inst:
            logger.error(
                '[{}] : [ERROR] Exception has occured while connecting to PR endpoint with type {} at arguments {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
            sys.exit(2)
        return resp.json()

    def pr_targets(self):
        """
        Get Monitored Target Info
        :return: Targets Dict
        """
        pr_target_string = '/api/v1/targets'
        try:
            resp = requests.get("http://{}:{}{}".format(self.prEndpoint, self.MInstancePort, pr_target_string))
        except Exception as inst:
            logger.error(
                '[{}] : [ERROR] Exception has occured while connecting to PR endpoint with type {} at arguments {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
            sys.exit(2)
        return resp.json()

    def pr_labels(self, label=None):
        if label is None:
            pr_target_string = '/api/v1/labels'
        else:
            pr_target_string = '/api/v1/label/{}/values'.format(label)
        try:
            resp = requests.get("http://{}:{}{}".format(self.prEndpoint, self.MInstancePort, pr_target_string))
        except Exception as inst:
            logger.error(
                '[{}] : [ERROR] Exception has occured while connecting to PR endpoint with type {} at arguments {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
            sys.exit(2)
        return resp.json()

    def pr_query(self, query):
        """
        QUery Monitoring Data From PR backend
        :param query: Query string for PR backend
        :return: Monitoring Data
        """
        try:
            url = '/api/v1/query'
            resp = requests.get('http://{}:{}{}'.format(self.prEndpoint, self.MInstancePort, url), params=query)
        except Exception as inst:
            logger.error(
                '[{}] : [ERROR] Exception has occured while connecting to PR endpoint with type {} at arguments {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
            sys.exit(2)
        return resp.json()

    def query(self,
              queryBody,
              allm=True,
              dMetrics=[],
              debug=False):
        # self.__check_valid_es()
        res = self.esInstance.search(index=self.myIndex, body=queryBody, request_timeout=230)
        if debug:
            print("%---------------------------------------------------------%")
            print("Raw JSON Ouput")
            print(res)
            print(("%d documents found" % res['hits']['total']))
            print("%---------------------------------------------------------%")
        termsList = []
        termValues = []
        ListMetrics = []
        for doc in res['hits']['hits']:
            if not allm:
                if not dMetrics:
                    sys.exit("dMetrics argument not set. Please supply valid list of metrics!")
                for met in dMetrics:
                    # prints the values of the metrics defined in the metrics list
                    if debug:
                        print("%---------------------------------------------------------%")
                        print("Parsed Output -> ES doc id, metrics, metrics values.")
                        print(("doc id %s) metric %s -> value %s" % (doc['_id'], met, doc['_source'][met])))
                        print("%---------------------------------------------------------%")
                    termsList.append(met)
                    termValues.append(doc['_source'][met])
                dictValues = dict(list(zip(termsList, termValues)))
            else:
                for terms in doc['_source']:
                    # prints the values of the metrics defined in the metrics list
                    if debug:
                        print("%---------------------------------------------------------%")
                        print("Parsed Output -> ES doc id, metrics, metrics values.")
                        print(("doc id %s) metric %s -> value %s" % (doc['_id'], terms, doc['_source'][terms])))
                        print("%---------------------------------------------------------%")
                    termsList.append(terms)
                    termValues.append(doc['_source'][terms])
                    dictValues = dict(list(zip(termsList, termValues)))
            ListMetrics.append(dictValues)
        return ListMetrics, res

    def info(self):
        # self.__check_valid_es()
        try:
            res = self.esInstance.info()
        except Exception as inst:
            logger.error('[%s] : [ERROR] Exception has occured while connecting to ES dmon with type %s at arguments %s',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args)
            sys.exit(2)
        return res

    def roles(self):
        # self.__check_valid_es()
        nUrl = "http://%s:%s/dmon/v1/overlord/nodes/roles" % (self.esEndpoint, self.dmonPort)
        logger.info('[%s] : [INFO] dmon get roles url -> %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), nUrl)
        try:
            rRoles = requests.get(nUrl)
        except Exception as inst:
            logger.error('[%s] : [ERROR] Exception has occured while connecting to dmon with type %s at arguments %s',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args)
            sys.exit(2)
        rData = rRoles.json()
        return rData

    def createIndex(self, indexName):
        # self.__check_valid_es()
        try:
            self.esInstance.create(index=indexName, ignore=400)
            logger.info('[%s] : [INFO] Created index %s',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), indexName)
        except Exception as inst:
            logger.error('[%s] : [ERROR] Failed to created index %s with %s and %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), indexName, type(inst), inst.args)

    def closeIndex(self, indexName):
        try:
            self.esInstance.close(index=indexName)
            logger.info('[%s] : [INFO] Closed index %s',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), indexName)
        except Exception as inst:
            logger.error('[%s] : [ERROR] Failed to close index %s with %s and %s',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), indexName, type(inst),
                         inst.args)

    def deleteIndex(self, indexName):
        try:
            res = self.esInstance.indices.delete(index=indexName, ignore=[400, 404])
            logger.info('[%s] : [INFO] Deleted index %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), indexName)
        except Exception as inst:
            logger.error('[%s] : [ERROR] Failed to delete index %s with %s and %s',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), indexName, type(inst),
                         inst.args)
            return 0
        return res

    def openIndex(self, indexName):
        res = self.esInstance.indices.open(index=indexName)
        logger.info('[%s] : [INFO] Open index %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), indexName)
        return res

    def getIndex(self, indexName):
        res = self.esInstance.indices.get(index=indexName, human=True)
        return res

    def getIndexSettings(self, indexName):
        res = self.esInstance.indices.get_settings(index=indexName, human=True)
        return res

    def clusterHealth(self):
        res = self.esInstance.cluster.health(request_timeout=15)
        return res

    def clusterSettings(self):
        res = self.esInstance.cluster.get_settings(request_timeout=15)
        return res

    def clusterState(self):
        res = self.esInstance.cluster.stats(human=True, request_timeout=15)
        return res

    def nodeInfo(self):
        res = self.esInstance.nodes.info(request_timeout=15)
        return res

    def nodeState(self):
        res = self.esInstance.nodes.stats(request_timeout=15)
        return res

    def getStormTopology(self):
        nUrl = "http://%s:%s/dmon/v1/overlord/detect/storm" % (self.esEndpoint, self.dmonPort)
        logger.info('[%s] : [INFO] dmon get storm topology url -> %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), nUrl)
        try:
            rStormTopology = requests.get(nUrl)
        except Exception as inst:
            logger.error('[%s] : [ERROR] Exception has occured while connecting to dmon with type %s at arguments %s',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args)
            print("Can't connect to dmon at %s port %s" % (self.esEndpoint, self.dmonPort))
            sys.exit(2)
        rData = rStormTopology.json()
        return rData

    def pushAnomalyES(self, anomalyIndex, doc_type, body):
        try:
            res = self.esInstance.index(index=anomalyIndex, doc_type=doc_type, body=body)
        except Exception as inst:
            logger.error('[%s] : [ERROR] Exception has occured while pushing anomaly with type %s at arguments %s',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args)
            sys.exit(2)
        return res

    def pushAnomalyKafka(self, body):
        if self.producer is None:
            logger.warning('[{}] : [WARN] Kafka reporter not defined, skipping reporting'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
        else:
            try:
                self.producer.send(self.prKafkaTopic, body)
                # self.producer.flush()
                logger.info('[{}] : [INFO] Anomalies reported to kafka topic {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), self.prKafkaTopic))
            except Exception as inst:
                logger.error('[{}] : [ERROR] Failed to report anomalies to kafka topic {} with {} and {}'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), self.prKafkaTopic, type(inst), inst.args))
        return 0

    def getModel(self):
        return "getModel"

    def pushModel(self):
        return "push model"

    def localData(self, data):
        data_loc = os.path.join(self.dataDir, data)
        try:
            df = pd.read_csv(data_loc)
        except Exception as inst:
            logger.error('[{}] : [ERROR] Cannot load local data with  {} and {}'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
            sys.exit(2)
        logger.info('[{}] : [INFO] Loading local data from {} with shape {}'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), data_loc, df.shape))
        return df

    def getInterval(self):
        nUrl = "http://%s:%s/dmon/v1/overlord/aux/interval" % (self.esEndpoint, self.dmonPort)
        logger.info('[%s] : [INFO] dmon get interval url -> %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), nUrl)
        try:
            rInterval = requests.get(nUrl)
        except Exception as inst:
            logger.error('[%s] : [ERROR] Exception has occured while connecting to dmon with type %s at arguments %s',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args)
            sys.exit(2)
        rData = rInterval.json()
        return rData

    def aggQuery(self, queryBody):
        adt_timeout = os.environ['ADP_TIMEOUT'] = os.getenv('ADP_TIMEOUT', str(60)) # Set timeout as env variable ADT_TIMEOUT, if not set use default 60
        # print "QueryString -> {}".format(queryBody)
        try:
            res = self.esInstance.search(index=self.myIndex, body=queryBody, request_timeout=float(adt_timeout))
        except Exception as inst:
            logger.error('[%s] : [ERROR] Exception while executing ES query with %s and %s',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args)
            sys.exit(2)
        return res

    def getNodeList(self):
        '''
        :return: -> returns the list of registered nodes from dmon
        '''
        nUrl = "http://%s:%s/dmon/v1/observer/nodes" % (self.esEndpoint, self.dmonPort)
        logger.info('[%s] : [INFO] dmon get node url -> %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), nUrl)
        try:
            rdmonNode = requests.get(nUrl)
        except Exception as inst:
            logger.error('[%s] : [ERROR] Exception has occured while connecting to dmon with type %s at arguments %s',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args)
            sys.exit(2)
        rdata = rdmonNode.json()
        nodes = []
        for e in rdata['Nodes']:
            for k in e:
                nodes.append(k)
        return nodes

    def getDmonStatus(self):
        nUrl = "http://%s:%s/dmon/v1/overlord/core/status" % (self.esEndpoint, self.dmonPort)
        logger.info('[%s] : [INFO] dmon get core status url -> %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), nUrl)
        try:
            rdmonStatus = requests.get(nUrl)
        except Exception as inst:
            logger.error('[%s] : [ERROR] Exception has occured while connecting to dmon with type %s at arguments %s',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args)
            sys.exit(2)
        return rdmonStatus.json()

    # def __check_valid_es(self, func):
    #     @functools.wraps(func)
    #     def wrap(self, *args, **kwargs):
    #         if not self.esEndpoint:
    #             logger.error('[%s] : [ERROR] ES Endpoint not defined in config',
    #                          datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    #             sys.exit(1)
    #         else:
    #             return func(self, *args, **kwargs)
    #     return wrap
    #
    # def __check_valid_pr(self, func):
    #     def wrapper(*args, **kwargs):
    #         if not prEndpoint:
    #             logger.error('[%s] : [ERROR] PR Endpoint not defined in config',
    #                          datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
    #             sys.exit(1)
    #         else:
    #             func(*args, **kwargs)
    #     return wrapper


