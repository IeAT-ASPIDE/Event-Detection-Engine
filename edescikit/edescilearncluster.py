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
import joblib
import numpy as np
from sklearn.cluster import DBSCAN
from sklearn.ensemble import IsolationForest
from sklearn import metrics
# from sklearn.datasets.samples_generator import make_blobs
from sklearn.preprocessing import StandardScaler
import pickle as pickle
import os
from util import str2Bool
import pandas as pd
from edelogger import logger
from datetime import datetime
import time
import sys
import glob
from util import ut2hum


class SciCluster:
    def __init__(self, modelDir):
        self.modelDir = modelDir

    def dask_sdbscanTrain(self, settings,
                          mname,
                          data,
                          scaler=None):
        '''
        :param data: -> dataframe with data
        :param settings: -> settings dictionary
        :param mname: -> name of serialized clusterer
        :param scaler: -> scaler to use on data
        :return: -> clusterer
        :example settings: -> {eps:0.9, min_samples:10, metric:'euclidean' ,
        algorithm:'auto, leaf_size:30, p:0.2, n_jobs:1}
        '''

        if scaler is None:
            logger.warning('[{}] : [WARN] Scaler not defined'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
        else:
            logger.info('[{}] : [INFO] Scaling data ...'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
            data = scaler.fit_transform(data)

        if not settings or settings is None:
            logger.warning('[{}] : [WARN] No DBScan parameters defined using default'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
            settings = {}
        else:
            for k, v in settings.items():
                logger.info('[{}] : [INFO] DBScan parameter {} set to {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), k, v))

        try:
            db = DBSCAN(**settings).fit(data)
        except Exception as inst:
            logger.error('[{}] : [INFO] Failed to instanciate DBScan with {} and {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
            sys.exit(1)
        labels = db.labels_
        logger.info('[{}] : [INFO] DBScan labels: {} '.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), labels))
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        logger.info('[{}] : [INFO] DBScan estimated number of clusters {} '.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), n_clusters_))
        self.__serializemodel(db, 'sdbscan', mname)
        return db

    def sdbscanTrain(self, settings,
                     mname,
                     data):
        '''
        :param data: -> dataframe with data
        :param settings: -> settings dictionary
        :param mname: -> name of serialized clusterer
        :return: -> clusterer
        :example settings: -> {eps:0.9, min_samples:10, metric:'euclidean' ,
        algorithm:'auto, leaf_size:30, p:0.2, n_jobs:1}
        '''
        for k, v in settings.items():
            logger.info('[%s] : [INFO] SDBSCAN %s set to %s',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), k, v)
        sdata = StandardScaler().fit_transform(data)
        try:
            db = DBSCAN(eps=float(settings['eps']), min_samples=int(settings['min_samples']), metric=settings['metric'],
                        algorithm=settings['algorithm'], leaf_size=int(settings['leaf_size']), p=float(settings['p']),
                        n_jobs=int(settings['n_jobs'])).fit(sdata)
        except Exception as inst:
            logger.error('[%s] : [ERROR] Cannot instanciate sDBSCAN with %s and %s',
                           datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args)
            print("Error while  instanciating sDBSCAN with %s and %s" % (type(inst), inst.args))
            sys.exit(1)
        labels = db.labels_
        print(labels)
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)
        print('Estimated number of clusters: %d' % n_clusters_)
        self.__serializemodel(db, 'sdbscan', mname)
        return db

    def dask_isolationForest(self, settings,
                             mname,
                             data
                             ):
        '''
        :param settings: -> settings dictionary
        :param mname: -> name of serialized clusterer
        :param scaler: -> scaler to use on data
        :return: -> isolation forest instance
        :example settings: -> {n_estimators:100, max_samples:100, contamination:0.1, bootstrap:False,
                        max_features:1.0, n_jobs:1, random_state:None, verbose:0}
        '''
        if not settings or settings is None:
            logger.warning('[{}] : [WARN] No IsolationForest parameters defined using defaults'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
            # print(settings)
            settings = {}
        else:
            for k, v in settings.items():
                logger.info('[{}] : [INFO] IsolationForest parameter {} set to {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), k, v))
        try:

            clf = IsolationForest(**settings)
            # print(clf)
        except Exception as inst:
            logger.error('[{}] : [INFO] Failed to instanciate IsolationForest with {} and {}'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
            sys.exit(1)

        try:
            with joblib.parallel_backend('dask'):
                logger.info('[{}] : [INFO] Using Dask backend for IsolationForest'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
                clf.fit(data)
        except Exception as inst:
            logger.error('[{}] : [ERROR] Failed to fit IsolationForest with {} and {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
            sys.exit(1)

        predict = clf.predict(data)
        anoOnly = np.argwhere(predict == -1)
        logger.info('[{}] : [INFO] Found {} anomalies in training dataset of shape {}.'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), len(anoOnly), data.shape))
        logger.info('[{}] : [DEBUG] Predicted Anomaly Array {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), predict))
        self.__serializemodel(clf, 'isoforest', mname)
        self.__appendPredictions(method='isoforest', mname=mname, data=data, pred=predict)


    def isolationForest(self, settings,
                        mname,
                        data):
        '''
        :param settings: -> settings dictionary
        :param mname: -> name of serialized clusterer
        :return: -> isolation forest instance
        :example settings: -> {n_estimators:100, max_samples:100, contamination:0.1, bootstrap:False,
                        max_features:1.0, n_jobs:1, random_state:None, verbose:0}
        '''
        # rng = np.random.RandomState(42)
        if settings['random_state'] == 'None':
            settings['random_state'] = None

        if isinstance(settings['bootstrap'], str):
            settings['bootstrap'] = str2Bool(settings['bootstrap'])

        if isinstance(settings['verbose'], str):
            settings['verbose'] = str2Bool(settings['verbose'])

        if settings['max_samples'] != 'auto':
            settings['max_samples'] = int(settings['max_samples'])
        # print type(settings['max_samples'])
        for k, v in settings.items():
            logger.info('[%s] : [INFO] IsolationForest %s set to %s',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), k, v)
            print("IsolationForest %s set to %s" % (k, v))
        try:
            clf = IsolationForest(n_estimators=int(settings['n_estimators']), max_samples=settings['max_samples'], contamination=float(settings['contamination']), bootstrap=settings['bootstrap'],
                        max_features=float(settings['max_features']), n_jobs=int(settings['n_jobs']), random_state=settings['random_state'], verbose=settings['verbose'])
        except Exception as inst:
            logger.error('[%s] : [ERROR] Cannot instanciate isolation forest with %s and %s',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args)
            sys.exit(1)
        # clf = IsolationForest(max_samples=100, random_state=rng)
        # print "*&*&*&& %s" % type(data)
        try:
            clf.fit(data)
        except Exception as inst:
            logger.error('[%s] : [ERROR] Cannot fit isolation forest model with %s and %s',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args)
            sys.exit(1)
        predict = clf.predict(data)
        print("Anomaly Array:")
        print(predict)
        self.__serializemodel(clf, 'isoforest', mname)
        return clf

    def detect(self, method,
               model,
               data):
        '''
        :param method: -> method name
        :param model: -> trained clusterer
        :param data: -> dataframe with data
        :return: -> dictionary that contains the list of anomalous timestamps
        '''
        smodel = self.__loadClusterModel(method, model)
        anomalieslist = []
        if not smodel:
            dpredict = 0
        else:
            if data.shape[0]:
                if isinstance(smodel, IsolationForest):
                    logger.info('[{}] : [INFO] Loading predictive model IsolationForest ').format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                    for k, v in smodel.get_params().items():
                        logger.info('[{}] : [INFO] Predict model parameter {} set to {}'.format(
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), k, v))
                    # print("Contamination -> %s" % smodel.contamination)
                    # print("Max_Features -> %s" % smodel.max_features)
                    # print("Max_Samples -> %s" % smodel.max_samples_)
                    # print("Threashold -> %s " % smodel.threshold_)
                    try:
                        dpredict = smodel.predict(data)
                        logger.debug('[{}] : [DEBUG] IsolationForest prediction array: {}').format(
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(dpredict))
                    except Exception as inst:
                        logger.error('[%s] : [ERROR] Error while fitting isolationforest model to event with %s and %s',
                             datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args)
                        dpredict = 0

                elif isinstance(smodel, DBSCAN):
                    logger.info('[{}] : [INFO] Loading predictive model DBSCAN ').format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                    for k, v in smodel.get_params().items():
                        logger.info('[{}] : [INFO] Predict model parameter {} set to {}'.format(
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), k, v))
                    # print("Leaf_zise -> %s" % smodel.leaf_size)
                    # print("Algorithm -> %s" % smodel.algorithm)
                    # print("EPS -> %s" % smodel.eps)
                    # print("Min_Samples -> %s" % smodel.min_samples)
                    # print("N_jobs -> %s" % smodel.n_jobs)
                    try:
                        dpredict = smodel.fit_predict(data)
                    except Exception as inst:
                        logger.error('[%s] : [ERROR] Error while fitting sDBSCAN model to event with %s and %s',
                                     datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst),
                                     inst.args)
                        dpredict = 0
            else:
                dpredict = 0
                logger.warning('[%s] : [WARN] Dataframe empty with shape (%s,%s)',
                             datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(data.shape[0]),
                             str(data.shape[1]))
                print("Empty dataframe received with shape (%s,%s)" % (str(data.shape[0]),
                             str(data.shape[1])))
            print("dpredict type is %s" % (type(dpredict)))
        if type(dpredict) is not int:
            anomalyarray = np.argwhere(dpredict == -1)
            for an in anomalyarray:
                anomalies = {}
                anomalies['utc'] = int(data.iloc[an[0]].name)
                anomalies['hutc'] = ut2hum(int(data.iloc[an[0]].name))
                anomalieslist.append(anomalies)
        anomaliesDict = {}
        anomaliesDict['anomalies'] = anomalieslist
        logger.info('[%s] : [INFO] Detected anomalies with model %s using method %s are -> %s',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), model, method, str(anomaliesDict))
        return anomaliesDict

    def dask_detect(self,
                    method,
                    model,
                    data
                    ):
        smodel = self.__loadClusterModel(method, model)
        anomaliesList = []
        if not smodel:
            dpredict = 0
        else:
            if data.shape[0]:
                try:
                    logger.info('[{}] : [INFO] Loading predictive model {} '.format(
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(smodel).split('(')[0]))
                    for k, v in smodel.get_params().items():
                        logger.info('[{}] : [INFO] Predict model parameter {} set to {}'.format(
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), k, v))
                        dpredict = smodel.predict(data)
                except Exception as inst:
                    logger.error('[{}] : [ERROR] Failed to load predictive model with {} and {}'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
                    dpredict = 0
            else:
                dpredict = 0
                logger.warning('[{}] : [WARN] DataFrame is empty with shape {} '.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(data.shape)))
        if type(dpredict) is not int:
            anomalyArray = np.argwhere(dpredict == -1)
            for an in anomalyArray:
                anomalies = {}
                anomalies['utc'] = int(data.iloc[an[0]].name)
                anomalies['hutc'] = ut2hum(int(data.iloc[an[0]].name))
                anomaliesList.append(anomalies)
        anomaliesDict = {}
        anomaliesDict['anomalies'] = anomaliesList
        logger.info('[{}] : [INFO] Detected {} anomalies with model {} using method {} '.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), len(anomaliesList), model,
            str(smodel).split('(')[0]))
        return anomaliesDict

    def dask_clusterMethod(self, cluster_method,
                           mname,
                           data
                           ):
        try:
            logger.info('[{}] : [INFO] Loading Clustering method {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(cluster_method)))
            # delattr(cluster_method, 'behaviour')
            # del cluster_method.__dict__['behaviour']
            for k, v in cluster_method.get_params().items():
                logger.info('[{}] : [INFO] Method parameter {} set to {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), k, v))
            try:
                with joblib.parallel_backend('dask'):
                    logger.info('[{}] : [INFO] Using Dask backend for user defined method'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
                    clf = cluster_method.fit(data)
            except Exception as inst:
                logger.error('[{}] : [ERROR] Failed to fit user defined method with dask backedn with {} and {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
                logger.warning('[{}] : [WARN] using default process based backedn for user defined method'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
                clf = cluster_method.fit(data)
        except Exception as inst:
            logger.error('[{}] : [ERROR] Failed to fit {} with {} and {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(cluster_method),
                type(inst), inst.args))
            sys.exit(1)
        predictions = clf.predict(data)
        logger.debug('[{}] : [DEBUG] Predicted Anomaly Array {}'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), predictions))
        fname = str(clf).split('(')[0]
        self.__serializemodel(clf, fname, mname)
        return clf

    def __appendPredictions(self, method, mname, data, pred):
        fpath = "{}_{}.csv".format(method, mname)
        fname = os.path.join(self.modelDir, fpath)
        logger.info('[{}] : [INFO] Appending predictions to data ... Saving to {}.'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), fname))
        data['ano'] = pred
        data.to_csv(fname, index=True)

    def __serializemodel(self, model,
                         method,
                         mname):
        '''
        :param model: -> model
        :param method: -> method name
        :param mname: -> name to be used for saved model
        :result: -> Serializez current clusterer/classifier
        '''
        fpath = "%s_%s.pkl" % (method, mname)
        fname = os.path.join(self.modelDir, fpath)
        pickle.dump(model, open(fname, "wb"))
        logger.info('[{}] : [INFO] Serializing model {} at {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), method, fpath))

    def __loadClusterModel(self, method,
                           model):
        '''
        :param method: -> method name
        :param model: -> model name
        :return: -> instance of serialized object
        '''
        lmodel = glob.glob(os.path.join(self.modelDir, ("%s_%s.pkl" % (method, model))))
        if not lmodel:
            logger.warning('[%s] : [WARN] No %s model with the name %s found',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), method, model)
            return 0
        else:
            smodel = pickle.load(open(lmodel[0], "rb"))
            logger.info('[%s] : [INFO] Succesfully loaded %s model with the name %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), method, model)
            return smodel