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

import os
import importlib
from edelogger import logger
from datetime import datetime
import time
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_validate, cross_val_score
from sklearn.ensemble import IsolationForest
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.neural_network import MLPClassifier
from sklearn import model_selection
from sklearn.decomposition import PCA
from sklearn.ensemble import AdaBoostClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from sklearn.cluster import DBSCAN
from sklearn.metrics import make_scorer, SCORERS, get_scorer, classification_report, confusion_matrix
from imblearn.metrics import classification_report_imbalanced
from yellowbrick.model_selection import LearningCurve, ValidationCurve, RFECV
from yellowbrick.classifier import PrecisionRecallCurve, ROCAUC
from yellowbrick.contrib.classifier import DecisionViz
from yellowbrick.style import set_palette
import matplotlib.pyplot as plt
import seaborn as sns
import yaml
import joblib
import pickle as pickle
from util import str2Bool
import glob
from util import ut2hum
import shap
import itertools

pd.options.mode.chained_assignment = None
import warnings

warnings.filterwarnings("ignore")

# set color palette yellowbrick
set_palette('sns_deep')


class SciClassification:
    def __init__(self, modelDir,
                 dataDir,
                 checkpoint,
                 export,
                 training,
                 validation,
                 validratio,
                 compare,
                 cv=None,
                 verbose=False,
                 learningcurve=None,
                 validationcurve=None,
                 prc=None,
                 rocauc=None,
                 rfe=None,
                 dboundary=None,
                 pred_analysis=False,
                 trainscore=False,
                 scorers=None,
                 returnestimators=False):
        self.modelDir = modelDir
        self.dataDir = dataDir
        self.checkpoint = checkpoint
        self.export = export
        self.training = training
        self.validation = validation
        self.validratio = validratio
        self.compare = compare
        self.cv = cv
        self.verbose = verbose
        self.learningcurve = learningcurve
        self.validationcurve = validationcurve
        self.prc = prc
        self.rocauc = rocauc
        self.rfe = rfe
        self.dboundary = dboundary
        self.pred_analysis = pred_analysis
        self.trainscore = trainscore
        self.scorers = scorers
        self.returnestimators = returnestimators
        self.skscorer = 'sklearn.metrics'
        self.sksplitgen = 'sklearn.model_selection'

    def detect(self, method, model, data):
        smodel = self.__loadClassificationModel(method, model)
        anomalieslist = []
        if not smodel:
            dpredict = 0
        else:
            if data.shape[0]:
                if isinstance(smodel, RandomForestClassifier):
                    print("Detected RandomForest model")
                    print("n_estimators -> %s" % smodel.n_estimators)
                    print("Criterion -> %s" % smodel.criterion)
                    print("Max_Features -> %s" % smodel.max_features)
                    print("Max_Depth -> %s" % smodel.max_depth)
                    print("Min_sample_split -> %s " % smodel.min_samples_split)
                    print("Min_sample_leaf -> %s " % smodel.min_samples_leaf)
                    print("Min_weight_fraction_leaf -> %s " % smodel.min_weight_fraction_leaf)
                    print("Max_leaf_nodes -> %s " % smodel.max_leaf_nodes)
                    print("Min_impurity_split -> %s " % smodel.min_impurity_split)
                    print("Bootstrap -> %s " % smodel.bootstrap)
                    print("Oob_score -> %s " % smodel.oob_score)
                    print("N_jobs -> %s " % smodel.n_jobs)
                    print("Random_state -> %s " % smodel.random_state)
                    print("Verbose -> %s " % smodel.verbose)
                    print("Class_weight -> %s " % smodel.class_weight)
                    try:
                        dpredict = smodel.predict(data)
                        print("RandomForest Prediction Array -> %s" % str(dpredict))
                    except Exception as inst:
                        logger.error('[%s] : [ERROR] Error while fitting randomforest model to event with %s and %s',
                                     datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst),
                                     inst.args)
                        dpredict = 0
                elif isinstance(smodel, AdaBoostClassifier):
                    print("Detected AdaBoost model")
                    print("base_estimator -> %s" % smodel.base_estimator)
                    print("n_estimators -> %s" % smodel.n_estimators)
                    print("Learning_rate -> %s" % smodel.learning_rate)
                    print("Algorithm -> %s" % smodel.algorithm)
                    print("Random State -> %s" % smodel.random_state)
                    try:
                        dpredict = smodel.predict(self.df)
                        print("AdaBoost Prediction Array -> %s" % str(dpredict))
                    except Exception as inst:
                        logger.error('[%s] : [ERROR] Error while fitting AdaBoost model to event with %s and %s',
                                     datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst),
                                     inst.args)
                        sys.exit(1)
                elif isinstance(smodel, DecisionTreeClassifier):
                    print("Detected Decision Tree model")
                    print("Criterion -> %s" % smodel.criterion)
                    print("Spliter -> %s" % smodel.splitter)
                    print("Max_Depth -> %s" % smodel.max_depth)
                    print("Min_sample_split -> %s " % smodel.min_samples_split)
                    print("Min_sample_leaf -> %s " % smodel.min_samples_leaf)
                    print("Min_weight_fraction_leaf -> %s " % smodel.min_weight_fraction_leaf)
                    print("Max_Features -> %s" % smodel.max_features)
                    print("Random_state -> %s " % smodel.random_state)
                    print("Max_leaf_nodes -> %s " % smodel.max_leaf_nodes)
                    print("Min_impurity_split -> %s " % smodel.min_impurity_split)
                    print("Class_weight -> %s " % smodel.class_weight)
                    try:
                        dpredict = smodel.predict(self.df)
                        print("Decision Tree Prediction Array -> %s" % str(dpredict))
                    except Exception as inst:
                        logger.error('[%s] : [ERROR] Error while fitting Decision Tree model to event with %s and %s',
                                     datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst),
                                     inst.args)
                        sys.exit(1)

                elif isinstance(smodel, MLPClassifier):
                    print("Detected Neural Network model")
                    print("Hidden Layer size -> %s" % str(smodel.hidden_layer_sizes))
                    print("Activation -> %s" % smodel.activation)
                    print("Solver -> %s" % smodel.solver)
                    print("Alpha -> %s" % smodel.alpha)
                    print("Batch Size -> %s" % smodel.batch_size)
                    print("Learning rate -> %s" % smodel.learning_rate)
                    print("Max Iterations -> %s" % smodel.max_iter)
                    print("Shuffle -> %s" % smodel.shuffle)
                    print("Momentum -> %s" % smodel.momentum)
                    print("Epsilon -> %s" % smodel.epsilon)
                    try:
                        dpredict = smodel.predict(self.df)
                        print("MLP Prediction Array -> %s" % str(dpredict))
                    except Exception as inst:
                        logger.error('[%s] : [ERROR] Error while fitting MLP model to event with %s and %s',
                                     datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst),
                                     inst.args)
                        sys.exit(1)
                else:
                    logger.error('[%s] : [ERROR] Unsuported model loaded: %s!',
                                 datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(smodel))
                    sys.exit(1)
            else:
                dpredict = 0
                logger.warning('[%s] : [WARN] Dataframe empty with shape (%s,%s)',
                               datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(data.shape[0]),
                               str(data.shape[1]))
                print("Empty dataframe received with shape (%s,%s)" % (str(data.shape[0]),
                                                                       str(data.shape[1])))
            print("dpredict type is %s" % (type(dpredict)))
            if type(dpredict) is not int:
                data['AType'] = dpredict
                for index, row in data.iterrows():
                    anomalies = {}
                    if row['AType'] != 0:
                        print(index)
                        print(data.get_value(index, 'AType'))
                        anomalies['utc'] = int(index)
                        anomalies['hutc'] = ut2hum(int(index))
                        anomalies['anomaly_type'] = data.get_value(index, 'AType')
                        anomalieslist.append(anomalies)
        anomaliesDict = {}
        anomaliesDict['anomalies'] = anomalieslist
        logger.info('[%s] : [INFO] Detected anomalies with model %s using method %s are -> %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), model, method,
                    str(anomaliesDict))
        return anomaliesDict

    def dask_detect(self, method,
                    model,
                    data,
                    normal_label=None):
        smodel = self.__loadClassificationModel(method=method, model=model)
        anomaliesList = []
        nl = 0
        explainer = 0
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
        anomaliesDict = {}
        if type(dpredict) is not int:
            if normal_label is None:  # Todo make normal_label  user definable
                nl = 0
            else:
                nl = normal_label
            anomalyArray = np.argwhere(dpredict != nl)  # Pandas bug where np.argwhere not working on dataframes
            if self.pred_analysis and anomalyArray.shape[0]:
                df_anomaly_data = data.copy(deep=True)  # copy for second filter using pandas
                df_anomaly_data['target'] = dpredict
                anomaliesDict['complete_shap_analysis'], explainer, shap_values = self.__shap_analysis(smodel,
                                                                                                       df_anomaly_data,
                                                                                                       normal_value=nl)
            count = 0 # todo merge filtering of dpred detection; related to np.argwhere bug for pandas and __shap_analysis data refiltering
            for an in anomalyArray:
                anomalies = {}
                anomalies['utc'] = int(data.iloc[an[0]].name)
                anomalies['hutc'] = ut2hum(int(data.iloc[an[0]].name))
                anomalies['type'] = dpredict[an[0]]
                if explainer:
                    anomalies['analysis'] = self.__shap_values_processing(explainer=explainer,
                                                                          shap_values=shap_values,
                                                                          label=dpredict[an[0]],
                                                                          feature_names=data.columns,
                                                                          instance=count)
                anomaliesList.append(anomalies)
                count += 1
        anomaliesDict['anomalies'] = anomaliesList

        logger.info('[{}] : [INFO] Detected {} anomalies with model {} using method {} '.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), len(anomaliesList), model,
            str(smodel).split('(')[0]))
        return anomaliesDict

    def __shap_analysis(self,
                        model,
                        data,
                        normal_value):
        # todo use non tokenized labels for data
        """
        Shap analysis of incoming data

        :param model: Predictive model to be analyzed
        :param data: data for analysis
        :param normal_value: denotes the normal (majority) class label
        :return: feature importance
        """
        logger.info('[%s] : [INFO] Executing classification prediction analysis ...',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))

        data_filtered = data.loc[data['target'] != normal_value]
        data_filtered.drop(['target'], inplace=True, axis=1)

        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data_filtered)

        try:
            labels = model.classes_
        except Exception as inst:
            logger.error('[%s] : [ERROR] Prediction analysis failed with {} and {}',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args)
            return 0

        feature_imp = self.__shap_feature_importance(shap_values=shap_values,
                                                     data=data,
                                                     label=labels)
        return feature_imp, explainer, shap_values

    def __shap_values_processing(self,
                                 explainer,
                                 shap_values,
                                 feature_names,
                                 label,
                                 instance):
        """
        Used to export data as used by shap.force_plot on a per detection basis

        :param explainer: Shape explainer object
        :param shap_values: shapely values
        :param feature_names: name of features from dataset
        :param label: label after detection
        :param instance: instance number as used in df.iloc
        :return: shap_values on a per detection instance basis
        """
        shap_values_d = {}
        try:
            shap_values_d['shap_values'] = dict(zip(feature_names, shap_values[label][instance]))
            shap_values_d['expected_value'] = explainer.expected_value[label]
        except Exception as inst:
            logger.error('[{}] : [ERROR] Error while executing shap processing with {} and {} '.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
        return shap_values_d

    def __shap_feature_importance(self,
                                shap_values,
                                data,
                                label):
        """
        Extracts feature importance from shapely values.

        :param shap_values: shapely values
        :param data: dataframe to be analysied
        :param label: labels to be used, as extracted from predictive model
        :return: dictionary containing feature importance for each label separately
        """
        feature_importance_d = {}
        if isinstance(label, list) or isinstance(label, np.ndarray):
            for l in label:
                feature_importance = pd.DataFrame(list(zip(data.columns.tolist(), shap_values[l].sum(0))),
                                                  columns=['feature_name', 'feature_importance_vals'])
                feature_importance = feature_importance.iloc[
                    (-np.abs(feature_importance['feature_importance_vals'].values)).argsort()]
                feature_importance_d[l] = feature_importance.to_dict()
        else:
            feature_importance = pd.DataFrame(list(zip(data.columns.tolist(), shap_values[label].sum(0))),
                                              columns=['feature_name', 'feature_importance_vals'])
            feature_importance = feature_importance.iloc[
                (-np.abs(feature_importance['feature_importance_vals'].values)).argsort()]
            feature_importance_d[label] = feature_importance.to_dict()
        return feature_importance_d

    def score(self, model, X, y):
        return model.score(X, y)

    def compare(self, modelList, X, y):
        scores = []
        for model in modelList:
            scores.append(model.score(X, y))
        logger.info('[%s] : [INFO] Best performing model score is -> %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), max(scores))
        # for a, b in itertools.combinations(modelList, 2):
        #     a.score(X, y)
        #     b.score(X, y)
        return modelList.index(max(scores))

    def crossvalid(self, model, X, y, kfold):
        return model_selection.cross_val_score(model, X, y, cv=kfold)

    def naiveBayes(self):
        return True

    def adaBoost(self, settings,
                 data=None,
                 dropna=True):
        if "n_estimators" not in settings:
            print("Received settings for Ada Boost are %s invalid!" % str(settings))
            logger.error('[%s] : [ERROR] Received settings for Decision Tree %s are invalid',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(settings))
            sys.exit(1)
        dtallowedSettings = ["n_estimators", "learning_rate"]
        for k, v in settings.items():
            if k in dtallowedSettings:
                logger.info('[%s] : [INFO] Ada Boost %s set to %s',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), k, v)
                print("Ada Boost %s set to %s" % (k, v))

        if not isinstance(self.export, str):
            mname = 'default'
        else:
            mname = self.export
        df = self.__loadData(data, dropna)
        features = df.columns[:-1]
        X = df[features]
        y = df.iloc[:, -1].values
        seed = 7
        # num_trees = 500
        kfold = model_selection.KFold(n_splits=10, random_state=seed)
        print(kfold)
        ad = AdaBoostClassifier(n_estimators=settings['n_estimators'], learning_rate=settings['learning_rate'],
                                random_state=seed)
        if self.validratio:
            trainSize = 1.0 - self.validratio
            print("Decision Tree training to validation ratio set to: %s" % str(self.validratio))
            logger.info('[%s] : [INFO] Ada Boost training to validation ratio set to: %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(self.validratio))
            d_train, d_test, f_train, f_test = self.__dataSplit(X, y, testSize=self.validratio, trainSize=trainSize)
            ad.fit(d_train, f_train)
            predict = ad.predict(d_train)
            print("Prediction for Ada Boost Training:")
            print(predict)

            print("Actual labels of training set:")
            print(f_train)

            predProb = ad.predict_proba(d_train)
            print("Prediction probabilities for Ada Boost Training:")
            print(predProb)

            score = ad.score(d_train, f_train)
            print("Ada Boost Training Score: %s" % str(score))
            logger.info('[%s] : [INFO] Ada Boost training score: %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(score))

            feature_imp = list(zip(d_train, ad.feature_importances_))
            print("Feature importance Ada Boost Training: ")
            print(list(zip(d_train, ad.feature_importances_)))
            logger.info('[%s] : [INFO] Ada Boost feature importance: %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(feature_imp))

            pred_valid = ad.predict(d_test)
            print("Ada Boost Validation set prediction: ")
            print(pred_valid)
            print("Actual values of validation set: ")
            print(d_test)
            score_valid = ad.score(d_test, f_test)
            print("Ada Boost validation set score: %s" % str(score_valid))
            logger.info('[%s] : [INFO] Ada Boost validation score: %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(score_valid))
        else:
            ad.fit(X, y)
            predict = ad.predict(X)
            print("Prediction for Ada Boost Training:")
            print(predict)

            print("Actual labels of training set:")
            print(y)

            predProb = ad.predict_proba(X)
            print("Prediction probabilities for Ada Boost Training:")
            print(predProb)

            score = ad.score(X, y)
            print("Ada Boost Training Score: %s" % str(score))

            fimp = list(zip(X, ad.feature_importances_))
            print("Feature importance Ada Boost Training: ")
            print(fimp)
            dfimp = dict(fimp)
            dfimp = pd.DataFrame(list(dfimp.items()), columns=['Metric', 'Importance'])
            sdfimp = dfimp.sort('Importance', ascending=False)
            dfimpCsv = 'Feature_Importance_%s.csv' % mname
            sdfimp.to_csv(os.path.join(self.modelDir, dfimpCsv))
            if self.validation is None:
                logger.info('[%s] : [INFO] Validation is set to None',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                # return True
            else:
                vfile = os.path.join(self.dataDir, self.validation)
                logger.info('[%s] : [INFO] Validation data file is set to %s',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(vfile))
                if not os.path.isfile(vfile):
                    print("Validation file %s not found" % vfile)
                    logger.error('[%s] : [ERROR] Validation file %s not found',
                                 datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(vfile))
                else:
                    df_valid = pd.read_csv(vfile)
                    if dropna:
                        df_valid = df_valid.dropna()
                    features_valid = df_valid.columns[:-1]
                    X_valid = df_valid[features_valid]
                    y_valid = df_valid.iloc[:, -1].values
                    pred_valid = ad.predict(X_valid)
                    print("Ada Boost Validation set prediction: ")
                    print(pred_valid)
                    print("Actual values of validation set: ")
                    print(y_valid)
                    score_valid = ad.score(X_valid, y_valid)
                    print("Ada Boost set score: %s" % str(score_valid))
                    # return True
        self.__serializemodel(ad, 'DecisionTree', mname)
        return ad

    def neuralNet(self, settings,
                  data=None,
                  dropna=True):
        if "activation" not in settings:
            print("Received settings for Neural Networks are %s invalid!" % str(settings))
            logger.error('[%s] : [ERROR] Received settings for Neural Networks %s are invalid',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(settings))
            sys.exit(1)

        rfallowedSettings = ["max_iter", "activation", "solver", "batch_size", "learning_rate",
                             "momentum", "alpha"]

        for k, v in settings.items():
            if k in rfallowedSettings:
                logger.info('[%s] : [INFO] Neural Network %s set to %s',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), k, v)
                print("Neural Network %s set to %s" % (k, v))

        if not isinstance(self.export, str):
            mname = 'default'
        else:
            mname = self.export

        df = self.__loadData(data, dropna)
        features = df.columns[:-1]
        X = df[features]
        y = df.iloc[:, -1].values

        mlp = MLPClassifier(hidden_layer_sizes=(50, 20), max_iter=settings['max_iter'],
                            activation=settings['activation'],
                            solver=settings['solver'], batch_size=settings['batch_size'],
                            learning_rate=settings['learning_rate'], momentum=settings['momentum'],
                            alpha=settings['alpha'])

        if self.validratio:
            trainSize = 1.0 - self.validratio
            print("Neural Network training to validation ratio set to: %s" % str(self.validratio))
            logger.info('[%s] : [INFO] Neural Netowork training to validation ratio set to: %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(self.validratio))
            d_train, d_test, f_train, f_test = self.__dataSplit(X, y, testSize=self.validratio, trainSize=trainSize)
            mlp.fit(d_train, f_train)
            predict = mlp.predict(d_train)
            print("Prediction for Neural Network Training:")
            print(predict)

            print("Actual labels of training set:")
            print(f_train)

            predProb = mlp.predict_proba(d_train)
            print("Prediction probabilities for Neural Network Training:")
            print(predProb)

            score = mlp.score(d_train, f_train)
            print("Neural Network Training Score: %s" % str(score))
            logger.info('[%s] : [INFO] Neural Network training score: %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(score))
            pred_valid = mlp.predict(d_test)
            print("Neural Network Validation set prediction: ")
            print(pred_valid)
            print("Actual values of validation set: ")
            print(d_test)
            score_valid = mlp.score(d_test, f_test)
            print("Neural Network validation set score: %s" % str(score_valid))
            logger.info('[%s] : [INFO] Random forest validation score: %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(score_valid))
        else:
            mlp.fit(X, y)
            predict = mlp.predict(X)
            print("Prediction for Neural Network Training:")
            print(predict)

            print("Actual labels of training set:")
            print(y)

            predProb = mlp.predict_proba(X)
            print("Prediction probabilities for Neural Network Training:")
            print(predProb)

            score = mlp.score(X, y)
            print("Random Forest Training Score: %s" % str(score))

            if self.validation is None:
                logger.info('[%s] : [INFO] Validation is set to None',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                # return True
            else:
                vfile = os.path.join(self.dataDir, settings['validation'])
                logger.info('[%s] : [INFO] Validation data file is set to %s',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(vfile))
                if not os.path.isfile(vfile):
                    print("Validation file %s not found" % vfile)
                    logger.error('[%s] : [ERROR] Validation file %s not found',
                                 datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(vfile))
                else:
                    df_valid = pd.read_csv(vfile)
                    if dropna:
                        df_valid = df_valid.dropna()
                    features_valid = df_valid.columns[:-1]
                    X_valid = df_valid[features_valid]
                    y_valid = df_valid.iloc[:, -1].values
                    pred_valid = mlp.predict(X_valid)
                    print("Neural Network Validation set prediction: ")
                    print(pred_valid)
                    print("Actual values of validation set: ")
                    print(y_valid)
                    score_valid = mlp.score(X_valid, y_valid)
                    print("Neural Network validation set score: %s" % str(score_valid))
                    # return True
        self.__serializemodel(mlp, 'RandomForest', mname)
        return mlp

    def decisionTree(self,
                     settings,
                     data=None,
                     dropna=True):
        if "splitter" not in settings:
            print("Received settings for Decision Tree are %s invalid!" % str(settings))
            logger.error('[%s] : [ERROR] Received settings for Decision Tree %s are invalid',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(settings))
            sys.exit(1)

        if settings['random_state'] == 'None':
            settings['random_state'] = None
        else:
            settings['random_state'] = int(settings['random_state'])

        if settings['max_depth'] == 'None':
            max_depth = None
        else:
            max_depth = int(settings['max_depth'])

        if settings['max_features'] == 'auto':
            max_features = settings['max_features']
        else:
            max_features = int(settings['max_features'])

        dtallowedSettings = ["criterion", "splitter", "max_features", "max_depth",
                             "min_weight_faction_leaf", "random_state"]
        for k, v in settings.items():
            if k in dtallowedSettings:
                logger.info('[%s] : [INFO] DecisionTree %s set to %s',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), k, v)
                print("DecisionTree %s set to %s" % (k, v))

        if not isinstance(self.export, str):
            mname = 'default'
        else:
            mname = self.export

        df = self.__loadData(data, dropna)
        features = df.columns[:-1]
        X = df[features]
        y = df.iloc[:, -1].values

        # dt = DecisionTreeClassifier(min_samples_split=20, random_state=99)
        dt = DecisionTreeClassifier(criterion=settings["criterion"], splitter=settings["splitter"],
                                    max_features=max_features, max_depth=max_depth,
                                    min_samples_split=float(settings["min_sample_split"]),
                                    min_weight_fraction_leaf=float(settings["min_weight_faction_leaf"]),
                                    random_state=settings["random_state"])
        if self.validratio:
            trainSize = 1.0 - self.validratio
            print("Decision Tree training to validation ratio set to: %s" % str(self.validratio))
            logger.info('[%s] : [INFO] Random forest training to validation ratio set to: %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(self.validratio))
            d_train, d_test, f_train, f_test = self.__dataSplit(X, y, testSize=self.validratio, trainSize=trainSize)
            dt.fit(d_train, f_train)
            predict = dt.predict(d_train)
            print("Prediction for Decision Tree Training:")
            print(predict)

            print("Actual labels of training set:")
            print(f_train)

            predProb = dt.predict_proba(d_train)
            print("Prediction probabilities for Decision Tree Training:")
            print(predProb)

            score = dt.score(d_train, f_train)
            print("Decision Tree Training Score: %s" % str(score))
            logger.info('[%s] : [INFO] Decision Tree training score: %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(score))

            feature_imp = list(zip(d_train, dt.feature_importances_))
            print("Feature importance Decision Tree Training: ")
            print(list(zip(d_train, dt.feature_importances_)))
            logger.info('[%s] : [INFO] Decision Tree feature importance: %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(feature_imp))

            pred_valid = dt.predict(d_test)
            print("Decision Tree Validation set prediction: ")
            print(pred_valid)
            print("Actual values of validation set: ")
            print(d_test)
            score_valid = dt.score(d_test, f_test)
            print("Decision Tree validation set score: %s" % str(score_valid))
            logger.info('[%s] : [INFO] Random forest validation score: %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(score_valid))
        else:
            dt.fit(X, y)
            predict = dt.predict(X)
            print("Prediction for Decision Tree Training:")
            print(predict)

            print("Actual labels of training set:")
            print(y)

            predProb = dt.predict_proba(X)
            print("Prediction probabilities for Decision Tree Training:")
            print(predProb)

            score = dt.score(X, y)
            print("Decision Tree Training Score: %s" % str(score))

            fimp = list(zip(X, dt.feature_importances_))
            print("Feature importance Random Forest Training: ")
            print(fimp)
            dfimp = dict(fimp)
            dfimp = pd.DataFrame(list(dfimp.items()), columns=['Metric', 'Importance'])
            sdfimp = dfimp.sort('Importance', ascending=False)
            dfimpCsv = 'Feature_Importance_%s.csv' % mname
            sdfimp.to_csv(os.path.join(self.modelDir, dfimpCsv))
            if self.validation is None:
                logger.info('[%s] : [INFO] Validation is set to None',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                # return True
            else:
                vfile = os.path.join(self.dataDir, self.validation)
                logger.info('[%s] : [INFO] Validation data file is set to %s',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(vfile))
                if not os.path.isfile(vfile):
                    print("Validation file %s not found" % vfile)
                    logger.error('[%s] : [ERROR] Validation file %s not found',
                                 datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(vfile))
                else:
                    df_valid = pd.read_csv(vfile)
                    if dropna:
                        df_valid = df_valid.dropna()
                    features_valid = df_valid.columns[:-1]
                    X_valid = df_valid[features_valid]
                    y_valid = df_valid.iloc[:, -1].values
                    pred_valid = dt.predict(X_valid)
                    print("Decision Tree Validation set prediction: ")
                    print(pred_valid)
                    print("Actual values of validation set: ")
                    print(y_valid)
                    score_valid = dt.score(X_valid, y_valid)
                    print("Random Decision Tree set score: %s" % str(score_valid))
                    # return True
        self.__serializemodel(dt, 'DecisionTree', mname)
        return dt

    def dask_tpot(self,
                  settings,
                  X,
                  y):
        from tpot import TPOTClassifier
        if self.cv is None:
            cv = self.cv
            logger.info('[{}] : [INFO] TPOT Cross Validation not set, using default'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
        elif isinstance(self.cv, int):
            logger.info('[{}] : [INFO] TPOT Cross Validation set to {} folds:'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), self.cv))
            cv = self.cv
        else:
            try:
                logger.info('[{}] : [INFO] TPOT Cross Validation set to use {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), self.cv['Type']))
            except:
                logger.error('[{}] : [ERROR] TPOT Cross Validation split generator type not set!'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
                sys.exit(1)
            cv = self.__crossValidGenerator(self.cv)
        settings.update({'cv': cv})
        tp = TPOTClassifier(**settings)
        for k, v in settings.items():
            logger.info('[{}] : [INFO] TPOT parame {} set to {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), k, v))
        logger.info('[{}] : [INFO] Starting TPOT Optimization ...'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
        try:
            pipeline_model = tp.fit(X, y)
        except Exception as inst:
            logger.error('[{}] : [ERROR] Failed to run TPOT optimization with {} and {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
            sys.exit(1)
        # print(pipeline_model.score(X, y))
        logger.info('[{}] : [INFO] TPOT optimized best pipeline is: {}'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
            str(pipeline_model.fitted_pipeline_.steps)))
        # print(str(pipeline_model.fitted_pipeline_.steps))
        # print(pipeline_model.pareto_front_fitted_pipelines_)
        # print(pipeline_model.evaluated_individuals_)
        self.__serializemodel(model=pipeline_model.fitted_pipeline_, method='TPOT', mname=self.export)

        return 0

    def dask_classifier(self, settings,
                        mname,
                        X,
                        y, classification_method=None):
        # Factorize input
        y_factor = pd.factorize(y)
        y = y_factor[0]
        y_definitions = y_factor[1]

        # y = y.astype(int) # fix y being set as object
        user_m = False
        if classification_method is not None and classification_method != 'randomforest':  # TODO fix
            user_m = True
            try:
                classification_type = str(classification_method).split('(')[0]
            except Exception:
                classification_type = type(classification_method)

            logger.info('[{}] : [INFO] Classification Method set to {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type))
            for k, v in classification_method.get_params().items():
                logger.info('[{}] : [INFO] Classification parameter {} set to {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), k, v))
            # print(classification_type)
            # sys.exit()
        else:
            classification_type = 'RandomForest'
            logger.info('[{}] : [INFO] Method set to {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type))
            if not settings or settings is None:
                logger.warning('[{}] : [WARN] No {} parameters defined using defaults'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type))
                settings = {}
            else:
                for k, v in settings.items():
                    logger.info('[{}] : [INFO] {} parameter {} set to {}'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type, k, v))
            try:
                clf = RandomForestClassifier(**settings)
            except Exception as inst:
                logger.error('[{}] : [INFO] Failed to instanciate {} with {} and {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type, type(inst),
                    inst.args))
                sys.exit(1)

        if self.cv is None:
            trainSize = 1.0 - self.validratio
            logger.info('[{}] : [INFO] {} training to validation ratio set to: {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type,
                str(self.validratio)))
            if self.validation:
                d_train, d_test, f_train, f_test = self.__dataSplit(X, y, testSize=self.validratio, trainSize=trainSize)
                try:
                    with joblib.parallel_backend('dask'):
                        logger.info('[{}] : [INFO] Using Dask backend for {}'.format(
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type))
                        if user_m:
                            classification_method.fit(d_train, f_train)
                        else:
                            clf.fit(d_train, f_train)
                except Exception as inst:
                    logger.error('[{}] : [ERROR] Failed to fit {} with {} and {}'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type,
                        type(inst), inst.args))
                    sys.exit(1)
                logger.info('[{}] : [INFO] Running Prediction on training data ...'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
                if user_m:
                    predict = classification_method.predict(d_train)
                    logger.info('[{}] : [INFO] Calculating predict probabilities ...'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
                    predict_proba = classification_method.predict_proba(d_train)
                    score = classification_method.score(d_train, f_train)
                    logger.info('[{}] : [INFO] Score on training set is {}'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), score))
                else:
                    predict = clf.predict(d_train)
                    logger.info('[{}] : [INFO] Calculating predict probabilities ...'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
                    predict_proba = clf.predict_proba(d_train)
                    score = clf.score(d_train, f_train)
                    logger.info('[{}] : [INFO] Score on training set is {}'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), score))

                try:
                    feature_imp = list(zip(d_train, clf.feature_importances_))
                    logger.info('[{}] : [INFO] Exporting Feature Importance ...'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), score))
                    df_fimp = pd.DataFrame(feature_imp, columns=["Feature_Name", "Importance"])
                    df_fimp.to_csv(
                        os.path.join(self.modelDir, "{}_{}_Feature_Importance.csv".format(classification_type,
                                                                                          self.export)), index=False)
                except:
                    logger.warning('[{}] : [WARN] Failed to export feature importance, not available for {}'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type))
                logger.info('[{}] : [INFO] Predicting on validation set ...'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), score))

                if user_m:
                    pred_valid = classification_method.predict(d_test)
                    predict_val_proba = classification_method.predict_proba(d_test)
                    score_valid = classification_method.score(d_test, f_test)
                    logger.info('[{}] : [INFO] Score on validation set is {}'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), score_valid))
                    logger.info('[{}] : [INFO] Exporting Training data with predictions ...'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
                    self.__appendPredictions("{}_train".format(classification_type), self.export, d_train, predict,
                                             predict_proba)
                    logger.info('[{}] : [INFO] Exporting Validation data with predictions ...'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
                    self.__appendPredictions("{}_valid".format(classification_type), self.export, d_test, pred_valid,
                                             predict_val_proba)
                else:
                    pred_valid = clf.predict(d_test)
                    predict_val_proba = clf.predict_proba(d_test)
                    score_valid = clf.score(d_test, f_test)
                    logger.info('[{}] : [INFO] Score on validation set is {}'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), score_valid))
                    logger.info('[{}] : [INFO] Exporting Training data with predictions ...'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
                    self.__appendPredictions("{}_train".format(classification_type), self.export, d_train, predict,
                                             predict_proba)
                    logger.info('[{}] : [INFO] Exporting Validation data with predictions ...'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
                    self.__appendPredictions("{}_valid".format(classification_type), self.export, d_test, pred_valid,
                                             predict_val_proba)
            else:
                try:
                    with joblib.parallel_backend('dask'):
                        logger.info('[{}] : [INFO] Using Dask backend for {}'.format(
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type))
                        if user_m:
                            classification_method.fit(X, y)
                        else:
                            clf.fit(X, y)
                except Exception as inst:
                    logger.error('[{}] : [ERROR] Failed to fit {} with {} and {}'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type,
                        type(inst), inst.args))
                    sys.exit(1)
                logger.info('[{}] : [INFO] Running Prediction on training data ...'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
                if user_m:
                    predict = classification_method.predict(X)
                    logger.info('[{}] : [INFO] Calculating predict probabilities ...'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
                    predict_proba = classification_method.predict_proba(X)
                    score = classification_method.score(X, y)
                    logger.info('[{}] : [INFO] Score on training set is {}'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), score))
                else:
                    predict = clf.predict(X)
                    logger.info('[{}] : [INFO] Calculating predict probabilities ...'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
                    predict_proba = clf.predict_proba(X)
                    score = clf.score(X, y)
                    logger.info('[{}] : [INFO] Score on training set is {}'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), score))
                try:
                    feature_imp = list(zip(X, clf.feature_importances_))
                    logger.info('[{}] : [INFO] Exporting Feature Importance ...'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), score))
                    df_fimp = pd.DataFrame(feature_imp, columns=["Feature_Name", "Importance"])
                    df_fimp.to_csv(
                        os.path.join(self.modelDir, "{}_{}_Feature_Importance.csv".format(classification_type,
                                                                                          self.export)), index=False)
                except:
                    logger.warning('[{}] : [WARN] Failed to export feature importance, not available for {}'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type))
                logger.info('[{}] : [INFO] Exporting Training data with predictions ...'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
                self.__appendPredictions("{}_train".format(classification_type), self.export, X, predict, predict_proba)
                self.__serializemodel(clf, classification_type, mname)
        else:
            if isinstance(self.cv, int):
                logger.info('[{}] : [INFO] {} Cross Validation set to {} folds:'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type, self.cv))
                cv = self.cv
            else:
                try:
                    logger.info('[{}] : [INFO] {} Cross Validation set to use {}'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type,
                        self.cv['Type']))
                except:
                    logger.error('[{}] : [ERROR] Cross Validation split generator type not set!'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
                    sys.exit(1)
                cv = self.__crossValidGenerator(self.cv)
            if self.scorers is None:
                scorer = None
            else:
                scorer = self.__dask_classification_scorer(self.scorers)
            try:
                with joblib.parallel_backend('dask'):
                    logger.info('[{}] : [INFO] Using Dask backend for CV of {}'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type))
                    if user_m:
                        if self.verbose:

                            cv_results = self.__ede_cross_validate(classification_method, X, y, scoring=scorer,
                                                                   return_estimator=self.returnestimators, cv=cv,
                                                                   definitions=y_definitions,
                                                                   model_name=classification_type)

                        else:
                            cv_results = cross_validate(classification_method, X, y, scoring=scorer,
                                                        return_train_score=self.trainscore,
                                                        return_estimator=self.returnestimators, cv=cv,
                                                        error_score='raise')
                    else:
                        if self.verbose:
                            cv_results = self.__ede_cross_validate(clf, X, y, scoring=scorer,
                                                                   return_estimator=self.returnestimators, cv=cv,
                                                                   definitions=y_definitions,
                                                                   model_name=classification_type)
                        else:
                            cv_results = cross_validate(clf, X, y, scoring=scorer, return_train_score=self.trainscore,
                                                        # todo check clf, method definitions in yaml string only
                                                        return_estimator=self.returnestimators, cv=cv)
            except Exception as inst:
                logger.error('[{}] : [ERROR] Failed to fit {} during Cross Validation with {} and {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type, type(inst),
                    inst.args))
                sys.exit(1)

            if self.returnestimators:
                cv_estimators = cv_results['estimator']
                del cv_results['estimator']
                i = 0
                for est in cv_estimators:
                    logger.info('[{}] : [INFO] Saving CV Model {} ...'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), i))
                    mname = "{}_{}".format(self.export, i)
                    self.__serializemodel(est, classification_type, mname)
                    i += 1
            logger.info('[{}] : [INFO] Saving CV Metrics ...'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
            cv_res = pd.DataFrame(cv_results)
            if isinstance(self.cv, dict):
                cv_name = self.cv['Type']
            elif self.cv is None:
                cv_name = 5  # Default value for random and Grid
            else:
                cv_name = cv
            cv_res_loc = os.path.join(self.modelDir, "{}_CV_{}_restults.csv".format(classification_type, cv_name))
            cv_res.to_csv(cv_res_loc, index=False)

            if not user_m:  # todo: better solution for switching between user defined method and ede predefined
                classification_method = clf

            # Learning Curve
            if self.learningcurve is not None:
                logger.info('[{}] : [INFO] Computing Learning Curve for {} of type {} ...'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), self.export,
                    classification_type))
                self.__learning_curve(classification_method, X, y, cv, model_name=classification_type)

            # Validation Curve
            if self.validationcurve is not None:
                logger.info('[{}] : [INFO] Computing Validation Curve for {} of type {} ...'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), self.export,
                    classification_type))
                self.__validation_curve(classification_method, X, y, cv, model_name=classification_type)

            # Recurrent feature elimination
            try:
                if self.rfe is not None:
                    logger.info('[{}] : [INFO] Computing Recursive Feature Elimination for {} of type {} ...'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), self.export,
                        classification_type))
                    rfe_model = self.__rfe(classification_method, X, y, cv=cv, model_name=classification_type)
                    mname = f"RFE_{self.export}"
                    self.__serializemodel(rfe_model, classification_type, mname)
            except Exception as inst:
                logger.warning('[{}] : [WARN] Recursive Feature Elimination is not executable on {} with {} and {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                    classification_type, type(inst), inst.args))

            # Precision-Recall Curve
            if self.prc is not None:
                logger.info('[{}] : [INFO] Computing Precision Recall Curve for {} of type {} ...'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), self.export,
                    classification_type))
                self.__precision_recall_curve(classification_method, X, y, definitions=y_definitions,
                                              model_name=classification_type)
            # ROC-AUC
            if self.rocauc is not None:
                logger.info('[{}] : [INFO] Computing ROC Curves for {} of type {} ...'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), self.export,
                    classification_type))
                self.__rocauc_curves(classification_method, X, y, definitions=y_definitions,
                                     model_name=classification_type)

            # DecisionBoundaries Vizualizer
            if self.dboundary is not None:
                logger.info('[{}] : [INFO] Computing Decision Boundary for {} of type {} ...'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), self.export,
                    classification_type))
                self.__decision_boundary(classification_method, X, y, definitions=y_definitions,
                                         model_name=classification_type )

    def __decision_boundary(self,
                            model,
                            X,
                            y,
                            definitions,
                            model_name):

        """
        Decision Boundary is a bivariate data visualization algorithm which plots the boundaries between
        each class. It operates on a limited set of features. Complex boundaries are to be expected
        when dealing with initially high dimensional data and/or large number of possible classes.

        :param model: model instance created from conf yaml parameters
        :param X: training dataframe
        :param y: ground truth from dataframe
        :param definitions: Class definitions as defined by tokenization
        :param model_name: name of the model set by export
        """
        pca_transformer = PCA(n_components=2)
        X_tranformed = pca_transformer.fit_transform(X)

        # Split data into training and testing  # todo add customizability
        XTrain, XTest, yTrain, yTest = train_test_split(X_tranformed, y, test_size=.33, shuffle=True, random_state=42)

        viz = DecisionViz(
            model,
            title=f"Decision Boundary {self.export} {model_name}",
            features=['PC1', 'PC2'], classes=definitions
        )

        viz.fit(XTrain, yTrain)
        viz.draw(XTest, yTest)
        db_fig = f"Decision Boundary_{self.export}_{model_name}.png"
        viz.show(outpath=os.path.join(self.modelDir, db_fig))
        plt.close()

    def __rfe(self,
              model,
              X,
              y,
              cv,
              model_name):
        """
        Recursive Feature elimination will fit the user defined model and removes the weakest feature
        until the an optimum combination of features is reached. Features are ranked by coef_ or feature_importance_
        attribute. Thus methods does not work with methods without these attributes. It utilizes cross-validation
        as defined by the user in the config yaml.  Saves a report containing the feature mask and a yaml drop list
        that can be used by the EDE filter.

        The step attribute must be set from the general yaml conf file and represents how many features should be
        eliminated at every iteration.

        :param model: model instance created from conf yaml parameters
        :param X: training dataframe
        :param y: ground truth from dataframe
        :param cv: type can be int or dictionary describing sklearn compatible type
        :param model_name: name of the model set by export
        :return rfe_estimator: best estimator for the given best features
        """
        try:
            scorer = self.rfe['scorer']
            step = self.rfe['step']
        except Exception as inst:
            logger.error('[{}] : [ERROR] Recursive Feature Elimination parameter error with {} and {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
            sys.exit(1)

        viz = RFECV(model, cv=cv, scoring=scorer, step=step)
        viz.fit(X, y)
        rfe_report = {}
        # cv_scores = viz.cv_scores_  # cv scores
        feature_ranking = viz.ranking_  # feature ranking
        # viz.rfe_estimator_  # best estimator model
        feature_mask = viz.support_ # mask of features to be removed
        rfe_report['Features'] = list(X.columns)
        rfe_report['Mask'] = feature_mask
        rfe_report['Ranking'] = feature_ranking
        df_rfe_report = pd.DataFrame(rfe_report)

        # Save feature to be dropped as yaml so that it can be used by the filter component.
        drop_list = []
        for index, row in df_rfe_report.iterrows():
            if not row['Mask']:
                drop_list.append(row['Features'])
        if drop_list:
            with open(os.path.join(self.modelDir, f"RFE_drop_list.yaml"), 'w')as drop_yaml:
                yaml.dump(drop_list, drop_yaml, default_flow_style=False)
            logger.info('[{}] : [INFO] Features eliminated:  {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), len(drop_list)))

        # Saving report
        df_rfe_report.to_csv(os.path.join(self.modelDir, f"RFE_Report_{self.export}_{model_name}.csv"))

        # Saving Visualization
        rfe_fig = f"Recursive_Feature_Elimination_{self.export}_{model_name}.png"
        viz.show(outpath=os.path.join(self.modelDir, rfe_fig))
        plt.close()
        return viz.rfe_estimator_

    def __rocauc_curves(self,
                        model,
                        X,
                        y,
                        definitions,
                        model_name):

        """
        Receiver Operating Charactersitic/Area Under the Curve plots the tradeoff between
        clasifier sensitivity and specificity i.e. predictive quality. The steepness of the curve
        is important as it represents the maximization of the true positive rate while minimising
        the false positive one.

        :param model: model instance created from conf yaml parameters
        :param X: training dataframe
        :param y: ground truth from dataframe
        :param definitions: Class definitions as defined by tokenization
        :param model_name: name of the model set by export
        """

        # Split data into training and testing  # todo add customizability
        XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=.33, shuffle=True, random_state=42)

        # Compute ROC AUC
        viz = ROCAUC(model, classes=definitions)
        viz.fit(XTrain, yTrain)
        viz.score(XTest, yTest)
        ROCAUC_curve_fig = f"ROCAUC_Curve_{self.export}_{model_name}.png"
        viz.show(outpath=os.path.join(self.modelDir, ROCAUC_curve_fig))
        plt.close()

    def __precision_recall_curve(self,
                                 model,
                                 X,
                                 y,
                                 definitions,
                                 model_name):

        """
        Precision Recall Curve reprezents the tradeoff between precision and recall. This
        version also includes each individual curve as well as F1-score ISO curves.

        :param model: model instance created from conf yaml parameters
        :param X: training dataframe
        :param y: ground truth from dataframe
        :param definitions: Class definitions as defined by tokenization
        :param model_name: name of the model set by export
        """

        # Split data into training and testing  # todo add customizability
        XTrain, XTest, yTrain, yTest = train_test_split(X, y, test_size=.33, shuffle=True, random_state=42)

        # Compute PRC
        viz = PrecisionRecallCurve(model, per_class=True, cmap="Set1", classes=definitions, iso_f1_curves=True,
                                   micro=False)
        viz.fit(XTrain, yTrain)
        viz.score(XTest, yTest)
        precisison_recall_curve_fig = f"Precision_Recall_Curve_{self.export}_{model_name}.png"
        viz.show(outpath=os.path.join(self.modelDir, precisison_recall_curve_fig))
        plt.close()

    def __validation_curve(self,
                           model,
                           X,
                           y,
                           cv,
                           model_name):

        """
        Model validation isused to determine how efective an estimator is on training data and out of core data.
        A hyperparameter of the model can be selected which the user feels can be further tweaked for better
        operation in the desired feature space. We should mention that hyperparameter optimization is done via other
        methods from EDE much better, however finetuning can be done much easier by end-users using this method.

        :param model: model instance created from conf yaml parameters
        :param X: training dataframe
        :param y: ground truth from dataframe
        :param cv: type can be int or dictionary describing sklearn compatible type
        :param model_name: name of the model set by export
        """

        try:
            param_name = self.validationcurve['param_name']
            param_range = self.validationcurve['param_range']
            scorer = self.validationcurve['scoring']
            n_jobs = self.validationcurve['n_jobs']
        except Exception as inst:
            logger.error('[{}] : [ERROR] Validation Curve parameter error with {} and {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
            sys.exit(1)
        viz = ValidationCurve(model, param_name=param_name, param_range=param_range,
                              logx=True, cv=cv, scoring=scorer, n_jobs=n_jobs)
        viz.fit(X, y)
        validation_curve_fig = f"Validation_Curve_{self.export}_{model_name}.png"
        viz.show(outpath=os.path.join(self.modelDir, validation_curve_fig))
        plt.close()

    def __learning_curve(self,
                         model,
                         X,
                         y,
                         cv,
                         model_name
                         ):
        """
        A learning curve shows the relationship between training scores vs cv test scores for
        an estimator with varying number of training sample (see sizes param).
        The visualization generaly shows two things:
        1. How much an estimator benefits from more data.
        2. If the estimator is more sensitive to error due to variance vs bias.


        :param model: model instance created from conf yaml parameters
        :param X: training dataframe
        :param y: ground truth from dataframe
        :param cv: type can be int or dictionary describing sklearn compatible type
        :param model_name: name of the model set by export
        :return:
        """
        try:
            sizes = self.learningcurve['sizes']  # size is set in yaml as a linspace
            scorer = self.learningcurve['scorer']  # scorer must be string from sklearn metrics
            n_jobs = self.learningcurve['n_jobs']
        except Exception as inst:
            logger.error('[{}] : [ERROR] Learning Curve parameter error with {} and {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
            sys.exit(1)
        viz = LearningCurve(model, cv=cv, scoring=scorer, train_sizes=sizes, n_jobs=n_jobs)
        viz.fit(X, y)
        learning_curve_fig = f"Learning_Curve_{self.export}_{model_name}.png"
        viz.show(outpath=os.path.join(self.modelDir, learning_curve_fig))
        plt.close()

    def __ede_cross_validate(self,
                             model,
                             X,
                             y,
                             scoring,
                             cv,
                             return_estimator,
                             definitions,
                             model_name
                             ):
        """
        Used to generate a more clear view of the performance of the initialied model on a fold by fold basis.
        Full Classification and Imbalanced Reports are generated and saved. Confusion metrics and
        Feature importance if applicable to the model.

        :param model: model instance created from conf yaml parameters
        :param X: training dataframe
        :param y: ground truth from dataframe
        :param scoring: scoring functions as dict
        :param cv: type can be int or dictionary describing sklearn compatible type
        :param return_estimator: if True estimators will be returned
        :param definitions: factorization of ground truth defitnitions
        :param model_name: set by export
        :return: 0
        """
        cv_results = {}
        cv_results['fit_time'] = []
        cv_results['scorer_time'] = []
        cv_results['estimator'] = []
        for scorer_name in scoring.keys():
            cv_results[f"train_{scorer_name}"] = []
            cv_results[f"test_{scorer_name}"] = []

        y = pd.Series(y)  # convert y to pd.series for iloc
        fold = 1
        for train_index, test_index in cv.split(X, y):
            # print("Starting fold {}".format(fold))
            logger.info('[{}] : [INFO] Starting CV Fold {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), fold))
            Xtrain, Xtest = X.iloc[train_index], X.iloc[test_index]
            ytrain, ytest = y.iloc[train_index], y.iloc[test_index]

            start_training_time = time.time()
            model.fit(Xtrain, ytrain)
            total_training_time = time.time() - start_training_time
            cv_results['fit_time'].append(total_training_time)
            if return_estimator:
                cv_results['estimator'].append(model)

            start_prediction_time = time.time()
            ypred_test = model.predict(Xtest)
            total_prediction_time = time.time() - start_prediction_time
            cv_results['scorer_time'].append(total_prediction_time)
            for k, v in scoring.items():  # scoring similar to cv eval and HPO
                scorer = get_scorer(v)
                score_testing = scorer(model, Xtest, ytest)
                score_training = scorer(model, Xtrain, ytrain)
                logger.info('[{}] : [INFO] Fold {} {} training score is {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), fold, k, score_training))

                logger.info('[{}] : [INFO] Fold {} {} testing score is {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), fold, k, score_testing))

                cv_results[f"train_{k}"].append(score_training)
                cv_results[f"test_{k}"].append(score_testing)

                # Full classification report
                logger.info('[{}] : [INFO] Computing classification report for fold {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), fold))
                print(classification_report(ytest, ypred_test, digits=4, target_names=definitions))
                cf_report = classification_report(ytest, ypred_test, output_dict=True, digits=4,
                                                  target_names=definitions)
                df_classification_report = pd.DataFrame(cf_report).transpose()
                cf_report_name = f"Classification_Report_{model_name}_Fold_{fold}.csv"
                df_classification_report.to_csv(os.path.join(self.modelDir, cf_report_name))

                # Full imbalanced classification report
                logger.info('[{}] : [INFO] Computing imbalanced classification report for fold {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), fold))
                print(classification_report_imbalanced(ytest, ypred_test, digits=4, target_names=definitions))
                imb_cf_report = classification_report_imbalanced(ytest, ypred_test, output_dict=True, digits=4,
                                                                 target_names=definitions)
                df_imb_classification_report = pd.DataFrame(imb_cf_report).transpose()
                imb_df_report_name = f"Imbalanced_classification_Report_{model_name}_Fold_{fold}.csv"
                df_imb_classification_report.to_csv(os.path.join(self.modelDir, imb_df_report_name))

                # Confusion matrix
                logger.info('[{}] : [INFO] Computing confusion matrix for fold {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), fold))
                self.__confusion_matrix(ytest, ypred_test, definitions, model_name=model_name, fold=fold)

                # Feature Importance, if applicable
                try:
                    self.__feature_imp(model, model_name, X, fold)
                except Exception:
                    logger.debug('[{}] : [DEBUG] Feature Importance not computable for {}'.format(
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), model_name))
                fold += 1
        return cv_results

    def __feature_imp(self,
                      model,
                      model_name,
                      X,
                      fold=None):
        feat_importances = pd.Series(model.feature_importances_, index=X.columns)
        sorted_feature = feat_importances.sort_values(ascending=True)
        sorted_feature = sorted_feature.tail(30)
        n_col = len(sorted_feature)
        # plt.figure(figsize=(30, 10))
        plt.figure()
        plt.autoscale()
        if fold is None:
            plt.title("Feature importances {}".format(model_name), fontsize=15)
            fi_fig = "FI_{}.png".format(model_name)
        else:
            plt.title("Feature importances {} Fold {}".format(model_name, fold), fontsize=15)
            fi_fig = "FI_{}_{}.png".format(model_name, fold)
        plt.barh(range(n_col), sorted_feature,
                 color="r", align="center")
        # If you want to define your own labels,
        # change indices to a list of labels on the following line.
        plt.yticks(range(n_col), sorted_feature.index)
        plt.ylim([-1, n_col])
        plt.savefig(os.path.join(self.modelDir, fi_fig), bbox_inches="tight")
        plt.close()
        return 0

    def __confusion_matrix(self,
                           ytest,
                           ypred_test,
                           definitions,
                           model_name,
                           fold=None):

        cf_matrix = confusion_matrix(ytest, ypred_test)
        ht_cf = sns.heatmap(cf_matrix, annot=True, yticklabels=list(definitions), xticklabels=list(definitions))
        if fold is None:
            plt.title('Confusion Matrix', fontsize=15)  # title with fontsize 20
            cf_fig = "CM_{}.png".format(model_name)
        else:
            plt.title('Confusion Matrix Fold {}'.format(fold), fontsize=15)  # title with fontsize 20
            cf_fig = "CM_{}_{}.png".format(model_name, fold)
        plt.xlabel('Ground Truth', fontsize=10)  # x-axis label with fontsize 15
        plt.ylabel('Predictions', fontsize=10)  # y-axis label with fontsize 15
        ht_cf.figure.savefig(os.path.join(self.modelDir, cf_fig), bbox_inches="tight")
        plt.close()
        return 0

    def dask_hpo(self, param_dist,
                 mname,
                 X,
                 y,
                 hpomethod,
                 hpoparam,
                 classification_method=None):
        user_m = False
        if classification_method is not None and not 'randomforest':  # TODO fix
            user_m = True
            try:
                classification_type = str(classification_method).split('(')[0]
            except Exception:
                classification_type = type(classification_method)

            logger.info('[{}] : [INFO] Classification Method set to {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type))
            for k, v in classification_method.get_params().items():
                logger.info('[{}] : [INFO] Classification parameter {} set to {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), k, v))
        else:
            classification_type = 'RandomForest'
            logger.info('[{}] : [INFO] Method set to {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type))
            try:
                clf = RandomForestClassifier()
            except Exception as inst:
                logger.error('[{}] : [INFO] Failed to instanciate {} with {} and {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type, type(inst),
                    inst.args))
                sys.exit(1)
        if self.cv is None:
            cv = self.cv
            logger.info('[{}] : [INFO] {} Cross Validation not set, using default'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type, self.cv))
        elif isinstance(self.cv, int):
            logger.info('[{}] : [INFO] {} Cross Validation set to {} folds:'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type, self.cv))
            cv = self.cv
        else:
            try:
                logger.info('[{}] : [INFO] {} Cross Validation set to use {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type,
                    self.cv['Type']))
            except Exception:
                logger.error('[{}] : [ERROR] Cross Validation split generator type not set!'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
                sys.exit(1)
            cv = self.__crossValidGenerator(self.cv)
        if self.scorers is None:
            scorer = None
        else:
            scorer = self.__dask_classification_scorer(self.scorers)
        if hpomethod == 'Random':
            rs_settings = {
                "param_distributions": param_dist,
                "cv": cv,
                "scoring": scorer,
                "return_train_score": self.returnestimators
            }
            rs_settings.update(hpoparam)
            for k, v in rs_settings.items():
                logger.info('[{}] : [INFO] RandomSearch param {} set to {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), k, v))
            if user_m:
                rs_settings.update({"estimator": classification_method})
                search = RandomizedSearchCV(**rs_settings)
            else:
                rs_settings.update({"estimator": clf})
                search = RandomizedSearchCV(**rs_settings)

        elif hpomethod == 'Grid':
            gs_settings = {
                "param_grid": param_dist,
                "cv": cv,
                "scoring": scorer,
                "return_train_score": self.returnestimators
            }
            gs_settings.update(hpoparam)
            for k, v in gs_settings.items():
                logger.info('[{}] : [INFO] GridSearch param {} set to {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), k, v))
            if user_m:
                gs_settings.update({"estimator": classification_method})
                search = GridSearchCV(**gs_settings)
            else:
                gs_settings.update({"estimator": clf})
                search = GridSearchCV(**gs_settings)
        else:
            logger.error('[{}] : [ERROR] Invalid HPO method specified: {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), hpomethod))
            sys.exit(2)
        try:
            refit = hpoparam['refit']
        except Exception as inst:
            logger.error('[{}] : [ERROR] HPO param must be set! Exiting!'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
            sys.exit(1)
        try:
            with joblib.parallel_backend('dask'):
                logger.info('[{}] : [INFO] Using Dask backend for HPO of {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type))
                sr_clf = search.fit(X, y)
        except Exception as inst:
            logger.error('[{}] : [ERROR] Failed to fit {} during Cross Validation with {} and {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type, type(inst),
                inst.args))
            sys.exit(1)

        if isinstance(refit, str) or refit:
            # print(search.best_score_)
            if isinstance(refit, str):
                logger.info('[{}] : [INFO] Best HPO {} score {} with refit time of {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), refit, sr_clf.best_score_,
                    sr_clf.refit_time_))
            else:
                logger.info('[{}] : [INFO] Best HPO score {} with refit time of {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), sr_clf.best_score_,
                    sr_clf.refit_time_))
            for k, v in sr_clf.best_params_.items():
                logger.info('[{}] : [INFO] Classifier {} param {} best value {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), classification_type, k, v))
            best_conf_loc = os.path.join(self.modelDir,
                                         "{}_HPO_{}_best_{}_config.yaml".format(classification_type, hpomethod,
                                                                                self.export))
            logger.info('[{}] : [INFO] Saving best configuration to {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), best_conf_loc))

            with open(best_conf_loc, 'w') as yaml_config:
                yaml.dump(sr_clf.best_params_, yaml_config, default_flow_style=False)

            if isinstance(self.cv, dict):
                cv_name = self.cv['Type']
            elif self.cv is None:
                cv_name = 5  # Default value for random and Grid
            else:
                cv_name = cv
            cv_res_loc = os.path.join(self.modelDir,
                                      "{}_HPO{}_CV_{}_restults.csv".format(classification_type, hpomethod, cv_name))
            logger.info('[{}] : [INFO] Saving HPO {} {} CV Metrics to {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), hpomethod, cv, cv_res_loc))
            cv_res = pd.DataFrame(sr_clf.cv_results_)
            cv_res.to_csv(cv_res_loc, index=False)
            logger.info('[{}] : [INFO] Saving Best Estimator ...'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), hpomethod, cv, cv_res_loc))
            # print(sr_clf.best_estimator_)
            # print(sr_clf.cv_results_)
            self.__serializemodel(model=sr_clf.best_estimator_, method=classification_type, mname=self.export)
        else:
            logger.warning('[{}] : [WARN] Skipping HPO Report ...'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))

    def __appendPredictions(self,
                            method,
                            mname,
                            data,
                            pred,
                            pred_proba=None):
        fpath = "{}_{}.csv".format(method, mname)
        fname = os.path.join(self.modelDir, fpath)
        logger.info('[{}] : [INFO] Appending predictions to  data ... '.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
        # print(list(pred))
        data['pred'] = pred
        if pred_proba is not None:
            logger.info('[{}] : [INFO] Appending prediction probabilities to  data ... '.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
            data['prob_0'] = pred_proba[:, 0]
            data['prob_1'] = pred_proba[:, 1]
        logger.info('[{}] : [INFO] Saving to {}.'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), fname))
        data.to_csv(fname, index=True)

    def randomForest(self, settings,
                     data=None,
                     dropna=True):
        if "min_weight_faction_leaf" not in settings:
            print("Received settings for RandomForest are %s invalid!" % str(settings))
            logger.error('[%s] : [ERROR] Received settings for RandomForest %s are invalid',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(settings))
            sys.exit(1)

        if settings['random_state'] == 'None':
            settings['random_state'] = None
        else:
            settings['random_state'] = int(settings['random_state'])

        if settings['max_depth'] == 'None':
            max_depth = None
        else:
            max_depth = int(settings['max_depth'])

        if isinstance(settings['bootstrap'], str):
            settings['iso_bootstrap'] = str2Bool(settings['bootstrap'])

        rfallowedSettings = ["n_estimators", "criterion", "max_features", "max_depth", "min_sample_split",
                             "min_sample_leaf", "min_weight_faction_leaf", "min_impurity_split", "bootstrap", "n_jobs",
                             "random_state", "verbose"]
        for k, v in settings.items():
            if k in rfallowedSettings:
                logger.info('[%s] : [INFO] RandomForest %s set to %s',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), k, v)
                print("RandomForest %s set to %s" % (k, v))

        if not isinstance(self.export, str):
            mname = 'default'
        else:
            mname = self.export

        df = self.__loadData(data, dropna)
        features = df.columns[:-1]
        X = df[features]
        y = df.iloc[:, -1].values

        # clf = RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1, n_jobs=2)
        clf = RandomForestClassifier(n_estimators=int(settings["n_estimators"]), criterion=settings["criterion"],
                                     max_features=settings["max_features"], max_depth=max_depth,
                                     min_samples_split=int(settings["min_sample_split"]),
                                     min_samples_leaf=int(settings["min_sample_leaf"]),
                                     min_weight_fraction_leaf=int(settings["min_weight_faction_leaf"]),
                                     bootstrap=settings["bootstrap"],
                                     n_jobs=int(settings["n_jobs"]),
                                     random_state=settings["random_state"], verbose=int(settings["verbose"]))

        if self.validratio:
            trainSize = 1.0 - self.validratio
            print("Random forest training to validation ratio set to: %s" % str(self.validratio))
            logger.info('[%s] : [INFO] Random forest training to validation ratio set to: %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(self.validratio))
            d_train, d_test, f_train, f_test = self.__dataSplit(X, y, testSize=self.validratio, trainSize=trainSize)
            clf.fit(d_train, f_train)
            predict = clf.predict(d_train)
            print("Prediction for Random Forest Training:")
            print(predict)

            print("Actual labels of training set:")
            print(f_train)

            predProb = clf.predict_proba(d_train)
            print("Prediction probabilities for Random Forest Training:")
            print(predProb)

            score = clf.score(d_train, f_train)
            print("Random Forest Training Score: %s" % str(score))
            logger.info('[%s] : [INFO] Random forest training score: %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(score))

            feature_imp = list(zip(d_train, clf.feature_importances_))
            print("Feature importance Random Forest Training: ")
            print(list(zip(d_train, clf.feature_importances_)))
            logger.info('[%s] : [INFO] Random forest feature importance: %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(feature_imp))

            pred_valid = clf.predict(d_test)
            print("Random Forest Validation set prediction: ")
            print(pred_valid)
            print("Actual values of validation set: ")
            print(d_test)
            score_valid = clf.score(d_test, f_test)
            print("Random Forest validation set score: %s" % str(score_valid))
            logger.info('[%s] : [INFO] Random forest validation score: %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(score_valid))
        else:
            clf.fit(X, y)
            predict = clf.predict(X)
            print("Prediction for Random Forest Training:")
            print(predict)

            print("Actual labels of training set:")
            print(y)

            predProb = clf.predict_proba(X)
            print("Prediction probabilities for Random Forest Training:")
            print(predProb)

            score = clf.score(X, y)
            print("Random Forest Training Score: %s" % str(score))

            fimp = list(zip(X, clf.feature_importances_))
            print("Feature importance Random Forest Training: ")
            print(fimp)
            dfimp = dict(fimp)
            dfimp = pd.DataFrame(list(dfimp.items()), columns=['Metric', 'Importance'])
            sdfimp = dfimp.sort('Importance', ascending=False)
            dfimpCsv = 'Feature_Importance_%s.csv' % mname
            sdfimp.to_csv(os.path.join(self.modelDir, dfimpCsv))
            if self.validation is None:
                logger.info('[%s] : [INFO] Validation is set to None',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                # return True
            else:
                vfile = os.path.join(self.dataDir, settings['validation'])
                logger.info('[%s] : [INFO] Validation data file is set to %s',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(vfile))
                if not os.path.isfile(vfile):
                    print("Validation file %s not found" % vfile)
                    logger.error('[%s] : [ERROR] Validation file %s not found',
                                 datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(vfile))
                else:
                    df_valid = pd.read_csv(vfile)
                    if dropna:
                        df_valid = df_valid.dropna()
                    features_valid = df_valid.columns[:-1]
                    X_valid = df_valid[features_valid]
                    y_valid = df_valid.iloc[:, -1].values
                    pred_valid = clf.predict(X_valid)
                    print("Random Forest Validation set prediction: ")
                    print(pred_valid)
                    print("Actual values of validation set: ")
                    print(y_valid)
                    score_valid = clf.score(X_valid, y_valid)
                    print("Random Forest validation set score: %s" % str(score_valid))
                    # return True
        self.__serializemodel(clf, 'RandomForest', mname)
        return clf

    def trainingDataGen(self, settings, data=None, dropna=True, onlyAno=True):
        print("Starting training data generation ....")
        logger.info('[%s] : [INFO] Starting training data generation ...',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
        df = self.__loadData(data, dropna)
        print(df.index.name)
        if df.index.name is None:
            df.set_index('key', inplace=True)
        logger.info('[%s] : [INFO] Input Dataframe shape: %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(df.shape))

        if 'iso_n_estimators' not in list(settings.keys()):
            settings['iso_n_estimators'] = 100
        if 'iso_max_samples' not in list(settings.keys()):
            settings['iso_max_samples'] = 'auto'
        if 'iso_contamination' not in list(settings.keys()):
            settings['iso_contamination'] = 0.1
        if 'iso_bootstrap' not in list(settings.keys()):
            settings['iso_bootstrap'] = True
        if 'iso_max_features' not in list(settings.keys()):
            settings['iso_max_features'] = 1.0
        if 'iso_n_jobs' not in list(settings.keys()):
            settings['iso_n_jobs'] = 1
        if 'iso_random_state' not in list(settings.keys()):
            settings['iso_random_state'] = None
        if 'iso_verbose' not in list(settings.keys()):
            settings['iso_verbose'] = 0

        if settings['iso_random_state'] == 'None':
            settings['iso_random_state'] = None

        if isinstance(settings['iso_bootstrap'], str):
            settings['iso_bootstrap'] = str2Bool(settings['bootstrap'])

        if isinstance(settings['iso_verbose'], str):
            settings['iso_verbose'] = str2Bool(settings['verbose'])

        if settings['iso_max_samples'] != 'auto':
            settings['iso_max_samples'] = int(settings['max_samples'])

        if isinstance(settings['iso_max_features'], str):
            settings["iso_max_features"] = 1.0
        # print type(settings['max_samples'])
        allowedIso = ['iso_n_estimators', 'iso_max_samples', 'iso_contamination', 'iso_bootstrap', 'iso_max_features',
                      'iso_n_jobs',
                      'iso_random_state', 'iso_verbose']
        for k, v in settings.items():
            if k in allowedIso:
                logger.info('[%s] : [INFO] IsolationForest %s set to %s',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), k, v)
                print("IsolationForest %s set to %s" % (k, v))

        try:
            clf = IsolationForest(n_estimators=int(settings['iso_n_estimators']),
                                  max_samples=settings['iso_max_samples'],
                                  contamination=float(settings['iso_contamination']), bootstrap=settings['bootstrap'],
                                  max_features=float(settings['iso_max_features']), n_jobs=int(settings['iso_n_jobs']),
                                  random_state=settings['iso_random_state'], verbose=settings['iso_verbose'])
        except Exception as inst:
            logger.error('[%s] : [ERROR] Cannot instanciate isolation forest with %s and %s',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args)
            print("Error while  instanciating isolation forest with %s and %s" % (type(inst), inst.args))
            sys.exit(1)

        clf.fit(df)
        pred = clf.predict(df)
        # print data.shape
        # print len(pred)
        print("Prediction for IsolationForest:")
        print(pred)
        anomalies = np.argwhere(pred == -1)
        normal = np.argwhere(pred == 1)

        print("Number of anomalies detected: %s" % str(len(anomalies)))

        logger.info('[%s] : [INFO] Number of anomalies detected: %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(len(anomalies)))

        # Generate anomalydataframe/ anomalies only
        slist = []
        for an in anomalies:
            slist.append(df.iloc[an[0]])
        anomalyFrame = pd.DataFrame(slist)
        # if df.index.name.index is None:
        #     anomalyFrame.set_index('key', inplace=True)

        logger.info('[%s] : [INFO] Anomaly Dataframe shape: %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(anomalyFrame.shape))

        if self.checkpoint:
            logger.info('[%s] : [INFO] Anomalies checkpointed.',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(len(anomalies)))
            anomalyFrame.to_csv(os.path.join(self.dataDir, 'AnomalyFrame.csv'))

        print("Normalizing data ...")
        X = StandardScaler().fit_transform(anomalyFrame)

        print("Started Anomaly clustering ...")
        if 'db_eps' not in list(settings.keys()):
            settings['db_eps'] = 0.9
        if 'db_min_samples' not in list(settings.keys()):
            settings['db_min_samples'] = 40
        if 'db_metric' not in list(settings.keys()):
            settings['db_metric'] = 'euclidean'
        if 'db_algorithm' not in list(settings.keys()):
            settings['db_algorithm'] = 'auto'
        if 'db_leaf_size' not in list(settings.keys()):
            settings['db_leaf_size'] = 30
        if 'db_p' not in list(settings.keys()):
            settings['db_p'] = 0.2
        if 'db_n_jobs' not in list(settings.keys()):
            settings['db_n_jobs'] = 1

        allowedDB = ['db_eps', 'db_min_samples', 'db_metric', 'db_algorithm', 'db_leaf_size', 'db_p', 'db_n_jobs']
        for k, v in settings.items():
            if k in allowedDB:
                logger.info('[%s] : [INFO] SDBSCAN %s set to %s',
                            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), k, v)
                print("SDBSCAN %s set to %s" % (k, v))

        # db = DBSCAN(eps=0.9, min_samples=40).fit(X)
        try:
            db = DBSCAN(eps=float(settings['db_eps']), min_samples=int(settings['db_min_samples']),
                        metric=settings['db_metric'],
                        algorithm=settings['db_algorithm'], leaf_size=int(settings['db_leaf_size']),
                        p=float(settings['db_p']),
                        n_jobs=int(settings['db_n_jobs'])).fit(X)
        except Exception as inst:
            logger.error('[%s] : [ERROR] Cannot instanciate sDBSCAN with %s and %s',
                         datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args)
            print("Error while  instanciating sDBSCAN with %s and %s" % (type(inst), inst.args))
            sys.exit(1)
        print("Finshed  Anomaly clustering.")
        labels = db.labels_

        # print len(labels)
        # print anomalyFrame.shape
        # print X[labels == -1]
        # Number of clusters in labels, ignoring noise if present.
        n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

        print('Estimated number of clusters: %d' % n_clusters_)
        logger.info('[%s] : [INFO] Estimated number of clusters: %d',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), n_clusters_)

        # Add labels to target column
        print("Adding labels to data ...")
        anomalyFrame['Target'] = labels

        if onlyAno:
            # Remove noise from data
            print("Removing noise from data ...")
            data_labeled = anomalyFrame[anomalyFrame["Target"] != -1]
            if self.checkpoint:
                data_labeled.to_csv(os.path.join(self.dataDir, 'AnomalyFrame_Labeled.csv'))
            print("Finished training data generation")
            return data_labeled
        else:
            bval = np.amax(labels)
            print(bval)
            # replace noise with new label
            labels[labels == -1] = bval + 1
            # add one to all elements in array so that 0 is free for normal events
            nlabels = labels + 1
            anomalyFrame['Target2'] = nlabels
            # print data_labeled[['Target', 'Target2']]
            # initialize empty column
            df['TargetF'] = np.nan
            print(df.shape)
            print(anomalyFrame.shape)
            # add clustered anomalies to original dataframe
            for k in anomalyFrame.index.values:
                try:
                    df.set_value(k, 'TargetF', anomalyFrame.loc[k, 'Target2'])
                except Exception as inst:
                    print(inst.args)
                    print(type(inst))
                    print(k)
                    sys.exit()
            # Mark all normal instances as 0
            df = df.fillna(0)
            if df.isnull().values.any():
                logger.error('[%s] : [ERROR] Found null values in anomaly dataframe after processing!',
                             datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'))
                print('Found null values in anomaly dataframe after processing!')
                sys.exit(1)
            if self.checkpoint:
                df.to_csv(os.path.join(self.dataDir, 'AnomalyFrame_Labeled_Complete.csv'))
            print("Finished training data generation")
            return df

    def __dask_classification_scorer(self, scorers):
        scorer_dict = {}
        if 'Scorer_list' in scorers.keys():
            logger.info('[{}] : [INFO] Found scorer list in config'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S')))
            try:
                scorer_lst = scorers['Scorer_list']
                for s in scorer_lst:
                    scorer_dict[s['Scorer']['Scorer_name']] = s['Scorer']['skScorer']
                del scorers['Scorer_list']
            except Exception as inst:
                logger.error('[{}] : [ERROR] Error while parsing scorer list with {} and {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type(inst), inst.args))
                sys.exit(2)
        for user_name, sc_name in scorers.items():
            try:
                sc_mod = importlib.import_module(self.skscorer)  # todo add support for non sklearn based scoring
                scorer_instance = getattr(sc_mod, sc_name)
                scorer = make_scorer(scorer_instance)
                logger.info('[{}] : [INFO] Found user defined scorer. Initializing {} '.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), sc_name))
            except Exception as inst:
                logger.error('[{}] : [ERROR] Error while initializing scorer {} with {} and {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), sc_name, type(inst), inst.args))
                sys.exit(2)
            scorer_dict[user_name] = scorer
        return scorer_dict

    def __loadData(self, data=None, dropna=True):
        if not self.checkpoint:
            dfile = os.path.join(self.dataDir, self.training)
            logger.info('[%s] : [INFO] Data file is set to %s',
                        datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(dfile))
            if not os.path.isfile(dfile):
                print("Training file %s not found" % dfile)
                logger.error('[%s] : [ERROR] Training file %s not found',
                             datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(dfile))
                sys.exit(1)
            else:
                df = pd.read_csv(dfile)
        else:
            if not isinstance(data, pd.core.frame.DataFrame):
                print("Data is of type %s and not dataframe, exiting!" % type(data))
                logger.error('[%s] : [ERROR] Data is of type %s and not dataframe, exiting!',
                             datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), str(type(data)))
                sys.exit(1)
            df = data
        if df.index.name is None:
            df.set_index('key', inplace=True)
        if dropna:
            df = df.dropna()
        return df

    def __gridSearch(self, est, X, y):
        if isinstance(est, RandomForestClassifier):
            param_grid = {
                'n_estimators': [200, 300, 400, 500, 800, 1000],
                'max_features': ['auto', 'sqrt', 'log2'],
                'max_depth': [5, 15, 25]
            }

        CV_rfc = GridSearchCV(estimator=est, param_grid=param_grid, cv=5)
        CV_rfc.fit(X, y)
        logger.info('[%s] : [INFO] Best parameters are: %s',
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), CV_rfc.best_params_)
        print('Best parameters are: %s' % CV_rfc.best_params_)
        return CV_rfc.best_params_

    def __loadClassificationModel(self,
                                  method,
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

    def __crossValidGenerator(self, cv_dict):

        type_cv = cv_dict['Type']
        logger.info('[{}] : [INFO] Instanciating CV Generator {} ...'.format(
            datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type_cv))
        try:
            params = cv_dict['Params']
            for param_name, param_value in params.items():
                logger.info('[{}] : [DEBUG] {} parameter {} set to {}'.format(
                    datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'),
                    type_cv, param_name, param_value))
        except:
            logger.warning('[{}] : [WARN] Params not set for {}, using default'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type_cv))
            params = {}
        try:
            cv_gen_mod = importlib.import_module(self.sksplitgen)
            cv_gen_instance = getattr(cv_gen_mod, type_cv)
            cv_gen = cv_gen_instance(**params)
        except Exception as inst:
            logger.error('[{}] : [ERROR] Failed to instanciate CV Generator {} with {} and {}'.format(
                datetime.fromtimestamp(time.time()).strftime('%Y-%m-%d %H:%M:%S'), type_cv, type(inst), inst.args))
        return cv_gen

    def __serializemodel(self, model, method, mname):
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

    def __normalize(self, data):
        normalized_data = StandardScaler().fit_transform(data)
        return normalized_data

    def __dataSplit(self, X, y, trainSize=None, testSize=None):
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=testSize, train_size=trainSize)
        return X_train, X_test, y_train, y_test
