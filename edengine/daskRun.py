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
from sklearn.linear_model import SGDClassifier
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestClassifier
from dask_ml.datasets import make_classification
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
# from dask_ml.model_selection import IncrementalSearchCV, GridSearchCV, RandomizedSearchCV, HyperbandSearchCV
from dask.distributed import Client, LocalCluster, progress
import warnings
from dask.utils import format_bytes
import numpy as np
import pandas as pd
import dask.array as da
import os


class EdeDaskEngine:

    def __init__(self, local,
                 n_workers,
                 report_dir='reports',
                 models_dir='models',
                 schedueler_endpoint=None,
                 schedueler_port='8787',
                 enforce_check=False):
        self.local = local
        self.n_workers = n_workers
        self.schedueler_endpoint = schedueler_endpoint
        self.schedueler_port = schedueler_port
        self.enforce_check = enforce_check
        self.report_dir = report_dir
        self.models_dir = models_dir

    def __engine_init(self):
        if self.local:
            print("Staring local cluster with {} workers".format(self.n_workers))
            cluster = LocalCluster(n_workers=self.n_workers)  # TODO: add more arguments
            print("Cluster settings:")
            print(cluster)
            client = Client(cluster)
            print("Client settings:")
            print(client)
        else:
            print("Connecting to remote schedueler at {}".format(self.schedueler_endpoint))
            scheduler_address = '{}:{}'.format(self.schedueler_endpoint, self.schedueler_port)
            client = Client(address=scheduler_address)
            print("Client settings:")
            print(client)
            client.get_versions(check=self.enforce_check)
        return client

    def __check_dirs(self):
        if not os.path.isdir(self.report_dir):
            print("Report directory not found creating")
            os.makedirs(self.report_dir)
        if not os.path.isdir(self.models_dir):
            print("Models directory not found, creating")
            os.makedirs(self.models_dir)

    def load_data(self, data, dropList, colover=True, forceFill=True, target='Alert'):
        def check_data(X_train, X_test):
            for name, X in [("train", X_train), ("test", X_test)]:
                print("dataset =", name)
                print("shape =", X.shape)
                print("bytes =", format_bytes(X.nbytes))
                print("-" * 20)

        def ohEncoding(data, cols, replace=False):
            vec = DictVectorizer()
            mkdict = lambda row: dict((col, row[col]) for col in cols)
            vecData = pd.DataFrame(vec.fit_transform(data[cols].apply(mkdict, axis=1)).toarray())
            vecData.columns = vec.get_feature_names()
            vecData.index = data.index
            if replace is True:
                data = data.drop(cols, axis=1)
                data = data.join(vecData)
            return data, vecData, vec

        df = pd.read_csv(data)
        print("Shape after loading is: {}".format(df.shape))

        print("Droped columns are: %s" % dropList)
        df = df.drop(dropList, axis=1)
        print("Index Name: %s" % df.index.name)
        print("Dataframe shape (row, col): %s" % str(df.shape))

        # encode dataframe
        col = []
        for el, v in df.dtypes.iteritems():
            # print el
            if v == 'object':
                col.append(el)
        print("Categorical columns detected: %s" % str(col))
        if colover == True:
            col = ['SD_RETL_ID', 'SD_PROD_IND', 'SD_TERM_NAME_LOC', 'SD_TERM_CITY_OLD', 'SD_TERM_ST', 'SD_TERM_CNTRY',
                   'SD_CR_DB_IND', 'SD_CASH_IND', 'SD_CRD_PLASTIC_TYP', 'SD_TERM_CITY', 'SD_TRAN_RSN_CDE', 'SD_TERM_ID',
                   'Alert']
            print("Categorical column overide detected, using: %s" % str(col))
        else:
            print("Categorical columns used: %s" % str(col))

        df, t, v = ohEncoding(df, col, replace=True)

        print("Shape of the dataframe after encoding: {}".format(df.shape))

        if forceFill == True:
            df = df.fillna(0)

        # Target always last column of dataframe
        features = df.columns[:-1]

        print("Detected Features are: %s" % features)

        X = df[features]
        # Target always last column of dataframe

        if target:
            y = df[target].values
            # Remove Target column from training set
            X = df.drop([target], axis=1)
        else:
            y = df.iloc[:, -1].values
            # Remove Target column from training set
            X = df.iloc[:, :-1]

        X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.75, test_size=0.25)

        # Normalization
        print("Normalizing data ....")
        scaler = StandardScaler()

        scaler.fit(X_train)
        # Now apply the transformations to the data:
        X_train = scaler.transform(X_train)
        X_test = scaler.transform(X_test)
        check_data(X_train, X_test)
        print(type(X_train))
        print("Done")
        return X_train, X_test, y_train, y_test

    def hpo_dask(self, model,
                 params,
                 X,
                 y,
                 exp_name='exp_0',
                 joblib=True,
                 cv=2,
                 n_iter=10,
                 verbose=10,
                 n_jobs=-1,
                 random_state=0,
                 report=True):

        self.__engine_init()
        if joblib:
            from sklearn.model_selection import RandomizedSearchCV
            import joblib
            search = RandomizedSearchCV(model, params, cv=cv, n_iter=n_iter, verbose=verbose,
                                        n_jobs=n_jobs, random_state=random_state)
            with joblib.parallel_backend('dask'):
                print("Using dask backend")
                print("Started fitting")
        else:
            from dask_ml.model_selection import RandomizedSearchCV
            search = RandomizedSearchCV(model, params, cv=cv, n_iter=n_iter, n_jobs=-1,
                                        random_state=random_state)
            print("Started fitting")
        search.fit(X, y)
        best = search.best_estimator_
        print("Best score {}".format(search.best_score_))
        print("Best params {}".format(search.best_params_))
        if report:
            from joblib import dump, load
            import json
            self.__check_dirs()
            print("Saving report and best model")
            cv_report = pd.DataFrame(search.cv_results_)
            rep_name = "{}_cv_results.csv".format(exp_name)
            path_report = os.path.join(self.report_dir, rep_name)
            cv_report.to_csv(path_report)
            best_name = "{}_best.pkl"
            path_model = os.path.join(self.models_dir, best_name)
            dump(best, path_model)
            param_name = "{}_best_params.json".format(search.best_params_)
            path_best_params = os.path.join(self.report_dir, param_name)
            with open(path_best_params, 'w') as fp:
                json.dump(search.best_params_, fp)
        return best

    def hpo_local(self):
        return 1

    def tpot(self):
        return 1