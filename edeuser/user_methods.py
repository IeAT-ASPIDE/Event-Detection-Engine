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

from sklearn.ensemble import IsolationForest
import pandas as pd
import numpy as np
import seaborn as sns
import os
import matplotlib.pyplot as plt
from yellowbrick.features import rank2d


def user_iso(n_estimators,
             contamination,
             max_features,
             n_jobs,
             warm_start,
             random_state,
             bootstrap,
             verbose,
             max_samples):
    '''
    Example of user defined unsupervised predictive model
    :param behaviour:
    :param n_estimators:
    :param contamination:
    :param max_features:
    :param n_jobs:
    :param warm_start:
    :param random_state:
    :param bootstrap:
    :param verbose:
    :param max_samples:
    :return: model instance
    '''
    clf = IsolationForest(
        n_estimators=n_estimators,
             contamination=contamination,
             max_features=max_features,
             n_jobs=n_jobs,
             warm_start=warm_start,
             random_state=random_state,
             bootstrap=bootstrap,
             verbose=verbose,
             max_samples=max_samples)
    return clf


def wrapper_add_columns(columns=(None, None),
                        column_name=None):
    '''
    Wrapper function example which wraps user defined
    feature engineering function
    :param columns:
    :param column_name:
    :return:
    '''
    def add_columns(df,
                    columns=columns,
                    column_name=column_name):
        if columns[0] is None:
            return df
        sum_col = df[columns[0]]+df[columns[1]]
        df[column_name] = sum_col
        return df
    return add_columns


def wrapper_analysis_corr(name,
                     annot=True,
                     cmap='RdBu_r',
                     columns=[],
                     location=None):
    def pearson_corelation_heatmap(data,
                                   name=name,
                                   annot=annot,
                                   cmap=cmap,
                                   columns=columns,
                                   location=location):
        data = data[columns]
        data = data.set_index('time', inplace=False)
        data = data.astype(float)
        # print(data.shape)
        # print(data.dtypes)
        corr = data.corr(method='pearson')
        # print(corr.shape)
        plt.subplots(figsize=(20, 15))
        sns_plot = sns.heatmap(corr,
                               xticklabels=corr.columns,
                               yticklabels=corr.columns,
                               annot=annot,
                               cmap=cmap)
        fig = sns_plot.get_figure()
        fig_loc = os.path.join(location, "pearson_corr_{}.png".format(name))
        file_loc = os.path.join(location, "pearson_corr_{}.csv".format(name))
        corr.to_csv(file_loc)
        fig.savefig(fig_loc)
        plt.close()
        return name
    return pearson_corelation_heatmap


def wrapper_analysis_plot(name,
                          columns,
                          location):
    def line_plot(data,
                  name=name,
                  columns=columns,
                  location=location):
        data = data.astype(float)
        sns_plot = sns.lineplot(
            x='time',
            y='node_load1_10.211.55.101:9100',
            data=data[columns]
        )
        # sns_plot = sns.replot(x=)
        fig = sns_plot.get_figure()
        fig_loc = os.path.join(location, "lineplot_{}.png".format(name))
        fig.savefig(fig_loc)
        plt.close()
        return name
    return line_plot


def wrapper_rank2_pearson(name,
                          location,
                          dcol=[],
                          show=False):
    def rank2_pearson(data,
                      name=name,
                      location=location,
                      dcol=dcol,
                      show=show):
        df_data = data.drop(dcol, axis=1)
        ax = plt.axes()
        try:
            rank2d(df_data, ax=ax, show_feature_names=show)
        except Exception as inst:
            print(type(inst), inst.args)
        ax.set_title(name)
        plt.savefig(os.path.join(location, f"{name}_Pearson_Corr.png"))
        return name
    return rank2_pearson


def wrapper_improved_pearson(name,
                             location,
                             dcol=[],
                             cmap='coolwarm',
                             show=False):
    """
    Computes the Pearson correlation between features. If the augmentation step is
    not used and categorical features are still present these will be converted from object dtypes
    to float
    :param name: name to be used for visuzliation
    :param location: location to save the heatmap
    :param dcol: columns to be droped
    :param cmap: color map
    :param show: show feature names
    :return: name
    """

    def improved_pearsons(data,
                         name=name,
                         location=location,
                         dcol=dcol,
                         cmap=cmap,
                         show=show):

        # Dectect object columns and convert them to float
        s = data.select_dtypes(include='object').columns
        data[s] = data[s].astype("float")
        df_data = data.drop(dcol, axis=1)
        # Compute pearson corelation
        p_test = df_data.corr()
        # Generate mask for upper half
        mask = np.triu(np.ones_like(p_test, dtype=bool))

        # Set up the matplotlib figure
        f, ax = plt.subplots(figsize=(25, 15))

        # Custom color map
        # cmap = sns.diverging_palette(230, 20, as_cmap=True)
        # ax = plt.axes()

        ht_hm = sns.heatmap(p_test, mask=mask, ax=ax, cmap=cmap, annot=show)
        ax.set_title(f'Person correlation {name}', fontsize=20)
        hm_fig = "Pearson_{}.png".format(name)
        ht_hm.figure.savefig(os.path.join(location, hm_fig))
        return name
    return improved_pearsons




