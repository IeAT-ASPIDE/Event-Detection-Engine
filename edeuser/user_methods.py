from sklearn.ensemble import IsolationForest
import pandas as pd
import seaborn as sns
import os
import matplotlib.pyplot as plt


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

