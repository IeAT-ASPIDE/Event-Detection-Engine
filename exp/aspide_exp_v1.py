import sklearn
import itertools
from collections import Counter
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.model_selection import train_test_split, validation_curve, learning_curve, GridSearchCV, RandomizedSearchCV, StratifiedShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
import sklearn.metrics as skm
import numpy as np
from sklearn.utils import resample
# import shap  # TODO: Not working Power9
import time
import sys
import warnings
warnings.filterwarnings("ignore")


def plot_learning_curve(estimator, title, X, y, axes=None, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate 3 plots: the test and training learning curve, the training
    samples vs fit times curve, the fit times vs score curve.

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    axes : array of 3 axes, optional (default=None)
        Axes to use for plotting the curves.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 5-fold cross-validation,
          - integer, to specify the number of folds.
          - :term:`CV splitter`,
          - An iterable yielding (train, test) splits as arrays of indices.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : int or None, optional (default=None)
        Number of jobs to run in parallel.
        ``None`` means 1 unless in a :obj:`joblib.parallel_backend` context.
        ``-1`` means using all processors. See :term:`Glossary <n_jobs>`
        for more details.

    train_sizes : array-like, shape (n_ticks,), dtype float or int
        Relative or absolute numbers of training examples that will be used to
        generate the learning curve. If the dtype is float, it is regarded as a
        fraction of the maximum size of the training set (that is determined
        by the selected validation method), i.e. it has to be within (0, 1].
        Otherwise it is interpreted as absolute sizes of the training sets.
        Note that for classification the number of samples usually have to
        be big enough to contain at least one sample from each class.
        (default: np.linspace(0.1, 1.0, 5))
    """
    if axes is None:
        _, axes = plt.subplots(1, 3, figsize=(20, 5))

    axes[0].set_title(title)
    if ylim is not None:
        axes[0].set_ylim(*ylim)
    axes[0].set_xlabel("Training examples")
    axes[0].set_ylabel("Score")

    train_sizes, train_scores, test_scores, fit_times, _ = \
        learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                       train_sizes=train_sizes,
                       return_times=True)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    fit_times_mean = np.mean(fit_times, axis=1)
    fit_times_std = np.std(fit_times, axis=1)

    # Plot learning curve
    axes[0].grid()
    axes[0].fill_between(train_sizes, train_scores_mean - train_scores_std,
                         train_scores_mean + train_scores_std, alpha=0.1,
                         color="r")
    axes[0].fill_between(train_sizes, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1,
                         color="g")
    axes[0].plot(train_sizes, train_scores_mean, 'o-', color="r",
                 label="Training score")
    axes[0].plot(train_sizes, test_scores_mean, 'o-', color="g",
                 label="Cross-validation score")
    axes[0].legend(loc="best")

    # Plot n_samples vs fit_times
    axes[1].grid()
    axes[1].plot(train_sizes, fit_times_mean, 'o-')
    axes[1].fill_between(train_sizes, fit_times_mean - fit_times_std,
                         fit_times_mean + fit_times_std, alpha=0.1)
    axes[1].set_xlabel("Training examples")
    axes[1].set_ylabel("fit_times")
    axes[1].set_title("Scalability of the model")

    # Plot fit_time vs score
    axes[2].grid()
    axes[2].plot(fit_times_mean, test_scores_mean, 'o-')
    axes[2].fill_between(fit_times_mean, test_scores_mean - test_scores_std,
                         test_scores_mean + test_scores_std, alpha=0.1)
    axes[2].set_xlabel("fit_times")
    axes[2].set_ylabel("Score")
    axes[2].set_title("Performance of the model")

    return plt


def evaluate_exp(y_test,
                 y_pred,
                 binarize=False):
    print("Accuracy = {0}%".format(100 * np.sum(y_pred == y_test) / len(y_test)))
    balanced_acc = skm.balanced_accuracy_score(y_test, y_pred)
    print("Ballanced Accuracy: {}".format(balanced_acc))
    ck = skm.cohen_kappa_score(y_test, y_pred)
    print("Cohen Kappa Coefficient: {}".format(ck))
    mcc = skm.matthews_corrcoef(y_test, y_pred)
    print("Matthews Correlation Coefficient: {}".format(mcc))
    # https://scikit-learn.org/stable/modules/generated/sklearn.metrics.brier_score_loss.html?highlight=metrics#sklearn.metrics.brier_score_loss
    brier = skm.brier_score_loss(y_test, y_pred)
    print("Brier loss: {}".format(brier))
    if binarize:
        # Only for binary classification problems
        roc_auc = skm.roc_auc_score(y_test, y_pred)
        print("ROC AUC Score: {}".format(roc_auc))
        precision, recall, threashold = skm.precision_recall_curve(y_test, y_pred)
        print("Precision {}; Recall {}; Threashold {}".format(precision, recall, threashold))
        cfm = skm.confusion_matrix(y_test, y_pred)
        print(cfm)
        fig, ax = plt.subplots()
        plot_confusion_matrix(cfm, classes=np.unique(y_test), ax=ax,
                              title='Confusion Matrix')
    else:
        cm = skm.multilabel_confusion_matrix(y_test, y_pred)
        print(cm)
    print(skm.classification_report(y_test, y_pred))


def pca_pot(X,
            y,
            fname='antarex',
            n_components=2):

    def plot_2d_space(X,
                      y,
                      fname,
                      label='Classes'):
        colors = ['#1F77B4', '#FF7F0E']
        markers = ['o', 's']
        for l, c, m in zip(np.unique(y), colors, markers):
            plt.scatter(
                X[y == l, 0],
                X[y == l, 1],
                c=c, label=l, marker=m
            )
        plt.title(label)
        plt.legend(loc='upper right')
        plt_name = 'pca_{}.png'.format(fname)
        plt.savefig(fname=plt_name, format='png')
        plt.show()

    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X)

    plot_2d_space(X_pca, y, fname=fname, label='Imbalanced dataset (2 PCA components)')


def plot_confusion_matrix(cm, classes, ax,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print(cm)
    print('')

    # Clear plot before starting
    plt.gca().cla()

    ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.sca(ax)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        ax.text(j, i, format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black")

    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    fname = "{}_cm.png".format("exp")
    plt.savefig(fname, format='png')



def undersample(X,
                y,
                class_type='not majority',
                type=None):
    from imblearn.under_sampling import RandomUnderSampler
    print("Starting undersampling of {}".format(class_type))
    rus = RandomUnderSampler(sampling_strategy=class_type, return_indices=True)
    X_rus, y_rus, id_rus = rus.fit_sample(X, y)

    print('Removed indexes:', id_rus)
    print("Values after undersampling {} are now: {}".format(class_type, Counter(y_rus)))

    return X_rus, y_rus


def minority_undersample(X,
                         y,
                         target='#Fault',
                         n_samples=None,
                         min_label=0,
                         majority_label=1):
    # Concatenate the training data back togheter
    X[target] = y
    # print(X[target])
    X_major = X[X[target] == majority_label]
    X_minority = X[X[target] == min_label]
    if n_samples is None:
        n_samples = int(len(X_minority)/10)
        print("Subsampling minority with n_samples set to: {}".format(n_samples))
    # print(Counter(y))
    down_majority = resample(X_minority, replace=False, n_samples=n_samples)
    # combine minority and downsampled majority
    X_minority_downsampled = pd.concat([down_majority, X_major])
    y_train = X_minority_downsampled[target]
    X_train = X_minority_downsampled.drop(target, axis=1)
    return X_train, y_train


def prep_data(data_file,
              target,
              test_size,
              drop_list,
              binarize=False,
              scale=True,
              under_sample=False,
              custom_under=False,
              pca_plot=False,
              row_filter=None,
              n_components=0):
    print("Started loading data ...")
    df = pd.read_csv(data_file)

    # Fill nan values
    df.fillna(value=0, inplace=True)

    # Filter rows based in column value

    # Classes['None' 'cpufreq_0' 'pagefail_0' 'leak_0' 'ddot_0' 'memeater_0' 'dial_0']
    # df.drop(df[df.score < 50].index, inplace=True)
    if row_filter != None:
        print('Dropping rows: {}'.format(row_filter))
        try:
            for row in row_filter:
                df.drop(df[df[target] == row].index, inplace=True)
        except Exception as inst:
            print("Failed to drop rows with {} and {}".format(type(inst), inst.args))


    # Get target column
    y = df[target]

    # Drop unneeded columns
    df.drop(drop_list, axis=1, inplace=True)
    print("Columns in dataframe: {}.".format(df.columns.values))
    print("Dataframe Shape {}".format(df.shape))
    print("Unique target column values: {}".format(y.unique()))
    print("Unique Value counts:")
    print(y.value_counts())

    if binarize:
        y = np.where(y == 'None', 1, -1)
        print(y)
        print(len(y))
        print('Finished binerizing ground truth')
        y_unique, y_count = np.unique(y)
        print("Values are now: {}".format(Counter(y)))
        print(y_count)
        ano = -1
        norm = 1
        if pca_plot:
            print("Creating PCA for 2D visualization")
            pca_pot(df, y)
    else:
        # Encode target labels
        print("Encoding Labels")
        le = preprocessing.LabelEncoder()
        le.fit(y)
        y = le.transform(y)
        ano = 0
        norm = 1
    if under_sample:
        df, y = undersample(df, y)

    if custom_under:
        if under_sample:
            print("Can't use both undersampling techniques at the same time!")
            sys.exit()
        df, y = minority_undersample(df, pd.Series(y), target=target, min_label=ano, majority_label=norm)
        print("Values are now: {}".format(Counter(y)))

    if scale:
        # Scale the data
        df = StandardScaler().fit_transform(df)

    if n_components:
        print("Running PCA with {}".format(n_components))
        pca = PCA(n_components=n_components)
        df = pca.fit_transform(df)
        print("Data shape is not {}".format(df.shape))
    # Splitting dataset
    print("Splitting dataset, test size {}".format(test_size))
    X_train, X_test, y_train, y_test = train_test_split(df, y, test_size=test_size)
    print("Length of train set: {}".format(len(X_train)))
    print("Unique value for train set {}".format(Counter(y_train)))
    print("Length of test set: {}".format(len(X_test)))
    print("Unique value for test set {}".format(Counter(y_test)))
    print("Data loaded!")
    return X_train, X_test, y_train, y_test, df, y


def validation_curve_experiment(fname,
                                X_train,
                                y_train,
                                method,
                                param_name,
                                param_range,
                                cv,
                                scoring):
    print("#" * 60)
    print("Starting validation curve for {}".format(fname))
    train_scores, test_scores = validation_curve(method,
                                                 X_train,
                                                 y_train,
                                                 param_name=param_name,
                                                 param_range=param_range,
                                                 cv=cv,
                                                 scoring=scoring,
                                                 n_jobs=-1)

    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    print(train_scores)
    print(test_scores)

    plt.title("Validation Curve with {}".format(fname))
    plt.xlabel(r"{}".format(param_name))
    plt.ylabel("Score")
    plt.ylim(0.0, 1.1)
    lw = 2
    plt.semilogx(param_range, train_scores_mean, label="Training score",
                 color="darkorange", lw=lw)
    plt.fill_between(param_range, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.2,
                     color="darkorange", lw=lw)
    plt.semilogx(param_range, test_scores_mean, label="Cross-validation score",
                 color="navy", lw=lw)
    plt.fill_between(param_range, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.2,
                     color="navy", lw=lw)
    plt.legend(loc="best")
    plt_name = 'valid_curve_{}.png'.format(fname)
    plt.savefig(plt_name, format='png')
    plt.show()
    print("#" * 60)


def experiment(X_train,  # TODO
               X_test,
               y_train,
               y_test,
               fname,
               binarize=False):
    print("#"*60)
    print('Started Experiment ..')
    # KNN
    # knn = sklearn.neighbors.KNeighborsClassifier(n_jobs=-1)
    # knn.fit(X_train, y_train)
    # print_accuracy(knn.predict)


    # Random Forest
    rforest = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0, n_jobs=-1)
    rforest.fit(X_train, y_train)
    # print_accuracy(rforest.predict)
    y_pred = rforest.predict(X_test)

    evaluate_exp(y_test, y_pred, binarize)
    try:
        from sklearn.metrics import plot_roc_curve
        rforest_disp = skm.plot_roc_curve(rforest, X_test, y_test)
        plt_name = 'roc_curve_{}.png'.format(fname)
        plt.savefig(plt_name, format='png')
        plt.show()
    except:
        print("Not working, plot_roc_curve!")

    disp = skm.plot_precision_recall_curve(rforest, X_test, y_test)
    average_precision = skm.average_precision_score(y_test, y_pred)
    disp.ax_.set_title('2-class Precision-Recall curve: '
                       'AP={0:0.2f}'.format(average_precision))

    print("#"*60)


def exp_pipeline(method): # TODO
    from imblearn import pipeline as pl
    rforest = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0, n_jobs=-1)
    pipeline = pl.make_pipeline(rforest)


def iso(X_train,
        X_test,
        y_train,
        y_test,
        binarize=True):
    from sklearn.ensemble import IsolationForest
    print("#"*60)
    print("Starting IsolationForest Train")
    clf = IsolationForest(max_samples=100, verbose=1, n_jobs=-1)
    clf.fit(X_train)

    # Predictions
    y_train_predictions = clf.predict(X_train)
    y_test_predictions = clf.predict(X_test)
    evaluate_exp(y_train, y_train_predictions, binarize)
    print("-" * 60)
    evaluate_exp(y_test, y_test_predictions, binarize)
    from sklearn.metrics import roc_curve

    # Clear plot before starting
    plt.gca().cla()

    fpr, tpr, thresholds = roc_curve(y_test, y_test_predictions)
    plt.plot(fpr, tpr, 'k-', lw=2)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.savefig('isolationForest_roc_auc.png', format='png')
    plt.show()
    print("#" * 60)

def iso_learning_curve(expname,
                       X_train,
                       X_test,
                       y_train,
                       y_test,
                       params,
                       n_splits=100,
                       test_size=.2):
    cv = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size)
    from sklearn.ensemble import IsolationForest
    print("#" * 60)
    print("Starting IsolationForest Train (Learning Curve")
    clf = IsolationForest(**params)
    plot_learning_curve(estimator=clf, title=expname, X=X_train, y=None, ylim=(0.7, 1.01), cv=cv, n_jobs=-1)
    fname = "{}_learning_curve.png".format(expname)
    plt.savefig(fname, format='png')
    plt.show()
    print("#" * 60)



def plot_hpo_results(expname,
                     results,
                     column,
                     scoring,
                     xlim=(0, 100),
                     ylim=(0.1, 1.0)
                     ):
    plt.figure(figsize=(13, 13))
    plt.title("HPO evaluating using multiple scorers simultaneously",
              fontsize=16)

    plt.xlabel(column)
    plt.ylabel("Score")

    ax = plt.gca()
    # Set the limits of y and x axis
    ax.set_xlim(xlim[0], xlim[1])
    ax.set_ylim(ylim[0], ylim[1])

    # Get the regular numpy array from the MaskedArray
    X_axis = np.array(results[column].data, dtype=float)

    for scorer, color in zip(sorted(scoring), ['g', 'k']):
        for sample, style in (('train', '--'), ('test', '-')):
            sample_score_mean = results['mean_%s_%s' % (sample, scorer)]
            sample_score_std = results['std_%s_%s' % (sample, scorer)]
            ax.fill_between(X_axis, sample_score_mean - sample_score_std,
                            sample_score_mean + sample_score_std,
                            alpha=0.1 if sample == 'test' else 0, color=color)
            ax.plot(X_axis, sample_score_mean, style, color=color,
                    alpha=1 if sample == 'test' else 0.7,
                    label="%s (%s)" % (scorer, sample))

        best_index = np.nonzero(results['rank_test_%s' % scorer] == 1)[0][0]
        best_score = results['mean_test_%s' % scorer][best_index]

        # Plot a dotted vertical line at the best score for that scorer marked by x
        ax.plot([X_axis[best_index], ] * 2, [0, best_score],
                linestyle='-.', color=color, marker='x', markeredgewidth=3, ms=8)

        # Annotate the best score for that scorer
        ax.annotate("%0.2f" % best_score,
                    (X_axis[best_index], best_score + 0.005))

    plt.legend(loc="best")
    plt.grid(False)
    fname = "hpo_{}.png".format(expname)
    plt.savefig(fname, format='png')
    plt.show()


def hyper_iso(
        expname,
        X_train,
        X_test,
        y_train,
        y_test,
        column,
        param_grid=None,
        scoring=None,
        cv=10,
        n_jobs=-1,
        verbose=1,
        n_iter=None,
        refit=True,
        binarize=True,
        xlim=(0, 100),
        ylim=(0.1, 1.0)):
    print('#'*75)
    print("Starting HPO for Isolation Forest")
    from sklearn.ensemble import IsolationForest
    clf = IsolationForest(random_state=42)
    if param_grid is None:
        param_grid = {
            'n_estimators': [100, 200],
            'max_samples': [5, 20, 100],
            'contamination': [0.4, 0.5],
            'max_features': [5, 10, 15],
            'bootstrap': [True]
        }
    if scoring is None:
        scoring = skm.make_scorer(skm.f1_score, average='micro')
    if n_iter is None:
        hpo_method = 'GridSearch'
        hpo_estimator = GridSearchCV(clf,
                                         param_grid,
                                         scoring=scoring,
                                         refit=refit,
                                         cv=cv,
                                         return_train_score=True,
                                         n_jobs=n_jobs,
                                         verbose=verbose)
    else:
        hpo_method = 'RandomSearch'
        hpo_estimator = RandomizedSearchCV(clf,
                                         param_grid,
                                         scoring=scoring,
                                         refit=refit,
                                         cv=cv,
                                         return_train_score=True,
                                         n_jobs=n_jobs,
                                         verbose=verbose,
                                         n_iter=n_iter)
    print("Using HPO method: {}".format(hpo_method))
    giso = hpo_estimator.fit(X_train, y_train)
    df_cv_results = pd.DataFrame(giso.cv_results_)
    fname = '{}.csv'.format(expname)
    df_cv_results.to_csv(fname)
    best_iso = giso.best_estimator_
    print("Best Score: {}".format(giso.best_score_))
    print("Best paramters: {}".format(giso.best_params_))

    # Predictions
    y_train_predictions = best_iso.predict(X_train)
    y_test_predictions = best_iso.predict(X_test)
    evaluate_exp(y_train, y_train_predictions, binarize)
    print("-" * 60)
    evaluate_exp(y_test, y_test_predictions, binarize)

    print("Plotting result")
    plot_hpo_results(expname,
                     giso.cv_results_,
                     column=column,
                     scoring=scoring,
                     xlim=xlim,
                     ylim=ylim)

    print('#' * 75)


if __name__ == '__main__':
    # Parameters
    data_file = 'processed.csv'
    target = '#Fault'
    test_size = 0.2
    drop_list = ['#Benchmark', '#Fault', '#Time', 'Time_usec', 'Time_usec.1', 'Time_usec.3', 'Time_usec.2', 'Time_usec.4']
    row_filter = ['cpufreq_0', 'pagefail_0', 'leak_0', 'ddot_0', 'memeater_0']  #  Set to None to do nothing
    # row_filter = None
    binarize = True
    under_sample = True  # Must be false if custom_under is true
    custom_under = False
    pca_plot = False
    fname = 'randomforest'
    n_components = 3


    #TODO: Plot learning curve for Isolation forest, plot roc curve for isolation forest,
    # isolation forest plot_precision_recall_curve, plot ROC with Cross valid, Visualizing cross-validation behavior in scikit-learn
    # https://scikit-learn.org/stable/auto_examples/calibration/plot_calibration_curve.html#sphx-glr-auto-examples-calibration-plot-calibration-curve-py

    X_train, X_test, y_train, y_test, df, y_encoded = prep_data(data_file=data_file, target=target, test_size=test_size,
                                                                drop_list=drop_list, binarize=binarize, under_sample=under_sample,
                                                                custom_under=custom_under, pca_plot=pca_plot, row_filter=row_filter,
                                                                n_components=n_components)
    # experiment(X_train, X_test, y_train, y_test, fname=fname, binarize=binarize)




    # Test valid_cirve_experiment
    # fname = 'validation_curve_randomforest'
    # scorer = skm.make_scorer(skm.cohen_kappa_score) # scorer = 'accuracy
    # method = RandomForestClassifier(n_estimators=100, max_depth=None, min_samples_split=2, random_state=0, n_jobs=-1, verbose=1)
    # param_range = [5, 25, 50, 100]  # param_range = np.logspace(-6, -1, 5)
    #
    # validation_curve_experiment(fname, df, y_encoded, method, param_name="n_estimators", param_range=param_range,
    # cv=2, scoring=scorer)

    # IsolationForest Test
    # iso(X_train,
    #     X_test,
    #     y_train,
    #     y_test)



    # HPO Isolation forest
    expname = 'HPo_Iso_PCA'
    # scoring = {'AUC': 'roc_auc',
    #            'Accuracy': skm.make_scorer(skm.accuracy_score),
    #            # 'Im_f1': skm.make_scorer(skm.f1_score, average='micro')
    #            }
    # scoring = {'AUC': 'roc_auc', 'Accuracy': skm.make_scorer(skm.accuracy_score)}
    scoring = {'AUC': 'roc_auc', 'BAccuracy': skm.make_scorer(skm.balanced_accuracy_score), 'Accuracy': skm.make_scorer(skm.accuracy_score)}
    refit = 'AUC'
    column = 'param_max_samples'

    # Set visualization limit for HPO
    # NOTE: Check with HPO max values and column value, if the parameter from the column
    # value is big increase maximum for of xlim accordingly.
    xlim = (0, 100)
    ylim = (0.1, 1.0)

    param_grid = {
        'n_estimators': [200, 220, 250, 300],
        'max_samples': [5, 10, 15, 20, 100, 250],
        'contamination': [0.01, 0.02, 0.03, 0.05],
        'max_features': [1, 2, 3],
        'bootstrap': [True]
    }

    hyper_iso(expname,
        X_train,
        X_test,
        y_train,
        y_test,
        column=column,
        param_grid=param_grid,
        scoring=scoring,
        cv=2,
        n_jobs=-1,
        verbose=1,
        n_iter=None,
        binarize=True,
        refit=refit,
        xlim=xlim,
        ylim=ylim
    )