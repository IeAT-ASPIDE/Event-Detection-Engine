# Event-Detection-Engine - H2020 SERRANO

Event Detection Engine  based on the work done during the DICE H2020 project specifically the [DICE Anomaly Detection Platform.](https://github.com/dice-project/DICE-Anomaly-Detection-Tool)
 for the [SERRANO](https://ict-serrano.eu/) H2020 project.

## Context

In the following section we will use the term events and anomalies seemingly interchangeably. 
However, we should note that the methods used for detecting anomalies are applicable in the case of events. 
The main difference lies in the fact that anomalies pose an additional level of complexity by their spars nature, 
some anomalies might have an occurrence rate well under 0.01%.
Event and anomaly detection can be split up into several categories based on the methods and the characteristics of 
the available data. The most simple form of anomalies are point anomalies which can be characterized by only one metric (feature). 
These types of anomalies are fairly easy to detect by applying simple rules (i.e. CPU is above 70%). Other types of anomalies 
are more complex but ultimately yield a much deeper understanding about the inner workings of a monitored exascale system or application. 
These types of anomalies are fairly common in complex systems.

Contextual anomalies are extremely interesting in the case of complex systems. These types of anomalies happen when a 
certain constellation of feature values are encountered. In isolation these values are not anomalous but when viewed in 
context they represent an anomaly. These type of anomalies represent application bottlenecks, imminent hardware failure 
or software miss-configuration. The last major type of anomaly which are relevant are temporal or sometimes sequential 
anomalies where a certain event takes place out of order or at the incorrect time. These types of anomalies are very 
important in systems which have a strong spatio-temporal relationship between features, which is very much the case for exascale metrics.


## Architecture

The Event detection engine (**EDE**) has several sub-components which are based on lambda type architecture where we have a 
_speed_, _batch_ and _serving_ layer. Because of the heterogeneous nature of most modern computing systems (including exascale and mesh networks) 
and the substantial variety of solutions which could constitute a monitoring services the _data ingestion component_ has to be 
able to contend with fetching data from a plethora of systems. _Connectors_ is implemented such that it  serves as adapters for each solution. 
Furthermore, this component also is be able to load data directly from static file (_HDF5_, _CSV_ , _JSON_, or even _raw format_). 

![EDE Architecture](https://github.com/IeAT-ASPIDE/Event-Detection-Engine/blob/master/architecture/ASPIDE-EDE-Page-2.png)

This aids in fine tuning of event and anomaly detection methods. We can also see that _data ingestion can be done directly_ 
via query from the monitoring solution or _streamed directly from the queuing service_ (after ETL if necessary). 
This ensures that we have the best chance of reducing the time between the event or anomaly happening and it being detected.

The _pre-processing component_ is in charge of taking the raw data from the data ingestion component and apply several transformations. 
It handles _data formatting_ (i.e. one-hot encoding), _analysis_ (i.e. statistical information), _splitter_ (i.e. splitting the 
data into training and validation sets) and finally _augmentation_ (i.e. oversampling and undersampling). 

As an example the analysis and splitter are responsible for creating stratified shuffle split for K-fold cross
validation for training while the augmentation step might involve under or oversampling techniques such as ADASYN or SMOTE. 
This component is also responsible for any feature engineering of the incoming monitoring data.

The _training component_ (batch layer) is used to instantiate and train methods that can be used for event and anomaly detection. 
The end user is able to configure the hyper-parameters of the selected models as well as run automatic optimization on these (i.e. Random Search, Bayesian search etc.). 
Users are not only able to set the parameters to be optimized but to define the objectives of the optimization. 
More specifically users can define what should be optimized including but not limited to predictive performance, 
_transprecise_ objectives (inference time, computational limitations, model size etc.).

_Evaluation_ of the created predictive model on a holdout set is also handled in this component. 
Current research and rankings of machine learning competitions show that creating an _ensemble_ of 
different methods may yield statistically better results than single model predictions. Because of this 
ensembling capabilities have to be included.
 
Finally, the trained and validated models have to be saved in such a way that enables them to be easily instantiated and used in a production environment. 
Several predictive model formats have to be supported, such as; PMML, ONNX, HDF5, JSON.

It is important to note at this time that the task of event and anomaly detection can be broadly split into two main types of machine learning tasks; 
classification and clustering. Classification methods such as Random Forest, Gradient Boosting, Decision Trees, Naive Bayes, Neural networks, Deep Neural Networks 
are widely use in the field of anomaly and event detection. While in the case of clustering we have methods such as IsolationForest, DBSCAN and Spectral Clustering.
Once a predictive model is trained and validated it is saved inside a model repository. Each saved model has to have 
metadata attached to it denoting its performance on the holdout set as well as other relevant information such as size, throughput etc.
 
The _prediction component_ (speed layer) is in charge of retrieving the predictive model form the model repository and feed metrics from the monitored  system. 
If and when an event or anomaly is detected EDE is responsible with signaling this to both the Monitoring service reporting component and to other tools such as the 
Resource manager and/or scheduler any decision support system. Figure 1 also shows the fact that the prediction component gets it’s data from both 
the monitoring service via direct query or directly from the queuing service via the data ingestion component.

For some situations a rule based approach is better suited. For these circumstances the prediction component has to include a rule based engine and a rule repository.
Naturally, detection of anomalies or any other events is of little practical significance if there is no way of handling them. 
There needs to be a component which once the event has been identified tries to resolve the underlying issues. 

## Utilization

EDE is designed around the utilization of a yaml based configuration scheme. This allows the complete configuration of the tool by the end user with limited to no intervention in the source code.
It should be mentioned that some of these features are considered unsave as they allow the execution of arbitrary code.  
The configuration file is split up into several categories:
* **Connector** - Deals with connection to the data sources
* **Mode** - Selects the mode of operation for EDE
* **Filter** - Used for applying filtering on the data
* **Augmentation** - User defined augmentations on the data
* **Training** - Settings for training of the selected predictive models
* **Detect** - Settings for the detection using a pre-trained predictive model
* **Point** - Settings for point anomaly detection
* **Misc** - Miscellaneous settings

### Connector

The current version of EDE support 3 types of data sources: _ElasticSearch_, _Prometheus_ and _CSV/Excel_. Conversely it supports also reporting mechanisms for ElasticSearch and Kafka.
In the former case, a new index is created in ElasticSearch which contains the detected anomalies while in the latter a new Kafka topic is created where the detected anomalies are pushed.

This sections parameters are:
* _PREndpoint_ - Endpoint for fetching Prometheus data
* _ESEndpoint_ - Endpoint for fetching ElasticSearch data
* _MPort_ - Sets the monitoring port for the selected Endpoint (defaults to 9200)
* _KafkaEndpoint_ - Endpoint for a pre-existing Kafka deployment
* _KafkaPort_ - Sets the Kafka port for the selected Kafka Endpoint (defaults to 9092)
* _KafkaTopic_ - Name of the kafka topic to be used
* _GrafanaUrl_ - Endpoint for Grafana used to report detected anomalies as annotations.
* _GrafanaToken_ - User token used for authentication
* _GrafanaTag_ - Tag used to identify dashboard to mark annotations. If not found a default dash will be created with this tag.
* _Query_ - The query string to be used for fetching data:
    * In the case of ElasticSearch please consult the official [documentation](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-query-string-query.html).
    * In the case of Prometheus please consult the official [documentation](https://prometheus.io/docs/prometheus/latest/querying/basics/)
        * For fetching all queryable data:  `{"query": '{__name__=~"node.+"}[1m]'}`
        * For fetching specific metric data: `{ "query": 'node_disk_written_bytes_total[1m]'}`
* _MetricsInterval_ - Metrics datapoint interval definition
* _QSize_ -  size in MB of the data to be feteched (only if ESEndpoint is used)
    * For no limit use `QSize: 0`
* _Index_ - The name of the column to be set as index
    * The column has to have unique values, by default it is set to the column denoting the time when the metric was read
* _QDelay_ - Polling period for metrics fetching
* _Dask_
    * _ScheduelerEndpoint_ - Denotes the Dask scheduler endpoint
        * If no pre-deployed Dask instance is available EDE can deploy a local Scheduler by setting this parameter to `local`
    * _SchedulerPort_ -  Endpoint for Dask scheduler endpoint
    * _Scale_ - Sets the number of workers if `local` scheduler is used
    * _EnforceCheck_ - if set to true it will check if the libraries from the python environment used on each Dask worker are the same versions as the origination source
        * If this check fails the job will exit with an error message
        * This parameter can be omitted in the case of local deployment
* _Local_ - path to csv or Excel file to be used
     

**Notes**: 
* Only one of type of connector endpoint (PREndpoint or ESEndpoint) is supported at any given time.
* If Local is defined than it will ignore any other data sources.

### Mode

The following settings set the mode in which EDE operates. There are 3 modes available in this version; _Training_, _Validate_, _Detect_

* _Training_ - If set to true a Dask worker or Python process for training is started
* _Validate_  - If set to true a Dask worker or Python process for validation is started
* _Detect_ - If set to true a Dask worker for Python process fof Detection is started

**Notes:**
* In case of a local Dask deployment it is advised to have at least 3 workers started (see the _Scale_ parameter in the previouse section). 

### Filter
Once the data has been loaded by the EDE connector it is tranformed into DataFrames. The data in these Dataframes can be filtered by using the
parameters listed bellow:

* __Columns__ - listing of columns which are to remain
* __Rows__ - Filters rows in a given bound
    * __gd__ - Lower bound
    * __ld__ - Upper bound
* __DColumns__ - list of columns to be deleted (dropped)
  * __Dlist__ - expects an external yaml file containing a list of columns to be dropped (usefull removing large number of features)
* __Fillna__ - fills `None` values with `0`
* __Dropna__ - deletes columns wiith `None` values
* __LowVariance__ - used to detect and remove low variance features automatically
* __DWild__ - Removes columns based on regex
    * __Regex__ - Regex to be used for fitlering
    * __Keep__ - If `True` all selected columns are kept the rest are dropped, otherwise selected columns are dropped.
* __CoreMetrics__ - Yaml descriptor containing common core metrics (i.e. features). This is important as the number of metrics can differ from training vs prediction for several reasons.
  * If set to `True` then default name is used. If file with yaml extension is given it will check for that.
  

**Notes:**
* Some machine learning models cannot deal with `None` values to this end the __Fillna__ or __Dropna__ parameters where introduced. It is important to note
the __Dropna__ will drop any column which has at least one `None` value.
* If __CoreMetrics__ is set during prediction EDE will try to load the descriptor and apply it. During training it will be created.
* Although __CoreMetrics__ is set as a filter it is applied during dataframe creation not after it. As a mismatch in metric/feature names will raise an index out of bounds exception.

### Augmentation

The following parameters are used to define augmentations to be executed on the loaded dataframes. The augmentations are chained togheter as defined by the user.
The available parameters are:

* __Scaler__ - Scaling/Normalizing the data. If not defined no scaler is used
    * __ScalerType__ - We currently support all scaler types from scikitlearn. Please consult the official [documentation](https://scikit-learn.org/stable/modules/preprocessing.html) for further details
        * When utilizing a scikit-learn scaler we have to use the exact name of the scaler followed by its parameters. Bellow you can find an exampele utilizing the [StandardScaler](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html#sklearn.preprocessing.StandardScaler):
             ```yaml 
             Scaler:
                StandardScaler:
                    copy: True
                    with_mean: True
                    with_std: True
            ```
* __Operations__ - set of predefined operations that can be executed
    * __STD__ - calculates the standard deviation, excepts a name and a list of metrics to use
    * __Mean__ - Calculates the mean, excepts a name and a list of metrics to use
    * __Median__ - Calculates the median, excepts a name and a list of metrics to use
    * Example usage:
    ```yaml
    Operations:
        STD:
          - cpu_load1:
              - node_load1_10.211.55.101:9100
              - node_load1_10.211.55.102:9100
              - node_load1_10.211.55.103:9100
          - memory:
              - node_memory_Active_anon_bytes_10.211.55.101:9100
              - node_memory_Active_anon_bytes_10.211.55.101:9100
              - node_memory_Active_anon_bytes_10.211.55.101:9100
        Mean:
          - network_flags:
              - node_network_flags_10.211.55.101:9100
              - node_network_flags_10.211.55.102:9100
              - node_network_flags_10.211.55.103:9100
          - network_out:
              - node_network_mtu_bytes_10.211.55.101:9100
              - node_network_mtu_bytes_10.211.55.102:9100
              - node_network_mtu_bytes_10.211.55.103:9100
        Median:
          - memory_file:
              - node_memory_Active_file_bytes_10.211.55.101:9100
              - node_memory_Active_file_bytes_10.211.55.102:9100
              - node_memory_Active_file_bytes_10.211.55.103:9100
          - memory_buffered:
              - node_memory_Buffers_bytes_10.211.55.101:9100
              - node_memory_Buffers_bytes_10.211.55.102:9100
              - node_memory_Buffers_bytes_10.211.55.103:9100
    ```
    * __RemoveFiltered__ - If set to `True` the metrics used during these operations will be deleted, the resulting augmented columns remaining
    * __Method__ - Excepts User defined augmentations (i.e. python functions) for feature engineering
        * Methods should be wrapped as can bee seen in the [wrapper_add_column](https://github.com/DIPET-UVT/EDE-Dipet/blob/master/edeuser/user_methods.py#L44) example
        * All keyword arguments should be passable to the wrapped function
        * Here is an example of a user defined method invocation:
            ```
            Method: !!python/object/apply:edeuser.user_methods.wrapper_add_columns # user defined operation
              kwds:
                columns: !!python/tuple [node_load15_10.211.55.101:9100, node_load15_10.211.55.102:9100]
                column_name: sum_load15
            ```
    * __Categorical__ - Excepts a list of categorical columns, if not defined EDE can try to automatically detect categorical columns
        * __OH__ - If set to True oneHot encoding is used for categorical features

### Training

The following parameters are used to set up training mode and machine learning model selection and initialization.
* __Type__ - Sets the type of machine learning problem. Currently supported are: __clustering__, __classification__, __hpo__ and __tpot__.
* __Method__ - Sets the machine learning method to be used. We support all acikit-learn based models as well as other machine learning libraries which
support scikit-learn API conventions such as: Tensorflow, Keras, LightGBM, XGBoost, CatBoost etc.
* __Export__ - Name of the preictive model  to be exported (serialized)
* __MethodSettings__ - Setting dependant on machine learning method selected.
* __Target__ - Denotes the ground truth column name to be used. This is mandatory in the case of classification. If no `target` is defined the last column is used instead.

In the case of classification we have several additional options we can select:

* __Verbose__ - Will save a full classification report, confusion matrix, feature importance (if applicable) for all folds 
* __PrecisionRecallCurve__ - Will plot the Precision Recall curve of the selected model
* __ROCAUC__ - Will plot the ROCAUC for the selected model
* __RFE__ - Will execute and plot recursive feature elimination. It will save a yaml file containing a list of features to be eliminated, usable by _DList_ from __DColumn__.
    * _scorer_ - Defines the scorer to be used
    * _step_ - Defines the step for feature elimination. If there are a lot of features in the data this can take a long time to execute. In this case a larger step function is advised.
* __DecisionBoundary__ - Will plot the decision boundary after executing _PCA_ with 2 components. For large number of classes the process of dimensionality reduction can result in noisy plots.
* __LearningCurve__ - Shows the relationship between model preformance and the amount of features/ training samples
  *_sizes_ - Used to define the training samples for ploting, can be list or use generator function as seen in the example bellow.
  *_scorer_ - Scorer to be used
  *_n_jobs_ - Number of jobs to be executed, if Dask backend is used it will handle schedueling of these jobs.
* __ValidationCurve__ - Used to finetune a specific parameter, checking  out of sample performance
    * _param_name_ - Name of Hyper-parameter to be optimized
    * _param_range_ - Range of values to check (list can be generated using generator functions same as for __LearningCurve__).
    * _scorer_ - Scorer to be used
    * _n_jobs_ - Number of jobs to be executed, if Dask backend is used it will handle schedueling of these jobs.
    
__Note__: When training an unsupervised method, by default it will generate a decision boundary and feature separation plots for the selected Model.



Example for clustering:

```yaml
# Clustering example
Training:
  Type: clustering
  Method: isoforest
  Export: clustering_1
  MethodSettings:
    n_estimators: 10
    max_samples: 10
    contamination: 0.1
    verbose: True
    bootstrap: True
```

Example for classification:
```yaml
Training:
  Type: classification
  Method: randomforest
  Export: classifier_1
  MethodSettings:
    n_estimators: 10
    max_samples: 10
    verbose: True
    bootstrap: True
  Target: target
  LearningCurve:
    sizes: !!python/object/apply:numpy.core.function_base.linspace
      kwds:
        start: 0.3
        stop: 1.0
        num: 10
    scorer: f1_weighted
    n_jobs: 5
  ValidationCurve:
    param_name: n_estimators
    param_range:
    - 10
    - 20
    - 60
    - 100
    - 200
    - 600
    scoring: f1_weighted
    n_jobs: 8
  PrecisionRecallCurve: 1
  ROCAUC: 1
  RFE:
    scorer: f1_weighted
    step: 10
  DecisionBoundary: 1
  Verbose: 1
```

Similar to how users can add their own implementations for augmentations it is also possible to add custom machine learning
methods. An example implementation can be found in the edeuser [section](https://github.com/DIPET-UVT/EDE-Dipet/tree/master/edeuser), namely [user_iso](https://github.com/DIPET-UVT/EDE-Dipet/blob/6d92fe4203053b6a6cab294553b81e87bf6ba11d/edeuser/user_methods.py#L8).
The wrapper function should contain all parameters which are necessary for the defined method and the return value should be an object which abides by scikit-learn API conventions.  
Example for user defined method:
```yaml
# User defined clustering custom
Training:
  Type: clustering
  Method: !!python/object/apply:edeuser.user_methods.user_iso
    kwds:
      n_estimators: 100
      contamination: auto
      max_features: 1
      n_jobs: 2
      warm_start: False
      random_state: 45
      bootstrap: True
      verbose: True
      max_samples: 1
  Export: clustering_2
```


#### Cross Validation

EDE supports a variety of cross validation methods ([all from scikit-learn](https://scikit-learn.org/stable/modules/cross_validation.html)). The parameters are as follows:

* __CV__ - If an integer is used then standard (scikit-learn) CV is used with the integer representing the number of folds.
    * __Type__ - Type is required if __CV__ denotes a scikit-learn or user defined CV method is used.
    * __Params__ - Parameters for CV method

For defining simple __CV__ with 5 folds:

`CV: 5`

For defining __CV__ using a  specific method such as [StratifiedKFold](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html?highlight=stratifiedkfold#sklearn.model_selection.StratifiedKFold):
```yaml
CV:
    Type: StratifiedKFold  # user defined all from sklearn
    Params:
      n_splits: 5
      shuffle: True
      random_state: 5
```


#### Scoring

EDE supports the inclusion of different scoring methods. These are all [scikit-learn scoring](https://scikit-learn.org/stable/modules/model_evaluation.html) methods as well as user defined scoring methods.
The scoring functions are defined using the following parameters:

* __Scorers__
    * __Scorers_list__ - List of scoreres to be used
        * __Scorer__
            * __Scorer_name__ - User defined name of the scorer
            * __skScorer__ - Scikit-learn scorer name
In the case of user defined scorers the user has to define a key value pair; `scorer_name` and `scorer instance`.

An example Scorer chain definition can be found here:

```yaml
Scorers:
    Scorer_list:
      - Scorer:
          Scorer_name: F1_weighted
          skScorer: f1_weighted
      - Scorer:
          Scorer_name: Jaccard_Index
          skScorer: jaccard_weighted # changes in scoring sklearn, for multiclass add suffix micro, weighted or sample
      - Scorer:
          Scorer_name: AUC
          skScorer: roc_auc_ovr_weighted
    User_scorer1: balanced_accuracy_score # key is user defined, can be changed same as Scorer_name
``` 
 

#### Hyper-parameter optimization

EDE also supports hyper parameter optimization methods such as: _grid_ and _random_ search, _bayesian_ and _evolutionary_ search and the _tpot_ framework. The following parameters are used for HPO:

* __HPOMethod__ - Name of the hyper parameter optimization to be used
* __HPOParam__ - HPO parameters:
    * __n_iter__ - Number of iterations. In case of Grid search this is ignored.
    * __n_jobs__ - number of threads (`-1` for all available)
    * __refit__ - Name of scoring metric to be used to determine the best performing hyperparameters. If multi metric is used, refit should be a metric name (mandatory)
    * __verbose__ - If set to true, it outputs metrics about each iteration
* __ParamDistribution__ - should contain a dictionary which has as a key the parameter name and a list (or python code which generates a list according to a particular distribution) which should be used:
    ```yaml
    ParamDistribution:
        n_estimators:
          - 10
          - 100
        max_depth:
          - 2
          - 3
    ```  

Example of HPO including CV and Scorers examples:

```yaml
# For HPO methods
Training:
  Type: hpo
  HPOMethod: Random  # random, grid, bayesian, tpot
  HPOParam:
    n_iter: 2
    n_jobs: -1
    refit: Balanced_Acc  # if multi metric used, refit should be metric name, mandatory
    verbose: True
  Method: randomforest
  ParamDistribution:
    n_estimators:
      - 10
      - 100
    max_depth:
      - 2
      - 3
  Target: target
  Export: hpo_1
  CV:
    Type: StratifiedKFold  # user defined all from sklearn
    Params:
      n_splits: 5
      shuffle: True
      random_state: 5
  Scorers:
    Scorer_list:
      - Scorer:
          Scorer_name: AUC
          skScorer: roc_auc
      - Scorer:
          Scorer_name: Jaccard_Index
          skScorer: jaccard
      - Scorer:
          Scorer_name: Balanced_Acc
          skScorer: balanced_accuracy
    User_scorer1: f1_score # key is user defined, can be changed same as Scorer_name
```

Example of HPO using the evolutionary search method:
```yaml
Training:
  Type: hpo
  HPOMethod: Evol  # Random, Grid, Bayesian, tpot, Evol
  HPOParam:
    n_jobs: 1 # must be number, not -1 for all in case of Evol
    scoring: f1_weighted
    gene_mutation_prob: 0.20
    gene_crossover_prob: 0.5
    tournament_size: 4
    generations_number: 30
    population_size: 40  # if multi metric used, refit should be metric name, mandatory
    verbose: 4
  Method: randomforest
  ParamDistribution:
    n_estimators:
      - 10
      - 100
    max_depth:
      - 2
      - 3
  Target: target
  Export: hpo_1_y2
  CV:
    Type: StratifiedKFold  # user defined all from sklearn
    Params:
      n_splits: 5
      shuffle: True
      random_state: 5
  Scorers:
    Scorer_list:
      - Scorer:
          Scorer_name: F1_weighted
          skScorer: f1_weighted
      - Scorer:
          Scorer_name: Jaccard_Index
          skScorer: jaccard_weighted # changes in scoring sklearn, for multiclass add suffix micro, weighted or sample
      - Scorer:
          Scorer_name: AUC
          skScorer: roc_auc_ovr_weighted
    User_scorer1: balanced_accuracy_score
```
#### TPOT

[TPOT](http://epistasislab.github.io/tpot/) is an automated machine learning framework designed around scikit-learn (and nay other framework which conforms to the scikit-learn API conventions). In contrast to other such tools it does not focus solely on the hyper-parameters
of machine learning models but it tries to optimize the pre and postprocessing methods as well. It does this by generating scikit-learn [pipelines](https://scikit-learn.org/stable/modules/generated/sklearn.pipeline.Pipeline.html).
The optimizations are based around a Genetic Programming stochastic global search procedure. TPOT parameters are as follows:

* __TPOTParam__
    * __generations__ - Number of generations to run
    * __population_size__ - Size of the population (candidate configurations)
    * __offspring_size__ - Number of new members added to a population between each generation
    * __mutation_rate__ - Rate of mutation to be used by the genetic algorithm
    * __crossover_rate__ - Value used in defining the crossover used when generating new offsprings
    * __scoring__ - Scoring functon to be used, it is different than scikit-learn, see TPOT documentation for details about [TPOT scoring](http://epistasislab.github.io/tpot/using/#scoring-functions).
    * __max_time_mins__ - Limits the time for computing a generation
    * __max_eval_time_mins__ - Limits the amount of time for evaluating a pipeline.
    * __random_state__ - Random seed so that it enables consistency between experiments.
    * __n_Jobs__ - Number of concurent jobs. If set to `-1` it will enable unlimited number of potential jobs
    * __verbosity__ - Logging detail
    * __config_dict__ -  This sets the amount of detail and methods to add to the pipelines from a population. Possible values are:
        * Default - Includes all scikit-learn methods
        * TPOT light - restricted range of methods
        * TPTO MDR - Extended feature selectors and Multi dimensional reduction models
    * __use_dask__ - Use Dask backend to run phenotypes from the population (i.e each Dask worker runs a phenotype)
    
Example of TPOT

```yaml
# TPOT Optimizer
Training:
  Type: tpot
  TPOTParam:
    generations: 2
    population_size: 2
    offspring_size: 2
    mutation_rate: 0.9
    crossover_rate: 0.1
    scoring: balanced_accuracy # Scoring different from HPO check TPOT documentation
    max_time_mins: 1
    max_eval_time_mins: 5
    random_state: 42
    n_jobs: -1
    verbosity: 2
    config_dict: TPOT light # "TPOT light", "TPOT MDR", "TPOT sparse" or None
    use_dask: True
  Target: target
  Export: tpotopt
  CV:
    Type: StratifiedKFold  # user defined all from sklearn
    Params:
      n_splits: 5
      shuffle: True
      random_state: 5
```

**Notes:**
* Both HPO and TPOT are heavily based around Dask and utilize Dask workers for running different hyper parameter configurations. Because of this it is recommended to utilise a pre-existent distributed Dask worker cluster.
* In contrast to other methods TPOT return the entire pipeline not just the predictive mode.

### Prediction

Prediction is largely unchanged between the various EDE modes. Its parameters are:

* __Method__ - Name of the predictive model type used
* __Type__ - Specifies what type the model is (i.e. clustering, classsification, tpot etc.)
* __Load__ - Name of the serialized predictive model to be instantiated. See the export from training.
* __Scaler__ - Name of the scaler (if used). Once the scaler has been invoced during training the result will be serialized by EDE and can be reused for prediction.
* __Analysis__ - Will attach root cause analysis in the form of computed Shapely values and feature importance for all detected anomalous instances.
    * _Plot_- If set to `True` it will generate plots for each detected anomalous instance;
        * _Clustering_: feature importance, summary and heatmap
        * _Classification_: force, summary
    
Example of a prediction:

```yaml
Detect:
  Method: isoforest
  Type: clustering
  Load: clustering_1
  Scaler: StandardScaler  # Same as for training
  #Analysis: True
  Analysis: # if plotting of heatmap, summary and feature importance is require, if not set False or use previous example
    Plot: True
```

### Analysis

EDE is capable of running any user defined analysis methods on the data. Users can add data exploration methods. Its parameters are:

* __Analysis__
    * __Methods__ - List of methods to be used
        * __Method__ - Information required for instantiation of user methods (including keyword arguments)
    * __Solo__ - If it is set to true it will run only the analysis and ignore any other Training or Prediction tasks.

Example analysis implementations included in EDE are [Pearson correlation](https://github.com/DIPET-UVT/EDE-Dipet/blob/80efea55545fefb7ba6d71f4e0f18fc962a16ef5/edeuser/user_methods.py#L64) and a [Line plot](https://github.com/DIPET-UVT/EDE-Dipet/blob/80efea55545fefb7ba6d71f4e0f18fc962a16ef5/edeuser/user_methods.py#L98):

```yaml
# Analysis example
Analysis:
 Methods:
   - Method: !!python/object/apply:edeuser.user_methods.wrapper_analysis_corr
       kwds:
         name: Pearson1
         annot: False
         cmap: RdBu_r
         columns:
           - node_load1_10.211.55.101:9100
           - node_load1_10.211.55.102:9100
           - node_load1_10.211.55.103:9100
           - node_memory_Cached_bytes_10.211.55.101:9100
           - node_memory_Cached_bytes_10.211.55.102:9100
           - node_memory_Cached_bytes_10.211.55.103:9100
           - time
         location: /Users/Gabriel/Documents/workspaces/Event-Detection-Engine/edeuser/analysis
   - Method: !!python/object/apply:edeuser.user_methods.wrapper_analysis_plot
       kwds:
         name: line1
         columns:
           - node_load1_10.211.55.101:9100
           - node_load1_10.211.55.102:9100
           - node_load1_10.211.55.103:9100
           - time
         location: /Users/Gabriel/Documents/workspaces/Event-Detection-Engine/edeuser/analysis
   - Method: !!python/object/apply:edeuser.user_methods.wrapper_improved_pearson
       kwds:
         name: Test_Training
         dcol:
           - target
         location: /Users/Gabriel/Documents/workspaces/Event-Detection-Engine/edeuser/analysis
         show: False
   - Method: !!python/object/apply:edeuser.user_methods.wrapper_rank2
       kwds:
         name: Test_rank
         dcol:
           - target
         location: /Users/Gabriel/Documents/workspaces/Event-Detection-Engine/edeuser/analysis
         algorithm: spearman
         show: False
   - Method: !!python/object/apply:edeuser.user_methods.wrapper_rank1
       kwds:
         name: Test_rank1
         dcol:
           - target
         location: /Users/Gabriel/Documents/workspaces/Event-Detection-Engine/edeuser/analysis
         algorithm: shapiro
   - Method: !!python/object/apply:edeuser.user_methods.wrapper_pca_plot
       kwds:
         name: Test_PCA
         location: /Users/Gabriel/Documents/workspaces/Event-Detection-Engine/edeuser/analysis
         projection: 3
         target: target
#         show: False
   - Method: !!python/object/apply:edeuser.user_methods.wrapper_manifold
       kwds:
         name: Test_manifold
         target: target
         location: /Users/Gabriel/Documents/workspaces/Event-Detection-Engine/edeuser/analysis
         manifold: tsne
         n_neighbors: 10
   - Method: !!python/object/apply:edeuser.user_methods.wrapper_manifold
       kwds:
         name: Test_manifold
         target: target
         location: /Users/Gabriel/Documents/workspaces/Event-Detection-Engine/edeuser/analysis
         manifold: hessian
   - Method: !!python/object/apply:edeuser.user_methods.wrapper_plot_on_features
       kwds:
         name: complete_columns
         target: target
         location: /Users/Gabriel/Documents/workspaces/Event-Detection-Engine/edeuser/analysis

 Solo: True
```


### Point

```yaml
Point:
  Memory:
    cached:
      gd: 231313
      ld: 312334
    buffered:
      gd: 231313
      ld: 312334
    used:
      gd: 231313
      ld: 312334
  Load:
    shortterm:
      gd: 231313
      ld: 312334
    midterm:
      gd: 231313
      ld: 312334
  Network:
    tx:
      gd: 231313
      ld: 312334
    rx:
      gd: 231313
      ld: 312334
```

### Misc
mMiscellaneous settings:
* __heap__: Size of JVM heap used for weka based methods (now deprecated, will be removed in next version)
* __checkpoint__: All filtering, augmentation steps will can be set to save to disk their results, thus in case of faliure processing can be resumed from the last step succesfully executed from the EDE processing pipeline.
* __delay__ : Used to set how often new data is to be fetched from the datasource. Same as __QDelay__'
* __interval__: Query interval to be used when generating query strings. It will be ignored if user defined query string is used.
* __resetindex__: Deletes anomaly index in case ElasticSearch is used for reporting.
* __point__: Toggles point anomaly execution (now deprecated, will be removed in next version)

```yaml
Misc:
  heap: 512m
  checkpoint: True
  delay: 15s
  interval: 30m
  resetindex: False
  point: False
 ```

## Complete example configurations

* [EDE Analysis](https://github.com/IeAT-ASPIDE/Event-Detection-Engine/blob/master/1_ede_analysis_y2.yaml)
* [EDE Clustering](https://github.com/IeAT-ASPIDE/Event-Detection-Engine/blob/master/2_ede_clustering_y2.yaml)
* [EDE Clustering user defined](https://github.com/IeAT-ASPIDE/Event-Detection-Engine/blob/master/3_ede_clustering_user_y2.yaml)
* [EDE Clustering Prediction](https://github.com/IeAT-ASPIDE/Event-Detection-Engine/blob/master/4_ede_clustering_predict_y2.yaml)
* [EDE Classification](https://github.com/IeAT-ASPIDE/Event-Detection-Engine/blob/master/5_ede_classification_y2.yaml)
* [EDE Classification Predicton](https://github.com/IeAT-ASPIDE/Event-Detection-Engine/blob/master/6_ede_classification_predict_y2.yaml)
* [EDE HPO](https://github.com/IeAT-ASPIDE/Event-Detection-Engine/blob/master/7_ede_hpo_y2.yaml)
* [EDE TPOT](https://github.com/IeAT-ASPIDE/Event-Detection-Engine/blob/master/8_ede_tpot.yaml)
* [EDE TPOT Predict](https://github.com/IeAT-ASPIDE/Event-Detection-Engine/blob/master/9_ede_tpot_predict.yaml)