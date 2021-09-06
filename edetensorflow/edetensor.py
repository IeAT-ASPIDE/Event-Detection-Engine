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

import tensorflow as tf
from tensorflow.keras.wrappers.scikit_learn import KerasClassifier
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv1D, Dropout, LSTM, SimpleRNN
from tensorflow.keras.optimizers import Adam, Adagrad, SGD
from tensorflow.keras.utils import plot_model
import numpy as np
import pandas as pd
import os


def dnn_aspide(X,
               y,
               optimizer='adam',  # adam, adagrad, sgd
               learning_r=0.01,
               kernel_init='he_normal',
               layer_1=0,
               layer_2=50,
               layer_3=100,
               layer_0=50,
               drop=0.3,
               loss='categorical_crossentropy',
               activation_1='relu',  # elu, selu
               out_activation='sigmoid'):

    y_oh = pd.get_dummies(y, prefix='target')
    # print(np.asarray(X).shape[1], len(y_oh.nunique()))
    n_inputs, n_outputs = X.shape[1], len(y_oh.nunique())
    model = Sequential()
    # model.add(Conv1D(filters=32, kernel_size=2,activation=activation_1, input_shape=n_inputs, kernel_initializer=kernel_init))
    model.add(Dense(layer_0, input_dim=n_inputs, kernel_initializer=kernel_init, activation=activation_1))
    if drop:
      model.add(Dropout(drop))
    if layer_1:
      model.add(Dense(layer_1, input_dim=n_inputs, kernel_initializer=kernel_init, activation=activation_1))
      if drop:
        model.add(Dropout(drop))
    if layer_2:
      model.add(Dense(layer_2, input_dim=n_inputs, kernel_initializer=kernel_init, activation=activation_1))
      if drop:
        model.add(Dropout(drop))
    if layer_3:
      model.add(Dense(layer_2, input_dim=n_inputs, kernel_initializer=kernel_init, activation=activation_1))
      if drop:
        model.add(Dropout(drop))
    model.add(Dense(n_outputs, activation=out_activation))
    if optimizer == 'adam':
      opt = Adam(learning_rate=learning_r)
    elif optimizer == 'adagrad':
      opt = Adagrad(learning_rate=learning_r)
    elif optimizer == 'sgd':
      opt = SGD(learning_rate=learning_r)
    else:
      opt = Adam(learning_rate=0.01)
    model.compile(optimizer=opt, loss=loss, metrics=['accuracy', 'categorical_crossentropy', 'binary_crossentropy'])
    return model


def ede_dnn(dnn_model,
            Xtrain,
            ytrain,
            Xtest,
            ytest,
            batch_size,
            epochs,
            model_dir,
            patience=3,
            factor=0.2,
            export='DNN_y2',
            verbose=0
            ):
    """
    Used to generate DNN model instance and training.

    :param dnn_model: Model to be generated
    :param Xtrain: Training input data
    :param ytrain: Training ground truth
    :param Xtest: Testing input data
    :param ytest: Testing ground truth
    :param batch_size: DNN Batch size
    :param epochs: Training Epochs
    :param model_dir: Model directory location
    :param patience: Patiance for early stopping callback
    :param factor: Factor for reduce learning rate
    :param export: name used for exporting
    :return: tf.history
    """
    # One hot encoding of groundtruth both testing and training
    y_oh_train = pd.get_dummies(ytrain, prefix='target')
    y_oh_test = pd.get_dummies(ytest, prefix='target')

    early_stopping = EarlyStopping(monitor="loss", patience=patience)# early stop patience
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=factor,
                                patience=5, min_lr=0.00001)
    model = KerasClassifier(build_fn=dnn_model, verbose=verbose, callbacks=[early_stopping, reduce_lr])
    history = model.fit(np.asarray(Xtrain), np.asarray(y_oh_train),
                      batch_size=batch_size, epochs=epochs,
                      callbacks=[early_stopping, reduce_lr],
                      verbose=0, validation_data=(np.asarray(Xtest), np.asarray(y_oh_test)))
    # Saving History
    df_history = pd.DataFrame(history.history)
    history_name = "DNN_history_{}.csv".format(export)
    df_history.to_csv(os.path.join(model_dir, history_name), index=False)
    return history


