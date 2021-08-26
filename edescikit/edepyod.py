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

###
Single-Objective Generative Adversarial Active Learning.
Part of the codes are adapted from
https://github.com/leibinghe/GAAL-based-outlier-detection

# Author: Winston Li <jk_zhengli@hotmail.com>
# Modified: Gabriel Iuhasz <iuhasz.gabriel@e-uvt.ro>
# License: BSD 2 clause

###
Multiple-Objective Generative Adversarial Active Learning.
Part of the codes are adapted from
https://github.com/leibinghe/GAAL-based-outlier-detection

# Author: Winston Li <jk_zhengli@hotmail.com>
# Modified: Gabriel Iuhasz <iuhasz.gabriel@e-uvt.ro>
# License: BSD 2 clause

###
Variational Auto Encoder (VAE)
and beta-VAE for Unsupervised Outlier Detection

Reference:
:cite:`kingma2013auto` Kingma, Diederik, Welling
'Auto-Encodeing Variational Bayes'
https://arxiv.org/abs/1312.6114

:cite:`burgess2018understanding` Burges et al
'Understanding disentangling in beta-VAE'
https://arxiv.org/pdf/1804.03599.pdf


# Author: Andrij Vasylenko <andrij@liverpool.ac.uk>
# Modified: Gabriel Iuhasz <iuhasz.gabriel@e-uvt.ro>
# License: BSD 2 clause

###
Using Auto Encoder with Outlier Detection

# Author: Yue Zhao <zhaoy@cmu.edu>
# Modified: Gabriel Iuhasz <iuhasz.gabriel@e-uvt.ro>
# License: BSD 2 clause

"""


from __future__ import division
from __future__ import print_function

import pyod
from pyod.models.abod import ABOD
from pyod.models.cblof import CBLOF
from pyod.models.feature_bagging import FeatureBagging
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.knn import KNN
from pyod.models.lof import LOF
from pyod.models.mcd import MCD
from pyod.models.ocsvm import OCSVM
from pyod.models.lscp import LSCP

from collections import defaultdict

import numpy as np

from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils.validation import check_is_fitted
from pyod.utils.utility import check_parameter
from pyod.utils.stat_models import pairwise_distances_no_broadcast
from pyod.models.base import BaseDetector
from pyod.models.base_dl import _get_tensorflow_version

from pyod.models.gaal_base import create_discriminator
from pyod.models.gaal_base import create_generator


# if tensorflow 2, import from tf directly
if _get_tensorflow_version() == 1:
    from keras.layers import Input, Lambda, Input, Dense, Dropout
    from keras.regularizers import l2
    from keras.losses import mse, binary_crossentropy, mean_squared_error
    from keras.models import Model, Sequential
    from keras.optimizers import SGD
    from keras import backend as K
else:
    from tensorflow.keras.layers import Input, Lambda, Input, Dense, Dropout
    from tensorflow.keras.regularizers import l2
    from tensorflow.keras.models import Model, Sequential
    from tensorflow.keras.losses import mse, binary_crossentropy, mean_squared_error
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras import backend as K


class SO_GAAL_ASPIDE(BaseDetector):
    """Single-Objective Generative Adversarial Active Learning.

    SO-GAAL directly generates informative potential outliers to assist the
    classifier in describing a boundary that can separate outliers from normal
    data effectively. Moreover, to prevent the generator from falling into the
    mode collapsing problem, the network structure of SO-GAAL is expanded from
    a single generator (SO-GAAL) to multiple generators with different
    objectives (MO-GAAL) to generate a reasonable reference distribution for
    the whole dataset.
    Read more in the :cite:`liu2019generative`.

    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    stop_epochs : int, optional (default=20)
        The number of epochs of training.

    lr_d : float, optional (default=0.01)
        The learn rate of the discriminator.

    lr_g : float, optional (default=0.0001)
        The learn rate of the generator.

    decay : float, optional (default=1e-6)
        The decay parameter for SGD.

    momentum : float, optional (default=0.9)
        The momentum parameter for SGD.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, stop_epochs=20, lr_d=0.01, lr_g=0.0001,
                 decay=1e-6, momentum=0.9, contamination=0.1, verbose=0):
        super(SO_GAAL_ASPIDE, self).__init__(contamination=contamination)
        self.stop_epochs = stop_epochs
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.decay = decay
        self.momentum = momentum
        self.verbose = verbose

    def fit(self, X, y=None, **kwargs):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        X = check_array(X)
        self._set_n_classes(y)
        latent_size = X.shape[1]
        data_size = X.shape[0]
        stop = 0
        epochs = self.stop_epochs * 3
        self.train_history = defaultdict(list)

        self.discriminator = create_discriminator(latent_size, data_size)
        self.discriminator.compile(
            optimizer=SGD(lr=self.lr_d, decay=self.decay,
                          momentum=self.momentum), loss='binary_crossentropy')

        self.generator = create_generator(latent_size)
        latent = Input(shape=(latent_size,))
        fake = self.generator(latent)
        self.discriminator.trainable = False
        fake = self.discriminator(fake)
        self.combine_model = Model(latent, fake)
        self.combine_model.compile(
            optimizer=SGD(lr=self.lr_g, decay=self.decay,
                          momentum=self.momentum), loss='binary_crossentropy')

        # Start iteration
        for epoch in range(epochs):
            if self.verbose:
                print('Epoch {} of {}'.format(epoch + 1, epochs))
            batch_size = min(500, data_size)
            num_batches = int(data_size / batch_size)

            for index in range(num_batches):
                if self.verbose:
                    print('\nTesting for epoch {} index {}:'.format(epoch + 1, index + 1))

                # Generate noise
                noise_size = batch_size
                noise = np.random.uniform(0, 1, (int(noise_size), latent_size))

                # Get training data
                data_batch = X[index * batch_size: (index + 1) * batch_size]

                # Generate potential outliers
                generated_data = self.generator.predict(noise, verbose=0)

                # Concatenate real data to generated data
                x = np.concatenate((data_batch, generated_data))
                y = np.array([1] * batch_size + [0] * int(noise_size))

                # Train discriminator
                discriminator_loss = self.discriminator.train_on_batch(x, y)
                self.train_history['discriminator_loss'].append(
                    discriminator_loss)

                # Train generator
                if stop == 0:
                    trick = np.array([1] * noise_size)
                    generator_loss = self.combine_model.train_on_batch(noise,
                                                                       trick)
                    self.train_history['generator_loss'].append(generator_loss)
                else:
                    trick = np.array([1] * noise_size)
                    generator_loss = self.combine_model.evaluate(noise, trick)
                    self.train_history['generator_loss'].append(generator_loss)

            # Stop training generator
            if epoch + 1 > self.stop_epochs:
                stop = 1

        # Detection result
        self.decision_scores_ = self.discriminator.predict(X).ravel()
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['discriminator'])
        X = check_array(X)
        pred_scores = self.discriminator.predict(X).ravel()
        return pred_scores


class MO_GAAL_ASPIDE(BaseDetector):
    """Multi-Objective Generative Adversarial Active Learning.

    MO_GAAL directly generates informative potential outliers to assist the
    classifier in describing a boundary that can separate outliers from normal
    data effectively. Moreover, to prevent the generator from falling into the
    mode collapsing problem, the network structure of SO-GAAL is expanded from
    a single generator (SO-GAAL) to multiple generators with different
    objectives (MO-GAAL) to generate a reasonable reference distribution for
    the whole dataset.
    Read more in the :cite:`liu2019generative`.

    Parameters
    ----------
    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. Used when fitting to
        define the threshold on the decision function.

    k : int, optional (default=10)
        The number of sub generators.

    stop_epochs : int, optional (default=20)
        The number of epochs of training.

    lr_d : float, optional (default=0.01)
        The learn rate of the discriminator.

    lr_g : float, optional (default=0.0001)
        The learn rate of the generator.

    decay : float, optional (default=1e-6)
        The decay parameter for SGD.

    momentum : float, optional (default=0.9)
        The momentum parameter for SGD.

    Attributes
    ----------
    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, k=10, stop_epochs=20, lr_d=0.01, lr_g=0.0001,
                 decay=1e-6, momentum=0.9, contamination=0.1, verbose=0):
        super(MO_GAAL_ASPIDE, self).__init__(contamination=contamination)
        self.k = k
        self.stop_epochs = stop_epochs
        self.lr_d = lr_d
        self.lr_g = lr_g
        self.decay = decay
        self.momentum = momentum
        self.verbose = verbose

    def fit(self, X, y=None, **kwargs):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """

        X = check_array(X)
        self._set_n_classes(y)
        self.train_history = defaultdict(list)
        names = locals()
        epochs = self.stop_epochs * 3
        stop = 0
        latent_size = X.shape[1]
        data_size = X.shape[0]
        # Create discriminator
        self.discriminator = create_discriminator(latent_size, data_size)
        self.discriminator.compile(
            optimizer=SGD(lr=self.lr_d, decay=self.decay,
                          momentum=self.momentum), loss='binary_crossentropy')

        # Create k combine models
        for i in range(self.k):
            names['sub_generator' + str(i)] = create_generator(latent_size)
            latent = Input(shape=(latent_size,))
            names['fake' + str(i)] = names['sub_generator' + str(i)](latent)
            self.discriminator.trainable = False
            names['fake' + str(i)] = self.discriminator(names['fake' + str(i)])
            names['combine_model' + str(i)] = Model(latent,
                                                    names['fake' + str(i)])
            names['combine_model' + str(i)].compile(
                optimizer=SGD(lr=self.lr_g,
                              decay=self.decay,
                              momentum=self.momentum),
                loss='binary_crossentropy')

        # Start iteration
        for epoch in range(epochs):
            if self.verbose:
                print('Epoch {} of {}'.format(epoch + 1, epochs))
            batch_size = min(500, data_size)
            num_batches = int(data_size / batch_size)

            for index in range(num_batches):
                if self.verbose:
                    print('\nTesting for epoch {} index {}:'.format(epoch + 1,
                                                                    index + 1))

                # Generate noise
                noise_size = batch_size
                noise = np.random.uniform(0, 1, (int(noise_size), latent_size))

                # Get training data
                data_batch = X[index * batch_size: (index + 1) * batch_size]

                # Generate potential outliers
                block = ((1 + self.k) * self.k) // 2
                for i in range(self.k):
                    if i != (self.k - 1):
                        noise_start = int(
                            (((self.k + (self.k - i + 1)) * i) / 2) * (
                                    noise_size // block))
                        noise_end = int(
                            (((self.k + (self.k - i)) * (i + 1)) / 2) * (
                                    noise_size // block))
                        names['noise' + str(i)] = noise[noise_start:noise_end]
                        names['generated_data' + str(i)] = names[
                            'sub_generator' + str(i)].predict(
                            names['noise' + str(i)], verbose=0)
                    else:
                        noise_start = int(
                            (((self.k + (self.k - i + 1)) * i) / 2) * (
                                    noise_size // block))
                        names['noise' + str(i)] = noise[noise_start:noise_size]
                        names['generated_data' + str(i)] = names[
                            'sub_generator' + str(i)].predict(
                            names['noise' + str(i)], verbose=0)

                # Concatenate real data to generated data
                for i in range(self.k):
                    if i == 0:
                        x = np.concatenate(
                            (data_batch, names['generated_data' + str(i)]))
                    else:
                        x = np.concatenate(
                            (x, names['generated_data' + str(i)]))
                y = np.array([1] * batch_size + [0] * int(noise_size))

                # Train discriminator
                discriminator_loss = self.discriminator.train_on_batch(x, y)
                self.train_history['discriminator_loss'].append(
                    discriminator_loss)

                # Get the target value of sub-generator
                pred_scores = self.discriminator.predict(X).ravel()

                for i in range(self.k):
                    names['T' + str(i)] = np.percentile(pred_scores,
                                                        i / self.k * 100)
                    names['trick' + str(i)] = np.array(
                        [float(names['T' + str(i)])] * noise_size)

                # Train generator
                noise = np.random.uniform(0, 1, (int(noise_size), latent_size))
                if stop == 0:
                    for i in range(self.k):
                        names['sub_generator' + str(i) + '_loss'] = \
                            names['combine_model' + str(i)].train_on_batch(
                                noise, names['trick' + str(i)])
                        self.train_history[
                            'sub_generator{}_loss'.format(i)].append(
                            names['sub_generator' + str(i) + '_loss'])
                else:
                    for i in range(self.k):
                        names['sub_generator' + str(i) + '_loss'] = names[
                            'combine_model' + str(i)].evaluate(noise, names[
                            'trick' + str(i)])
                        self.train_history[
                            'sub_generator{}_loss'.format(i)].append(
                            names['sub_generator' + str(i) + '_loss'])

                generator_loss = 0
                for i in range(self.k):
                    generator_loss = generator_loss + names[
                        'sub_generator' + str(i) + '_loss']
                generator_loss = generator_loss / self.k
                self.train_history['generator_loss'].append(generator_loss)

                # Stop training generator
                if epoch + 1 > self.stop_epochs:
                    stop = 1

        # Detection result
        self.decision_scores_ = self.discriminator.predict(X).ravel()
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['discriminator'])
        X = check_array(X)
        pred_scores = self.discriminator.predict(X).ravel()
        return pred_scores



class VAE_ASPIDE(BaseDetector):
    """ Variational auto encoder
    Encoder maps X onto a latent space Z
    Decoder samples Z from N(0,1)
    VAE_loss = Reconstruction_loss + KL_loss

    Reference
    See :cite:`kingma2013auto` Kingma, Diederik, Welling
    'Auto-Encodeing Variational Bayes'
    https://arxiv.org/abs/1312.6114 for details.

    beta VAE
    In Loss, the emphasis is on KL_loss
    and capacity of a bottleneck:
    VAE_loss = Reconstruction_loss + gamma*KL_loss

    Reference
    See :cite:`burgess2018understanding` Burges et al
    'Understanding disentangling in beta-VAE'
    https://arxiv.org/pdf/1804.03599.pdf for details.


    Parameters
    ----------
    encoder_neurons : list, optional (default=[128, 64, 32])
        The number of neurons per hidden layer in encoder.

    decoder_neurons : list, optional (default=[32, 64, 128])
        The number of neurons per hidden layer in decoder.

    hidden_activation : str, optional (default='relu')
        Activation function to use for hidden layers.
        All hidden layers are forced to use the same type of activation.
        See https://keras.io/activations/

    output_activation : str, optional (default='sigmoid')
        Activation function to use for output layer.
        See https://keras.io/activations/

    loss : str or obj, optional (default=keras.losses.mean_squared_error
        String (name of objective function) or objective function.
        See https://keras.io/losses/

    gamma : float, optional (default=1.0)
        Coefficient of beta VAE regime.
        Default is regular VAE.

    capacity : float, optional (default=0.0)
        Maximum capacity of a loss bottle neck.

    optimizer : str, optional (default='adam')
        String (name of optimizer) or optimizer instance.
        See https://keras.io/optimizers/

    epochs : int, optional (default=100)
        Number of epochs to train the model.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    dropout_rate : float in (0., 1), optional (default=0.2)
        The dropout to be used across all layers.

    l2_regularizer : float in (0., 1), optional (default=0.1)
        The regularization strength of activity_regularizer
        applied on each layer. By default, l2 regularizer is used. See
        https://keras.io/regularizers/

    validation_size : float in (0., 1), optional (default=0.1)
        The percentage of data to be used for validation.

    preprocessing : bool, optional (default=True)
        If True, apply standardization on the data.

    verbose : int, optional (default=1)
        verbose mode.

        - 0 = silent
        - 1 = progress bar
        - 2 = one line per epoch.

        For verbose >= 1, model summary may be printed.

    random_state : random_state: int, RandomState instance or None, opti
        (default=None)
        If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the r
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. When fitting this is
        to define the threshold on the decision function.

    Attributes
    ----------
    encoding_dim_ : int
        The number of neurons in the encoding layer.

    compression_rate_ : float
        The ratio between the original feature and
        the number of neurons in the encoding layer.

    model_ : Keras Object
        The underlying AutoEncoder in Keras.

    history_: Keras Object
        The AutoEncoder training history.

    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, encoder_neurons=None, decoder_neurons=None,
                 latent_dim=2, hidden_activation='relu',
                 output_activation='sigmoid', loss=mse, optimizer='adam',
                 epochs=100, batch_size=32, dropout_rate=0.2,
                 l2_regularizer=0.1, validation_size=0.1, preprocessing=True,
                 verbose=1, random_state=None, contamination=0.1,
                 gamma=1.0, capacity=0.0):
        super(VAE_ASPIDE, self).__init__(contamination=contamination)
        self.encoder_neurons = encoder_neurons
        self.decoder_neurons = decoder_neurons
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_regularizer = l2_regularizer
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.random_state = random_state
        self.latent_dim = latent_dim
        self.gamma = gamma
        self.capacity = capacity

        # default values
        if self.encoder_neurons is None:
            self.encoder_neurons = [128, 64, 32]

        if self.decoder_neurons is None:
            self.decoder_neurons = [32, 64, 128]

        self.encoder_neurons_ = self.encoder_neurons
        self.decoder_neurons_ = self.decoder_neurons

        check_parameter(dropout_rate, 0, 1, param_name='dropout_rate',
                        include_left=True)

    def sampling(self, args):
        """Reparametrisation by sampling from Gaussian, N(0,I)
        To sample from epsilon = Norm(0,I) instead of from likelihood Q(z|X)
        with latent variables z: z = z_mean + sqrt(var) * epsilon

        Parameters
        ----------
        args : tensor
            Mean and log of variance of Q(z|X).

        Returns
        -------
        z : tensor
            Sampled latent variable.
        """

        z_mean, z_log = args
        batch = K.shape(z_mean)[0]  # batch size
        dim = K.int_shape(z_mean)[1]  # latent dimension
        epsilon = K.random_normal(shape=(batch, dim))  # mean=0, std=1.0

        return z_mean + K.exp(0.5 * z_log) * epsilon

    def vae_loss(self, inputs, outputs, z_mean, z_log):
        """ Loss = Recreation loss + Kullback-Leibler loss
        for probability function divergence (ELBO).
        gamma > 1 and capacity != 0 for beta-VAE
        """

        reconstruction_loss = self.loss(inputs, outputs)
        reconstruction_loss *= self.n_features_
        kl_loss = 1 + z_log - K.square(z_mean) - K.exp(z_log)
        kl_loss = -0.5 * K.sum(kl_loss, axis=-1)
        kl_loss = self.gamma * K.abs(kl_loss - self.capacity)

        return K.mean(reconstruction_loss + kl_loss)

    def _build_model(self):
        """Build VAE = encoder + decoder + vae_loss"""

        # Build Encoder
        inputs = Input(shape=(self.n_features_,))
        # Input layer
        layer = Dense(self.n_features_, activation=self.hidden_activation)(
            inputs)
        # Hidden layers
        for neurons in self.encoder_neurons:
            layer = Dense(neurons, activation=self.hidden_activation,
                          activity_regularizer=l2(self.l2_regularizer))(layer)
            layer = Dropout(self.dropout_rate)(layer)
        # Create mu and sigma of latent variables
        z_mean = Dense(self.latent_dim)(layer)
        z_log = Dense(self.latent_dim)(layer)
        # Use parametrisation sampling
        z = Lambda(self.sampling, output_shape=(self.latent_dim,))(
            [z_mean, z_log])
        # Instantiate encoder
        encoder = Model(inputs, [z_mean, z_log, z])
        if self.verbose >= 1:
            encoder.summary()

        # Build Decoder
        latent_inputs = Input(shape=(self.latent_dim,))
        # Latent input layer
        layer = Dense(self.latent_dim, activation=self.hidden_activation)(
            latent_inputs)
        # Hidden layers
        for neurons in self.decoder_neurons:
            layer = Dense(neurons, activation=self.hidden_activation)(layer)
            layer = Dropout(self.dropout_rate)(layer)
        # Output layer
        outputs = Dense(self.n_features_, activation=self.output_activation)(
            layer)
        # Instatiate decoder
        decoder = Model(latent_inputs, outputs)
        if self.verbose >= 1:
            decoder.summary()
        # Generate outputs
        outputs = decoder(encoder(inputs)[2])

        # Instantiate VAE
        vae = Model(inputs, outputs)
        vae.add_loss(self.vae_loss(inputs, outputs, z_mean, z_log))
        vae.compile(optimizer=self.optimizer)
        if self.verbose >= 1:
            vae.summary()
        return vae

    def fit(self, X, y=None, **kwargs):
        """Fit detector. y is optional for unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : numpy array of shape (n_samples,), optional (default=None)
            The ground truth of the input samples (labels).
        """
        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        # Verify and construct the hidden units
        self.n_samples_, self.n_features_ = X.shape[0], X.shape[1]

        # Standardize data for better performance
        if self.preprocessing:
            self.scaler_ = StandardScaler()
            X_norm = self.scaler_.fit_transform(X)
        else:
            X_norm = np.copy(X)

        # Shuffle the data for validation as Keras do not shuffling for
        # Validation Split
        np.random.shuffle(X_norm)

        # Validate and complete the number of hidden neurons
        if np.min(self.encoder_neurons) > self.n_features_:
            raise ValueError("The number of neurons should not exceed "
                             "the number of features")

        # Build VAE model & fit with X
        self.model_ = self._build_model()
        self.history_ = self.model_.fit(X_norm,
                                        epochs=self.epochs,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        validation_split=self.validation_size,
                                        verbose=self.verbose,
                                        **kwargs).history
        # Predict on X itself and calculate the reconstruction error as
        # the outlier scores. Noted X_norm was shuffled has to recreate
        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        pred_scores = self.model_.predict(X_norm)
        self.decision_scores_ = pairwise_distances_no_broadcast(X_norm,
                                                                pred_scores)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['model_', 'history_'])
        X = check_array(X)

        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        # Predict on X and return the reconstruction errors
        pred_scores = self.model_.predict(X_norm)
        return pairwise_distances_no_broadcast(X_norm, pred_scores)

class AutoEncoder(BaseDetector):
    """Auto Encoder (AE) is a type of neural networks for learning useful data
    representations unsupervisedly. Similar to PCA, AE could be used to
    detect outlying objects in the data by calculating the reconstruction
    errors. See :cite:`aggarwal2015outlier` Chapter 3 for details.

    Parameters
    ----------
    hidden_neurons : list, optional (default=[64, 32, 32, 64])
        The number of neurons per hidden layers.

    hidden_activation : str, optional (default='relu')
        Activation function to use for hidden layers.
        All hidden layers are forced to use the same type of activation.
        See https://keras.io/activations/

    output_activation : str, optional (default='sigmoid')
        Activation function to use for output layer.
        See https://keras.io/activations/

    loss : str or obj, optional (default=keras.losses.mean_squared_error)
        String (name of objective function) or objective function.
        See https://keras.io/losses/

    optimizer : str, optional (default='adam')
        String (name of optimizer) or optimizer instance.
        See https://keras.io/optimizers/

    epochs : int, optional (default=100)
        Number of epochs to train the model.

    batch_size : int, optional (default=32)
        Number of samples per gradient update.

    dropout_rate : float in (0., 1), optional (default=0.2)
        The dropout to be used across all layers.

    l2_regularizer : float in (0., 1), optional (default=0.1)
        The regularization strength of activity_regularizer
        applied on each layer. By default, l2 regularizer is used. See
        https://keras.io/regularizers/

    validation_size : float in (0., 1), optional (default=0.1)
        The percentage of data to be used for validation.

    preprocessing : bool, optional (default=True)
        If True, apply standardization on the data.

    verbose : int, optional (default=1)
        Verbosity mode.

        - 0 = silent
        - 1 = progress bar
        - 2 = one line per epoch.

        For verbose >= 1, model summary may be printed.

    random_state : random_state: int, RandomState instance or None, optional
        (default=None)
        If int, random_state is the seed used by the random
        number generator; If RandomState instance, random_state is the random
        number generator; If None, the random number generator is the
        RandomState instance used by `np.random`.

    contamination : float in (0., 0.5), optional (default=0.1)
        The amount of contamination of the data set, i.e.
        the proportion of outliers in the data set. When fitting this is used
        to define the threshold on the decision function.

    Attributes
    ----------
    encoding_dim_ : int
        The number of neurons in the encoding layer.

    compression_rate_ : float
        The ratio between the original feature and
        the number of neurons in the encoding layer.

    model_ : Keras Object
        The underlying AutoEncoder in Keras.

    history_: Keras Object
        The AutoEncoder training history.

    decision_scores_ : numpy array of shape (n_samples,)
        The outlier scores of the training data.
        The higher, the more abnormal. Outliers tend to have higher
        scores. This value is available once the detector is
        fitted.

    threshold_ : float
        The threshold is based on ``contamination``. It is the
        ``n_samples * contamination`` most abnormal samples in
        ``decision_scores_``. The threshold is calculated for generating
        binary outlier labels.

    labels_ : int, either 0 or 1
        The binary labels of the training data. 0 stands for inliers
        and 1 for outliers/anomalies. It is generated by applying
        ``threshold_`` on ``decision_scores_``.
    """

    def __init__(self, hidden_neurons=None,
                 hidden_activation='relu', output_activation='sigmoid',
                 loss=mean_squared_error, optimizer='adam',
                 epochs=100, batch_size=32, dropout_rate=0.2,
                 l2_regularizer=0.1, validation_size=0.1, preprocessing=True,
                 verbose=1, random_state=None, contamination=0.1):
        super(AutoEncoder, self).__init__(contamination=contamination)
        self.hidden_neurons = hidden_neurons
        self.hidden_activation = hidden_activation
        self.output_activation = output_activation
        self.loss = loss
        self.optimizer = optimizer
        self.epochs = epochs
        self.batch_size = batch_size
        self.dropout_rate = dropout_rate
        self.l2_regularizer = l2_regularizer
        self.validation_size = validation_size
        self.preprocessing = preprocessing
        self.verbose = verbose
        self.random_state = random_state

        # default values
        if self.hidden_neurons is None:
            self.hidden_neurons = [64, 32, 32, 64]

        # Verify the network design is valid
        if not self.hidden_neurons == self.hidden_neurons[::-1]:
            print(self.hidden_neurons)
            raise ValueError("Hidden units should be symmetric")

        self.hidden_neurons_ = self.hidden_neurons

        check_parameter(dropout_rate, 0, 1, param_name='dropout_rate',
                        include_left=True)

    def _build_model(self):
        model = Sequential()
        # Input layer
        model.add(Dense(
            self.hidden_neurons_[0], activation=self.hidden_activation,
            input_shape=(self.n_features_,),
            activity_regularizer=l2(self.l2_regularizer)))
        model.add(Dropout(self.dropout_rate))

        # Additional layers
        for i, hidden_neurons in enumerate(self.hidden_neurons_, 1):
            model.add(Dense(
                hidden_neurons,
                activation=self.hidden_activation,
                activity_regularizer=l2(self.l2_regularizer)))
            model.add(Dropout(self.dropout_rate))

        # Output layers
        model.add(Dense(self.n_features_, activation=self.output_activation,
                        activity_regularizer=l2(self.l2_regularizer)))

        # Compile model
        model.compile(loss=self.loss, optimizer=self.optimizer)
        if self.verbose >= 1:
            print(model.summary())
        return model

    # noinspection PyUnresolvedReferences
    def fit(self, X, y=None, **kwargs):
        """Fit detector. y is ignored in unsupervised methods.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The input samples.

        y : Ignored
            Not used, present for API consistency by convention.

        Returns
        -------
        self : object
            Fitted estimator.
        """
        # validate inputs X and y (optional)
        X = check_array(X)
        self._set_n_classes(y)

        # Verify and construct the hidden units
        self.n_samples_, self.n_features_ = X.shape[0], X.shape[1]

        # Standardize data for better performance
        if self.preprocessing:
            self.scaler_ = StandardScaler()
            X_norm = self.scaler_.fit_transform(X)
        else:
            X_norm = np.copy(X)

        # Shuffle the data for validation as Keras do not shuffling for
        # Validation Split
        np.random.shuffle(X_norm)

        # Validate and complete the number of hidden neurons
        if np.min(self.hidden_neurons) > self.n_features_:
            raise ValueError("The number of neurons should not exceed "
                             "the number of features")
        self.hidden_neurons_.insert(0, self.n_features_)

        # Calculate the dimension of the encoding layer & compression rate
        self.encoding_dim_ = np.median(self.hidden_neurons)
        self.compression_rate_ = self.n_features_ // self.encoding_dim_

        # Build AE model & fit with X
        self.model_ = self._build_model()
        self.history_ = self.model_.fit(X_norm, X_norm,
                                        epochs=self.epochs,
                                        batch_size=self.batch_size,
                                        shuffle=True,
                                        validation_split=self.validation_size,
                                        verbose=self.verbose,
                                        **kwargs).history
        # Reverse the operation for consistency
        self.hidden_neurons_.pop(0)
        # Predict on X itself and calculate the reconstruction error as
        # the outlier scores. Noted X_norm was shuffled has to recreate
        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        pred_scores = self.model_.predict(X_norm)
        self.decision_scores_ = pairwise_distances_no_broadcast(X_norm,
                                                                pred_scores)
        self._process_decision_scores()
        return self

    def decision_function(self, X):
        """Predict raw anomaly score of X using the fitted detector.

        The anomaly score of an input sample is computed based on different
        detector algorithms. For consistency, outliers are assigned with
        larger anomaly scores.

        Parameters
        ----------
        X : numpy array of shape (n_samples, n_features)
            The training input samples. Sparse matrices are accepted only
            if they are supported by the base estimator.

        Returns
        -------
        anomaly_scores : numpy array of shape (n_samples,)
            The anomaly score of the input samples.
        """
        check_is_fitted(self, ['model_', 'history_'])
        X = check_array(X)

        if self.preprocessing:
            X_norm = self.scaler_.transform(X)
        else:
            X_norm = np.copy(X)

        # Predict on X and return the reconstruction errors
        pred_scores = self.model_.predict(X_norm)
        return pairwise_distances_no_broadcast(X_norm, pred_scores)

def ede_abod(contamination):
    clf = ABOD(contamination=contamination)
    return clf


def ede_cblof(contamination,
              check_estimator,
              random_state):

    clf = CBLOF(contamination=contamination,
                check_estimator=check_estimator,
                random_state=random_state)
    return clf


def ede_hbos(contamination):
    clf = HBOS(contamination=contamination)
    return clf


