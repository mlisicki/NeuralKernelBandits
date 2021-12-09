"""
Copyright 2021 Michal Lisicki

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.


Neural kernel bandits
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from sklearn.preprocessing import StandardScaler
import neural_tangents as nt
from neural_tangents import stax
import jax
print(jax.devices())
from jax.config import config
# Enable float64 for JAX
config.update("jax_enable_x64", True)
from jax import jit

from bandits.core.bandit_algorithm import BanditAlgorithm
from bandits.core.contextual_dataset import ContextualDataset


class NKBandit(BanditAlgorithm):
    """
    This class implements a sampling process for the neural kernel bandit.
    The policy is guided by GP predictive distribution with NTK or NNGP
    kernels. The type of the used GP is specified through the `mode`parameter.
    For information on available modes see http://arxiv.org/abs/2007.05864.

    Parameters
    ---------

    name: str
        Label for the chosen configuration, used for displaying the results

    Parameters specified through hparams
    ------------------------------------
    alg: str
        Bandit policy (Upper Confidence Bounds / Thompson Sampling)

    mode : str
        GP type (nngp / deep_ensemble / rand_prior / ntkgp)

    joint: bool
        Joint GP for all arms or separate GP per arm

    num_layers: int
        Number of layers in a network corresponding to the chosen neural kernel

    gamma : float
        Kernel regularizer

    eta: float
        Bandit exploration parameter

    num_actions: int
        Number of arms

    context_dim: int
        Dimensionality of contexts / input data

    normalize_y: bool
        Whether to standardize the outputs before GP inference

    """

    def __init__(self, name, hparams):
        self.name = name
        self.hparams = hparams
        if hasattr(hparams,'mode'):
            # Following (He 2020) categorization
            if hparams.mode == "nngp":
                self.mode = "nngp"
                self.gamma = hparams.gamma
            elif hparams.mode == "deep_ensemble":
                self.mode = "ntk"
                self.gamma = 0
            elif hparams.mode == "rand_prior":
                self.mode = "ntk"
                self.gamma = hparams.gamma
            elif hparams.mode == "ntkgp":
                self.mode = "ntkgp"
                self.gamma = hparams.gamma
            else:
                raise Exception("Incorrect GP method. Use 'nngp', 'deep_ensemble', 'rand_prior' or 'ntkgp'")
        else:
            self.mode = 'ntk'
            self.gamma = hparams.gamma
        self.alg = hparams.alg
        self.mab_size = hparams.num_actions
        self.context_dim = hparams.context_dim

        self.eta = hparams.eta
        self.training_freq = hparams.training_freq

        # Normalize the outputs before computing the predictive distribution
        # See sklearn GP regression model for similar use
        if hasattr(hparams, 'normalize_y'):
            self.normalize_y = hparams.normalize_y
        else:
            self.normalize_y = False
        self.y_scaler = StandardScaler()

        self.joint = hparams.joint
        if self.joint:
            self.memory = None
        else:
            self.memory = [None] * self.hparams.num_actions

        # store student-t beta for debugging
        self._beta = np.zeros(self.mab_size)

        self.t = 0

        self.data_h = ContextualDataset(hparams.context_dim,
                                        hparams.num_actions,
                                        intercept=False)

        net = [stax.Dense(512), stax.Relu()] * self.hparams.num_layers + [stax.Dense(1)]
        init_fn, apply_fn, kernel_fn = stax.serial(*net)
        self.kernel_fn = jit(kernel_fn, static_argnums=(2,))

        model_type = "joint" if self.joint else "disjoint"
        print('NK-GP mode: {}, diag_reg: {}, model type: {}'.format(self.mode, self.gamma, model_type))
        print("Num of NK layers: {}".format((len(net)-1)/2))

    def action(self, context):
        # set reward estimate (e.g. ucb) to infinity first for initial exploration
        _r = np.full(self.mab_size, np.inf)

        for a in range(self.mab_size):
            # Retrieve data from the memory
            if self.joint:
                if self.memory is None:
                    break
                X, y, actions, predict_fn, model_size = self.memory
                if a in actions and self.t < self.mab_size:
                    # Ensure that the same action is not going to be picked up again until after k steps
                    _r[a] = -np.inf
                    continue
            else:
                if self.memory[a] is None:
                    continue
                if self.t < self.mab_size:
                    # Ensure that the same action is not going to be picked up again until after k steps
                    _r[a] = -np.inf
                    continue
                X, y, _, predict_fn, model_size = self.memory[a]

            X = X[:model_size]
            y = y[:model_size]
            if self.normalize_y:
                self.y_scaler.fit(y)
                y = self.y_scaler.transform(y)

            if self.joint:
                # zero padding
                x = np.zeros((1, self.context_dim * self.mab_size))
                x[0, a * self.context_dim:(a + 1) * self.context_dim] = context
            else:
                x = context[np.newaxis, :]

            kxX = self.kernel_fn(x, X, ('nngp', 'ntk'))
            kxx = self.kernel_fn(x, x, ('nngp', 'ntk'))

            y_pred = predict_fn(get=self.mode, k_test_train=kxX, k_test_test=kxx)

            if self.normalize_y:
                _mu = self.y_scaler.inverse_transform(y_pred.mean)[:,0]
            else:
                _mu = y_pred.mean[:,0]

            _cov = y_pred.covariance + np.finfo(float).eps # prevent negative covariance
            if (self.alg == "ucb"):
                _sigma = np.sqrt(np.diag(_cov))
                _r[a] = _mu + self.eta * _sigma
            elif (self.alg == "ts"):
                _r[a] = np.random.multivariate_normal(_mu, self.eta * _cov)
            else:
                raise Exception("Incorrect algorithm. Choose UCB or TS.")

        # Choose the max reward, randomization is only to resolve ties (e.g. in the beginning when all are infinite)
        action = np.random.choice(np.argwhere(_r == np.amax(_r)).flatten())

        return action

    def _update_internal_model(self, memory, action):
        if memory is None:
            return memory
        elif self.t < self.mab_size:
            return memory

        X, y, actions, predict_fn, model_size = memory
        if self.normalize_y:
            self.y_scaler.fit(y)
            y = self.y_scaler.transform(y)

        kXX = self.kernel_fn(X, X, ('nngp', 'ntk'))
        predict_fn = nt.predict.gp_inference(kXX, y, diag_reg=self.gamma)
        model_size = X.shape[0]
        memory = (X, y, actions, predict_fn, model_size)
        print("t = {}, Action = {}, X.shape = {}, y.shape = {}, kXX.shape = {}".format(self.t, action, X.shape, y.shape,
                                                                                       kXX.ntk.shape))
        return memory

    def update(self, context, action, reward):
        self.t += 1
        self.data_h.add(context, action, reward)

        if self.joint:
            # zero padding
            x = np.zeros((1, self.context_dim*self.mab_size))
            x[0,action*self.context_dim:(action+1)*self.context_dim] = context
            memory = self.memory
        else:
            x = context[np.newaxis, :]
            memory = self.memory[action]

        if memory is None:
            X = np.array(x)
            y = np.array([[reward]])
            actions = np.array([action])
            _y = np.array([[0]]) if self.normalize_y else y
            kXX = self.kernel_fn(X, X, ('nngp', 'ntk'))
            _predict_fn = nt.predict.gp_inference(kXX, _y, diag_reg=self.gamma)
            model_size = X.shape[0]
        else:
            X, y, actions, _predict_fn, model_size = memory
            actions = np.append(actions, action)
            X = np.append(X, x, axis=0)
            y = np.append(y, [[reward]], axis=0)

        if self.joint:
            self.memory = (X, y, actions, _predict_fn, model_size)
        else:
            self.memory[action] = (X, y, [], _predict_fn, model_size)

        # Update NK
        if self.t % self.training_freq == 0:
            if self.joint:
                self.memory = self._update_internal_model(self.memory,"joint")
            else:
                for a in range(self.mab_size):
                    self.memory[a] = self._update_internal_model(self.memory[a],a)


