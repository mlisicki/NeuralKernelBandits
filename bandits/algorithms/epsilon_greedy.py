# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Thompson Sampling with linear posterior over a learnt deep representation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import invgamma

from bandits.core.bandit_algorithm import BanditAlgorithm
from bandits.core.contextual_dataset import ContextualDataset
from bandits.algorithms.neural_bandit_model import NeuralBanditModel,TextCNN
import tensorflow as tf

import mpmath as mp

import random

class NeuralLinearEpsilonGreedy(BanditAlgorithm):
  """Full Bayesian linear regression on the last layer of a deep neural net."""

  def __init__(self, name, hparams,textflag ='no', optimizer='RMS'):

    self.name = name
    self.hparams = hparams
    self.epsilon= self.hparams.epsilon
    self.latent_dim = self.hparams.layer_sizes[-1]
    self.intercept = True
    if self.intercept:
      self.param_dim=1+self.latent_dim
    else:
      self.param_dim = self.latent_dim
    # Gaussian prior for each beta_i





    # Regression and NN Update Frequency
    self.update_freq_lr = hparams.training_freq
    self.update_freq_nn = hparams.training_freq_network

    self.t = 0
    self.optimizer_n = optimizer

    self.num_epochs = hparams.training_epochs
    self.data_h = ContextualDataset(hparams.context_dim,
                                    hparams.num_actions,
                                    intercept=False)
    self.latent_h = ContextualDataset(self.latent_dim,
                                      hparams.num_actions,
                                      intercept=self.intercept)
    if textflag=='yes':
      self.bnn = TextCNN('adam', self.hparams.num_actions,self.hparams.batch_size, '{}-bnn'.format(name))
    else:
      self.bnn = NeuralBanditModel(optimizer, hparams, '{}-bnn'.format(name))



  def action(self, context):
    """Samples beta's from posterior, and chooses best action accordingly."""

    # Round robin until each action has been selected "initial_pulls" times
    if self.t < self.hparams.num_actions * self.hparams.initial_pulls:
      return self.t % self.hparams.num_actions

    with self.bnn.graph.as_default():
      c = context.reshape((1, self.hparams.context_dim))
      y = self.bnn.sess.run(self.bnn.y_pred, feed_dict={self.bnn.x: c})
      if random.random() > self.epsilon:
        return np.argmax(y)
      else:
        return random.randrange(self.hparams.num_actions)


  def update(self, context, action, reward):
    """Updates the posterior using linear bayesian regression formula."""

    self.t += 1
    self.data_h.add(context, action, reward)
    c = context.reshape((1, self.hparams.context_dim))
    z_context = self.bnn.sess.run(self.bnn.nn, feed_dict={self.bnn.x: c})
    self.latent_h.add(z_context, action, reward)

    # Retrain the network on the original data (data_h)
    if self.t % self.update_freq_nn == 0:

      if self.hparams.reset_lr:
        self.bnn.assign_lr()
      #self.bnn.set_last_layer(self.mu)
      self.bnn.train(self.data_h, self.num_epochs)


  @property
  def a0(self):
    return self._a0

  @property
  def b0(self):
    return self._b0

  @property
  def lambda_prior(self):
    return self._lambda_prior
