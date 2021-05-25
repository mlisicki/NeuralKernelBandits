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


class NeuralLinearPosteriorSampling(BanditAlgorithm):
  """Full Bayesian linear regression on the last layer of a deep neural net."""

  def __init__(self, name, hparams,textflag ='no', optimizer='RMS'):

    self.name = name
    self.hparams = hparams
    self.latent_dim = self.hparams.layer_sizes[-1]
    self.intercept = False
    if self.intercept:
      self.param_dim=1+self.latent_dim
    else:
      self.param_dim = self.latent_dim
    # Gaussian prior for each beta_i
    self._lambda_prior = self.hparams.lambda_prior

    self.mu = [
        np.zeros(self.param_dim)
        for _ in range(self.hparams.num_actions)
    ]

    self.f = [
      np.zeros(self.param_dim)
      for _ in range(self.hparams.num_actions)
    ]
    self.yy = [0 for _ in range(self.hparams.num_actions)]

    self.cov = [(1.0 / self.lambda_prior) * np.eye(self.param_dim)
                for _ in range(self.hparams.num_actions)]

    self.precision = [
        self.lambda_prior * np.eye(self.param_dim)
        for _ in range(self.hparams.num_actions)
    ]

    # Inverse Gamma prior for each sigma2_i
    self._a0 = self.hparams.a0
    self._b0 = self.hparams.b0

    self.a = [self._a0 for _ in range(self.hparams.num_actions)]
    self.b = [self._b0 for _ in range(self.hparams.num_actions)]

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

    # Sample sigma2, and beta conditional on sigma2
    sigma2_s = [
        self.b[i] * invgamma.rvs(self.a[i])
        for i in range(self.hparams.num_actions)
    ]

    try:
      beta_s = [
          np.random.multivariate_normal(self.mu[i], sigma2_s[i] * self.cov[i])
          for i in range(self.hparams.num_actions)
      ]
    except np.linalg.LinAlgError as e:
      # Sampling could fail if covariance is not positive definite

      d = self.param_dim
      beta_s = [
          np.random.multivariate_normal(np.zeros((d)), np.eye(d))
          for i in range(self.hparams.num_actions)
      ]

    # Compute last-layer representation for the current context
    with self.bnn.graph.as_default():
      c = context.reshape((1, self.hparams.context_dim))
      z_context = self.bnn.sess.run(self.bnn.nn, feed_dict={self.bnn.x: c})
      if self.intercept:
        z_context = np.append(z_context, 1.0).reshape((1, self.latent_dim + 1))
    # Apply Thompson Sampling to last-layer representation
    vals = [
        np.dot(beta_s[i], z_context.T) for i in range(self.hparams.num_actions)
    ]
    return np.argmax(vals)

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

      # Update the latent representation of every datapoint collected so far
      new_z = self.bnn.sess.run(self.bnn.nn,
                                feed_dict={self.bnn.x: self.data_h.contexts})
      self.latent_h.replace_data(contexts=new_z)
      for action_v in range(self.hparams.num_actions):

        # Update action posterior with formulas: \beta | z,y ~ N(mu_q, cov_q)
        z, y = self.latent_h.get_data(action_v)

        # The algorithm could be improved with sequential formulas (cheaper)
        self.precision[action_v] = (np.dot(z.T, z)+self.lambda_prior * np.eye(self.param_dim)) #the new PHI_0
        self.f[action_v] = np.dot(z.T, y)
    else:
      if self.intercept:
        z_context = np.append(z_context, 1.0).reshape((1, self.latent_dim + 1))
      self.precision[action] += np.dot(z_context.T, z_context)
      self.f[action] += (z_context.T * reward)[:, 0]
    self.yy[action] += reward ** 2
    self.cov[action] = np.linalg.inv(self.precision[action])
    self.mu[action] = np.dot(self.cov[action], self.f[action])

    # Inverse Gamma posterior update
    self.a[action] += 0.5
    b_upd = 0.5 * (self.yy[action] - np.dot(self.mu[action].T, np.dot(self.precision[action], self.mu[action])))
    self.b[action] = self.b0 + b_upd

    #print(self.calc_model_evidence())

  @property
  def a0(self):
    return self._a0

  @property
  def b0(self):
    return self._b0

  @property
  def lambda_prior(self):
    return self._lambda_prior
  def calc_model_evidence(self):
    vval = 0
    mp.mp.dps = 50
    for action in range(self.hparams.num_actions):
      #  val=1
      #  aa = self.a[action]
      #  for i in range(int(self.a[action]-self.a0)):
      #      aa-=1
      #      val*=aa
      #      val/=(2.0*math.pi)
      #      val/=self.b[action]
      #  val*=gamma(aa)
      #  val/=(self.b[action]**aa)
      #  val *= np.sqrt(np.linalg.det(self.lambda_prior * np.eye(self.hparams.context_dim + 1)) / np.linalg.det(self.precision[action]))
      #  val *= (self.b0 ** self.a0)
      #  val/= gamma(self.a0)
      #  vval += val
      #val= 1/float((2.0 * math.pi) ** (self.a[action]-self.a0))
      #val*= (float(gamma(self.a[action]))/float(gamma(self.a0)))
      #val*= np.sqrt(float(np.linalg.det(self.lambda_prior * np.eye(self.hparams.context_dim + 1)))/float(np.linalg.det(self.precision[action])))
      #val*= (float(self.b0**self.a0)/float(self.b[action]**self.a[action]))
      val= mp.mpf(mp.fmul(mp.fneg(mp.log(mp.fmul(2.0 , mp.pi))) , mp.fsub(self.a[action],self.a0)))
      val+= mp.loggamma(self.a[action])
      val-= mp.loggamma(self.a0)
      val+= 0.5*mp.log(np.linalg.det(self.lambda_prior * np.eye(self.hparams.context_dim + 1)))
      val -= 0.5*mp.log(np.linalg.det(self.precision[action]))
      val+= mp.fmul(self.a0,mp.log(self.b0))
      val-= mp.fmul(self.a[action],mp.log(self.b[action]))
      vval+=mp.exp(val)


    vval/=float(self.hparams.num_actions)

    return vval