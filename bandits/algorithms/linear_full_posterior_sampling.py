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

"""Contextual algorithm that keeps a full linear posterior for each arm."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from scipy.stats import invgamma

from bandits.core.bandit_algorithm import BanditAlgorithm
from bandits.core.contextual_dataset import ContextualDataset
from scipy.special import gamma,gammaln

import math
import mpmath as mp
class LinearFullPosteriorSampling(BanditAlgorithm):
  """Thompson Sampling with independent linear models and unknown noise var."""

  def __init__(self, name, hparams):
    """Initialize posterior distributions and hyperparameters.

    Assume a linear model for each action i: reward = context^T beta_i + noise
    Each beta_i has a Gaussian prior (lambda parameter), each sigma2_i (noise
    level) has an inverse Gamma prior (a0, b0 parameters). Mean, covariance,
    and precision matrices are initialized, and the ContextualDataset created.

    Args:
      name: Name of the algorithm.
      hparams: Hyper-parameters of the algorithm.
    """

    self.name = name
    self.hparams = hparams

    if hasattr(hparams, 'ucb'):
      self.ucb = self.hparams.ucb
    else:
      self.ucb = False
    if hasattr(hparams, 'ucb_eta'):
      self.ucb_eta = self.hparams.ucb_eta
    else:
      self.ucb_eta = 0.001
    # Gaussian prior for each beta_i
    self._lambda_prior = self.hparams.lambda_prior

    self.mu = [
        np.zeros(self.hparams.context_dim + 1)
        for _ in range(self.hparams.num_actions)
    ]
    self.f= [
        np.zeros(self.hparams.context_dim + 1)
        for _ in range(self.hparams.num_actions)
    ]
    self.yy = [0 for _ in range(self.hparams.num_actions)]
    self.cov = [(1.0 / self.lambda_prior) * np.eye(self.hparams.context_dim + 1)
                for _ in range(self.hparams.num_actions)]

    self.precision = [
        self.lambda_prior * np.eye(self.hparams.context_dim + 1)
        for _ in range(self.hparams.num_actions)
    ]

    # Inverse Gamma prior for each sigma2_i
    self._a0 = self.hparams.a0
    self._b0 = self.hparams.b0

    self.a = [self._a0 for _ in range(self.hparams.num_actions)]
    self.b = [self._b0 for _ in range(self.hparams.num_actions)]

    self.t = 0
    self.intercept = True
    self.data_h = ContextualDataset(hparams.context_dim,
                                    hparams.num_actions,
                                    intercept=self.intercept)

  def action(self, context):
    """Samples beta's from posterior, and chooses best action accordingly.

    Args:
      context: Context for which the action need to be chosen.

    Returns:
      action: Selected action for the context.
    """

    # Round robin until each action has been selected "initial_pulls" times
    if self.t < self.hparams.num_actions * self.hparams.initial_pulls:
      return self.t % self.hparams.num_actions

    # Sample sigma2, and beta conditional on sigma2
    sigma2_s = [
        self.b[i] * invgamma.rvs(self.a[i])
        for i in range(self.hparams.num_actions)
    ]
    if self.ucb:
      if self.intercept:
        c = np.array(context[:])
        c = np.append(c, 1.0).reshape((1, self.hparams.context_dim + 1))
      else:
        c = np.array(context[:]).reshape((1, self.hparams.context_dim))
      try:
        vals = [self.mu[i] @ c.T + np.sqrt(np.squeeze(self.ucb_eta * c @ (sigma2_s[i] * self.cov[i]) @ c.T)) for i in
                range(self.hparams.num_actions)]
      except:
        d = self.latent_dim
        vals = [np.sqrt(np.squeeze(self.ucb_eta * c @ (sigma2_s[i] * np.eye(d)) @ c.T)) for i in range(self.hparams.num_actions)]
    else:
      try:
        beta_s = [
            np.random.multivariate_normal(self.mu[i], sigma2_s[i] * self.cov[i])
            for i in range(self.hparams.num_actions)
        ]
      except np.linalg.LinAlgError as e:
        # Sampling could fail if covariance is not positive definite

        d = self.hparams.context_dim + 1
        beta_s = [
            np.random.multivariate_normal(np.zeros((d)), np.eye(d))
            for i in range(self.hparams.num_actions)
        ]

      # Compute sampled expected values, intercept is last component of beta
      vals = [
          np.dot(beta_s[i][:-1], context.T) + beta_s[i][-1]
          for i in range(self.hparams.num_actions)
      ]

    return np.argmax(vals)

  def update(self, context, action, reward):
    """Updates action posterior using the linear Bayesian regression formula.

    Args:
      context: Last observed context.
      action: Last observed action.
      reward: Last observed reward.
    """

    self.t += 1
    self.data_h.add(context, action, reward)
    if self.intercept:
      c = np.array(context[:])
      c = np.append(c, 1.0).reshape((1, self.hparams.context_dim + 1))
    else:
      c = np.array(context[:]).reshape((1, self.hparams.context_dim))
    # Update posterior of action with formulas: \beta | x,y ~ N(mu_q, cov_q)
    #x, y = self.data_h.get_data(action)

    # Some terms are removed as we assume prior mu_0 = 0.
    self.precision[action] += np.dot(c.T, c)
    self.f[action] += (c.T*reward)[:,0]
    self.yy[action] += reward**2
    self.cov[action] = np.linalg.inv(self.precision[action])
    self.mu[action] = np.dot(self.cov[action], self.f[action])

    # Inverse Gamma posterior update
    self.a[action] +=  0.5
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