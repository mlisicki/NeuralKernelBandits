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

"""Define a data buffer for contextual bandit algorithms."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np


class ContextualDataset(object):
  """The buffer is able to append new data, and sample random minibatches."""

  def __init__(self, context_dim, num_actions, buffer_s=-1, intercept=False):
    """Creates a ContextualDataset object.

    The data is stored in attributes: contexts and rewards.
    The sequence of taken actions are stored in attribute actions.

    Args:
      context_dim: Dimension of the contexts.
      num_actions: Number of arms for the multi-armed bandit.
      buffer_s: Size of buffer for training. Only last buffer_s will be
        returned as minibatch. If buffer_s = -1, all data will be used.
      intercept: If True, it adds a constant (1.0) dimension to each context X,
        at the end.
    """

    self._context_dim = context_dim
    self._num_actions = num_actions
    self._contexts = None
    self._rewards = None
    self.actions = []
    self.buffer_s = buffer_s
    self.intercept = intercept
    self.action_del = True
  def add(self, context, action, reward):
    """Adds a new triplet (context, action, reward) to the dataset.

    The reward for the actions that weren't played is assumed to be zero.

    Args:
      context: A d-dimensional vector with the context.
      action: Integer between 0 and k-1 representing the chosen arm.
      reward: Real number representing the reward for the (context, action).
    """

    if self.intercept:
      c = np.array(context[:])
      c = np.append(c, 1.0).reshape((1, self.context_dim + 1))
    else:
      c = np.array(context[:]).reshape((1, self.context_dim))

    if self.contexts is None:
      self.contexts = c
    else:
      self.contexts = np.vstack((self.contexts, c))
      if len(self.contexts)>self.buffer_s:
        if self.action_del: # per action fifo
          rem_ind = self.actions.index(action)
          self.contexts = np.delete(self.contexts, rem_ind, 0)
        else: # fifo
          self.contexts = np.delete(self.contexts,0,0)

    r = np.zeros((1, self.num_actions))
    r[0, action] = reward
    if self.rewards is None:
      self.rewards = r
    else:
      self.rewards = np.vstack((self.rewards, r))
      if len(self.rewards)>self.buffer_s:
        if self.action_del:  # per action fifo
          self.rewards = np.delete(self.rewards, rem_ind, 0)
        else:
          self.rewards = np.delete(self.rewards,0,0)

    self.actions.append(action)
    if len(self.actions) > self.buffer_s:
      if self.action_del:  # per action fifo
        self.actions.pop(rem_ind)
      else:
        self.actions.pop(0)

  def replace_data(self, contexts=None, actions=None, rewards=None):
    if contexts is not None:
      i_contexts = None
      for context in contexts:
        c = np.array(context[:])
        if self.intercept:
          c = np.append(c, 1.0).reshape((1, self.context_dim+1))
        if i_contexts is None:
          i_contexts = c
        else:
          i_contexts = np.vstack((i_contexts, c))
      self.contexts = i_contexts
    if actions is not None:
      self.actions = actions
    if rewards is not None:
      self.rewards = rewards

  def get_batch(self, batch_size):
    """Returns a random minibatch of (contexts, rewards) with batch_size."""
    n, _ = self.contexts.shape
    ind = np.random.choice(range(n), batch_size)
    return self.contexts[ind, :], self.rewards[ind, :]

  def get_data(self, action):
    """Returns all (context, reward) where the action was played."""
    n, _ = self.contexts.shape
    ind = np.array([i for i in range(n) if self.actions[i] == action])
    if len(ind)>0:
      return self.contexts[ind, :], self.rewards[ind, action]
    else:
      return None, None

  def get_data_with_weights(self):
    """Returns all observations with one-hot weights for actions."""
    weights = np.zeros((self.contexts.shape[0], self.num_actions))
    a_ind = np.array([(i, val) for i, val in enumerate(self.actions)])
    weights[a_ind[:, 0], a_ind[:, 1]] = 1.0
    return self.contexts, self.rewards, weights

  def get_batch_with_weights(self, batch_size):
    """Returns a random mini-batch with one-hot weights for actions."""
    n, _ = self.contexts.shape
    ind = np.random.choice(range(n), batch_size)
    weights = np.zeros((batch_size, self.num_actions))
    sampled_actions = np.array(self.actions)[ind]
    a_ind = np.array([(i, val) for i, val in enumerate(sampled_actions)])
    weights[a_ind[:, 0], a_ind[:, 1]] = 1.0
    return self.contexts[ind, :], self.rewards[ind, :], weights

  def num_points(self, f=None):
    """Returns number of points in the buffer (after applying function f)."""
    if f is not None:
      return f(self.contexts.shape[0])
    return self.contexts.shape[0]

  @property
  def context_dim(self):
    return self._context_dim

  @property
  def num_actions(self):
    return self._num_actions

  @property
  def contexts(self):
    return self._contexts

  @contexts.setter
  def contexts(self, value):
    self._contexts = value

  @property
  def actions(self):
    return self._actions

  @actions.setter
  def actions(self, value):
    self._actions = value

  @property
  def rewards(self):
    return self._rewards

  @rewards.setter
  def rewards(self, value):
    self._rewards = value
