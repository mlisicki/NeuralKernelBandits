"""Copyright 2021 Michal Lisicki

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License."""

import os
import pickle as pkl
import time

import numpy as np
import tensorflow as tf
from absl import app, flags

from bandits.algorithms.linear_full_posterior_sampling import (
    LinearFullPosteriorSampling)
from bandits.algorithms.uniform_sampling import UniformSampling
from bandits.core.contextual_bandit import run_contextual_bandit
from bandits.data.data_sampler import (sample_adult_data, sample_census_data,
                                       sample_covertype_data,
                                       sample_jester_data, sample_mushroom_data,
                                       sample_statlog_data, sample_stock_data)
from bandits.data.synthetic_data_sampler import sample_linear_data

# Set up your file routes to the data files.
BASE_ROUTE = os.getcwd()
DATA_ROUTE = 'contextual_bandits/datasets'

# experiment output directory
OUTDIR = "./outputs/"

FLAGS = flags.FLAGS
FLAGS.set_default('alsologtostderr', True)

# Hyperparameters
flags.DEFINE_integer('seed', None, 'Random seed')
flags.DEFINE_list(
    'methods', ['linucb'], 'Methods list. Choose between: uniform '
    '/ linucb . You can specify multiple '
    'methods in a list. Warning: Running multiple NKs will '
    'result in a heavy computational load.')
flags.DEFINE_boolean('joint', False, 'Use a joint or disjoint model')
flags.DEFINE_boolean('normalizey', False,
                     'Normalize the targets before passing them to GP')
flags.DEFINE_string('nkmode', 'rand_prior', 'NK GP posterior type')
flags.DEFINE_float('nkreg', 0.2, 'NK regularizer')
flags.DEFINE_integer('nlayers', 2, 'Number of layers in neural models')
flags.DEFINE_float('eta', 0.1, 'Bandit exploration parameter')
flags.DEFINE_integer('steps', 5000, 'Number of MAB steps')
flags.DEFINE_integer('trainfreq', 1, 'Training frequency of NK bandits')

flags.DEFINE_string('logdir', '/tmp/bandits/', 'Base directory to save output')

flags.DEFINE_string('mushroom_data',
                    os.path.join(BASE_ROUTE, DATA_ROUTE, 'mushroom.data'),
                    'Directory where Mushroom data is stored.')
flags.DEFINE_string('financial_data',
                    os.path.join(BASE_ROUTE, DATA_ROUTE, 'raw_stock_contexts'),
                    'Directory where Financial data is stored.')
flags.DEFINE_string(
    'jester_data',
    os.path.join(BASE_ROUTE, DATA_ROUTE, 'jester_data_40jokes_19181users.npy'),
    'Directory where Jester data is stored.')
flags.DEFINE_string('statlog_data',
                    os.path.join(BASE_ROUTE, DATA_ROUTE, 'shuttle.trn'),
                    'Directory where Statlog data is stored.')
flags.DEFINE_string('adult_data',
                    os.path.join(BASE_ROUTE, DATA_ROUTE, 'adult.full'),
                    'Directory where Adult data is stored.')
flags.DEFINE_string('covertype_data',
                    os.path.join(BASE_ROUTE, DATA_ROUTE, 'covtype.data'),
                    'Directory where Covertype data is stored.')
flags.DEFINE_string(
    'census_data', os.path.join(BASE_ROUTE, DATA_ROUTE,
                                'USCensus1990.data.txt'),
    'Directory where Census data is stored.')

flags.DEFINE_integer("task_id", None, "ID of task")


class HParams(dict):

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    self.__dict__ = self


def sample_data(data_type, num_contexts=None):
  """Sample data from given 'data_type'.

  Args:
    data_type: Dataset from which to sample.
    num_contexts: Number of contexts to sample.

  Returns:
    dataset: Sampled matrix with rows: (context, reward_1, ..., reward_num_act).
    opt_rewards: Vector of expected optimal reward for each context.
    opt_actions: Vector of optimal action for each context.
    num_actions: Number of available actions.
    context_dim: Dimension of each context.
  """
  if data_type == 'linear':
    # Create linear dataset
    num_actions = 8
    context_dim = 10
    noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
    dataset, _, opt_linear = sample_linear_data(num_contexts,
                                                context_dim,
                                                num_actions,
                                                sigma=noise_stds)
    opt_rewards, opt_actions = opt_linear
    return dataset, opt_rewards, opt_actions, num_actions, context_dim, None

  elif data_type == 'mushroom':
    # Create mushroom dataset
    num_actions = 2
    context_dim = 117
    file_name = FLAGS.mushroom_data
    dataset, opt_mushroom = sample_mushroom_data(file_name, num_contexts)
    opt_rewards, opt_actions = opt_mushroom
    return dataset, opt_rewards, opt_actions, num_actions, context_dim, None

  elif data_type == 'financial':
    num_actions = 8
    context_dim = 21
    num_contexts = min(3713, num_contexts)
    noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
    file_name = FLAGS.financial_data
    dataset, opt_financial = sample_stock_data(file_name,
                                               context_dim,
                                               num_actions,
                                               num_contexts,
                                               noise_stds,
                                               shuffle_rows=True)
    opt_rewards, opt_actions = opt_financial
    return dataset, opt_rewards, opt_actions, num_actions, context_dim, None

  elif data_type == 'jester':
    num_actions = 8
    context_dim = 32
    num_contexts = min(19181, num_contexts)
    file_name = FLAGS.jester_data
    dataset, opt_jester = sample_jester_data(file_name,
                                             context_dim,
                                             num_actions,
                                             num_contexts,
                                             shuffle_rows=True,
                                             shuffle_cols=True)
    opt_rewards, opt_actions = opt_jester
    return dataset, opt_rewards, opt_actions, num_actions, context_dim, None

  elif data_type == 'statlog':
    file_name = FLAGS.statlog_data
    num_actions = 7
    num_contexts = min(43500, num_contexts)
    sampled_vals = sample_statlog_data(file_name,
                                       num_contexts,
                                       shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
    return dataset, opt_rewards, opt_actions, num_actions, context_dim, None

  elif data_type == 'adult':
    file_name = FLAGS.adult_data
    num_actions = 2
    num_contexts = min(45222, num_contexts)
    sampled_vals = sample_adult_data(file_name, num_contexts, shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
    return dataset, opt_rewards, opt_actions, num_actions, context_dim, None

  elif data_type == 'covertype':
    file_name = FLAGS.covertype_data
    num_actions = 7
    num_contexts = min(150000, num_contexts)
    sampled_vals = sample_covertype_data(file_name,
                                         num_contexts,
                                         shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]  # 54
    return dataset, opt_rewards, opt_actions, num_actions, context_dim, None

  elif data_type == 'census':
    file_name = FLAGS.census_data
    num_actions = 9
    num_contexts = min(150000, num_contexts)
    sampled_vals = sample_census_data(file_name,
                                      num_contexts,
                                      shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
    return dataset, opt_rewards, opt_actions, num_actions, context_dim, None


def display_final_results(algos, opt_rewards, opt_actions, res, name):
  """Displays summary statistics of the performance of each algorithm."""

  print('---------------------------------------------------')
  print('---------------------------------------------------')
  print('{} bandit completed.'.format(name))
  print('---------------------------------------------------')

  performance_triples = []
  for j, a in enumerate(algos):
    performance_triples.append((a.name, np.mean(res[j]), np.std(res[j])))

  performance_pairs = sorted(performance_triples,
                             key=lambda elt: elt[1],
                             reverse=True)

  for i, (name, mean_reward, std_reward) in enumerate(performance_pairs):
    print('{:3}) {:20}| \t \t total reward = {:10} +- {:10}.'.format(
        i, name, mean_reward, std_reward))

  print('---------------------------------------------------')
  print('Optimal total reward = {}.'.format(np.sum(opt_rewards)))
  print('Frequency of optimal actions (action, frequency):')
  print([[elt, list(opt_actions).count(elt)] for elt in set(opt_actions)])
  print('---------------------------------------------------')
  print('---------------------------------------------------')


def get_algorithm(method, num_actions, context_dim):
  if method == 'linucb':
    hparams = HParams(num_actions=num_actions,
                      context_dim=context_dim,
                      ucb=True,
                      ucb_eta=0.1,
                      a0=6,
                      b0=6,
                      lambda_prior=0.25,
                      initial_pulls=3)
    algo = LinearFullPosteriorSampling('LinearUCB / LinFullPost', hparams)

  elif method == 'uniform':
    # Uniform and Fixed
    hparams = HParams(num_actions=num_actions)
    algo = UniformSampling('Uniform Sampling', hparams)

  else:
    raise ValueError(f"Method name {method} is not found")

  return algo


def experiment(methods, dataset, token):
  # Problem parameters
  num_contexts = FLAGS.steps
  data_type = dataset
  Nruns = 1

  # Create dataset
  sampled_vals = sample_data(data_type, num_contexts)
  dataset, opt_rewards, opt_actions, num_actions, context_dim, vocab_processor = sampled_vals

  os.makedirs(OUTDIR, exist_ok=True)

  res = np.zeros((len(methods), len(dataset)))
  totalreward = [0] * len(methods)
  rewards = [[]] * len(methods)

  for i_run in range(Nruns):

    algos = [
        get_algorithm(method, num_actions, context_dim) for method in methods
    ]

    results = run_contextual_bandit(context_dim, num_actions, dataset, algos)

    h_actions, h_rewards, optimal_actions, optimal_rewards, times = results

    for j, a in enumerate(algos):
      print(np.sum(h_rewards[:, j]))
      totalreward[j] += ((np.sum(h_rewards[:, j])) / Nruns)
      rewards[j].append((np.sum(h_rewards[:, j])))

    actions = [[] for i in range(len(h_actions[0]))]
    for aa in h_actions:
      for i, a in enumerate(aa):
        actions[i].append(a)

    for i_alg in range(len(algos)):
      res[i_alg, :] += 1 * ((actions[i_alg] != opt_actions))

    pkl_path = os.path.join(
        OUTDIR, "linucb_experiment_{}_{}_run{}_{}.pkl".format(
            num_contexts, str(token), str(i_run), data_type))

    with open(pkl_path, "wb") as fp:
      # Collect experiment statistics
      pkl.dump(
          {
              'desc': 'LinUCB experiment',
              'seed': FLAGS.seed,
              'times': times,
              'models': [alg.name for alg in algos],
              'dataset': data_type,
              'hparams': [dict(alg.hparams) for alg in algos],
              'flags': FLAGS.flag_values_dict(),
              'actions': h_actions,
              'rewards': h_rewards,
              'opt_actions': optimal_actions,
              'opt_rewards': optimal_rewards,
              'opt_actions_data': opt_actions,
              'opt_rewards_data': opt_rewards
          }, fp)

    print('Run number {}'.format(i_run + 1))
    display_final_results(algos, opt_rewards, opt_actions, rewards, data_type)

  display_final_results(algos, opt_rewards, opt_actions, rewards, data_type)


def main(argv):
  timestr = time.strftime("%Y%m%d-%H%M%S")
  token = timestr + "_" + str(np.random.randint(9999))
  print(token)

  if FLAGS.seed is not None:
    np.random.seed(FLAGS.seed)
    tf.random.set_seed(FLAGS.seed)

  methods = FLAGS.methods
  datasets = [
      'financial', 'jester', 'statlog', 'adult', 'covertype', 'census',
      'mushroom'
  ]

  for dataset in datasets:
    print("================")
    print(dataset)
    print("================")
    experiment(methods, dataset, token)


if __name__ == "__main__":
  app.run(main)
