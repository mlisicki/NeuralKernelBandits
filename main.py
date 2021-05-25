from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import flags
from absl import app
import numpy as np
import os
import tensorflow as tf
import multiprocessing

num_cores = multiprocessing.cpu_count()

# from bandits.algorithms.bootstrapped_bnn_sampling import BootstrappedBNNSampling
from bandits.core.contextual_bandit import run_contextual_bandit
from bandits.data.data_sampler import sample_adult_data
from bandits.data.data_sampler import sample_census_data
from bandits.data.data_sampler import sample_covertype_data
from bandits.data.data_sampler import sample_jester_data
from bandits.data.data_sampler import sample_mushroom_data,sample_txt_data
from bandits.data.data_sampler import sample_statlog_data
from bandits.data.data_sampler import sample_stock_data
from bandits.data.data_sampler import sample_eeg_data,sample_diabetic_data,sample_phone_data,sample_aps_data,sample_amazon_data
from bandits.algorithms.linear_full_posterior_sampling import LinearFullPosteriorSampling
from bandits.algorithms.neural_linear_sampling import NeuralLinearPosteriorSampling
from bandits.data.synthetic_data_sampler import sample_linear_data
from bandits.data.synthetic_data_sampler import sample_sparse_linear_data
from bandits.data.synthetic_data_sampler import sample_wheel_bandit_data,sample_wheel2_bandit_data
from bandits.algorithms.neural_linear_sampling_lm import NeuralLinearPosteriorSamplingLM
from bandits.algorithms.neural_linear_sampling_ntk import NeuralLinearPosteriorSamplingNTK
# Set up your file routes to the data files.
base_route = os.getcwd()
data_route = 'contextual_bandits/datasets'

FLAGS = flags.FLAGS
FLAGS.set_default('alsologtostderr', True)
flags.DEFINE_string('logdir', '/tmp/bandits/', 'Base directory to save output')
flags.DEFINE_string(
    'mushroom_data',
    os.path.join(base_route, data_route, 'mushroom.data'),
    'Directory where Mushroom data is stored.')
flags.DEFINE_string(
    'financial_data',
    os.path.join(base_route, data_route, 'raw_stock_contexts'),
    'Directory where Financial data is stored.')
flags.DEFINE_string(
    'jester_data',
    os.path.join(base_route, data_route, 'jester_data_40jokes_19181users.npy'),
    'Directory where Jester data is stored.')
flags.DEFINE_string(
    'statlog_data',
    os.path.join(base_route, data_route, 'shuttle.trn'),
    'Directory where Statlog data is stored.')
flags.DEFINE_string(
    'adult_data',
    os.path.join(base_route, data_route, 'adult.full'),
    'Directory where Adult data is stored.')
flags.DEFINE_string(
    'covertype_data',
    os.path.join(base_route, data_route, 'covtype.data'),
    'Directory where Covertype data is stored.')
flags.DEFINE_string(
    'census_data',
    os.path.join(base_route, data_route, 'USCensus1990.data.txt'),
    'Directory where Census data is stored.')
flags.DEFINE_string(
    'eeg_data',
    os.path.join(base_route, data_route, 'eeg.csv'),
    'Directory where Census data is stored.')
flags.DEFINE_string(
    'diabetic_data',
    os.path.join(base_route, data_route, 'diabetic.csv'),
    'Directory where Census data is stored.')
flags.DEFINE_string(
    'amazon_data_file',
    os.path.join(base_route, data_route, 'Amazon.csv'),
    'Directory where Census data is stored.')
flags.DEFINE_string(
    'phone_data',
    os.path.join(base_route, data_route, 'samsung.csv'),
    'Directory where Census data is stored.')
flags.DEFINE_string(
    'aps_data',
    os.path.join(base_route, data_route, 'aps.csv'),
    'Directory where Census data is stored.')

flags.DEFINE_string(
    'positive_data_file',
    os.path.join(base_route, data_route, 'rt-polarity.pos'),
    'Directory where Census data is stored.')

flags.DEFINE_string(
    'negative_data_file',
    os.path.join(base_route, data_route, 'rt-polarity.neg'),
    'Directory where Census data is stored.')

flags.DEFINE_integer("task_id",None,"ID of task")

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
  if data_type == '2linear':
    # Create linear dataset
    num_actions = 2
    context_dim = 10
    noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
    dataset, _, opt_linear = sample_linear_data(num_contexts, context_dim,
                                                num_actions, sigma=noise_stds)
    opt_rewards, opt_actions = opt_linear
    return dataset, opt_rewards, opt_actions, num_actions, context_dim, None
  if data_type == 'linear':
    # Create linear dataset
    num_actions = 8
    context_dim = 10
    noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
    dataset, _, opt_linear = sample_linear_data(num_contexts, context_dim,
                                                num_actions, sigma=noise_stds)
    opt_rewards, opt_actions = opt_linear
    return dataset, opt_rewards, opt_actions, num_actions, context_dim, None
  elif data_type == 'sparse_linear':
    # Create sparse linear dataset
    num_actions = 7
    context_dim = 10
    noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
    num_nnz_dims = int(context_dim / 3.0)
    dataset, _, opt_sparse_linear = sample_sparse_linear_data(
        num_contexts, context_dim, num_actions, num_nnz_dims, sigma=noise_stds)
    opt_rewards, opt_actions = opt_sparse_linear
    return dataset, opt_rewards, opt_actions, num_actions, context_dim, None
  elif data_type == 'mushroom':
    # Create mushroom dataset
    num_actions = 2
    context_dim = 117
    file_name = FLAGS.mushroom_data
    dataset, opt_mushroom = sample_mushroom_data(file_name, num_contexts)
    opt_rewards, opt_actions = opt_mushroom
    return dataset, opt_rewards, opt_actions, num_actions, context_dim,None
  elif data_type == 'financial':
    num_actions = 8
    context_dim = 21
    num_contexts = min(3713, num_contexts)
    noise_stds = [0.01 * (i + 1) for i in range(num_actions)]
    file_name = FLAGS.financial_data
    dataset, opt_financial = sample_stock_data(file_name, context_dim,
                                               num_actions, num_contexts,
                                               noise_stds, shuffle_rows=True)
    opt_rewards, opt_actions = opt_financial
    return dataset, opt_rewards, opt_actions, num_actions, context_dim,None
  elif data_type == 'jester':
    num_actions = 8
    context_dim = 32
    num_contexts = min(19181, num_contexts)
    file_name = FLAGS.jester_data
    dataset, opt_jester = sample_jester_data(file_name, context_dim,
                                             num_actions, num_contexts,
                                             shuffle_rows=True,
                                             shuffle_cols=True)
    opt_rewards, opt_actions = opt_jester
    return dataset, opt_rewards, opt_actions, num_actions, context_dim,None
  elif data_type == 'statlog':
    file_name = FLAGS.statlog_data
    num_actions = 7
    num_contexts = min(43500, num_contexts)
    sampled_vals = sample_statlog_data(file_name, num_contexts,
                                       shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
    return dataset, opt_rewards, opt_actions, num_actions, context_dim,None
  elif data_type == 'adult':
    file_name = FLAGS.adult_data
    num_actions = 2
    num_contexts = min(45222, num_contexts)
    sampled_vals = sample_adult_data(file_name, num_contexts,
                                     shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
    return dataset, opt_rewards, opt_actions, num_actions, context_dim,None
  elif data_type == 'covertype':
    file_name = FLAGS.covertype_data
    num_actions = 7
    num_contexts = min(150000, num_contexts)
    sampled_vals = sample_covertype_data(file_name, num_contexts,
                                         shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1] #54
    return dataset, opt_rewards, opt_actions, num_actions, context_dim,None
  elif data_type == 'census':
    file_name = FLAGS.census_data
    num_actions = 9
    num_contexts = min(150000, num_contexts)
    sampled_vals = sample_census_data(file_name, num_contexts,
                                      shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
    return dataset, opt_rewards, opt_actions, num_actions, context_dim,None
  elif data_type == 'wheel':
    delta = 0.5
    num_actions = 5
    context_dim = 2
    mean_v = [0.1,0.1,0.1,0.1,0.2]
    std_v = [0.1, 0.1, 0.1, 0.1, 0.1]
    mu_large = 0.4
    std_large = 0.1
    dataset, opt_wheel = sample_wheel_bandit_data(num_contexts, delta,
                                                  mean_v, std_v,
                                                  mu_large, std_large)
    opt_rewards, opt_actions = opt_wheel

    return dataset, opt_rewards, opt_actions, num_actions, context_dim,None
  elif data_type == 'wheel2':
    delta = 0.7
    num_actions = 2
    context_dim = 2
    mean_v = [0.0, 1]
    std_v = [0.1, 0.1]
    mu_large = 2
    std_large = 0.1
    dataset, opt_wheel = sample_wheel2_bandit_data(num_contexts, delta,
                                                  mean_v, std_v,
                                                  mu_large, std_large)
    opt_rewards, opt_actions = opt_wheel

    return dataset, opt_rewards, opt_actions, num_actions, context_dim,None
  elif data_type == 'eeg': #Epileptic
    file_name = FLAGS.eeg_data
    num_actions = 5
    num_contexts = min(11500, num_contexts)
    sampled_vals = sample_eeg_data(file_name, num_contexts,
                                      shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
    return dataset, opt_rewards, opt_actions, num_actions, context_dim,None
  elif data_type == 'diabetic':
    file_name = FLAGS.diabetic_data
    num_actions = 3
    num_contexts = min(100000, num_contexts)
    sampled_vals = sample_diabetic_data(file_name, num_contexts,
                                      shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
    return dataset, opt_rewards, opt_actions, num_actions, context_dim,None
  elif data_type == 'phone':
    file_name = FLAGS.phone_data
    num_actions = 6
    num_contexts = min(7767, num_contexts)
    sampled_vals = sample_phone_data(file_name, num_contexts,
                                      shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
    return dataset, opt_rewards, opt_actions, num_actions, context_dim,None
  elif data_type == 'aps': #scania
    file_name = FLAGS.aps_data
    num_actions = 2
    num_contexts = min(76000, num_contexts)
    sampled_vals = sample_aps_data(file_name, num_contexts,
                                      shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions) = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
    return dataset, opt_rewards, opt_actions, num_actions, context_dim,None
  elif data_type == 'txt':
    file_name = [FLAGS.positive_data_file,FLAGS.negative_data_file]
    num_actions = 2
    num_contexts = min(10000, num_contexts)
    sampled_vals = sample_txt_data(file_name, num_contexts,
                                      shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions),vocab_processor = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
    return dataset, opt_rewards, opt_actions, num_actions, context_dim,vocab_processor
  elif data_type == 'amazon':
    file_name = FLAGS.amazon_data_file
    num_actions = 5
    num_contexts = min(10000, num_contexts)
    sampled_vals = sample_amazon_data(file_name, num_contexts,shuffle_rows=True)
    contexts, rewards, (opt_rewards, opt_actions),vocab_processor = sampled_vals
    dataset = np.hstack((contexts, rewards))
    context_dim = contexts.shape[1]
    return dataset, opt_rewards, opt_actions, num_actions, context_dim,vocab_processor



def display_final_results(algos, opt_rewards,opt_actions,res, name):
  """Displays summary statistics of the performance of each algorithm."""

  print('---------------------------------------------------')
  print('---------------------------------------------------')
  print('{} bandit completed.'.format(
    name))
  print('---------------------------------------------------')

  performance_triples = []
  for j, a in enumerate(algos):
      performance_triples.append((a.name, np.mean(res[j]),np.std(res[j])))
  performance_pairs = sorted(performance_triples,
                             key=lambda elt: elt[1],
                             reverse=True)
  for i, (name, mean_reward,std_reward) in enumerate(performance_pairs):
    print('{:3}) {:20}| \t \t total reward = {:10} +- {:10}.'.format(i, name, mean_reward,std_reward))

  print('---------------------------------------------------')
  print('Optimal total reward = {}.'.format(np.sum(opt_rewards)))
  print('Frequency of optimal actions (action, frequency):')
  print([[elt, list(opt_actions).count(elt)] for elt in set(opt_actions)])
  print('---------------------------------------------------')
  print('---------------------------------------------------')


def get_algorithm(method, num_actions, context_dim, l_sizes, tfn, tfe, textflag):
    if method == 'linear':
        hparams = tf.contrib.training.HParams(num_actions=num_actions,
                                              context_dim=context_dim,
                                              a0=6,
                                              b0=6,
                                              lambda_prior=0.25,
                                              initial_pulls=2)
        algo = LinearFullPosteriorSampling('LinearTS', hparams)
    elif method == 'neural-linear':

        hparams = tf.contrib.training.HParams(num_actions=num_actions,
                                              context_dim=context_dim,
                                              init_scale=0.3,
                                              activation=tf.nn.relu,
                                              layer_sizes=l_sizes,
                                              batch_size=num_actions * 16,
                                              activate_decay=True,
                                              initial_lr=0.1,
                                              max_grad_norm=5.0,
                                              show_training=False,
                                              freq_summary=1000,
                                              buffer_s=-1,
                                              initial_pulls=2,
                                              reset_lr=True,
                                              lr_decay_rate=0.5,
                                              training_freq=1,
                                              training_freq_network=tfn,
                                              training_epochs=tfe,
                                              a0=6,
                                              b0=6,
                                              lambda_prior=0.25,
                                              verbose=False)
        algo= NeuralLinearPosteriorSampling('NeuralLinearTS', hparams, textflag=textflag)

    elif method == 'neural-linear-lm':
        hparams = tf.contrib.training.HParams(num_actions=num_actions,
                                              context_dim=context_dim,
                                              init_scale=0.3,
                                              activation=tf.nn.relu,
                                              layer_sizes=l_sizes,
                                              batch_size=num_actions * 16,
                                              activate_decay=True,
                                              initial_lr=0.1,
                                              max_grad_norm=5.0,
                                              show_training=False,
                                              freq_summary=1000,
                                              buffer_s=-1,
                                              initial_pulls=2,
                                              reset_lr=True,
                                              lr_decay_rate=0.5,
                                              training_freq=1,
                                              training_freq_network=1,  # tfn,
                                              training_epochs=1,  # tfe,
                                              a0=6,
                                              b0=6,
                                              lambda_prior=1,
                                              mem=num_actions * 100,
                                              mu_prior_flag=1,
                                              sigma_prior_flag=1,
                                              pgd_freq=1,
                                              pgd_steps=1,
                                              pgd_batch_size=20,
                                              verbose=False)
        algo = NeuralLinearPosteriorSamplingLM('NeuralLinearTS-LM', hparams, textflag=textflag)

    elif method == 'neural-linear-ntk':
        hparams = tf.contrib.training.HParams(num_actions=num_actions,
                                              context_dim=context_dim,
                                              init_scale=0.3,
                                              activation=tf.nn.relu,
                                              layer_sizes=l_sizes,
                                              batch_size=num_actions * 16,
                                              activate_decay=True,
                                              initial_lr=0.1,
                                              max_grad_norm=5.0,
                                              show_training=False,
                                              freq_summary=1000,
                                              buffer_s=-1,
                                              initial_pulls=2,
                                              reset_lr=True,
                                              lr_decay_rate=0.5,
                                              training_freq=1,
                                              training_freq_network=1,  # tfn,
                                              training_epochs=1,  # tfe,
                                              a0=6,
                                              b0=6,
                                              lambda_prior=1,
                                              mem=num_actions * 100,
                                              mu_prior_flag=1,
                                              sigma_prior_flag=1,
                                              pgd_freq=1,
                                              pgd_steps=2,
                                              pgd_batch_size=20,
                                              verbose=False)
        algo = NeuralLinearPosteriorSamplingNTK('NeuralLinearTS-NTK', hparams,
                                                        textflag=textflag)  #
    else:
        assert False,'method name is unknown.'
    return algo

def experiment(method, dataset):

  # Problem parameters
  num_contexts = 5000
  tfn=400
  tfe=tfn*2
  data_type = dataset
  l_sizes=[50]
  outdir  ="./"
  Nruns = 10
  if data_type == 'amazon':
      textflag = 'yes'
  else:
      textflag = 'no'

  # Create dataset
  sampled_vals = sample_data(data_type, num_contexts)
  dataset, opt_rewards, opt_actions, num_actions, context_dim,vocab_processor = sampled_vals

  if not os.path.exists(outdir):
      os.makedirs(outdir)

  res = np.zeros((1,num_contexts))
  totalreward=[0]
  rewards = [[]]

  for i_run in range(Nruns):
      algo = get_algorithm(method, num_actions, context_dim, l_sizes, tfn, tfe, textflag)
      algos = [algo]
      results = run_contextual_bandit(context_dim, num_actions, dataset, algos)
      h_actions, h_rewards = results
      for j, a in enumerate(algos):
          print(np.sum(h_rewards[:, j]))
          totalreward[j]+=((np.sum(h_rewards[:, j]))/Nruns)
          rewards[j].append((np.sum(h_rewards[:, j])))
      actions = [[] for i in range(len(h_actions[0]))]
      for aa in h_actions:
          for i, a in enumerate(aa):
              actions[i].append(a)
      for i_alg in range(len(algos)):
          res[i_alg,:]+=1*((actions[i_alg] != opt_actions))

      print('Run number {}'.format(i_run+1))
      display_final_results(algos, opt_rewards, opt_actions, rewards, data_type)

  display_final_results(algos,opt_rewards,opt_actions,rewards,data_type)


def set_gpu(gpu_id=0):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)


def main(argv):

    method = 'neural-linear-lm' # linear/ neural-linear/ neural-linear-lm/ neural-linear-ntk
    gpu = 0
    dataset = 'statlog'
    set_gpu(gpu)
    experiment(method, dataset)

if __name__ == "__main__":
    app.run(main)


