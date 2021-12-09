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
from collections import defaultdict
from glob import glob
from statistics import median

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from tabulate import tabulate

RESULTS_PATH = "./outputs/"
FIGURE_PATH = "./figures/"
NUM_COLORS = 30
LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
NUM_STYLES = len(LINE_STYLES)
PLOT_SIZE = (14, 8)
FONT_SIZE = 16

CLRS = sns.color_palette('husl', n_colors=NUM_COLORS)

# Plot statistics
DSLIST = [
    'adult', 'census', 'covertype', 'financial', 'jester', 'mushroom', 'statlog'
]

data = {}
reward_table = None

os.makedirs(FIGURE_PATH, exist_ok=True)

for dataset_name in DSLIST:
  data[dataset_name] = {}

  for fn in glob("{}/*{}.pkl".format(RESULTS_PATH, dataset_name)):

    with open(fn, "rb") as fp:
      d = pkl.load(fp)

    for i in range(len(d['models'])):
      hparams = d['hparams'][i]
      model = d['models'][i]

      if hparams['joint']:
        model = "Joint{}".format(model)

      model += "_{}_g{}e{}l{}".format(hparams['mode'], hparams['gamma'],
                                      hparams['eta'], hparams['num_layers'])
      if hparams['training_freq'] > 1:
        model += "f{}".format(hparams['training_freq'])

      if model not in data[dataset_name]:
        data[dataset_name][model] = defaultdict(list)

      data[dataset_name][model]['cum_regret'] += [
          np.cumsum(d['opt_rewards_data'] - d['rewards'][:, i])
      ]
      data[dataset_name][model]['cum_reward'] += [np.cumsum(d['rewards'][:, i])]

      if "times" in d:
        times = np.array(d['times'])
        data[dataset_name][model]['cum_time'] += [
            np.cumsum(times[:, i + 1] - times[:, i])
        ]
        data[dataset_name][model]['times'] += [d['times']]
        data[dataset_name][model]['min_times'] += [
            min(times[:, i + 1] - times[:, i])
        ]
        data[dataset_name][model]['max_times'] += [
            max(times[:, i + 1] - times[:, i])
        ]
        data[dataset_name][model]['median_times'] += [
            median(times[:, i + 1] - times[:, i])
        ]

  if reward_table is None:
    reward_table = defaultdict(list)
    reward_table_std = defaultdict(list)
    reward_table_full = defaultdict(list)
    times_table = defaultdict(list)
    times_table_std = defaultdict(list)

  plt.figure(figsize=PLOT_SIZE)
  plt.rcParams['font.size'] = FONT_SIZE

  num_exp = []
  for clr_idx, m in enumerate(data[dataset_name]):
    num_exp += [len(np.unique(data[dataset_name][m]['cum_reward'], axis=0))]
    mean_rewards = np.mean(np.unique(data[dataset_name][m]['cum_reward'],
                                     axis=0),
                           axis=0)
    std_rewards = np.std(np.unique(data[dataset_name][m]['cum_reward'], axis=0),
                         axis=0)
    print((dataset_name, m, num_exp[-1]))
    reward_table[m] += [mean_rewards[-1]]
    reward_table_std[m] += [std_rewards[-1]]
    reward_table_full[m] += [
        np.unique(data[dataset_name][m]['cum_reward'], axis=0)
    ]
    plt.plot(mean_rewards,
             label=m,
             color=CLRS[clr_idx],
             linestyle=LINE_STYLES[clr_idx % NUM_STYLES])

  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.ylabel("Cumulative Reward")
  plt.xlabel("Step")
  plt.title(dataset_name +
            " (avg over {}-{} runs)".format(min(num_exp), max(num_exp)))
  plt.savefig(FIGURE_PATH + "reward_{}.pdf".format(dataset_name),
              bbox_inches='tight')
  plt.close()

  plt.figure(figsize=PLOT_SIZE)
  plt.rcParams['font.size'] = FONT_SIZE

  num_exp = []
  for clr_idx, m in enumerate(data[dataset_name]):

    num_exp += [len(np.unique(data[dataset_name][m]['cum_reward'], axis=0))]
    mean_regrets = np.mean(np.unique(data[dataset_name][m]['cum_regret'],
                                     axis=0),
                           axis=0)
    plt.plot(mean_regrets,
             label=m,
             color=CLRS[clr_idx],
             linestyle=LINE_STYLES[clr_idx % NUM_STYLES])

  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.ylabel("Cumulative Regret")
  plt.xlabel("Step")
  plt.title(dataset_name +
            " (avg over {}-{} runs)".format(min(num_exp), max(num_exp)))
  plt.savefig(FIGURE_PATH + "regret_{}.pdf".format(dataset_name),
              bbox_inches='tight')
  plt.close()

  plt.figure(figsize=PLOT_SIZE)
  plt.rcParams['font.size'] = FONT_SIZE

  num_exp = []
  for clr_idx, m in enumerate(data[dataset_name]):

    if m in data[dataset_name].keys() and data[dataset_name][m]['cum_time']:
      num_exp += [len(np.unique(data[dataset_name][m]['cum_time'], axis=0))]
      mean_times = np.mean(np.unique(data[dataset_name][m]['cum_time'], axis=0),
                           axis=0)
      std_times = np.std(np.unique(data[dataset_name][m]['cum_time'], axis=0),
                         axis=0)
      times_table[m] += [mean_times[-1]]
      times_table_std[m] += [std_times[-1]]
      plt.plot(mean_times,
               label=m,
               color=CLRS[clr_idx],
               linestyle=LINE_STYLES[clr_idx % NUM_STYLES])
    else:
      times_table[m] += [None]
      times_table_std[m] += [None]

  plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
  plt.ylabel("Clock Time [s]")
  plt.xlabel("Step")
  plt.title(dataset_name +
            " (avg over {}-{} runs)".format(min(num_exp), max(num_exp)))
  plt.savefig(FIGURE_PATH + "time_{}.pdf".format(dataset_name),
              bbox_inches='tight')
  plt.close()

# Prepare the results table
df = pd.DataFrame(reward_table, index=DSLIST)
df_std = pd.DataFrame(reward_table_std, index=DSLIST)

mask = np.zeros(df.T.shape)
mask[np.argmax(df.T.sort_index().values, axis=0), np.arange(mask.shape[1])] = 1
df_total = (df.astype('int').astype('str') + " ± " +
            df_std.astype('int').astype('str')).T.sort_index()

print(tabulate(df_total, headers='keys', tablefmt='psql'))

with open("{}/table.tex".format(FIGURE_PATH), "w") as f:
  df_total = (df.astype('int').astype('str') + " ± " +
              df_std.astype('int').astype('str')).T

  f.write(df_total.sort_index().style.apply(
      lambda x: np.where(mask, 'bfseries: ;', None),
      axis=None).to_latex(position_float="centering",
                          hrules=True,
                          label="table:1",
                          caption="Cumulative rewards").replace('_', '\_'))
