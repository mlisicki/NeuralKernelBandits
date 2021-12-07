import pickle as pkl
from tabulate import tabulate
import seaborn as sns
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from glob import glob
import os
from statistics import median

RESULTS_PATH= "./outputs/"
FIGURE_PATH = "./figures/"
NUM_COLORS = 30
LINE_STYLES = ['solid', 'dashed', 'dashdot', 'dotted']
NUM_STYLES = len(LINE_STYLES)
PLOT_SIZE = (14, 8)
FONT_SIZE = 16

clrs = sns.color_palette('husl', n_colors=NUM_COLORS)


def parse(s):
    s = s.strip('[]')
    tuples = s.split('), ')
    out = []
    for x in tuples:
        a,b = x.strip('()').split(', ')
        a = eval(a)
        try:
            b = eval(b)
        except (ValueError, SyntaxError):
            b = None
        out.append((a, b))
    return out


def init_dict(keys):
    return  {k: [] for k in keys}

# Plot statistics
dslist = ['adult', 'census', 'covertype', 'financial', 'jester', 'mushroom', 'statlog']
data = {}
try:
    os.makedirs(FIGURE_PATH)
except OSError:
    pass
reward_table = None
for dataset_name in dslist:
    data[dataset_name] = {}
    for fn in glob("{}/*{}.pkl".format(RESULTS_PATH, dataset_name)):
        d = pkl.load(open(fn, "rb"))
        for i in range(len(d['models'])):
            hparams = dict(parse(d['hparams'][i]))
            model = d['models'][i]
            if hparams['joint']:
                model = "Joint{}".format(model)
            model += "_{}_g{}e{}l{}".format(hparams['mode'],hparams['gamma'],hparams['eta'],hparams['num_layers'])
            if hparams['training_freq']>1:
                model += "f{}".format(hparams['training_freq'])
            if model not in data[dataset_name].keys():
                data[dataset_name][model] = {
                    'cum_regret': [],
                    'cum_reward': [],
                    'cum_time': [],
                    'times': [],
                    'min_times': [],
                    'max_times': [],
                    'median_times': []}
            data[dataset_name][model]['cum_regret'] += [np.cumsum(d['opt_rewards_data'] - d['rewards'][:, i])]
            data[dataset_name][model]['cum_reward'] += [np.cumsum(d['rewards'][:, i])]
            if "times" in d:
                times = np.array(d['times'])
                data[dataset_name][model]['cum_time'] += [np.cumsum(times[:,i+1]-times[:,i])]
                data[dataset_name][model]['times'] += [d['times']]
                data[dataset_name][model]['min_times'] += [min(times[:,i+1]-times[:,i])]
                data[dataset_name][model]['max_times'] += [max(times[:,i+1]-times[:,i])]
                data[dataset_name][model]['median_times'] += [median(times[:,i+1]-times[:,i])]
    if reward_table is None:
        keys = data[dataset_name].keys()
        reward_table = init_dict(keys)
        reward_table_std = init_dict(keys)
        reward_table_full = init_dict(keys)
        times_table = init_dict(keys)
        times_table_std = init_dict(keys)

    plt.figure(figsize=PLOT_SIZE)
    plt.rcParams['font.size'] = FONT_SIZE
    num_exp = []
    clr = 0
    for m in data[dataset_name].keys():
        num_exp += [len(np.unique(data[dataset_name][m]['cum_reward'], axis=0))]
        mean_rewards = np.mean(np.unique(data[dataset_name][m]['cum_reward'], axis=0), axis=0)
        std_rewards = np.std(np.unique(data[dataset_name][m]['cum_reward'], axis=0), axis=0)
        print((dataset_name, m, num_exp[-1]))
        reward_table[m] += [mean_rewards[-1]]
        reward_table_std[m] += [std_rewards[-1]]
        reward_table_full[m] += [np.unique(data[dataset_name][m]['cum_reward'], axis=0)]
        plt.plot(mean_rewards, label=m, color=clrs[clr], linestyle=LINE_STYLES[clr % NUM_STYLES])
        clr += 1
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel("Cumulative Reward")
    plt.xlabel("Step")
    plt.title(dataset_name + " (avg over {}-{} runs)".format(min(num_exp), max(num_exp)))
    plt.savefig(FIGURE_PATH + "reward_{}.pdf".format(dataset_name), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=PLOT_SIZE)
    plt.rcParams['font.size'] = FONT_SIZE
    num_exp = []
    clr = 0
    for m in data[dataset_name].keys():
        num_exp += [len(np.unique(data[dataset_name][m]['cum_reward'], axis=0))]
        mean_regrets = np.mean(np.unique(data[dataset_name][m]['cum_regret'], axis=0), axis=0)
        plt.plot(mean_regrets, label=m, color=clrs[clr], linestyle=LINE_STYLES[clr % NUM_STYLES])
        clr += 1
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel("Cumulative Regret")
    plt.xlabel("Step")
    plt.title(dataset_name + " (avg over {}-{} runs)".format(min(num_exp), max(num_exp)))
    plt.savefig(FIGURE_PATH + "regret_{}.pdf".format(dataset_name), bbox_inches='tight')
    plt.close()

    plt.figure(figsize=PLOT_SIZE)
    plt.rcParams['font.size'] = FONT_SIZE
    num_exp = []
    clr = 0
    for m in data[dataset_name].keys():
        if m in data[dataset_name].keys() and data[dataset_name][m]['cum_time']:
            num_exp += [len(np.unique(data[dataset_name][m]['cum_time'],axis=0))]
            mean_times = np.mean(np.unique(data[dataset_name][m]['cum_time'],axis=0),axis=0)
            std_times = np.std(np.unique(data[dataset_name][m]['cum_time'],axis=0),axis=0)
            times_table[m] += [mean_times[-1]]
            times_table_std[m] += [std_times[-1]]
            plt.plot(mean_times, label=m, color=clrs[clr], linestyle=LINE_STYLES[clr%NUM_STYLES])
            clr += 1
        else:
            times_table[m] += [None]
            times_table_std[m] += [None]
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylabel("Clock Time [s]")
    plt.xlabel("Step")
    plt.title(dataset_name + " (avg over {}-{} runs)".format(min(num_exp),max(num_exp)))
    plt.savefig(FIGURE_PATH+"time_{}.pdf".format(dataset_name), bbox_inches='tight')
    plt.close()

# Prepare the results table
df = pd.DataFrame(reward_table,index=dslist)
df_std = pd.DataFrame(reward_table_std,index=dslist)
mask = np.zeros(df.T.shape)
mask[np.argmax(df.T.sort_index().values,axis=0),np.arange(mask.shape[1])] = 1
df_total = (df.astype('int').astype('str')+" ± "+df_std.astype('int').astype('str')).T.sort_index()
print(tabulate(df_total, headers='keys', tablefmt='psql'))

with open("{}/table.tex".format(FIGURE_PATH),"w") as f:
    df_total = (df.astype('int').astype('str')+" ± "+df_std.astype('int').astype('str')).T
    f.write(df_total.sort_index().style.apply(lambda x: np.where(mask, 'bfseries: ;', None),axis=None).to_latex(
        position_float="centering", hrules=True, label="table:1", caption="Cumulative rewards").replace('_','\_'))
