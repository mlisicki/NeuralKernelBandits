# Online Limited Memory Neural-Linear Bandits with Likelihood Matching #

This library corresponds to the "Online Limited Memory Neural-Linear Bandits with Likelihood Matching" paper, ICML 2021.
[[paper]](https://arxiv.org/abs/2102.03799)

The code is based on the "Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling" github repository https://github.com/tensorflow/models/tree/master/research/deep_contextual_bandits, published in ICLR 2018.

### Datasets ###

All the datasets can be found in https://archive.ics.uci.edu/ml/index.php, and should be placed under  contextual_bandtis/datasets folder. 

### How to run the code? ###
Run at terminal:
```
 python3 main.py 
```
### How to configure the experiment? ###
At the main.py, set the method into:
1. neural-linear-lm (Our method)
2. neural-linear (full memory NeuralTS)
3. linear (LinearTS)
4. neural-linear-ntk (NTK version of limited memory NeuralTS)

At dataset sepcify the wanted dataset (an unknown dataset will cause an error).
For amazon dataset, LinearTS do not work. 

### Requirements ###

* tensorflow-gpu 1.15
* absl-py 0.11
* scipy 1.5.4

