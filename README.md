# Neural Kernel Bandits

Neural kernel bandits are contextual bandit algorithms guided by a neural kernel-induced Gaussian process predictive distribution. The model is most suitable for small data (per arm) structured problems, requiring non-linear function approximation and accurate exploration strategy. The implementation is a part of a larger contextual bandit framework, introduced by [1] and expanded by [2].

Currently, the project provides access the following neural kernels:

* Neural tangent kernel (NTK)
* Conjugate kernel (CK, aka NNGP)

and GP predictive distributions:

* NNGP
* Deep ensembles
* Randomized Priors
* NTKGP

as specified in [3] (Table 1) and implemented in neural-tangents library (link). The predictive distribution inform the following bandit policies:

* Upper Confidence Bounds (UCB)
* Thompson Sampling (TS)

## Citing the work

This project accompanies the paper:

Lisicki, Michal, Arash Afkanpour, and Graham W. Taylor. "An Empirical Study of Neural Kernel Bandits." Neural Information Processing Systems (NeurIPS) Workshop on Bayesian Deep Learning, 2021. https://arxiv.org/abs/2111.03543.

### BibTeX

```
@inproceedings{lisicki2021empirical,
  title={An Empirical Study of Neural Kernel Bandits},
  author={Lisicki, Mihal and Afkanpour, Arash and Taylor, Graham W},
  booktitle={Neural Information Processing Systems (NeurIPS) Workshop on Bayesian Deep Learning},
  year={2021}  
}
```

## Dependencies

To install the dependencies, enter a Python 3.7+ virtual environment of your choice, and run:

```bash
python -m pip install -r requirements.txt
```

## How to download datasets?

```bash
cd contextual_bandits/datasets/
wget -i wget_list.txt
```

## How to run an experiment?

Run the script with default parameters to perform a full experiment with NK-TS. Optionally change the training frequency to perform a significantly faster run without much loss in overall performance:

```bash
python neural_kernel_experiment.py [--trainfreq=20]
```

To list all the available options, type:

```bash
python neural_kernel_experiment.py --help
```

For consistency in reporting the results, I recommend running the script with a fixed seed (`--seed` flag). All the experiments in the paper were run with seeds in range `1234-1244`.

## How to analyze the results?

The results are saved in the `./outputs` directory. The experiment file names include the general name of the experiment and the most significant hyperparameters. Plots and a summary can obtained by running:

```bash
python analyze_results.py
```

## Acknowledgements

We thank the [Vector AI](https://vectorinstitute.ai/) Engineering team (Gerald Shen, [Maria Koshkina](https://mkoshkina.github.io/) and Deval Pandya) for code review.

## References

[1] Riquelme, Carlos, George Tucker, and Jasper Snoek. “Deep Bayesian Bandits Showdown: An Empirical Comparison of Bayesian Deep Networks for Thompson Sampling.” *ArXiv:1802.09127 [Cs, Stat]*, February 25, 2018. http://arxiv.org/abs/1802.09127.

[2] Nabati, Ofir, Tom Zahavy, and Shie Mannor. “Online Limited Memory Neural-Linear Bandits with Likelihood Matching.” *ArXiv:2102.03799 [Cs]*, June 8, 2021. http://arxiv.org/abs/2102.03799.

[3] He, Bobby, Balaji Lakshminarayanan, and Yee Whye Teh. “Bayesian Deep Ensembles via the Neural Tangent Kernel.” ArXiv:2007.05864 [Cs, Stat], October 24, 2020. http://arxiv.org/abs/2007.05864.

Bandits Code based on repos: [Online Limited Memory Neural-Linear Bandits with Likelihood Matching](https://github.com/ofirnabati/Neural-Linear-Bandits-with-Likelihood-Matching) and [Deep Bayesian Bandits Library](https://github.com/tensorflow/models/tree/archive/research/deep_contextual_bandits)
