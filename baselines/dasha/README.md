---
title: DASHA: Distributed Nonconvex Optimization with Communication Compression and Optimal Oracle Complexity
url: https://openreview.net/forum?id=VA1YpcNr7ul
labels: [compression, heterogeneous setting, variance reduction, image classification]
dataset: [cifar10, mushrooms, libsvm]
---

# DASHA: Distributed Nonconvex Optimization with Communication Compression and Optimal Oracle Complexity

> Note: If you use this baseline in your work, please remember to cite the original authors of the paper as well as the Flower paper.

****Paper:**** https://openreview.net/forum?id=VA1YpcNr7ul

****Authors:**** Alexander Tyurin, Peter Richtárik

****Abstract:**** We develop and analyze DASHA: a new family of methods for nonconvex distributed optimization problems. When the local functions at the nodes have a finite-sum or an expectation form, our new methods, DASHA-PAGE, DASHA-MVR and DASHA-SYNC-MVR, improve the theoretical oracle and communication complexity of the previous state-of-the-art method MARINA by Gorbunov et al. (2020). In particular, to achieve an $\varepsilon$-stationary point, and considering the random sparsifier Rand$K$ as an example, our methods compute the optimal number of gradients $O\left(\frac{\sqrt{m}}{\varepsilon\sqrt{n}}\right)$ and $O\left(\frac{\sigma}{\varepsilon^{\frac{3}{2}}n}\right)$ in finite-sum and expectation form cases, respectively, while maintaining the SOTA communication complexity $O\left(\frac{d}{\varepsilon \sqrt{n}}\right)$. Furthermore, unlike MARINA, the new methods DASHA, DASHA-PAGE and DASHA-MVR send compressed vectors only, which makes them more practical for federated learning. We extend our results to the case when the functions satisfy the Polyak-Lojasiewicz condition. Finally, our theory is corroborated in practice: we see a significant improvement in experiments with nonconvex classification and training of deep learning models.


## About this baseline

****What’s implemented:**** The code in this directory implements the experiments from the DASHA paper.

****Datasets:**** Mushrooms from LIBSVM and CIFAR10 from PyTorch's Torchvision

****Hardware Setup:**** These experiments were run on a desktop machine with 64 CPU cores. Any machine with 1 CPU would be able to run this code with the mushrooms dataset. The experiments with CIFAR10 would require slightly more CPU resources (e.g., 4 cores would be sufficient) and 1 GPU with CUDA.

****Contributors:**** Alexander Tyurin (https://github.com/k3nfalt)


## Experimental Setup

****Task:**** Image Classification and Linear Regression

****Model:**** This baseline implements two models:
* A logistic regression model with a nonconvex loss from the DASHA paper (Section A.1).
* A neural network with the cross entropy loss (Section A.4).

**Dataset:** This baseline only includes the MNIST dataset. By default, the datasets are partitioned randomly between $n$ clients:

| Dataset | #classes | partitioning method |
| :------ | :---: | :---: |
| mushrooms | 2 | random |
| cifar10 | 10 | random |

****Training Hyperparameters:**** In all experiments, we take parameters of algorithms predicted by the theory, except for the step sizes. In the case of the mushrooms's experiments, the step sizes are fine-tuned from the set of powers of two $\{2^i\,|\,i \in [-10, 10]\}.$ In the case of cifar10's experiments, the step sizes are fixed to $0.01.$


## Environment Setup

To construct the Python environment follow these steps:

```bash
# Install the base Poetry environment
# By default, it is assumed that the user has Python 3.8 and CUDA 11.8. 
# If you have a different setup, then change the "torch" and "torchvision" lines in [tool.poetry.dependencies] to the relevant system setup.
poetry install

# Activate the environment
poetry shell
```


## Running the Experiments

:warning: _Provide instructions on the steps to follow to run all the experiments._
```bash  
# The main experiment implemented in your baseline using default hyperparameters (that should be setup in the Hydra configs) should run (including dataset download and necessary partitioning) by executing the command:

poetry run -m <baseline-name>.main <no additional arguments> # where <baseline-name> is the name of this directory and that of the only sub-directory in this directory (i.e. where all your source code is)

# If you are using a dataset that requires a complicated download (i.e. not using one natively supported by TF/PyTorch) + preprocessing logic, you might want to tell people to run one script first that will do all that. Please ensure the download + preprocessing can be configured to suit (at least!) a different download directory (and use as default the current directory). The expected command to run to do this is:

poetry run -m <baseline-name>.dataset_preparation <optional arguments, but default should always run>

# It is expected that you baseline supports more than one dataset and different FL settings (e.g. different number of clients, dataset partitioning methods, etc). Please provide a list of commands showing how these experiments are run. Include also a short explanation of what each one does. Here it is expected you'll be using the Hydra syntax to override the default config.

poetry run -m <baseline-name>.main  <override_some_hyperparameters>
.
.
.
poetry run -m <baseline-name>.main  <override_some_hyperparameters>
```


## Expected Results

:warning: _Your baseline implementation should replicate several of the experiments in the original paper. Please include here the exact command(s) needed to run each of those experiments followed by a figure (e.g. a line plot) or table showing the results you obtained when you ran the code. Below is an example of how you can present this. Please add command followed by results for all your experiments._

```bash
# it is likely that for one experiment you need to sweep over different hyperparameters. You are encouraged to use Hydra's multirun functionality for this. This is an example of how you could achieve this for some typical FL hyperparameteres

poetry run -m <baseline-name>.main --multirun num_client_per_round=5,10,50 dataset=femnist,cifar10
# the above command will run a total of 6 individual experiments (because 3client_configs x 2datasets = 6 -- you can think of it as a grid).

[Now show a figure/table displaying the results of the above command]

# add more commands + plots for additional experiments.
```
