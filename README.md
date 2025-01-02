# Intuitions on difusion models

In this repository contains a collection of scripts that I used to understand Diffusion models for image generation.

This set of scripts are based on the course from Maxime Vandergar that is shared on Udemy. Course highly recommended. I hope the author does not mind that snippets of his code are used here. If so, please let me know that I will take this repositiory down.

As you can see, I don't use jupyter notebooks, I prefer to use scripts, because I think it is easier to understand the flow of the code and also it is easier to debug. I like a lot using ipdb to debug my code.

**How to follow this README? Please, open the README in one window and the scripts in another window, so you can follow the explanation of the code.**

The installation can be done via pixi. Once this repo is cloned you can install the dependencies by running the following command:


```bash
pixi shell
pip install -r requirements.txt
```


## Basic mechanics of the diffusion model - "Deep Unsupservised Learning using Nonequilibrium Thermodynamics"

In this section the mechanics of the diffusion model are explained and this comes from the paper "Deep Unsupservised Learning using Nonequilibrium Thermodynamics" by Sohl-Dickstein et al. The paper can be found [here](https://arxiv.org/abs/1503.03585).

A Diffusion Probabilistic Model, or Diffusion Model is a generative model modeled using a parametrized Markov Chain that is trained to generate samples from a given distribution (similar to VAE, or GANs).
The so called Difusion Process consists in to each add gaussian noise in each step of the Markov Chain until the input signal is totally corrupted, and then a simple MLP is used to recover the original signal as showed in the image below.

<img src="images/markov_chain.png" alt="Markov Chain" style="width:50%;">

The forward process (corruption phase) is given by:

$q(x_{1:T}|x_0)=\prod\limits_{t=1}^{T}q(x_t|x_{t-1})$

where $q(x_t|x_{t-1})$ can be parametrized as any well behaved distribution, but in this case it is parametrized as a Gaussian distribution as.

$q(x_t|x_{t-1})=\mathcal{N}(x_t ; \sqrt{1-\beta_t}x_{t-1},\beta_tI)$

where $\beta_t$ is the variance schedule of these distributions for each step of the Markov Chain (in an increasing order)
and $x_0$ is your training data.

The reverse process (recovery phase) is untractable to be computed, but it can be approximated by a simple MLP hence:

$p_\theta(x_{0:T})=p_\theta(x_{T}) \prod\limits_{i=1}^{T} p_\theta(x_{t-1}|x_t)$

where 

$p_\theta(x_{t-1}|x_t)=\mathcal{N}(x_{t-1};\mu_\theta(x_t, t), \sigma_\theta(x_t, t)$

where $\mu_\theta(x_t, t)$ and $\sigma_\theta(x_t, t)$ are the outputs of the MLP for each step $t$.
