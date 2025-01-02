# In this we will implement some basic functions from the paper
# "Deep Unsupservised Learning using Nonequilibrium Thermodynamics",

import torch

# import numpy as np
import matplotlib.pyplot as plt
import argparse

from sklearn.datasets import make_swiss_roll


def main():
    n_samples = 5000
    data, _ = make_swiss_roll(n_samples=n_samples)

    # Trying to make the same visualization as in the paper
    data = data[:, [2, 0]] / 10  # We will only use the first and third dimension

    # Converting to torch
    data = torch.tensor(data, dtype=torch.float32)

    # Testing the forward function
    T = 40
    # NOTE: Here betas does not give you a good behaved Gaussian distribution
    # betas = torch.linspace(1e-4, 1e-2, T)

    # Here the author crafted a set of betas that gives a good behaved Gaussian distribution
    betas = torch.sigmoid(torch.linspace(-18, 10, T)) * (3e-1 - 1e-5) + 1e-5

    # xT = forward_standard(data, T, betas)
    xT_20 = Diffusion(T).forward_closed_form(data, t=20)
    xT_40 = Diffusion(T).forward_closed_form(data, t=40)

    # Plotting the data at the first and the last noisy step after T steps
    # into 2 subplots
    fig, ax = plt.subplots(1, 3, figsize=(10, 5))
    ax[0].scatter(data[:, 0], data[:, 1])
    ax[0].set_title("Original Data")
    ax[1].scatter(xT_20[:, 0], xT_20[:, 1], alpha=0.25)
    ax[1].set_title("T=20")
    ax[2].scatter(xT_40[:, 0], xT_40[:, 1], alpha=0.25)
    ax[2].set_title("T=40")

    plt.show()

    pass


class Diffusion:

    def __init__(self, T):
        self.T = T
        # Here the author crafted a set of betas that gives a good behaved Gaussian distribution
        self.betas = torch.sigmoid(torch.linspace(-18, 10, T)) * (3e-1 - 1e-5) + 1e-5
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, 0)

    def forward_standard(self, data):
        """
        Basically applying the  Forward difusion kernel from `Tab App 1` in the paper.
        """

        for t in range(self.T):
            beta_t = self.betas[t]
            mu = data * torch.sqrt(1 - beta_t)
            std = torch.sqrt(beta_t)
            # Sampling from the normal distribution q(x_t | x_{t-1})
            # data = mu + torch.random.randn(data.shape[0], data.shape[1]) * std # Data ~ N(mu, std)
            data = mu + torch.randn_like(data) * std  # Data ~ N(mu, std)

        return data

    def forward_closed_form(self, x0, t=40):
        """
        Applying the closed form solution for the forward diffusion kernel from `Tab App 1` in the paper.

        The idea is that since `q(x_t | x_{t-1})` is a Gaussian distribution, we can directly compute
        `q(x_t | x_0)` in a closed form.

        If you do the math:

        $x^{(t)} = \prod\limits_{i=1}^{t} = \sqrt{1-\beta_i}x^{(0)} + \sqrt{1-\prod\limits_{i=1}^{t}} \epsilon$

        So defining $\alpha_t = 1 - \beta_t$ and $a_t = \prod\limits_{i=1}^{t} \alpha_i$, we can write:

        x^{(t)} = \sqrt{a_t}x^{(0)} + \sqrt{1-a_t} \epsilon$

        In this way you will no longer need the loop for
        """
        assert t > 0, "t must be greater than 0"

        t = t - 1  # Since the paper is 1-indexed

        mu = x0 * torch.sqrt(self.alpha_bar[t])
        sigma = torch.sqrt(1 - self.alpha_bar[t])
        epsilon = torch.randn_like(x0)

        return mu + sigma * epsilon


def forward_standard(data, T, betas):
    """
    Basically applying the  Forward difusion kernel from `Tab App 1` in the paper.
    """

    for t in range(T):
        beta_t = betas[t]
        mu = data * torch.sqrt(1 - beta_t)
        std = torch.sqrt(beta_t)
        # Sampling from the normal distribution q(x_t | x_{t-1})
        # data = mu + torch.random.randn(data.shape[0], data.shape[1]) * std # Data ~ N(mu, std)
        data = mu + torch.randn_like(data) * std  # Data ~ N(mu, std)

    return data


if __name__ == "__main__":
    main()
