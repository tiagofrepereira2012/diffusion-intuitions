# In this we will implement some basic functions from the paper 
# "Deep Unsupservised Learning using Nonequilibrium Thermodynamics",

import torch
#import numpy as np
import matplotlib.pyplot as plt
import argparse

from sklearn.datasets import make_swiss_roll


def main():
    n_samples = 5000
    data, _ = make_swiss_roll(n_samples=n_samples)

    # Trying to make the same visualization as in the paper    
    data = data[:, [2, 0]] / 10 # We will only use the first and third dimension

    # Converting to torch
    data = torch.tensor(data, dtype=torch.float32)

    # Testing the forward function
    T = 40
    # NOTE: Here betas does not give you a good behaved Gaussian distribution
    betas = torch.linspace(1e-4, 1e-2, T)

    # Here the author crafted a set of betas that gives a good behaved Gaussian distribution
    #betas = torch.sigmoid(torch.linspace(-18, 10, T)).numpy() * (3e-1 - 1e-5) + 1e-5



    xT = forward(data, T, betas)

    # Plotting the data at the first and the last noisy step after T steps
    # into 2 subplots
    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    ax[0].scatter(data[:, 0], data[:, 1])
    ax[0].set_title("Original Data")
    ax[1].scatter(xT[:, 0], xT[:, 1], alpha=0.1)
    ax[1].set_title("Noisy Data after T steps")
    plt.show()


    #plt.scatter(data[:, 0], data[:, 1], c='r')
    #plt.show()


    pass

def forward(data, T, betas):
    """
    Basically applying the  Forward difusion kernel from `Tab App 1` in the paper.
    """

    for t in range(T):
        beta_t = betas[t]
        mu = data * torch.sqrt(1 - beta_t)
        std = torch.sqrt(beta_t)
        # Sampling from the normal distribution
        #data = mu + torch.random.randn(data.shape[0], data.shape[1]) * std # Data ~ N(mu, std)
        data = mu + torch.randn_like(data) * std  # Data ~ N(mu, std)

    return data

if __name__ == "__main__":
    main()

