# In this we will implement some basic functions from the paper
# "Deep Unsupservised Learning using Nonequilibrium Thermodynamics",

# The part b of the paper is about the diffusion process visualization.
# Here we will just organize the previous code and rewire some other stuff.

import torch
import torch.nn as nn

# import numpy as np
import matplotlib.pyplot as plt
import argparse
from tqdm import tqdm

from sklearn.datasets import make_swiss_roll


def sample_batch(batch_size, device="cpu"):
    data, _ = make_swiss_roll(n_samples=batch_size)

    # Trying to make the same visualization as in the paper
    data = data[:, [2, 0]] / 10  # We will only use the first and third dimension

    # Converting to torch
    data = torch.tensor(data, dtype=torch.float32).to(device)

    return data


class MLP(nn.Module):
    """
    A simple MLP model to learn the mean and the variance of the Gaussian distribution

    Args:
        N: Number of NNs
        data_dim: Dimension of the input data
        hidden_dim: Hidden dimension of the NNs
    """

    def __init__(self, N=40, data_dim=2, hidden_dim=64):

        super(MLP, self).__init__()

        self.network_head = nn.Sequential(
            nn.Linear(data_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        self.network_tail = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Linear(
                        hidden_dim, data_dim * 2
                    ) ,  # We will output the mean and the and the diagonal of the covariance matrix
                )
                for _ in range(N)
            ]
        )

    def forward(self, x, t):

        h = self.network_head(x)

        # mu, sigma = self.nn_tail[t](h).chunk(
        # 2, dim=-1
        # )  # This will split the output in two parts. Very nice trick, didn't know about it
        tmp = self.network_tail[t](h)
        mu, h = torch.chunk(tmp, 2, dim=1)
        sigma = torch.exp(h)
        std = torch.sqrt(sigma)

        return mu, std


class Diffusion:

    def __init__(self, T, model: nn.Module, dim=2, device="cpu"):
        self.T = T
        # Here the author crafted a set of betas that gives a good behaved Gaussian distribution
        self.betas = torch.sigmoid(torch.linspace(-18, 10, T)) * (3e-1 - 1e-5) + 1e-5
        self.betas = self.betas.to(device)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, 0)
        
        self.model = model
        self.dim = dim
        self.device = device

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

        t -= 1  # Since the paper is 1-indexed

        mu = x0 * torch.sqrt(self.alpha_bar[t])
        sigma = torch.sqrt(1 - self.alpha_bar[t])
        epsilon = torch.randn_like(x0)
        xt = mu + sigma * epsilon


        sigma_q = torch.sqrt((( 1 - self.alpha_bar[t-1])/(1 - self.alpha_bar[t])) * self.betas[t])
        
        # m1 and m2 are two scale factors that multiply x0 and xt
        m1 = torch.sqrt(self.alpha_bar[t-1]) * self.betas[t] / (1 - self.alpha_bar[t])
        #m2 = torch.sqrt(1 - self.alpha_bar[t]) * (1 - self.alpha_bar[t-1]) / (1 - self.alpha_bar[t])
        m2 = torch.sqrt(self.alphas[t]) * (1 - self.alpha_bar[t-1]) / (1 - self.alpha_bar[t])
        mu_q = m1 * x0 + m2 * xt
 

        return mu_q, sigma_q, xt

    def reverse_process(self, xt, t):
        """
        The reversed trajectory is given by the following equation:

        $p(x^{(0..T)}) = p(x^{(T)}) \prod\limits_{t=1}^{T} p(x^{(t-1)} | x^{(t)})$ Eq. 5

        For the reversed trajectory described in Table App 1, you start from a gaussian distribution
        and go back to the original data distribution.
        The gaussian distribution has a mean $\mu$ and a covarians $\sigma$ that we don't know
        its values, and this we learn with a Neural Network.
        """

        assert t >= 0, "t must be greater than or equal to 0"
        assert t <= self.T, "t must be less than T"

        t -= 1  # Since the paper is 1-indexed

        mu, sigma = self.model(xt, t)  # Now the model is responsible to get the mean
        epsilon = torch.randn_like(xt).to(self.device)

        return mu, sigma, mu + sigma * epsilon
    
    def sample(self, batch_size):
        """
        Sample a batch of data from the diffusion process
        """

        noise = torch.randn(batch_size, self.dim).to(self.device)
        x = noise
        samples = [x]
        for t in range(self.T, 0, -1): # Reversing the process
            if not (t == 1):
                _, _, x = self.reverse_process(x, t)
            samples.append(x)

        return samples[::-1] # Reversing the list, since we started from the end
    
    def get_loss(self, x0):
        """
        Get the loss of the diffusion process
        
        Args:
            x0: The initial data point [batch_size, dim]
        """
        
        # First thing to do is to sample t
        t = torch.randint(2, self.T + 1, (1,))
        mu_q, sigma_q, xt = self.forward_closed_form(x0, t) # q(x_t | x_0)
        mu_p, sigma_p, xt_minus1 = self.reverse_process(xt, t) # p(x_{t-1} | x_t)
        
        
        #kl_divergence = torch.distributions.kl.kl_divergence(
        #    torch.distributions.Normal(mu_q, sigma_q),
        #    torch.distributions.Normal(mu_p, sigma_p)
        #)
        
        kl_divergence = torch.log(sigma_p) - torch.log(sigma_q) + (sigma_q ** 2 + (mu_q - mu_p) ** 2) / (2 * sigma_p ** 2) 
        kl = -kl_divergence.mean()
        
        loss = -kl # The loss is the negative of the KL divergence because we want to maximize it
        
        return loss
        
def training_loop(model, optimizer, batch_size, nb_epochs, device="cpu"):
    
    training_loss = []    
    for epochs in tqdm(range(nb_epochs)):
                
        x0 = sample_batch(batch_size, device=device)
        loss = model.get_loss(x0)
    
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        training_loss.append(loss.item())
        
        if epochs % 10_000 == 0:
            plot(model, f"paper_1_partb_{epochs}.png")
        
        
    return training_loss

def plot(diffusion_model, file_name="paper_1_partb.png"):

    device = diffusion_model.device
    x0= sample_batch(3_000).to(device)
    samples = diffusion_model.sample(5_000)
        
    #import ipdb
    #ipdb.set_trace()

    # q(x_t | x_0)    
    _,_,xt_20 = diffusion_model.forward_closed_form(x0, 20)
    _,_,xt_40 = diffusion_model.forward_closed_form(x0, 40)
    

    # p(x_{t-1} | x_t)
    #_,_, xt_minus1_20 = diffusion_model.reverse_process(xt_20, 20)
    #_,_, xt_minus1_40 = diffusion_model.reverse_process(xt_40, 40)


    #fig, ax = plt.subplots(2, 3, figsize=(10, 5), sharex=True, sharey=True)
    _, ax = plt.subplots(2, 3, figsize=(10, 5))
    
    ax[0,0].scatter(x0[:, 0].detach().cpu().numpy(), x0[:, 1].detach().cpu().numpy())
    ax[0,0].set_title("Original Data")
    
    ax[0,1].scatter(xt_20[:, 0].detach().cpu().numpy(), xt_20[:, 1].detach().cpu().numpy(), alpha=0.25)
    ax[0,1].set_title("$T/2$")
    ax[0,2].scatter(xt_40[:, 0].detach().cpu().numpy(), xt_40[:, 1].detach().cpu().numpy(), alpha=0.25)
    ax[0,2].set_title("T")

    ax[1,0].scatter(samples[0][:, 0].detach().cpu().numpy(), samples[0][:, 1].detach().cpu().numpy(), alpha=0.25, color="red")
    ax[1,1].scatter(samples[20][:, 0].detach().cpu().numpy(), samples[20][:, 1].detach().cpu().numpy(), alpha=0.25, color="red")    
    ax[1,2].scatter(samples[40][:, 0].detach().cpu().numpy(), samples[40][:, 1].detach().cpu().numpy(), alpha=0.25, color="red")     
    
    
    #plt.show()
    plt.savefig(file_name)

    

def main():
    n_samples = 5_000
    #device = "cuda" if torch.cuda.is_available() else "cpu"
    device = "cpu"
    #data = sample_batch(n_samples)

    # What we will do basicallz is to take the data and train a NN to learn a
    # set of features that can be used to generate the mean and the variance
    # data --> NN--> h
    #       -> t ->  NN[t] --> sigma

    # Loading a pretrained model
    #mlp_model = torch.load("./data/model_paper1").to(device)    
    mlp_model = MLP(hidden_dim=128).to(device)    
    diffusion_model = Diffusion(40, mlp_model, device=device)
    
    optimizer = torch.optim.Adam(mlp_model.parameters(), lr=1e-2)
    training_loss = training_loop(diffusion_model, optimizer, 64_000, 300_000, device=device)

    plt.plot(training_loss)
    plt.savefig("training_loss.png")
    
    plot(diffusion_model)

    pass


if __name__ == "__main__":
    main()
