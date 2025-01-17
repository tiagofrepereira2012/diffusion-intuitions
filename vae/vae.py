# Auto-Encoder Variational Bayes (VAE) implementation
import torch
import torch.nn as nn
from torch.nn import functional as F
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

from scipy.stats import norm
import scipy.io

from torchvision.datasets import MNIST


def get_mnist():
    # Not my best moment here, but fuck it
    mnist = MNIST(root="data", download=True)

    # Getting the training and test tensors
    x_train = mnist.data[:50_000].unsqueeze(1).float() / 255.0
    x_test =  mnist.data[50_000:].unsqueeze(1).float() / 255.0
    
    # Interpolating to 32 to not break the code
    x_train = nn.functional.interpolate(x_train, 32)
    x_test = nn.functional.interpolate(x_test, 32)
    
    return x_train, x_test

def get_minibatch(x, batch_size, device="cpu"):
    
    indexes = torch.randperm(x.shape[0])[:batch_size]

    data = x[indexes].to(device)
    # Transforming B, C, H,W in B, C*H*W
    data = data.view(data.size(0), -1)    
    #data = data.view(data.size(0), 32, 32)
    
    # Interpolating to 32 to not break the code
    #data = nn.functional.interpolate(data, 32)

    return data


class Model(nn.Module):
    
    def __init__(self, data_dim=2, context_dim=2, hidden_dim=200, constrain_mean=False):
        super(Model, self).__init__()
        """
        This will model p(y|x) as N(mu, sigma) where here mu and sigma are parametrized with NN
        """
        
        self.h = nn.Sequential(
            nn.Linear(context_dim, hidden_dim),
            nn.Tanh()
        )
        self.log_var = nn.Sequential(nn.Linear(hidden_dim, data_dim),)
        
        if constrain_mean:
            self.mu = nn.Sequential(nn.Linear(hidden_dim, data_dim), nn.Sigmoid())
        else:
            self.mu = nn.Sequential(nn.Linear(hidden_dim, data_dim),)
            
        
    def get_mean_and_log_var(self, x):
        
        h = self.h(x)
        mu = self.mu(h)
        log_var = self.log_var(h)
        
        return mu, log_var 
    
    def forward(self, epsilon, x):
        """
        Sample y from p(y|x) = N(mu, sigma) using the reparametrization trick
        """        
        mu, log_var = self.get_mean_and_log_var(x)
        
        sigma = torch.sqrt(torch.exp(log_var))
                        
        return mu + sigma * epsilon
    
    def compute_log_density(self, y, x):
        """
        Compute the log(p(y|x)) for a given y and x
        """
        
        mu, log_var = self.get_mean_and_log_var(x)
        
        log_density = -0.5 * torch.log(2 * torch.tensor(np.pi)) + log_var + (((y-mu) ** 2) / (2 * torch.exp(log_var) + 1e-10)) 
        
        return log_density
        
    def compute_kl(self, x):
        """
        Compute the KL divergence between q(y|x) and p(y)
        
        Here we assume that p(x) = N(0, I)
        """
        
        mu, log_var = self.get_mean_and_log_var(x)
        
        kl = -0.5 * (1 + log_var - mu ** 2 - torch.exp(log_var)).sum(dim=1)
        
        return kl


def train_vae(encoder, decoder, encoder_optimizer, decoder_optimizer, nb_epochs, M=100, L=1, latent_dim=2, device="cpu"):
    """
    
    Args:
        encoder: The encoder model
        decoder: The decoder model
        encoder_optimizer: The optimizer for the encoder
        decoder_optimizer: The optimizer for the decoder
        nb_epochs: The number of epochs
        M: The batch size in the paper
        L: The number of samples to take
        latent_dim: The latent dimension
        device: The device to use
    """
    losses = []    
    
    x_train, x_test = get_mnist()

    import ipdb; ipdb.set_trace()
        
    for epoch in range(nb_epochs):
        x = get_minibatch(x_train, M, device=device)
        
        # Sample epsilon 
        epsilon = torch.normal(torch.zeros(M* L, latent_dim), torch.ones(latent_dim)).to(device)
        
        # Computing the loss
        z = encoder(epsilon, x)
        log_likelihoods = decoder.compute_log_density(x, z)
        kl_divergence = decoder.compute_kl(z)
        loss = (kl_divergence - log_likelihoods.view(-1, L).mean(dim=1)).mean()
        
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        loss.backward()
        encoder_optimizer.step()
        decoder_optimizer.step()
        
        losses.append(loss.item())
        
        if epoch % 100 == 0:
            print(f"Epoch {epoch} Loss: {loss.item()}")
        
    return losses

def main():
    
    img_size = (32, 32)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    encoder = Model(data_dim=2, context_dim=img_size[0]*img_size[1], hidden_dim=200, constrain_mean=False).to(device)
    decoder = Model(data_dim=img_size[0]*img_size[1], context_dim=2, hidden_dim=200, constrain_mean=True).to(device)
    
    encoder_optimizer = torch.optim.Adam(encoder.parameters(), lr=1e-3)
    decoder_optimizer = torch.optim.Adam(decoder.parameters(), lr=1e-3)
    
    losses = train_vae(encoder, decoder, encoder_optimizer, decoder_optimizer, 1000, M=100, L=1, latent_dim=2, device=device)
    
    pass
    
    

if __name__ == "__main__":
    main()
 
