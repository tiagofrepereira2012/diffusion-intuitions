import torch
from torchvision.datasets import CIFAR10, MNIST
import torch.nn as nn
import argparse
from tqdm import tqdm
from ddpm_paper import DiffusionModel

# from unet import UNet
from pathlib import Path
import matplotlib.pyplot as plt
import imageio
import numpy as np

cifar10 = CIFAR10(root="data", download=True)
mnist = MNIST(root="data", download=True)




def sample_batch_cifar10(batch_size, device="cpu"):

    # Getting the training and test tensor
    x_train = torch.from_numpy(cifar10.data[:40_000]).float() / 255.0

    # Move to channel first
    x_train = x_train.permute(0, 3, 1, 2)

    indexes = torch.randperm(x_train.shape[0])[:batch_size]

    data = x_train[indexes].to(device)

    return data


def sample_batch_mnist(batch_size, device="cpu"):
    # Not my best moment here, but fuck it
    mnist = MNIST(root="data", download=True)

    # Getting the training and test tensors
    # x = mnist.data[:50_000].unsqueeze(1).float() / 255.0
    # Test set
    x = mnist.data[50_000:].unsqueeze(1).float() / 255.0

    indexes = torch.randperm(x.shape[0])[:batch_size]

    data = x[indexes].to(device)
    # Interpolating to 32 to not break the code
    data = nn.functional.interpolate(data, 32)

    return data


def main():

    # MODEL_PATH = Path("paper_2_cifar10") / "model_30000.pt"
    MODEL_PATH = Path("paper_2") / "model_35000.pt"
    T = 1000
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = torch.load(MODEL_PATH, map_location=device)

    # diffusion_model = DiffusionModel(
    #    1000, model, device=device, sample_function=sample_batch_cifar10
    # )

    diffusion_model = DiffusionModel(
        1000, model, device=device, sample_function=sample_batch_mnist
    )

    # x0 = sample_batch_cifar10(1, device=device)
    # x0 = sample_batch_mnist(1, device)

    all_steps = animate(diffusion_model, image_channels=1, img_size=(32, 32))

    # Taking 100 samples uniformly from all_steps
    samples = all_steps[::10]

    import ipdb

    ipdb.set_trace()

    # imageio.mimsave("movie.gif", samples)

    samples = np.array(samples, dtype=np.uint8)

    # Channel last
    samples = samples.transpose(0, 2, 3, 1)
    imageio.mimsave("movie.mp4", samples)

    # plt.imshow(all_steps[-1], cmap="gray")
    # plt.savefig("movie.png")


@torch.no_grad()
def animate(diffusion_model, image_channels=1, img_size=(32, 32)):
    """
    This is the algorithm 2 from the paper.s
    """

    n_samples = 1
    all_steps = []

    x_t = torch.randn(
        (n_samples, image_channels, img_size[0], img_size[1]),
        device=diffusion_model.device,
    )

    for t in tqdm(range(diffusion_model.T, 1, -1)):

        z = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)  # z ~ N(0, I)

        # x_{t-1} = mean + sigma * z # look at line 4

        t = torch.ones(n_samples, device=diffusion_model.device, dtype=torch.long) * t

        beta_t = (
            diffusion_model.beta[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )  # sigma = sqrt(beta_t)
        alpha_t = diffusion_model.alpha[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        alpha_bar_t = (
            diffusion_model.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )
        sigma = torch.sqrt(
            beta_t
        )  # As mentioned in the paper, the variance has a fixed size

        epsilon_theta = diffusion_model.base_model(
            x_t, t
        )  # epsilon_theta = epsilon_{\theta}(x_t, t)
        mean = (1 / torch.sqrt(alpha_t)) * (
            x_t - ((1 - alpha_t) / (torch.sqrt(1 - alpha_bar_t))) * epsilon_theta
        )  # mean = (1 / sqrt(alpha_t)) * (x_t - ((1-alpha_t)/(sqrt(1-alpha_bar_t)))*epsilon_theta)

        # x_t = mean + sigma * z
        x_t = mean + sigma * z
        # all_steps.append(x_t[0].cpu().numpy())
        all_steps.append((x_t[0].cpu().numpy() * 255).astype(np.uint8))

    return all_steps


def forward(diffusion_model, big_t, x0):

    x_forward = []
    x = x0

    for t in range(big_t):
        # mu = torch.sqrt(diffusion_model.alpha_bar[t-1]) * x0

        # + (
        #    1 - torch.sqrt(diffusion_model.alpha_bar[t-1])
        # ) * diffusion_model.base_model(
        #    x0, t
        # )  # Look at section 3.2 of the paper

        # sigma = torch.sqrt(diffusion_model.beta[t-1])  # Look at section 3.2 of the paper
        # sigma = torch.sqrt(1-diffusion_model.alpha_bar[t-1])

        sigma = torch.sqrt(diffusion_model.beta[t - 1])
        x = x + sigma * torch.randn_like(x0)
        x_forward.append(x)

    return x_forward


if __name__ == "__main__":
    main()
