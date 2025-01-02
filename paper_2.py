# In this file we will go step-by-step the paper
# "Denoising Diffusion Probabilistic Models" by Ho et al.

# Using the UNet architecture from the course.
from unet import UNet
import argparse

import torch
import torch.nn as nn
from tqdm import tqdm
import matplotlib.pyplot as plt

# Importing the mnist dataset
from torchvision.datasets import MNIST, CIFAR10
from pathlib import Path


# PATH = Path("./paper_2")


class DiffusionModel(nn.Module):
    def __init__(self, T, base_model: nn.Module, sample_function, device="cpu"):

        super(DiffusionModel, self).__init__()

        # self.betas = torch.sigmoid(torch.linspace(-18, 10, T)) * (3e-1 - 1e-5) + 1e-5
        self.beta = torch.linspace(1e-4, 0.02, T).to(device)
        self.alpha = 1 - self.beta

        self.alpha_bar = torch.cumprod(self.alpha, 0)
        self.T = T
        self.base_model = base_model.to(device)  # The function approximator
        self.device = device
        self.sample_function = sample_function

    def train_one_step(self, optimizer, batch_size=32):
        """
        This is the algorithm 1 from the paper.
        Pretty simple
        """

        # x0 = sample_batch(batch_size, self.device)
        x0 = self.sample_function(batch_size, self.device)

        t = torch.randint(
            1, self.T + 1, (batch_size,), device=self.device, dtype=torch.long
        )
        # Sampling epsilon from N(0, I)
        epsilon = torch.randn_like(x0)

        # Moving to the loss function
        # $||\epsilon - \epsilon_{\theta}(\sqrt{\alpha_t}x_0 + \sqrt{1-\alpha_t}\epsilon),t||^2$  # alpha is alpha_bar

        # These unsequeeze are to make the dimensions compatible of doing the element-wise multiplication below
        alpha_bar_t = (
            self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        )  # Since the paper is 1-indexed # [B,C,H,W]

        # Epsilon is approximated by the model
        eps_pred = self.base_model(
            torch.sqrt(alpha_bar_t) * x0 + torch.sqrt(1 - alpha_bar_t) * epsilon, t
        )

        # loss = nn.MSELoss(eps_pred, epsilon)
        loss = nn.functional.mse_loss(eps_pred, epsilon)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        return loss.item()

    @torch.no_grad()
    def sample(self, n_samples=1, n_channels=1, img_size=(32, 32)):
        """
        This is the algorithm 2 from the paper.s
        """

        x_t = torch.randn(n_samples, n_channels, *img_size).to(
            self.device
        )  # x_0 ~ N(0, I)

        for t in range(self.T, 1, -1):

            z = torch.randn_like(x_t) if t > 1 else torch.zeros_like(x_t)  # z ~ N(0, I)

            # x_{t-1} = mean + sigma * z # look at line 4

            t = torch.ones(n_samples, device=self.device, dtype=torch.long) * t

            beta_t = (
                self.beta[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            )  # sigma = sqrt(beta_t)
            alpha_t = self.alpha[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            alpha_bar_t = (
                self.alpha_bar[t - 1].unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
            )
            sigma = torch.sqrt(
                beta_t
            )  # As mentioned in the paper, the variance has a fixed size

            epsilon_theta = self.base_model(
                x_t, t
            )  # epsilon_theta = epsilon_{\theta}(x_t, t)
            mean = (1 / torch.sqrt(alpha_t)) * (
                x_t - ((1 - alpha_t) / (torch.sqrt(1 - alpha_bar_t))) * epsilon_theta
            )  # mean = (1 / sqrt(alpha_t)) * (x_t - ((1-alpha_t)/(sqrt(1-alpha_bar_t)))*epsilon_theta)

            x_t = mean + sigma * z

        return x_t


def sample_batch_mnist(batch_size, device="cpu"):
    # Not my best moment here, but fuck it
    mnist = MNIST(root="data", download=True)

    # Getting the training and test tensors
    x_train = mnist.data[:50_000].unsqueeze(1).float() / 255.0
    
    indexes = torch.randperm(x_train.shape[0])[:batch_size]

    data = x_train[indexes].to(device)
    # Interpolating to 32 to not break the code
    data = nn.functional.interpolate(data, 32)

    return data

cifar10 = CIFAR10(root="data", download=True)
def sample_batch_cifar10(batch_size, device="cpu"):
    
    
    # Getting the training and test tensor
    x_train = torch.from_numpy(cifar10.data[:40_000]).float() / 255.0
    
    # Move to channel first
    x_train = x_train.permute(0,3,1,2)
    
    indexes = torch.randperm(x_train.shape[0])[:batch_size]
    
    data = x_train[indexes].to(device)
    
    return data

def main():

    parser = argparse.ArgumentParser(
        description="Denoising Diffusion Probabilistic Models"
    )
    parser.add_argument("output_path", type=Path)
    parser.add_argument("--epochs", type=int, default=40_000, help="Number of epochs")

    args = parser.parse_args()

    print("Running with number of epochs: ", args.epochs)

    batch_size = 64
    n_channels = 3  
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # device = "cpu"
    model = UNet(in_ch=n_channels)
    optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
    
    #sample_batch_cifar10
    #sample_batch_mnist
    difusion_model = DiffusionModel(
        1000, model, device=device, sample_function=sample_batch_cifar10
    )

    # Creating path
    #PATH.mkdir()
    path = args.output_path
    path.mkdir(exist_ok=True)

    training_loss = []

    for epoch in tqdm(range(args.epochs)):
        loss = difusion_model.train_one_step(optimizer, batch_size=batch_size)
        training_loss.append(loss)

        # Plotting the training loss every 100 epochs
        if epoch > 0 and epoch % 1000 == 0:
            _, ax = plt.subplots()
            ax.plot(training_loss)
            plt.savefig(path / f"training_loss_{epoch}.png")

            _, ax = plt.subplots()
            ax.plot(training_loss[-1000:])
            # plt.show()
            plt.savefig(path / f"training_loss_crop_{epoch}.png")

        if epoch > 0 and epoch % 5_000 == 0:
            # Saving the model
            # torch.save(model.state_dict(), f"model_{epoch}.pt")
            nb_images = 9
            samples = difusion_model.sample(nb_images, n_channels=n_channels)
            # Created a 3x3 figure
            fig, axs = plt.subplots(3, 3)
            for i in range(nb_images):
                ax = axs[i // 3, i % 3]                
                #ax.imshow(samples[i].squeeze(0).clip(0, 1).cpu().numpy(), cmap="gray")
                img = samples[i].permute(1,2,0).clip(0, 1).cpu().numpy()
                ax.imshow(img)
                ax.axis("off")
            plt.savefig(path / f"samples_{epoch}.png")

            # Saving the model
            torch.save(model.cpu(), path / f"model_{epoch}.pt")
            # Moving the model back to the device
            model.to(device)
            
    # Saving the final model
    torch.save(model.cpu(), path / "model_final.pt")


if __name__ == "__main__":
    main()
