import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from pathlib import Path
from tqdm import tqdm
import matplotlib.pyplot as plt


import argparse

# Define the Encoder
class Encoder(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, latent_dim)

    def forward(self, x):
        h = torch.relu(self.fc1(x))
        z = self.fc2(h)
        return z

# Define the Decoder
class Decoder(nn.Module):
    def __init__(self, latent_dim=20, hidden_dim=400, output_dim=784):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(latent_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, z):
        h = torch.relu(self.fc1(z))
        x_recon = self.sigmoid(self.fc2(h))
        return x_recon

# Define the Vector Quantizer
class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost=0.25):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost

        self.embeddings = nn.Embedding(num_embeddings, embedding_dim)
        self.embeddings.weight.data.uniform_(-1 / num_embeddings, 1 / num_embeddings)

    def forward(self, z):
        
        # Flatten input
        z_flattened = z.view(-1, self.embedding_dim)

        # Calculate distances
        distances = (torch.sum(z_flattened**2, dim=1, keepdim=True)
                     - 2 * torch.matmul(z_flattened, self.embeddings.weight.t())
                     + torch.sum(self.embeddings.weight**2, dim=1))

        # Get closest embeddings
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=z.device)
        encodings.scatter_(1, encoding_indices, 1)

        # Quantize and compute loss
        z_q = torch.matmul(encodings, self.embeddings.weight).view(z.shape)
        loss = torch.mean((z_q.detach() - z)**2) + self.commitment_cost * torch.mean((z_q - z.detach())**2)

        # Pass gradients
        z_q = z + (z_q - z).detach()

        return z_q, loss

# Define the VQ-VAE
class VQVAE(nn.Module):
    def __init__(self, input_dim=784, hidden_dim=400, latent_dim=20, num_embeddings=512):
        super(VQVAE, self).__init__()
        self.encoder = Encoder(input_dim, hidden_dim, latent_dim)
        self.decoder = Decoder(latent_dim, hidden_dim, input_dim)
        self.vq = VectorQuantizer(num_embeddings, latent_dim)

    def forward(self, x):
        z = self.encoder(x)
        z_q, vq_loss = self.vq(z)
        x_recon = self.decoder(z_q)
        return x_recon, vq_loss

# Loss function
def loss_function(recon_x, x, vq_loss):
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    return recon_loss + vq_loss

def main():

    parser = argparse.ArgumentParser(description='Train a VAE on MNIST')
    parser.add_argument("output_path", type=Path, help="Path to save the model and plot")
    parser.add_argument("--only-plot", action="store_true", help="Only plot the images")

    args = parser.parse_args()
    
    only_plot = args.only_plot
    output_path = args.output_path


    # Training settings
    batch_size = 128
    epochs = 10
    learning_rate = 1e-3
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model, optimizer
    input_dim = 784
    hidden_dim = 400
    latent_dim = 20
    num_embeddings = 512
    
    
    if only_plot:

        vqvae = torch.load(output_path)

        # Sampling and plotting
        vqvae.eval()
        with torch.no_grad():
            #z = torch.randn(64, latent_dim).to(device)
            z = torch.linspace(-5, 5, 64).view(-1, 1).repeat(1, latent_dim).to(device)
            sample = vqvae.decoder(z).cpu()

            fig, axes = plt.subplots(8, 8, figsize=(8, 8))
            for i, ax in enumerate(axes.flatten()):
                ax.imshow(sample[i].view(28, 28), cmap='gray')
                ax.axis('off')

            plt.tight_layout()
            plt.savefig("vq_vae.png")
    
    else:
            
        # Data loading
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Lambda(lambda x: x.view(-1).to(device))  # Flatten the image
        ])
        train_dataset = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        vqvae = VQVAE(input_dim, hidden_dim, latent_dim, num_embeddings).to(device)
        optimizer = optim.Adam(vqvae.parameters(), lr=learning_rate)

        # Training loop
        vqvae.train()
        for epoch in tqdm(range(epochs)):
            train_loss = 0
            for batch_idx, (data, _) in enumerate(train_loader):
                data = data.to(torch.float32)
                optimizer.zero_grad()
                recon_batch, vq_loss = vqvae(data)
                loss = loss_function(recon_batch, data, vq_loss)
                loss.backward()
                train_loss += loss.item()
                optimizer.step()

            print(f"Epoch {epoch + 1}, Loss: {train_loss / len(train_loader.dataset):.4f}")

        print("Training complete.")
        torch.save(vqvae, output_path)
        
        
if __name__ == "__main__":
    main()