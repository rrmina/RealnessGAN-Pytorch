import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils import spectral_norm

from torchvision import datasets, transforms
import torchvision

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import skewnorm

# Global Settings
NUM_EPOCHS = 40
BATCH_SIZE = 128
D_LR = 2e-4
G_LR = 2e-4
BETA_1 = 0.5
BETA_2 = 0.999

# Model Hyperparameters
LATENT_DIM = 128
CONV_DIM = 64
NUM_OUTCOMES = 10

class ReshapeLayer(nn.Module):
    def __init__(self, shape):
        super(ReshapeLayer, self).__init__()
        self.shape = shape
    
    def forward(self, x):
        return x.view(*self.shape)

class Generator(nn.Module):
    def __init__(self, latent_dim, conv_dim):
        super(Generator, self).__init__()
        c = conv_dim

        self.model = nn.Sequential(
            nn.Linear(latent_dim, c*8*2*2),
            ReshapeLayer(shape=(-1, c*8, 2, 2)),

            nn.ConvTranspose2d(c*8, c*4, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c*4),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(c*4, c*2, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c*2),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(c*2, c, kernel_size=4, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(c),
            nn.ReLU(inplace=True),

            nn.ConvTranspose2d(c, 3, kernel_size=4, stride=2, padding=1, bias=True),
            nn.Tanh()
        )

    def forward(self, x):
        return self.model(x)

class Discriminator(nn.Module):
    def __init__(self, conv_dim, num_outcomes):
        super(Discriminator, self).__init__()
        c = conv_dim

        self.model = nn.Sequential(
            spectral_norm(nn.Conv2d(  3,   c, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(  c,   c, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(  c, c*2, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(c*2, c*2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(c*2, c*4, kernel_size=3, stride=1, padding=1)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(c*4, c*4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),

            spectral_norm(nn.Conv2d(c*4, c*8, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2),

            ReshapeLayer(shape=(-1, c*8*2*2)),
            spectral_norm(nn.Linear(c*8*2*2, num_outcomes)),
            nn.Softmax()
        )

    def forward(self, x):
        return self.model(x)

def train():
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")
    print("Device: {}".format(device))

    # Dataset and DataLoader
    transform = transforms.Compose([
        transforms.ToTensor()
    ])
    train_dataset = datasets.SVHN('data', split="train", transform=transform, download=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Load Networks
    d = Discriminator(CONV_DIM, NUM_OUTCOMES).to(device)
    g = Generator(LATENT_DIM, CONV_DIM).to(device)

    # Optimizer Settings
    d_optim = optim.Adam(d.parameters(), lr=D_LR, betas=[BETA_1, BETA_2])
    g_optim = optim.Adam(g.parameters(), lr=G_LR, betas=[BETA_1, BETA_2])

    # Helper Functions
    def generate_latent(batch_size, latent_dim):
        return torch.empty(batch_size, latent_dim).uniform_(-1,1).to(device)

    # Generate a fixed latent vector. This will be used 
    # in monitoring the improvement of generator network
    fixed_z = generate_latent(64, LATENT_DIM)

    # Helper Functions
    def scale(tensor, mini=-1, maxi=1):
        return tensor * (maxi - mini) + mini

    def scale_back(tensor, mini=-1, maxi=1):
        return (tensor-mini)/(maxi-mini)

    # Define Anchors
    # # Anchor 0 = Normal
    # gauss = np.random.normal(0, 0.1, 1000)
    # count, bins = np.histogram(gauss, NUM_OUTCOMES)
    # anchor0 = count/sum(count)
    
    # # Anchor 1 = Uniform
    # unif = np.random.uniform(-1, 1, 1000)
    # count, bins = np.histogram(unif, NUM_OUTCOMES)
    # anchor1 = count/sum(count)
    
    # Anchor 0 = Skewed normal to the left
    skew = skewnorm.rvs(-5, size=1000)
    count, bins = np.histogram(skew, NUM_OUTCOMES)
    anchor0 = count / sum(count)

    # Anchor 1 = Skewed normal to the right
    skew = skewnorm.rvs(5, size=1000)
    count, bins = np.histogram(skew, NUM_OUTCOMES)
    anchor1 = count / sum(count)

    A0 = torch.from_numpy(np.array(anchor0)).to(device).float()
    A1 = torch.from_numpy(np.array(anchor1)).to(device).float()

    # Helper Function
    def saveimg(image, savepath):
        image = image.transpose(1,2,0)
        plt.imsave(savepath, image)

    # Compute KL Divergence
    def KLD(P, Q):
        return torch.mean(torch.sum(P * (P/Q).log(), dim=1))

    # Print KLD between the anchords
    print("KLD(A0||A1): {}".format(KLD(A0.view(1, -1), A1)))

    # Global Loss Logger
    losses = {"D": [], "G": []}

    # Train Proper
    for epoch in range(1, NUM_EPOCHS+1):
        print("========Epoch {}/{}========".format(epoch, NUM_EPOCHS))
        epoch_losses = {"D": [], "G": []}

        d.train()
        g.train()

        for real_images, _ in train_loader:
            # Preprocess tensor
            batch_size = real_images.shape[0]
            real_images = real_images.to(device)
            real_images = scale(real_images, -1, 1)

            # Discriminator Real Loss
            d_optim.zero_grad()
            d_real_out = d(real_images)
            d_real_loss = KLD(d_real_out, A1)
           
            # Discriminator Fake Loss
            z = generate_latent(batch_size, LATENT_DIM)
            fake_images = g(z)
            d_fake_out = d(fake_images)
            d_fake_loss = KLD(A0, d_fake_out)

            # Total Discriminator Loss, Backprop, and Gradient Descent
            d_loss = d_real_loss + d_fake_loss
            d_loss.backward()
            d_optim.step()

            # Generator Loss
            g_optim.zero_grad()
            z = generate_latent(batch_size, LATENT_DIM)
            g_images = g(z)
            d_g_out = d(g_images)
            g_loss = KLD(A1, d_g_out)

            # Generator Backprop and Gradient Descent
            g_loss.backward()
            g_optim.step()

            # Record Epoch Losses
            epoch_losses["D"].append(d_loss.item())
            epoch_losses["G"].append(g_loss.item())

        # Record Mean Epoch Losses
        losses["D"].append(np.mean(epoch_losses["D"]))
        losses["G"].append(np.mean(epoch_losses["G"]))
        print("D loss: {} G loss: {}".format(d_loss.item(), g_loss.item()))

        # Generate sample fake images after each epoch
        g.eval()
        with torch.no_grad():
            sample_tensor = g(fixed_z)
            concat_tensor = torchvision.utils.make_grid(sample_tensor)
            sample_images = concat_tensor.clone().detach().cpu().numpy()
            sample_images = scale_back(sample_images).clip(0,1)
            saveimg(sample_images, "resultsRealnessDCGAN/recon"+str(epoch)+".png")

train()