import numpy as np
import torch
from torch.nn import functional as F
import torch.nn as nn
import math
import os
from PIL import Image
import PIL
from tqdm import tqdm
from torchvision import datasets, transforms
import glob
import random
import matplotlib.pyplot as plt
dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
to_image = transforms.ToPILImage()


def load_pictures():
    bs = 64
    dataset = datasets.ImageFolder(root='./RL/vae/dataset_img/crop_data/crop_64_dataset/', transform=transforms.Compose([
    #dataset = datasets.ImageFolder(root='./RL/vae/dataset_img/crop_data/dataset/', transform=transforms.Compose([
        transforms.ToTensor(),
    ]))
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=bs, shuffle=True)
    print(len(dataset.imgs)) 
    print(len(dataloader))
    return dataloader



def reparameterize(means, logvar):
    stds = (0.5*logvar).exp()
    noises = torch.randn_like(means)
    acts = means + noises * stds
    return acts


class Flatten(nn.Module):
    def forward(self, inputs):
        return inputs.contiguous().view(inputs.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, inputs, size=512):#256
        #ans = inputs.view(inputs.size(0), size, 3, 8)
        ans = inputs.view(inputs.size(0), size, 2, 5)
        
        return ans


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=5120, z_dim=64):#6144#4096
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=3, stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64,  kernel_size=3, stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 256,  kernel_size=3, stride=2,padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 512,  kernel_size=3, stride=2,padding=1),
            nn.ReLU(),
            Flatten()
        ).to(dev)

        self.fc1 = nn.Linear(h_dim, z_dim).to(dev)
        self.fc2 = nn.Linear(h_dim, z_dim).to(dev)
        self.fc3 = nn.Linear(z_dim, h_dim).to(dev)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=2, stride=2),
            nn.Sigmoid(),
        ).to(dev)

    def reparameterize(self, mu, logvar):
        std = logvar.mul(0.5).exp_()
        esp = torch.randn(*mu.size()).to(dev)
        z = mu + std * esp
        return z

    def bottleneck(self, h):
        mu, logvar = self.fc1(h), F.softplus(self.fc2(h))
        z = self.reparameterize(mu, logvar)
        return z, mu, logvar

    def encode(self, x):
        h = self.encoder(x)
        z, mu, logvar = self.bottleneck(h)
        return z, mu, logvar

    def decode(self, z):
        z = self.fc3(z)
        z = self.decoder(z)
        return z

    def forward(self, x):
        z, mu, logvar = self.encode(x)
        r_image = self.decode(z)
        return r_image, mu, logvar, z

    def loss_fn(self, images, r_image, mean, logvar ,real=False):
        #KL = -0.5 * torch.sum((1 + logvar - mean.pow(2) - logvar.exp()), dim=0)
        KL = -0.5 * torch.sum(1 + logvar - mean**2 - torch.exp(logvar))
        if real :
            KL = torch.mean(KL)
        #print(images.size())
        #print(r_image.size())
        r_image = r_image.contiguous().view(-1, 30720)#38400
        images = images.contiguous().view(-1, 30720)#38400
        #print(images.size())
        r_image_loss = F.binary_cross_entropy(r_image, images, reduction='sum') 
        loss = r_image_loss+ 5.0 * KL 
        return loss,KL,r_image_loss

    def evaluate(self, image,epoch,output_dir):
        #plt.ion()
        r_image, mean, log_var, z = self.forward(image)
        pre_im = to_image(image[0].clone().detach().cpu().squeeze(0))
        im_now = to_image(r_image[0].clone().detach().cpu().squeeze(0))
        #z = to_image(z[0].clone().detach().cpu())
        plt.imshow(pre_im)
        plt.show()
        #plt.imsave(f'{output_dir}/original_{epoch}.png',pre_im)
        #plt.pause(0.1)
        plt.imshow(im_now)
        #plt.imsave(f'{output_dir}/Generate_{epoch}.png',im_now)
        #plt.pause(0.1)
        #plt.imshow(z)
        #plt.pause(0.1)
        #plt.figure()
        plt.show()


def train_vae(vae, epochs, train_datas,output_dir,real):
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    vae.train()
    flag = False
    for epoch in range(epochs):
        losses = []
        tmp = 0
        for idx, (images, _) in enumerate(train_datas):
            images = images.to(dev)
            optimizer.zero_grad()
            #recon_images, mu, logvar = vae(images)
            recon_images, mu, logvar, z = vae(images)
            
            loss,kl,re = vae.loss_fn(images, recon_images, mu, logvar,real)
            loss.backward()
            optimizer.step()
            losses.append(loss.cpu().detach().numpy())
            
        print("EPOCH: {} loss: {}".format(epoch+1, np.average(losses)))
        print(f'kl:{kl} re:{re}')

        
        if epoch%10==0:
            vae.evaluate(images,epoch,output_dir)
        torch.save(vae.cpu().state_dict(), f'{output_dir}/vae.pth')
        vae.to(dev)
        flag = False
        
    vae.evaluate(images,epoch,output_dir)





def main():
    output_dir='./output/test'
    real=False
    vae = VAE()
    pics = load_pictures()
    train_vae(vae, 50, pics,output_dir,real)


if __name__ == "__main__":
    main()
