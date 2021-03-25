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
    # dataset = datasets(root='/home/emile/Documents/Code/RL_car/train_data', transform=Picture())
    # dataset = datasets.ImageFolder(root='/home/emile/Documents/Code/RL_car/train_data', transform=transforms.Compose([
    #     transforms.ToTensor(),
    # ]))
    pic_dir = "/home/emile/Documents/Code/RL_car/train_data/pictures"
    file_name = "_cam-image_array_.jpg"
    num_file = sum(os.path.isfile(os.path.join(pic_dir, name)) for name in os.listdir(pic_dir))
    ans = []
    for index in tqdm(range(num_file)):
        path = pic_dir + "/" + str(index) + file_name
        img = np.array(Image.open(path).resize((160, 120)).crop((0, 40, 160, 120)))
        im = torch.from_numpy(img.reshape((1, 80, 160, 3))).to(dev).permute(0, 3, 1, 2).float().to(dev)
        ans.append(im/255.0)
    # ans = torch.utils.data.DataLoader(ans, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    # dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True, num_workers=2, pin_memory=True)
    random.shuffle(ans)
    return ans


def reparameterize(means, logvar):
    stds = (0.5*logvar).exp()
    noises = torch.randn_like(means)
    acts = means + noises * stds
    return acts


class Flatten(nn.Module):
    def forward(self, inputs):
        return inputs.contiguous().view(inputs.size(0), -1)


class UnFlatten(nn.Module):
    def forward(self, inputs, size=256):
        ans = inputs.view(inputs.size(0), size, 3, 8)
        return ans


class VAE(nn.Module):
    def __init__(self, image_channels=3, h_dim=6144, z_dim=32):
        super(VAE, self).__init__()
        self.z_dim = z_dim
        self.encoder = nn.Sequential(
            nn.Conv2d(image_channels, 32, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2),
            nn.ReLU(),
            Flatten()
        ).to(dev)

        self.fc1 = nn.Linear(h_dim, z_dim).to(dev)
        self.fc2 = nn.Linear(h_dim, z_dim).to(dev)
        self.fc3 = nn.Linear(z_dim, h_dim).to(dev)

        self.decoder = nn.Sequential(
            UnFlatten(),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, kernel_size=5, stride=2),
            nn.ReLU(),
            nn.ConvTranspose2d(32, image_channels, kernel_size=4, stride=2),
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

    def loss_fn(self, images, r_image, mean, logvar):
        KL = -0.5 * torch.sum((1 + logvar - mean.pow(2) - logvar.exp()), dim=0)
        KL = torch.mean(KL)
        r_image = r_image.contiguous().view(-1, 38400)
        images = images.contiguous().view(-1, 38400)
        r_image_loss = F.binary_cross_entropy(r_image, images, reduction='mean')  # size_average=False)
        # print("loss reconst {}".format(r_image_loss.clone().cpu().detach().numpy()))
        # print("loss KL {}".format(KL.clone().cpu().detach().numpy()))
        loss = r_image_loss + 5.0 * KL
        # print("loss {}".format(loss.clone().cpu().detach().numpy()))
        return loss

    def evaluate(self, image):
        plt.ion()
        r_image, mean, log_var, z = self.forward(image)
        pre_im = to_image(image.clone().detach().cpu().squeeze(0))
        im_now = to_image(r_image.clone().detach().cpu().squeeze(0))
        z = to_image(z.clone().detach().cpu())
        plt.imshow(pre_im)
        plt.pause(0.1)
        plt.imshow(im_now)
        plt.pause(0.1)
        plt.imshow(z)
        plt.pause(0.1)
        plt.figure()


def train_vae(vae, epochs, train_datas):
    optimizer = torch.optim.Adam(vae.parameters(), lr=1e-3)
    vae.train()
    flag = False
    for epoch in range(epochs):
        losses = []
        tmp = 0
        for data in tqdm(train_datas):
            tmp += 1
            images = data.to(dev)
            # if not flag:
            #     vae.evaluate(images)
            #     flag = True
            r_images, means, log_var, zs = vae(images)
            if tmp == 1:
                loss = vae.loss_fn(images, r_images, means, log_var).to(dev)
            else:
                loss += vae.loss_fn(images, r_images, means, log_var).to(dev)
            if (tmp//6) == 0:
                loss = loss.mean()
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                tmp = 0
                losses.append(loss.clone().cpu().detach().numpy())
        print("epoch{}: average loss {}".format(epoch, np.array(losses).mean()))
        vae.evaluate(random.choice(train_datas))
        torch.save(vae.cpu().state_dict(), './vae.pth')
        vae.to(dev)
        flag = False





def main():
    vae = VAE()
    pics = load_pictures()
    train_vae(vae, 50, pics)


if __name__ == "__main__":
    main()
