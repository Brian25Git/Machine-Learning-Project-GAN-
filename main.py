import os, time
import glob
import math
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.utils as vutils
import matplotlib.animation as animation
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.nn.utils import spectral_norm
from numpy.core.fromnumeric import size
from google.colab import drive
drive.mount('/content/drive')

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if torch.cuda.is_available():
    torch.cuda.empty_cache()

batchSize = 8
epochs = 150
noise = 100
save_interval = 2
saved_model_path = "/content/drive/My Drive/2022 AI/GAN Model/Saved Models/"
load_model = False

compose = transforms.Compose([
            transforms.Grayscale(1),
            transforms.ToTensor(),
        ])

data_set = torchvision.datasets.ImageFolder(root="/content/drive/My Drive/2022 AI/GAN Model/Images", transform=compose)
data_loader = DataLoader(dataset=data_set, batch_size=batchSize, shuffle=True)
enumeratePerBatch = int(len(data_set)/batchSize)

def main():
    t0 = time.time()
    loss_function = nn.BCELoss()
    discriminator = Discriminators(1).to(device)
    generator = Generators(1).to(device)
    discriminatorOptim = optim.Adam(discriminator.parameters(), lr=0.0004, betas=(0.5,0.999))
    generatorOptim = optim.Adam(generator.parameters(), lr=0.0001, betas=(0.5,0.999))

    if load_model:
        generator.load_state_dict(torch.load(saved_model_path + "generator.pt"))
        discriminator.load_state_dict(torch.load(saved_model_path + "discriminator.pt"))
        generatorOptim.load_state_dict(torch.load(saved_model_path + "generatorOptim.pt"))
        discriminatorOptim.load_state_dict(torch.load(saved_model_path + "discriminatorOptim.pt"))

    for epoch in range(epochs):
        for i, (images, _) in enumerate(data_loader):
            realImage = images.to(device)
            prediction = discriminator(realImage)
            label_for_realImages = torch.ones(images.size(0), device=device)
            loss_discriminator = loss_function(prediction.view(-1), label_for_realImages)

            discriminator.zero_grad()
            loss_discriminator.backward()

            random_noise = torch.randn(batchSize, noise, 1, 1, device=device)
            generatedImage = generator(random_noise)
            prediction = discriminator(generatedImage.detach())
            label_for_generatedImages = torch.zeros(np.prod(prediction.size()), device=device)
            loss_discriminator_fake = loss_function(prediction.view(-1), label_for_generatedImages)

            loss_discriminator_fake.backward()
            loss_discriminator = loss_discriminator + loss_discriminator_fake
            discriminatorOptim.step()

            prediction = discriminator(generatedImage).view(-1)
            label_for_realImages = torch.ones(np.prod(prediction.size()), device=device)
            loss_generator = loss_function(prediction, label_for_realImages)

            generator.zero_grad()
            loss_generator.backward()
            generatorOptim.step()

        if epoch % save_interval == 0:
          torch.save(generator.state_dict(), saved_model_path + "generatorCheckpoint-" + str(epoch) + ".pt")

        print(f"\n[{epoch+1:3d}/{epochs:3d}][{i:3d}/{enumeratePerBatch}] dLoss = {loss_discriminator:8.4f} | gLoss = {loss_generator:8.4f}", end='')

    tc = time.time()-t0
    print (f"\nTraining completed in {tc:.0f} seconds")

    torch.save(generator.state_dict(), saved_model_path + "generator.pt")
    torch.save(discriminator.state_dict(), saved_model_path + "discriminator.pt")
    torch.save(generatorOptim.state_dict(), saved_model_path + "generatorOptim.pt")
    torch.save(discriminatorOptim.state_dict(), saved_model_path + "discriminatorOptim.pt")
    print ('Model saved')

if __name__ == '__main__':
    for file in glob.glob(saved_model_path + '*Checkpoint*.pt'):
        os.remove(file)
    main()
