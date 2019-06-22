from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.optim import Optimizer
from torch.utils.data import DataLoader
from .autoencoder import EncoderCNN, DecoderCNN


class Discriminator(nn.Module):
    def __init__(self, in_size):
        """
        :param in_size: The size of on input image (without batch dimension).
        """
        super().__init__()
        self.in_size = in_size
        # TODO: Create the discriminator model layers.
        # To extract image features you can use the EncoderCNN from the VAE
        # section or implement something new.
        # You can then use either an affine layer or another conv layer to
        # flatten the features.
        # ====== YOUR CODE: ======
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1, stride=2)
        self.conv1_bn = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1, stride=2)
        self.conv2_bn = nn.BatchNorm2d(128)
        self.conv3 = nn.Conv2d(128, 256, 3, padding=1, stride=2)
        self.conv3_bn = nn.BatchNorm2d(256)
        self.fc1 = nn.Linear(4096*4, 64)
        self.fc_bn = nn.BatchNorm1d(64)
        self.fc2 = nn.Linear(64, 1)

        # ========================

    def forward(self, x):
        """
        :param x: Input of shape (N,C,H,W) matching the given in_size.
        :return: Discriminator class score (aka logits, not probability) of
        shape (N,).
        """
        # TODO: Implement discriminator forward pass.
        # No need to apply sigmoid to obtain probability - we'll combine it
        # with the loss due to improved numerical stability.
        # ====== YOUR CODE: ======
        x = F.leaky_relu(self.conv1_bn(self.conv1(x)), 0.2)
        x = F.leaky_relu(self.conv2_bn(self.conv2(x)), 0.2)
        x = F.leaky_relu(self.conv3_bn(self.conv3(x)), 0.2)
        x = x.view(-1, 4096*4)
        x = F.leaky_relu(self.fc1(x), 0.2)
        x = F.dropout(x, 0.2)
        y = self.fc2(x)
        # ========================
        return y

class Generator(nn.Module):
    def __init__(self, z_dim, featuremap_size=4, out_channels=3):
        """
        :param z_dim: Dimension of latent space.
        :featuremap_size: Spatial size of first feature map to create
        (determines output size). For example set to 4 for a 4x4 feature map.
        :out_channels: Number of channels in the generated image.
        """
        super().__init__()
        self.z_dim = z_dim
        # TODO: Create the generator model layers.
        # To combine image features you can use the DecoderCNN from the VAE
        # section or implement something new.
        # You can assume a fixed image size.
        # ====== YOUR CODE: ======
        self.deconv1 = nn.ConvTranspose2d(z_dim, 1024, featuremap_size, stride=2)
        self.deconv1_bn = nn.BatchNorm2d(1024)
        self.deconv2 = nn.ConvTranspose2d(1024, 512, featuremap_size, padding=1, stride=2)
        self.deconv2_bn = nn.BatchNorm2d(512)
        self.deconv3 = nn.ConvTranspose2d(512, 256, featuremap_size, padding=1, stride=2)
        self.deconv3_bn = nn.BatchNorm2d(256)
        self.deconv4 = nn.ConvTranspose2d(256, 128, featuremap_size, padding=1, stride=2)
        self.deconv4_bn = nn.BatchNorm2d(128)
        self.deconv5 = nn.ConvTranspose2d(128, out_channels, featuremap_size, padding=1, stride=2)

        # ========================

    def sample(self, n, with_grad=False):
        """
        Samples from the Generator.
        :param n: Number of instance-space samples to generate.
        :param with_grad: Whether the returned samples should track
        gradients or not. I.e., whether they should be part of the generator's
        computation graph or standalone tensors.
        :return: A batch of samples, shape (N,C,H,W).
        """
        device = next(self.parameters()).device
        # TODO: Sample from the model.
        # Generate n latent space samples and return their reconstructions.
        # Don't use a loop.
        # ====== YOUR CODE: ======
        z = torch.randn(n, self.z_dim, device=device)
        if not with_grad:
            with torch.no_grad():
                samples = self.forward(z)
        else:
            samples = self.forward(z)

        # ========================
        return samples

    def forward(self, z):
        """
        :param z: A batch of latent space samples of shape (N, latent_dim).
        :return: A batch of generated images of shape (N,C,H,W) which should be
        the shape which the Discriminator accepts.
        """
        # TODO: Implement the Generator forward pass.
        # Don't forget to make sure the output instances have the same scale
        # as the original (real) images.
        # ====== YOUR CODE: ======
        z = z.view(-1, self.z_dim, 1, 1)
        z = F.relu(
            self.deconv1_bn(
                self.deconv1(z)))
        z = F.relu(
            self.deconv2_bn(
                self.deconv2(z)))
        z = F.relu(
            self.deconv3_bn(
                self.deconv3(z)))
        z = F.relu(
            self.deconv4_bn(
                self.deconv4(z)))
        z = F.tanh(
            self.deconv5(z))
        # ========================
        return z


def discriminator_loss_fn(y_data, y_generated, data_label=0, label_noise=0.0):
    """
    Computes the combined loss of the discriminator given real and generated
    data using a binary cross-entropy metric.
    This is the loss used to update the Discriminator parameters.
    :param y_data: Discriminator class-scores of instances of data sampled
    from the dataset, shape (N,).
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :param label_noise: The range of the noise to add. For example, if
    data_label=0 and label_noise=0.2 then the labels of the real data will be
    uniformly sampled from the range [-0.1,+0.1].
    :return: The combined loss of both.
    """
    assert data_label == 1 or data_label == 0
    # TODO: Implement the discriminator loss.
    # See torch's BCEWithLogitsLoss for a numerically stable implementation.
    # ====== YOUR CODE: ======
    N = y_data.shape
    device = y_data.device
    e0 = -label_noise / 2
    e1 = -e0

    # make label tensor
    data_label_tensor = torch.full(N, data_label, device=device)
    generated_label_tensor = 1 - data_label_tensor

    # add noise
    data_label_tensor = data_label_tensor + torch.FloatTensor(N).uniform_(e0, e1).to(device)
    generated_label_tensor = generated_label_tensor + torch.FloatTensor(N).uniform_(e0, e1).to(device)

    loss_data = F.binary_cross_entropy_with_logits(y_data, data_label_tensor)
    loss_generated = F.binary_cross_entropy_with_logits(y_generated, generated_label_tensor)

    # ========================
    return loss_data + loss_generated


def generator_loss_fn(y_generated, data_label=0):
    """
    Computes the loss of the generator given generated data using a
    binary cross-entropy metric.
    This is the loss used to update the Generator parameters.
    :param y_generated: Discriminator class-scores of instances of data
    generated by the generator, shape (N,).
    :param data_label: 0 or 1, label of instances coming from the real dataset.
    :return: The generator loss.
    """
    # TODO: Implement the Generator loss.
    # Think about what you need to compare the input to, in order to
    # formulate the loss in terms of Binary Cross Entropy.
    # ====== YOUR CODE: ======
    N = y_generated.shape
    device = y_generated.device

    # make label tensor
    generated_label_tensor = torch.full(N, data_label, device=device)

    loss = F.binary_cross_entropy_with_logits(y_generated, generated_label_tensor)
    # ========================
    return loss

def train_batch(dsc_model: Discriminator, gen_model: Generator,
                dsc_loss_fn: Callable, gen_loss_fn: Callable,
                dsc_optimizer: Optimizer, gen_optimizer: Optimizer,
                x_data: DataLoader):
    """
    Trains a GAN for over one batch, updating both the discriminator and
    generator.
    :return: The discriminator and generator losses.
    """

    # TODO: Discriminator update
    # 1. Show the discriminator real and generated data
    # 2. Calculate discriminator loss
    # 3. Update discriminator parameters
    # ====== YOUR CODE: ======
    dsc_optimizer.zero_grad()

    # forward both batches
    real_data = x_data
    fake_data = gen_model.sample(x_data.shape[0], with_grad=True)

    # get the result for the real and fake images
    real_class_scores = dsc_model(real_data)
    fake_class_scores = dsc_model(fake_data.detach())

    # calculate d loss and make a backward calculation to calculate the gradients
    dsc_loss = dsc_loss_fn(real_class_scores, fake_class_scores)

    # train the weights using the optimizer
    dsc_loss.backward()
    dsc_optimizer.step()

    # ========================

    # TODO: Generator update
    # 1. Show the discriminator generated data
    # 2. Calculate generator loss
    # 3. Update generator parameters
    # ====== YOUR CODE: ======
    gen_optimizer.zero_grad()

    # forward
    fake_class_scores = dsc_model(fake_data)

    # calculate g and make a backward calculation to calculate the gradients
    gen_loss = gen_loss_fn(fake_class_scores)

    # train the weights using the optimizer
    gen_loss.backward()
    gen_optimizer.step()
    # ========================

    return dsc_loss.item(), gen_loss.item()
