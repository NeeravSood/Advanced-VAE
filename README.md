# Variational Autoencoder (VAE) with PyTorch

## Overview

This project implements a Variational Autoencoder (VAE) using a Convolutional Neural Network (CNN) in PyTorch. A VAE is a generative model that learns to encode input data into a lower-dimensional latent space and then decode it back to the original data space. VAEs are widely used for tasks such as image generation, data compression, and anomaly detection.

## Table of Contents
- [Introduction](#introduction)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Understanding the Code](#understanding-the-code)
  - [Model Architecture](#model-architecture)
  - [Training Loop](#training-loop)
- [Potential Uses](#potential-uses)
- [References](#references)

## Introduction

### What is a VAE?

A Variational Autoencoder (VAE) is a type of generative model that uses neural networks to encode input data into a latent space and then decode it back into the data space. Unlike traditional autoencoders, VAEs impose a probabilistic structure on the latent space, allowing for more meaningful and interpretable representations.

### How does a VAE work?

1. **Encoder**: Maps the input data to a mean and variance that define a distribution in the latent space.
2. **Reparameterization Trick**: Samples a point from the latent space distribution.
3. **Decoder**: Maps the sampled point back to the data space, reconstructing the input data.

### Potential Uses of VAEs

- **Image Generation**: Generate new images similar to the training data.
- **Data Compression**: Compress data by encoding it into a smaller latent space.
- **Anomaly Detection**: Detect anomalies by comparing reconstruction errors.
- **Data Imputation**: Fill in missing data by sampling from the latent space.

## Requirements

- Python 3.x
- PyTorch
- torchvision
- numpy

## Installation

1. Clone the repository:
    ```bash
    git clone https://github.com/NeeravSood/Advanced-VAE.git
    cd vae-pytorch
    ```
2. Install the required packages:
    ```bash
    pip install torch torchvision numpy
    ```

## Usage

1. Run the training script:
    ```bash
    python train.py
    ```

## Understanding the Code

### Model Architecture

The VAE model is defined in the `ConvVAE` class, which includes:

1. **Encoder**: Two convolutional layers followed by fully connected layers to map the input to a latent space.
2. **Reparameterization**: The `reparameterize` method samples from the latent space using the mean and variance.
3. **Decoder**: Fully connected layers followed by transposed convolutional layers to reconstruct the input data from the latent space.

### Training Loop

The training loop:
1. Loads the MNIST dataset.
2. Defines the VAE model and optimizer.
3. Trains the model for a specified number of epochs.
4. Computes the loss, which is a combination of reconstruction loss (Binary Cross-Entropy) and KL divergence.

```python
# Training loop
epochs = 50
for epoch in range(epochs):
    model.train()
    train_loss = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.view(-1, 1, 28, 28)
        optimizer.zero_grad()
        recon_batch, mu, logvar = model(data)
        loss = model.loss_function(recon_batch, data, mu, logvar)
        loss.backward()
        train_loss += loss.item()
        optimizer.step()
    
    print(f'Epoch {epoch+1}/{epochs}, Loss: {train_loss/len(train_loader.dataset)}')

print("Training complete.")
```

## Potential Uses

- **Image Generation**: Generate new, unseen images similar to the ones in the training dataset.
- **Data Compression**: Encode images into a lower-dimensional space for compression.
- **Anomaly Detection**: Detect anomalies by observing reconstruction errors; anomalies typically have higher reconstruction errors.
- **Data Imputation**: Fill in missing parts of images by reconstructing the missing parts from the latent representation.

Feel free to contribute to this project or raise issues if you find any bugs. Happy coding!
