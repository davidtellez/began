from os.path import exists, join, dirname, basename
import os
import time
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import scipy.ndimage
from PIL import Image
import scipy
import sys

from began_network import BEGANNet
from data_handler import DataHandler


class Trainer():

    def __init__(self, input_size=64, hidden_size=64, n_filters=16):

        # Copy params
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_filters = n_filters

        # Initialize data loader
        self.data = DataHandler(image_size=input_size)

        # Initialize model
        self.began_network = BEGANNet(input_size=input_size, hidden_size=hidden_size, n_filters=n_filters)

    def train(self, n_iterations, mini_batch_size, learning_rate_0=1e-4, gamma=0.5, lambda_=0.001, k_t_0=0.0):

        # Initialize
        k_t = k_t_0
        learning_rate = learning_rate_0

        # Iterations
        for iteration_i in range(n_iterations):

            # Get data
            input_x, _ = self.data.get_batch('train', mini_batch_size/2, use_target_distribution=False)
            input_z = np.random.uniform(0, 1, (mini_batch_size/2, self.hidden_size))

            # Train
            m_global, loss_generator, loss_discriminator, loss_reconstruction_real, loss_reconstruction_fake, k_t_1 = \
                self.began_network.backward_pass(input_x, input_z, learning_rate, gamma, lambda_, k_t)

            # Validate
            if iteration_i % 500 == 0:
                input_x_val, _ = self.data.get_batch('train', mini_batch_size / 2, use_target_distribution=False)
                input_z_val = np.random.uniform(-1, 1, (mini_batch_size / 2, self.hidden_size))

                prediction_generator, prediction_discriminator_real, prediction_discriminator_fake = \
                    self.began_network.forward_pass(input_x_val, input_z_val)

                plot_grid(input_x_val, output_path='resources/plots/input_x_%05d.png' % iteration_i)
                plot_grid(prediction_generator, output_path='resources/plots/prediction_generator_%05d.png' % iteration_i)
                plot_grid(prediction_discriminator_real, output_path='resources/plots/prediction_discriminator_real_%05d.png' % iteration_i)
                plot_grid(prediction_discriminator_fake, output_path='resources/plots/prediction_discriminator_fake_%05d.png' % iteration_i)

            # Print info
            print('[It %04d, LR %0.6f] m %0.3f - lg %0.3f - ld %0.3f - lrr %0.3f - lrf %0.3f - kt1 %0.3f' %
                  (iteration_i, learning_rate, m_global, loss_generator, loss_discriminator,
                   loss_reconstruction_real, loss_reconstruction_fake, k_t_1))

            # Update params
            k_t = k_t_1
            learning_rate *= 0.9999


def plot_grid(batch, output_path):

    # batch = 1.0 * (batch + 1) / 2.0
    n = int(np.ceil(np.sqrt(batch.shape[0])))
    for i in range(batch.shape[0]):
        plt.subplot(n, n, i + 1)
        plt.imshow(batch[i, :, :, :].transpose((1, 2, 0)), vmin=0, vmax=1)
        plt.axis('off')
    plt.savefig(output_path)
    plt.close()

if __name__ == "__main__":

    # Train model
    trainer = Trainer(input_size=16, hidden_size=128, n_filters=32)
    results = trainer.train(n_iterations=50000, mini_batch_size=64, learning_rate_0=1e-4, gamma=0.5, lambda_=0.001, k_t_0=0)



