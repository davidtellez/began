import theano
import theano.tensor as T
import lasagne
from lasagne.layers import InputLayer, Conv2DLayer, MaxPool2DLayer, DenseLayer, get_output, ReshapeLayer, TransposedConv2DLayer, Upscale2DLayer, batch_norm
from lasagne.nonlinearities import rectify, softmax, elu, tanh, sigmoid
from lasagne.objectives import categorical_crossentropy, categorical_accuracy
from lasagne.init import GlorotUniform, Constant
import numpy as np
from collections import OrderedDict

def print_network_architecture(network, name):
    print('### Network: %s ###' % name)
    for l in lasagne.layers.get_all_layers(network):
        try:
            print('%s: %s: %s -> %s' % (l.name, l.__class__.__name__, str(l.get_W_shape()), str(lasagne.layers.get_output_shape(l))))
        except:
            print('%s: %s' % (l.name, l.__class__.__name__))

def get_W(network, layer_name):
    if (network is not None) and (layer_name in network):
        W = network[layer_name].W
    else:
        W = GlorotUniform()  # default value in Lasagne
    return W

def get_b(network, layer_name):
    if (network is not None) and (layer_name in network):
        b = network[layer_name].b
    else:
        b = Constant(0.)  # default value in Lasagne
    return b

def conv_layer(input, n_filters, stride, name, network_weights, nonlinearity=elu, bn=False):

    layer = Conv2DLayer(input, num_filters=n_filters, filter_size=3, stride=stride, pad='same',
                        nonlinearity=nonlinearity, name=name, W=get_W(network_weights, name), b=get_b(network_weights, name))
    if bn:
        layer = batch_norm(layer)
    return layer

def transposed_conv_layer(input, n_filters, stride, name, network_weights, output_size, nonlinearity=elu, bn=False):

    layer = TransposedConv2DLayer(input, n_filters, filter_size=3, stride=stride, W=get_W(network_weights, name),
                                  b=get_b(network_weights, name), nonlinearity=nonlinearity, name=name, crop='same',
                                  output_size=output_size)
    if bn:
        layer = batch_norm(layer)
    return layer

def dense_layer(input, n_units, name, network_weights, nonlinearity=None, bn=False):

    layer = DenseLayer(input, num_units=n_units, nonlinearity=nonlinearity, name=name,
                       W=get_W(network_weights, name), b=get_b(network_weights, name))
    if bn:
        layer = batch_norm(layer)
    return layer

class BEGANNet(object):

    def __init__(self, input_size, hidden_size, n_filters):

        # Save parameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_filters = n_filters

        # Define inputs
        input_z_var = T.fmatrix('input_z_var')
        input_x_var = T.ftensor4('input_x_var')
        learning_rate = T.fscalar('learning_rate')
        gamma = T.fscalar('gamma')  # trade-off between image diversity and visual quality (0.5)
        lambda_ = T.fscalar('lambda_')  # closed-loop gain (0.001)
        k_t = T.fscalar('k_t')  # control loss

        # Define networks
        network_discriminator_real = self.network_discriminator(input_x_var)
        network_generator = self.network_generator(input_z_var)
        network_discriminator_fake = self.network_discriminator(network_generator['generator/output'], network_weights=network_discriminator_real)

        # Print architectures
        print_network_architecture(network_discriminator_real['discriminator/decoder/output'], name='Discriminator Real')
        print_network_architecture(network_generator['generator/output'], name='Generator')
        print_network_architecture(network_discriminator_fake['discriminator/decoder/output'], name='Discriminator Fake')

        # Define outputs
        prediction_generator = get_output(network_generator['generator/output'])  # artificial/fake images
        prediction_discriminator_real = get_output(network_discriminator_real['discriminator/decoder/output'])  # reconstructed from real images
        prediction_discriminator_fake = get_output(network_discriminator_fake['discriminator/decoder/output'])  # reconstructed from artificial images

        # Define losses
        loss_reconstruction_real = T.mean(abs(input_x_var - prediction_discriminator_real))
        loss_reconstruction_fake = T.mean(abs(prediction_generator - prediction_discriminator_fake))
        k_t_1 = k_t + lambda_ * (gamma * loss_reconstruction_real - loss_reconstruction_fake)
        loss_generator = loss_reconstruction_fake
        loss_discriminator = loss_reconstruction_real - k_t * loss_reconstruction_fake

        # Define performance
        m_global = loss_reconstruction_real + abs(gamma * loss_reconstruction_real - loss_reconstruction_fake)

        # Define params
        params_generator = lasagne.layers.get_all_params(network_generator['generator/output'], trainable=True)
        params_generator = [param for param in params_generator if 'generator' in param.name]
        params_discriminator = lasagne.layers.get_all_params(network_discriminator_real['discriminator/decoder/output'], trainable=True)
        params_discriminator = [param for param in params_discriminator if 'discriminator' in param.name]

        # Define gradients
        grads_generator = T.grad(loss_generator, params_generator)
        grads_discriminator = T.grad(loss_discriminator, params_discriminator)

        # Define updates
        # updates_generator = lasagne.updates.adam(loss_generator, params_generator, learning_rate=learning_rate)
        # updates_discriminator = lasagne.updates.adam(loss_discriminator, params_discriminator, learning_rate=learning_rate)
        # updates_all = OrderedDict(list(updates_generator.items()) + list(updates_discriminator.items()))
        # updates_all = lasagne.updates.adam(grads_generator + grads_discriminator, params_generator + params_discriminator, learning_rate=learning_rate)
        updates_all = lasagne.updates.adam(grads_generator + grads_discriminator, params_generator + params_discriminator, learning_rate=learning_rate)

        # Define training functions
        self.train_fn = theano.function(
            [input_x_var, input_z_var, learning_rate, gamma, lambda_, k_t],
            [m_global, loss_generator, loss_discriminator, loss_reconstruction_real, loss_reconstruction_fake, k_t_1],
            updates=updates_all)

        # Define validation functions
        self.valid_fn = theano.function(
            [input_x_var, input_z_var],
            [prediction_generator, prediction_discriminator_real, prediction_discriminator_fake]
        )

    def network_discriminator(self, input, network_weights=None):

        layers = []

        if isinstance(input, lasagne.layers.Layer):
            layers.append(input)

            # First convolution
            layers.append(conv_layer(input, n_filters=self.n_filters, stride=1, name='discriminator/encoder/conv%d' % len(layers), network_weights=network_weights))

        else:
            # Input layer
            layers.append(InputLayer(shape=(None, 3, self.input_size, self.input_size), input_var=input, name='discriminator/encoder/input'))

            # First convolution
            layers.append(conv_layer(layers[-1], n_filters=self.n_filters, stride=1, name='discriminator/encoder/conv%d' % len(layers), network_weights=network_weights))

        # Convolutional blocks (encoder)self.n_filters*i_block
        n_blocks = int(np.log2(self.input_size/8)) + 1  # end up with 8x8 output
        for i_block in range(1, n_blocks+1):
            layers.append(conv_layer(layers[-1], n_filters=self.n_filters*i_block, stride=1, name='discriminator/encoder/conv%d' % len(layers), network_weights=network_weights))
            layers.append(conv_layer(layers[-1], n_filters=self.n_filters*i_block, stride=1, name='discriminator/encoder/conv%d' % len(layers), network_weights=network_weights))
            if i_block != n_blocks:
                # layers.append(conv_layer(layers[-1], n_filters=self.n_filters*(i_block+1), stride=2, name='discriminator/encoder/conv%d' % len(layers), network_weights=network_weights))
                layers.append(MaxPool2DLayer(layers[-1], pool_size=2, stride=2, name='discriminator/encoder/pooling%d' % len(layers)))
            # else:
            #     layers.append(conv_layer(layers[-1], n_filters=self.n_filters*(i_block), stride=1, name='discriminator/encoder/conv%d' % len(layers), network_weights=network_weights))

        # Dense layers (linear outputs)
        layers.append(dense_layer(layers[-1], n_units=self.hidden_size, name='discriminator/encoder/dense%d' % len(layers), network_weights=network_weights))

        # Dense layer up (from h to n*8*8)
        layers.append(dense_layer(layers[-1], n_units=(8 * 8 * self.n_filters), name='discriminator/decoder/dense%d' % len(layers), network_weights=network_weights))
        layers.append(ReshapeLayer(layers[-1], (-1, self.n_filters, 8, 8), name='discriminator/decoder/reshape%d' % len(layers)))

        # Convolutional blocks (decoder)
        for i_block in range(1, n_blocks+1):
            layers.append(conv_layer(layers[-1], n_filters=self.n_filters, stride=1, name='discriminator/decoder/conv%d' % len(layers), network_weights=network_weights))
            layers.append(conv_layer(layers[-1], n_filters=self.n_filters, stride=1, name='discriminator/decoder/conv%d' % len(layers), network_weights=network_weights))
            if i_block != n_blocks:
                layers.append(Upscale2DLayer(layers[-1], scale_factor=2, name='discriminator/decoder/upsample%d' % len(layers)))

        # Final layer (make sure input images are in the range [-1, 1]
        layers.append(conv_layer(layers[-1], n_filters=3, stride=1, name='discriminator/decoder/output', network_weights=network_weights, nonlinearity=sigmoid))

        # Network in dictionary form
        network = {layer.name: layer for layer in layers}

        return network

    def network_generator(self, input_var, network_weights=None):

        # Input layer
        layers = []
        n_blocks = int(np.log2(self.input_size / 8)) + 1  # end up with 8x8 output
        layers.append(InputLayer(shape=(None, self.hidden_size), input_var=input_var, name='generator/input'))

        # Dense layer up (from h to n*8*8)
        layers.append(dense_layer(layers[-1], n_units=(8 * 8 * self.n_filters), name='generator/dense%d' % len(layers), network_weights=network_weights))
        layers.append(ReshapeLayer(layers[-1], (-1, self.n_filters, 8, 8), name='generator/reshape%d' % len(layers)))

        # Convolutional blocks (decoder)
        for i_block in range(1, n_blocks+1):
            layers.append(conv_layer(layers[-1], n_filters=self.n_filters, stride=1, name='generator/conv%d' % len(layers), network_weights=network_weights))
            layers.append(conv_layer(layers[-1], n_filters=self.n_filters, stride=1, name='generator/conv%d' % len(layers), network_weights=network_weights))
            if i_block != n_blocks:
                layers.append(Upscale2DLayer(layers[-1], scale_factor=2, name='generator/upsample%d' % len(layers)))

        # Final layer (make sure input images are in the range [-1, 1] if tanh used)
        layers.append(conv_layer(layers[-1], n_filters=3, stride=1, name='generator/output', network_weights=network_weights, nonlinearity=sigmoid))

        # Network in dictionary form
        network = {layer.name: layer for layer in layers}

        return network

    # def network_generator_alt(self, input_var, network_weights=None):
    #
    #     # Input layer
    #     layers = []
    #     n_blocks = int(np.log2(self.input_size / 8)) + 1  # end up with 8x8 output
    #     layers.append(InputLayer(shape=(None, self.hidden_size), input_var=input_var, name='generator/input'))
    #
    #     # Dense layer up (from h to n*8*8)
    #     layers.append(dense_layer(layers[-1], n_units=(8 * 8 * self.n_filters*n_blocks), name='generator/dense%d' % len(layers), network_weights=network_weights, nonlinearity=elu, bn=True))
    #     layers.append(ReshapeLayer(layers[-1], (-1, self.n_filters*n_blocks, 8, 8), name='generator/reshape%d' % len(layers)))
    #
    #     # Convolutional blocks (decoder)
    #     for i_block in range(1, n_blocks+1)[::-1]:
    #         # layers.append(conv_layer(layers[-1], n_filters=self.n_filters*(i_block), stride=1, name='generator/conv%d' % len(layers), network_weights=network_weights, bn=True))
    #         # layers.append(conv_layer(layers[-1], n_filters=self.n_filters*(i_block), stride=1, name='generator/conv%d' % len(layers), network_weights=network_weights, bn=True))
    #         if i_block != 1:
    #             layers.append(transposed_conv_layer(layers[-1], n_filters=self.n_filters*(i_block-1), stride=2, name='generator/upsample%d' % len(layers),
    #                                                 output_size=8*2**(n_blocks-i_block+1), network_weights=network_weights, nonlinearity=elu, bn=True))
    #
    #     # Final layer (make sure input images are in the range [-1, 1]
    #     layers.append(conv_layer(layers[-1], n_filters=3, stride=1, name='generator/output', network_weights=network_weights, nonlinearity=tanh, bn=False))
    #
    #     # Network in dictionary form
    #     network = {layer.name: layer for layer in layers}
    #
    #     return network


    def forward_pass(self, input_x, input_z):

        prediction_generator, prediction_discriminator_real, prediction_discriminator_fake = \
            self.valid_fn(input_x.astype('float32'), input_z.astype('float32'))

        return prediction_generator, prediction_discriminator_real, prediction_discriminator_fake

    def backward_pass(self, input_x, input_z, learning_rate, gamma, lambda_, k_t):

        m_global, loss_generator, loss_discriminator, loss_reconstruction_real, loss_reconstruction_fake, k_t_1 = \
            self.train_fn(input_x.astype('float32'), input_z.astype('float32'),
                            np.array(learning_rate).astype('float32'), np.array(gamma).astype('float32'),
                            np.array(lambda_).astype('float32'), np.array(k_t).astype('float32'))

        return m_global, loss_generator, loss_discriminator, loss_reconstruction_real, loss_reconstruction_fake, k_t_1

if __name__ == "__main__":

    model = BEGANNet(input_size=64, hidden_size=128, n_filters=64)