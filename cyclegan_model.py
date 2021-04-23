import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
import tensorflow.data as tf_data

from tensorflow.keras import layers

from tensorflow.keras.models import Model

from tensorflow_addons.layers import InstanceNormalization

from tensorflow.keras.initializers import RandomNormal

from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras.losses import MeanAbsoluteError

from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.optimizers import Adam

# Weights Initialisers for the layers
kernel_init = RandomNormal(mean=0.0, stddev=0.02)

# Gamma Initialisers for instance normalisation
gamma_init = RandomNormal(mean=0.0, stddev=0.02)


class ReflectionPadding2D(layers.Layer):
    """
    Implements Reflection Padding as a layer.

    Args:
        padding(tuple): Amount of padding for the spatial dimensions.

    Returns:
        A padded tensor with the same type as the input tensor.

    """

    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = tuple(padding)
        super(ReflectionPadding2D, self).__init__(**kwargs)

    def call(self, input_tensor, mask=None):
        padding_width, padding_height = self.padding
        padding_tensor = [
            [0, 0],
            [padding_height, padding_height],
            [padding_width, padding_width],
            [0, 0],
        ]

        return tf.pad(input_tensor, padding_tensor, mode="REFLECT")


def residual_block(x, activation, kernel_initializer=kernel_init, kernel_size=(3, 3), strides=(1, 1), padding="valid", gamma_initializer=gamma_init, use_bias=False):
    dim = x.shape[-1]
    input_tensor = x

    x = ReflectionPadding2D()(input_tensor)
    x = layers.Conv2D(dim, kernel_size, strides=strides,
                      kernel_initializer=kernel_initializer, padding=padding, use_bias=use_bias)(x)
    x = InstanceNormalization(gamma_initializer=gamma_initializer)(x)

    x = activation(x)

    x = ReflectionPadding2D()(x)
    x = layers.Conv2D(dim, kernel_size, strides=strides,
                      kernel_initializer=kernel_initializer, padding=padding, use_bias=use_bias)(x)
    x = InstanceNormalization(gamma_initializer=gamma_initializer)(x)

    x = layers.add([input_tensor, x])

    return x


def downsample(x, filters, activation, kernel_initializer=kernel_init, kernel_size=(3, 3), strides=(2, 2), padding='same', gamma_initializer=gamma_init, use_bias=False):
    x = layers.Conv2D(filters, kernel_size, strides=strides,
                      kernel_initializer=kernel_initializer, padding=padding, use_bias=use_bias)(x)
    x = InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if(activation):
        x = activation(x)

    return x


def upsample(x, filters, activation, kernel_size=(3, 3), strides=(2, 2), paddding='same', kernel_initializer=kernel_init, gamma_initializer=gamma_init, use_bias=False):
    x = layers.Conv2DTranspose(filters, kernel_size, strides=strides, padding=paddding,
                               kernel_initializer=kernel_initializer, use_bias=use_bias)(x)
    x = InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    if(activation):
        x = activation(x)

    return x

# Generator Model


def get_resnet_generator(num_residual_blocks, model_image_size, filters=64, num_downsampling_blocks=2,  num_upsample_blocks=2, gamma_initializer=gamma_init, name=None):
    # Input
    img_input = layers.Input(shape=model_image_size, name=name + "_img_input")
    x = ReflectionPadding2D(padding=(3, 3))(img_input)
    x = layers.Conv2D(
        filters, (7, 7), kernel_initializer=kernel_init, use_bias=False)(x)
    x = InstanceNormalization(gamma_initializer=gamma_initializer)(x)
    x = layers.Activation('relu')(x)

    # Downsampling
    for _ in range(num_downsampling_blocks):
        filters *= 2
        x = downsample(x, filters=filters,
                       activation=layers.Activation('relu'))

    # Residual blocks
    for _ in range(num_residual_blocks):
        x = residual_block(x, activation=layers.Activation('relu'))

    # Upsampling
    for _ in range(num_upsample_blocks):
        filters //= 2
        x = upsample(x, filters, activation=layers.Activation('relu'))

    # Final block
    x = ReflectionPadding2D(padding=(3, 3))(x)
    x = layers.Conv2D(3, (7, 7), padding='valid')(x)
    x = layers.Activation('tanh')(x)

    model = Model(img_input, x, name=name)
    return model


# Discriminator Model
def get_discriminator(model_image_size, filters=64, kernel_initializer=kernel_init, num_downsampling=3, name=None):
    img_input = layers.Input(shape=model_image_size, name=name + "_img_input")
    x = layers.Conv2D(filters, (4, 4), strides=(
        2, 2), padding="same", kernel_initializer=kernel_initializer,)(img_input)
    x = layers.LeakyReLU(0.2)(x)

    num_filters = filters
    for num_downsample_block in range(3):
        num_filters *= 2
        if(num_downsample_block < 2):
            x = downsample(x, filters=num_filters, activation=layers.LeakyReLU(
                0.2), kernel_size=(4, 4), strides=(2, 2))
        else:
            x = downsample(x, filters=num_filters, activation=layers.LeakyReLU(
                0.2), kernel_size=(4, 4), strides=(1, 1))

    x = layers.Conv2D(1, (4, 4), strides=(1, 1), padding="same",
                      kernel_initializer=kernel_initializer)(x)

    model = Model(inputs=img_input, outputs=x, name=name)
    return model


# Loss function for evaluating adversarial loss
adv_loss_fn = MeanSquaredError()


# Define the loss function for the generators
def generator_loss_fn(fake):
    fake_loss = adv_loss_fn(tf.ones_like(fake), fake)
    return fake_loss


# Define the loss function for the discriminators
def discriminator_loss_fn(real, fake):
    real_loss = adv_loss_fn(tf.ones_like(real), real)
    fake_loss = adv_loss_fn(tf.zeros_like(fake), fake)
    return (real_loss + fake_loss) * 0.5


# CycleGAN Class
class CycleGan(Model):
    def __init__(self, generator_G, generator_F, discriminator_X, discriminator_Y, lambda_cycle=10.0, lambda_identity=0.5):
        super(CycleGan, self).__init__()
        self.gen_G = generator_G
        self.gen_F = generator_F
        self.disc_X = discriminator_X
        self.disc_Y = discriminator_Y
        self.lambda_cycle = lambda_cycle
        self.lambda_identity = lambda_identity

    def compile(self, gen_G_optimizer, gen_F_optimizer, disc_X_optimizer, disc_Y_optimizer, gen_loss_fn, disc_loss_fn):
        super(CycleGan, self).compile()
        self.gen_G_optimizer = gen_G_optimizer
        self.gen_F_optimizer = gen_F_optimizer
        self.disc_X_optimizer = disc_X_optimizer
        self.disc_Y_optimizer = disc_Y_optimizer
        self.generator_loss_fn = gen_loss_fn
        self.discriminator_loss_fn = disc_loss_fn
        self.cycle_loss_fn = MeanAbsoluteError()
        self.identity_loss_fn = MeanAbsoluteError()

    def train_step(self, batch_data):
        real_x, real_y = batch_data

        # For CycleGAN, we need to calculate different kinds of losses for the generators and discriminators. We will perform the following steps here:
        #
        # 1. Pass real images through the generators and get the generated images
        # 2. Pass the generated images back to the generators to check if we can predict the original image from the generated image.
        # 3. Do an identity mapping of the real images using the generators.
        # 4. Pass the generated images in 1) to the corresponding discriminators.
        # 5. Calculate the generators total loss (adverserial + cycle + identity)
        # 6. Calculate the discriminators loss
        # 7. Update the weights of the generators
        # 8. Update the weights of the discriminators
        # 9. Return the losses in a dictionary

        with tf.GradientTape(persistent=True) as tape:

            # Generate fake samples
            fake_y = self.gen_G(real_x, training=True)
            fake_x = self.gen_F(real_y, training=True)

            # Forward Cycle
            cycled_x = self.gen_F(fake_y, training=True)
            cycled_y = self.gen_G(fake_x, training=True)

            # Identity Mapping
            same_x = self.gen_F(real_x, training=True)
            same_y = self.gen_G(real_y, training=True)

            # Discriminator output
            disc_real_x = self.disc_X(real_x, training=True)
            disc_fake_x = self.disc_X(fake_x, training=True)

            disc_real_y = self.disc_Y(real_y, training=True)
            disc_fake_y = self.disc_Y(fake_y, training=True)

            # Generator adverserial loss
            gen_G_loss = self.generator_loss_fn(disc_fake_y)
            gen_F_loss = self.generator_loss_fn(disc_fake_x)

            # Generator cycle loss
            cycle_loss_G = self.cycle_loss_fn(
                real_y, cycled_y) * self.lambda_cycle
            cycle_loss_F = self.cycle_loss_fn(
                real_x, cycled_x) * self.lambda_cycle

            # Generator identity loss
            id_loss_G = (self.identity_loss_fn(real_x, same_x) *
                         self.lambda_cycle * self.lambda_identity)
            id_loss_F = (self.identity_loss_fn(real_y, same_y) *
                         self.lambda_cycle * self.lambda_identity)

            # Total Generator loss
            total_loss_G = gen_G_loss + cycle_loss_G + id_loss_G
            total_loss_F = gen_F_loss + cycle_loss_F + id_loss_F

            # Discriminator loss
            disc_X_loss = self.discriminator_loss_fn(disc_real_x, disc_fake_x)
            disc_Y_loss = self.discriminator_loss_fn(disc_real_y, disc_fake_y)

        # Get the gradients for the generators
        grads_G = tape.gradient(total_loss_G, self.gen_G.trainable_variables)
        grads_F = tape.gradient(total_loss_F, self.gen_F.trainable_variables)

        # Get the gradients for the discriminators
        disc_X_grads = tape.gradient(
            disc_X_loss, self.disc_X.trainable_variables)
        disc_Y_grads = tape.gradient(
            disc_Y_loss, self.disc_Y.trainable_variables)

        # Update the weights of the generators
        self.gen_G_optimizer.apply_gradients(
            zip(grads_G, self.gen_G.trainable_variables))
        self.gen_F_optimizer.apply_gradients(
            zip(grads_F, self.gen_F.trainable_variables))

        # Update the weights of the discriminators
        self.disc_X_optimizer.apply_gradients(
            zip(disc_X_grads, self.disc_X.trainable_variables))
        self.disc_Y_optimizer.apply_gradients(
            zip(disc_Y_grads, self.disc_Y.trainable_variables))

        return {
            "G_loss": total_loss_G,
            "F_loss": total_loss_F,
            "D_X_loss": disc_X_loss,
            "D_Y_loss": disc_Y_loss
        }


# Callback Monitor
class GANMonitor(Callback):
    """
        A callback to generate and save images after each epoch
    """

    def __init__(self, num_img, test_A):
        self.num_img = num_img
        self.test_A = test_A

    def on_epoch_end(self, epoch, logs=None):
        inputs = []
        predictions = []

        for i, img in enumerate(self.test_A.take(self.num_img)):
            prediction = self.model.gen_G(img)[0].numpy()
            prediction = (prediction * 127.5 + 127.5).astype(np.uint8)
            img = (img[0] * 127.5 + 127.5).numpy().astype(np.uint8)

            inputs.append(img)
            predictions.append(prediction)

        fig, ax = plt.subplots(
            2, self.num_img, figsize=((2 * self.num_img), 5))
        for i in range(self.num_img):
            ax[0, i].axis('off')
            ax[0, i].imshow(inputs[i])
        for i in range(self.num_img):
            ax[1, i].axis('off')
            ax[1, i].imshow(predictions[i])

        filename = 'generated_plot_{epoch}.png'.format(epoch=epoch + 1)
        plt.tight_layout()
        plt.savefig(filename)
        plt.show()
        plt.close()


def create_default_cyclegan_model(resnet_blocks, model_image_size):
    # Default Adam parameters (initial learning rate and momentum decay rate)
    learn_rate = 2e-4
    beta_1_value = 0.5

    # Get Generators & Discriminators
    gen_G = get_resnet_generator(
        name="generator_G", num_residual_blocks=resnet_blocks, model_image_size=model_image_size)
    gen_F = get_resnet_generator(
        name="generator_F", num_residual_blocks=resnet_blocks, model_image_size=model_image_size)

    disc_X = get_discriminator(
        name="discriminator_X", model_image_size=model_image_size)
    disc_Y = get_discriminator(
        name="discriminator_Y", model_image_size=model_image_size)

    # Create CycleGAN model
    cycle_gan_model = CycleGan(
        generator_G=gen_G, generator_F=gen_F, discriminator_X=disc_X, discriminator_Y=disc_Y)

    # Compile the model
    cycle_gan_model.compile(
        gen_G_optimizer=Adam(learning_rate=learn_rate, beta_1=beta_1_value),
        gen_F_optimizer=Adam(learning_rate=learn_rate, beta_1=beta_1_value),
        disc_X_optimizer=Adam(learning_rate=learn_rate, beta_1=beta_1_value),
        disc_Y_optimizer=Adam(learning_rate=learn_rate, beta_1=beta_1_value),
        gen_loss_fn=generator_loss_fn,
        disc_loss_fn=discriminator_loss_fn
    )

    return cycle_gan_model
