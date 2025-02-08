import os
import random
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import VGG19
from tensorflow.keras.applications.vgg19 import preprocess_input
from skimage.metrics import structural_similarity as compare_ssim
from skimage.metrics import peak_signal_noise_ratio as compare_psnr
from PIL import Image

# Set GPU 3 as the visible device
os.environ["CUDA_VISIBLE_DEVICES"] = "3"
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)

# Directories
source_folder = "data/data"
train_folder = "trialv2_train"
val_folder = "trialv2_val"
output_folder = "trialv2_results"
log_file = os.path.join(output_folder, "log_metrics.txt")

os.makedirs(train_folder, exist_ok=True)
os.makedirs(val_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# Split Dataset
all_files = [f for f in os.listdir(source_folder) if f.endswith(('.jpg', '.png', '.jpeg'))]
random.shuffle(all_files)
split_index = int(0.9 * len(all_files))
train_files, val_files = all_files[:split_index], all_files[split_index:]
for file_list, destination in [(train_files, train_folder), (val_files, val_folder)]:
    for file_name in file_list:
        src_path = os.path.join(source_folder, file_name)
        dst_path = os.path.join(destination, file_name)
        tf.io.gfile.copy(src_path, dst_path, overwrite=True)

def calculate_metrics(hr_img, sr_img):
    """
    Calculate Structural Similarity Index (SSIM) and Peak Signal-to-Noise Ratio (PSNR)
    
    Args:
        hr_img (numpy.ndarray): High-resolution ground truth image
        sr_img (numpy.ndarray): Super-resolved image
    
    Returns:
        tuple: (SSIM, PSNR) metrics
    """
    # Squeeze the single-channel dimension for comparison
    hr_img = hr_img.squeeze()
    sr_img = sr_img.squeeze()
    
    # Ensure images are in the range [0, 255]
    hr_img = hr_img * 255
    sr_img = sr_img * 255
    
    # Calculate SSIM and PSNR
    ssim = compare_ssim(hr_img, sr_img, data_range=255)
    psnr = compare_psnr(hr_img, sr_img, data_range=255)
    
    return ssim, psnr

def save_comparison_images(lr_imgs, hr_imgs, sr_imgs, epoch, output_folder):
    """
    Save comparison images showing LR, HR, and SR images side by side.
    
    Args:
        lr_imgs (tf.Tensor): Low-resolution input images
        hr_imgs (tf.Tensor): High-resolution ground truth images
        sr_imgs (tf.Tensor): Super-resolved generated images
        epoch (int): Current training epoch
        output_folder (str): Folder to save comparison images
    """
    # Select first few images from the batch
    num_images = min(4, lr_imgs.shape[0])
    
    for i in range(num_images):
        # Convert images to numpy and rescale to 0-255
        lr_img = lr_imgs[i].numpy().squeeze() * 255
        hr_img = hr_imgs[i].numpy().squeeze() * 255
        sr_img = sr_imgs[i].numpy().squeeze() * 255
        
        # Create a side-by-side comparison
        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle(f'Epoch {epoch} - Image {i+1}')
        
        axs[0].imshow(lr_img, cmap='gray')
        axs[0].set_title('Low Resolution')
        axs[0].axis('off')
        
        axs[1].imshow(hr_img, cmap='gray')
        axs[1].set_title('High Resolution')
        axs[1].axis('off')
        
        axs[2].imshow(sr_img, cmap='gray')
        axs[2].set_title('Super-Resolved')
        axs[2].axis('off')
        
        # Save the figure
        plt.tight_layout()
        save_path = os.path.join(output_folder, f'comparison_epoch_{epoch}_img_{i}.png')
        plt.savefig(save_path)
        plt.close()

# Preprocess Images
def preprocess_image(image_path, crop_size, upscale_factor):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=1)  # Grayscale
    img = tf.image.resize_with_crop_or_pad(img, crop_size, crop_size)
    hr_img = tf.image.random_crop(img, size=[crop_size, crop_size, 1])
    lr_img = tf.image.resize(hr_img, [crop_size // upscale_factor, crop_size // upscale_factor],
                             method=tf.image.ResizeMethod.BICUBIC)
    return tf.cast(lr_img, tf.float32) / 255.0, tf.cast(hr_img, tf.float32) / 255.0

def prepare_dataset(folder, crop_size, upscale_factor, batch_size):
    img_paths = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(('.jpg', '.png'))]
    dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    return dataset.map(lambda x: preprocess_image(x, crop_size, upscale_factor), num_parallel_calls=tf.data.AUTOTUNE)\
                  .shuffle(1000).batch(batch_size).prefetch(tf.data.AUTOTUNE)

CROP_SIZE, UPSCALE_FACTOR, BATCH_SIZE = 256, 4, 16
train_dataset = prepare_dataset(train_folder, CROP_SIZE, UPSCALE_FACTOR, BATCH_SIZE)
val_dataset = prepare_dataset(val_folder, CROP_SIZE, UPSCALE_FACTOR, BATCH_SIZE)

# VGG Perceptual Loss
vgg_model = VGG19(include_top=False, input_shape=(256, 256, 3), weights='imagenet')
vgg_model = Model(inputs=vgg_model.input, outputs=vgg_model.get_layer('block2_conv2').output)

def vgg_perceptual_loss(hr_images, sr_images):
    hr_images_rgb = tf.image.grayscale_to_rgb(hr_images)
    sr_images_rgb = tf.image.grayscale_to_rgb(sr_images)
    hr_features = vgg_model(preprocess_input(hr_images_rgb * 255.0))
    sr_features = vgg_model(preprocess_input(sr_images_rgb * 255.0))
    return tf.reduce_mean(tf.square(hr_features - sr_features))

# Attention Mechanism
class CBAMLayer(layers.Layer):
    def __init__(self, ratio=8):
        super(CBAMLayer, self).__init__()
        self.ratio = ratio

    def call(self, x):
        channels = x.shape[-1]
        avg_pool = tf.reduce_mean(x, axis=[1, 2], keepdims=True)
        max_pool = tf.reduce_max(x, axis=[1, 2], keepdims=True)
        scale = layers.Dense(channels, activation="sigmoid")(
            layers.Dense(channels // self.ratio, activation="relu")(avg_pool + max_pool)
        )
        x = x * scale
        spatial_attn = layers.Conv2D(1, 7, padding="same", activation="sigmoid")(x)
        return x * spatial_attn

# Generator
def build_generator(input_shape=(64, 64, 1)):
    def residual_block(x):
        res = layers.SeparableConv2D(64, 3, padding="same")(x)
        res = layers.BatchNormalization()(res)
        res = CBAMLayer()(res)
        return layers.Add()([x, res])

    def subpixel_conv(x, scale=2):
        """Custom Sub-pixel Convolution implemented with Keras Lambda layer."""
        return tf.keras.layers.Lambda(lambda t: tf.nn.depth_to_space(t, scale))(x)

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 9, padding="same", activation="relu")(inputs)
    res = x
    for _ in range(8):
        x = residual_block(x)
    x = layers.Conv2D(64, 3, padding="same")(x)
    x = layers.Add()([x, res])

    for _ in range(2):
        x = layers.Conv2D(256, 3, padding="same")(x)
        x = subpixel_conv(x, scale=2)  # Replace tf.nn.depth_to_space with Lambda layer
        x = layers.PReLU()(x)

    outputs = layers.Conv2D(1, 9, activation="tanh", padding="same")(x)
    return Model(inputs, outputs, name="Generator")

# Discriminator
def build_discriminator():
    inputs = layers.Input(shape=(256, 256, 1))
    x = inputs
    for filters in [64, 128, 256, 512]:
        x = layers.SeparableConv2D(filters, 3, strides=2, padding="same")(x)
        x = layers.LeakyReLU(0.2)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation="relu")(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    return Model(inputs, outputs, name="Discriminator")

# Models
generator = build_generator()
discriminator = build_discriminator()
gen_optimizer = tf.keras.optimizers.Adam(1e-4)
disc_optimizer = tf.keras.optimizers.Adam(1e-4)

# Checkpoint object
checkpoint = tf.train.Checkpoint(generator=generator,
                                 discriminator=discriminator,
                                 generator_optimizer=gen_optimizer,
                                 discriminator_optimizer=disc_optimizer)

# Loss Functions
def generator_loss(fake_output, hr_images, generated_images):
    adv_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(fake_output), fake_output)
    perceptual_loss = vgg_perceptual_loss(hr_images, generated_images)
    content_loss = tf.reduce_mean(tf.square(hr_images - generated_images))
    return adv_loss + 0.1 * perceptual_loss + 0.01 * content_loss

def discriminator_loss(real_output, fake_output):
    real_loss = tf.keras.losses.BinaryCrossentropy()(tf.ones_like(real_output), real_output)
    fake_loss = tf.keras.losses.BinaryCrossentropy()(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

# Training Loop
with open(log_file, "w") as log:
    for epoch in range(1, 101):
        print(f"Epoch {epoch}")
        g_losses, d_losses, ssim_vals, psnr_vals = [], [], [], []
        for lr_imgs, hr_imgs in train_dataset:
            with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
                sr_imgs = generator(lr_imgs, training=True)
                real_output = discriminator(hr_imgs, training=True)
                fake_output = discriminator(sr_imgs, training=True)

                g_loss = generator_loss(fake_output, hr_imgs, sr_imgs)
                d_loss = discriminator_loss(real_output, fake_output)

            gen_optimizer.apply_gradients(zip(gen_tape.gradient(g_loss, generator.trainable_variables),
                                              generator.trainable_variables))
            disc_optimizer.apply_gradients(zip(disc_tape.gradient(d_loss, discriminator.trainable_variables),
                                               discriminator.trainable_variables))
            for i in range(hr_imgs.shape[0]):
                ssim, psnr = calculate_metrics(hr_imgs[i].numpy(), sr_imgs[i].numpy())
                ssim_vals.append(ssim)
                psnr_vals.append(psnr)
            g_losses.append(g_loss.numpy())
            d_losses.append(d_loss.numpy())

        save_comparison_images(lr_imgs, hr_imgs, sr_imgs, epoch, output_folder)
        
        if epoch % 10 == 0:
            generator.save(os.path.join(output_folder, f'generator_epoch_{epoch}.h5'))
            discriminator.save(os.path.join(output_folder, f'discriminator_epoch_{epoch}.h5'))

            checkpoint.save(os.path.join(output_folder, f'ckpt_epoch_{epoch}'))
