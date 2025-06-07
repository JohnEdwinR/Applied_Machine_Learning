import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import layers
import time

# --- Load CIFAR-10 dataset ---
(x_train, y_train), (_, _) = tf.keras.datasets.cifar10.load_data()
print(f"Training images shape: {x_train.shape}")
print(f"Training labels shape: {y_train.shape}")

# --- Simple EDA: Display first 10 images ---
class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

plt.figure(figsize=(12,4))
for i in range(10):
    plt.subplot(2, 5, i+1)
    plt.imshow(x_train[i])
    plt.title(class_names[y_train[i][0]])
    plt.axis('off')
plt.show()

# --- Preprocess images: Normalize to [-1, 1] ---
x_train = x_train.astype('float32')
x_train = (x_train - 127.5) / 127.5

# --- Define Generator ---
def make_generator_model():
    model = tf.keras.Sequential([
        layers.Dense(8*8*256, use_bias=False, input_shape=(100,)),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Reshape((8, 8, 256)),
        layers.Conv2DTranspose(128, (5,5), strides=(1,1), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(64, (5,5), strides=(2,2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),
        layers.Conv2DTranspose(3, (5,5), strides=(2,2), padding='same', use_bias=False, activation='tanh')
    ])
    return model

# --- Define Discriminator ---
def make_discriminator_model():
    model = tf.keras.Sequential([
        layers.Conv2D(64, (5,5), strides=(2,2), padding='same', input_shape=[32,32,3]),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Conv2D(128, (5,5), strides=(2,2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),
        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

# --- Loss and optimizers ---
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator = make_generator_model()
discriminator = make_discriminator_model()

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)

# --- Prepare dataset ---
BUFFER_SIZE = 50000
BATCH_SIZE = 256
EPOCHS = 1000
NOISE_DIM = 100
NUM_EXAMPLES_TO_GENERATE = 16

train_dataset = tf.data.Dataset.from_tensor_slices(x_train).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
seed = tf.random.normal([NUM_EXAMPLES_TO_GENERATE, NOISE_DIM])

# --- Training step ---
@tf.function
def train_step(images):
    noise = tf.random.normal([BATCH_SIZE, NOISE_DIM])
    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)
        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)
        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)
    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)
    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))
    return gen_loss, disc_loss

# --- Generate images ---
def generate_and_show_image(model, noise_vector):
    generated_image = model(noise_vector, training=False)
    generated_image = (generated_image + 1) / 2.0  # scale to [0,1]
    img = generated_image[0].numpy()
    plt.imshow(img)
    plt.axis('off')
    plt.show()

# --- Training loop ---
def train(dataset, epochs):
    for epoch in range(epochs):
        start = time.time()
        for image_batch in dataset:
            gen_loss, disc_loss = train_step(image_batch)
        print(f'Epoch {epoch+1}, Gen Loss: {gen_loss:.4f}, Disc Loss: {disc_loss:.4f}, Time: {time.time()-start:.2f}s')
        if (epoch + 1) % 10 == 0:
            print("Generated images at epoch", epoch+1)
            generate_and_show_image(generator, seed[:1])

train(train_dataset, EPOCHS)

# --- Input part: generate image from custom noise vector ---
print("\nNow you can input 100 values separated by spaces as noise vector to generate an image.")
print("If you want to use random noise, just press Enter.")

user_input = input("Enter 100 float values (or press Enter for random noise): ").strip()

if user_input == '':
    noise_vector = tf.random.normal([1, NOISE_DIM])
else:
    try:
        vals = list(map(float, user_input.split()))
        if len(vals) != NOISE_DIM:
            raise ValueError("Incorrect number of values.")
        noise_vector = tf.convert_to_tensor([vals], dtype=tf.float32)
    except Exception as e:
        print("Invalid input, using random noise instead.")
        noise_vector = tf.random.normal([1, NOISE_DIM])

print("Generating image from your input noise vector...")
generate_and_show_image(generator, noise_vector)