# %%
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dropout,
    Input,
    Conv2DTranspose,
)
import os


# %%

path = ""
image_path = os.path.join(path + "./data/CameraRGB/")
mask_path = os.path.join(path + "./data/CameraMask/")

image_list = os.listdir(image_path)
mask_list = os.listdir(mask_path)

image_path_list = [image_path + i for i in image_list]
mask_path_list = [mask_path + i for i in mask_list]
# %%
image_path_list = tf.constant(image_path_list)
mask_path_list = tf.constant(mask_path_list)
dataset = tf.data.Dataset.from_tensor_slices((image_path_list, mask_path_list))


# %%
def preprocess_path(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, dtype=tf.float32)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=3)
    mask = tf.math.reduce_max(mask, axis=-1, keepdims=True)
    return img, mask


def preprocess_image(img, mask):
    img = tf.image.resize(img, size=(96, 128), method="nearest")
    mask = tf.image.resize(mask, size=(96, 128), method="nearest")

    mask = tf.cast(mask, tf.uint8)
    return img, mask


# %%
imgread = dataset.map(preprocess_path)
preprocessed_dataset = imgread.map(preprocess_image)


# %%
def downsampling(input, n_filters, drop_prob, max_pooling):
    conv1 = Conv2D(
        filters=n_filters,
        kernel_size=3,
        padding="same",
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.HeNormal,
    )(input)

    conv2 = Conv2D(
        filters=n_filters,
        kernel_size=3,
        padding="same",
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.HeNormal,
    )(conv1)

    if drop_prob > 0:
        conv2 = Dropout(drop_prob)(conv2)

    skip_connections = conv2

    if max_pooling:
        conv2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    return skip_connections, conv2


# %%


def upsampling(input, skip_connections, n_filters):
    transpose_conv1 = Conv2DTranspose(
        filters=n_filters, kernel_size=3, strides=(2, 2), padding="same"
    )(input)

    concated_input = tf.keras.layers.concatenate(
        [transpose_conv1, skip_connections], axis=3
    )

    conv1 = Conv2D(
        filters=n_filters,
        kernel_size=3,
        padding="same",
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.HeNormal,
    )(concated_input)

    conv2 = Conv2D(
        filters=n_filters,
        kernel_size=3,
        padding="same",
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.HeNormal,
    )(conv1)

    return conv2


# %%
def unet_model(input_size=(96, 128, 3), n_filters=64, n_classes=23):
    input = Input(input_size)
    skip_block1, down_block1 = downsampling(input, n_filters, 0, True)
    skip_block2, down_block2 = downsampling(down_block1, n_filters * 2, 0, True)
    skip_block3, down_block3 = downsampling(down_block2, n_filters * 4, 0, True)
    skip_block4, down_block4 = downsampling(down_block3, n_filters * 8, 0.3, True)

    _, down_block5 = downsampling(down_block4, n_filters * 16, 0.3, False)

    up_block1 = upsampling(down_block5, skip_block4, n_filters * 8)
    up_block2 = upsampling(up_block1, skip_block3, n_filters * 4)
    up_block3 = upsampling(up_block2, skip_block2, n_filters * 2)
    up_block4 = upsampling(up_block3, skip_block1, n_filters)

    conv5 = Conv2D(
        n_filters,
        3,
        padding="same",
        activation=tf.keras.activations.relu,
        kernel_initializer=tf.keras.initializers.HeNormal,
    )(up_block4)

    output = Conv2D(n_classes, 1, padding="same")(conv5)

    model = tf.keras.Model(inputs=input, outputs=output)

    return model


# %%

unet = unet_model()
# %%
unet.compile(
    optimizer="adam",
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"],
)
# %%
EPOCHS = 5
BUFFER_SIZE = 500
BATCH_SIZE = 32

train_dataset = preprocessed_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
print(train_dataset.element_spec)
# %%
model_history = unet.fit(train_dataset, epochs=EPOCHS)


# %%
def show_prediction(num=3):
    for image, mask in preprocessed_dataset.take(num):
        input_image = tf.expand_dims(image, axis=0)

        pred_mask = unet.predict(input_image)

        pred_mask = tf.argmax(pred_mask, axis=-1)
        pred_mask = pred_mask[..., tf.newaxis]

        pred_mask = pred_mask[0]

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))

        axs[0].set_title("Input Image")
        axs[0].imshow(tf.keras.preprocessing.image.array_to_img(image))
        axs[0].axis("off")

        axs[1].set_title("True Mask")
        axs[1].imshow(tf.keras.preprocessing.image.array_to_img(mask))
        axs[1].axis("off")

        axs[2].set_title("Predicted Mask")
        axs[2].imshow(tf.keras.preprocessing.image.array_to_img(pred_mask))
        axs[2].axis("off")

        plt.show()


# %%
show_prediction(5)

# %%
