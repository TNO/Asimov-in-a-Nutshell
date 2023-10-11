import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


class NNUtilities:
    @staticmethod
    def generate_keras_ds(directory, class_names, image_size, colormode):
        train_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="int" if len(class_names) == 2 else "categorical",
            class_names=class_names,
            color_mode=colormode,
            batch_size=32,
            image_size=image_size,
            shuffle=True,
            seed=221,
            subset="training",
            validation_split=0.2,
            follow_links=False,
            crop_to_aspect_ratio=False
        )
        val_ds = tf.keras.preprocessing.image_dataset_from_directory(
            directory,
            labels="inferred",
            label_mode="int" if len(class_names) == 2 else "categorical",
            class_names=class_names,
            color_mode=colormode,
            batch_size=32,
            image_size=image_size,
            shuffle=True,
            seed=221,
            subset="validation",
            validation_split=0.2,
        )
        return train_ds, val_ds


class NNModel:
    @staticmethod
    def make_model_lenet5_original(image_size, colormode, num_classes, strides=1):
        # Note: original Lenet5 input_shape is 28x28 pixels, zero-padded to 32x32 pixels
        input_shape = (image_size[0], image_size[1], 1) if colormode == 'grayscale' else (image_size[0], image_size[1], 3)
        inputs = keras.Input(shape=input_shape)
        x = layers.Rescaling(scale=1.0/255)(inputs)
        x = layers.Conv2D(filters=6, kernel_size=5, strides=strides, activation='relu')(x)
        x = layers.MaxPool2D(pool_size=2, strides=2)(x)
        x = layers.Conv2D(filters=16, kernel_size=5, strides=strides, activation='relu')(x)
        x = layers.MaxPool2D(pool_size=2, strides=2)(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=120, activation='relu')(x)
        x = layers.Dense(units=84, activation='relu')(x)
        x = layers.Dense(units=num_classes, activation='softmax')(x)
        return keras.Model(inputs, x)

    @staticmethod
    def make_model_lenet5(image_size, colormode, num_classes, strides=1):
        input_shape = (image_size[0], image_size[1], 1) if colormode == 'grayscale' else (image_size[0], image_size[1], 3)
        inputs = keras.Input(shape=input_shape)
        x = layers.Rescaling(scale=1.0/255)(inputs)
        x = layers.Conv2D(filters=32, kernel_size=1, strides=2, padding="same")(x)
        x = layers.MaxPool2D(pool_size=2, strides=None, padding="same")(x)
        x = layers.Conv2D(filters=32, kernel_size=1, strides=2, padding="same")(x)
        x = layers.MaxPool2D(pool_size=2, strides=None, padding="same")(x)
        x = layers.Flatten()(x)
        x = layers.Dense(units=120, activation="tanh")(x)
        x = layers.Dense(units=84, activation="tanh")(x)

        if num_classes == 2:
            activation = "sigmoid"
            units = 1
        else:
            activation = "softmax"
            units = num_classes
        x = layers.Dense(units=units, activation=activation)(x)
        return keras.Model(inputs, x)

    @staticmethod
    def make_model_lenet5_reg(image_size, colormode, num_classes, strides=1):
        input_shape = (image_size[0], image_size[1], 1) if colormode == 'grayscale' else (image_size[0], image_size[1], 3)
        inputs = keras.Input(shape=input_shape)
        x = layers.Rescaling(1.0/ 255)(inputs)
        x = layers.Conv2D(32, 1, strides=2, padding="same")(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="same")(x)
        x = layers.Conv2D(32, 1, strides=2, padding="same")(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="same")(x)
        x = layers.Conv2D(32, 1, strides=2, padding="same")(x)                   # added
        x = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="same")(x)  # added
        x = layers.Flatten()(x)
        x = layers.Dense(120, activation="tanh")(x)
        x = layers.Dense(84, activation="tanh")(x)

        x = layers.Dropout(0.1)(x)
        # x = layers.Dense(units=1, activation='linear')(x)
        x = layers.Dense(units=num_classes, activation='softmax')(x)
        return keras.Model(inputs, x)

    # implementar VGG16

    @staticmethod
    def make_model_xception_reg(image_size, colormode, num_classes, strides=1):
        input_shape = (image_size[0], image_size[1], 1) if colormode == 'grayscale' else (image_size[0], image_size[1], 3)
        inputs = keras.Input(shape=input_shape)
        x = layers.Rescaling(1.0 / 255)(inputs)
        x = layers.Conv2D(32, 3, strides=2, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.Conv2D(64, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        previous_block_activation = x  # Set aside residual

        for size in [128, 256, 512, 728]:
            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.Activation("relu")(x)
            x = layers.SeparableConv2D(size, 3, padding="same")(x)
            x = layers.BatchNormalization()(x)

            x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

            # Project residual
            residual = layers.Conv2D(size, 1, strides=2, padding="same")(
                previous_block_activation
            )
            x = layers.add([x, residual])  # Add back residual
            previous_block_activation = x  # Set aside next residual

        x = layers.SeparableConv2D(1024, 3, padding="same")(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation("relu")(x)

        x = layers.GlobalAveragePooling2D()(x)

        x = layers.Dense(units=num_classes, activation='softmax')(x)
        return keras.Model(inputs, x)

    @staticmethod
    def make_model_vgg16_reg(image_size, colormode, num_classes, strides=1):
        input_shape = (image_size[0], image_size[1], 1) if colormode == 'grayscale' else (image_size[0], image_size[1], 3)
        inputs = keras.Input(shape=input_shape)
        x = layers.Rescaling(1.0 / 255)(inputs)
        x = layers.Conv2D(64, 1, strides=2, padding="same")(x)
        x = layers.Conv2D(64, 1, strides=2, padding="same")(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="same")(x)

        x = layers.Conv2D(128, 1, strides=2, padding="same")(x)
        x = layers.Conv2D(128, 1, strides=2, padding="same")(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="same")(x)

        x = layers.Conv2D(256, 1, strides=2, padding="same")(x)
        x = layers.Conv2D(256, 1, strides=2, padding="same")(x)
        x = layers.Conv2D(256, 1, strides=2, padding="same")(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="same")(x)

        x = layers.Conv2D(512, 1, strides=2, padding="same")(x)
        x = layers.Conv2D(512, 1, strides=2, padding="same")(x)
        x = layers.Conv2D(512, 1, strides=2, padding="same")(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="same")(x)

        x = layers.Conv2D(512, 1, strides=2, padding="same")(x)
        x = layers.Conv2D(512, 1, strides=2, padding="same")(x)
        x = layers.Conv2D(512, 1, strides=2, padding="same")(x)
        x = layers.MaxPool2D(pool_size=(2, 2), strides=None, padding="same")(x)

        x = layers.Flatten()(x)
        x = layers.Dense(4096, activation="tanh")(x)
        x = layers.Dense(1000, activation="tanh")(x)

        x = layers.Dense(units=num_classes, activation='softmax')(x)
        return keras.Model(inputs, x)
