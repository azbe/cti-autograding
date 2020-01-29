import argparse
import copy

import keras
import numpy as np


def build_model():
    inp = keras.layers.Input((28, 28, 1))
    net = inp
    net = keras.layers.Conv2D(32, (3, 3), activation="relu")(net)
    net = keras.layers.Conv2D(64, (3, 3), activation="relu")(net)
    net = keras.layers.MaxPooling2D(pool_size=(2, 2))(net)
    net = keras.layers.Dropout(0.25)(net)
    net = keras.layers.Flatten()(net)
    net = keras.layers.Dense(128, activation="relu")(net)
    net = keras.layers.Dropout(0.5)(net)
    net = keras.layers.Dense(4, activation="softmax")(net)

    model = keras.models.Model(inputs=inp, outputs=net)
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=["accuracy"])

    return model


def build_discriminator():
    inp = keras.layers.Input((28, 28, 1))
    net = inp
    net = keras.layers.Conv2D(32, (3, 3), activation="relu")(net)
    net = keras.layers.Conv2D(64, (3, 3), activation="relu")(net)
    net = keras.layers.MaxPooling2D(pool_size=(2, 2))(net)
    net = keras.layers.Dropout(0.25)(net)
    net = keras.layers.Flatten()(net)
    net = keras.layers.Dense(128, activation="relu")(net)
    net = keras.layers.Dropout(0.5)(net)
    net = keras.layers.Dense(1, activation="sigmoid")(net)

    model = keras.models.Model(inputs=inp, outputs=net)
    model.compile(
        loss=keras.losses.binary_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=["binary_accuracy"])

    return model


def load_model(path):
    return keras.models.load_model(path)


def train_model(dataset_path, epochs, model_path):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = X_train[(y_train >= 1) & (y_train <= 4)]
    X_test = X_test[(y_test >= 1) & (y_test <= 4)]
    y_train = y_train[(y_train >= 1) & (y_train <= 4)] - 1
    y_test = y_test[(y_test >= 1) & (y_test <= 4)] - 1

    X_train = 255 - X_train
    X_test = 255 - X_test
    for idx in range(1, 3):
        x = copy.deepcopy(X_train)
        x[:, :idx, :] = 0
        x[:, :, :idx] = 0
        x[:, -idx:, :] = 0
        x[:, :, -idx:] = 0
        X_train = np.concatenate([X_train, x], axis=0)
        y_train = np.concatenate([y_train, y_train], axis=0)
    X_train = np.expand_dims(X_train / 255.0, axis=3)
    X_test = np.expand_dims(X_test / 255.0, axis=3)

    y_train_oh = keras.utils.to_categorical(y_train, 4)
    y_test_oh = keras.utils.to_categorical(y_test, 4)

    model = build_model()
    model.fit(
        X_train, y_train_oh, batch_size=32, epochs=10, verbose=1, validation_data=(X_test, y_test_oh), shuffle=True,
        callbacks=[keras.callbacks.ModelCheckpoint(model_path, save_best_only=True)])

    data_preproc = keras.preprocessing.image.ImageDataGenerator(
        rotation_range=30,
        width_shift_range=0.2, 
        height_shift_range=0.2,
        zoom_range=0.2,
        brightness_range=(0.75, 1.25),
        rescale=1/255.0,
        validation_split=0.2)
    train_gen = data_preproc.flow_from_directory(dataset_path, shuffle=True, batch_size=32, target_size=(28, 28), color_mode="grayscale", subset="training")
    val_gen = data_preproc.flow_from_directory(dataset_path, shuffle=True, batch_size=32, target_size=(28, 28), color_mode="grayscale", subset="validation")
    model.fit_generator(train_gen, epochs=epochs, callbacks=[keras.callbacks.ModelCheckpoint(model_path, save_best_only=True)],
                        validation_data=val_gen, validation_steps=10)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_path", type=str, default="./digits", help="Path to digits dataset")
    parser.add_argument("--model_path", type=str, default="./digit_recognizer.h5", help="Path to save model to.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    args, _ = parser.parse_known_args()
    
    train_model(args.dataset_path, args.epochs, args.model_path)
