import argparse

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
    net = keras.layers.Dense(10, activation="softmax")(net)

    model = keras.models.Model(inputs=inp, outputs=net)
    model.compile(
        loss=keras.losses.categorical_crossentropy,
        optimizer=keras.optimizers.Adadelta(),
        metrics=["accuracy"])

    return model


def load_model(path):
    return keras.models.load_model(path)


def train_model(model, epochs, save_path):
    (X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
    X_train = np.expand_dims(X_train / 255.0, axis=3)
    X_test = np.expand_dims(X_test / 255.0, axis=3)
    y_train = keras.utils.to_categorical(y_train, 10)
    y_test = keras.utils.to_categorical(y_test, 10)

    model.fit(
        X_train, y_train, batch_size=32, epochs=epochs, verbose=1, validation_data=(X_test, y_test),
        callbacks=[keras.callbacks.ModelCheckpoint(save_path, save_best_only=True)])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--save_path", type=str, default="./model.h5", help="Path to save model to.")
    parser.add_argument("--epochs", type=int, default=10, help="Number of epochs to train for")
    args, _ = parser.parse_known_args()
    
    model = build_model()
    train_model(model, args.epochs, args.save_path)
