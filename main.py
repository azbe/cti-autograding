import copy
import glob
import os
import pdb

import keras
import numpy as np
from PIL import Image
import tensorflow as tf
import tqdm


class Dataset:
    LABELS = ["F 1", "F 2", "F 3", "F 4", "I 1", "I 2", "I 3", "I 4"]
    ANSWERS = ["A", "B", "C", "D"]

    def __init__(self, path, image_size):
        # Load images
        images = sorted(glob.glob(os.path.join(path, "*.jpg")))
        images = [np.array(Image.open(file).resize(image_size, Image.LANCZOS)) for file in tqdm.tqdm(images)]
        self.images = np.array(images)
        
        # Load label files
        labels = sorted(glob.glob(os.path.join(path, "*.txt")))
        labels = [open(file).readlines() for file in tqdm.tqdm(labels)]
        labels = [[line.strip() for line in lines] for lines in labels]
        labels = copy.deepcopy(labels) + copy.deepcopy(labels) + copy.deepcopy(labels)
        labels, answers, scores = [lines[0] for lines in labels], [lines[1:-1] for lines in labels], [lines[-1] for lines in labels]
        assert all([len(answer) == 30 for answer in answers]), answers
        
        # Save labels (which type of test)
        labels = np.array([Dataset.LABELS.index(label) for label in labels])
        self.labels = np.eye(len(Dataset.LABELS))[labels]
        
        # Save test answers
        answers = np.array([[Dataset.ANSWERS.index(ans.split()[1]) for ans in answer] for answer in answers])
        self.answers = np.eye(len(Dataset.ANSWERS))[answers]
        
        # Self final scores
        self.scores = np.array([int(score.split()[1]) for score in scores])


class Model:
    def __init__(self, input_shape, num_answers, num_type_answers, dropout=False):
        self.inputs = keras.layers.Input(input_shape)
        net = keras.layers.BatchNormalization()(self.inputs)
        net = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(net)
        net = keras.layers.BatchNormalization()(net)
        net = keras.layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(net)
        net = keras.layers.BatchNormalization()(net)
        net = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(net)
        net = keras.layers.BatchNormalization()(net)
        net = keras.layers.Conv2D(filters=64, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(net)
        net = keras.layers.BatchNormalization()(net)
        net = keras.layers.Conv2D(filters=128, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(net)
        net = keras.layers.BatchNormalization()(net)
        net = keras.layers.Conv2D(filters=256, kernel_size=(3, 3), strides=(2, 2), padding="same", activation="relu")(net)
        net = keras.layers.BatchNormalization()(net)
        net = keras.layers.GlobalAveragePooling2D()(net)
        if dropout:
            net = keras.layers.Dropout(rate=0.5)(net)
        net = keras.layers.Dense(units=num_answers * num_type_answers, activation="softmax", use_bias=False)(net)
        self.outputs = keras.layers.Reshape((num_answers, num_type_answers))(net)
        self.model = keras.models.Model(inputs=self.inputs, outputs=self.outputs)
        self.model.compile(optimizer="rmsprop", loss="categorical_crossentropy", metrics=["accuracy"])
    
    def train(self, dataset):
        try:
            self.model.fit(x=dataset.images, y=dataset.answers, epochs=10000, verbose=1, validation_split=0.25, shuffle=True)
        except KeyboardInterrupt:
            pass

    def save(self, path):
        self.model.save(path)


def main():
    dataset = Dataset("./data", image_size=(566, 400))
    model = Model(input_shape=(400, 566, 3), num_answers=30, num_type_answers=4, dropout=True)
    model.train(dataset)
    model.save("./model.h5")


if __name__ == "__main__":
    main()