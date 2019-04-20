import os

import matplotlib.pyplot as plt
from keras import layers
from keras import models
from image_avg_size import width_height_of_images
from keras import optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
from keras.utils.np_utils import to_categorical
import pickle
import plaidml.keras


class DogBreedTraining:
    def __init__(self, base_dir='static/data'):
        plaidml.keras.install_backend()
        self.avg_width, self.avg_height = width_height_of_images(base_dir)
        self.conv_base = VGG16(weights='imagenet',
                               include_top=False,
                               input_shape=(self.avg_width, self.avg_height, 3))

        base_dir = base_dir

        self.train_dir = os.path.join(base_dir, 'train')

        self.validation_dir = os.path.join(base_dir, 'train')

        self.test_dir = os.path.join(base_dir, 'test')

        self.bulldog = 'bulldog'
        self.german_shepherd = 'german_shepherd'
        self.golden_retriever = 'golden_retriever'
        self.husky = 'husky'
        self.poodle = 'poodle'
        self.breeds_of_dogs = [self.bulldog, self.german_shepherd, self.golden_retriever, self.husky, self.poodle]

        self.train_bulldog_dir = None
        self.train_german_shepherd_dir = None
        self.train_golden_retriever_dir = None
        self.train_husky_dir = None
        self.train_poodle_dir = None

        self.validation_bulldog_dir = None
        self.validation_german_shepherd_dir = None
        self.validation_golden_retriever_dir = None
        self.validation_husky_dir = None
        self.validation_poodle_dir = None

        self.test_bulldog_dir = None
        self.test_german_shepherd_dir = None
        self.test_golden_retriever_dir = None
        self.test_husky_dir = None
        self.test_poodle_dir = None

        self.train_generator = None
        self.validation_generator = None

    def directory_init(self):
        self.train_bulldog_dir = os.path.join(self.train_dir, self.bulldog)
        self.train_german_shepherd_dir = os.path.join(self.train_dir, self.german_shepherd)
        self.train_golden_retriever_dir = os.path.join(self.train_dir, self.golden_retriever)
        self.train_husky_dir = os.path.join(self.train_dir, self.husky)
        self.train_poodle_dir = os.path.join(self.train_dir, self.poodle)

        self.validation_bulldog_dir = os.path.join(self.validation_dir, self.bulldog)
        self.validation_german_shepherd_dir = os.path.join(self.validation_dir, self.german_shepherd)
        self.validation_golden_retriever_dir = os.path.join(self.validation_dir, self.golden_retriever)
        self.validation_husky_dir = os.path.join(self.validation_dir, self.husky)
        self.validation_poodle_dir = os.path.join(self.validation_dir, self.poodle)

        self.test_bulldog_dir = os.path.join(self.test_dir, self.bulldog)
        self.test_german_shepherd_dir = os.path.join(self.test_dir, self.german_shepherd)
        self.test_golden_retriever_dir = os.path.join(self.test_dir, self.golden_retriever)
        self.test_husky_dir = os.path.join(self.test_dir, self.husky)
        self.test_poodle_dir = os.path.join(self.test_dir, self.poodle)

    def image_data_gen(self):
        train_datagen = ImageDataGenerator(rescale=1. / 255)
        test_datagen = ImageDataGenerator(rescale=1. / 255)

        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.avg_width, self.avg_height),
            batch_size=20,
            class_mode='categorical'
        )

        self.validation_generator = test_datagen.flow_from_directory(
            self.validation_dir,
            target_size=(self.avg_width, self.avg_height),
            batch_size=20,
            class_mode='categorical'
        )

    def dogs_breed_modeling(self):
        model = models.Sequential()
        model.add(self.conv_base)
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(5, activation='softmax'))
