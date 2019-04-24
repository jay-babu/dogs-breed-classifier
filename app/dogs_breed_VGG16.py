import os

import matplotlib.pyplot as plt
from keras import layers
from keras import models
from image_avg_size import width_height_of_images
from keras import optimizers
from keras.applications import VGG16
from keras.preprocessing.image import ImageDataGenerator
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
        self.test_datagen = None

        self.history = None

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
        validation_datagen = ImageDataGenerator(rescale=1. / 255)
        self.test_datagen = ImageDataGenerator(rescale=1. / 255)

        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(self.avg_width, self.avg_height),
            batch_size=20,
            class_mode='categorical'
        )

        self.validation_generator = validation_datagen.flow_from_directory(
            self.validation_dir,
            target_size=(self.avg_width, self.avg_height),
            batch_size=20,
            class_mode='categorical'
        )

    def dogs_breed_modeling(self):
        model = models.Sequential()
        model.add(self.conv_base)
        print(model.summary())
        model.add(layers.Flatten())
        model.add(layers.Dense(256, activation='relu'))
        model.add(layers.Dense(5, activation='softmax'))

        print(model.summary())

        model.compile(loss='categorical_crossentropy',
                      optimizer=optimizers.Adam(),
                      metrics=['accuracy'])

        self.history = model.fit_generator(
            self.train_generator,
            steps_per_epoch=100,
            epochs=30,
            validation_data=self.validation_generator,
            validation_steps=50
        )

        model.save('dogs_breed_1.h5')

    def results(self):
        acc = self.history.history['acc']
        val_acc = self.history.history['val_acc']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(1, len(acc) + 1)

        plt.plot(epochs, acc, 'bo', label='Training acc')
        plt.plot(epochs, val_acc, 'b', label='Validation acc')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, 'bo', label='Training loss')
        plt.plot(epochs, val_loss, 'b', label='Validation loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()


if __name__ == '__main__':
    dogs = DogBreedTraining('../static/data')
    dogs.directory_init()
    dogs.image_data_gen()
    dogs.dogs_breed_modeling()
    dogs.results()
