from keras.models import load_model
import plaidml.keras
from keras.preprocessing.image import ImageDataGenerator

plaidml.keras.install_backend()
# import numpy as np
# from keras.preprocessing import image

model = load_model('dogs_breed_4 (1).h5')
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
# print(model.predict('static/bulldog/1. Regular/91.Regular.png'))
test_dir = "static/data/test"
test_datagen = ImageDataGenerator(rescale=1. / 255)
test_generator = test_datagen.flow_from_directory(test_dir,
                                                  target_size=(256, 256),
                                                  batch_size=20,
                                                  class_mode='categorical')

# predict the result

test_loss, test_acc = model.evaluate_generator(test_generator, steps=50)
print(test_loss, test_acc)
