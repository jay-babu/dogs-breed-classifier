# **Deep Learning With Python**
The book I used to learn more about Machine Learning and, more specifically, Keras. I read the entire book, but will discuss the more important chapters in summary and the most important chapter in greater detail.

### **Chapter 2**
##### Section 2.1
This section gave me my first exposure to an example of a neural network. It used the MNIST dataset, which is considered the "Hello World" of deep learning. It is usually used to verify that your algorithms are working as expected. I learned about optimizer, loss functions, and metrics during this section. 

##### Section 2.2 Onwards
The rest of the sections gave exposure to how data is represented and what a tensor is. Numpy only accepts certain types of data in certain formats. Machine Learning usually uses tensors instead of regular data types such as lists. I learned about the relu operation/activation, which is widely used. In reality, it uses a derivative to obtain a gradient descent. Without using the derivative, the operation would be time & memory consuming, since it would have to freeze the entire network and then guess values until a meaningful representation was being made. A derivative allows the function to quickly find what values would be best for the iteration.

### **Chapter 3**
##### Section 3.1 - 3.3
These sections went over how to setup a deep-learning workstation and install Keras on your local machine. It also gave a brief overview on how to use AWS if you do not have access to a GPU. I used Anaconda as my Python package manager. 

##### Section 3.4
A certain type of classification in machine learning is machine learning. It is called binary because it can only have two results. The code for it was as follows:
```python
from keras.datasets import imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data( num_words=10000)

import numpy as np
def vectorize_seq(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')


from keras import models from keras import layers

model = models.Sequential() 
model.add(layers.Dense(16, activation='relu', input_shape=(10000,))) 
model.add(layers.Dense(16, activation='relu')) 
model.add(layers.Dense(1, activation='sigmoid'))

from keras import losses from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001), loss=losses.binary_crossentropy, metrics=[metrics.binary_accuracy])

history = model.fit(partial_x_train, partial_y_train, epochs=4, batch_size=512, validation_data=(x_val, y_val))
```
I learned how to vectorize data manually, but I could also use the built-in numpy function to do so. The optimizer has to have a good learning rate, so it does not asymptote too quickly but also learns as much as possible from the data. Using this naive approach, I obtained an accuracy of 88%.

##### Section 3.5
```python
from keras.datasets import reuters

(train_data, train_labels), (test_data, test_labels) = reuters.load_data(num_words=10000)

import numpy as np

def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results

# Our vectorized training data
x_train = vectorize_sequences(train_data)
# Our vectorized test data
x_test = vectorize_sequences(test_data)

from keras.utils.np_utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64, activation='relu', input_shape=(10000,)))
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(46, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=20,
                    batch_size=512,
                    validation_data=(x_val, y_val))
```
This model obtained approximately 88% accuracy, which is considered good because it did not have state of art techniques implemented and the dataset was large. The model did overfit quickly though because of the high learning rate.

### **Chapter 4**
##### Section 4.1
Within this section, I learned about four different branches of machine learning. The first is called Supervised learning, which is the one I used for my Dogs Breed Classifier. It is done by mapping input data such as images to a target such as its label. 

Another branch is unsupervised learning. This is done by finding interesting patterns and transformations without any targets (labels). This type is often used to do tasks such as clustering and dimensionality reduction.

The third branch is called self-supervised learning, which is basically supervised learning, but without human generated labels. The machine creates its own labels based on the input data and its patterns.

The last and least known branch is reinforcement learning. The machine receives information about its environment and then chooses an action that maximizes its reward. For example, in a game like Mortal Kombat, the machine would pick a move that would cause the most damage to its opponent. 

##### Section 4.4
This section goes over various ways to prevent overfitting and underfitting. The first is reducing the network's size (aka the learnable parameters). If the size is too big then it creates a perfect map, where it has not really learned the features of the data, but is able to recognize the same images. There is no formula to figure out the right number of layers, but estimations can be made. I had to experiment with this throughout the training I used.

Another method is to regularize the weights for each layer. This means you explicitly tell the model what the maximum weight can be and it scales based on that. 
```python
from keras import regularizers

model = models.Sequential() 
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu', input_shape=(10000,))) 
model.add(layers.Dense(16, kernel_regularizer=regularizers.l2(0.001), activation='relu')) 
model.add(layers.Dense(1, activation='sigmoid'))
```

The last and most effective method is dropout. This method involves randomly dropping out certain features within the data and the model would have to train without using a certain feature for that instance. This is done randomly throughout model training.

```python
model = models.Sequential() 
model.add(layers.Dense(16, activation='relu', input_shape=(10000,))) 
model.add(layers.Dropout(0.5)) 
model.add(layers.Dense(16, activation='relu')) 
model.add(layers.Dropout(0.5)) model.add(layers.Dense(1, activation='sigmoid'))
```
To recap, these are the most common ways to prevent overfitting in neural networks:
- Get more training data
- Reduce the capacity of the network.
- Add weight regularization
- Add dropout

### **Chapter 5**
This is the last chapter I will review from the book.

##### Section 5.1
```python
from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation='relu'))
model.add(layers.Dense(10, activation='softmax'))


from keras.datasets import mnist
from keras.utils import to_categorical

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

train_images = train_images.reshape((60000, 28, 28, 1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000, 28, 28, 1))
test_images = test_images.astype('float32') / 255

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)
model.compile(optimizer='rmsprop',
              loss='categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5, batch_size=64)
test_loss, test_acc = model.evaluate(test_images, test_labels)
```

The code above goes over various benefits of using convnets and the reason why they are used the most within computer vision. Whereas the densely connected network from chapter 2 had a test accuracy of 97.8%, the basic convnet has a test accuracy of 99.3%: we decreased the error rate by 68% (relative).

![Convolution Filtering](README_images/Convolution_Filtering.png)

The images shows how it filters are used to learn about the image. As more layers are added to the model, it starts using more filters to learn more and more details about the data.

###### Section 5.2
I have linked my Jupyter Notebook to show what was done for this section. 
[Training a convnet on a small dataset](https://github.com/jayp0521/dogs-breed-classifier/blob/dev/app/convnets_on_small_datasets.ipynb)

# **Dogs Breed Classifier**

### **Dogs_file_renamer**
This python file was used to rename all different breeds of dogs to format this format:\
Number (This was an arbitrary integer). Subtype of breed. File Extension

The reason each breed has subtypes is so the model is able to train on various types of breeds 
and not see them some for the first time. For example, the model should train on white bulldogs, so it is able to expect
them in the model, instead of there being a risk of it being seen the first time in the test.

### **Dogs_file_organizer**
The purpose of this file was to get 70% of each subtype of each breed and place it in the training folder, 20% in the
validation folder, 10% in the test folder.

### **Image_Avg_Size**
This file enabled us to find the average size of all of the images in the dataset. It is allowed one parameter which is used to navigate the train, test, and validation subfolders and then the images. The images are converted to RGB channels and saved as a JPG or PNG.

The output of this code is used in the generators for the target_size parameter.

### **Dogs_Breed_VGG16**
The purpose of this file was to train the model for classifying various breeds of dogs. Five various breeds were used, namely: bulldogs, german shepherds, golden retrievers, huskies, poodles.

At first, I tried training the model using my AMD GPU using plaidml to help utilize it. On a smaller dataset, it was working fine, but the Dog Dataset was too big, so a better GPU had to be found. I used Google Colab to train the model as it was tractable.

##### Import Statements
I imported matplotlib to plot my models training and validation accuracy and loss overtime.

I used "os" and "shutil" libraries to navigate and import various training folders.

The keras libraries I used were layers, models, optimizers, VGG16 (pre-trained mode), ImageDataGenerator.

The mean from statistics was imported to and used in Image_Avg_Size.

from PIL import Image was used to help convert all images to RGB and all GIFs to JPGs. 

##### PyDrive
```python
!pip install PyDrive
from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from google.colab import files
from oauth2client.client import GoogleCredentials

auth.authenticate_user()
gauth = GoogleAuth()
gauth.credentials = GoogleCredentials.get_application_default()
drive = GoogleDrive(gauth)

download = drive.CreateFile({'id': '1EgZuR10bXkNxK3g1ppfz_T2aRuUxXwC'})
download.GetContentFile('dataset.tar')
```

These commands were used in a second code block to help download my dataset directly to Google Colab's temporary runtime storage. The dataset was uploaded to Google Drive as a tar and extracted in Colab. All the authentication modules were required to connect to Google Drive and then download the dataset.

##### dataset_cleaner
The dataset_cleaner was created to clean the dataset, since Google Colab adds its own metadata when extracting a tar file. The metadata interfered with the trainer, which is why it was cleaned. 

#### class DogBreedTraining
##### Init Method
The init method setup all of the various classes and various folders. The convolution base was setup to have weights of imagenet, shape of 256, 256. Initially the average image width and height was used, but after thoroughly reading the book, I realized a square image (meaning of same width and height). I used 256 because, after reading about VGG16, it seemed it would be optimal.

The directories for all of the images were initially set to none.
##### Directory Init
The call to directory_init initializes all of the directories using
```python
os.path.join()
```
and the base_dir that was passed in.

##### Image Data Gen
This method allows for data augmentation to be used on the training data and the validation generator. The keras function *flow_from_directory* was used to gather images from each directory for training and validation.

ImageDataGenerator used these parameters:
- rescale - Color values are between 0 and 1
- rotation_range - How much an image can be rotated
- width_shift_range - How much an image can be shifted to the left and right
- height_shift_range - How much an image can be shifted up and down
- shear_range - How much an image can be distorted
- zoom_range - How much an image and be zoomed in or out
- horizontal_flip - If an image can be flipped across its horizontal (y) axis or not
- fill_mode - How to fill an image's black spots because of shifting

Flow_from_directory has these parameters:
- target_size - What the image's target size should be
- batch_size - How many images should be in a batch
- class_mode - The types of labels an image could have. For example, if there were only two options, it would be binary.

 The code for it goes as follows.
 ```python
 train_datagen = ImageDataGenerator(
            rescale=1. / 255,
            rotation_range=40, 
            width_shift_range=0.2, 
            height_shift_range=0.2, 
            shear_range=0.2, 
            zoom_range=0.2, 
            horizontal_flip=True, 
            fill_mode='nearest')
        
        validation_datagen = ImageDataGenerator(rescale=1. / 255)
        
        self.test_datagen = ImageDataGenerator(rescale=1. / 255)

        self.train_generator = train_datagen.flow_from_directory(
            self.train_dir,
            target_size=(256, 256),
            batch_size=batch_size,
            class_mode='categorical'
        )

        self.validation_generator = validation_datagen.flow_from_directory(
            self.validation_dir,
            target_size=(256, 256),
            batch_size=batch_size,
            class_mode='categorical'
        )
```