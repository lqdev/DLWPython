"""
Listing 2.1 Loading MNIST dataset in Keras
"""

from keras.datasets import mnist

# Load Training and Testing Datasets
(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

# A look at the training data
train_images.shape
len(train_labels)
train_labels

# A look at the testing data
test_images.shape
len(test_labels)
test_labels

"""
Listing 2.2 The network architecture
"""
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512,activation='relu',input_shape=(28*28,)))
network.add(layers.Dense(10,activation='softmax'))

"""
Listing 2.3 The compilation step
"""

# Loss Function - Performace metric on training
# Optimizer - How network updates itself
# Metric - Performance on training/testing
network.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

"""
Listing 2.4 Preparing the image data
"""

train_images = train_images.reshape((60000,28*28))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000,28*28))
test_images = test_images.astype('float32') / 255

"""
Listing 2.5 Preparing the labels
"""

from keras.utils import to_categorical

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

# Train the network
network.fit(train_images,train_labels,epochs=5,batch_size=128)

test_loss,test_acc = network.evaluate(test_images,test_labels)

print("Test Loss: {0:f} | Test Accuracy: {1:f}".format(test_loss,test_acc))

"""
Listing 2.6 Displaying the fourth digit
"""

(train_images,train_labels),(test_images,test_labels) = mnist.load_data()
print(train_images.ndim)
print(train_images.shape)
print(train_images.dtype)

digit = train_images[4]

import matplotlib.pyplot as plt
plt.imshow(digit,cmap=plt.cm.binary)
plt.show()

