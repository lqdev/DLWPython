"""
Listing 5.1 Instantiating a small convnet
"""

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(28,28,1)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))

model.summary()

"""
Listing 5.2 Adding a classifier on top of the convnet
"""

model.add(layers.Flatten())
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(10,activation='softmax'))

model.summary()

"""
Listing 5.3 Training the convnet on MNIST images
"""
from keras.datasets import mnist
from keras.utils import to_categorical

(train_images,train_labels),(test_images,test_labels) = mnist.load_data()

train_images = train_images.reshape((60000,28,28,1))
train_images = train_images.astype('float32') / 255

test_images = test_images.reshape((10000,28,28,1))
test_images = test_images.astype('float32') / 32

train_labels = to_categorical(train_labels)
test_labels = to_categorical(test_labels)

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=5,batch_size=64)

test_loss,test_acc = model.evaluate(test_images,test_labels)

test_acc

"""
Listing 5.4 Copying images to training,validation, and test directories
"""

import os, shutil

original_dataset_dir = '/home/lqdev/Downloads/kaggle_original_data/train'

base_dir = '/home/lqdev/Downloads/cats_and_dogs_small'

# Create Reduced Data Directory
os.mkdir(base_dir)

# Create Train/Validation/Test Directories
train_dir = os.path.join(base_dir,'train')
os.mkdir(train_dir)
validation_dir = os.path.join(base_dir,'validation')
os.mkdir(validation_dir)
test_dir = os.path.join(base_dir,'test')
os.mkdir(test_dir)

# Create Individual Tag Directories
train_cats_dir = os.path.join(train_dir,'cats')
os.mkdir(train_cats_dir)

train_dogs_dir = os.path.join(train_dir,'dogs')
os.mkdir(train_dogs_dir)

validation_cats_dir = os.path.join(validation_dir,'cats')
os.mkdir(validation_cats_dir)

validation_dogs_dir = os.path.join(validation_dir,'dogs')
os.mkdir(validation_dogs_dir)

test_cats_dir = os.path.join(test_dir,'cats')
os.mkdir(test_cats_dir)

test_dogs_dir = os.path.join(test_dir,'dogs')
os.mkdir(test_dogs_dir)

# Copy Cat Files
fnames = ['cat.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_cats_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['cat.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_cats_dir, fname)
    shutil.copyfile(src, dst)

# Copy Dog Files
fnames = ['dog.{}.jpg'.format(i) for i in range(1000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(train_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1000,1500)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(validation_dogs_dir, fname)
    shutil.copyfile(src, dst)

fnames = ['dog.{}.jpg'.format(i) for i in range(1500,2000)]
for fname in fnames:
    src = os.path.join(original_dataset_dir, fname)
    dst = os.path.join(test_dogs_dir, fname)
    shutil.copyfile(src, dst)


# Check number of files in directories
print('total training cat images:', len(os.listdir(train_cats_dir)))
print('total training dog images:', len(os.listdir(train_dogs_dir)))
print('total validation cat images:', len(os.listdir(validation_cats_dir)))
print('total validation dog images:', len(os.listdir(validation_dogs_dir)))
print('total test cat images:', len(os.listdir(test_cats_dir)))
print('total test dog images:', len(os.listdir(test_dogs_dir)))

"""
Listing 5.5 Instantiating a small convnet for dogs vs. cats classification
"""

from keras import layers
from keras import models

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D(2,2))
model.add(layers.Flatten())
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.summary()

"""
Listing 5.6 Configuring the model for training
"""

from keras import optimizers

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])

"""
Listing 5.7 Using ImageDataGenerator to read images from directories
"""

from keras.preprocessing.image import ImageDataGenerator

#Rescale Images by 1/255
train_datagen = ImageDataGenerator(rescale=1./255)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary')

# See shape of generator
for data_batch,labels_batch in train_generator:
    print('data batch shape:', data_batch.shape)
    print('labels batch shape:', labels_batch.shape)
    break

"""
Listing 5.8 Fiting the model using a batch generator
"""    

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50)

"""
Listing 5.9 Saving the model
"""

model.save('cats_and_dogs_small_1.h5')

"""
Listing 5.10 Displaying curves of loss and accuracy during training
"""

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc) + 1)

plt.clf()

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

"""
Listing 5.11 Setting up a data augmentation configuration via ImageDataGenerator
"""

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest')

"""
Listing 5.12 Displaying some randomly augmented training images
"""    

from keras.preprocessing import image

fnames = [os.path.join(train_cats_dir,fname) for fname in os.listdir(train_cats_dir)]

img_path = fnames[3]

img = image.load_img(img_path,target_size=(150,150))

x = image.img_to_array(img)

x = x.reshape((1,) + x.shape)

i = 0

for batch in datagen.flow(x,batch_size=1):
    plt.figure(i)
    imgplot = plt.imshow(image.array_to_img(batch[0]))
    i += 1
    if i %4 == 0:
        break

plt.show()

"""
Listing 5.13 Defining a new convnet that includes dropout
"""

model = models.Sequential()
model.add(layers.Conv2D(32,(3,3),activation='relu',input_shape=(150,150,3)))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(64,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Conv2D(128,(3,3),activation='relu'))
model.add(layers.MaxPooling2D((2,2)))
model.add(layers.Flatten())
model.add(layers.Dropout(0.5))
model.add(layers.Dense(512,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-4),metrics=['acc'])

model.summary()

"""
Listing 5.14 Training the convnet using data-augmentation generators
"""

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

validation_generator = test_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=32,
    class_mode='binary'
)

# Should be this
# history = model.fit_generator(
#     train_generator,
#     steps_per_epoch=100,
#     epochs=100,
#     validation_data=validation_generator,
#     validation_steps=50
# )

# Added to make it run quicker on CPU
history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=5,
    validation_data=validation_generator,
    validation_steps=50
)

"""
Listing 5.15 Saving the model
"""

model.save('cats_and_dogs_small_2.h5')

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc) + 1)

plt.clf()

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

"""
Listing 5.16 Instantiating the VGG16 convolutional base
"""

from keras.applications import VGG16

conv_base = VGG16(weights='imagenet',include_top=False,input_shape=(150,150,3))

conv_base.summary()

"""
Listing 5.17 Extracting features using the pretrained convolutional base
"""

import os
import numpy as np
from keras.preprocessing.image import ImageDataGenerator

base_dir = '/home/lqdev/Downloads/cats_and_dogs_small'
train_dir = os.path.join(base_dir,'train')
validation_dir = os.path.join(base_dir,'validation')
test_dir = os.path.join(base_dir,'test')

datagen = ImageDataGenerator(rescale=1./255)

batch_size = 20

def extract_features(directory,sample_count):
    features = np.zeros(shape=(sample_count,4,4,512))
    labels = np.zeros(shape=(sample_count))
    generator = datagen.flow_from_directory(
        directory,
        target_size=(150,150),
        batch_size=batch_size,
        class_mode='binary'
    )
    i = 0
    for inputs_batch,labels_batch in generator:
        features_batch = conv_base.predict(inputs_batch)
        features[i * batch_size:(i + 1) * batch_size] = features_batch
        labels[i * batch_size:(i + 1) * batch_size] = labels_batch
        i += 1
        if i * batch_size >= sample_count:
            break
    return features,labels

train_features,train_labels = extract_features(train_dir,2000)
validation_features,validation_labels = extract_features(validation_dir,1000)
test_features,test_labels = extract_features(test_dir,1000)

train_features = np.reshape(train_features,(2000, 4 * 4 * 512))
validation_features = np.reshape(validation_features,(1000, 4 * 4 * 512))
test_features = np.reshape(test_features,(1000, 4 * 4 * 512))

"""
Listing 5.18 Defining and training the densely connected classifier
"""

from keras import models
from keras import layers
from keras import optimizers

model = models.Sequential()
model.add(layers.Dense(256,activation='relu',input_dim=4*4*512))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer=optimizers.RMSprop(lr=2e-5),loss='binary_crossentropy',metrics=['acc'])

history = model.fit(train_features,train_labels,epochs=30,batch_size=20,validation_data=(validation_features,validation_labels))

"""
Listing 5.19 Plotting the results
"""

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc) + 1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

"""
Listing 5.20 Adding a densely connected classifier on top of the convolutional base
"""

from keras import models
from keras import layers

model = models.Sequential()
model.add(conv_base)
model.add(layers.Flatten())
model.add(layers.Dense(256,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.summary()

# Freeze weights (prevent weights of convolutional base from being updated on training. This must be done before compilation)
print('This is the number of trainable weights before freezing the conv base:', len(model.trainable_weights))
conv_base.trainable = False
print('This is the number of trainable weights before freezing the conv base:', len(model.trainable_weights))

"""
Listing 5.21 Training the model end to end with a frozen convolutional base
"""

from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers

train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

validation_generator = train_datagen.flow_from_directory(
    validation_dir,
    target_size=(150,150),
    batch_size=20,
    class_mode='binary'
)

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=2e-5),metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=30,
    validation_data=validation_generator,
    validation_steps=50
)

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc) + 1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

"""
Listing 5.22 Freezing all layers up to a specific one
"""

# This step takes place after training frozen conv base with newly added Dense layers

conv_base.trainable = True

set_trainable = False
for layer in conv_base.layers:
    if layer.name == 'block5_conv1':
        set_trainable = True
    if set_trainable:
        layer.trainable = True
    else:
        layer.trainable = False

"""
Listing 5.23 Fine-tuning the model
"""        

model.compile(loss='binary_crossentropy',optimizer=optimizers.RMSprop(lr=1e-5),metrics=['acc'])

history = model.fit_generator(
    train_generator,
    steps_per_epoch=100,
    epochs=100,
    validation_data=validation_generator,
    validation_steps=50
)

"""
Listing 5.24 Smoothing the plots
"""

def smooth_curve(points, factor=0.8):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc) + 1)

plt.plot(epochs,smooth_curve(acc), 'bo', label='Smoothed training acc')
plt.plot(epochs,smooth_curve(val_acc), 'b', label='Smoothed validation acc')
plt.title('Training and validation accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,smooth_curve(loss), 'bo', label='Smoothed training loss')
plt.plot(epochs,smooth_curve(val_loss), 'b', label='Smoothed validation loss')
plt.title('Training and validation loss')
plt.legend()

plt.show()

# Evaluate Model on Test Data
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(150, 150),
    batch_size=20,
    class_mode='binary')

test_loss,test_acc = model.evaluate_generator(test_generator,steps=50)
print('test acc:', test_acc)

"""
Listing 5.25 Preprocessing an image
"""

img_path = '/home/lqdev/Downloads/cats_and_dogs_small/test/cats/cat.1700.jpg'

from keras.preprocessing.image import image
import numpy as np

img = image.load_img(img_path,target_size=(150,150))
img_tensor = image.img_to_array(img)
img_tensor = np.expand_dims(img_tensor,axis=0)
img_tensor /= 255.

print(img_tensor.shape)

"""
Listing 5.26 Displaying the test picture
"""

import matplotlib.pyplot as plt

plt.imshow(img_tensor[0])
plt.show()

"""
Listing 5.27 Instantiating a model from an input tensor and a list of output tensors
"""

from keras.models import load_model
model = load_model('cats_and_dogs_small_2.h5')

from keras import models

layer_outputs = [layer.output for layer in model.layers[:8]]

activation_model = models.Model(inputs=model.input,outputs=layer_outputs)

"""
Listing 5.28 Running the model in predict mode
"""

activations = activation_model.predict(img_tensor)

first_layer_activation = activations[0]
print(first_layer_activation.shape)

"""
Listing 5.29 Visualizing the fourth channel
"""

import matplotlib.pyplot as plt

plt.matshow(first_layer_activation[0,:,:,4],cmap='viridis')
plt.show()

"""
Listing 5.30 Visualizing the seventh channel
"""

plt.matshow(first_layer_activation[0,:,:,7],cmap='viridis')
plt.show()

"""
Listing 5.31 Visualizing every channel in every intermediate activation
"""

layer_names = []

# Names of layers so you can have them as part of your plot
for layer in model.layers[:8]:
    layer_names.append(layer.name)

images_per_row = 16

for layer_name,layer_activation in zip(layer_names,activations): # Displays the feature maps
    n_features = layer_activation.shape[-1] #Number of features in the feature map

    size = layer_activation.shape[1] #The feature map has ashape (l,size,size,n_features)

    n_cols = n_features // images_per_row # Tiles the activation channels channels in this matrix

    display_grid = np.zeros((size * n_cols,images_per_row * size))

    for col in range(n_cols): # Tiles each filter into a big horizontal grid
        for row in range(images_per_row): # Post processes the feature to make it visually palatable
            channel_image = layer_activation[0,:,:,col * images_per_row + row]
        
            channel_image -= channel_image.mean()
            channel_image /= channel_image.std()
            channel_image *= 64
            channel_image += 128
            channel_image = np.clip(channel_image,0,255).astype('uint8')
            display_grid[col * size : (col + 1) * size,row * size : (row + 1) * size] = channel_image

    scale = 1. /size
    plt.figure(figsize=(scale * display_grid.shape[0],scale * display_grid.shape[1]))
    plt.title(layer_name)
    plt.grid(False)
    plt.imshow(display_grid,aspect='auto',cmap='viridis')

plt.show()

"""
Listing 5.32 Defining the loss tensor for filter visualization
"""

from keras.applications import VGG16
from keras import backend as K

model = VGG16(weights='imagenet',include_top=False)

layer_name = 'block3_conv1'
filter_index = 0

layer_output = model.get_layer(layer_name).output
loss = K.mean(layer_output[:,:,:,filter_index])

"""
Listing 5.33 Obtainig the gradient of the loss with regard to the input
"""

# The call to gradients returns a list of tensors (of size 1 in this case). 
# Hence keep only first elemtent which is a tensor
grads = K.gradients(loss,model.input)[0] 

"""
Listing 5.34 Gradient-normalization trick
"""

grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5) # Smmoth SGD with L2 Regularization

"""
Listing 5.35 Fetching Numpy output values given Numpy input values
"""

iterate = K.function([model.input],[loss,grads])

import numpy as np
loss_value,grads_value = iterate([np.zeros((1,150,150,3))])

"""
Listing 5.36 Loss maximization via stochastic gradient descent
"""

input_img_data = np.random.random((1,150,150,3)) * 20 + 128. # Starts from a gray image with some noise

step = 1. # Magnitude of each gradient update
for i in range(40): # Runs gradient ascent for 40 steps
    loss_value,grads_value = iterate([input_img_data]) # Compute loss value and gradient value

    input_img_data += grads_value * step # Adjusts the input image in the direction that maximizes the loss

"""
Listing 5.37 Utility function to convert a tensor into a valid image
"""

def deprocess_image(x):
    # Normalizes the tensor;Centers on 0;Ensures that std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    x += 0.5
    x = np.clip(x,0,1)

    x *= 255
    x = np.clip(x,0,255).astype('uint8') #Convert to an RGB Array
    return x

"""
Listing 5.38 Function to generate filter visualizations
"""

def generate_pattern(layer_name,filter_index,size=150):
    # Builds a loss function tat maximizes the activaiton of the nth filter of the layer under consideration
    layer_output = model.get_layer(layer_name).output
    loss = K.mean(layer_output[:,:,:,filter_index])

    grads = K.gradients(loss,model.input)[0] # Computes the gradient of the input pucture with regard to this loss

    grads /= (K.sqrt(K.mean(K.square(grads))) + 1e-5) # Normalization trick: normalizes the gradient

    iterate = K.function([model.input],[loss,grads]) # Return the loss and grads given the input picture

    input_img_data = np.random.random((1,size,size,3)) * 20 + 128. # Starts from a gray image with some noise

    step = 1.

    # Runs graduent ascent for 40 steps
    for i in range(40):
        loss_value,grads_value = iterate([input_img_data])
        input_img_data += grads_value * step
    
    img = input_img_data[0]
    return deprocess_image(img)

plt.imshow(generate_pattern('block3_conv1',0))
plt.show()

"""
Listing 5.39 Generating a grid of all filter response patterns in a layer
"""

layer_name = 'block3_conv1'
size = 64
margin = 5

results = np.zeros((8 * size + 7 * margin, 8 * size + 7 * margin, 3)) # Empty (black) image to store results

for i in range(8):
    for j in range(8):
        # Generates the pattern for filter i + (j * 8) in layer_name
        filter_img = generate_pattern(layer_name, i + (j * 8), size=size)

        # Puts the result in the square (i,j) of the results grid
        horizontal_start = i * size + i * margin
        horizontal_end = horizontal_start + size
        vertical_start = j * size + j * margin
        vertical_end = vertical_start + size
        results[horizontal_start: horizontal_end,vertical_start: vertical_end, :] = filter_img

plt.figure(figsize=(20, 20))
plt.imshow(results)
plt.show()

"""
Listing 5.40 Loading the VGG16 network with pretrained weights
"""

from keras.applications import VGG16

model = VGG16(weights='imagenet')

"""
Listing 5.41 Preprocessing an input image for VGG16
"""

from keras.preprocessing import image
from keras.applications.vgg16 import preprocess_input, decode_predictions
import numpy as np

img_path = "/home/lqdev/Downloads/creative_commons_elephant.jpg" # Local path to the target image

img = image.load_img(img_path,target_size=(224,224)) # Python Imaging Library (PIL) image of size 224 x 224

x = image.img_to_array(img) # float32 Numpy array of shape (224,224,3)

x = np.expand_dims(x, axis=0) # Adds a dimension to transform the array into a batch of size (1,224,224,3)

x = preprocess_input(x) # Preprocess the batch (this does channel-wise color normalization)

preds = model.predict(x)

print('Predicted:', decode_predictions(preds, top=3)[0])

np.argmax(preds[0])

"""
Listing 5.42 Setting up the Grad-CAM algorithm
"""

# African elephant entryu in the prediction vector
african_elephant_output = model.output[:,386]

# Output feature map of the block5_conv3 layer, the last convolutional layer in VGG16
last_conv_layer = model.get_layer('block5_conv3') 

# Gradient of the "African elephant" class with regard to the output feature map of block5_conv3
grads = K.gradients(african_elephant_output,last_conv_layer.output)[0]

# Vector of shape (512,) where each entry is the m,ean intensity of the gradient over a specific feature-map channel
pooled_grads = K.mean(grads,axis=(0,1,2))

# Lets you access the values of the quantities you just defined
# pooled_grads and the output feature map of block5_conv3, given a sample image
iterate = K.function([model.input],[pooled_grads,last_conv_layer.output[0]])

# Values of these two quantities, as Numpy arrays, given the sample image of two elephants
pooled_grads_value,conv_layer_output_value = iterate([x])

# Multiplies each channel in the feature-map array by "how important this cahnnel is" with the "elephant" class
for i in range(512):
    conv_layer_output_value[:,:,i] *= pooled_grads_value[i]

# The channel-wise mean of the resulting feature map is the heatmap of the class activation
heatmap = np.mean(conv_layer_output_value,axis=-1)

"""
Listing 5.43 Heatmap post-processing
"""

heatmap = np.maximum(heatmap,0)
heatmap /= np.max(heatmap)
plt.matshow(heatmap)
plt.show()

"""
Listing 5.44 Superimposing the heatmap with the original picture
"""

import cv2
img = cv2.imread(img_path)
heatmap = cv2.resize(heatmap,(img.shape[1],img.shape[0]))
heatmap = np.uint8(255 * heatmap)
heatmap = cv2.applyColorMap(heatmap,cv2.COLORMAP_JET)
superimposed_img = heatmap * 0.4 + img
cv2.imwrite('/home/lqdev/Downloads/elephant_cam.jpg',superimposed_img)