# IMDB Dataset

"""
Listing 3.1 Loading the IMDB dataset
"""

from keras.datasets import imdb

(train_data,train_labels),(test_data,test_labels) = imdb.load_data(num_words=10000)

train_data.shape

# Decode reviews back to English
word_index = imdb.get_word_index()
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])
decoded_review = ' '.join([reverse_word_index.get(i-3,'?') for i in train_data[0]])
print(decoded_review)

"""
Listing 3.2 Encoding the integer sequencies into a binary matrix
"""
import numpy as np

def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

x_train[0]

# Vectorize Labels
y_train = np.asarray(train_labels).astype('float32')
y_test = np.asarray(test_labels).astype('float32')

"""
Listing 3.3 The model definition
"""

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

"""
Listing 3.4 Compiling the model
"""
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

"""
Listing 3.5 Configuring the optimizer
"""
from keras import optimizers

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy'])

"""
Listing 3.6 Using custom losses and metrics
"""

from keras import losses
from keras import metrics

model.compile(optimizer=optimizers.RMSprop(lr=0.001),
    loss=losses.binary_crossentropy,
    metrics=[metrics.binary_accuracy])

"""
Listing 3.7 Setting aside a validation set
"""

x_val = x_train[:10000]
partial_x_train = x_train[10000:]

y_val = y_train[:10000]
partial_y_train = y_train[10000:]

"""
Listing 3.8 Training your model
"""
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))

history_dict = history.history
history_dict.keys()

"""
Listing 3.9  Plotting the training and validation loss
"""

import matplotlib.pyplot as plt

history_dict = history.history
loss_values = history_dict['loss']
val_loss_values = history_dict['val_loss']

acc = history_dict['acc']

epochs = range(1,len(acc) + 1)

plt.plot(epochs,loss_values,'bo',label="Training loss")
plt.plot(epochs,val_loss_values,'b',label="Validation loss")
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

"""
Listing 3.10 Plotting the training and validation accuracy
"""

plt.clf()
acc_values = history_dict['acc']
val_acc_values = history_dict['val_acc']

plt.plot(epochs,acc_values,'bo',label="Training acc")
plt.plot(epochs,val_acc_values,'b',label="Validation acc")
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

"""
Listing 3.11 Retraining a model from scratch
"""

model = models.Sequential()

model.add(layers.Dense(16,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(16,activation='relu'))
model.add(layers.Dense(1,activation='sigmoid'))

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['accuracy'])

model.fit(x_train,y_train,epochs=4,batch_size=512)

results = model.evaluate(x_test,y_test)

# Reuters Dataset

"""
Listing 3.12 Loading the Reuters dataset
"""

from keras.datasets import reuters

(train_data,train_labels),(test_data,test_labels) = reuters.load_data(num_words=10000)

len(train_data)
len(test_data)

train_data[10]

"""
Listing 3.13 Decoding newswires back to text
"""
word_index = reuters.get_word_index()
reverse_word_index = dict([(value,key) for (key,value) in word_index.items()])
decoded_newswire = ' '.join([reverse_word_index.get(i - 3, '?') for i in train_data[0]])
decoded_newswire
train_labels[0]

"""
Listing 3.14 Encoding the data
"""

import numpy as np

def vectorize_sequences(sequences,dimension=10000):
    results = np.zeros((len(sequences),dimension))
    for i,sequence in enumerate(sequences):
        results[i,sequence] = 1.
    return results

x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)

# One Hot Encoding Manual
def to_one_hot(labels,dimension=46):
    results = np.zeros((len(labels),dimension))
    for i,label in enumerate(labels):
        results[i,label] = 1.
    return results

# One Hot Encoding Keras Built In
from keras.utils.np_utils import to_categorical

one_hot_train_labels = to_categorical(train_labels)
one_hot_test_labels = to_categorical(test_labels)

"""
Listing 3.15 Model definition
"""

from keras import models
from keras import layers

model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))

"""
Listing 3.16 Compiling the model
"""

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

"""
Listing 3.17 Setting aside a validation set
"""

x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

"""
Listing 3.18 Training the model
"""

history = model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=512,validation_data=(x_val,y_val))

"""
Listing 3.19 Plotting the training and validation loss
"""

import matplotlib.pyplot as plt

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(loss) + 1)

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.show()

"""
Listing 3.20 Plotting the training and validation accuracy
"""

plt.clf()

acc = history.history['acc']
val_acc = history.history['val_acc']

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.show()

"""
Listing 3.21 Retraining a model from scratch
"""

model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(64,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(partial_x_train,partial_y_train,epochs=9,batch_size=512,validation_data=(x_val,y_val))

results = model.evaluate(x_test,one_hot_test_labels)

results

import copy
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
float(np.sum(hits_array)) / len(test_labels)

"""
Listing 3.22 Generating predictions on new data
"""

predictions = model.predict(x_test)

predictions[0].shape
np.sum(predictions[0])
np.argmax(predictions[0])

"""
Listing 3.23 A model with an information bottleneck
"""

model = models.Sequential()
model.add(layers.Dense(64,activation='relu',input_shape=(10000,)))
model.add(layers.Dense(4,activation='relu'))
model.add(layers.Dense(46,activation='softmax'))

model.compile(optimizer='rmsprop',loss='categorical_crossentropy',metrics=['accuracy'])

model.fit(partial_x_train,partial_y_train,epochs=20,batch_size=128,validation_data=(x_val,y_val))

"""
Listing 3.24 Loading the Boston housing dataset
"""

from keras.datasets import boston_housing

(train_data,train_targets),(test_data,test_targets) = boston_housing.load_data()

train_data.shape
test_data.shape

train_targets

"""
Listing 3.25 Normalizing the data
"""

mean = train_data.mean(axis=0)
train_data -= mean
std = train_data.std(axis=0)
train_data /= std

test_data -= mean
test_data /= std

"""
Listing 3.26 Model definition
"""

from keras import models
from keras import layers

def build_model():
    model = models.Sequential()
    model.add(layers.Dense(64,activation='relu',input_shape=(train_data.shape[1],)))
    model.add(layers.Dense(64,activation='relu'))
    model.add(layers.Dense(1))
    model.compile(optimizer='rmsprop',loss='mse',metrics=['mae'])
    return model

"""
Listing 3.27 K-fold validation
"""   
import numpy as np

k = 4
num_val_samples = len(train_data) // k
num_epochs = 100
all_scores = []

for i in range(k):
    print("procesing fold # ",i)
    val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],train_data[(i + 1) * num_val_samples:]],axis=0
    )

    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],train_targets[(i + 1) * num_val_samples:]],axis=0
    )

    model = build_model()
    model.fit(partial_train_data,partial_train_targets,epochs=num_epochs,batch_size=1,verbose=0)
    val_mse,val_mae = model.evaluate(val_data,val_targets,verbose=0)
    all_scores.append(val_mae)

all_scores    
np.mean(all_scores)

"""
Listing 3.28 Saving the validation logs at each fold
"""

num_epochs = 500
all_mae_histories = []
for i in range(k):
    print("procesing fold # ",i)
    val_data = train_data[i * num_val_samples:(i + 1) * num_val_samples]
    val_targets = train_targets[i * num_val_samples: (i + 1) * num_val_samples]

    partial_train_data = np.concatenate(
        [train_data[:i * num_val_samples],train_data[(i + 1) * num_val_samples:]],axis=0
    )

    partial_train_targets = np.concatenate(
        [train_targets[:i * num_val_samples],train_targets[(i + 1) * num_val_samples:]],axis=0
    )

    model = build_model()
    model.fit(partial_train_data,partial_train_targets,validation_data=(val_data, val_targets),epochs=num_epochs,batch_size=1,verbose=0)
    mae_history = history.history['val_mean_absolute_error']
    all_mae_histories.append(mae_history)

"""
Listing 3.29 Building the history of successive mean K-fold validation scores
"""    

average_mae_history = [np.mean([x[i] for x in all_mae_histories]) for i in range(num_epochs)]

"""
Listing 3.30 Plotting validation scores
"""

import matplotlib.pyplot as plt

plt.plot(range(1,len(average_mae_history) + 1),average_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

"""
Listing 3.31 Plotting validation scores, excluding the first 10 data plints
"""

def smooth_curve(points,factor=0.9):
    smoothed_points = []
    for point in points:
        if smoothed_points:
            previous = smoothed_points[-1]
            smoothed_points.append(previous * factor + point * (1 - factor))
        else:
            smoothed_points.append(point)
    return smoothed_points

smooth_mae_history = smooth_curve(average_mae_history[10:])

plt.plot(range(1, len(smooth_mae_history) + 1), smooth_mae_history)
plt.xlabel('Epochs')
plt.ylabel('Validation MAE')
plt.show()

"""
Listing 3.32 Training the final model
"""
model = build_model()
model.fit(train_data, train_targets,
  epochs=80, batch_size=16, verbose=0)
test_mse_score, test_mae_score = model.evaluate(test_data, test_targets)
test_mae_score