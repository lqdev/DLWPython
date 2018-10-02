"""
Listing 6.1 Word-level one-hot encoding (toy example)
"""

import numpy as np

# Initial data: one entry persample (in this example, a sampe is a sentence but it could be an entire document)
samples = ['The cat sat on the mat.','The dog ate my homework.']

# Builds an index of all tokens in the data
token_index = {}
for sample in samples:
    for word in sample.split(): # Tokenizes samples via the split method. In real life punctuation and special characters would be stripped
        if word not in token_index:
            token_index[word] = len(token_index) + 1 # Assign unique index to each unique word. Note that you don't attribute index 0 to anything.

# Vectorizes the samples. You'll only consider the first max_length words in each sample.
max_length = 10

# Store results
results = np.zeros(shape=(len(samples),max_length,max(token_index.values()) + 1))

for i,sample in enumerate(samples):
    for j,word in list(enumerate(sample.split()))[:max_length]:
        index = token_index.get(word)
        results[i,j,index] = 1.

"""
Listing 6.2 Character-level one-hot encoding (toy example)
"""        

import string

samples = ['The cat sat on the mat.','The dog ate my homework.']
characters = string.printable # All printable ASCII characters
token_index = dict(zip(range(1,len(characters) + 1),characters))

max_length = 50
results = np.zeros((len(samples),max_length,max(token_index.keys()) + 1))
for i,sample in enumerate(samples):
    for j,character in enumerate(sample):
        index = token_index.get(character)
        results[i,j,index] = 1.

"""
Listing 6.3 Using Keras for word-level one-hot encoding
"""

from keras.preprocessing.text import Tokenizer

samples = ['The cat sat on the mat.','The dog ate my homework.']

tokenizer = Tokenizer(num_words=1000) # Creates a tokenizer, configured to only take into account the 1000 most common words
tokenizer.fit_on_texts(samples) # Builds the word index

sequences = tokenizer.texts_to_sequences(samples) #Turns strings into a lists of integer indices

# You could also directly get the ont-hot binary representations
# Vectorization modes other than one-hot encoding are supported by this tokenizer
one_hot_results = tokenizer.texts_to_matrix(samples,mode='binary')

word_index = tokenizer.word_index
print('Found %s unique tokens. ' % len(word_index))

"""
Listing 6.4 Word-level one-hot encoding with hashing trick (toy example)
"""

samples = ['The cat sat on the mat.','The dog ate my homework.']

# Stores the words as vectors of size 1000.
# If you have close to 1000 words (or more) you'll see many hash collisions,
# Which will decrease the accuracy of this encoding method.

dimensionality = 1000
max_length = 10
results = np.zeros((len(samples),max_length,dimensionality))
for i,sample in enumerate(samples):
    for j,word in list(enumerate(sample.split()))[:max_length]:
        index = abs(hash(word)) % dimensionality # Hashes the words into a random integer index between 0 and 1000
        results[i,j,index] = 1.

"""
Listing 6.5 Instantiating an Embedding layer
"""

from keras.layers import Embedding

# The Embedding layer takes at least two arguments:
# the number of possible tokens (here,1000: 1 + maximum word index)
# and the dimensionality of the embeddings (here,64).
embedding_layer = Embedding(1000,64)

"""
Listing 6.6 Loading the IMDB data for use in an Embedding layer
"""

from keras.datasets import imdb
from keras import preprocessing

max_features = 10000 # Number of words to consider as features
maxlen = 20 #Cuts off the text after this number of words (among the max_features most common words)

# Load the data as a list of integers
(x_train,y_train),(x_test,y_test) = imdb.load_data(num_words=max_features)

# Turns the list into a 2D integer tensor of shape (samples,maxlen)
x_train = preprocessing.sequence.pad_sequences(x_train,maxlen=maxlen) 
x_test = preprocessing.sequence.pad_sequences(x_test,maxlen=maxlen)

"""
Listing 6.7 Using an Embedding layer and classifier on the IMDB data
"""

from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()

# Specified the maximum input length to the Embedding layer so you can later flatten the embedded inputs.
# After the Embedding layer, the activations have shape (samples,maxlen,8)
model.add(Embedding(10000,8,input_length=maxlen))

# Flattens the 3D tensor of embeddings into a 2D tensor of shape (samples,maxlen * 8)
model.add(Flatten())

model.add(Dense(1,activation='sigmoid')) # Adds the classifier on top
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
model.summary()

history = model.fit(x_train,y_train,
    epochs=10,
    batch_size=32,
    validation_split=0.2)


"""
Listing 6.8 Processing the labels of the raw IMDB data (Downloaded from http://mng.bz/0tIo)
"""

import os

imdb_dir = '/home/lqdev/Downloads/aclImdb'
train_dir = os.path.join(imdb_dir,'train')

labels = []
texts = []

for label_type in ['neg','pos']:
    dir_name = os.path.join(train_dir,label_type)
    for fname in os.listdir(dir_name):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name,fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)


"""
Listing 6.9 Tokenizing the text of the raw IMDB data
"""

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import numpy as np

maxlen = 100 # Cuts off reviews after 100 words
training_samples = 200 # Trains on 200 samples
validation_samples = 10000 # Validates on 10000 samples
max_words = 10000 # Consider onyl the top 10000 words in the dataset

tokenizer = Tokenizer(num_words=max_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

word_index = tokenizer.word_index
print("Found %s unique tokens" % len(word_index))

data = pad_sequences(sequences,maxlen=maxlen)

labels = np.asarray(labels)
print('Shape of data tensor: ', data.shape)
print('Shape of label tensor: ', labels.shape)

# Splits the data into a training set and a validation set,
# but first shuffles the data, because you're starting with data
# in which samples are ordered (all negative first, then all positive)
indices = np.arange(data.shape[0]) 
np.random.shuffle(indices)
data = data[indices]
labels = labels[indices]

x_train = data[:training_samples]
y_train = labels[:training_samples]
x_val = data[training_samples:training_samples + validation_samples]
y_val = labels[training_samples:training_samples + validation_samples]

"""
Listing 6.10 Parsing the GloVe word-embeddings file
"""

glove_dir = '/home/lqdev/Downloads/glove.6B'

embeddings_index = {}

f = open(os.path.join(glove_dir,'glove.6B.100d.txt'))
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:],dtype='float32')
    embeddings_index[word] = coefs
f.close()

print('Found %s word vectors.' % len(embeddings_index))

"""
Listing 6.11 Preparing the GloVe word-embeddings matrix
"""

embedding_dim = 100

embedding_matrix = np.zeros((max_words,embedding_dim))

for word,i in word_index.items():
    if i < max_words:
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector # Words not found in the embedding index will be all zeros

"""
Listing 6.12 Model definition
"""

from keras.models import Sequential
from keras.layers import Embedding,Flatten,Dense

model = Sequential()
model.add(Embedding(max_words,embedding_dim,input_length=maxlen))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()

"""
Listing 6.13 Loading pretrained word embeddings into the Embedding layer
"""

model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable=False

"""
Listing 6.14 Training and evaluation
"""

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

history = model.fit(x_train,y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_val,y_val))

model.save_weights('pre_trained_glove_model.h5')

"""
Listing 6.15 Plotting the results
"""

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

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

"""
Listing 6.16 Training the same model without pretrained word embeddings
"""

from keras.models import Sequential
from keras.layers import Embedding,Flatten,Dense

model = Sequential()
model.add(Embedding(max_words,embedding_dim,input_length=maxlen))
model.add(Flatten())
model.add(Dense(32,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.summary()

model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])

history = model.fit(x_train,y_train,
    epochs=10,
    batch_size=32,
    validation_data=(x_val,y_val))

"""
Listing 6.17 Tokenizing the data of the test set
"""

test_dir = os.path.join(imdb_dir,'test')

labels = []
texts = []

for label_type in ['neg', 'pos']:
    dir_name = os.path.join(test_dir, label_type)
    for fname in sorted(os.listdir(dir_name)):
        if fname[-4:] == '.txt':
            f = open(os.path.join(dir_name, fname))
            texts.append(f.read())
            f.close()
            if label_type == 'neg':
                labels.append(0)
            else:
                labels.append(1)

sequences = tokenizer.texts_to_sequences(texts)
x_test = pad_sequences(sequences, maxlen=maxlen)
y_test = np.asarray(labels)

"""
Listing 6.18 Evaluating the model on the test set
"""

model.load_weights('pre_trained_glove_model.h5')
model.evaluate(x_test, y_test)

"""
Listing 6.19 Pseudocode RNN
"""

# state_t = 0 # The state at t
# for input_t in input_sequence: # Iterates over sequence elements
#     output_t = f(input_t,state_t)
#     state_t = output_t

"""
Listing 6.20 More detailed pseudocode for the RNN
"""

# state_t = 0
# for input_t in input_sequence:
#     output_t = activation(dot(W,input_t) + dot(U,state_t) + b)
#     state_t = output_t

"""
Listing 6.21 Numpy implementation of a simple RNN
"""

import numpy as np

timesteps = 1000 # Number of steps in the input sequence
input_features = 32 # Dimensionality of the input feature space
output_features = 64 # Dimensionality of the output feature space

inputs = np.random.random((timesteps,input_features)) # Input data: random noise for the sake of the example

state_t = np.zeros((output_features,)) # Initiali state: an all-zero vector

# Create random weight matrices
W = np.random.random((output_features,input_features))
U = np.random.random((output_features,output_features))
b = np.random.random((output_features,))

successive_outputs = []
for input_t in inputs: # input_t is a vector of shape (input_featurs,)
    output_t = np.tanh(np.dot(W,input_t) + np.dot(U,state_t) + b) # Combines the input with the current state (the previous output) to obtain the current output

    successive_outputs.append(output_t) # Stores this output in a list

    state_t = output_t # Updates the state of the network for the next timestep

final_output_sequence = np.concatenate(successive_outputs,axis=0) # The final output is a 2D tensor of shape (timesteps,output_features)

"""
Listing 6.22 Preparing the IMDB data
"""

from keras.datasets import imdb
from keras.preprocessing import sequence

max_features = 10000 # Number of words to consider as features
maxlen = 500 # Cuts off texts after this many words (among the max_features of most common words)
batch_size = 32

print('Loading data...')
(input_train,y_train),(input_test,y_test) = imdb.load_data(num_words=max_features)
print(len(input_train),'train sequences')
print(len(input_test),'test sequences')

print('Pad sequences (samples x time)')
input_train = sequence.pad_sequences(input_train,maxlen=maxlen)
input_test = sequence.pad_sequences(input_test,maxlen=maxlen)
print('input_train shape:',input_train.shape)
print('input test shape:',input_test.shape)

"""
Listing 6.23 Training the model with Embedding and SimpleRNN layers
"""

from keras.models import Sequential
from keras.layers import Dense, Embedding,SimpleRNN

model = Sequential()
model.add(Embedding(max_features,32))
model.add(SimpleRNN(32))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history = model.fit(input_train,y_train,
                    epochs=10,
                    batch_size=128,
                    validation_split=0.2)

"""
Listing 6.24 Plotting results
"""

import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc) + 1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validationa accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validationa loss')
plt.legend()

plt.show()

"""
Listing 6.25 Pseudocode details of the LSTM architecture (1/2)
"""

# output_t = activation(dot(state_t,Uo) = dot(input_t,Wo) + dot(C_t,Vo) + bo)

# i_t = activation(dot(state_t,Ui) + dot(input_t,Wi) + bi)
# f_t = activation(dot(state_t,Uf) + dot(input_t,Wf) + bf)
# k_t = activation(dot(state_t,Uk) + dot(input_t,Wk) + bk)

"""
Listing 6.26 Pseudocode details of the LSTM architecture (2/2)
"""

# c_t+1 = i_t * k+t + c_t * f_t

"""
Listing 6.27 Using the LSTM layer in Keras
"""

from keras.layers import LSTM

model = Sequential()
model.add(Embedding(max_features,32))
model.add(LSTM(32))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy',metrics=['acc'])
history = model.fit(input_train,y_train,epochs=10,batch_size=128,validation_split=0.2)

# Added plotting logic 
import matplotlib.pyplot as plt

acc = history.history['acc']
val_acc = history.history['val_acc']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs = range(1,len(acc) + 1)

plt.plot(epochs,acc,'bo',label='Training acc')
plt.plot(epochs,val_acc,'b',label='Validation acc')
plt.title('Training and validationa accuracy')
plt.legend()

plt.figure()

plt.plot(epochs,loss,'bo',label='Training loss')
plt.plot(epochs,val_loss,'b',label='Validation loss')
plt.title('Training and validationa loss')
plt.legend()

plt.show()

"""
Listing 6.28 Inspecfing the data of the Jena weather dataset
"""

import os

data_dir = "/home/lqdev/Downloads/jena_climate"
fname = os.path.join(data_dir,'jena_climate_2009_2016.csv')

f = open(fname)
data = f.read()
f.close()

lines = data.split('\n')
header = lines[0].split(',')
lines = lines[1:]

print(header)
print(len(lines))

"""
Listing 6.29 Parsing the data
"""

import numpy as np

float_data = np.zeros((len(lines),len(header)-1))
for i,line in enumerate(lines):
    values = [float(x) for x in line.split(',')[1:]]
    float_data[i,:] = values

"""
6.30 Plotting the temperature timeseries
"""

import matplotlib.pyplot as plt

temp = float_data[:,1]
plt.plot(range(len(temp)),temp)
plt.show()

"""
Listing 6.31 PLotting the first 10 days of the temperature timeseries
"""

plt.plot(range(1440),temp[:1440])
plt.show()

"""
Listing 6.32 Normalizing the data
"""

mean = float_data[:200000].mean(axis=0)
float_data -= mean
std = float_data[:200000].std(axis=0)
float_data /= std

"""
Listing 6.33 Generator yielding timeseries samples and their targets
"""

def generator(data,lookback,delay,min_index,max_index,shuffle=False,batch_size=128,step=6):
    if max_index is None:
        max_index = len(data) - delay - 1
    i = min_index + lookback
    while 1:
        if shuffle:
            rows = np.random.randint(min_index + lookback,max_index,size=batch_size)
        else:
            if i + batch_size >= max_index:
                i = min_index + lookback
            rows = np.arange(i,min(i + batch_size,max_index))
            i += len(rows)
        
        samples = np.zeros((len(rows),lookback // step,data.shape[-1]))
        targets = np.zeros((len(rows),))
        for j,row in enumerate(rows):
            indices = range(rows[j] - lookback,rows[j],step)
            samples[j] = data[indices]
            targets[j] = data[rows[j] + delay][1]
        yield samples,targets

"""
Listing 6.34 Preparing the training, validation and test generators
"""

lookback = 1440
step = 6
delay = 144
batch_size = 128

train_gen = generator(float_data,
                            lookback=lookback,
                            delay=delay,
                            min_index=0,
                            max_index=200000,
                            shuffle=True,
                            step=step,
                            batch_size=batch_size)

val_gen = generator(float_data,
                            lookback=lookback,
                            delay=delay,
                            min_index=200001,
                            max_index=300000,
                            shuffle=True,
                            step=step,
                            batch_size=batch_size)

test_gen = generator(float_data,
                            lookback=lookback,
                            delay=delay,
                            min_index=300001,
                            max_index=None,
                            shuffle=True,
                            step=step,
                            batch_size=batch_size)

val_steps = (300000 - 200001 - lookback) # How many steps to draw from val_gen in order to see the entire validation set
test_steps = (len(float_data) - 300001 - lookback) # How many steps to draw from test_gen in order to see the entire test set

"""
Listing 6.35 Computing the common-sense baseline MAE
"""

def evaluate_naive_method():
    batch_maes = []
    for step in range(val_steps):
        print("On step ",step)
        samples,targets = next(val_gen)
        print(step)
        preds = samples[:,-1,1]
        mae = np.mean(np.abs(preds-targets))
        batch_maes.append(mae)
    print(np.mean(batch_maes))

evaluate_naive_method()

"""
Listing 6.36 Converting the MAE back to Celcious error
"""

celsius_mae = 0.29 * std[1]

"""
Listing 6.37 Training and evaluating a densely connected model
"""

from keras.models import Sequential
from keras import layers
from keras.optimizers import RMSprop

model = Sequential()
model.add(layers.Flatten(input_shape=(lookback // step, float_data.shape[-1])))
model.add(layers.Dense(32, activation='relu'))
model.add(layers.Dense(1))
model.compile(optimizer=RMSprop(),loss='mae')

history = model.fit_generator(train_gen,
                              steps_per_epoch=500,
                              epochs=20,
                              validation_data=val_gen,
                              validation_steps=val_steps)
