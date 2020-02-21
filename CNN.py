from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
import pandas as pd 
import matplotlib.pyplot as plt
df = pd.read_csv('elon_musk.csv')
#select input and output 
from keras.utils import to_categorical
#one-hot encode target column
y = df['class'].values
print(y)
Y = (y)
print(Y)
Y = to_categorical(df['class'])
#vcheck that target column has been converted
df = df.drop('class',1)
dataset = df.values
X = dataset[:,0:dataset.shape[1]]
#splitting the data into training and testing
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2 ) # training to testing ratio is 0.8:0.2
#Now {X_train, X_test, Y_train, Y_test} can be fed to Keras model
from keras.models import Sequential
from keras.layers import Dense
#create model
model = Sequential()
#get number of columns in training data
n_cols = X_train.shape[1]
model.add(Dense(100, activation='relu', input_shape=(n_cols,)))
model.add(Dense(100, activation='relu'))
model.add(Dense(2, activation ='softmax'))
#compile model using mse as a measure of model performance
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=['accuracy'])
# add model layers
from keras.callbacks import EarlyStopping
#set early stopping monitor so the model stops training when it won't improve anymore
early_stopping_monitor = EarlyStopping(patience=3)
#train model
history=model.fit(X,Y, validation_split=0.2, epochs=30, callbacks=[early_stopping_monitor])
Y_pred = model.predict(X_test)
# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()
Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show() 
