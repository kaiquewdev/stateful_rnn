from keras.models import Sequential
from keras.layers import LSTM,Dense
import numpy as np

data_dim = 128
timesteps = 64
nb_classes = 80
batch_size = 256

model = Sequential()
model.add(LSTM(256,return_sequences=True,stateful=True,
               batch_input_shape=(batch_size,timesteps,data_dim)))
model.add(LSTM(256,return_sequences=True,stateful=True))
model.add(LSTM(256,stateful=True))
model.add(Dense(80,activation='softmax'))
model.compile(loss='categorical_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
x_train = np.random.random((batch_size*80,timesteps,data_dim))
y_train = np.random.random((batch_size*80,nb_classes))
x_val = np.random.random((batch_size*28,timesteps,data_dim))
y_val = np.random.random((batch_size*28,nb_classes))
model.fit(x_train,y_train,
          batch_size=batch_size,nb_epoch=300,
          validation_data=(x_val,y_val))
