import tensorflow as tf
import numpy as np

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout, Bidirectional, BatchNormalization
from keras import regularizers
from keras.callbacks import ReduceLROnPlateau
from sklearn.metrics import precision_recall_fscore_support

def lstm_model(time_steps, num_features):
    
#     model = Sequential()
#     model.add(LSTM(256, input_shape=(time_steps, num_features), return_sequences=True))  # LSTM layer with 256 units
#     model.add(Dropout(0.2))
#     model.add(LSTM(128, return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(LSTM(64, return_sequences=True))
#     model.add(Dropout(0.2))
#     model.add(Dense(64, activation="relu"))
#     model.add(Dropout(0.2))
#     model.add(Dense(3, activation="softmax"))    

    model = Sequential()
    model.add(Bidirectional(LSTM(256, return_sequences=True), input_shape=(time_steps, num_features)))
    model.add(Bidirectional(LSTM(128, return_sequences=False)))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(64, activation="relu"))
    model.add(Dropout(0.2))
    model.add(Dense(32, activation="relu"))
    model.add(Dense(3, activation="softmax")) 

    
#     opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    opt = tf.keras.optimizers.RMSprop(learning_rate=0.001) #Root Mean Squared Propagation
    model.compile(loss="sparse_categorical_crossentropy", metrics=['accuracy'], optimizer='rmsprop')
    
    # print summary of the model
    model.summary()
    
    return model