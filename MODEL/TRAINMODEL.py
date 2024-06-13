from keras.layers import Attention, Reshape, Input, Dense, Dropout, Flatten, LSTM, Flatten, GRU, Conv1D, GlobalMaxPooling2D, Layer, Concatenate, GlobalAveragePooling2D, GlobalAveragePooling1D, BatchNormalization
from keras.models import Sequential
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
import tensorflow as tf
import argparse
import numpy as np

input_shape = (300, 4)

NF = np.loadtxt('NF.csv', delimiter = ",")

seq_len = 300
test_ratio = 0.3
filenames = ['1PSC','2PSC','VREC','OP','REVD']

from sklearn.preprocessing import MinMaxScaler
def divide_dataset(data):
    data = np.asarray(data)
    X_train = data[0:int(len(data)*(1-test_ratio)),[0,3,6,7]]
    X_test = data[0:int(len(data)*test_ratio),[0,3,6,7]]
    Scaler = MinMaxScaler()
    X_train = Scaler.fit_transform(X_train)
    X_test = Scaler.transform(X_train)
    X_train = X_train.reshape(-1,300,4)
    X_test = X_test.reshape(-1,300,4)
    y_train = data[:int(len(data)/seq_len*(1-test_ratio)),9]
    y_test = data[:int(len(data)/seq_len*test_ratio),9]
    return X_train, X_test, y_train, y_test
X_train, X_test, y_train, y_test = divide_dataset(NF)

for filename in filenames:
  origin = np.loadtxt('%s.csv'%(filename), delimiter = ",")
  X_train1, X_test1, y_train1, y_test1 = divide_dataset(origin)
  X_train = np.concatenate([X_train, X_train1])
  X_test = np.concatenate([X_test,X_test1])
  y_train = np.concatenate([y_train, y_train1])
  y_test = np.concatenate([y_test, y_test1])

from sklearn.preprocessing import OneHotEncoder
encoder = OneHotEncoder()
y_train = y_train.reshape(-1, 1)
y_train = encoder.fit_transform(y_train).toarray()
y_test = y_test.reshape(-1, 1)
y_test = encoder.transform(y_test).toarray()

np.save('X_test.npy', X_test)
np.save('y_test.npy', y_test)

def lstm_model(input_shape, a):
    inputs = Input(shape=input_shape)
    x = LSTM(300, return_sequences=True)(inputs)
    x = Dense(30)(x)  # y_train의 shape에 맞게 출력 shape 변경
    if a == 'y':
        x = Attention()([x, x, x])  # Attention 레이어 사용법 점검 필요
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(6, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

def gru_model(input_shape,a):
    inputs = Input(shape=input_shape)
    x = GRU(300, return_sequences=True)(inputs)
    x = Dense(30)(x)
    if a == 'y':
      x = Attention()([x, x, x])
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(6, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

def cnn_model(input_shape,a):
    inputs = Input(shape=input_shape)
    x = Conv1D(128, kernel_size=3, padding='same', activation='relu')(inputs)
    x = Dense(64)(x)
    if a == 'y':
      x = Attention()([x, x, x])
    x = GlobalAveragePooling1D()(x)
    outputs = Dense(6, activation='softmax')(x)
    model = Model(inputs, outputs)
    return model

def combined_model(input_shape, a, b):
    lstm = lstm_model(input_shape, a)
    gru = gru_model(input_shape, a)
    cnn = cnn_model(input_shape, a)

    combined_input = Input(shape=input_shape)

    lstm_output = lstm(combined_input)
    gru_output = gru(combined_input)
    cnn_output = cnn(combined_input)

    combined_output = Concatenate()([lstm_output, gru_output, cnn_output])
    combined_output = Dense(128, activation='relu')(combined_output)
    combined_output = Dropout(0.5)(combined_output)
    
    if b == 'y':
        combined_output = Attention()([combined_output,combined_output,combined_output])

    combined_output = BatchNormalization()(combined_output)
    final_output = Dense(6, activation='softmax')(combined_output)  # Ensure this matches y_train shape  
    model = Model(combined_input, final_output)
    return model

def save_train(model, name):
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    # Ensure that the y_train shape matches the model's output shape
    model.fit(X_train, y_train, epochs=100, batch_size=32)
    model.save('%s'%(name))

save_train(lstm_model(input_shape,'y'), 'lstmA')
save_train(lstm_model(input_shape,'n'), 'lstmB')
save_train(gru_model(input_shape,'y'), 'gruA')
save_train(gru_model(input_shape,'n'), 'gruB')
save_train(cnn_model(input_shape,'y'), 'cnnA')
save_train(cnn_model(input_shape,'n'), 'cnnB')
save_train(combined_model(input_shape, 'n', 'n'), 'comBB')
save_train(combined_model(input_shape, 'y', 'n'), 'comAB')
save_train(combined_model(input_shape, 'n', 'y'), 'comBA')
save_train(combined_model(input_shape, 'y', 'y'), 'comAA')
