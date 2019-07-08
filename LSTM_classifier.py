import matplotlib.pyplot as plt
import numpy as np
import sys
import os

from keras.models import Model
from keras.layers import LSTM, Input
from keras.activations import hard_sigmoid
from keras.backend.tensorflow_backend import set_session
from tensorflow import Session, ConfigProto
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

set_session(Session(config=ConfigProto(device_count={'GPU': 0})))

sample_info = []
datadir = 'Data/Features/'

for fname in os.listdir(datadir):
    length = len(np.load(datadir + fname))
    class_no, mmsi, seg_no = fname.split('.')[0].split('_')
    sample_info.append([length, int(class_no), int(mmsi), int(seg_no)])

sample_info = np.array(sample_info)
# np.random.shuffle(sample_info)
break_inx = int(len(sample_info) * 0.8)

train = sample_info[:break_inx]
test = sample_info[break_inx:]

features = 6

inputs = Input(shape=(None, features))
lstm_1 = LSTM(32, activation=hard_sigmoid, return_sequences=True)(inputs)
lstm_2 = LSTM(32, activation=hard_sigmoid, return_sequences=True)(lstm_1)
output = LSTM(1, activation=hard_sigmoid)(lstm_2)

LSTM_model = Model(inputs, output)
LSTM_model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

print(LSTM_model.summary())

for epoch in range(100):
    train_x_info = train[27]  # np.random.randint(len(train))]
    fname = datadir + '_'.join(str(_) for _ in train_x_info[1:]) + '.npy'
    train_x = np.expand_dims(np.load(fname), axis=0)
    train_y = np.array([train_x_info[1]])

    LSTM_model.train_on_batch(train_x, train_y)

    pred = float(LSTM_model.predict(train_x))
    true = float(train_y)

    print('{}/{}'.format(pred, true))
