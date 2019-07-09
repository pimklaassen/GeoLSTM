import numpy as np
import sys
import os

from keras.models import Model
from keras.layers import LSTM, Input, Masking, Dense
from keras.activations import hard_sigmoid
from keras.backend.tensorflow_backend import set_session
from tensorflow import Session, ConfigProto
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

set_session(Session(config=ConfigProto(device_count={'GPU': 0})))


bin_size = [1000, 6000]


def pad_sequences(sequences, timesteps=6000):
    # set dimensions
    no_sequences = len(sequences)
    features = len(sequences[0][0])

    # create template
    template = np.ones((no_sequences, timesteps, features)) * -1.

    # fill in template
    for i, sequence in enumerate(sequences):
        template[i, :len(sequence), :] = sequence

    return template


sample_info = []
datadir = 'Data/Features/'

for fname in os.listdir(datadir):
    length = len(np.load(datadir + fname))
    class_no, mmsi, seg_no = fname.split('.')[0].split('_')
    sample_info.append([length, int(class_no), int(mmsi), int(seg_no)])

sample_info = np.array(sample_info)

# put in a bin
sample_info = sample_info[(sample_info[:, 0] > 1000) & (sample_info[:, 0] < 6000)]

# a priori
sample_info = sample_info[10:]
np.random.shuffle(sample_info)
break_inx = int(len(sample_info) * 0.8)

train = sample_info[:break_inx]
test = sample_info[break_inx:]

features = 6

inputs = Input(shape=(None, features))
mask = Masking(mask_value=-1.)(inputs)
lstm_1 = LSTM(200, activation='relu')(mask)
output = Dense(1, activation='sigmoid')(lstm_1)

LSTM_model = Model(inputs, output)
LSTM_model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

print(LSTM_model.summary())

for epoch in range(100):
    train_x_info = train[np.random.choice(range(len(train)), size=16, replace=False)]
    train_y = train_x_info[:, 1, None]
    train_x = []

    for info in train_x_info:
        fname = datadir + '_'.join(str(_) for _ in info[1:]) + '.npy'
        train_x.append(np.load(fname))

    train_x = pad_sequences(train_x)

    loss, acc = LSTM_model.train_on_batch(train_x, train_y)

    pred = LSTM_model.predict(train_x)
    res = np.hstack((pred, train_y))

    print('epoch: {}\n\n{}\n\nloss: {} accuracy: {}\n'.format(epoch, res, loss, acc))
