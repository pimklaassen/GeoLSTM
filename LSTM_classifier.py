import numpy as np
import sys
import os

from keras.models import Model
from keras.layers import LSTM, Input, Masking, Dense, BatchNormalization
from keras.activations import hard_sigmoid
from keras.backend.tensorflow_backend import set_session
from tensorflow import Session, ConfigProto
from keras.optimizers import SGD
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
normalization_values_max = np.array([0., 0., 0., 0., 0., 0.])

for fname in os.listdir(datadir):
    sample = np.load(datadir + fname)
    max_values = np.max(sample, axis=0)
    normalization_values_max[normalization_values_max < max_values] = max_values[normalization_values_max < max_values]

    length = len(sample)
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
lstm_1 = LSTM(512)(mask)
norm = BatchNormalization()(lstm_1)
output = Dense(1, activation='sigmoid')(norm)

LSTM_model = Model(inputs, output)

# optimizers
sgd = SGD(lr=0.01, clipvalue=0.25, momentum=0.0, decay=0.0, nesterov=True)

LSTM_model.compile(optimizer='adadelta', loss='binary_crossentropy', metrics=['accuracy'])

print(LSTM_model.summary())


def batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i + n]


for epoch in range(100):

    print('epoch: {}\n'.format(epoch))
    print('#')

    for batch in batches(train, 32):
        train_y = batch[:, 1, None]
        train_x = []

        for info in batch:
            fname = datadir + '_'.join(str(_) for _ in info[1:]) + '.npy'
            sample = np.load(fname) / normalization_values_max
            train_x.append(sample)

        train_x = pad_sequences(train_x)

        LSTM_model.train_on_batch(train_x, train_y)

        print('|')

    print('#\n')

    train_val_x = []
    train_val_batch = train[np.random.choice(len(train), size=32)]
    train_val_y = train_val_batch[:, 1, None]

    for info in train_val_batch:
        fname = datadir + '_'.join(str(_) for _ in info[1:]) + '.npy'
        sample = np.load(fname) / normalization_values_max
        train_val_x.append(sample)

    train_val_x = pad_sequences(train_val_x)
    loss, acc = LSTM_model.test_on_batch(train_val_x, train_val_y)
    pred = LSTM_model.predict(train_val_x, train_val_y)

    res = np.hstack((pred, train_val_y))

    print('{}\n\ntraining loss: {} accuracy: {}'.format(res, loss, acc))

    test_val_x = []
    test_val_batch = test[np.random.choice(len(train), size=16)]
    test_val_y = test_val_batch[:, 1, None]

    for info in test_val_batch:
        fname = datadir + '_'.join(str(_) for _ in info[1:]) + '.npy'
        sample = np.load(fname) / normalization_values_max
        test_val_batch.append(sample)

    test_val_batch = pad_sequences(test_val_batch)
    loss, acc = LSTM_model.test_on_batch(test_val_batch, test_val_y)

    print('test loss: {} accuracy: {}\n'.format(loss, acc))
