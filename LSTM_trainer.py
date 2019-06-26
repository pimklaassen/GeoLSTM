import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import seaborn as sns

from keras.models import Model
from keras.layers import LSTM, Input, RepeatVector, Masking, Dropout, Dense
from keras.backend.tensorflow_backend import set_session
from keras.utils import to_categorical
from tensorflow import Session, ConfigProto
from random import shuffle
from sklearn.metrics import confusion_matrix
from keras import optimizers

set_session(Session(config=ConfigProto(device_count={'GPU': 0})))

labels = []

with open('labeled_data/labels.txt', 'r') as fh:
    for line in fh.readlines():
        label = line.split(': ')[1].strip()

        if label == 'NULL':
            labels.append('NULL')
        else:
            label = int(label) - 1

        labels.append(label)

samples = []

for i in range(len(labels) - 1):
    sample = np.load('labeled_data/features_{}.npy'.format(i))
    if labels[i] == 'NULL':
        continue
    if len(sample) == 0:
        labels[i] = 'NULL'
        continue
    samples.append(sample)

labels = filter(lambda a: a != 'NULL', labels)

temp = list(zip(samples, labels))
shuffle(temp)

samples, labels = zip(*temp)

labels = np.array(labels)
labels = to_categorical(labels)


def pad_sequences(sequences):
    # set dimensions
    no_sequences = len(sequences)
    timesteps = (max(len(_) for _ in sequences))
    features = len(sequences[0][0])

    # create template
    template = np.zeros((no_sequences, timesteps, features))

    # fill in template
    for i, sequence in enumerate(sequences):
        template[i, :len(sequence), :] = sequence

    return template


samples = pad_sequences(samples)

print(samples.shape)
print(labels)
print(samples)

no_samples, timesteps, features = samples.shape

validate_x = samples[:200]
validate_y = labels[:200]

train_x = samples[200:]
train_y = labels[200:]

inputs = Input(shape=(timesteps, features))
mask = Masking(mask_value=0.0)(inputs)
output = LSTM(3, activation='sigmoid')(mask)

LSTM_model = Model(inputs, output)
optimizer = optimizers.Adam()
LSTM_model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])

# fit model
history = LSTM_model.fit(train_x, train_y, epochs=1000, batch_size=256, validation_data=(validate_x, validate_y), verbose=2)

prediction = LSTM_model.predict(validate_x)

prediction = np.argmax(prediction, axis=1)
validate_y = np.argmax(validate_y, axis=1)

sns.heatmap(confusion_matrix(validate_y, prediction), annot=True, fmt='.5g')
plt.show()

# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['acc'])
plt.plot(history.history['val_loss'])
plt.plot(history.history['val_acc'])
plt.title('model loss & accuracy')
plt.ylabel('loss & accuracy')
plt.xlabel('epoch')
plt.legend(['training_loss', 'training_loss_accuracy', 'validation_loss', 'validation_accuracy'], loc='upper left')
plt.show()
