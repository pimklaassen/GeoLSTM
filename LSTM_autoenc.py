import matplotlib.pyplot as plt
import numpy as np
import sys
import os

from keras.models import Model
from keras.layers import LSTM, Input, RepeatVector, Masking
from keras.backend.tensorflow_backend import set_session
from tensorflow import Session, ConfigProto
from sklearn.manifold import TSNE

set_session(Session(config=ConfigProto(device_count={'GPU': 0})))

# fishing_names = os.listdir('dutch_cargo_dynamic')
# sequences = [np.load('dutch_cargo_dynamic/{}'.format(filename)) for filename in fishing_names]
# fishing_names = [_.split('.')[0] for _ in fishing_names]

# first_num = len(sequences) + 1

# cargo_names = os.listdir('dutch_fishing_dynamic')
# next_sequences = [np.load('dutch_fishing_dynamic/{}'.format(filename)) for filename in cargo_names]
# cargo_names = [_.split('.')[0] for _ in cargo_names]

# sequences += next_sequences
# del next_sequences


def pad_sequences(sequences):
    # set dimensions
    samples = len(sequences)
    timesteps = (max(len(_) for _ in sequences))
    features = len(sequences[0][0])

    # create template
    template = np.zeros((samples, timesteps, features))

    # fill in template
    for i, sequence in enumerate(sequences):
        template[i, :len(sequence), :] = sequence

    return template


# dataset = pad_sequences(sequences)
# dataset = dataset[:1, :20, :1]


dataset = np.array([n / 10 for n in range(1, 11)])[None, :, None]

print(dataset)


samples, timesteps, features = dataset.shape
latent_dim = 16

inputs = Input(shape=(timesteps, features))
mask = Masking(mask_value=0.0)(inputs)
encoded = LSTM(latent_dim)(mask)

decoded = RepeatVector(timesteps)(encoded)
decoded = LSTM(features, return_sequences=True)(decoded)

sequence_autoencoder = Model(inputs, decoded)
encoder = Model(inputs, encoded)

sequence_autoencoder.compile(optimizer='adam', loss='mse')

# fit model
history = sequence_autoencoder.fit(dataset, dataset, epochs=500, verbose=0, batch_size=1)

# demonstrate recreation
yhat = sequence_autoencoder.predict(dataset, verbose=0)
repr_vector = encoder.predict(dataset, verbose=0)
print(yhat)

# summarize history for loss
plt.plot(history.history['loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
sys.exit()

X_embedded = TSNE(n_components=2).fit_transform(repr_vector)

X_fishing = X_embedded[first_num:, 0]
Y_fishing = X_embedded[first_num:, 1]
X_cargo = X_embedded[:first_num, 0]
Y_cargo = X_embedded[:first_num, 1]

for name, x, y in zip(fishing_names, X_fishing, Y_fishing):
    plt.scatter(x, y, marker='o', color='red')
    plt.text(x, y, name, fontsize=9)

for name, x, y in zip(cargo_names, X_cargo, Y_cargo):
    plt.scatter(x, y, marker='o', color='blue')
    plt.text(x, y, name, fontsize=9)

plt.show()


# X_embedded = TSNE(n_components=2).fit_transform(yhat[0])

# X = X_embedded[:, 0]
# Y = X_embedded[:, 1]

# plt.plot(X, Y, 'r.')
# plt.show()
