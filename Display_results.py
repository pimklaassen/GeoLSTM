import numpy as np
import matplotlib.pyplot as plt

X_cargo = np.loadtxt('X_cargo.log')
Y_cargo = np.loadtxt('Y_cargo.log')
X_fishing = np.loadtxt('X_fishing.log')
Y_fishing = np.loadtxt('Y_fishing.log')
history = np.loadtxt('history.log')

plt.plot(X_cargo, Y_cargo, 'bo')
plt.plot(X_fishing, Y_fishing, 'ro')
plt.show()

plt.plot(history)
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
