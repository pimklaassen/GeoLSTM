import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

history = np.load('history.npy')
sns.set()

plt.plot(history[:, 0], label='L-train')
plt.plot(history[:, 1], label='A-train')
plt.plot(history[:, 2], label='L-test')
plt.plot(history[:, 3], label='A-test')

plt.legend(loc='upper right')
plt.show()
