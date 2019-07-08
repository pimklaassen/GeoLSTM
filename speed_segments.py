import sys
import os
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


def load_raw(directory):
    """Yields raw trajectories

    Parameter(s):
        -- directory: directory name where files live
    """
    for filename in os.listdir(directory):
        yield np.load(directory + filename)


if __name__ == '__main__':
    np.seterr(divide='ignore', invalid='ignore')
    to_file = False
    MINIMUM_SEGMENT_SIZE = 100

    for trajectory in load_raw('Data/Raw/'):
        trajectory['speed'][trajectory['speed'] > 14.] = 14.
        sns.lineplot(x=trajectory['ts'], y=trajectory['speed'])
        plt.show()
