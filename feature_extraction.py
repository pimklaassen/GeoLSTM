import sys
import os
import numpy as np
from sklearn.preprocessing import normalize


def load_raw(directory):
    """Yields raw trajectories

    Parameter(s):
        -- directory: directory name where files live
    """
    for filename in os.listdir(directory):
        yield np.load(directory + filename), filename


def extract_window_records(segment, window_size=600, window_offset=300, min_size=50):
    """Extracts time windows with offset from a temporally continuous segment.

    Parameter(s):
        -- window_size: size of temporal window in seconds
        -- window_offset: size of temporal offset in seconds (TODO)
        -- min_size: discard segments with less points than min_size
    """
    assert window_offset <= window_size

    # if the segment is too small, discard it
    if len(segment) < min_size:
        return

    current = segment[0][1]
    last = segment[-1][1]
    windows = []

    # each window contains the first time of the next, and the last time of its own
    while current <= last:
        windows.append((current + datetime.timedelta(seconds=window_offset), current + datetime.timedelta(seconds=window_size)))
        current += datetime.timedelta(seconds=window_offset)

    segment = segment[::-1]
    current_window_records = []
    next_window_records = []
    tby = segment.pop()
    current = tby[1]

    # roll through all windows
    for window in windows:

        # while the records are still within this window, add them
        while current < window[1]:
            current_window_records.append(tby)

            # if they are also in the next window, add them to the next
            if current >= window[0]:
                next_window_records.append(tby)

            if segment:
                tby = segment.pop()
                current = tby[1]
            else:
                break

        # current record is not in this window anymore, yield all previous records
        if len(current_window_records) > 1:
            yield current_window_records

        # next window becomes current window
        current_window_records = next_window_records
        next_window_records = []


def extract_features(window_records):
    """Extracts representative features from a window of records.

    Parameter(s):
        -- window_records: the records from which the features have to be extracted
    """
    difference = np.diff(np.array(window_records)[:, 1:], axis=0)

    # calculate distance between records
    location_difference = np.sqrt(np.sum(difference[:, 1:3]**2, axis=1, keepdims=True).astype(float))

    # function to extract seconds from datetime
    get_seconds = np.frompyfunc(lambda _: _.seconds + _.microseconds / 1000000., 1, 1)

    # calculate differences in time
    time_difference = get_seconds(difference[:, 0])

    try:
        # calculate average speed per record
        average_speed = location_difference / time_difference[:, None]
    except:
        return False, False

    # get change of speed per record
    speed_change = difference[:, 3][:, None]

    # get the ROT per record
    ROT = np.divide(difference[:, 1].astype(np.float64), difference[:, 2].astype(np.float64))[:, None]
    ROT[np.isnan(ROT)] = np.inf
    ROT = np.arctan(ROT)

    # calculate the change of ROT (first row is 0)
    ROT_change = np.insert(np.diff(ROT, axis=0), 0, ROT[0], 0)

    # stack all features
    # column 1: average speed between records
    # column 2: change of speed per record
    # column 3: change of ROT per record, first record 0
    features = np.hstack((average_speed, speed_change, ROT_change))

    # extract meaningful statistics
    a = features.mean(axis=0)
    b = features.max(axis=0)
    c = np.quantile(features, 0.75, axis=0)
    d = np.quantile(features, 0.5, axis=0)
    e = np.quantile(features, 0.25, axis=0)
    f = features.min(axis=0)

    # concatenate and normalize
    features = np.concatenate((a, b, c, d, e, f))
    features = normalize([features]).ravel()
    return True, features


def speed_only(content):
    speed = content['speed']
    speed[speed > 25.] = 25
    speed[speed < 0.] = 0.

    f1 = min(speed)
    f2 = np.quantile(speed, 0.25)
    f3 = np.quantile(speed, 0.5)
    f4 = np.mean(speed)
    f5 = np.quantile(speed, 0.75)
    f6 = max(speed)

    return normalize([[f1, f2, f3, f4, f5, f6]]).ravel()


def rot_only(content):
    # TODO: finish function
    diflon = np.diff(content['lon'])
    diflat = np.diff(content['lat'])
    diflat[diflat == 0] = 1e-10

    sign_x = np.sign(diflon)
    sign_x[sign_x == 0] = -1.

    angle = sign_x * (diflat * np.pi + np.arctan(diflon / diflat))
    angle[angle < 0] += 2 * np.pi
    print(angle)

    return content


def create_features(trajectory, window=600, offset=300, function=speed_only):
    begin = trajectory[0][0]
    end = trajectory[-1][0]

    window = np.timedelta64(window, 's')
    offset = np.timedelta64(offset, 's')

    while begin < end:
        content = trajectory[(trajectory['ts'] >= begin) &
                             (trajectory['ts'] < begin + window)]

        if len(content) < 2:
            begin += offset
            continue

        yield function(content)

        begin += offset


if __name__ == '__main__':
    np.seterr(divide='ignore', invalid='ignore')
    to_file = False
    MINIMUM_SEGMENT_SIZE = 100

    for trajectory, fname in load_raw('Data/Raw/'):
        features = np.array([f for f in create_features(trajectory)])
        np.save('Data/Features/{}'.format(fname), features)
        print('-> {} features pickled'.format(fname))
