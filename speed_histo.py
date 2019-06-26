import psycopg2
import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import normalize

print('-> imports loaded.')


# parameters for psycopg2
params = eval(open('cred.auth', 'r').read())

# set up connection with DB
try:
    connection = psycopg2.connect(**params)
    print('-> CONNECTION ESTABLISHED.')
    print()
except:
    print('-> CONNECTION FAILED.')
    sys.exit()


# define how many ships to retrieve per class
NUM = 10

cur = connection.cursor()
cur.execute('select mmsi from dutch_fishing_dynamic group by mmsi order by count(*) desc limit {};'.format(NUM))
fishing = [_[0] for _ in cur.fetchall()]
cur.close()

# SELECT count(*) ... GROUP BY mmsi ORDER BY count(*) DESC LIMIT 20
TOP = [fishing]


def retrieve_trajectory(mmsi, database):
    """Retrieves a generator of records for the given MMSI.

    Parameter(s):
        -- mmsi: unique identifier of vessel
    """
    cur = connection.cursor()
    cur.execute('select * from {} where mmsi = {} order by ts;'.format(database, mmsi))
    print('-> records retrieved for mmsi {}.'.format(mmsi))
    return cur


def segment_continuous(cur, fh, threshold=3, to_file=False):
    """Segments a trajectory into temporally continuous parts.

    Parameter(s):
        -- threshold: a gap of this much hours cuts a trajecory into a new segment
    """
    # read out first record
    mmsi, ts, lon, lat, stat, turn, speed, course, head = cur.fetchone()
    prev = ts
    first = ts

    if to_file:
        # write first to file (ORIGINAL)
        fh.write(('{},' * 7 + '{}\n').format(ts, lon, lat, stat, turn, speed, course, head))

    # set up empty list of segments
    segments = []
    segment = [(mmsi, ts, lon, lat, speed, turn)]  # , stat, turn, speed, course, head)]

    for i, record in enumerate(cur):
        mmsi, ts, lon, lat, stat, turn, speed, course, head = record

        if to_file:
            # write all to file (ORIGINAL)
            fh.write(('{},' * 7 + '{}\n').format(ts, lon, lat, stat, turn, speed, course, head))

        difference = ts - prev
        prev = ts

        if difference > datetime.timedelta(hours=threshold):
            segments.append(segment)
            segment = [(mmsi, ts, lon, lat, speed, turn)]  # , stat, turn, speed, course, head)]
        else:
            segment.append((mmsi, ts, lon, lat, speed, turn))

    segments.append(segment)

    # logging relevant information
    print('-> time interval of {} days.'.format((prev - first).days))
    print('-> {} records read.'.format(i))
    print('-> {} segment(s) created.'.format(len(segments)))

    return segments


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


if __name__ == '__main__':
    np.seterr(divide='ignore', invalid='ignore')
    to_file = False
    MINIMUM_SEGMENT_SIZE = 100
    speed_list = []

    speed = []
    rot = []

    # iterate through vessels
    for mmsis, database in zip(TOP, ['dutch_fishing_dynamic']):

        for mmsi in mmsis:
            fh = open('segments/{}.csv'.format(mmsi), 'w') if to_file else False

            # retrieve trajectory
            cur = retrieve_trajectory(mmsi, database)

            # segment trajectory into continuous segments
            segments = segment_continuous(cur, fh, 3, to_file)

            # iterate through segments
            for segment_no, segment in enumerate(segments):
                for record in segment:
                    speed_list.append((record[2], record[3], record[4], record[1]))

            speed_list = np.array(speed_list)

            difference = np.diff(speed_list[:, :2], axis=0)
            left = np.sign(difference[:, 0])
            left[left == 0] = -1
            conv = 180 / np.pi

            time_list = speed_list[1:, 3]
            speed_list = speed_list[1:, 2]

            ROT = np.divide(difference[:, 0].astype(np.float64), difference[:, 1].astype(np.float64))[:, None]

            ROT[np.isnan(ROT)] = np.inf
            ROT = np.arctan(ROT).flatten()

            ROT = left * (difference[:, 1] * np.pi + ROT)
            ROT[ROT < 0] += 2 * np.pi
            ROT *= conv

            ROT = ROT[(speed_list < 15) & (speed_list > 0.5)]
            time_list = time_list[(speed_list < 15) & (speed_list > 0.5)]
            speed_list = speed_list[(speed_list < 15) & (speed_list > 0.5)]

            ROT = np.diff(ROT)
            # ROT = np.abs(ROT)
            speed_list = speed_list[1:]

            time_list = np.diff(time_list, axis=0)

            # function to extract seconds from datetime
            get_seconds = np.frompyfunc(lambda _: _.seconds + _.microseconds / 1000000., 1, 1)

            # calculate differences in time in minutes
            time_list = get_seconds(time_list) / 60.

            ROT = ROT[time_list != 0]
            speed_list = speed_list[time_list != 0]
            time_list = time_list[time_list != 0]

            ROT = ROT / time_list

            rot += list(ROT.flatten())
            speed += list(speed_list.flatten())

            speed_list = []

    records = np.array(list(zip(speed, rot)))
    np.random.shuffle(records)
    records = records[:120000]

    speed = records[:, 0]
    rot = records[:, 1]

    # temp = speed[(rot > 90) & (rot < 360)]
    # speed = np.concatenate((temp, speed[(rot > -360) & (rot < -90)]))
    # temp = rot[(rot > 90) & (rot < 360)]
    # rot = np.concatenate((temp, rot[(rot > -360) & (rot < -90)]))

    # speed = speed[(rot > -20) & (rot < 20)]
    rot = rot[(rot > -360) & (rot < 360)]

    sns.set()
    # sns.set_style('white')
    sns.kdeplot(rot, shade=True, label='KDE')
    # sns.jointplot(x=speed, y=rot, kind='kde', space=0)
    plt.xlabel('Rate of Turn [deg/min]')
    plt.ylabel('Normalized Density')
    plt.title('Kernel Density Estimation of rate of turn')
    plt.legend()
    plt.show()

    print('DONE.')
