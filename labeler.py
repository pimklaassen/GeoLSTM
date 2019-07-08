import psycopg2
import datetime
import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import TextBox

from sklearn.preprocessing import normalize

print('-> imports loaded.')


# parameters for psycopg2
params = eval(open('../pwd/cred.auth', 'r').read())

# set up connection with DB
try:
    connection = psycopg2.connect(**params)
    print('-> CONNECTION ESTABLISHED.')
    print()
except:
    print('-> CONNECTION FAILED.')
    sys.exit()


# define how many ships to retrieve per class
NUM = 50

cur = connection.cursor()
cur.execute('select mmsi from dutch_fishing_dynamic group by mmsi order by count(*) desc limit {};'.format(NUM))
fishing = [_[0] for _ in cur.fetchall()]
cur.close()

cur = connection.cursor()
cur.execute('select mmsi from dutch_cargo_dynamic group by mmsi order by count(*) desc limit {};'.format(NUM))
cargo = [_[0] for _ in cur.fetchall()]
cur.close()

# SELECT count(*) ... GROUP BY mmsi ORDER BY count(*) DESC LIMIT 20
TOP = [fishing, cargo]


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


def extract_window_records(segment, window_size=600, window_offset=300, min_size=20):
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


def autofocus():
    text_box.cursor_index = 0
    text_box._rendercursor()
    text_box.begin_typing(None)


def submit(text):
    if (text != '1') and (text != '2') and (text != '3'):
        text = 'NULL\n'
    else:
        text += '\n'

    with open('labeled_data/labels.txt', 'a') as fh:
        fh.write(text)

    plt.close()


if __name__ == '__main__':
    np.seterr(divide='ignore', invalid='ignore')
    to_file = False
    MINIMUM_SEGMENT_SIZE = 100
    SAMPLE_NO = 0

    # iterate through vessels
    for mmsis, database in zip(TOP, ['dutch_fishing_dynamic', 'dutch_cargo_dynamic']):
        print(mmsis)
        sys.exit()

        for mmsi in mmsis:
            fh = open('segments/{}.csv'.format(mmsi), 'w') if to_file else False

            # retrieve trajectory
            cur = retrieve_trajectory(mmsi, database)

            # segment trajectory into continuous segments
            segments = segment_continuous(cur, fh, 3, to_file)

            # iterate through segments
            for segment_no, segment in enumerate(segments):
                for sample in extract_window_records(segment, window_size=60 * 60 * 1, window_offset=60 * 60 * 1):
                    return_sample = []

                    for window_records in extract_window_records(sample):
                        if window_records:
                            # extract features of window
                            check, features = extract_features(window_records)

                            if check:
                                return_sample.append(features)

                    np.save('labeled_data/features_{}.npy'.format(SAMPLE_NO), np.array(return_sample))

                    coordinates = np.array([[_[2], _[3], _[4]] for _ in sample])
                    np.savetxt('labeled_data/coords_{}.csv'.format(SAMPLE_NO), coordinates[:, :2])

                    SAMPLE_NO += 1
                    longitudes = coordinates[:, 0]
                    latitudes = coordinates[:, 1]
                    speeds = coordinates[:, 2]
                    speeds[speeds > 14.] = 14.
                    speeds[speeds < 0.] = 0.
                    # colormap = speeds / 14.

                    sc = plt.scatter(longitudes, latitudes, marker='.', c=speeds, cmap='hot', vmin=0., vmax=14.)

                    plt.scatter(longitudes[0], latitudes[0], marker='o', c='green')
                    plt.scatter(longitudes[-1], latitudes[-1], marker='o', c='red')
                    plt.colorbar(sc)

                    axbox = plt.axes([0.1, 0.05, 0.8, 0.075])
                    text_box = TextBox(axbox, '')
                    text_box.on_submit(submit)

                    timer = plt.gcf().canvas.new_timer()
                    timer.single_shot = True
                    timer.add_callback(autofocus)
                    timer.start()

                    plt.autoscale(enable=True, axis='both')

                    mng = plt.get_current_fig_manager()
                    mng.full_screen_toggle()

                    with open('labeled_data/labels.txt', 'a') as fh:
                        fh.write('{}: '.format(SAMPLE_NO))

                    title = 'sample {}'.format(SAMPLE_NO)
                    plt.title(title)
                    plt.show()

            sys.exit()
            # # extract all windows
            # for window_records in extract_window_records(segment):
            #     if window_records:

            #         # extract features of window
            #         check, features = extract_features(window_records)

            #         if check:
            #             return_segment.append(features)

            # if len(return_segment) >= MINIMUM_SEGMENT_SIZE:
            #     print('  -> {} features.'.format(len(return_segment)))
            #     return_segment = np.array(return_segment)
            #     np.save('{}/{}_{}.npy'.format(database, mmsi, segment_no), return_segment)

            #     with open('{}/{}_{}.csv'.format(database.split('_')[1], mmsi, segment_no), 'w') as fh:
            #         for record in segment:
            #             fh.write('{},{}\n'.format(record[2], record[3]))

            # print('-> features extracted.\n')

            # if fh != False:
            #     fh.close()

    print('DONE.')
