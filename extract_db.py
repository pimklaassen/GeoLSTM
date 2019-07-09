import psycopg2
import datetime
import sys
import numpy as np


# parameters for psycopg2
# put cred.auth file in ./pwd/ containing parameters
params = eval(open('../pwd/cred.auth', 'r').read())

# set up connection with DB
try:
    connection = psycopg2.connect(**params)
    print('-> CONNECTION ESTABLISHED.')
    print()
except:
    print('-> CONNECTION FAILED.')
    sys.exit()


def fetch_mmsis(table_names, amount):
    """Fetches n MMSI numbers for list of ship

    Parameter(s):
        -- shiptypes: list of shiptype codes
        -- amount: number of mmsi's per shiptype to fetch
    """
    # select the ships that are most dense in AIS points
    mmsi_lists = []

    for table_name in table_names:
        cur = connection.cursor()
        cur.execute('select mmsi from {} group by mmsi order by count(*) desc limit {};'.format(table_name, amount))
        mmsi_lists.append([_[0] for _ in cur.fetchall()])

    cur.close()
    return mmsi_lists


def retrieve_trajectory(mmsi, table_name):
    """Retrieves a generator of records for the given MMSI.

    Parameter(s):
        -- mmsi: unique identifier of vessel
    """
    cur = connection.cursor()
    cur.execute('select * from {} where mmsi = {} order by ts;'.format(table_name, mmsi))
    print('-> records retrieved for mmsi {}.'.format(mmsi))
    return cur


def segment_continuous(shiptype, cur, threshold=3, min_size=50):
    """Segments a trajectory into temporally continuous parts.

    Parameter(s):
        -- threshold: a gap of this much hours cuts a trajecory into a new segment
    """
    # read out first record
    # FIX: UTC WRONG
    dtype = [('ts', 'datetime64[us]'),
             ('lon', float),
             ('lat', float),
             ('stat', int),
             ('turn', int),
             ('speed', float),
             ('course', float),
             ('head', int)]

    record = cur.fetchone()
    prev = record[1]
    j = 0

    # set up empty list of segments
    segment = [record[1:]]

    for record in cur:
        difference = record[1] - prev
        prev = record[1]

        # if time gap exceeds threshold, create new segment
        if difference > datetime.timedelta(hours=threshold):

            if len(segment) > min_size:
                segment = np.array(segment, dtype=dtype)
                np.save('Data/Raw/{}_{}_{}.npy'.format(shiptype, record[0], j), segment)
                print(' -> {}_{}_{} pickled.'.format(shiptype, record[0], j))
                j += 1

            # create new empty
            segment = [record[1:]]
        else:
            segment.append(record[1:])

    if len(segment) > min_size:
        segment = np.array(segment, dtype=dtype)
        np.save('Data/Raw/{}_{}_{}.npy'.format(shiptype, record[0], j), segment)
        print(' -> {}_{}_{} pickled.'.format(shiptype, record[0], j))


if __name__ == '__main__':
    table_names = ['dutch_fishing_dynamic', 'dutch_cargo_dynamic']
    amount = 2
    threshold = 3

    # retrieve mmsi's
    mmsi_lists = fetch_mmsis(table_names, amount)

    # iterate through list of mmsi's
    for shiptype, (mmsis, table_name) in enumerate(zip(mmsi_lists, table_names)):
        for mmsi in mmsis:
            cur = retrieve_trajectory(mmsi, table_name)
            segment_continuous(shiptype, cur, threshold=threshold)
