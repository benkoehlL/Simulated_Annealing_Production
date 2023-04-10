import numpy as np
import pandas as pd

def gen_processing_time(mean=10, stdd=10):
    return abs(np.random.normal(mean * np.arange(N_JOBS), np.full(N_JOBS, stdd)))


def gen_due_date(processing_times, stdd=10):
    # mean_ptime = np.mean(processing_times)
    # due_date_offsets = np.random.normal(mean_ptime * np.arange(N_JOBS), np.full(N_JOBS, stdd))
    # return processing_times + due_date_offsets
    due_dates = abs(np.random.normal(len(processing_times), np.full(N_JOBS, stdd))) * np.arange(N_JOBS)
    due_dates = due_dates[np.random.permutation(len(processing_times))]
    return processing_times + due_dates


def gen_setup():
    return np.random.randint(0, N_TYPES, N_JOBS)

for N_JOBS in [10, 50, 100, 500, 1000]:
    N_MACHINES = 4
    N_TYPES = 10
    PATH = './data/J{}S{}'.format(N_JOBS, N_TYPES)

    processing_times = gen_processing_time(mean=10, stdd=10)
    due_dates = gen_due_date(processing_times, stdd=10)
    setups = gen_setup()

    # print([(p, d) for p, d in zip(processing_times, due_dates)])

    df = pd.DataFrame([{'id': i, 'pt': pt, 'dd': dd, 'type': type}
                       for i, (pt, dd, type) in enumerate(zip(processing_times, due_dates, setups))])

    df.to_pickle(PATH + ".pkl")

    # df = pd.read_pickle(PATH + ".pkl")
    with open(PATH + ".dat", 'w+') as f:
        for index, row in df.iterrows():
            f.write('{}\t{}\t{}\t{}\n'.format(
                int(row[0]), row[2], int(row[3]), row[1]
            ))
