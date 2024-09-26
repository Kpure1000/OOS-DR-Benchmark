import argparse
import os
import json
import traceback
import h5py as hf
from metrics import *
from datetime import datetime
from logger import getLogger
import pandas as pd

from utils import *

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Benchmark command line arguments.')
    parser.add_argument('-t', '--time', nargs=1, help='Specify bench time')

    args = parser.parse_args()

    if args.time:
        print(f"Given time: {args.time[0]}")
        bench_fold = args.time[0]
    else:
        raise Exception("No bench time given")

    create_time = datetime.now()
    logger = getLogger(name='measure', path=f'results/runtime/{bench_fold}', create_time=create_time)
    logger.propagate = False

    logger.info(f'bench_fold: {bench_fold}')

    with open(f'results/runtime/{bench_fold}/meta.json', 'r') as f:
        meta = json.loads(f.read())

    datasets = meta['datasets']
    methods_name = meta['methods']
    n_methods = len(methods_name)

    df_train = pd.read_csv(f'results/runtime/{bench_fold}/train_runtimes.csv')
    df_test  = pd.read_csv(f'results/runtime/{bench_fold}/test_runtimes.csv')

    method_u = df_train['method'].unique()

    props=[
        (3,7)
        (5,5),
        (7,3),
    ]

    mean_train = []
    mean_test = []

    for idx, prop in enumerate(props):
        for method in method_u:

            try:
                q_train = df_train.query(f'method == "{method}" and prop == {idx}')
                N_train = q_train['n_samples'].to_numpy()
                T_train = q_train['train_time'].to_numpy()

                meanTrainTime = np.dot(N_train, T_train) / N_train.sum()

                logger.info(f'[{prop[0]}:{prop[1]}] {method} mean train runtime: {meanTrainTime}')

                q_test = df_test.query(f'method == "{method}" and prop == {idx}')
                N_test = q_test['n_samples'].to_numpy()
                T_test = q_test['test_time'].to_numpy()

                meanTestTime = np.dot(N_test, T_test) / N_test.sum()

                logger.info(f'[{prop[0]}:{prop[1]}] {method} mean test runtime: {meanTestTime}')

                mean_train.append({
                    'method': method,
                    'prop': idx,
                    'runtime': meanTrainTime
                })
                mean_test.append({
                    'method': method,
                    'prop': idx,
                    'runtime': meanTestTime
                })

            except Exception as e:
                logger.error(f'[{prop[0]}:{prop[1]}] {method} failed, Info: {traceback.format_exc()}')
                continue

    pd.DataFrame(mean_train).to_csv(f'results/runtime/{bench_fold}/mean_train.csv', index=False)
    pd.DataFrame(mean_test).to_csv(f'results/runtime/{bench_fold}/mean_test.csv', index=False)