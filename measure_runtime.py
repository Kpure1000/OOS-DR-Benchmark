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

def load_runtime(dataset_name:str, n_stages:int, is_Labeled:bool):
    with hf.File(f'datasets/runtime/{dataset_name}.h5', 'r') as f:
        datas = []
        for i in range(n_stages):
            datas.append((
                f['E'][f'X{i}'][:],
                f['O'][f'X{i}'][:],
                f['E'][f'y{i}'][:] if is_Labeled else None,
                f['O'][f'y{i}'][:] if is_Labeled else None,
            ))
        return datas

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

    with open('benchmark.yml', 'r', encoding='utf-8') as f:
        bench_conf = yaml.load(f, Loader=yaml.FullLoader)
        metrics_name = bench_conf['metrics']
        logger.info(f'Load Config Success')

    datasets = meta['datasets']
    methods_name = meta['methods']

    metrics = Metrics()

    metric_res = []

    result_saved_path = f'results/runtime/{bench_fold}/results.csv'
    
    logger.info("------- START MEASURING -------")

    # 度量
    for dataset in datasets:
        [dataset_name, n_stages, is_Labeled] = dataset
        logger.info(f"Dataset '{dataset_name}-{n_stages}'")

        # load data
        stage_data = load_runtime(dataset_name, n_stages, is_Labeled)
        for oos_stage_idx, (X_train, X_test, y_train, y_test) in enumerate(stage_data):
            for method_idx, method_name in enumerate(methods_name):

                logger.info(f">>>>> Metrics for Method '{method_name}' on Dataset '{dataset_name}{oos_stage_idx}' calculating...")
                try:
                    proj_saved_path = f'results/runtime/{bench_fold}/projections/{dataset_name}_{method_name}_{oos_stage_idx}_test.csv'
                    proj = load_projection(proj_saved_path)
                    proj_org_saved_path = f'results/runtime/{bench_fold}/projections/{dataset_name}_{method_name}_{oos_stage_idx}_train.csv'
                    proj_org = load_projection(proj_org_saved_path)

                    # update distance, label etc.
                    metrics.update_metrics(X_train=X_train,
                                           X_train_Embedded=proj_org,
                                           X_test=X_test,
                                           X_test_Embedded=proj,
                                           y_train=y_train,
                                           y_test=y_test)

                except Exception as e:
                    logger.error(f"Metrics error, skip. Info: \n{traceback.format_exc()}, skip")
                    continue

                # 计算各项指标
                for metric_idx, metric_name in enumerate(metrics_name):
                    try:
                        (res, m_name) = metrics.run_single(metric_name)
                        res = float(res)
                        if res is None:
                            continue
                        logger.info(f"Metric '{m_name}' done, result: {res:.4g}")
                        # 保存度量结果
                        metric_res.append({
                            'method': method_name,
                            'metric': metric_name,
                            'dataset_name': dataset_name,
                            'stage': oos_stage_idx,
                            'result': res,
                        })
                    except Exception as e:
                        logger.error(f"Metric '{metric_name}' error, skip. Info: \n{traceback.format_exc()}")
                        continue

    df_metrics = pd.DataFrame(metric_res)
    df_metrics.to_csv(result_saved_path, index=False, header=True, encoding='utf-8')

    # 计算平均时间
    df_train = pd.read_csv(f'results/runtime/{bench_fold}/train_runtimes.csv')
    df_test  = pd.read_csv(f'results/runtime/{bench_fold}/test_runtimes.csv')

    method_u = df_train['method'].unique()

    props=[
        (9, 1),
        (7, 3),
        (5, 5),
        (3, 7),
        (1, 9)
    ]

    mean_train = []
    mean_test = []

    logger.info("------- START CALCULATING MEAN RUNTIME -------")

    for idx, prop in enumerate(props):
        for method in method_u:

            try:
                q_train = df_train.query(f'method == "{method}" and prop == {idx}')
                N_train = q_train['n_samples'].to_numpy()
                T_train = q_train['train_time'].to_numpy()

                meanTrainTime = np.dot(N_train, T_train) / N_train.sum()

                logger.info(f"prop = {prop[0]}:{prop[1]}, '{method}' mean train runtime: {meanTrainTime}")

                q_test = df_test.query(f'method == "{method}" and prop == {idx}')
                N_test = q_test['n_samples'].to_numpy()
                T_test = q_test['test_time'].to_numpy()

                meanTestTime = np.dot(N_test, T_test) / N_test.sum()

                logger.info(f"prop = {prop[0]}:{prop[1]}, '{method}' mean test runtime: {meanTestTime}")

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
                logger.error(f"prop = {prop[0]}:{prop[1]}, '{method}' failed, skip. Info: {traceback.format_exc()}")
                continue

    pd.DataFrame(mean_train).to_csv(f'results/runtime/{bench_fold}/mean_train.csv', index=False)
    pd.DataFrame(mean_test).to_csv(f'results/runtime/{bench_fold}/mean_test.csv', index=False)