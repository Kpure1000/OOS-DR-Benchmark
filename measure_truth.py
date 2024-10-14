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

def load_truth(dataset_name, n_samples:int, n_stages:int):
    with hf.File(f'datasets/truth/{dataset_name}_{n_samples}.h5', 'r') as f:
        datas = []
        for i in range(n_stages):
            gE = f['E']
            gO = f['O']
            datas.append({
                'E':        np.array(gE[f'X{i}']),
                'O':        np.array(gO[f'X{i}']),
                'y_train':  np.array(gE[f'y{i}'] if f'y{i}' in gE else None),
                'y_test':   np.array(gO[f'y{i}'] if f'y{i}' in gO else None),
            })
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
    logger = getLogger(name='measure', path=f'results/truth/{bench_fold}', create_time=create_time)
    logger.propagate = False

    os.makedirs(f'results/truth/{bench_fold}/', exist_ok=True)
    os.makedirs(f'results/truth/{bench_fold}/projections/', exist_ok=True)

    logger.info(f'bench_fold: {bench_fold}')

    with open(f'results/truth/{bench_fold}/meta.json', 'r') as f:
        meta = json.loads(f.read())

    datasets = meta['datasets']
    methods_name = meta['methods']
    n_methods = len(methods_name)

    # 指标
    metrics_name = [
        't',
        'c',
        'nh',
        'lc',
        'tp',
        'sd',
        'sc',
        'dsc',
        'acc_oos',
        'acc_e',
    ]
    n_metrics = len(metrics_name)

    metrics = Metrics()

    metric_res = []

    result_saved_path = f'results/truth/{bench_fold}/results.csv'
    # 度量
    for dataset in datasets:
        [dataset_name, n_samples, n_stages] = dataset
        logger.info(f"Dataset '{dataset_name}-{n_samples}'")

        # load data
        stage_data = load_truth(dataset_name=dataset_name, n_samples=n_samples, n_stages=n_stages)
        for oos_stage_idx, data in enumerate(stage_data):
            for method_idx, method_name in enumerate(methods_name):
                
                logger.info(f"Metrics for '{method_name}-{dataset_name}-{oos_stage_idx}' updating")

                try:
                    
                    # load data and projection
                    E = data['E']
                    O = data['O']
                    y_train = data['y_train']
                    y_test = data['y_test']

                    train_proj_saved_path = f'results/truth/{bench_fold}/projections/{dataset_name}_{method_name}_{oos_stage_idx}_train.csv'
                    proj_train = load_projection(train_proj_saved_path)

                    test_proj_saved_path = f'results/truth/{bench_fold}/projections/{dataset_name}_{method_name}_{oos_stage_idx}_test.csv'
                    proj_test = load_projection(test_proj_saved_path)

                    # update distance, label etc.
                    metrics.update_metrics(X_train=E, X_train_Embedded=proj_train, X_test=O, X_test_Embedded=proj_test, y_train=y_train, y_test=y_test)
                    
                except Exception as e:
                    logger.error(f"Metrics error at '{method_name}-{dataset_name}-{oos_stage_idx}', skip. Info: \n{traceback.format_exc()}, skip")
                    continue

                logger.info(f"Metrics ready")
                
                # 计算各项指标
                for metric_idx, metric_name in enumerate(metrics_name):
                    try:
                        (res, m_name) = metrics.run_single(metric_name)
                        res = float(res)
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
                        logger.error(f"Metric '{metric_name}' error at '{method_name}-{dataset_name}-{oos_stage_idx}', skip. Info: \n{traceback.format_exc()}")
                        continue


    df_metrics = pd.DataFrame(metric_res)
    df_metrics.to_csv(result_saved_path, index=False, header=True, encoding='utf-8')
