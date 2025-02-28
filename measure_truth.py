import argparse
import os
import json
import traceback
import h5py as hf
import yaml
from metrics import *
from datetime import datetime
from logger import getLogger
import pandas as pd

from utils import *

def load_truth(dataset_name, n_samples:int, n_stages:int, is_Labeled:bool):
    with hf.File(f'datasets/truth/{dataset_name}_{n_samples}.h5', 'r') as f:
        datas = []
        for i in range(n_stages):
            gE = f['E']
            gO = f['O']
            datas.append((
                gE[f'X{i}'][:],
                gO[f'X{i}'][:],
                gE[f'y{i}'][:] if is_Labeled else None,
                gO[f'y{i}'][:] if is_Labeled else None,
            ))
        return datas

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser(description='Benchmark command line arguments.')
    parser.add_argument('-t', '--time', nargs=1, help='Specify bench time (folder name)')
    parser.add_argument('-i', '--ignore-meta', action='store_true', required=False, help="Using benckmark.yml instead of meta.json")
    
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

    with open('benchmark.yml', 'r', encoding='utf-8') as f:
        bench_conf = yaml.load(f, Loader=yaml.FullLoader)
        metrics_name = bench_conf['metrics']
        logger.info(f'Load Config Success')

    if args.ignore_meta:
        datasets = bench_conf['datasets']['runtime']
        methods_name = bench_conf['methods']
    else:
        with open(f'results/truth/{bench_fold}/meta.json', 'r') as f:
            meta = json.loads(f.read())
            datasets = meta['datasets']
            methods_name = meta['methods']

    metrics = Metrics()

    metric_res = []

    result_saved_path = f'results/truth/{bench_fold}/results.csv'
    # 度量
    for dataset in datasets:
        [dataset_name, n_samples, n_stages, is_Labeled] = dataset
        logger.info(f"Dataset '{dataset_name}-{n_samples}'")

        # load data
        stage_data = load_truth(dataset_name=dataset_name, n_samples=n_samples, n_stages=n_stages, is_Labeled=is_Labeled)
        for oos_stage_idx, (X_train, X_test, y_train, y_test) in enumerate(stage_data):
            for method_idx, method_name in enumerate(methods_name):
                
                logger.info(f">>>>> Metrics for Method '{method_name}' on Dataset '{dataset_name}-{oos_stage_idx}' calculating...")

                try:
                    train_proj_saved_path = f'results/truth/{bench_fold}/projections/{dataset_name}_{method_name}_{oos_stage_idx}_train.csv'
                    proj_train = load_projection(train_proj_saved_path)

                    test_proj_saved_path = f'results/truth/{bench_fold}/projections/{dataset_name}_{method_name}_{oos_stage_idx}_test.csv'
                    proj_test = load_projection(test_proj_saved_path)

                    # update distance, label etc.
                    metrics.update_metrics(X_train=X_train, X_train_Embedded=proj_train, X_test=X_test, X_test_Embedded=proj_test, y_train=y_train, y_test=y_test)
                    
                except Exception as e:
                    logger.error(f"Metrics error at '{method_name}-{dataset_name}-{oos_stage_idx}', skip. Info: \n{traceback.format_exc()}, skip")
                    continue

                logger.info(f"Metrics ready")
                
                # 计算各项指标
                for metric_idx, metric_name in enumerate(metrics_name):
                    try:
                        (res, m_name) = metrics.run_single(metric_name)
                        if res is None:
                            continue
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
