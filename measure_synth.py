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

def load_synth(manifold:str, diff:str, n_samples:int, n_stages:int):
    with hf.File(f'datasets/synth/{manifold}_{diff}_{n_samples}.h5', 'r') as f:
        datas = []
        for i in range(n_stages):
            datas.append({
                'E':        np.array(f['E'][f'X{i}']),
                'O':        np.array(f['O'][f'X{i}']),
                'y_train':  np.array(f['E'][f'y{i}']),
                'y_test':   np.array(f['O'][f'y{i}']),
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
    logger = getLogger(name='measure', path=f'results/synth/{bench_fold}', create_time=create_time)
    logger.propagate = False

    os.makedirs(f'results/synth/{bench_fold}/', exist_ok=True)
    os.makedirs(f'results/synth/{bench_fold}/projections/', exist_ok=True)

    logger.info(f'bench_fold: {bench_fold}')

    with open(f'results/synth/{bench_fold}/meta.json', 'r') as f:
        meta = json.loads(f.read())

    datasets = meta['datasets']
    methods_name = meta['methods']
    n_methods = len(methods_name)

    # 指标
    metrics_name = [
        't',
        'c',
        'ns',
        'shp',
        'sil',
        'dsc',
        'nh',
        'ale',
        'acc_test',
        'acc_train',
    ]
    n_metrics = len(metrics_name)

    metrics = Metrics()

    metric_res = []

    result_saved_path = f'results/synth/{bench_fold}/results.csv'
    # 度量
    for dataset in datasets:
        [manifold, diff, n_samples, n_stages] = dataset
        logger.info(f"Dataset '{manifold}-{diff}-{n_samples}'")

        # load data
        stage_data = load_synth(manifold, diff, n_samples, n_stages)
        for oos_stage_idx, data in enumerate(stage_data):
            for method_idx, method_name in enumerate(methods_name):
                
                logger.info(f"Metrics running")
                try:
                    
                    # load data and projection
                    E = data['E']
                    O = data['O']
                    y_train = data['y_train']
                    y_test = data['y_test']

                    proj_saved_path = f'results/synth/{bench_fold}/projections/{manifold}_{diff}_{method_name}_{oos_stage_idx}_test.csv'
                    proj = load_projection(proj_saved_path)
                    proj_org_saved_path = f'results/synth/{bench_fold}/projections/{manifold}_{diff}_{method_name}_{oos_stage_idx}_train.csv'
                    proj_org = load_projection(proj_org_saved_path)

                    # update distance, label etc.
                    metrics.update_metrics(X_train=E, X_train_Embedded=proj_org, X_test=O, X_test_Embedded=proj, y_train=y_train, y_test=y_test)
                    
                except Exception as e:
                    logger.error(f"Metrics error, skip. Info: \n{traceback.format_exc()}, skip")
                    continue
                
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
                            'manifold': manifold,
                            'diff': diff,
                            'stage': oos_stage_idx,
                            'result': res,
                        })
                    except Exception as e:
                        logger.error(f"Metric '{metric_name}' error, skip. Info: \n{traceback.format_exc()}")
                        continue


    df_metrics = pd.DataFrame(metric_res)
    df_metrics.to_csv(result_saved_path, index=False, header=True, encoding='utf-8')
