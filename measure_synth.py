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

def load_synth(dataset_name:str, diff:str, n_stages:int, is_Labeled:bool):
    with hf.File(f'datasets/synth/{dataset_name}_{diff}.h5', 'r') as f:
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
    parser.add_argument('-t', '--time', nargs=1, help='Specify bench time (folder name)')
    parser.add_argument('-i', '--ignore-meta', action='store_true', required=False, help="Using benckmark.yml instead of meta.json")

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

    with open('benchmark.yml', 'r', encoding='utf-8') as f:
        bench_conf = yaml.load(f, Loader=yaml.FullLoader)
        metrics_name = bench_conf['metrics']
        logger.info(f'Load Config Success')

    if args.ignore_meta:
        datasets = bench_conf['datasets']['runtime']
        methods_name = bench_conf['methods']
    else:
        with open(f'results/synth/{bench_fold}/meta.json', 'r') as f:
            meta = json.loads(f.read())
            datasets = meta['datasets']
            methods_name = meta['methods']

    metrics = Metrics()

    metric_res = []

    result_saved_path = f'results/synth/{bench_fold}/results.csv'
    # 度量
    for dataset in datasets:
        [manifold, diff, n_stages, is_Labeled] = dataset
        logger.info(f"Dataset '{manifold}-{diff}-{n_stages}'")

        # load data
        stage_data = load_synth(manifold, diff, n_stages, is_Labeled)
        for oos_stage_idx, (X_train, X_test, y_train, y_test) in enumerate(stage_data):
            for method_idx, method_name in enumerate(methods_name):

                logger.info(f">>>>> Metrics for Method '{method_name}' on Dataset '{manifold}-{diff}-{oos_stage_idx}' calculating...")
                try:
                    proj_saved_path = f'results/synth/{bench_fold}/projections/{manifold}_{diff}_{method_name}_{oos_stage_idx}_test.csv'
                    proj = load_projection(proj_saved_path)
                    proj_org_saved_path = f'results/synth/{bench_fold}/projections/{manifold}_{diff}_{method_name}_{oos_stage_idx}_train.csv'
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
