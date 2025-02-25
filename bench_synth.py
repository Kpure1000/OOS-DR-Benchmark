import os
import traceback

import yaml
from methods.methods import Methods
from datetime import datetime
import time
from logger import getLogger
import pandas as pd
import json
import h5py as hf
import numpy as np
from utils import *

def load_synth(dataset_name:str, diff:str, n_stages:int):
    with hf.File(f'datasets/synth/{dataset_name}_{diff}.h5', 'r') as f:
        datas = []
        for i in range(n_stages):
            datas.append((
                f['E'][f'X{i}'][:],
                f['O'][f'X{i}'][:]
            ))
        return datas

if __name__ == '__main__':
    
    create_time = datetime.now()
    bench_time = create_time.strftime("%Y.%m.%d-%H.%M.%S")
    
    os.makedirs(f'results/synth/', exist_ok=True)
    logger = getLogger(name='bench', path=f'results/synth/{bench_time}', create_time=create_time)
    logger.propagate = False

    os.makedirs(f'results/synth/{bench_time}/', exist_ok=True)
    os.makedirs(f'results/synth/{bench_time}/projections/', exist_ok=True)
    logger.info(f'Benchmark folder created, "meta.json" and "projections/" will be saved in "results/truth/{bench_time}"')

    with open('benchmark.yml', 'r', encoding='utf-8') as f:
        bench_conf = yaml.load(f, Loader=yaml.FullLoader)
        methods_name = bench_conf['methods']
        datasets = bench_conf['datasets']['synth']
        logger.info(f'Load Config Success')

    with open(f'results/synth/{bench_time}/meta.json', 'w') as f:
        meta = {
            "methods": methods_name,
            "datasets": datasets
        }
        f.write(json.dumps(meta, indent=4))
        logger.info(f'Save Meta Data Success')

    logger.info(f'Accessible Datasets: {list(map(lambda e:e[0], datasets))}')

    methods = Methods(gpu_accel=True)
    logger.info(f'Accessible Methods: {methods_name}')

    train_runtime_res = []
    test_runtime_res = []

    logger.info(f"------- START BENCHMARK -------")

    for dataset in datasets:
        [dataset_name, diff, n_stages, _] = dataset

        # get dataset
        stage_data = load_synth(dataset_name, diff, n_stages)

        for oos_stage_idx, (X_train, X_test) in enumerate(stage_data):
            
            logger.info(f"***** Dataset '{dataset_name}-{diff} [oos_stage={oos_stage_idx}]' start, train samples '{X_train.shape[0]}', test samples '{X_test.shape[0]}' *****")
            
            for method_idx, method_name in enumerate(methods_name):
                try:
                    proj_train_saved_path = f'results/synth/{bench_time}/projections/{dataset_name}_{diff}_{method_name}_{oos_stage_idx}_train.csv'
                    proj_test_saved_path = f'results/synth/{bench_time}/projections/{dataset_name}_{diff}_{method_name}_{oos_stage_idx}_test.csv'
                    
                    # get method
                    method = methods.get(method_name)

                    logger.info(f"Method '{method_name}' running... ")
                    
                    # train
                    start_time = time.perf_counter()
                    method.fit(X_train)
                    train_time =  time.perf_counter() - start_time
                    train_runtime_res.append({
                        'method': method_name,
                        'manifold': dataset_name,
                        'diff': diff,
                        'n_samples': X_train.shape[0],
                        'train_time': train_time
                    })
                    
                    # train projection
                    proj_train = method.transform(X_train)
                    
                    # test projection
                    start_time = time.perf_counter()
                    proj_test = method.transform_oos(X_test)
                    test_time = time.perf_counter() - start_time
                    test_runtime_res.append({
                        'method': method_name,
                        'manifold': dataset_name,
                        'diff': diff,
                        'n_samples': X_test.shape[0],
                        'test_time': test_time
                    })

                    # report
                    logger.info(f"*** Method '{method_name}' on '{dataset_name}-{diff}[oos_stage={oos_stage_idx}]' done, fitting time:{train_time:.4f}s, oos projecting time:{test_time:.4f}s ***")

                    # save projection
                    save_projection(proj_train_saved_path, proj=proj_train)
                    save_projection(proj_test_saved_path, proj=proj_test)
                    logger.info(f"Projections saved successfully")

                except Exception as e:
                    logger.error(f"Method '{method_name}' error with: \n{traceback.format_exc()}")
                    continue
    
    df_train_time = pd.DataFrame(train_runtime_res)
    df_test_time = pd.DataFrame(test_runtime_res)

    train_runtime_saved_path = f'results/synth/{bench_time}/train_runtimes.csv'
    test_runtime_saved_path = f'results/synth/{bench_time}/test_runtimes.csv'
    
    df_train_time.to_csv(train_runtime_saved_path, index=False, header=True, encoding='utf-8')
    df_test_time.to_csv(test_runtime_saved_path, index=False, header=True, encoding='utf-8')

    logger.info(f"Runtime saved to '{train_runtime_saved_path}' and '{test_runtime_saved_path}' successfully")

    # report
    logger.info(f"------- FINISHED BENCHMARKING, saved in 'results/synth/{bench_time}/' -------")

