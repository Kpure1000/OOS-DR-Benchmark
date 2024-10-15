import os
import traceback
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
            datas.append({
                'E':        np.array(f['E'][f'X{i}']),
                'O':        np.array(f['O'][f'X{i}']),
                'y_train':  np.array(f['E'][f'y{i}']),
                'y_test':   np.array(f['O'][f'y{i}']),
            })
        return datas

if __name__ == '__main__':
    
    create_time = datetime.now()
    bench_time = create_time.strftime("%Y.%m.%d-%H.%M.%S")
    
    logger = getLogger(name='bench', path=f'results/synth/{bench_time}', create_time=create_time)
    logger.propagate = False

    os.makedirs(f'results/synth/{bench_time}/', exist_ok=True)
    os.makedirs(f'results/synth/{bench_time}/projections/', exist_ok=True)

    logger.info(f'bench_time: {bench_time}')

    methods_name = [
        'pca',
        'ae',
        'cdr',

        'oos-mds',
        'lmds',
        'kmds',
        'mimds',

        'oos-isomap',
        'lisomap',
        'kisomap',

        'ktsne',
        'ptsne',
        'ptsne22',
        'dlmp-tsne',
        
        'pumap',
        'dlmp-umap',
    ]

    n_methods = len(methods_name)
    
    methods = Methods(gpu_accel=True)

    # 要使用的数据集
    datasets = [
        ['syn1', 'dist', 1],
        ['syn1', 'prop', 4],
        
        ['syn2', 'dist', 1],
        ['syn2', 'prop', 4],
        
        ['syn4', 'dist', 1],
        ['syn4', 'prop', 4],
    ]

    meta = {
        "methods": methods_name,
        "datasets": datasets
    }

    with open(f'results/synth/{bench_time}/meta.json', 'w') as f:
        f.write(json.dumps(meta, indent=4))

    train_runtime_res = []
    test_runtime_res = []


    for dataset in datasets:
        [manifold, diff, n_stages] = dataset

        # get dataset
        stage_data = load_synth(manifold, diff, n_stages)

        for oos_stage_idx, data in enumerate(stage_data):
            E = data['E']
            O = data['O']
            y_train = data['y_train']
            y_test = data['y_test']
            
            logger.info(f"***** Dataset '{manifold}-{diff} [oos_stage={oos_stage_idx}]' start, train samples '{E.shape[0]}', test samples '{O.shape[0]}' *****")
            
            for method_idx, method_name in enumerate(methods_name):
                try:
                    proj_train_saved_path = f'results/synth/{bench_time}/projections/{manifold}_{diff}_{method_name}_{oos_stage_idx}_train.csv'
                    proj_test_saved_path = f'results/synth/{bench_time}/projections/{manifold}_{diff}_{method_name}_{oos_stage_idx}_test.csv'
                    
                    # get method
                    method = methods.get(method_name)

                    logger.info(f"Method '{method_name}' running... ")
                    
                    # train
                    start_time = time.perf_counter()
                    method.fit(E)
                    train_time =  time.perf_counter() - start_time
                    train_runtime_res.append({
                        'method': method_name,
                        'manifold': manifold,
                        'diff': diff,
                        'n_samples': E.shape[0],
                        'train_time': train_time
                    })
                    
                    # train projection
                    proj_train = method.transform(E)
                    
                    # test projection
                    start_time = time.perf_counter()
                    proj_test = method.transform_oos(O)
                    test_time = time.perf_counter() - start_time
                    test_runtime_res.append({
                        'method': method_name,
                        'manifold': manifold,
                        'diff': diff,
                        'n_samples': O.shape[0],
                        'test_time': test_time
                    })

                    # save projection
                    save_projection(proj_train_saved_path, proj=proj_train, label=y_train)
                    save_projection(proj_test_saved_path, proj=proj_test, label=y_test)
                    logger.info(f"Projections saved successfully")

                    # report
                    logger.info(f"*** Method '{method_name}' on '{manifold}-{diff}[oos_stage={oos_stage_idx}]' done, fitting time:{train_time:.4f}s, oos projecting time:{test_time:.4f}s ***")

                except Exception as e:
                    logger.error(f"Method '{method_name}' error with: \n{traceback.format_exc()}")
                    continue
    
    df_train_time = pd.DataFrame(train_runtime_res)
    df_test_time = pd.DataFrame(test_runtime_res)

    train_runtime_saved_path = f'results/synth/{bench_time}/train_runtimes.csv'
    test_runtime_saved_path = f'results/synth/{bench_time}/test_runtimes.csv'
    
    df_train_time.to_csv(train_runtime_saved_path, index=False, header=True, encoding='utf-8')
    df_test_time.to_csv(test_runtime_saved_path, index=False, header=True, encoding='utf-8')

    logger.info(f"Runtime saved to {train_runtime_saved_path} and {test_runtime_saved_path} successfully")

    # report
    logger.info(f"Finished benchmarking {bench_time} at {datetime.now()}")

