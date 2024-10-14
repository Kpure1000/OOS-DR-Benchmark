import argparse
import os
import traceback
import h5py as hf
from methods.methods import Methods
from datetime import datetime
import time
from logger import getLogger
import pandas as pd
import numpy as np
import json
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
    
    create_time = datetime.now()
    bench_time = create_time.strftime("%Y.%m.%d-%H.%M.%S")
    
    logger = getLogger(name='bench', path=f'results/truth/{bench_time}', create_time=create_time)
    logger.propagate = False

    os.makedirs(f'results/truth/', exist_ok=True)
    os.makedirs(f'results/truth/{bench_time}/', exist_ok=True)
    os.makedirs(f'results/truth/{bench_time}/projections/', exist_ok=True)

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
    
    methods = Methods()

    # 要使用的数据集
    datasets = [
        # ['cnae9', 1080, 1],         # 跑完啦
        # ['fashionMNIST', 4200, 1],  # 跑完啦
        # ['cifar10', 3250, 1],       # 跑完啦
        # ['imdb', 5000, 1],          # 跑完啦

        # ['dendritic', 576, 1],      
        # ['retina', 7606, 1],
        # ['pancreas', 7383, 1],

        # ['raid', 1955, 1],
        ['sensor', 7213, 1],
        # ['merchant', 3897, 1],
         
        # ['isomapFace', 698, 1],   # 跑完啦
        # ['freyFace', 1965, 1],    # 跑完啦
    ]

    meta = {
        "methods": methods_name,
        "datasets": datasets
    }

    with open(f'results/truth/{bench_time}/meta.json', 'w') as f:
        f.write(json.dumps(meta, indent=4))

    train_runtime_res = []
    test_runtime_res = []


    for dataset in datasets:
        [dataset_name, n_samples, n_stages] = dataset

        # get dataset
        stage_data = load_truth(dataset_name, n_samples, n_stages)

        for oos_stage_idx, data in enumerate(stage_data):
            E = data['E']
            O = data['O']
            y_train = data['y_train']
            y_test = data['y_test']
            
            logger.info(f"***** Dataset '{dataset_name} [oos_stage={oos_stage_idx}]' start, train samples '{E.shape[0]}', test samples '{O.shape[0]}' *****")
            
            for method_idx, method_name in enumerate(methods_name):
                try:
                    proj_train_saved_path = f'results/truth/{bench_time}/projections/{dataset_name}_{method_name}_{oos_stage_idx}_train.csv'
                    proj_test_saved_path = f'results/truth/{bench_time}/projections/{dataset_name}_{method_name}_{oos_stage_idx}_test.csv'
                    
                    # get method
                    method = methods.get(method_name)

                    logger.info(f"Method '{method_name}' running... ")
                    
                    # train
                    start_time = time.perf_counter()
                    method.fit(E)
                    train_time =  time.perf_counter() - start_time
                    train_runtime_res.append({
                        'method': method_name,
                        'dataset': dataset_name,
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
                        'dataset': dataset_name,
                        'n_samples': O.shape[0],
                        'test_time': test_time
                    })

                    # save projection
                    save_projection(proj_train_saved_path, proj=proj_train, label=y_train)
                    save_projection(proj_test_saved_path, proj=proj_test, label=y_test)
                    logger.info(f"Projections saved successfully")

                    # report
                    logger.info(f"*** Method '{method_name}' on '{dataset_name}[oos_stage={oos_stage_idx}]' done, fitting time:{train_time:.4f}s, oos projecting time:{test_time:.4f}s ***")

                except Exception as e:
                    logger.error(f"Method '{method_name}' error with: \n{traceback.format_exc()}")
                    continue
    
    df_train_time = pd.DataFrame(train_runtime_res)
    df_test_time = pd.DataFrame(test_runtime_res)

    train_runtime_saved_path = f'results/truth/{bench_time}/train_runtimes.csv'
    test_runtime_saved_path = f'results/truth/{bench_time}/test_runtimes.csv'
    
    df_train_time.to_csv(train_runtime_saved_path, index=False, header=True, encoding='utf-8')
    df_test_time.to_csv(test_runtime_saved_path, index=False, header=True, encoding='utf-8')

    logger.info(f"Runtime saved to {train_runtime_saved_path} and {test_runtime_saved_path} successfully")

    # report
    logger.info(f"Finished benchmarking {bench_time} at {datetime.now()}")

