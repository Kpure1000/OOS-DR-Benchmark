import argparse
import os
import traceback
from gen_dataset import *
from methods.methods import Methods
from metrics import *
from datetime import datetime
import time
from logger import getLogger
import matplotlib
matplotlib.use('Qt5Agg')
import seaborn as sns

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


def mat_plot(mat_data: np.ndarray, save_path:str = None, xlabels: list = None, ylabels: list = None, showGUI: bool = False, fmt='.4g', cm = None, vmax = None):
    
    fig = plt.figure(figsize=(len(xlabels), len(ylabels) / 4))
    ax = fig.add_subplot(111)
    sns.set_style('whitegrid')
    cm = sns.diverging_palette(20, 220, n=200) if cm is None else cm
    sns.set_theme(font_scale=1.0)
    sns.heatmap(
        mat_data,
        vmin=0,
        vmax=1 if vmax is None else vmax,
        annot=True,
        fmt='.4g',
        cmap=cm,
        xticklabels=xlabels if xlabels is not None else 'auto',
        yticklabels=ylabels if ylabels is not None else 'auto',
        annot_kws={'size': 8},
    )
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=8)
    ax.tick_params(axis='y', labelright=False, labelleft = True)

    plt.tight_layout()
    if showGUI:
        plt.show()

    if save_path is not None:
        fig.savefig(save_path + '.png', dpi=300)
        fig.savefig(save_path + '.eps')


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Benchmark command line arguments.')
    parser.add_argument('-t', '--time', nargs=1, help='Specify bench time')
    
    args = parser.parse_args()

    given_time = False
    if args.time:
        print(f"Given time: {args.time[0]}")
        given_time = True
        create_time = datetime.strptime(args.time[0], "%Y.%m.%d-%H.%M.%S")
    else:
        create_time = datetime.now()

    logger = getLogger('bench', create_time=create_time)
    logger.propagate = False

    bench_time = create_time.strftime("%Y.%m.%d-%H.%M.%S")

    if not given_time:
        os.makedirs(f'results/{bench_time}/', exist_ok=True)
        os.makedirs(f'imgs/{bench_time}/', exist_ok=True)
    elif os.path.exists(f'results/{bench_time}/') is False or os.path.exists(f'imgs/{bench_time}/') is False:
        logger.error(f'No such directory: results/{bench_time}/ or imgs/{bench_time}/')
        exit(1)

    methods = Methods()

    # 要评估的方法
    methods_name = [
        'pca',

        # 'ae',
        # 'cdr',
        # 'dlmp-tsne',
        # 'dlmp-umap',

        # 'ptsne',
        # 'ptsne22',
        # 'pumap',

        # 'oos-mds',
        # 'oos-isomap',
        # 'mimds',
        # 'kmds',
        # 'kisomap',
        # 'ktsne',
        # 'lmds',
        # 'lisomap',
    ]
    n_methods = len(methods_name)
    
    # 要使用的数据集
    datasets = [
        ('plane','structure', 2000, 1),
        ('plane','dist', 2000, 2),
        ('plane','prop', 2000, 4),
        
        ('swissroll','structure', 2000, 1),
        ('swissroll','dist', 2000, 2),
        ('swissroll','prop', 2000, 4),
        
        ('hybrid','structure', 2000, 1),
        ('hybrid','dist', 2000, 2),
        ('hybrid','prop', 2000, 4),
    ]

    metrics = Metrics()

    # 指标
    metrics_name = [
        # 't',
        # 'c',
        # 'ns',
        # 'shp',
        'sil',
        'dsc',
        'nh',
        'ale',
    ]
    # 指标数量
    n_metrics = len(metrics_name)

    for dataset in datasets:
        (manifold, diff, n_samples, n_stages) = dataset

        logger.info(f"dataset '{manifold}-{diff}'")

        # 加载生成数据集
        datas = load_synth(manifold, diff, n_samples, n_stages)
    
        result_saved_path = f'results/{bench_time}/{manifold}_{diff}_metrics.csv'
        train_runtime_saved_path = f'results/{bench_time}/{manifold}_{diff}_train_runtimes.csv'
        test_runtime_saved_path = f'results/{bench_time}/{manifold}_{diff}_test_runtimes.csv'

        # 获取OOS数据集的阶段数
        n_oos_stages = len(datas)

        res_mat = np.zeros(shape=(n_metrics * n_methods, n_oos_stages), dtype=np.float64)
        train_runtime_mat = np.zeros(shape=(n_methods, n_oos_stages), dtype=np.float64)
        test_runtime_mat =  np.zeros(shape=(n_methods, n_oos_stages), dtype=np.float64)

        if not given_time:

            for oos_stage_idx, data in enumerate(datas):
                E = data['E']
                O = data['O']
                y_train = data['y_train']
                y_test = data['y_test']
                
                logger.info(f"Dataset '{manifold}-{diff}'[oos_stage={oos_stage_idx}] start, train samples '{E.shape[0]}', test samples '{O.shape[0]}'")
                
                for method_idx, method_name in enumerate(methods_name):
                    # 进行投影
                    try:
                        logger.info(f"Method '{method_name}' running... ")
                        # 获取投影方法
                        method = methods.get(method_name)
                        # 训练
                        start_time = time.perf_counter()
                        method.fit(E)
                        train_time =  time.perf_counter() - start_time
                        train_runtime_mat[method_idx, oos_stage_idx] = train_time # 保存训练时间
                        logger.info(f"Fitting time: {train_time:.4f}s")
                        # 训练集投影
                        proj_org = method.transform(E)
                        # 测试集投影
                        start_time = time.perf_counter()
                        proj = method.transform_oos(O)
                        test_time = time.perf_counter() - start_time
                        test_runtime_mat[method_idx, oos_stage_idx] = test_time # 保存测试集投影时间
                        logger.info(f"OOS projecting time: {test_time:.4f}s")

                        proj_saved_path = f'results/{bench_time}/{manifold}_{diff}_{method_name}_{oos_stage_idx}.csv'
                        np.savetxt(fname=proj_saved_path, X=proj, header='x,y', delimiter=',', comments='')

                        logger.info(f"Method '{method_name}' done, fitting time:{train_time:.4f}s, OOS projecting time:{test_time:.4f}s, projection saved to {proj_saved_path}")
                    except Exception as e:
                        logger.error(f"Method '{method_name}' error with: \n{traceback.format_exc()}")
                        continue

                    logger.info(f"Metrics running")
                    try:
                        # 更新指标
                        metrics.update_metrics(O, proj, X_train_Embedded=proj_org, y_test=y_test)
                        # 计算各项指标
                        for metric_idx, metric_name in enumerate(metrics_name):
                            try:
                                (res, m_name) = metrics.run_single(metric_name)
                                logger.info(f"Metric '{m_name}' done, result: {res:.5g}")
                                # 结果保存在指标矩阵中
                                res_mat[method_idx * n_metrics + metric_idx, oos_stage_idx] = res
                            except Exception as e:
                                # res_mat[method_idx * n_metrics + metric_idx, oos_stage_idx] = np.nan
                                logger.error(f"Metric '{m_name}' error with: \n{traceback.format_exc()}, setting to nan, continue")
                                continue

                    except Exception as e:
                        logger.error(f"Metrics error with: \n{traceback.format_exc()}")
                        continue

            try:
                # 处理需要归一化的结果
                # for met_idx, (met_name, (need_nor, need_inv)) in enumerate(metrics.post_processing(metrics_name)):
                #     if need_nor:
                #         ele = res_mat[[meth_idx * metrics_len + met_idx for meth_idx in range(len(methods_name))], :]
                #         vmin = np.min(ele)
                #         vmax = np.max(ele)
                #         logger.info(f"Metrics '{met_name}' need normalize, values: '{ele}', min: {vmin:.5g}, max: {vmax:.5g}")
                #         res_mat[[meth_idx * metrics_len + met_idx for meth_idx in range(len(methods_name))], :] = (ele - vmin) / (vmax - vmin)
                #     if need_inv:
                #         ele = res_mat[[meth_idx * metrics_len + met_idx for meth_idx in range(len(methods_name))], :]
                #         logger.info(f"Metrics '{met_name}' need inverse, values: '{ele}'")
                #         res_mat[[meth_idx * metrics_len + met_idx for meth_idx in range(len(methods_name))], :] = 1 - ele
                    
                # 保存结果矩阵到本地
                np.savetxt(result_saved_path, X=res_mat, delimiter=',', comments='')
                    
                np.savetxt(train_runtime_saved_path, X=train_runtime_mat, delimiter=',', comments='')
                np.savetxt(test_runtime_saved_path, X=test_runtime_mat, delimiter=',', comments='')

                logger.info(f"Results saved to '{result_saved_path}'")
            
            except Exception as e:
                logger.error(f"Save results error with: \n{traceback.format_exc()}")

        try:
            # 画结果热力图

            res_mat = np.loadtxt(result_saved_path, delimiter=',')
            res_mat = res_mat.reshape(n_metrics * n_methods, n_oos_stages)

            ylabels_metric = [f'{method_name}-{met_abs_name}' for method_name in methods_name for met_abs_name in metrics_name]
            xlabels_metric = [f'{diff}.{i + 1}' for i in range(n_stages)]
            mat_plot(res_mat, save_path=f'imgs/{bench_time}/{manifold}_{diff}_res', ylabels=ylabels_metric, xlabels=xlabels_metric)
            
            # 画运行时间热力图
        
            train_runtime_mat = np.loadtxt(train_runtime_saved_path, delimiter=',')
            train_runtime_mat = train_runtime_mat.reshape(n_methods, n_oos_stages)

            test_runtime_mat = np.loadtxt(test_runtime_saved_path, delimiter=',')
            test_runtime_mat = test_runtime_mat.reshape(n_methods, n_oos_stages)

            ylabel_methods = methods_name
            xlabel_samples = [f"N={data['E'].shape[0]}" for data in datas]
            cm = sns.diverging_palette(240, 20, n=200)
            mat_plot(train_runtime_mat, save_path=f'imgs/{bench_time}/{manifold}_{diff}_train_runtime', ylabels=ylabel_methods, xlabels=xlabel_samples, fmt='.5g', cm=cm, vmax=np.max(train_runtime_mat))
            xlabel_samples = [f"N={data['O'].shape[0]}" for data in datas]
            mat_plot(test_runtime_mat, save_path=f'imgs/{bench_time}/{manifold}_{diff}_test_runtime',  ylabels=ylabel_methods, xlabels=xlabel_samples, fmt='.5g', cm=cm, vmax=np.max(test_runtime_mat))

        except Exception as e:
            logger.error(f"Save results error with: \n{traceback.format_exc()}")
