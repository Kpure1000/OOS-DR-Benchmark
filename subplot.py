from datetime import datetime
import os
from exp_plot import *
import pandas as pd

def subplot_structure(data, methods, path, large_is_best):

    structures = ["plane", "roll", "hybrid"]
    
    # 定义颜色列表
    colors = ['#ffffff', '#bcbcbc']
    # 创建自定义的离散色图
    custom_cmap = mcolors.ListedColormap(colors)

    fig_size = (8, 2)

    fig = plt.figure(figsize=fig_size)

    # data = np.random.rand(len(structures), len(methods)) * 1

    highlight_table(data, structures, methods, large_is_best=large_is_best, show_xlabel=False, cmap=custom_cmap)

    fig.savefig(path, transparent=True)
    
    matplotlib.pyplot.close()



def subplot_dist(data, methods, path):

    fig_size = (8, 2)

    fig = plt.figure(figsize=fig_size)

    # data = np.random.rand(len(structures), len(methods)) * 1

    diff_bar(data, methods, show_xlabel=False, show_legend=False)

    fig.savefig(path, transparent=True)
    
    matplotlib.pyplot.close()


def subplot_proportion(data, methods, path):

    proportions = ["0.9", "0.7", "0.5", "0.3"]
    
    # 定义颜色列表
    # colors = ['#ebf0f5', '#71aacc', '#597cab'] legacy
    # colors = ['#ebf0f5', '#91bcd6', '#6694bc'] legacy2
    # colors = ['#ebf0f5', '#accbdf', '#7ca8c9'] legacy3
    colors = ['#fafafa', '#dbdbdb', '#bdbdbd'] # 400
    n_bins = 200
    # 创建自定义的离散色图
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("my_cm", colors, N=n_bins)

    fig_size = (8, 2)

    fig = plt.figure(figsize=fig_size)

    # data = np.random.rand(len(proportions), len(methods)) * 100.0

    proportion_heatmap(data, row_labels=proportions, col_labels=methods, show_xlabel=False, show_ylabel=True, cmap=custom_cmap)

    fig.savefig(path, transparent=True)
    
    matplotlib.pyplot.close()



if __name__ == '__main__':

    methods = [
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

    metrics = [
        ('t',       True),
        ('c',       True),
        ('nh',      True),
        ('ale',     False),
        ('ns',      False),
        ('shp',     True),
        ('dsc',     True),
        ('sil',     True),
        ('acc_train',   True),
        ('acc_test',   True),
    ]

    manifolds = ['plane', 'swissroll', 'hybrid']
    diffs = ['structure' 'dist' 'prop']

    save_time = datetime.now()
    save_time = save_time.strftime("%Y.%m.%d-%H.%M.%S")
    os.makedirs(f'./imgs/subplots/{save_time}/structure', exist_ok=True)
    os.makedirs(f'./imgs/subplots/{save_time}/prop', exist_ok=True)
    os.makedirs(f'./imgs/subplots/{save_time}/dist', exist_ok=True)

    res = pd.read_csv('./results/synth/res2.csv')

    # sturcture
    res_structure = res[res['diff'] == 'structure']
    print(pd.unique(res_structure['manifold']))
    for metric_idx, (metric, large_is_best) in enumerate(metrics):
        res_metric = res_structure[res_structure['metric'] == metric]
        if (len(res_metric) == 0):
            print(f"sturcture result for '{metric}' is not found")
            continue
        res_data = np.zeros((len(manifolds), len(methods)))
        mask = np.zeros((len(manifolds), len(methods)), dtype=bool)
        for method_idx, method in enumerate(methods):
            res_method = res_metric[res_metric['method'] == method]
            if (len(res_method) == 0):
                print(f"sturcture result for '{method}'-'{metric}' is not found")
                continue
            for manifold_idx, manifold in enumerate(manifolds):
                res_manifold = res_method[res_method['manifold'] == manifold]
                if (len(res_manifold) == 0):
                    print(f"sturcture result for '{method}'-'{metric}'-'{manifold_idx}' is not found")
                    continue
                res_data[manifold_idx, method_idx] = float(res_manifold['result'])
                mask[manifold_idx, method_idx] = True
        
            mask_data = np.ma.masked_where(mask == False, res_data)

            subplot_structure(data=mask_data, methods=methods, path=f'./imgs/subplots/{save_time}/structure/{metric_idx + 1}-{metric}.svg', large_is_best=large_is_best)

    # dist
    # n_dist_stage = 2
    # res_dist = res[res['diff'] == 'dist']
    # for metric_idx, (metric, large_is_best) in enumerate(metrics):
    #     res_metric = res_dist[res_dist['metric'] == metric]
    #     if (len(res_metric) == 0):
    #         print(f"dist result for '{metric}' is not found")
    #         continue

    #     for manifold_idx, manifold in enumerate(manifolds):
    #         res_manifold = res_metric[res_metric['manifold'] == manifold]
    #         if (len(res_manifold) == 0):
    #             print(f"dist result for '{method}'-'{metric}'-'{manifold}' is not found")
    #             continue
        
    #         res_data = np.zeros((n_dist_stage, len(methods)))
    #         mask = np.zeros((n_dist_stage, len(methods)), dtype=bool)

    #         for method_idx, method in enumerate(methods):
    #             res_method = res_manifold[res_manifold['method'] == method]
    #             if (len(res_method) == 0):
    #                 print(f"dist result for '{method}'-'{metric}' is not found")
    #                 continue

    #             for dist_idx in range(n_dist_stage):
    #                 res_dist_stage = res_method[res_method['stage'] == dist_idx]
    #                 if (len(res_dist_stage) == 0):
    #                     print(f"dist result for '{method}'-'{metric}'='{manifold}'-'{dist_idx}' is not found")
    #                     continue
    #                 res_data[dist_idx, method_idx] = float(res_dist_stage['result'])
    #                 mask[dist_idx, method_idx] = True
            
    #         mask_data = np.ma.masked_where(mask == False, res_data)

    #         subplot_dist(data=mask_data, methods=methods, path=f'./imgs/subplots/{save_time}/dist/{manifold}-{metric_idx + 1}-{metric}.svg')

    # prop
    # n_prop_stage = 4
    # res_prop = res[res['diff'] == 'prop']
    # for metric_idx, (metric, large_is_best) in enumerate(metrics):
    #     res_metric = res_prop[res_prop['metric'] == metric]
    #     if (len(res_metric) == 0):
    #         print(f"prop result for '{metric}' is not found")
    #         continue

    #     for manifold_idx, manifold in enumerate(manifolds):
    #         res_manifold = res_metric[res_metric['manifold'] == manifold]
    #         if (len(res_manifold) == 0):
    #             print(f"prop result for '{method}'-'{metric}'-'{manifold}' is not found")
    #             continue
        
    #         res_data = np.zeros((n_prop_stage, len(methods)))
    #         mask = np.zeros((n_prop_stage, len(methods)), dtype=bool)

    #         for method_idx, method in enumerate(methods):
    #             res_method = res_manifold[res_manifold['method'] == method]
    #             if (len(res_method) == 0):
    #                 print(f"prop result for '{method}'-'{metric}' is not found")
    #                 continue

    #             for prop_idx in range(n_prop_stage):
    #                 res_prop_stage = res_method[res_method['stage'] == prop_idx]
    #                 if (len(res_prop_stage) == 0):
    #                     print(f"prop result for '{method}'-'{metric}'='{manifold}'-'{prop_idx}' is not found")
    #                     continue
    #                 res_data[prop_idx, method_idx] = float(res_prop_stage['result'])
    #                 mask[prop_idx, method_idx] = True
            
    #         mask_data = np.ma.masked_where(mask == False, res_data)

    #         subplot_proportion(data=mask_data, methods=methods, path=f'./imgs/subplots/{save_time}/prop/{manifold}-{metric_idx + 1}-{metric}.svg')


    # subplot_dist(methods)
    # subplot_proportion(None, methods)
