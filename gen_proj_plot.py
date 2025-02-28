from glob import glob
import json
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes
from matplotlib.pyplot import Normalize
import numpy as np
from tqdm import tqdm
from utils import load_projection, load_dataset, load_datasets
import os
import pandas as pd
from collections import defaultdict
import yaml
import h5py as h5


def subplot_projection(ax: Axes,
                       method_name,
                       train_set_path,
                       test_set_path,
                       label_train,
                       label_test,
                       alpha_train,
                       alpha_test,
                       size_train,
                       size_test,
                       cmap,
                       show_label=False,
                       show_train=True,
                       colornorm=None,
                       **other):

    proj_train = load_projection(train_set_path)

    norm = None if colornorm is None else Normalize(vmin=colornorm[0], vmax=colornorm[1])

    if show_train:
        ax.scatter(proj_train[:,0], proj_train[:,1], norm=norm, c=label_train, cmap=cmap if show_label else'gray', s=size_train, marker='o', alpha=alpha_train)

    proj_test = load_projection(test_set_path)

    ax.scatter(proj_test[:,0], proj_test[:,1], norm=norm, c=label_test, cmap=cmap, s=size_test, marker='o', alpha=alpha_test)

    ax.set_xticks([])
    ax.set_yticks([])

    ax.set_box_aspect(1)

    for spine in ax.spines:
        ax.spines[spine].set_alpha(0.2)

    ax.set_title(method_name.upper(), y=1.0, loc='right', pad=-13, fontsize=16, alpha=0.8)


def plot_projection(save_params,
                    methods,
                    proj_train,
                    proj_test,
                    label_train,
                    label_test,
                    n_rows,
                    n_cols,
                    title,
                    show_label,
                    show_train,
                    **plot_args):

    fig = plt.figure(figsize=(n_cols * 4, n_rows * 4))

    fig.suptitle(title, verticalalignment='bottom', y=0.89)

    axs = fig.subplots(nrows=n_rows, ncols=n_cols, gridspec_kw=dict(wspace=0.03, hspace=0.03))

    for idx in range(n_rows * n_cols):
        ax = axs[idx // n_cols, idx % n_cols] if n_rows > 1 else axs[idx] if n_cols > 1 else axs

        if idx >= len(methods):
            ax.set_visible(False)
            continue

        method_name = methods[idx]

        if method_name not in proj_train or method_name not in proj_test:
            ax.set_visible(False)
            print(f"Projections of '{method_name}' is lack")
            continue

        subplot_projection(ax,
                           method_name,
                           proj_train[method_name],
                           proj_test[method_name],
                           label_train=label_train,
                           label_test=label_test,
                           show_label=show_label,
                           show_train=show_train,
                           **plot_args)

    [fig.savefig(param[0], **param[1],) for param in save_params]

    matplotlib.pyplot.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--type', type=str, default='truth')
    parser.add_argument('-i', '--input', type=str, required=True)
    parser.add_argument('-n', '--nrows', type=int, default=4)
    parser.add_argument('-c', '--cover', action='store_true', required=False, help="Cover plots if exist")
    parser.add_argument('--hide-label', action='store_false', required=False, help="Hide train set label")
    parser.add_argument('--show-train', action='store_true', required=False, help="Show train set")

    args = parser.parse_args()

    # load data
    data_type = args.type
    if args.type == 'truth':
        result_path = f"results/truth/{args.input}"
    elif args.type == 'synth':
        result_path = f"results/synth/{args.input}"
    elif args.type == 'runtime':
        result_path = f"results/runtime/{args.input}"
    else:
        raise ValueError('type must be "truth", "synth" or "runtime"')

    if not os.path.exists(result_path):
        raise OSError(f"Result path '{result_path}' does not exist")

    cover = args.cover

    os.makedirs(f"imgs/{args.type}/", exist_ok=True)

    with open("plot.yml", 'r', encoding='utf-8') as f:
        plot_config = yaml.load(f, Loader=yaml.FullLoader)[data_type]

    with open('benchmark.yml', 'r', encoding='utf-8') as f:
        bench_conf = yaml.load(f, Loader=yaml.FullLoader)
        datasets = bench_conf['datasets'][data_type]
        methods  = bench_conf['methods']

    # plot param
    n_rows = int(args.nrows)
    n_cols = int(np.ceil(len(methods) / n_rows))
    show_label = args.hide_label
    show_train = args.show_train

    # save path
    save_path = f"imgs/{args.type}/{args.input}/"
    os.makedirs(save_path, exist_ok=True)
    
    save_path = os.path.join(save_path, f'{n_rows}x{n_cols}')
    
    if os.path.exists(save_path):
        if not cover:
            raise OSError(f"Plot already exist in '{save_path}', use -c to cover")
    
    os.makedirs(save_path, exist_ok=True)

    # check projections info
    proj_path = os.path.join(result_path, 'projections')

    if not os.path.exists(proj_path):
        raise OSError(f"Projections path '{proj_path}' does not exist")

    proj_files = glob(os.path.join(proj_path, '*.csv'))

    projections = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: {})))

    for filename in proj_files:
        proj_file_abs = os.path.abspath(filename)
        root, _ = os.path.splitext(os.path.basename(filename))

        if data_type == "truth" or data_type == "runtime":
            [dataset_name, method_name, stage_idx, batch_name] = root.split('_')
        elif data_type == "synth":
            [dataset_name, diff_name, method_name, stage_idx, batch_name] = root.split('_')
            dataset_name = dataset_name + '_' + diff_name

        projections[dataset_name][stage_idx][batch_name][method_name] = proj_file_abs

    for dataset_name in tqdm(projections):
        if dataset_name not in plot_config:
            continue
        for stage_idx in projections[dataset_name]:
            
            print(f"Generate plot for '{dataset_name}' Stage.{stage_idx}")

            plot_args = plot_config[dataset_name]
            label_name = plot_args['label_name']

            if data_type == "truth":
                n_samples = [item for item in datasets if item[0] == dataset_name][0][1]
                with h5.File(f"datasets/truth/{dataset_name}_{n_samples}.h5") as f:
                    label_train = f["E"][f"{label_name}{stage_idx}"][:]
                    label_test  = f["O"][f"{label_name}{stage_idx}"][:]
            elif data_type == "synth":
                with h5.File(f"datasets/synth/{dataset_name}.h5") as f:
                    label_train = f["E"][f"{label_name}{stage_idx}"][:]
                    label_test  = f["O"][f"{label_name}{stage_idx}"][:]
            elif data_type == "runtime":
                with h5.File(f"datasets/runtime/{dataset_name}.h5") as f:
                    label_train = f["E"][f"{label_name}{stage_idx}"][:]
                    label_test  = f["O"][f"{label_name}{stage_idx}"][:]

            save_params = [(f'{save_path}/{dataset_name}_{stage_idx}.png', {'dpi':400, 'bbox_inches':'tight'})]

            plot_projection(save_params,
                            methods,
                            projections[dataset_name][stage_idx]['train'],
                            projections[dataset_name][stage_idx]['test'],
                            label_train=label_train,
                            label_test=label_test,
                            title=f"{dataset_name}_{stage_idx}",
                            n_rows=n_rows,
                            n_cols=n_cols,
                            show_label=show_label,
                            show_train=show_train,
                            **plot_args)
