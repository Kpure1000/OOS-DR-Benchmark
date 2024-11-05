import matplotlib.pyplot as plt
from matplotlib.pyplot import Axes 
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import matplotlib
import matplotlib.colors as mcolors

def highlight_table(data, row_labels, col_labels, large_is_best=True, show_xlabel=True, show_ylabel=True, ax=None, cbar_kw=None, cbarlabel="", fmt='{x:.2g}', **kwargs):
    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    if large_is_best:
        best_idxs = np.argmax(data, axis=1)
    else:
        best_idxs = np.argmin(data, axis=1)
    data_re = np.zeros_like(data)
    data_re[np.arange(data.shape[0]), best_idxs] = 1

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar=None
    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels if show_xlabel else [], fontsize=6)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels if show_ylabel else [], fontsize=8)

    # Let the horizontal axes labeling appear on top.
    # ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    if show_xlabel:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.1)
    ax.tick_params(which="minor", bottom=False, left=False)

    tdata = data

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center", fontsize=9)
    # kw.update(text_kw)

    if isinstance(fmt, str):
        fmt = matplotlib.ticker.StrMethodFormatter(fmt)

    for i in range(tdata.shape[0]):
        for j in range(tdata.shape[1]):
            d = im.norm(tdata[i, j])
            kw.update(color='black')
            val = tdata[i, j]
            im.axes.text(j, i, f'{val:.2f}' if abs(val) < 10 else f'{val:.1e}', **kw)


def diff_bar(data, col_labels, show_xlabel=True, show_legend=False, ax:Axes=None, cbar_kw=None, cbarlabel="", ymin=0, **kwargs):
    if ax is None:
        ax = plt.gca()

    ax.spines[:].set_visible(False)

    total_min = np.min(data)
    total_max = np.max(data)
    total_delta = total_max - total_min
    middle = (data[1, :] + data[0, :]) * 0.5
    delta = data[1, :] - data[0, :]
    x=np.arange(data.shape[1])

    ax.margins(y=0.1)

    # ax.set_ylim(total_min - total_delta * 0.12, total_max + total_delta * 0.12)

    kw = dict(horizontalalignment="center", verticalalignment="center", fontsize=7.5) # plot label 大小

    # By using ``transform=vax.get_xaxis_transform()`` the y coordinates are scaled
    # such that 0 maps to the bottom of the Axes and 1 to the top.

    # background bar
    ax.vlines(x, 0, 1, transform=ax.get_xaxis_transform(), linewidth=14, color="#f2f2f2")

    # y baseline
    # ax.axhline(ymin, 0, 1, linewidth=1, color="#a2a2a2", linestyle='dashed')

    # delta line
    ax.vlines(x, data[0, :], data[1, :], linewidth=1, linestyles='dashed', color="gray", alpha=0.5)

    raw_color = "#d77281"
    oos_color = "#354050"
    delta_color = '#727272'

    # raw value
    ax.plot(x, data[0,:], 'o', color=raw_color, markersize=4)
    # oos value
    ax.plot(x, data[1,:], '^', color=oos_color, markersize=4)

    # label_fmt = matplotlib.ticker.StrMethodFormatter('{x:.2f}')
    # # up&down labels
    # for j,xpos in enumerate(x):
    #     inv = data[0,j] > data[1,j] # idx0 is greater
    #     yoffset_up = 0.07 * (total_max - total_min)
    #     yoffset_down = -0.08 * (total_max - total_min)
    #     # raw
    #     kw.update(color=raw_color, fontweight="semibold", horizontalalignment="center")
    #     ax.text(xpos, data[0,j] + (yoffset_up if inv else yoffset_down), label_fmt(data[0,j], None), **kw)
    #     # oos
    #     kw.update(color=oos_color)
    #     ax.text(xpos, data[1,j] + (yoffset_down if inv else yoffset_up), label_fmt(data[1,j], None), **kw)

    # delta value
    offset_x = 0
    # ax.plot(x + offset_x, middle, "_", color=delta_color, markersize=4)
    
    fmt_delta = matplotlib.ticker.StrMethodFormatter('{x:.2f}')

    # # delta label
    # offset_x_2 = 0.12
    # for j,xpos in enumerate(x):
    #     kw.update(color=delta_color, fontweight="heavy", horizontalalignment="left")
    #     d = delta[j]
    #     ax.text(xpos + offset_x + offset_x_2, middle[j] , f"{'-' if d < 0 else '+'}{fmt_delta(abs(d), None)}", **kw)


    if show_legend:
        ax.legend(loc='upper left', ncols=3)

    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels if show_xlabel else [], minor=False)
    # ax.set_yticks(np.arange(data.shape[0]), minor=True)

    ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

    ax.tick_params(axis='x', which='major', labelsize=8)
    ax.tick_params(axis='y', which='major', labelsize=9) # y轴label大小

    ax.set_ylim(total_min - total_delta * 0.12, total_max + total_delta * 0.12)

    if show_xlabel:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")


def proportion_heatmap(data, row_labels, col_labels, show_xlabel=True, show_ylabel=True, ax=None, cbar_kw=None, cbarlabel="", fmt="{x:.2f}", **kwargs):
    if ax is None:
        ax = plt.gca()

    if cbar_kw is None:
        cbar_kw = {}

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    cbar=None
    # cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    # cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")

    # Show all ticks and label them with the respective list entries.
    ax.set_xticks(np.arange(data.shape[1]), labels=col_labels if show_xlabel else [], fontsize=9)
    ax.set_yticks(np.arange(data.shape[0]), labels=row_labels if show_ylabel else [], fontsize=9)

    # Let the horizontal axes labeling appear on top.
    # ax.tick_params(top=True, bottom=False, labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    if show_xlabel:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right", rotation_mode="anchor")

    # Turn spines off and create white grid.
    ax.spines[:].set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    # ax.grid(which="minor", color="gray", linestyle='-', linewidth=0.1)
    ax.tick_params(which="minor", bottom=False, left=False)

    tdata = data

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center", verticalalignment="center", fontsize=9) # 内部label
    # kw.update(text_kw)

    if isinstance(fmt, str):
        fmt = matplotlib.ticker.StrMethodFormatter(fmt)

    texts = []
    for i in range(tdata.shape[0]):
        for j in range(tdata.shape[1]):
            d = im.norm(tdata[i, j])
            kw.update(color='black')
            text = im.axes.text(j, i, fmt(tdata[i, j], None), **kw)
            texts.append(text)


def structure_plot():

    structures = ["plane", "roll", "hybrid"]
    methods = ["PCA", "AE", "CDR", "PUMAP", "PTSNE", "PTSNE22", "DLMP-TSNE", "DLMP-UMAP", "K-TSNE", "OOS-ISO", "K-ISO", "L-ISO", "OOS-MDS", "K-MDS", "L-MDS", "MI-MDS" ]

    # gridspec inside gridspec
    fig = plt.figure(layout='constrained', figsize=(13, 11))
    subfigs = fig.subfigures(6, 1, hspace=0, height_ratios=[1,1,1,1,1,1.1])

    # 定义颜色列表
    colors = ['#ffffff', '#bcbcbc']
    # 创建自定义的离散色图
    custom_cmap = mcolors.ListedColormap(colors)

    cmap = custom_cmap

    subfig = subfigs[0]
    subfig.set_facecolor('#d9e7f4')
    ax = subfig.subplots(1, 2)
    highlight_table(np.random.rand(len(structures), len(methods)), structures, methods, show_xlabel = False, show_ylabel = True , ax=ax[0], cmap=cmap   )
    highlight_table(np.random.rand(len(structures), len(methods)), structures, methods, show_xlabel = False, show_ylabel = False, ax=ax[1], cmap=cmap   )
    ax[0].set_title("trustworthiness")
    ax[1].set_title("continuity")

    subfig = subfigs[1]
    subfig.set_facecolor('#d9e7f4')
    ax = subfig.subplots(1, 2)
    highlight_table(5 * np.random.rand(len(structures), len(methods)), structures, methods, show_xlabel = False, show_ylabel = True , ax=ax[0], cmap=cmap   )
    highlight_table(np.random.rand(len(structures), len(methods)), structures, methods, show_xlabel = False, show_ylabel = False, ax=ax[1], cmap=cmap   )
    ax[0].set_title("normalized stress")
    ax[1].set_title("shepard diagrams")


    subfig = subfigs[2]
    subfig.set_facecolor('#d9e7f4')
    ax = subfig.subplots(1, 2)
    highlight_table(np.random.rand(len(structures), len(methods)), structures, methods, show_xlabel = False, show_ylabel = True , ax=ax[0], cmap=cmap   )
    ax[0].set_title("neighborhood hit")
    ax[1].axis('off')


    subfig = subfigs[3]
    subfig.set_facecolor('#f4ccd3')
    ax = subfig.subplots(1, 2)
    highlight_table(np.random.rand(len(structures), len(methods)), structures, methods, show_xlabel = False, show_ylabel = True, ax=ax[0], cmap=cmap)
    highlight_table(np.random.rand(len(structures), len(methods)), structures, methods, show_xlabel = False, show_ylabel = False, ax=ax[1], cmap=cmap)
    ax[0].set_title("area under curve")
    ax[1].set_title("average local error")


    subfig = subfigs[4]
    subfig.set_facecolor('#d1ddcc')
    ax = subfig.subplots(1, 2)
    highlight_table(np.random.rand(len(structures), len(methods)), structures, methods, show_xlabel = False, show_ylabel = True, ax=ax[0], cmap=cmap)
    highlight_table(2 * np.random.rand(len(structures), len(methods)) - 1, structures, methods, show_xlabel = False, show_ylabel = False, ax=ax[1], cmap=cmap)
    ax[0].set_title("distance consistency")
    ax[1].set_title("silhouette correlation")


    subfig = subfigs[5]
    subfig.set_facecolor('#fae4d5')
    ax = subfig.subplots(1, 2)
    highlight_table(np.random.rand(len(structures), len(methods)), structures, methods, show_xlabel = True, show_ylabel = True,  ax=ax[0], cmap=cmap)
    highlight_table(np.random.rand(len(structures), len(methods)), structures, methods, show_xlabel = True, show_ylabel = False, ax=ax[1], cmap=cmap)
    ax[0].set_title("accuracy")
    ax[1].set_title("overfitting")


    plt.show()


def distribution_plot():

    methods = ["PCA", "AE", "CDR", "PUMAP", "PTSNE", "PTSNE22", "DLMP-TSNE", "DLMP-UMAP", "K-TSNE", "OOS-ISO", "K-ISO", "L-ISO", "OOS-MDS", "K-MDS", "L-MDS", "MI-MDS" ]

    def random_data():
        rd_data = np.ones((2, len(methods)))
        rd_start = np.random.rand(len(methods))
        rd_delta = (2 * np.random.rand(len(methods)) - 1) * 0.1
        rd_data[0, :] = rd_start + rd_delta
        rd_data[1, :] = rd_data[0, :] + rd_delta
        return rd_data

    # gridspec inside gridspec
    fig = plt.figure(layout='constrained', figsize=(13, 11))
    subfigs = fig.subfigures(6, 1, hspace=0, height_ratios=[1,1,1,1,1,1.2])

    # 定义颜色列表
    colors = ['#ffffff', '#bcbcbc']
    # 创建自定义的离散色图
    custom_cmap = mcolors.ListedColormap(colors)

    cmap = custom_cmap

    subfig = subfigs[0]
    subfig.set_facecolor('#d9e7f4')
    ax = subfig.subplots(1, 2)
    diff_bar(random_data(), col_labels=methods, show_xlabel=False, ax=ax[0], cmap=cmap   )
    diff_bar(random_data(), col_labels=methods, show_xlabel=False, ax=ax[1], cmap=cmap   )
    ax[0].set_title("trustworthiness")
    ax[1].set_title("continuity")

    subfig = subfigs[1]
    subfig.set_facecolor('#d9e7f4')
    ax = subfig.subplots(1, 2)
    diff_bar(200 * random_data(), col_labels=methods, show_xlabel=False, ax=ax[0], cmap=cmap ,ymin=0  )
    diff_bar(random_data(), col_labels=methods, show_xlabel=False, ax=ax[1], cmap=cmap   )
    ax[0].set_title("normalized stress")
    ax[1].set_title("shepard diagrams")


    subfig = subfigs[2]
    subfig.set_facecolor('#d9e7f4')
    # ax = subfig.subplots(1, 2)
    ax = subfig.subplot_mosaic("AX", width_ratios=[1, 1.07], empty_sentinel="X")
    diff_bar(random_data(), col_labels=methods, show_xlabel=False, ax=ax['A'], cmap=cmap   )
    ax['A'].set_title("neighborhood hit")
    # ax[1].axis('off')


    subfig = subfigs[3]
    subfig.set_facecolor('#f4ccd3')
    ax = subfig.subplots(1, 2)
    diff_bar(random_data(), col_labels=methods, show_xlabel=False, ax=ax[0], cmap=cmap   )
    diff_bar(random_data(), col_labels=methods, show_xlabel=False, ax=ax[1], cmap=cmap   )
    ax[0].set_title("area under curve")
    ax[1].set_title("average local error")


    subfig = subfigs[4]
    subfig.set_facecolor('#d1ddcc')
    ax = subfig.subplots(1, 2)
    diff_bar(random_data(), col_labels=methods, show_xlabel=False, ax=ax[0], cmap=cmap   )
    diff_bar(2 * random_data() - 1, col_labels=methods, show_xlabel=False, ax=ax[1], cmap=cmap, ymin=-1  )
    ax[0].set_title("distance consistency")
    ax[1].set_title("silhouette correlation")


    subfig = subfigs[5]
    subfig.set_facecolor('#fae4d5')
    ax = subfig.subplots(1, 2)
    diff_bar(random_data(), col_labels=methods, show_xlabel=True, ax=ax[0], cmap=cmap   )
    diff_bar(random_data(), col_labels=methods, show_xlabel=True, ax=ax[1], cmap=cmap   )
    ax[0].set_title("accuracy")
    ax[1].set_title("overfitting")


    plt.show()


def proportion_plot():

    proportions = ["0.9", "0.7", "0.5", "0.3"]
    methods = ["PCA", "AE", "CDR", "PUMAP", "PTSNE", "PTSNE22", "DLMP-TSNE", "DLMP-UMAP", "K-TSNE", "OOS-ISO", "K-ISO", "L-ISO", "OOS-MDS", "K-MDS", "L-MDS", "MI-MDS" ]

    def random_data():
        rd_data = np.ones((len(proportions), len(methods)))
        rd_start = np.random.rand(len(methods))
        rd_delta = (2 * np.random.rand(len(methods)) - 1) * 0.1
        rd_data[0, :] = rd_start + rd_delta
        for i in range(1, len(proportions)):
            rd_data[i, :] = rd_data[i-1, :] + rd_delta

        return rd_data


    # gridspec inside gridspec
    fig = plt.figure(layout='constrained', figsize=(10, 13))
    subfigs = fig.subfigures(6, 1, hspace=0, height_ratios=[1,1,1,1,1,1.4])

    # 定义颜色列表
    colors = ['#ebf0f5', '#71aacc', '#597cab']
    n_bins = 200
    # 创建自定义的离散色图
    custom_cmap = mcolors.LinearSegmentedColormap.from_list("my_cm", colors, N=n_bins)

    cmap = custom_cmap

    subfig = subfigs[0]
    subfig.set_facecolor('#d9e7f4')
    ax = subfig.subplots(1, 2)
    proportion_heatmap(random_data(), row_labels=proportions, col_labels=methods, show_xlabel=False, show_ylabel=True,  ax=ax[0], cmap=cmap   )
    proportion_heatmap(random_data(), row_labels=proportions, col_labels=methods, show_xlabel=False, show_ylabel=False, ax=ax[1], cmap=cmap   )
    ax[0].set_title("trustworthiness")
    ax[1].set_title("continuity")

    subfig = subfigs[1]
    subfig.set_facecolor('#d9e7f4')
    ax = subfig.subplots(1, 2)
    proportion_heatmap(4*random_data(), row_labels=proportions, col_labels=methods, show_xlabel=False, show_ylabel=True , ax=ax[0], cmap=cmap   )
    proportion_heatmap(random_data(), row_labels=proportions, col_labels=methods, show_xlabel=False, show_ylabel=False, ax=ax[1], cmap=cmap   )
    ax[0].set_title("normalized stress")
    ax[1].set_title("shepard diagrams")


    subfig = subfigs[2]
    subfig.set_facecolor('#d9e7f4')
    # ax = subfig.subplots(1, 2)
    ax = subfig.subplot_mosaic("AX", width_ratios=[1, 1], empty_sentinel="X")
    proportion_heatmap(random_data(), row_labels=proportions, col_labels=methods, show_xlabel=False, show_ylabel=True, ax=ax['A'], cmap=cmap   )
    ax['A'].set_title("neighborhood hit")
    # ax[1].axis('off')


    subfig = subfigs[3]
    subfig.set_facecolor('#f4ccd3')
    ax = subfig.subplots(1, 2)
    proportion_heatmap(random_data(), row_labels=proportions, col_labels=methods, show_xlabel=False, show_ylabel=True, ax=ax[0], cmap=cmap   )
    proportion_heatmap(random_data(), row_labels=proportions, col_labels=methods, show_xlabel=False, show_ylabel=False, ax=ax[1], cmap=cmap   )
    ax[0].set_title("area under curve")
    ax[1].set_title("average local error")


    subfig = subfigs[4]
    subfig.set_facecolor('#d1ddcc')
    ax = subfig.subplots(1, 2)
    proportion_heatmap(random_data(), row_labels=proportions, col_labels=methods, show_xlabel=False, show_ylabel=True, ax=ax[0], cmap=cmap   )
    proportion_heatmap(2*random_data()-1, row_labels=proportions, col_labels=methods, show_xlabel=False, show_ylabel=False, ax=ax[1], cmap=cmap   )
    ax[0].set_title("distance consistency")
    ax[1].set_title("silhouette correlation")


    subfig = subfigs[5]
    subfig.set_facecolor('#fae4d5')
    ax = subfig.subplots(1, 2)
    proportion_heatmap(random_data(), row_labels=proportions, col_labels=methods, show_xlabel=True, show_ylabel=True, ax=ax[0], cmap=cmap   )
    proportion_heatmap(random_data(), row_labels=proportions, col_labels=methods, show_xlabel=True, show_ylabel=False, ax=ax[1], cmap=cmap   )
    ax[0].set_title("accuracy")
    ax[1].set_title("overfitting")


    plt.show()

if __name__ == '__main__':
    # structure_plot()
    distribution_plot()
    # proportion_plot()
