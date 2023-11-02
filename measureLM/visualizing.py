import matplotlib as mpl
import seaborn as sns

import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from measureLM import helpers

from sklearn.decomposition import PCA

matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False


def plot_synth_data(df, sort_by="good-bad"):

    ## preprocessing__________________
    df = df.sort_values(f"bias {sort_by}")
    df = df.reset_index(drop=True)

    ent1, ent2, labels = df["ent1"].to_list(), df["ent2"].to_list(), df["label"].to_list()
    ent1_2 = [f"{e1}–{e2}" for e1, e2 in list(zip(ent1, ent2))]
    x_vals = np.arange(0, len(ent1_2))

    # draw plot_____________________
    titlefont = 20
    labelfont = 14

    fig, ax = plt.subplots(1, 1, figsize=(20, 3), gridspec_kw={'hspace': 0.4})

    vals = df[f"bias {sort_by}"].to_numpy()
    prior_scatter = ax.scatter(x_vals, vals, s=200, alpha=1.0, c=vals, cmap=cm.coolwarm_r)

    pos_vals = df[f"pos {sort_by}"].to_numpy()
    pos_scatter = ax.scatter(x_vals, pos_vals, s=100, alpha=0.8, marker="v", color="blue")

    neg_vals = df[f"neg {sort_by}"].to_numpy()
    neg_scatter = ax.scatter(x_vals, neg_vals, s=100, alpha=0.8, marker="^", color="red")

    ax.hlines(y=vals.mean(), xmin=x_vals.min() - 1, xmax=x_vals.max() + 1, linewidth=2, linestyle='--', color='grey')
    # for x, y in zip(x_vals, vals):
    # t = ax.text(x, y, round(y, 1), horizontalalignment='center',
    # verticalalignment='center', fontdict={'color':'white'})

    ax.xaxis.set_ticks(x_vals)
    ax.tick_params(axis='both', which='major', labelsize=labelfont)
    ax.set_xticklabels(ent1_2, fontsize=labelfont, rotation=90)
    ax.set_ylim(vals.min() - 0.05, vals.max() + 0.05)
    ax.set_xlim(-0.5)
    # ax.set_title(scale_name, fontsize=titlefont, color="black", loc='center')
    ax.set_ylabel(f"{sort_by} scale", fontsize=labelfont)

    for i, x_tick_label in enumerate(ax.get_xticklabels()):
        label = labels[i]
        if label == "enemy":
            label_name = "E"
            color = "red"
        elif label == "friend":
            color = "blue"
            label_name = "F"
        position = x_tick_label.get_position()
        ax.text(position[0] - 0.33, 0.435, label_name, fontsize=labelfont, color=color, verticalalignment='top')
        # x_tick_label.set_color(color)
        x_tick_label.set_y(-.1)

    prior_scatter = Line2D([0], [0], label='The relationship between A and B is', marker='.', markersize=22,color='grey', linestyle='')
    pos_scatter = Line2D([0], [0], label='prepended context: A loves B.', marker='v', markersize=10, color='blue',linestyle='')
    neg_scatter = Line2D([0], [0], label='prepended context: A hates B.', marker='^', markersize=10, color='red',linestyle='')

    # add manual symbols to auto legend
    handles, labels = plt.gca().get_legend_handles_labels()
    handles.extend([prior_scatter, pos_scatter, neg_scatter])

    plt.legend(handles=handles, ncol=len(handles), prop={'size': labelfont}, facecolor='white', framealpha=0,loc='upper left', bbox_to_anchor=(0.0, 1.1))
    plt.show()
    #fig.savefig(helpers.ROOT_DIR / "results" / "plots" / f"{sort_by}.pdf", bbox_inches='tight', dpi=200,transparent=True)


def dim_reduction(embs, reduction="pca"):
    x = embs.view(-1, embs.shape[-1]).cpu().detach().numpy()

    y_len = int((embs.shape[0] * embs.shape[1]) / 2)
    y = ([0] * y_len + [1] * y_len)
    colormap = np.array(['g', 'r'])
    y = colormap[y]

    if reduction == "pca":
        pca = PCA(n_components=2)
        x_2D = pca.fit_transform(x)

    fig, (ax) = plt.subplots(1, figsize=(5, 5), gridspec_kw={'hspace': 0.40})
    ax.scatter(x_2D[:, 0], x_2D[:, 1], c=y)
    plt.show()


def plot_heatmap(array, title='Patching Effect', xticklabels=["attn_out", "mlp_out"], cmap="binary"):
    titlefont, labelsize = 12, 10
    array_abs_max = np.max(np.abs(array))
    fig, ax = plt.subplots(1, 1, figsize=(2, 4), gridspec_kw={'hspace': 0.4})
    #ax = sns.heatmap(array, vmin=-array_abs_max, center=0, vmax=array_abs_max, cmap=mpl.colormaps[cmap], xticklabels=xticklabels, square=False)
    ax = sns.heatmap(array, cmap=mpl.colormaps[cmap], xticklabels=xticklabels, square=False)
    ax.set_title(title, fontsize=titlefont, color="black", loc='center', y=1.22)
    ax.set_ylabel('layers', fontsize=labelsize)

    mean_effect = list(map(lambda x: "%.3f" % x, list(array.mean(0))))
    max_effect = list(map(lambda x: "%.3f" % x, list(array.max(0))))
    #min_effect = list(map(lambda x: "%.3f" % x, list(array.min(0))))
    for i, x_tick_label in enumerate(ax.get_xticklabels()):
        ax.text(x_tick_label.get_position()[0] - 0.5, -0.5, f"max:\n{max_effect[i]}", fontsize=labelsize,
        color="black", verticalalignment='bottom')
        ax.text(x_tick_label.get_position()[0] - 0.5, -3.0, f"mean:\n{mean_effect[i]}", fontsize=labelsize,
        color="black", verticalalignment='bottom')
        #ax.text(x_tick_label.get_position()[0] - 0.5, -0.2, f"min:\n{min_effect[i]}", fontsize=labelsize,
        #color="black", verticalalignment='bottom')
    plt.show()


def plot_2D_scatter(X_trans, y):
    fig, (ax) = plt.subplots(1, figsize=(5, 5), gridspec_kw={'hspace': 0.40})
    ax.scatter(X_trans[:, 0], X_trans[:, 1], c=y)
    plt.show()