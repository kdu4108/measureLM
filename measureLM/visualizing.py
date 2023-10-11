import torch
import numpy as np
import matplotlib
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from measureLM import helpers

matplotlib.rcParams['axes.spines.right'] = False
matplotlib.rcParams['axes.spines.top'] = False


def scale_scatter(df, scale_name="good-bad", save=False):

    ## preprocessing__________________
    df = df.sort_values(scale_name)
    df = df.reset_index(drop=True)

    ent1, ent2, labels = df["ent1"].to_list(), df["ent2"].to_list(), df["label"].to_list()
    ent1_2 = [f"{e1}–{e2}" for e1, e2 in list(zip(ent1, ent2))]
    vals = df[scale_name].to_numpy()
    x_vals = np.arange(0, len(vals))

    # Draw plot
    titlefont = 20
    labelfont = 14

    fig, ax = plt.subplots(1, 1, figsize=(20, 3), gridspec_kw={'hspace': 0.4})
    ax.scatter(x_vals, vals, s=200, alpha=1.0, c=vals, cmap=cm.coolwarm_r)
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
    ax.set_ylabel(scale_name, fontsize=labelfont)

    for i, x_tick_label in enumerate(ax.get_xticklabels()):
        label = labels[i]
        if label == "enemy":
            color = "red"
        elif label == "friend":
            color = "blue"
        x_tick_label.set_color(color)

    plt.show()
    if save:
        fig.savefig(helpers.ROOT_DIR / "results" / "plots" / "gpt-xl.pdf", bbox_inches='tight', dpi=200,
                    transparent=True)


if __name__ == "__main__":
    pass