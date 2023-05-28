import numpy as np
import os

from matplotlib import pyplot as plt
from matplotlib import ticker

os.makedirs("./figs", exist_ok=True)


def boxplot(data, labels, title=None, x_label=None, y_label=None):
    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    ax.boxplot(data, labels=labels)
    plt.show()


def bargraph(data, labels, title=None, x_label=None, y_label=None):
    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.subplots_adjust(left=0.25)
    ax.barh(labels, data)
    plt.show()


def biplot(pc1, pc2, labels, features, loadings, title=None, x_label="PC1", y_label="PC2"):
    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)

    for i, feature in enumerate(features):
        ax.arrow(0, 0, loadings[0, i] * 10,
                 loadings[1, i] * 10, color='b', alpha=0.25)
        ax.text(loadings[0, i] * 10.1,
                loadings[1, i] * 10.1,
                feature, fontsize=7.5)

    ax.scatter(pc1, pc2, s=1)
    for (i, label) in enumerate(labels):
        ax.annotate(label,
                    (pc1[i], pc2[i]),
                    fontsize=5)

    plt.figure()

    plt.show()



def pattern(data):
    fig, ax = plt.subplots()
    ax.spines[:].set_visible(False)
    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="tab:gray", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(left=False, bottom=False)
    plt.style.use('grayscale')
    plt.imshow(data)
    ax.xaxis.set_major_locator(ticker.NullLocator())
    ax.yaxis.set_major_locator(ticker.NullLocator())
    plt.show()
    

def heatmap(matrix, file, title=None, text=None):
    if text is None:
        text = []
    plt.imshow(matrix, cmap='RdYlGn')
    plt.colorbar()

    # Annotate the heatmap with the values
    for i in range(len(text)):
        for j in range(len(text[0])):
            plt.text(j, i, text[i][j], ha='center', va='center', color='black', fontsize=9)

    plt.title(title)

    plt.show()

    plt.savefig(f"./figs/{file}")

