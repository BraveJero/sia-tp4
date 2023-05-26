from matplotlib import pyplot as plt


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
    ax.grid(zorder=0)
    ax.barh(labels, data, zorder=3)
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
