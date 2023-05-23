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
    ax.barh(labels, data)
    plt.show()


def biplot(pc1, pc2, labels, features, loadings, title=None, x_label=None, y_label=None):
    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    plt.xticks(rotation=45, ha='right')

    scale_pc1 = 1.0 / (pc1.max() - pc1.min())
    scale_pc2 = 1.0 / (pc2.max() - pc2.min())

    for i, feature in enumerate(features):
        ax.arrow(0, 0, loadings[0, i],
                 loadings[1, i], color='b', alpha=0.25)
        ax.text(loadings[0, i] * 1.1,
                loadings[1, i] * 1.1,
                feature, fontsize=7.5)

    ax.scatter(pc1 * scale_pc1, pc2 * scale_pc2, s=1)
    for (i, label) in enumerate(labels):
        ax.annotate(label,
                    (pc1[i] * scale_pc1, pc2[i] * scale_pc2),
                    fontsize=5)

    ax.set_xlabel('PC1', fontsize=20)
    ax.set_ylabel('PC2', fontsize=20)
    ax.set_title('Figure 1', fontsize=20)
    plt.figure()

    plt.show()
