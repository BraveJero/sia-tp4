from matplotlib import pyplot as plt


def boxplot(data, labels, title=None, x_label=None, y_label=None):
    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.boxplot(data, labels=labels)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show(bbox_inches='tight')


def bargraph(data, labels, title=None, x_label=None, y_label=None):
    fig, ax = plt.subplots()
    plt.title(title)
    plt.xlabel(x_label)
    plt.ylabel(y_label)
    ax.bar(labels, data)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show(bbox_inches='tight')
