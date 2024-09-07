import pickle as pkl
import pandas as pd
import numpy as np
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl


def get_sorted_counts(data):
    item_data = data.iloc[:, 1:]
    sorted_items = []
    for item, freq in item_data.sum().items():
        sorted_items.append((item, freq))

    sorted_items = np.array(sorted_items, dtype=[('item', '<U30'), ('count', int)])
    sorted_items.sort(order='count')
    return sorted_items


def top_k_items(item_counts: np.ndarray, k, labels: bool = False):
    items = []
    freqs = []
    for (item, freq) in item_counts[::-1]:
        if k > 0:
            items.append(item)
            freqs.append(freq)
        else:
            break
        k -= 1

    color_scheme = sns.color_palette('crest', n_colors=169)
    color_scheme.reverse()
    ax = sns.barplot(x=items, y=freqs, palette=color_scheme, width=3.2)
    if not labels:
        ax.set(xticklabels=[])
    else:
        plt.xticks(rotation=45)
    plt.show()


def stacked_item_counts(items: pd.DataFrame):
    cmap = mpl.colormaps['viridis']
    colors = cmap(np.linspace(0, 1, 169))
    product_colors = {}
    for color, product in zip(colors, items.columns):
        product_colors[product] = mpl.colors.rgb2hex(color, True)
    fig = items.plot.bar(stacked=True, color=product_colors)
    fig.get_legend().remove()
    fig.set(xticklabels=[])
    plt.show()


with open("processed_data/one_hot.pkl", "rb") as f:
    one_hot_data: pd.DataFrame = pkl.load(f)
sorted_counts = get_sorted_counts(one_hot_data)
top_k_items(sorted_counts, k=169)

with open("processed_data/item_product_counts.pkl", "rb") as f:
    item_product_counts = pkl.load(f)
stacked_item_counts(item_product_counts)
