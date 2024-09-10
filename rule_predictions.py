import pickle as pkl
import pandas as pd
import numpy as np


def get_sorted_counts(data: pd.DataFrame):
    sorted_items = []
    for item, freq in data.sum().items():
        sorted_items.append((item, freq))

    sorted_items = np.array(sorted_items, dtype=[('item', '<U30'), ('count', int)])
    sorted_items.sort(order='count')
    return sorted_items[::-1]


with open("processed_data/test/test_transactions.pkl", "rb") as f:
    transactions: pd.DataFrame = pkl.load(f)

with open("models/association_rules.pkl", "rb") as f:
    arl_dict: pd.DataFrame = pkl.load(f)

with open("processed_data/one_hot.pkl", "rb") as f:
    one_hot_data: pd.DataFrame = pkl.load(f)
most_frequent_items = get_sorted_counts(one_hot_data)

# for transaction in transactions.itertuples(index=False, name="Transaction"):
#     total_items = transaction[0]
#     lhs = (transaction[1:total_items])
#     print(total_items, lhs)
