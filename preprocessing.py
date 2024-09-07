import pandas as pd
import pickle as pkl
import numpy as np


def get_distinct(raw_df) -> pd.DataFrame:
    distinct = pd.get_dummies(raw_df["Item 1"])
    item_lists = raw_df.iloc[:, 2:]
    distinct_items = set()
    for col in distinct.columns:
        distinct_items.add(col)

    for col in item_lists.columns:
        curr_item_distinct = pd.get_dummies(item_lists[col])
        for col in curr_item_distinct.columns:
            if col not in distinct_items:
                distinct_items.add(col)
                distinct[col] = curr_item_distinct[col]

    for col in distinct.columns:
        distinct[col].values[:] = 0
    return distinct


def get_one_hot(raw: pd.DataFrame, distinct: pd.DataFrame) -> pd.DataFrame:
    for (index, data) in raw.iterrows():
        for col in data:
            if pd.notna(col):
                distinct[col].values[index] = 1

    return distinct


def get_product_count_by_item(raw: pd.DataFrame, tp: pd.DataFrame):
    item_by_product_count = pd.DataFrame()
    for item_num in raw.columns:
        items = raw[item_num].value_counts()
        item_by_product_count[item_num] = pd.Series(items.values, items.index)
    item_by_product_count.fillna(0, inplace=True)
    return item_by_product_count.T


raw_df = pd.read_csv("raw_data/transactions.csv")
transaction_product_df = get_distinct(raw_df)
one_hot_df = get_one_hot(raw_df.iloc[:, 1:], transaction_product_df)
item_product_counts = get_product_count_by_item(raw_df.iloc[:, 1:], transaction_product_df)


with open("processed_data/one_hot.pkl", "wb") as f:
    pkl.dump(one_hot_df, f)

with open("processed_data/item_product_counts.pkl", "wb") as f:
    pkl.dump(item_product_counts, f)
