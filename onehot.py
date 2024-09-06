import pandas as pd
import pickle as pkl


def get_one_hot(raw_df) -> pd.DataFrame:
    one_hot = pd.get_dummies(raw_df["Item 1"])
    item_lists = raw_df.iloc[:, 2:]
    distinct_items = set()
    for col in one_hot.columns:
        distinct_items.add(col)
    for col in item_lists.columns:
        curr_item_one_hot = pd.get_dummies(item_lists[col])
        for col in curr_item_one_hot.columns:
            if col not in distinct_items:
                distinct_items.add(col)
                one_hot[col] = curr_item_one_hot[col]
    return one_hot


raw_df = pd.read_csv("raw_data/transactions.csv")
one_hot_df = get_one_hot(raw_df)
one_hot_df.insert(0, "item_count", raw_df["Item(s)"])

with open("processed_data/transactions.pkl", "wb") as f:
    pkl.dump(one_hot_df, f)
