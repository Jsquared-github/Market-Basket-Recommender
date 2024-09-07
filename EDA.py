import pickle as pkl
import pandas as pd
import numpy as np

with open("processed_data/transactions.pkl", "rb") as f:
    transaction_data: pd.DataFrame = pkl.load(f)

item_data = transaction_data.iloc[:, 1:]
print(item_data)
print(np.sum(item_data.mean().sort_values(ascending=False) * 9835))
print(np.sum(transaction_data["item_count"]))
