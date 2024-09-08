import pickle as pkl

with open("processed_data/transactions.pkl", "rb") as f:
    transactions = pkl.load(f)

print(transactions)
