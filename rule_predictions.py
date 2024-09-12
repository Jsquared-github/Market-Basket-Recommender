import pickle as pkl
import pandas as pd
import numpy as np
from itertools import permutations


def get_best_prediction(arl_predictions):
    best_item = ""
    best_score = 0
    for item, score in arl_predictions:
        if score > best_score:
            best_score = score
            best_item = item
    return best_item


def get_prediction(transaction, total_items: int, max_lhs: int, arl_dict: dict):
    prediction = None
    if max_lhs > 2:
        arl_predictions = []
        lhs = (transaction[2:min(total_items+1, max_lhs)])
        for item_set in permutations(lhs):
            if arl_dict.get(item_set, False):
                prediction = ((item[0], score) for item, score in arl_dict[item_set].items())
                arl_predictions.append(next(prediction))
        if arl_predictions:
            prediction = get_best_prediction(arl_predictions)
        else:
            prediction = get_prediction(transaction, total_items, max_lhs-1, arl_dict)
    return prediction


def predict_items(transactions: pd.DataFrame, one_hot_transactions: pd.DataFrame):
    results = {"predictions": [], "unclassified": 0, "metrics": {"coverage": None, "accuracy": None}}
    total_correct = 0
    for transaction in transactions.itertuples(name="Transaction"):
        index = transaction[0]
        total_items = transaction[1]
        if total_items > 1 and total_items < 32:
            item = get_prediction(transaction, total_items, arl_dict["max_lhs"]+2, arl_dict)
            if item:
                correct = one_hot_transactions.at[index, item]
                results["predictions"].append((index, item, correct))
                total_correct += correct
            else:
                results["unclassified"] += 1
        else:
            results["unclassified"] += 1
    results["metrics"]["coverage"] = len(results["predictions"])/(len(results["predictions"])+results["unclassified"])
    results["metrics"]["accuracy"] = total_correct/len(results["predictions"])
    return results


with open("processed_data/test/test_transactions.pkl", "rb") as f:
    transactions: pd.DataFrame = pkl.load(f)

with open("models/association_rules.pkl", "rb") as f:
    arl_dict: pd.DataFrame = pkl.load(f)

with open("processed_data/test/test_one_hot.pkl", "rb") as f:
    one_hot_data: pd.DataFrame = pkl.load(f)

results = predict_items(transactions, one_hot_data)
print(len(results["predictions"]))
print(results["unclassified"])
print(results["metrics"]["coverage"])
print(results["metrics"]["accuracy"])

# with open("results/rule_predictions.pkl", "wb") as f:
#     pkl.dump(results, f)
