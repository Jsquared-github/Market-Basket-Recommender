import pickle as pkl
import pandas as pd
from sklearn.neural_network import MLPClassifier


def modify_transactions(transactions: pd.DataFrame):
    modified_oh = transactions.copy(deep=True)
    for transaction in transactions.itertuples(name=None):
        index = transaction[0]
        last_item = transaction[transaction[1]+1]
        modified_oh.at[index, last_item] = 0
    return modified_oh.iloc[:, :-2]


def predict_items(nn_model: MLPClassifier, modified_transactions: pd.DataFrame):
    X_test = []
    for transaction in modified_transactions.itertuples(name=None):
        X_test.append([*transaction[1:]])
    return nn_model.predict(X_test)


def get_results(predictions: list, unmodified_transactions: pd.DataFrame):
    result = {"predictions": [], "unclassified": 0, "metrics": {"coverage": 1, "accuracy": None}}
    total_correct = 0
    for index, prediction in enumerate(predictions):
        correct = unmodified_transactions.at[index+7376, prediction]
        result["predictions"].append((index, prediction, correct))
        total_correct += correct
    result["metrics"]["accuracy"] = total_correct/len(predictions)
    return result


with open("models/nn_predictor.pkl", "rb") as f:
    nn_model: MLPClassifier = pkl.load(f)

with open("processed_data/test/test_one_hot.pkl", "rb") as f:
    oh_transactions: pd.DataFrame = pkl.load(f)

with open("processed_data/test/test_transactions.pkl", "rb") as f:
    transactions: pd.DataFrame = pkl.load(f)

modified_transactions = modify_transactions(oh_transactions)
predictions = predict_items(nn_model, modified_transactions)
results = get_results(predictions, oh_transactions)
print(results)

# with open("results/nn_predictions.pkl", "wb") as f:
#     pkl.dump(results, f)
