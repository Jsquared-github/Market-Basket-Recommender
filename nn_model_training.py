import pickle as pkl
import pandas as pd
from sklearn.neural_network import MLPClassifier

with open("processed_data/train/train_transactions.pkl", "rb") as f:
    transactions: pd.DataFrame = pkl.load(f)

with open("processed_data/train/one_hot.pkl", "rb") as f:
    one_hot_data: pd.DataFrame = pkl.load(f)

Y_train = []
X_train = []
for transaction in transactions.itertuples(name=None):
    index = transaction[0]
    last_item = transaction[transaction[1]+1]
    one_hot_data.at[index, last_item] = 0
    Y_train.append(last_item)
    one_hot_vector = one_hot_data.iloc[index].values.flatten().tolist()
    X_train.append(one_hot_vector)

trained_classifier = MLPClassifier(hidden_layer_sizes=(78, 78), learning_rate_init=.0001, max_iter=3000).fit(X=X_train, y=Y_train)
with open("models/nn_predictor.pkl", "wb") as f:
    pkl.dump(trained_classifier, f)
