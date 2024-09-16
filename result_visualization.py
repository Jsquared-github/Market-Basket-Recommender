import pickle as pkl
import seaborn as sns
import pandas as pd
from sklearn.neural_network import MLPClassifier
import matplotlib.pyplot as plt

with open("results/rule_predictions.pkl", "rb") as f:
    rule_results = pkl.load(f)

with open("results/nn_predictions.pkl", "rb") as f:
    nn_results = pkl.load(f)

with open("models/nn_predictor.pkl", "rb") as f:
    nn_model: MLPClassifier = pkl.load(f)


combined_results = pd.DataFrame([{"coverage": 1.0, "accuracy": .031, "composite_score": .031, "model": "Neural Network"},
                                 {"coverage": .139, "accuracy": .316, "composite_score": .044, "model": "ARL Model"}
                                 ])

sns.barplot(data=combined_results, x="model", y="coverage", hue="model")
plt.show()

sns.barplot(data=combined_results, x="model", y="accuracy", hue="model")
plt.show()

sns.barplot(data=combined_results, x="model", y="composite_score", hue="model")
plt.show()
