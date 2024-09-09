import pickle as pkl
import pandas as pd
from numpy import exp, abs
from efficient_apriori import apriori
from FIM import eclat
from mlxtend.frequent_patterns import fpgrowth

with open("processed_data/train/tuple_itemsets.pkl", "rb") as tf:
    transactions: pd.DataFrame = pkl.load(tf)

with open("processed_data/train/one_hot.pkl", "rb") as ohf:
    oh_transactions: pd.DataFrame = pkl.load(ohf)

itemsets, rules = apriori(transactions, min_support=.01, min_confidence=.501)
for rule in rules:
    sigmoid_conv = (2/(1 + exp(-(rule.conviction-1)))) - 1
    sigmoid_lift = (2/(1 + exp(-(abs(rule.lift-1))))) - 1
    sup_conf_score = rule.support*rule.confidence
    lift_conv_score = sigmoid_lift*sigmoid_conv
    print(rule.lhs, rule.rhs, (sup_conf_score+lift_conv_score))

eclat_itemsets = eclat(oh_transactions, min_support=.05)
print(eclat_itemsets)

oh_transactions = oh_transactions.astype('bool')
fp_itemsets = fpgrowth(oh_transactions, min_support=.01, use_colnames=True)
print(fp_itemsets)
