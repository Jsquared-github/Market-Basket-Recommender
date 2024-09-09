import pickle as pkl
import pandas as pd
from numpy import exp, abs, log
from efficient_apriori import apriori
from FIM import eclat
from mlxtend.frequent_patterns import fpgrowth

with open("processed_data/train/tuple_itemsets.pkl", "rb") as tf:
    transactions: pd.DataFrame = pkl.load(tf)

with open("processed_data/train/one_hot.pkl", "rb") as ohf:
    oh_transactions: pd.DataFrame = pkl.load(ohf)

final_rules = {}
_, apriori_rules = apriori(transactions, min_support=.01)
for rule in apriori_rules:
    sigmoid_conv = (2/(1 + exp(-(log(rule.conviction))))) - 1
    sigmoid_lift = (2/(1 + exp(-(abs(rule.lift-1))))) - 1
    sup_conf_score = rule.support*rule.confidence
    lift_conv_score = sigmoid_lift*sigmoid_conv
    lhs = rule.lhs
    rhs = rule.rhs
    candidate_rule_score = (sup_conf_score+lift_conv_score)

    if final_rules.get(lhs, False):
        final_rhs: dict = final_rules.get(lhs)
        for score in final_rhs.values():
            if score < candidate_rule_score:
                final_rules[lhs] = {rhs: candidate_rule_score}
    else:
        final_rules[lhs] = {rhs: candidate_rule_score}

_, eclat_rules = eclat(transactions, min_support=.005)
for rule in eclat_rules:
    sigmoid_conv = (2/(1 + exp(-(log(rule.conviction))))) - 1
    sigmoid_lift = (2/(1 + exp(-(abs(rule.lift-1))))) - 1
    sup_conf_score = rule.support*rule.confidence
    lift_conv_score = sigmoid_lift*sigmoid_conv
    lhs = rule.lhs
    rhs = rule.rhs
    candidate_rule_score = (sup_conf_score+lift_conv_score)

    if final_rules.get(lhs, False):
        final_rhs: dict = final_rules.get(lhs)
        for score in final_rhs.values():
            if score < candidate_rule_score:
                final_rules[lhs] = {rhs: candidate_rule_score}
    else:
        final_rules[lhs] = {rhs: candidate_rule_score}


_, fpgrowth_rules = fpgrowth(transactions, min_support=.001)
for rule in fpgrowth_rules:
    sigmoid_conv = (2/(1 + exp(-(log(rule.conviction))))) - 1
    sigmoid_lift = (2/(1 + exp(-(abs(rule.lift-1))))) - 1
    sup_conf_score = rule.support*rule.confidence
    lift_conv_score = sigmoid_lift*sigmoid_conv
    lhs = rule.lhs
    rhs = rule.rhs
    candidate_rule_score = (sup_conf_score+lift_conv_score)

    if final_rules.get(lhs, False):
        final_rhs: dict = final_rules.get(lhs)
        for score in final_rhs.values():
            if score < candidate_rule_score:
                final_rules[lhs] = {rhs: candidate_rule_score}
    else:
        final_rules[lhs] = {rhs: candidate_rule_score}

with open('models/association_rules.pkl', "wb") as f:
    pkl.dump(final_rules, f)
