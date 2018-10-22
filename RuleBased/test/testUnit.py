import numpy as np
from sklearn.model_selection import train_test_split

from RuleBased.unit.Rule import Rule

rule_list_sorted_by_recall = []
with open("../data/dbo_birthPlace/filtered_sorted_rule.txt", "r", encoding="UTF-8") as f:
    for idx, line in enumerate(f.readlines()):
        if idx > 200: break
        rule_chain, precision, recall, f1 = line.strip().split()
        rule_list_sorted_by_recall.append(Rule(rule_chain, precision, recall, f1))
rule_list_sorted_by_recall=sorted(rule_list_sorted_by_recall, key=lambda k: k.recall,reverse=True)
for rule in rule_list_sorted_by_recall:
    print("{}".format(rule))