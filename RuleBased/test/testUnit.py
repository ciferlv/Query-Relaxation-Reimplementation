import numpy as np
from sklearn.model_selection import train_test_split

from RuleBased.unit.Rule import Rule
import threading


# rule_list_sorted_by_recall = []
# with open("../data/dbo_birthPlace/filtered_sorted_rule.txt", "r", encoding="UTF-8") as f:
#     for idx, line in enumerate(f.readlines()):
#         if idx > 200: break
#         rule_chain, precision, recall, f1 = line.strip().split()print("sdsfsdfdssdfsdfsfddsfs")
#         rule_list_sorted_by_recall.append(Rule(rule_chain, precision, recall, f1))
# rule_list_sorted_by_recall=sorted(rule_list_sorted_by_recall, key=lambda k: k.recall,reverse=True)
# for rule in rule_list_sorted_by_recall:
#     print("{}".format(rule))

def test():
    for i in range(5):
        print(" {}".format( i))


array_n = ["se", "et", "etdg"]

for n in array_n:
    print("{}".format(n))
    tt = threading.Thread(target=test,)
    tt.start()

print("dsfsdfsdfsdfsdf")
