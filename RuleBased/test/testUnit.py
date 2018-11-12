import numpy as np
from SPARQLWrapper import SPARQLWrapper, JSON
from sklearn.model_selection import train_test_split

from RuleBased.Params import ht_conn, ht_seg, mydb
from RuleBased.unit.Rule import Rule
import threading

# rule_list_sorted_by_recall = []
# with open("../data/dbo_birthPlace/filtered_rule_sorted_by_accuracy.txt", "r", encoding="UTF-8") as f:
#     for idx, line in enumerate(f.readlines()):
#         if idx > 200: break
#         rule_chain, precision, recall, f1 = line.strip().split()print("sdsfsdfdssdfsdfsfddsfs")
#         rule_list_sorted_by_recall.append(Rule(rule_chain, precision, recall, f1))
# rule_list_sorted_by_recall=sorted(rule_list_sorted_by_recall, key=lambda k: k.recall,reverse=True)
# for rule in rule_list_sorted_by_recall:
#     print("{}".format(rule))

# def FTEST():
#     for i in range(5):
#         print(" {}".format( i))
#
#
# array_n = ["se", "et", "etdg"]
#
# for n in array_n:
#     print("{}".format(n))
#     tt = threading.Thread(target=FTEST,)
#     tt.start()
#
# print("dsfsdfsdfsdfsdf")

# entity_query = """SELECT *
# WHERE{
# ?film <http://dbpedia.org/ontology/director> ?p.
# }"""
# sparql = SPARQLWrapper("http://210.28.132.61:8898/sparql")
# sparql.setTimeout(10)
# sparql.setQuery(entity_query)
# sparql.setReturnFormat(JSON)
# results = sparql.query().convert()
# vars_list = results['head']['vars']
# bindings_list = results['results']['bindings']
# for binding in bindings_list:
#     for var in vars_list:
#         print("{} {}".format(var, binding[var]['value']))

# import json
#
#
# class TTTT:
#     def __init__(self):
#         self.dict1 = {}
#         self.dict2 = []
#         self.generate()
#
#     def generate(self):
#         self.dict1['t'] = [1, 2, 3, 6]
#         self.dict1['y'] = [5, 9, 78, 9]
#         self.dict2 = [[1, 5, 9, 7], [7, 8, 9, 6], [4, 5, 6, 3]]


if __name__ == "__main__":
    # query = "INSERT INTO dbpediarule ( relation_idx,rule_key,correct_ht,wrong_ht,no_idea_ht,P,R,F1 ) " \
    #         "VALUES ({},'{}','{}','{}','{}',{},{},{})".format(1, "2:6", "6,4;8,9", "8,9;7,6",
    #                                                   "98,56;15,89", 0.1, 0.2, 0.3)
    # print(query)
    # mycursor = mydb.cursor()
    # mycursor.execute(query)
    # mydb.commit()

    query = "select * from dbpediarule where relation_idx = {} and rule_key = '{}'".format(1, "2:6")
    print(query)
    mycursor = mydb.cursor()
    mycursor.execute(query)

    fetched = mycursor.fetchall()
    assert len(fetched) <= 2, "Duplicate relation:rulepath in MYSQL."
    for row in fetched:
        correct_ht = [list(map(int, ht2)) for ht2 in [ht.split(ht_conn) for ht in row[3].split(ht_seg)]]
        wrong_ht = [ht.split(ht_conn) for ht in row[4].split(ht_seg)]
        no_idea_ht = [ht.split(ht_conn) for ht in row[5].split(ht_seg)]
        P = row[6]
        R = row[7]
        F1 = row[8]
        print("dfsd")
    # for x in mycursor:
    #     print(x)
