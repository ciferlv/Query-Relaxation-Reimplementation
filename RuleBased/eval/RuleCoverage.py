from Empty_Answer_Query import eaqs
import numpy as np
from RuleBased.SparqlParser import SparqlParser
from Util import Util

util = Util()

coverage_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000, 1100]

r_name_list = []
for sparql in eaqs:
    sp = SparqlParser(sparql=sparql)
    sp.parse_sparql()
    for relation in sp.r_name_list:
        r_name_list.append(relation)
    r_name_list = list(set(r_name_list))


def get_coverage(x_data):
    res_ratio = []
    res_str = ""
    for c_num in coverage_list:
        feature = np.sum(x_data[:c_num], -1)
        ratio = np.sum(feature > 0) / x_data.shape[0]
        res_ratio.append(ratio)
        res_str += "%.4f\t" % ratio
    return res_str, res_ratio


coverage_file = "./coverage.txt"
ratio_list = []
writer = open(coverage_file, 'w', encoding="UTF-8")
writer.write("{}\n".format("\t".join(list(map(str, coverage_list)))))

for r_name in r_name_list:
    writer.write("{}\n".format(r_name))
    feature_file = "../../MyData/DBO/United_States/model/{}/pca_train_feature_data.npy".format(r_name.split(":")[-1])
    rule_num_file = "../../MyData/DBO/United_States/model/{}/rule_num_to_use.txt".format(r_name.split(":")[-1])

    with open(rule_num_file, 'r', encoding="UTF-8") as f:
        rule_num = int(f.readline().split()[-1])
        if rule_num > 1000:
            print(r_name)

    train_data = np.load(feature_file)
    train_data = train_data[np.where(train_data[:, -1] == 1.0)]
    train_x = train_data[:, 0:-1]
    train_y = train_data[:, -1]

    assert train_x.shape[1] == rule_num, "Wrong rule_num and feature num"

    one_str, one_ratio = get_coverage(train_x)
    ratio_list.append(one_ratio)

    writer.write(one_str + "\n")

avg = np.sum(np.array(ratio_list), 0) / len(ratio_list)
writer.write("{}\n".format("\t".join(list(map(str, avg)))))
writer.close()
