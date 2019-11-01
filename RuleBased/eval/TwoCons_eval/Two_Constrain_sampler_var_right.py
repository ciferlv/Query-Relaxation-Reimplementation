import random

from SPARQLWrapper import SPARQLWrapper, JSON

from RuleBased.Graph import Graph
from RuleBased.FilePathConfig import FilePathConfig
from Util import Util


class TwoConsSampler:
    def __init__(self, graph):

        self.graph = graph
        self.first_r_name = ""
        self.second_r_name = ""
        self.first_r_idx = None
        self.second_r_idx = None

        self.restrain_num = 1000
        self.res_token = set()
        self.sparql_database = "http://114.212.86.67:8890/sparql"
        self.sparql_interface = SPARQLWrapper(self.sparql_database)
        self.util = Util()

        self.fileName = "./"

    def set_relation(self, first_relation, second_relation):
        self.first_r_name = first_relation
        self.second_r_name = second_relation
        self.first_r_idx = self.graph.r2idx["inv_" + self.first_r_name]
        self.second_r_idx = self.graph.r2idx["inv_" + self.second_r_name]

    def sample_data(self):
        print("Start sample data")
        f_r_name = "inv_" + self.first_r_name
        s_r_name = "inv_" + self.second_r_name
        folder = self.fileName + "{}_{}/".format(self.first_r_name.replace("dbo:", ""),
                                                 self.second_r_name.replace("dbo:", ""))
        self.util.createFolder(folder)
        self.fileName = folder + "2consQandA.txt"

        # if os.path.exists(self.fileName):
        #     return

        for e_idx in self.graph.idx2e.keys():
            c_node = self.graph.node_dict[e_idx]
            first_val_idx_list = c_node.get_r_value(self.first_r_idx)
            second_val_idx_list = c_node.get_r_value(self.second_r_idx)
            if len(first_val_idx_list) == 0 or len(second_val_idx_list) == 0:
                continue
            for f_val_idx in first_val_idx_list:
                for s_val_idx in second_val_idx_list:
                    token = "{}\t{}\t?x+{}\t{}\t?x".format(self.graph.idx2e[f_val_idx], f_r_name.replace("inv_", ""),
                                                           self.graph.idx2e[s_val_idx], s_r_name.replace("inv_", ""))
                    if token not in self.res_token:
                        self.res_token.add(token)
            # if len(self.res_token) >= self.restrain_num:
            #     break
        self.res_token = list(self.res_token)
        if len(self.res_token) > self.restrain_num:
            self.res_token = random.sample(self.res_token, self.restrain_num)
        print("Res token num: {}".format(len(self.res_token)))

    def get_e_r_idx(self, triple_pattern):
        _, r_name, e_name = triple_pattern.split()
        e_idx = self.graph.e2idx[e_name]
        r_idx = self.graph.r2idx[r_name]
        return e_idx, r_idx

    def record_test_sample(self):
        print("Start recording data.")
        res_list = []
        for token in self.res_token:
            f_cons, s_cons = token.split("+")
            f_e_name, f_r_name, _ = f_cons.split()
            s_e_name, s_r_name, _ = s_cons.split()
            tmp_f_e_name = self.util.getFullName(f_e_name)
            tmp_s_e_name = self.util.getFullName(s_e_name)

            res = [[f_e_name, f_r_name], [s_e_name, s_r_name], []]
            query = "select * where { " + f_cons.replace(f_e_name, tmp_f_e_name) \
                    + ".\n" + s_cons.replace(s_e_name, tmp_s_e_name) + "}"
            print(query)
            self.sparql_interface.setQuery(query)
            self.sparql_interface.setReturnFormat(JSON)
            results = self.sparql_interface.query().convert()
            # print(results)
            binding_list = results['results']['bindings']
            for binding in binding_list:
                x = binding['x']['value']
                x_shortcut = self.util.getShortcutName(x)
                res[2].append(x_shortcut)

            res_list.append(res)

        with open(self.fileName, 'w', encoding="UTF-8") as f:
            print("Res Num: {}".format(len(res_list)))
            for res_idx,res in enumerate(res_list):
                f.write("{}\t{}\n".format(res[0][0], res[0][1]))
                f.write("{}\t{}\n".format(res[1][0], res[1][1]))
                for answer_name in res[2]:
                    f.write("{}\t".format(answer_name))
                if res_idx != len(res_list) - 1:
                    f.write("\n")


if __name__ == "__main__":
    filePathConfig = FilePathConfig()
    # e2idx_file = "../" + self.filePathConfig.test_e2idx_file
    # r2idx_file = "../" + self.filePathConfig.test_r2idx_file
    # triple2idx_file = "../" + self.filePathConfig.test_triple2idx_file
    e2idx_file = "../" + filePathConfig.search_e2idx_file
    r2idx_file = "../" + filePathConfig.search_r2idx_file
    triple2idx_file = "../" + filePathConfig.search_triple2idx_file
    graph = Graph(e2idx_file, r2idx_file, triple2idx_file)
    graph.load_data()

    r_name_list_right = [["dbo:residence", "dbo:deathPlace"]]

    for r_name in r_name_list_right:
        tcs = TwoConsSampler(graph)
        tcs.set_relation(r_name[0], r_name[1])
        tcs.sample_data()
        tcs.record_test_sample()
