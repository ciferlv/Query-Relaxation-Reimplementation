import os
import time

from RuleBased.BiSearch.Graph import Graph
from RuleBased.BiSearch.SparqlParser import SparqlParser
from RuleBased.Params import file_path_seg, rules_num_to_search_cands, sort_candidate_criterion
from RuleBased.VirtuosoSearch.Util import Util


class QR:
    def __init__(self, sparql, root_folder, train_scope, test_scope, search_scope):
        self.sparql = sparql
        self.sp = SparqlParser(sparql=sparql)
        self.sp.parse_sparql()

        self.util = Util()

        self.root_folder = root_folder
        self.train_scope = train_scope
        self.test_scope = test_scope
        self.search_scope = search_scope

        self.search_root = root_folder + search_scope + "\\"
        self.train_root = root_folder + train_scope + "\\"
        self.test_root = root_folder + test_scope + "\\"

        self.r_name_list = self.sp.r_name_list

        '''
        Dict for r_idx and its list of rule object.
        {
            r_idx:[Rule(),Rule(),...],
            r_idx:[Rule(),Rule(),...],
            ...
        }
        '''
        self.r_rules_dict = {}
        '''
        Dict for r_idx and its trained model.
        {
            r_idx: LogisticRegression Model
            r_idx: LogisticRegression Model
            ...
        }
        '''
        self.r_model_dict = {}

    def train_rules(self):
        start_time = time.time()

        saved_model_folder = self.train_root + "model\\"
        e2idx_file = self.train_root + "e2idx_shortcut.txt"
        r2idx_file = self.train_root + "r2idx_shortcut.txt"
        triple2idx_file = self.train_root + "triple2idx.txt"

        graph = Graph(e2idx_file, r2idx_file, triple2idx_file)
        graph.load_data()
        self.r_idx_list = [graph.r2idx[relation] for relation in self.r_name_list]

        print("Start Training Rules.")
        for idx, r_idx in enumerate(self.r_idx_list):
            print("Going to train R:{}".format(graph.idx2r[r_idx]))
            r_folder = saved_model_folder + self.util.gen_prefix(self.r_name_list[idx]).replace(':',
                                                                                                '_') + file_path_seg
            if not os.path.isdir(r_folder):
                os.makedirs(r_folder)
            model = graph.get_pra_model4r(r_idx=r_idx, folder=r_folder)
            self.r_model_dict[r_idx] = model
        end_time = time.time()
        print("Finish train rules and get the model. Elapsed: {}.".format(end_time - start_time))

    def test_rules(self):
        e2idx_file = self.test_root + "e2idx_shortcut.txt"
        r2idx_file = self.test_root + "r2idx_shortcut.txt"
        triple2idx_file = self.test_root + "triple2idx.txt"

        test_file_folder = self.test_root + "test_model\\"
        if not os.path.isdir(test_file_folder):
            os.makedirs(test_file_folder)

        graph = Graph(e2idx_file, r2idx_file, triple2idx_file)
        graph.load_data()

        print("Start Testing Rules.")
        for idx, r_idx in enumerate(self.r_idx_list):
            metric_record_file = test_file_folder + graph.get_localname(
                graph.get_r_name_by_r_idx(r_idx)) + "_metric.txt"
            with open(metric_record_file, 'w', encoding="UTF-8") as f:
                f.write("Use Model trained from {}.\n".format(self.train_scope))
            print("Going to train R:{}".format(graph.idx2r[r_idx]))
            model = self.r_model_dict[r_idx]
            rule_list = graph.get_rule4train_from_mysql(r_idx)
            graph.test_model(r_idx, model, rule_list, metric_record_file)

    def gen_rules_dict(self, graph):
        print("Start collect top rules by Precision.")
        for r_idx in self.r_idx_list:
            self.r_rules_dict[r_idx] = graph.get_top_k_rules(r_idx, rules_num_to_search_cands)

    def get_candidates(self):
        e2idx_file = self.search_root + "e2idx_shortcut.txt"
        r2idx_file = self.search_root + "r2idx_shortcut.txt"
        triple2idx_file = self.search_root + "triple2idx.txt"

        graph = Graph(e2idx_file, r2idx_file, triple2idx_file)
        graph.load_data()

        self.gen_rules_dict(graph)

        start_time = time.time()
        print("Start generating candidates.")

        print("Start executing 1 var BGP.")
        self.sp.execute_var1BGP(self.r_rules_dict, graph)

        print("Start executin 2 var BGP.")
        self.sp.execute_var2BGP(self.r_rules_dict, graph)

        print("Start normalize searched res.")
        self.sp.normalize_searched_res()

        # print("Display result.")
        # self.sp.display_searched_res(graph)

        print("Calculate confidence for candidates.")
        self.sp.gen_conf_and_rule_path(self.r_rules_dict, self.r_model_dict, graph)

        print("Sort candidate list by {} score.".format(sort_candidate_criterion))
        self.sp.sort_cand_obj_list()

        print("Print cands.")
        self.sp.display_cands(graph)
        end_time = time.time()
        print("Finishing generating and displaying candidates. Epalsed: {}.".format(end_time - start_time))


if __name__ == "__main__":
    root_folder = "F:\\Data\\dbpedia\\"
    train_scope = "United_States"
    # train_scope = "Japan"
    test_scope = "Canada"
    search_scope = "All"
    # search_scope = "United_States"

    # sparql = """
    #     SELECT ?film WHERE{
    #         ?film dbo:director ?p.
    #         ?film dbo:starring ?p.
    #         ?p dbo:birthPlace dbr:North_America.
    #     }
    #     """

    # sparql = """
    #     SELECT ?film WHERE{
    #         ?film dbo:starring ?p.
    #         ?p dbo:birthPlace dbr:Asia.
    #     }
    #     """

    sparql = """
        SELECT * WHERE{
         dbr:Isaac_Newton dbo:doctoralAdvisor ?p.}
        """

    # sparql = """
    #         SELECT * WHERE{
    #          dbr:Albert_Einstein dbo:doctoralAdvisor ?p.}
    #         """

    qr = QR(sparql, root_folder, train_scope, test_scope, search_scope)

    only_train_rule = False
    if only_train_rule:
        qr.train_rules()
        qr.test_rules()
    else:
        qr.train_rules()
        qr.get_candidates()
