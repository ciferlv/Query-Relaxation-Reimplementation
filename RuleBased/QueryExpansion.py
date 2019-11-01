import random
import time
import argparse

from Empty_Answer_Query import eaqs
from ALogger import ALogger
from RuleBased.Graph import Graph
from RuleBased.SparqlParser import SparqlParser
from RuleBased.FilePathConfig import FilePathConfig
from RuleBased.Params import rules_num_to_search_cands, pca_or_cwa
from Util import Util


class QE:
    def __init__(self, args):
        self.logger = ALogger("Graph", True).getLogger()
        self.util = Util()

        self.sp = None
        self.args = args

        self._build_container()

    def _build_container(self):

        self.train_graph = Graph()
        self.search_graph = Graph()

        '''
        Dict for r_name and its list of rule object.
        It is used to search cands.
        {
            r_name:[Rule(),Rule(),...],
            r_name:[Rule(),Rule(),...],
            ...
        }
        '''
        self.r_name2rule_set = {}

        # '''
        # Dict for r_idx and its list of rule object.
        # It is used to feed test_model.
        # {
        #     r_idx:[Rule(),Rule(),...],
        #     r_idx:[Rule(),Rule(),...],
        #     ...
        # }
        # '''
        # self.r_rules_dict_4_feed_model = {}

        '''
        Dict for r_name and its trained test_model.
        {
            r_name: LogisticRegression Model
            r_name: LogisticRegression Model
            ...
        }
        '''
        self.r_name2model = {}

    def _setSparql(self, sparql):
        self.sparql = sparql
        self.sp = SparqlParser(sparql=sparql)
        self.sp.parse_sparql()
        self.r_name_list = self.sp.r_name_list

    def train_rules(self):
        for idx, r_name in enumerate(self.sp.r_name_list):
            self.logger.info("Train\t{}".format(r_name))
            train_args = self.get_train_args(r_name)
            self.train_graph._build_train_file_path(train_args)
            self.train_graph.load_data()
            r_idx = self.train_graph.r_name2id[r_name]
            model = self.train_graph.get_pra_model4r(r_idx=r_idx)
            self.r_name2model[r_name] = model

    # def test_rules(self):
    #     print("Start Testing Rules.")
    #     for idx, r_idx in enumerate(self.r_idx_list):
    #         metric_record_folder = self.args.test_root + "test_model" + file_path_seg \
    #                                + self.test_graph.get_localname(self.idx2r[r_idx]) + file_path_seg
    #         if os.path.isdir(metric_record_folder):
    #             continue
    #
    #         os.makedirs(metric_record_folder)
    #         metric_record_file = metric_record_folder + "pra_metric.txt"
    #         self.test_graph.load_data()
    #
    #         with open(metric_record_file, 'w', encoding="UTF-8") as f:
    #             f.write("Use Model trained from {}.\n".format(self.args.train_scope))
    #         print("Testing {}".format(self.idx2r[r_idx]))
    #         model = self.r_model_dict[r_idx]
    #         rule_list = self.test_graph.get_rule4train_from_mysql(r_idx)
    #         self.test_graph.test_model(r_idx, model, rule_list, metric_record_folder, metric_record_file)

    def get_rule_set_model(self, graph):
        for r_name in self.r_name_list:
            graph.rule_file = "../MyData/DBO/United_States/model/{}/rule.txt".format(r_name.split(":")[-1])
            graph.rule_num_to_use_file = "../MyData/DBO/United_States/model/{}/rule_num_to_use.txt".format(
                r_name.split(":")[-1])
            graph.model_file = "../MyData/DBO/United_States/model/{}/{}_model.tar".format(r_name.split(":")[-1],
                                                                                          pca_or_cwa)
            self.logger.info("Collect Rules for R: {}".format(r_name))
            self.r_name2rule_set[r_name] = graph.load_rule_obj_from_file(r_name)[:rules_num_to_search_cands]

            r_idx = graph.r_id_by_name(r_name)
            model = graph.get_pra_model4r(r_idx=r_idx)
            self.r_name2model[r_name] = model

    def get_candidates(self, query_name):
        self.logger.info("Get candidates.")
        search_args = self.get_search_args(query_name)
        self.search_graph._build_search_file_path(search_args)

        self.search_graph.load_data()

        self.get_rule_set_model(self.search_graph)

        start_time = time.time()

        self.logger.info("Get candidates and execute 1 var BGP.")
        self.sp.execute_var1BGP(self.r_name2rule_set, self.search_graph)

        self.logger.info("Execute 2 var BGP.")
        self.sp.execute_var2BGP(self.r_name2rule_set, self.search_graph)

        print("Start normalize searched res.")
        self.sp.normalize_searched_res()

        # print("Display result.")
        # self.sp.display_searched_res(graph)
        if len(self.sp.searched_res) > 1500:
            self.sp.searched_res = random.sample(self.sp.searched_res, 1500)

        self.logger.info("Calculate confidence for candidates.")
        self.sp.gen_pra_conf_and_rule_path(self.r_name2rule_set, self.r_name2model, self.search_graph)

        self.logger.info("Sorting and displaying.")
        self.sp.sort_cand_obj_list("pra")
        self.sp.display_cands(self.search_graph)

        end_time = time.time()
        self.logger.info("Finishing generating and displaying candidates. Epalsed: {}.".
                         format(end_time - start_time))

    def get_search_args(self, query_name):
        parser = argparse.ArgumentParser()
        search_folder = "../MyData/DBO/All/"
        # search_folder = "../MyData/DBO/United_States/"

        parser.add_argument('--e2id_file', type=str, default=search_folder + "entity2id.txt",
                            help='entity2id file')
        parser.add_argument('--r2id_file', type=str, default=search_folder + "relation2id.txt",
                            help='relation2id file')
        parser.add_argument('--triple2id_file', type=str, default=search_folder + "triple2id.txt",
                            help='triple2id file')

        parser.add_argument('--qe_res_all', type=str,
                            default="../MyData/DBO/United_States/EmptyQ/{}_{}_qe_res_all.txt".format(query_name,
                                                                                                        pca_or_cwa),
                            help='{} qe resutls with {}'.format(query_name, pca_or_cwa))
        parser.add_argument('--qe_res_topk', type=str,
                            default="../MyData/DBO/United_States/EmptyQ/{}_{}_qe_res_topk.txt".format(query_name,
                                                                                                         pca_or_cwa),
                            help='{} qe resutls with {}'.format(query_name, pca_or_cwa))

        args = parser.parse_args()
        return args

    def get_train_args(self, r_name):

        util = Util()
        scope = "United_States"
        root_folder = "../MyData/{}/{}/".format("DBO", scope)
        model_folder = root_folder + "model/{}/".format(r_name.split(":")[-1])

        util.createFolder(root_folder)
        util.createFolder(model_folder)

        parser = argparse.ArgumentParser()

        parser.add_argument('--root_folder', type=str, default=root_folder,
                            help='root folder file')

        parser.add_argument('--e2id_file', type=str, default=root_folder + "entity2id.txt",
                            help='entity2id file')
        parser.add_argument('--r2id_file', type=str, default=root_folder + "relation2id.txt",
                            help='relation2id file')
        parser.add_argument('--triple2id_file', type=str, default=root_folder + "triple2id.txt",
                            help='triple2id file')

        parser.add_argument('--model_folder', type=str, default=model_folder,
                            help='model folder for {}'.format(r_name))

        parser.add_argument('--rule_file', type=str, default="{}rule.txt".format(model_folder),
                            help='rule file for {}'.format(r_name))
        parser.add_argument('--rule_num_to_use_file', type=str, default="{}rule_num_to_use.txt".format(model_folder),
                            help='rule num to use for {}'.format(r_name))

        parser.add_argument('--train_id_data_file', type=str, default="{}train_id_data.npy".format(model_folder),
                            help='train id data for {}'.format(r_name))

        parser.add_argument('--train_feature_data_file', type=str,
                            default="{}{}_train_feature_data.npy".format(model_folder, pca_or_cwa),
                            help='train feature data for {}'.format(r_name))

        parser.add_argument('--model_file', type=str, default="{}{}_model.tar".format(model_folder, pca_or_cwa),
                            help='lg model for {}'.format(r_name))

        args = parser.parse_args()
        return args


if __name__ == "__main__":

    qr = QE(FilePathConfig())
    for idx, sparql in enumerate(eaqs):
        if idx != 21:
            continue
        print(sparql)
        print("{}/{}".format(idx, len(eaqs)))
        query_name = "E{}".format(idx)
        qr._setSparql(sparql)
        only_train_rule = False
        if only_train_rule:
            qr.train_rules()
        else:
            # qr.train_rules()
            qr.get_candidates(query_name)
