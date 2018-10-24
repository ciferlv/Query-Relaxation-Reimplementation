from SPARQLWrapper import SPARQLWrapper, JSON
import math
import random

from RuleBased.ALogger import ALogger
from RuleBased.unit.Triple import Triple

sparql_database = "http://210.28.132.61:8898/sparql"


class Util:
    def __init__(self):
        self.logger = ALogger("UTIL.py", True).getLogger()

    def get_num_by_sparql(self, query):
        sparql = SPARQLWrapper(sparql_database)
        sparql.setTimeout(3)
        try:
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            res_num = int(results['results']['bindings'][0]['callret-0']['value'])
            return res_num
        except Exception as my_exception:
            self.logger.info("Can't get num {}.".format(my_exception))
            return -1

    def get_s_e_by_sparql(self, entity_query, count_query, extracted_num):
        s_e_list = []
        res_num = self.get_num_by_sparql(count_query)
        if res_num == -1:
            return None

        search_times = math.ceil(res_num / 10000)
        extracted_num_per_time = math.ceil(extracted_num / search_times)

        if search_times > extracted_num: search_times = extracted_num

        sparql = SPARQLWrapper(sparql_database)
        sparql.setTimeout(10)

        self.logger.info("Search Times:{}\tExtracted num per time:{}".format(search_times, extracted_num_per_time))

        for idx in range(search_times):
            self.logger.info("No.{}".format(idx))
            try:
                sparql.setQuery(entity_query + "LIMIT 10000 OFFSET " + str(idx * 10000))
                sparql.setReturnFormat(JSON)
                results = sparql.query().convert()
                binding_list = results['results']['bindings']
                if len(binding_list) <= extracted_num_per_time:
                    for binding in binding_list:
                        s = "<" + binding['s']['value'] + ">"
                        e = "<" + binding['e']['value'] + ">"
                        s_e_list.append(Triple(searched_dict=None, subj=s, obj=e))
                else:
                    sampled_idx_list = random.sample(range(0, len(binding_list)), extracted_num_per_time)
                    for sampled_idx in sampled_idx_list:
                        s = "<" + binding_list[sampled_idx]['s']['value'] + ">"
                        e = "<" + binding_list[sampled_idx]['e']['value'] + ">"
                        s_e_list.append(Triple(searched_dict=None, subj=s, obj=e))
            except Exception as my_exception:
                self.logger.info("Can't get s e {}".format(my_exception))
                return None
        if len(s_e_list) > extracted_num:
            s_e_list = s_e_list[0:extracted_num]
        return s_e_list

    def ask_sparql(self, query):
        sparql = SPARQLWrapper(sparql_database)
        sparql.setTimeout(10)
        try:
            sparql.setQuery(query)
            sparql.setReturnFormat(JSON)
            results = sparql.query().convert()
            return int(results['boolean'])
        except Exception as my_exception:
            self.logger.info("Can't get num {}.".format(my_exception))
            return 0

    def generate_name(self, one_uri):
        if '/' not in one_uri:
            return one_uri

        one_uri = one_uri.strip().strip("<").strip(">")

        dbo = "http://dbpedia.org/ontology/"
        rdfs = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        dbr = "http://dbpedia.org/resource/"
        dbp = "http://dbpedia.org/property/"
        localname = one_uri.split("/")[-1]
        if "#" in localname:
            localname = localname.split("#")[-1]
        if one_uri.startswith(dbo):
            name = "dbo_" + localname
        elif one_uri.startswith(rdfs):
            name = "rdfs_" + localname
        elif one_uri.startswith(dbr):
            name = "dbr_" + localname
        elif one_uri.startswith(dbp):
            name = "dbp_" + localname
        else:
            name = localname
        return name

    def rule_parser(self, raw_rule):
        prev = "?s"
        triple_pattern = ""
        raw_rule_array = raw_rule.strip().strip(";").split(";")
        for idx, pred in enumerate(raw_rule_array):

            if idx == len(raw_rule_array) - 1:
                next_singal = "?e"
            else:
                next_singal = "?o" + str(idx)

            if pred[0] is "+":
                triple_pattern += prev + " <" + pred[1:] + "> " + next_singal + ".\n"
            else:
                triple_pattern += next_singal + " <" + pred[1:] + "> " + prev + ".\n"
            prev = next_singal
        return triple_pattern

    def get_entity_set_by_sparql(self, var_list, rewritted_body_triple_list, entity_set):
        sparql_endpoint = SPARQLWrapper(sparql_database)
        sparql_endpoint.setTimeout(5)

        rewritted_body = "\n".join(rewritted_body_triple_list)
        var_str = " ".join(var_list)
        count_query = "select count(*) where{" + rewritted_body + "}"
        entity_query = "select " + var_str + " where{" + rewritted_body + "}"

        res_num = self.get_num_by_sparql(count_query)
        search_times = math.ceil(res_num / 10000)

        for idx in range(search_times):
            try:
                sparql_endpoint.setQuery(entity_query + "LIMIT 10000 OFFSET " + str(idx * 10000))
                sparql_endpoint.setReturnFormat(JSON)
                results = sparql_endpoint.query().convert()
                binding_list = results['results']['bindings']
                for binding in binding_list:
                    one_result = []
                    for variable in var_list:
                        one_result.append(binding[variable.strip("?")]['value'])
                    entity_set.add(";".join(one_result))
            except Exception as my_exception:
                self.logger.info("Can't get results: {}".format(my_exception))


if __name__ == "__main__":
    utils = Util()
    query = """where { ?s <http://dbpedia.org/property/birthPlace> ?e.
                FILTER EXISTS {?s <http://dbpedia.org/ontology/birthPlace> ?random}. 
                FILTER regex(?e,\"^http\"). 
                MINUS {?s <http://dbpedia.org/ontology/birthPlace> ?e}
                }"""
    count_query = "select count(?s) " + query
    entity_query = "select ?s ?e " + query
    s_e_list = utils.get_s_e_by_sparql(entity_query, count_query, 20)
    print("{}".format(len(s_e_list)))
    for s_e in s_e_list:
        print(s_e)
