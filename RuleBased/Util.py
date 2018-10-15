from SPARQLWrapper import SPARQLWrapper, JSON
import math
import random
import logging

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

        sparql = SPARQLWrapper(sparql_database)
        sparql.setTimeout(10)

        self.logger.info("Search Times:{}\tExtracted num per time:{}".format(search_times,extracted_num_per_time))

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
