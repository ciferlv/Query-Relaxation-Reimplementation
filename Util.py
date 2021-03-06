import os

from RuleBased.Params import uri2shorcut, shortcut2uri
from ALogger import ALogger


class Util:
    def __init__(self):
        self.logger = ALogger("UTIL.py", True).getLogger()

    def gen_prefix(self, one_uri):
        if '/' not in one_uri:
            return one_uri

        one_uri = one_uri.strip().strip("<").strip(">")

        dbo = "http://dbo_plain.org/ontology/"
        rdfs = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"
        dbr = "http://dbo_plain.org/resource/"
        dbp = "http://dbo_plain.org/property/"
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

    def createFolder(self, folder_name):
        if not os.path.exists(folder_name):
            os.makedirs(folder_name)

    def load_idx2r(self, idx2r_file):
        r2idx = {}
        idx2r = {}
        with open(idx2r_file, 'r', encoding="UTF-8") as f:
            for line in f.readlines():
                line_array = line.strip().split()
                if len(line_array) == 1:
                    continue
                idx, name = line.strip().split()
                idx = int(idx)
                r2idx[name] = idx
                idx2r[idx] = name
        return r2idx, idx2r

    '''
    Change full name uri to its shortcut
    Parameters:
    -----------
    uri: string
    for example, <http://dbo_plain.org/resource/Catalan_language>

    Returns:
    -----------
    out: string
    input uri's shortcut, for example, 'dbr:Catalan_language'
    '''

    def getShortcutName(self, uri):
        shortcut_uri = None
        uri = uri.strip().strip("<").strip(">")
        for key in uri2shorcut.keys():
            if uri.startswith(key):
                length = len(key)
                shortcut_uri = "{}:{}".format(uri2shorcut[key], uri[length:])
                break
        if shortcut_uri is None:
            return "<" + uri + ">"
        return shortcut_uri

    def getFullName(self, uri):
        for shortcut in shortcut2uri:
            tmp_shortcut = shortcut + ":"
            if uri.startswith(tmp_shortcut):
                uri = uri.replace(tmp_shortcut, shortcut2uri[shortcut])
        if not uri.startswith("<"):
            uri = "<" + uri
        if not uri.endswith(">"):
            uri = uri + ">"
        return uri

    def get_inverse_r(self, r_name):
        if r_name.startswith("inv_"):
            return r_name.replace("inv_", "")
        else:
            return "inv_" + r_name

    def format_uri(self, uri):
        uri = uri.strip().strip(".")
        if not uri.startswith("<"):
            uri = "<" + uri
        if not uri.endswith(">"):
            uri = uri + ">"
        return uri

    # def getTripleConfByName(self, e1_name, r_name, e2_name):
    #     # return self.ekg.get_triple_simi_by_name(e1_name,r_name,e2_name)
    #     data = {'e1': e1_name,
    #             'r': r_name,
    #             'e2': e2_name}
    #     response = requests.post(embedding_http, json=data)
    #     return float(response.content)
    # def rule_parser(self, raw_rule):
    #     prev = "?s"
    #     triple_pattern = ""
    #     raw_rule_array = raw_rule.strip().strip(";").split(";")
    #     for idx, pred in enumerate(raw_rule_array):
    #
    #         if idx == len(raw_rule_array) - 1:
    #             next_singal = "?e"
    #         else:
    #             next_singal = "?o" + str(idx)
    #
    #         if pred[0] is "+":
    #             triple_pattern += prev + " <" + pred[1:] + "> " + next_singal + ".\n"
    #         else:
    #             triple_pattern += next_singal + " <" + pred[1:] + "> " + prev + ".\n"
    #         prev = next_singal
    #     return triple_pattern

    # def get_entity_set_by_sparql(self, var_list, rewritted_body_triple_list):
    #     cand_list = []
    #     sparql_endpoint = SPARQLWrapper(sparql_database)
    #     sparql_endpoint.setTimeout(5)
    #
    #     rewritted_body = "\n".join(rewritted_body_triple_list)
    #     var_str = " ".join(var_list)
    #     count_query = "select count(*) where{" + rewritted_body + "}"
    #     entity_query = "select " + var_str + " where{" + rewritted_body + "}"
    #
    #     res_num = self.get_num_by_sparql(count_query)
    #     search_times = math.ceil(res_num / 10000)
    #
    #     for idx in range(search_times):
    #         try:
    #             sparql_endpoint.setQuery(entity_query + "LIMIT 10000 OFFSET " + str(idx * 10000))
    #             sparql_endpoint.setReturnFormat(JSON)
    #             results = sparql_endpoint.query().convert()
    #             binding_list = results['results']['bindings']
    #             for binding in binding_list:
    #                 cand_list.append([[var_value, binding[var_value.strip("?")]['value']] for var_value in var_list])
    #         except Exception as my_exception:
    #             self.logger.info("Can't get results: {}".format(my_exception))
    #     return cand_list
    # def get_query_path(self, body):
    #     generated_path = []
    #     entity_query = "select * where {" + body + "}"
    #
    #     sparql = SPARQLWrapper(sparql_database)
    #     sparql.setTimeout(10)
    #     sparql.setQuery(entity_query)
    #     sparql.setReturnFormat(JSON)
    #     results = sparql.query().convert()
    #     vars_list = results['head']['vars']
    #     bindings_list = results['results']['bindings']
    #     for binding in bindings_list:
    #         temp = body
    #         for var in vars_list:
    #             temp = temp.replace("?" + var, self.gen_prefix(binding[var]['value']))
    #             generated_path.append(temp)
    #     return generated_path
    #
    # def get_s_e_by_sparql(self, entity_query, count_query, extracted_num):
    #     s_e_list = []
    #     res_num = self.get_num_by_sparql(count_query)
    #     if res_num == -1:
    #         return None
    #
    #     search_times = math.ceil(res_num / 10000)
    #     extracted_num_per_time = math.ceil(extracted_num / search_times)
    #
    #     sparql = SPARQLWrapper(sparql_database)
    #     sparql.setTimeout(10)
    #
    #     self.logger.info("Search Times:{}\tExtracted num per time:{}".format(search_times, extracted_num_per_time))
    #
    #     for idx in range(search_times):
    #         if len(s_e_list) > extracted_num: break
    #         self.logger.info("No.{}".format(idx))
    #         try:
    #             sparql.setQuery(entity_query + "LIMIT 10000 OFFSET " + str(idx * 10000))
    #             sparql.setReturnFormat(JSON)
    #             results = sparql.query().convert()
    #             binding_list = results['results']['bindings']
    #             if len(binding_list) <= extracted_num_per_time:
    #                 for binding in binding_list:
    #                     s = "<" + binding['s']['value'] + ">"
    #                     e = "<" + binding['e']['value'] + ">"
    #                     s_e_list.append(Triple(searched_dict=None, subj=s, obj=e))
    #             else:
    #                 sampled_idx_list = random.sample(range(0, len(binding_list)), extracted_num_per_time)
    #                 for sampled_idx in sampled_idx_list:
    #                     s = "<" + binding_list[sampled_idx]['s']['value'] + ">"
    #                     e = "<" + binding_list[sampled_idx]['e']['value'] + ">"
    #                     s_e_list.append(Triple(searched_dict=None, subj=s, obj=e))
    #         except Exception as my_exception:
    #             self.logger.info("Can't get s e {}".format(my_exception))
    #             return None
    #     if len(s_e_list) > extracted_num:
    #         s_e_list = s_e_list[0:extracted_num]
    #     return s_e_list

    # def ask_sparql(self, query):
    #     sparql = SPARQLWrapper(sparql_database)
    #     sparql.setTimeout(10)
    #     try:
    #         sparql.setQuery(query)
    #         sparql.setReturnFormat(JSON)
    #         results = sparql.query().convert()
    #         return int(results['boolean'])
    #     except Exception as my_exception:
    #         self.logger.info("Can't get num {}.".format(my_exception))
    #         return 0
    # def get_num_by_sparql(self, query):
    #     sparql = SPARQLWrapper(sparql_database)
    #     sparql.setTimeout(3)
    #     try:
    #         sparql.setQuery(query)
    #         sparql.setReturnFormat(JSON)
    #         results = sparql.query().convert()
    #         res_num = int(results['results']['bindings'][0]['callret-0']['value'])
    #         return res_num
    #     except Exception as my_exception:
    #         self.logger.info("Can't get num {}.".format(my_exception))
    #         return -1
