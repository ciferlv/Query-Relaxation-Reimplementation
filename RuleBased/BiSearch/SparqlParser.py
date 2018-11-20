import queue

from RuleBased.BiSearch.Triple import Candidate
from RuleBased.Params import ht_conn


class SparqlParser:
    def __init__(self, sparql):
        self.sparql = sparql
        self.sparql_BGP = []
        self.var2entity = {}
        self.r_name_list = []
        '''[var1,var2,var3], sorted by alphabet order'''
        self.var_list = []
        '''
        list of BGP which has two variable
        key: token => h,t, h and t are sorted by alphabet order
        value: list of splited BGP => [[h,r,t],[h,r,t],....] 
        '''
        self.var2BGP = {}

        '''
        list of BGP which has one variable
        key: h or t, it depends on which is varible
        value: list of splited BGP => [[h,r,t],[h,r,t],...]
        '''
        self.var1BGP = {}

        '''
        [
            [
                [var1],[var2],[var3]
            ],
            [
                [var1],[var2],[var3]
            ],
            ....
        ]
        '''
        self.res = []
        self.temp_res = []
        self.cand_obj_list = []

    def parse_sparql(self):
        body_start_index = self.sparql.find("{")
        body_end_index = self.sparql.find("}")
        body = self.sparql[body_start_index + 1:body_end_index].strip()
        for BGP in body.split("\n"):
            BGP = BGP.strip().strip(".")
            head, relation, tail = BGP.split()
            self.sparql_BGP.append([head, relation, tail])
            self.r_name_list.append(relation)
            if head.startswith('?') and tail.startswith("?"):
                token = ht_conn.join([head, tail])
                if token not in self.var2BGP:
                    self.var2BGP[token] = []
                self.var2BGP[token].append([head, relation, tail])
            elif head.startswith("?") and not tail.startswith("?"):
                if head not in self.var1BGP:
                    self.var1BGP[head] = []
                self.var1BGP[head].append([head, relation, tail])
            elif not head.startswith("?") and tail.startswith("?"):
                if tail not in self.var1BGP:
                    self.var1BGP[tail] = []
                self.var1BGP[tail].append([head, relation, tail])
            if head.startswith("?"):
                self.var_list.append(head)
            if tail.startswith("?"):
                self.var_list.append(tail)
        self.r_name_list = list(set(self.r_name_list))
        self.var_list = list(set(self.var_list))
        self.var_list.sort()

        for var in self.var_list:
            self.var2entity[var] = set()
            self.temp_res.append([])

    '''
    Update results searched. This is used for BGP with 2 variables.
    Parameters:
    -----------
    h_var: string, name of head variable
    t_var: string, name of tail variable
    passed_ht_list: list, list of passed ht, [[h,t],[h,t],...]
    passed_ht_token_set: set, set of tokens of passed ht, token => h,t
    
    Returns:
    -----------
    None
    Update self.var2entity[h_var], self.var2entity[t_var] and res.
    '''

    def update_res_var2entity(self, h_var, t_var, passed_ht_list, passed_ht_token_set):
        h_res_idx = self.var_list.index(h_var)
        t_res_idx = self.var_list.index(t_var)
        h_set = set()
        t_set = set()

        if len(self.res) == 0:
            for ht in passed_ht_list:
                copy_res = self.temp_res[:]
                copy_res[h_res_idx] = [ht[0]]
                copy_res[t_res_idx] = [ht[-1]]
                self.res.append(copy_res)
                h_set.add(ht[0])
                t_set.add(ht[-1])
        else:
            temp_store = []
            for one_res in self.res:
                h = one_res[h_res_idx][0]
                t = one_res[t_res_idx][0]
                if ht_conn.join([h, t]) in passed_ht_token_set:
                    temp_store.append(one_res)
                    h_set.add(h)
                    t_set.add(t)
            self.res = temp_store

        self.var2entity[h_var] = h_set
        self.var2entity[t_var] = t_set

    '''
    Get candidates of var_name
    Parameters:
    -----------
    var_name: string, for example, ?p
    the name of variable
    
    Returns:
    out: list
    list of candidates of var_name
    '''

    def get_candidate_by_var(self, var_name):
        return self.var2entity[var_name]

    def execute_var1BGP(self, r_rules_dict, graph):
        for var in self.var1BGP:
            for BGP in self.var1BGP[var]:
                if BGP[0].startswith("?"):
                    h_idx_list = []
                    t_idx_list = [graph.e2idx[BGP[2]]]
                    idx_of_var = 0
                else:
                    h_idx_list = [graph.e2idx[BGP[0]]]
                    t_idx_list = []
                    idx_of_var = 1

                rule_path_list = [[graph.r2idx[BGP[1]]]]
                rule_path_list.extend([rule_obj.r_path for rule_obj in r_rules_dict[graph.r2idx[BGP[1]]]])

                passed_ht_token = set()
                passed_ht = []
                for rule_path in rule_path_list:
                    temp_passed_ht = graph.get_ht_from_one_end(h_idx_list, t_idx_list, rule_path, passed_ht_token)
                    passed_ht.extend(temp_passed_ht)

                temp_res = set([ht[idx_of_var] for ht in passed_ht])
                self.var2entity[var] = temp_res if len(self.var2entity[var]) == 0 else temp_res & self.var2entity[var]

    def execute_var2BGP(self, r_rules_dict, graph):
        BGP_queue = queue.Queue()
        [BGP_queue.put(token) for token in self.var2BGP]

        while not BGP_queue.empty():
            token = BGP_queue.get()
            for BGP in self.var2BGP[token]:
                print("Start executin {}".format(" ".join(BGP)))
                h_var, r_name, t_var = BGP
                h_idx_list = list(self.var2entity[h_var])
                t_idx_list = list(self.var2entity[t_var])

                rule_path_list = [[graph.r2idx[r_name]]]
                rule_path_list.extend([rule_obj.r_path for rule_obj in r_rules_dict[graph.r2idx[r_name]]])

                if len(h_idx_list) == 0 and len(t_idx_list) == 0:
                    BGP_queue.put(token)
                    print("size of h_idx_list and size of t_idx_list are zero.")
                    continue

                if len(h_idx_list) == 0 or len(t_idx_list) == 0:
                    passed_ht_token_set = set()
                    for rule in rule_path_list:
                        graph.get_ht_from_one_end(h_idx_list, t_idx_list, rule, passed_ht_token_set)

                    passed_ht = [list(map(int, ht_token.split(ht_conn))) for ht_token in passed_ht_token_set]
                    self.update_res_var2entity(h_var, t_var, passed_ht, passed_ht_token_set)
                else:
                    passed_ht_list, passed_ht_token_set = graph.pass_verify(h_idx_list, t_idx_list, rule_path_list)
                    self.update_res_var2entity(h_var, t_var, passed_ht_list, passed_ht_token_set)

    def gen_confidence(self, r_rules_dict, rule_model_dict, graph):
        for cand in self.res:
            cand_obj = Candidate(cand)
            for one_bgp in self.sparql_BGP:
                h_name = one_bgp[0]
                r_idx = graph.r2idx[one_bgp[1]]
                t_name = one_bgp[2]

                h_name = cand[self.var_list.index(h_name)] if h_name.startswith("?") else h_name
                t_name = cand[self.var_list.index(t_name)] if t_name.startswith("?") else t_name

                one_bgp_str = "{}\t{}\t{}".format(h_name, one_bgp[1], t_name)
                cand_obj.add_complished_bgp(one_bgp_str)

                h_idx = graph.e2idx[h_name]
                t_idx = graph.e2idx[t_name]

                rule_list = r_rules_dict[r_idx]
                features, res_path = graph.get_passed_e_r_path(h_idx, t_idx, rule_list)
                pra_conf = rule_model_dict[r_idx].get_output_prob(features)
                cand_obj.add_path_idx_for_bgp(res_path, one_bgp_str)
                cand_obj.add_pra_conf_for_bgp(pra_conf, one_bgp_str)
            self.cand_obj_list.append(cand_obj)

    def display_cands(self, graph):
        for cand in self.cand_obj_list:
            print(cand.display_var2entity(self.var_list))
            print(cand.display_rule_path(graph))
        print()


if __name__ == "__main__":
    sparql = """
    SELECT ?film WHERE{
        ?film <http://dbpedia.org/ontology/director> ?p.
        ?film <http://dbpedia.org/ontology/starring> ?p.
        ?p <http://dbpedia.org/ontology/birthPlace> <http://dbpedia.org/resource/North_America>.
    }
    """

    sp = SparqlParser(sparql=sparql)
    sp.parse_sparql()
