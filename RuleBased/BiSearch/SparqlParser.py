from RuleBased.Params import ht_conn


class SparqlParser:
    def __init__(self, sparql):
        self.sparql = sparql
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

    def parse_sparql(self):
        body_start_index = self.sparql.find("{")
        body_end_index = self.sparql.find("}")
        body = self.sparql[body_start_index + 1:body_end_index].strip()
        for BGP in body.split("\n"):
            BGP = BGP.strip().strip(".")
            head, relation, tail = BGP.split()
            self.r_name_list.append(relation)
            if head.startswith('?') and tail.startswith("?"):
                token = ht_conn.join([head, tail].sort())
                if token not in self.var2BGP:
                    self.var2BGP[token] = []
                self.var2BGP.append([head, relation.tail])
            elif head.startswith("?") and not tail.startswith("?"):
                if head not in self.var1BGP:
                    self.var1BGP[head] = []
                self.var1BGP.append([head, relation, tail])
            elif not head.startswith("?") and tail.startswith("?"):
                if tail not in self.var1BGP:
                    self.var1BGP[tail] = []
                self.var1BGP.append([head, relation, tail])
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
    
    '''
    def update_res_var2entity(self, h_var, t_var, passed_ht_list, passed_ht_token_set):
        h_res_idx = sp.var_list.index(h_var)
        t_res_idx = sp.var_list.index(t_var)
        h_set = set()
        t_set = set()

        if len(self.res) == 0:
            for ht in passed_ht_list:
                copy_res = self.temp_res
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

    def get_candidate_by_var(self, var_name):
        return self.var2entity[var_name]


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
    print(sp.fixed_var_cnt)
