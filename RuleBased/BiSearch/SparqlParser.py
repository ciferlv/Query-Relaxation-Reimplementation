
class SparqlParser:
    def __init__(self,sparql):
        self.sparql = sparql
        self.fixed_var_cnt = {}
        self.relation2ht = {}
        self.var2entity = {}

    def parse_sparql(self):
        body_start_index = self.sparql.find("{")
        body_end_index = self.sparql.find("}")
        body = self.sparql[body_start_index:body_end_index].strip()
        for BGP in body.split("."):
            head,relation,tail = BGP.split()

