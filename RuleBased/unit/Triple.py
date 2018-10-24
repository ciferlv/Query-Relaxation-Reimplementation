from SPARQLWrapper import SPARQLWrapper, JSON

sparql_database = "http://210.28.132.61:8898/sparql"


class Triple:
    def __init__(self, searched_dict=None, subj=None, obj=None):
        self.sparql = SPARQLWrapper(sparql_database)
        if searched_dict is None:
            self.subj = subj
            self.obj = obj
        else:
            self.subj = searched_dict['s']['value']
            self.obj = searched_dict['o']['value']
        self.rule_set = set()
        self.check_subj_obj()

    def check_subj_obj(self):
        if not self.subj.startswith("<"):
            self.subj = "<" + self.subj
        if not self.obj.startswith("<"):
            self.obj = "<" + self.obj
        if not self.subj.endswith(">"):
            self.subj = self.subj + ">"
        if not self.obj.endswith(">"):
            self.obj = self.obj + ">"

    def get_pred(self, sparql, position):
        self.sparql.setQuery(sparql)
        self.sparql.setReturnFormat(JSON)
        results = self.sparql.query().convert()
        for res in results['results']['bindings']:
            one_rule = ""
            for i, key in enumerate(res.keys()):
                one_rule += position[i] + res[key]['value'] + ";"
            self.rule_set.add(one_rule)

    def display_rule(self):
        for one_rule in self.rule_set:
            print(one_rule)

    def search_rule(self):
        query_filter = "FILTER regex(?o, \"^http\"). FILTER (?o != " + self.subj + "). FILTER (?o != " + self.obj + ").}"
        query0 = "select ?p1 where { " + self.subj + " ?p1 " + self.obj + "}"
        query1 = "select ?p1 where { " + self.obj + " ?p1 " + self.subj + "}"
        query2 = "select ?p1 ?p2 where{ " + self.subj + " ?p1 ?o.\n?o ?p2 " + self.obj + "." + query_filter
        query3 = "select ?p1 ?p2 where{ " + self.subj + " ?p1 ?o.\n" + self.obj + " ?p2 ?o." + query_filter
        query4 = "select ?p1 ?p2 where{ ?o ?p1 " + self.subj + ".\n" + self.obj + " ?p2 ?o." + query_filter
        query5 = "select ?p1 ?p2 where{ ?o ?p1 " + self.subj + ".\n ?o ?p2" + self.obj + "." + query_filter

        print(query0)
        self.get_pred(query0, ['+'])
        self.get_pred(query1, ['-'])
        self.get_pred(query2, ['+', '+'])
        self.get_pred(query3, ['+', '-'])
        self.get_pred(query4, ['-', '-'])
        self.get_pred(query5, ['-', '+'])

    def __str__(self):
        msg = self.subj + "\t" + self.obj
        return msg
