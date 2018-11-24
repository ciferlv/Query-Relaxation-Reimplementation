class SparqlSeg:
    def __init__(self, sparql):
        self.sparql = sparql
        self.head_vars = []
        self.body_triple_list = []
        self.body_vars = set()

    def get_head_vars(self):
        select_idx = self.sparql.find("SELECT") + 6
        where_idx = self.sparql.find("WHERE")
        self.head_vars = self.sparql[select_idx:where_idx].strip().split()

    def get_body_triples(self):
        start_idx = self.sparql.find("WHERE") + 6
        end_idx = self.sparql.find("}") - 1

        self.body_triple_list = self.sparql[start_idx:end_idx].strip().split('\n')
        self.body_triple_list = [t_p.strip() for t_p in self.body_triple_list]

        for pattern in self.body_triple_list:
            head = pattern.split()[0].strip(".")
            tail = pattern.split()[2].strip(".")
            if head.startswith("?"): self.body_vars.add(head)
            if tail.startswith("?"): self.body_vars.add(tail)

    def body_vars_str(self):
        return " ".join(list(self.body_vars))

    def analyze_sparql(self):
        self.get_head_vars()
        self.get_body_triples()


if __name__ == "__main__":
    sparql = """
    SELECT ?film
    WHERE{
        ?film <http://dbpedia.org/ontology/director> ?p.
        ?film <http://dbpedia.org/ontology/starring> ?p.
        ?p <http://dbpedia.org/ontology/birthPlace> <http://dbpedia.org/resource/North_America>.
    }"""

    sparqlSeg = SparqlSeg(sparql=sparql)
    sparqlSeg.get_head_vars()
    sparqlSeg.get_body_triples()
    print(sparqlSeg.head_vars)
    print(sparqlSeg.body_vars)
