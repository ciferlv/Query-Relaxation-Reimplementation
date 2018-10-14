class SparqlSeg:
    def __init__(self, sparql):
        self.sparql = sparql
        self.variables = []
        self.triple_patterns = []

    def get_variables(self):
        select_idx = self.sparql.find("SELECT") + 6
        where_idx = self.sparql.find("WHERE")
        self.variables = self.sparql[select_idx:where_idx].strip().split()

    def get_triple_patterns(self):
        start_idx = self.sparql.find("WHERE") + 6
        end_idx = self.sparql.find("}") - 1
        self.triple_patterns = self.sparql[start_idx:end_idx].strip().split('\n')
        self.triple_patterns = [t_p.strip() for t_p in self.triple_patterns]


if __name__ == "__main__":
    sparql = """
    SELECT ?film
    WHERE{
        ?film <http://dbpedia.org/ontology/director> ?p.
        ?film <http://dbpedia.org/ontology/starring> ?p.
        ?p <http://dbpedia.org/ontology/birthPlace> <http://dbpedia.org/resource/North_America>.
    }"""

    sparqlSeg = SparqlSeg(sparql=sparql)
    sparqlSeg.get_variables()
    sparqlSeg.get_triple_patterns()
