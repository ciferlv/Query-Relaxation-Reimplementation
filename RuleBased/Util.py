from SPARQLWrapper import SPARQLWrapper, JSON

sparql_database = "http://210.28.132.61:8898/sparql"


class Util:
    def get_num_by_sparql(self, query):
        self.sparql = SPARQLWrapper(sparql_database)
        self.sparql.setTimeout(3)
        try:
            self.sparql.setQuery(query)
            self.sparql.setReturnFormat(JSON)
            results = self.sparql.query().convert()
            res_num = int(results['results']['bindings'][0]['callret-0']['value'])
            return res_num
        except:
            return -1


if __name__ == "__main__":
    utils = Util()
    query = """select count(?s)where{ ?s <http://dbpedia.org/property/birthPlace> ?o0.
?o0 <http://dbpedia.org/property/seat> ?e.
?s <http://dbpedia.org/ontology/birthPlace>?e.}"""
    print(utils.get_num_by_sparql(query))
