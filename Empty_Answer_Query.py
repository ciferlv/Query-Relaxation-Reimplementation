# eaqs = [
#     # 1.Who is the advisor of Newton?
#     """
#     SELECT * WHERE{
#         dbr:Yann_LeCun dbo:doctoralAdvisor ?p.}""",
#     # 2. Name the time zone of Glasgow.
#     """
#     SELECT * WHERE {
#         dbr:Glasgow dbo:timeZone ?uri}""",
#     # 3. Where is Apple Company?
#     """
#     SELECT * WHERE {
#         dbr:Apple_Inc. dbo:locationCountry ?uri.}""",
#     # 4. Name the origin of AMD_Alarus?
#     """
#     SELECT * WHERE {
#         dbr:AMD_Alarus dbo:origin ?uri.}""",
#     # 5. What is the owner of Singtel?
#     """
#     SELECT * WHERE {
#         ?q dbo:owner dbr:Singtel}""",
#     # 6. Who was born in USA and awarded Fellow of the Royal Society of Canada
#     """
#     select * where {
#         ?a dbo:birthPlace dbr:United_States.
#         ?a dbo:award dbr:Fellow_of_the_Royal_Society_of_Canada.}""",
#     # 7. Name the organizations that locate in Italy and serve Asia.
#     """
#     select * where {
#             ?p dbo:regionServed dbr:Asia.
#             ?p dbo:location dbr:Italy}""",
#     # 8. Name the organizations that are founded in England and product automobiles.
#     """
#     select * where {
#         ?p dbo:product dbr:Automobile.
#         ?p dbo:foundationPlace dbr:England.}""",
#     # 9. What is the official residence of the Lula J. Davis which is also the death location of the John McTaggart (jockey)?
#     """
#     SELECT * WHERE {
#         dbr:Lula_J._Davis dbo:residence ?uri.
#         dbr:John_McTaggart_(jockey) dbo:deathPlace ?uri} """,
#     # 10. Who starred in Gift of the Night Fury and was born in the USA?
#     """
#     select * where {
#         dbr:Gift_of_the_Night_Fury dbo:starring ?p.
#         ?p dbo:birthPlace  dbr:United_States.}""",
#     # 11. Name the works that locate in New York and their actors.
#     """
#     select * where {
#         ?p dbo:starring ?o.
#         ?p dbo:location dbr:Province_of_New_York.}""",
#     # 12. Name the actors that live in New York and their films.
#     """
#     select * where{
#         ?film dbo:starring ?p.
#         ?p dbo:residence dbr:Province_of_New_York.} """,
#
#     # 13. Show companies which serves Pacific Ocean and other regions served by them.
#     """
#     SELECT * WHERE {
#         ?x dbo:regionServed dbr:Pacific_Ocean.
#         ?x dbo:regionServed ?uri }""",
#     # 14. Where does the publisher of Game of Thrones live?
#     """SELECT * WHERE {
#         dbr:Game_of_Thrones dbo:publisher ?x .
#         ?x dbo:residence ?uri. }""",
#     # 15. What is the foundation place of companies owned by Google?
#     """
#     SELECT * WHERE {
#         ?x dbo:owner dbr:Google .
#         ?x dbo:foundationPlace ?uri.}""",
#     # 16
#     """
#     SELECT * WHERE {
#         dbr:Isaac_Newton dbo:doctoralAdvisor ?p.}""",
#     # 17
#     """
#     SELECT * WHERE {
#         dbr:Paris dbo:timeZone ?p.
#     }""",
#     # 18
#     """
#     SELECT * WHERE {
#         dbr:Amazon.com dbo:locationCountry ?p.
#     }
#     """,
#     # 19
#     """
#     SELECT * WHERE {
#         dbr:Jeep_Wrangler dbo:origin ?p.
#     }
#     """,
#     # 20
#     """
#     SELECT * WHERE {
#     dbr:Carmel_Winery dbo:locationCountry ?uri.}
#     """,
#
#     # 21 Name the organizations that are founded in Canada and are belong to Jim Pattison Group.
#     """
#     select * where {
#     ?p dbo:owner dbr:Jim_Pattison_Group.
#     ?p dbo:foundationPlace dbr:Canada.}
#     """,
#
#     # 22. Name the organizations that are founded in England and product automobiles.
#
#     """
#     select * where {
#     ?p dbo:product dbr:Automobile.
#     ?p dbo:foundationPlace dbr:Italy.}
#     """,
#
#     # 23. Show me the DVD producers that serve South Australia.
#     """
#      select * where {
#         ?p dbo:regionServed  dbr:South_Australia.
#         ?p dbo:product  dbr:DVD.}
#     """,
#
#     # 24. Name the organizations that serve South Australia and locate in Sydney.
#     """
#     select * where {
#     ?p dbo:locationCountry dbr:Canada.
#     ?p dbo:foundationPlace dbr:United_States.}
#     """,
#
#     # 25. Name the organizations that serve Texas and are belong to Google.
#     """
#     select * where {
#         ?p dbo:regionServed  dbr:Texas.
#         ?p dbo:owner  dbr:Google.}
#     """
# ]

eaqs = [

    # 1.Who is the advisor of Newton?
    """
    SELECT * WHERE{
        dbr:Yann_LeCun dbo:doctoralAdvisor ?p.}""",

    # 2. Name the time zone of Glasgow.
    """
    SELECT * WHERE { 
        dbr:Glasgow dbo:timeZone ?uri}""",

    # 3. Where is Apple Company?
    """
    SELECT * WHERE { 
        dbr:Apple_Inc. dbo:locationCountry ?uri.}""",

    # 4. Name the origin of AMD_Alarus?
    """
    SELECT * WHERE { 
        dbr:AMD_Alarus dbo:origin ?uri.}""",

    # 5. What is the owner of Singtel?
    """
    SELECT * WHERE {
        ?q dbo:owner dbr:Singtel}""",

    # 6. Who is the advisor of Newton?
    """
    SELECT * WHERE {
        dbr:Isaac_Newton dbo:doctoralAdvisor ?p.}""",

    # 7. What is the timezone of Paris?
    """
    SELECT * WHERE {
        dbr:Paris dbo:timeZone ?p.}""",

    # 8. Which country does Amazon.com locate?
    """
    SELECT * WHERE {
        dbr:Amazon.com dbo:locationCountry ?p.}""",

    # 9. What is the origin of Jeep Wrangler?
    """
    SELECT * WHERE {
        dbr:Jeep_Wrangler dbo:origin ?p.}""",

    # 10. Which country does Carmel Winery locate?
    """
    SELECT * WHERE { 
        dbr:Carmel_Winery dbo:locationCountry ?uri.}""",

    # 11. Who was born in USA and awarded Fellow of the Royal Society of Canada
    """
    select * where {
        ?a dbo:birthPlace dbr:United_States.
        ?a dbo:award dbr:Fellow_of_the_Royal_Society_of_Canada.}""",

    # 12. Name the organizations that locate in Italy and serve Asia.
    """
    select * where {
            ?p dbo:regionServed dbr:Asia.
            ?p dbo:location dbr:Italy}""",

    # 13. Name the organizations that are founded in England and product automobiles.
    """
    select * where {
        ?p dbo:product dbr:Automobile.
        ?p dbo:foundationPlace dbr:England.}""",

    # 14. What is the official residence of the Lula J. Davis which is also the death location of the John McTaggart (jockey)?
    """
    SELECT * WHERE { 
        dbr:Lula_J._Davis dbo:residence ?uri. 
        dbr:John_McTaggart_(jockey) dbo:deathPlace ?uri} """,

    # 15. Who starred in Gift of the Night Fury and was born in the USA?
    """
    select * where {
        dbr:Gift_of_the_Night_Fury dbo:starring ?p.
        ?p dbo:birthPlace  dbr:United_States.}""",

    # 16. Name the organizations that are founded in Canada and are belong to Jim Pattison Group.
    """
    select * where {
        ?p dbo:owner dbr:Jim_Pattison_Group.
        ?p dbo:foundationPlace dbr:Canada.}""",

    # 17. Name the organizations that are founded in England and product automobiles.
    """
    select * where {
        ?p dbo:product dbr:Automobile.
        ?p dbo:foundationPlace dbr:Italy.}""",

    # 18. Show me the DVD producers that serve South Australia.
    """
     select * where {
        ?p dbo:regionServed  dbr:South_Australia.
        ?p dbo:product  dbr:DVD.}""",

    # 19. Name the organizations that serve South Australia and locate in Sydney.
    """
    select * where {
        ?p dbo:locationCountry dbr:Canada.
        ?p dbo:foundationPlace dbr:United_States.}""",

    # 20. Name the organizations that serve Texas and are belong to Google.
    """
    select * where {
        ?p dbo:regionServed  dbr:Texas.
        ?p dbo:owner  dbr:Google.}""",

    # 21. Name the works that locate in New York and their actors.
    """
    select * where {
        ?p dbo:starring ?o.
        ?p dbo:location dbr:Province_of_New_York.}""",

    # 22. Name the actors that live in New York and their films.
    """
    select * where{
        ?film dbo:starring ?p.
        ?p dbo:residence dbr:Province_of_New_York.} """,

    # 23. Show companies which serves Pacific Ocean and other regions served by them.
    """
    SELECT * WHERE { 
        ?x dbo:regionServed dbr:Pacific_Ocean. 
        ?x dbo:regionServed ?uri }""",

    # 24. Where does the publisher of Game of Thrones live?
    """SELECT * WHERE { 
        dbr:Game_of_Thrones dbo:publisher ?x . 
        ?x dbo:residence ?uri. }""",

    # 25. What is the foundation place of companies owned by Google?
    """
    SELECT * WHERE { 
        ?x dbo:owner dbr:Google .
        ?x dbo:foundationPlace ?uri.}"""
]

if __name__ == "__main__":
    a_list = ["dbr:Google",
              "dbr:Game_of_Thrones",
              "dbr:Pacific_Ocean",
              "dbr:Province_of_New_York",
              "dbr:Texas",
              "dbr:Canada",
              "dbr:United_States",
              "dbr:South_Australia",
              "dbr:DVD",
              "dbr:Italy",
              "dbr:Automobile",
              "dbr:Canada",
              "dbr:Jim_Pattison_Group",
              "dbr:United_States",
              "dbr:Gift_of_the_Night_Fury",
              "dbr:John_McTaggart_(jockey)",
              "dbr:Lula_J._Davis",
              "dbr:England",
              "dbr:Automobile",
              "dbr:Asia",
              "dbr:Italy",
              "dbr:Fellow_of_the_Royal_Society_of_Canada",
              "dbr:United_States",
              "dbr:Carmel_Winery",
              "dbr:Jeep_Wrangler",
              "dbr:Amazon.com",
              "dbr:Paris",
              "dbr:Isaac_Newton",
              "dbr:Singtel",
              "dbr:AMD_Alarus",
              "dbr:Apple_Inc.",
              "dbr:Glasgow",
              "dbr:Yann_LeCun"]
    print(len(list(set(a_list))))
