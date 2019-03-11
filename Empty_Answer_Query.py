eaqs = [
    # 0
    """
    SELECT * WHERE{
        dbr:Isaac_Newton dbo:doctoralAdvisor ?p.}""",
    # 1
    """
    select * where {
        ?a dbo:birthPlace dbr:United_States.
        ?a dbo:award dbr:Fellow_of_the_Royal_Society_of_Canada.}""",
    # 2
    """
    select * where {
        ?a dbo:draftTeam dbr:New_York_Knicks.
        ?a dbo:birthPlace dbo:United_States.}""",
    # 3
    """
    select * where {
        ?a dbo:draftTeam dbr:New_York_Knicks.
        ?a dbo:birthPlace dbr:Canada.}""",
    # 4
    """
    select * where {
        dbr:Castleton_Lyons dbo:keyPerson ?a.
        ?a dbo:birthPlace dbr:United_Kingdom.}""",
    # 5
    """
    select * where {
        dbr:Gift_of_the_Night_Fury dbo:starring ?p.
        ?p dbo:birthPlace  dbr:United_States.}""",
    # 6
    """
    select * where {
        ?p dbo:starring ?o.
        ?p dbo:location dbr:Province_of_New_York.}""",
    # 7
    """
    select * where {
            ?p dbo:regionServed dbr:Asia.
            ?p dbo:location dbr:Italy}""",
    # 8
    """
    select * where {
        ?p dbo:regionServed dbr:Norway.
        ?p dbo:industry dbr:Bus_transport}""",
    # 9
    """
    select * where {
        ?p dbo:regionServed dbr:Alaska.
        ?p dbo:industry  dbr:Financial_services.}""",
    # 10
    """
    select * where {
        ?p dbo:regionServed  dbr:Texas.
        ?p dbo:owner  dbr:Google.}""",
    # 11
    """
    select * where {
        ?p dbo:regionServed  dbr:South_Australia.
        ?p dbo:locationCity  dbr:Sydney.}""",
    # 12
    """
    select * where {
        ?p dbo:regionServed  dbr:South_Australia.
        ?p dbo:product  dbr:DVD.}""",
    # 13
    """
    select * where{
        ?film dbo:starring ?p.
        ?p dbo:residence dbr:Province_of_New_York.} """,
    # 14
    """
    select * where {
        ?p dbo:product dbr:Automobile.
        ?p dbo:foundationPlace dbr:England.}""",
    # 15
    """
    select * where {
        ?p dbo:owner dbr:Jim_Pattison_Group.
        ?p dbo:foundationPlace dbr:Canada.}""",
    # 16
    """
    select * where {
        ?p dbo:locationCity dbr:Sheffield.
        ?p dbo:foundationPlace dbr:United_States.}"""
]
