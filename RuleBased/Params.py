import mysql.connector
import sys

rule_seg = "=>"  # break up the rule, birthPlace <= birthPlace:location
ht_conn = ","  # connect h and t, [Microsoft,Seattle] ==> "Microsoft,Seattle"
ht_seg = ";"  # break up between different hts, [[Micorsoft,Seattle],[Apple,LA]] ==> "Microsoft,Seattel;Apple,LA"
file_path_seg = '/'

# pca_or_cwa = 'pca'
pca_or_cwa = 'cwa'

'''
Parameters for classifier:
'''
epoch_num = 1000
batch_size = 128
'''
Parameters for rule
'''
max_step = 3
rule_num4train = 1000

'''
For a candidate, there may be many rules for a BGP, this param determines the num to display.
'''
rule_num2display = 3
'''
Filter the pattern relation=>inv_relation and inv_relation=>relation
'''
filter_inv_pattern = True
'''
When get passed ht for some rule, 
for example, dbo:distributor=>dbo:locationCountry=>inv_dbo:birthPlace
may cost so much time and the results is always nonsense,
so make a time limit, abandon the rule if time exceeds.
'''
check_time_for_get_passed_ht = True
time_limit_for_get_passed_ht = 20
'''
When searching passed ht for rule, there may be too much branch, 
so we limit the number of nodes from left path and right path
'''
limit_branch_node_num = False
branch_node_limit = 10000
'''
Sampled posis or neges may be too much,
we may need to restrain the number of them.
'''
restrain_num_of_posis_neges = True
# restrain_num = 5000
restrain_num = 1000
# '''
# When test a test_model, we want to sample neges from every relation,
# this parameter control the number of neges to sample.
# '''
# test_nege_sample_num = 10
'''
This parameter determines the num of rules to ues to search candidates.
'''
rules_num_to_search_cands = 50
# rules_num_to_search_cands = 100
# rules_num_to_search_cands = 200
# rules_num_to_search_cands = 400
# rules_num_to_search_cands = 800
# rules_num_to_search_cands = 1000
'''
sort rule by criterion
'''
sort_rule_criterion = 'F1'
'''
sort bgp candidate by criterion
'''
# sort_candidate_criterion = 'pra'
# sort_candidate_criterion = 'embedding'
sort_candidate_criterion = 'pra_embedding'
'''
Set the numbers to display of candidates
'''
numbers_to_display_of_cands = 10

dbpedia_folder = "../../MyData/dbo_pca_plain/"

rule_train_label_fixed_file = "./rule_train_label_fixed.txt"

train_data_num_upper_limit = 1000

alpha = 0.01

'''
Dict for the prefixes of uri
'''
uri2shorcut = {
    "http://dbpedia.org/ontology/": "dbo",
    "http://www.w3.org/1999/02/22-rdf-syntax-ns#": "rdf",
    "http://dbpedia.org/resource/": "dbr",
    "http://dbpedia.org/property/": "dbp",
    "http://www.w3.org/2000/01/rdf-schema#": "rdfs",
    "http://www.w3.org/2002/07/owl#": "owl",
    "http://xmlns.com/foaf/0.1/": "foaf",
    "http://purl.org/dc/elements/1.1/": "dc",
    "http://dbpedia.org/datatype/": "dbt",
    "http://www.w3.org/2001/XMLSchema#": "xsd",
    "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#": "dulo"
}
shortcut2uri = {
    "dbo": "http://dbpedia.org/ontology/",
    "rdf": "http://www.w3.org/1999/02/22-rdf-syntax-ns#",
    "dbr": "http://dbpedia.org/resource/",
    "dbp": "http://dbpedia.org/property/",
    "rdfs": "http://www.w3.org/2000/01/rdf-schema#",
    "owl": "http://www.w3.org/2002/07/owl#",
    "foaf": "http://xmlns.com/foaf/0.1/",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dbt": "http://dbpedia.org/datatype/",
    "xsd": "http://www.w3.org/2001/XMLSchema#",
    "dulo": "http://www.ontologydesignpatterns.org/ont/dul/DUL.owl#"
}

# transe_embed = False
transe_embed = True
# '''
# For a relation, we sample its ht to search the path which can conclude to it,
# this parameter is the number of ht sampled.
# '''
# sampled_num_to_search_rule = 500
# '''
# The first filter of rules searched, saving the rules tha occur frequently.
# '''
# top_frequency_rule_num = 600
# '''
# To train classifier, we sample positive and negetive instance for every rule saved,
# this is the number of instances sampled for every rule.
# '''
# posi_num_per_rule = 100
# nege_num_per_rule = 100

# test_model = False


# if sys.platform.startswith("linux"):
#     file_path_seg = "/"
#     mydb = mysql.connector.connect(
#         host='114.212.86.67',
#         user="xzlyu",
#         password="123456",
#         auth_plugin="mysql_native_password",
#         database="kg")
# else:
#     file_path_seg = "/"
#     # file_path_seg = "\\"
#     mydb = mysql.connector.connect(
#         host='localhost',
#         user="xzlyu",
#         password="123456",
#         auth_plugin="mysql_native_password",
#         database="kg")
