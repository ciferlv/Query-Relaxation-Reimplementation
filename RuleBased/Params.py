import mysql.connector

rule_seg = ":"  # break up the rule, birthPlace <= birthPlace:location
ht_conn = ","  # connect h and t, [Microsoft,Seattle] ==> "Microsoft,Seattle"
ht_seg = ";"  # break up between different hts, [[Micorsoft,Seattle],[Apple,LA]] ==> "Microsoft,Seattel;Apple,LA"

mydb = mysql.connector.connect(
    host='localhost',
    user="root",
    password="3721",
    auth_plugin="mysql_native_password",
    database="kg"
)

file_path_seg = "\\"

database = ' fb15k '

'''
For a relation, we sample its ht to search the path which can conclude to it,
this parameter is the number of ht sampled.
'''
sampled_num_to_search_rule = 10

'''
The first filter of rules searched, saving the rules tha occur frequently.
'''
top_frequency_rule_num = 10

'''
To train classifier, we sample positive and negetive instance for every rule saved,
this is the number of instances sampled for every rule.
'''
posi_num_per_rule = 100
nege_num_per_rule = 100

test_model = False
