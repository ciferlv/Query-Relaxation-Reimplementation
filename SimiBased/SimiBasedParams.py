import mysql.connector

mydb = mysql.connector.connect(
    host='localhost',
    user="root",
    password="3721",
    auth_plugin="mysql_native_password",
    database="kg")

"""
Unigram ration
"""
alpha = 0.8

"""
Subject ration for relation
"""
miu_s = 0.4

"""
Object ration for relation
"""
miu_o = 0.4

"""
Before computing js-divergence, compare the num of keys of two distribution,
cancel this calculation if they differ much.
"""
js_key_differ_num_threshold = 2000

"""
After merging the keys of two distribution,
threshold for the difference between the size of merged key set and original key set
"""
num_threshold_after_merging = 1000

"""
The table is used to store the context of entities.
"""
e_context_table = "e_context"

"""
The table is used to store the top K similar entities of entity e_idx.
"""
e_top_simi = "dbo_e_similarity"

"""
The table is used to store the context of relations.
"""
r_context_table = "r_context"

"""
The table is used to store the top K similar entites of entity r_idx.
"""
r_top_simi = "dbo_r_similarity"

"""
Top K to store
"""
top_k_to_store = 10

"""
Top K to use
"""
top_k_to_use = 2

"""
number of top kl value entities to reserve
"""
js_top_k = 5
