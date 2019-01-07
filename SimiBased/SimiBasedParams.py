import mysql.connector

mydb = mysql.connector.connect(
    host='localhost',
    user="root",
    password="3721",
    auth_plugin="mysql_native_password",
    database="kg")

"""
Before computing kl-divergence, compare the num of keys of two distribution,
cancel this calculation if they differ much.
"""
kl_key_differ_num_threshold = 10

"""
Unigram ration
"""
alpha = 0.8

"""
number of top kl value entities to reserve
"""
kl_top_k = 5
