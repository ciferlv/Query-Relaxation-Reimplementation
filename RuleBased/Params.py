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
