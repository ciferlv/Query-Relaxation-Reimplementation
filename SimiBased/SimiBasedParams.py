import mysql.connector

mydb = mysql.connector.connect(
    host='localhost',
    user="root",
    password="3721",
    auth_plugin="mysql_native_password",
    database="kg")
