
import psycopg2
import pandas as pd
import matplotlib.pyplot as plt
import os

db_params = {
    'host': 'localhost',
    'user': 'postgres',
    'password': 'Admin@'
}
revenue_streams = ['group']

connection = psycopg2.connect(**db_params)
output_directory = 'revenue_graphs/'
os.makedirs(output_directory,exist_ok=True)

for group in revenue_streams:
    query =f"SELECT posted_date,title FROM joblisting WHERE group='{group}'"
    df = pd.read_sql_query(query,connection)
