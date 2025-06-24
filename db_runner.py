import psycopg2
import os

def run_query(query):
    conn = psycopg2.connect(
        host="localhost",
        database="chinook",   # Your database name
        user="postgres",      # Your DB username
        password="postgres",  # Your DB password
        port="5432"
    )
    cur = conn.cursor()
    cur.execute(query)

    # Get column headers
    colnames = [desc[0] for desc in cur.description]

    # Get rows as list of tuples
    rows = cur.fetchall()

    cur.close()
    conn.close()

    return colnames, rows  # ✅ Return 2-tuple
