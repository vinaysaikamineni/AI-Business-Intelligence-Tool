import psycopg2
import os

def run_query(query):
    conn = psycopg2.connect(
        host="localhost",
        database="chinook",  # Replace with your DB
        user="postgres",           # Replace with your DB user
        password="postgres",       # Replace with your DB password
        port="5432"
    )
    cur = conn.cursor()
    cur.execute(query)
    
    # Fetch column names from cursor description
    colnames = [desc[0] for desc in cur.description]
    
    # Fetch all rows
    rows = cur.fetchall()

    # Close connection
    cur.close()
    conn.close()

    # Convert to list of dicts
    result = [dict(zip(colnames, row)) for row in rows]
    return result


# Test with static SQL
if __name__ == "__main__":
    test_sql = "SELECT FirstName, LastName FROM customer WHERE Country = 'Germany';"
    headers, data = run_query(test_sql)
    print("Results:")
    print(headers)
    for row in data:
        print(row)
