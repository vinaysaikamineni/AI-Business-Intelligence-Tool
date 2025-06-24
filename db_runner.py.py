import psycopg2

# Connection settings for your local DB
DB_CONFIG = {
    "host": "localhost",
    "port": 5432,
    "database": "chinook",
    "user": "postgres",
    "password": "postgres"
}

def run_query(sql):
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        cursor = conn.cursor()
        cursor.execute(sql)
        rows = cursor.fetchall()
        colnames = [desc[0] for desc in cursor.description]
        cursor.close()
        conn.close()
        return colnames, rows
    except Exception as e:
        return [], [f"Error: {e}"]

# Test with static SQL
if __name__ == "__main__":
    test_sql = "SELECT FirstName, LastName FROM customers WHERE Country = 'Germany';"
    headers, data = run_query(test_sql)
    print("Results:")
    print(headers)
    for row in data:
        print(row)
