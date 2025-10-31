import psycopg2

try:
    conn = psycopg2.connect("dbname='energydb' user='energy' host='localhost' password='energy' port='5432'")
    print("Successfully connected to the database!")
    conn.close()
except Exception as e:
    print(f"Failed to connect to the database: {e}")
