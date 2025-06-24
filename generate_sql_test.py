import os
from openai import OpenAI
from dotenv import load_dotenv
from db_runner import run_query

load_dotenv()

client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def prompt_to_sql(nl_query):
    prompt = f"""
You are a helpful assistant that converts natural language into SQL queries.

Use the following PostgreSQL tables:
customer, album, artist, employee, genre, invoice, invoice_line, media_type, playlist, playlist_track, track.

Translate this question into an accurate SQL query:
{nl_query}

SQL:
"""

    try:
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are a PostgreSQL SQL assistant."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )

        sql = response.choices[0].message.content.strip()
        return sql

    except Exception as e:
        return f"Error: {str(e)}"


if __name__ == "__main__":
    query = "List all customers from Germany."
    sql = prompt_to_sql(query)
    print("\nGenerated SQL:\n", sql)

    if sql.lower().startswith("select"):
        print("\nQuery Results:")
        try:
            results = run_query(sql)
            print(results)
        except Exception as e:
            print("Error:", str(e))
