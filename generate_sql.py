import os
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
def prompt_to_sql(prompt):
    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", "content": (
                "You are an expert SQL developer. "
                "Only respond with clean SQL queries using the Chinook database schema, "
                "especially the correct singular table names like 'customer', 'employee', 'track', etc. "
                "Do not include code block formatting (no ```sql)."
            )},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message.content.strip()
