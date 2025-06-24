import streamlit as st
from dotenv import load_dotenv
import os
from openai import OpenAI
from db_runner import run_query

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

st.set_page_config(page_title="AI SQL Assistant", layout="centered")
st.title("🧠 AI-Powered SQL Assistant")
st.write("Ask a question and get real-time insights from your Chinook database.")

nl_query = st.text_input("Enter your data question:", placeholder="e.g. List all customers from Germany")

if nl_query:
    with st.spinner("Generating SQL..."):
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

            sql = response.choices[0].message.content.strip().replace("```sql", "").replace("```", "")
            st.code(sql, language="sql")

            if sql.lower().startswith("select"):
                st.subheader("Query Results")
                headers, rows = run_query(sql)
                if headers and isinstance(rows[0], (tuple, list)):
                    import pandas as pd
                    df = pd.DataFrame(rows, columns=headers)
                    st.dataframe(df)
                else:
                    st.error(rows[0])  # display error
        except Exception as e:
            st.error(f"Error: {str(e)}")
