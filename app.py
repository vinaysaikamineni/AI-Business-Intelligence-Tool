
import streamlit as st
from dotenv import load_dotenv
import os
import pandas as pd
from openai import OpenAI
from db_runner import run_query
from datetime import datetime

load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# Streamlit UI setup
st.set_page_config(page_title="AI SQL Assistant", layout="wide")
st.title("🧠 AI-Powered SQL Assistant")
st.write("Ask a question and get real-time insights from your Chinook database.")

# Initialize session history
if "query_history" not in st.session_state:
    st.session_state.query_history = []

# Sidebar: Query History
st.sidebar.header("📜 Query History")
for entry in reversed(st.session_state.query_history):
    with st.sidebar.expander(entry["timestamp"]):
        st.code(entry["query"], language="sql")

# Main Input
nl_query = st.text_input("Enter your data question:", placeholder="e.g., List all customers from Germany")

if nl_query:
    with st.spinner("Generating SQL..."):
        # Prompt with table schema
        prompt = f"""
You are a helpful assistant that converts natural language into SQL queries.

Use the following PostgreSQL tables and columns:
- customer (customer_id, first_name, last_name, address, city, country, postal_code, phone, email)
- invoice (invoice_id, customer_id, invoice_date, billing_address, total)
- employee (employee_id, first_name, last_name, title, hire_date, birth_date, address, city, state)
- playlist (playlist_id, name)
- playlist_track (playlist_id, track_id)
- track (track_id, name, album_id, media_type_id, genre_id, composer, milliseconds, bytes, unit_price)
- genre (genre_id, name)

Translate this user question to a valid PostgreSQL SQL query:
{nl_query}

Only return the SQL statement.
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

            # Store query in session history
            st.session_state.query_history.append({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "query": sql
            })

            try:
                headers, rows = run_query(sql)
                df = pd.DataFrame(rows, columns=headers)

                st.subheader("Query Results")
                st.dataframe(df, use_container_width=True)

                with st.expander("📄 Download Results"):
                    csv = df.to_csv(index=False)
                    st.download_button("Download as CSV", data=csv, file_name="query_results.csv", mime="text/csv")

            except Exception as exec_err:
                st.error(f"❌ Query execution error: {exec_err}")

        except Exception as e:
            st.error(f"❌ OpenAI error: {e}")
