import streamlit as st
import pandas as pd
from generate_sql import prompt_to_sql
from db_runner import run_query

st.set_page_config(page_title="AI SQL Assistant", page_icon="🧠", layout="centered")

st.title("🧠 AI-Powered SQL Assistant")
st.markdown("Ask a question in natural language and get instant SQL with live results!")

user_question = st.text_input(
    "Enter your question (e.g., 'List all customers from Germany'):", ""
)

sql_query = ""
if user_question:
    sql_query = prompt_to_sql(user_question)
    st.subheader("Generated SQL")
    st.code(sql_query, language="sql")

if sql_query and st.button("Run SQL Query"):
    try:
        rows = run_query(sql_query)
        if rows:
            df = pd.DataFrame(rows)
            st.success("✅ Query executed successfully!")
            st.dataframe(df)
        else:
            st.info("Query executed but returned no results.")
    except Exception as e:
        st.error(f"❌ Error: {e}")
