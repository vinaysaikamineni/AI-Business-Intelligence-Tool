import streamlit as st
import pandas as pd
import logging
from datetime import datetime
from db_runner import run_query, test_connection, DatabaseError
from generate_sql import prompt_to_sql, SQLGenerationError
from config import Config, CHINOOK_SCHEMA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit configuration
st.set_page_config(
    page_title="AI SQL Assistant",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
.main-header {
    font-size: 3rem;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 2rem;
}
.stats-box {
    background-color: #f0f2f6;
    padding: 1rem;
    border-radius: 10px;
    margin: 0.5rem 0;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "db_connected" not in st.session_state:
    st.session_state.db_connected = None

# Header
st.markdown('<h1 class="main-header">🧠 AI-Powered SQL Assistant</h1>', unsafe_allow_html=True)
st.markdown("### Transform natural language into SQL queries and get instant insights from the Chinook database")

# Sidebar setup
with st.sidebar:
    st.header("🔧 Configuration")
    
    # Database connection status
    if st.button("Test Database Connection"):
        with st.spinner("Testing connection..."):
            st.session_state.db_connected = test_connection()
    
    if st.session_state.db_connected is True:
        st.success("✅ Database Connected")
    elif st.session_state.db_connected is False:
        st.error("❌ Database Connection Failed")
    else:
        st.info("ℹ️ Click to test database connection")
    
    st.divider()
    
    # Database schema
    with st.expander("📊 Database Schema", expanded=False):
        for table, columns in CHINOOK_SCHEMA.items():
            st.write(f"**{table}:**")
            st.write(", ".join(columns[:5]) + ("..." if len(columns) > 5 else ""))
    
    st.divider()
    
    # Query history
    st.header("📜 Query History")
    if st.button("Clear History"):
        st.session_state.query_history = []
        st.rerun()
    
    # Limit history display
    history_limit = min(10, len(st.session_state.query_history))
    for entry in reversed(st.session_state.query_history[-history_limit:]):
        with st.expander(f"{entry['timestamp']} (Score: {entry.get('score', 'N/A')})"):
            st.code(entry["query"], language="sql")
            if "execution_time" in entry:
                st.caption(f"Executed in {entry['execution_time']:.2f}s")

# Main content area
col1, col2 = st.columns([2, 1])

with col2:
    st.markdown('<div class="stats-box">', unsafe_allow_html=True)
    st.metric("Total Queries", len(st.session_state.query_history))
    if st.session_state.query_history:
        avg_score = sum(entry.get('score', 0) for entry in st.session_state.query_history) / len(st.session_state.query_history)
        st.metric("Avg Success Rate", f"{avg_score:.1%}")
    st.markdown('</div>', unsafe_allow_html=True)

with col1:
    # Example queries
    st.markdown("**💡 Example Questions:**")
    examples = [
        "Show me the top 5 customers by total spending",
        "What are the most popular music genres?",
        "List all tracks by AC/DC",
        "Which employee has the most customers?"
    ]
    
    selected_example = st.selectbox("Choose an example or type your own:", [""] + examples)
    
    # Main input
    nl_query = st.text_area(
        "Enter your data question:", 
        value=selected_example if selected_example else "",
        placeholder="e.g., Show me the top 10 best-selling tracks",
        height=100
    )

# Query execution
if nl_query:
    try:
        start_time = datetime.now()
        
        with st.spinner("🤖 Generating SQL query..."):
            # Generate SQL using enhanced module
            sql = prompt_to_sql(nl_query)
            
        # Display generated SQL
        st.subheader("📝 Generated SQL Query")
        st.code(sql, language="sql")
        
        # Execute query
        with st.spinner("⚡ Executing query..."):
            try:
                headers, rows = run_query(sql, timeout=Config.QUERY_TIMEOUT)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Create DataFrame
                if headers and rows:
                    df = pd.DataFrame(rows, columns=headers)
                    
                    # Display results
                    st.subheader("📊 Query Results")
                    
                    # Results summary
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Rows Returned", len(df))
                    with col2:
                        st.metric("Columns", len(df.columns))
                    with col3:
                        st.metric("Execution Time", f"{execution_time:.2f}s")
                    
                    # Data display with pagination
                    if len(df) > 100:
                        st.warning(f"Large result set ({len(df)} rows). Showing first 100 rows.")
                        st.dataframe(df.head(100), use_container_width=True)
                    else:
                        st.dataframe(df, use_container_width=True)
                    
                    # Download options
                    with st.expander("📥 Download Options"):
                        col1, col2 = st.columns(2)
                        with col1:
                            csv = df.to_csv(index=False)
                            st.download_button(
                                "📄 Download as CSV", 
                                data=csv, 
                                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv", 
                                mime="text/csv"
                            )
                        with col2:
                            json_data = df.to_json(orient='records', indent=2)
                            st.download_button(
                                "📋 Download as JSON", 
                                data=json_data, 
                                file_name=f"query_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json", 
                                mime="application/json"
                            )
                    
                    # Store successful query in history
                    st.session_state.query_history.append({
                        "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                        "query": sql,
                        "nl_query": nl_query,
                        "rows_returned": len(df),
                        "execution_time": execution_time,
                        "score": 1.0  # Successful execution
                    })
                    
                    st.success(f"✅ Query executed successfully! {len(df)} rows returned in {execution_time:.2f}s")
                    
                else:
                    st.info("Query executed successfully but returned no results.")
                    
            except DatabaseError as db_err:
                st.error(f"🚫 Database Error: {db_err}")
                # Store failed query
                st.session_state.query_history.append({
                    "timestamp": start_time.strftime("%Y-%m-%d %H:%M:%S"),
                    "query": sql,
                    "nl_query": nl_query,
                    "error": str(db_err),
                    "score": 0.0
                })
                
            except Exception as exec_err:
                st.error(f"❌ Query Execution Error: {exec_err}")
                logger.error(f"Query execution failed: {exec_err}")
                
    except SQLGenerationError as sql_err:
        st.error(f"🤖 SQL Generation Error: {sql_err}")
        logger.error(f"SQL generation failed: {sql_err}")
        
    except Exception as e:
        st.error(f"❌ Unexpected Error: {e}")
        logger.error(f"Unexpected error: {e}")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🎵 Powered by OpenAI GPT • Built with Streamlit • Data from Chinook Database</p>
    <p>💡 Tip: Try asking about customers, invoices, tracks, artists, or employees!</p>
</div>
""", unsafe_allow_html=True)
