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
    initial_sidebar_state="collapsed"  # Start collapsed to prevent initial shift
)

# Custom CSS for stable layout and better styling
st.markdown("""
<style>
/* Prevent content shifting when sidebar opens/closes */
.main .block-container {
    padding-left: 1rem;
    padding-right: 1rem;
    max-width: none;
    transition: none !important;
}

/* Ensure main content area has consistent width */
.stApp > div:first-child {
    transition: none !important;
}

/* Sidebar styling */
.css-1d391kg {
    transition: margin-left 0.2s ease;
}

/* Main header styling */
.main-header {
    font-size: 2.5rem;
    background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-align: center;
    margin-bottom: 0.5rem;
    padding: 0.5rem 0;
}

/* Stats box styling */
.stats-box {
    background: linear-gradient(135deg, #f0f2f6 0%, #e8ecf0 100%);
    padding: 1.5rem;
    border-radius: 12px;
    margin: 0.5rem 0;
    border: 1px solid #e1e5e9;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

/* Query input area */
.query-input-container {
    background: #ffffff;
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #e1e5e9;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* Results section */
.results-section {
    background: #ffffff;
    padding: 1.5rem;
    border-radius: 12px;
    border: 1px solid #e1e5e9;
    margin: 1rem 0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.05);
}

/* Sidebar content */
.sidebar .sidebar-content {
    padding: 1rem 0.5rem;
}

/* Fixed main content width to prevent shifting */
.main-content-wrapper {
    width: 100%;
    margin: 0;
    padding: 0;
}

/* Compact header section */
.header-section {
    margin-bottom: 1rem;
    padding-bottom: 0;
}

/* Subtitle spacing */
.subtitle {
    margin-top: 0;
    margin-bottom: 1.5rem;
}

/* Responsive design improvements */
@media (max-width: 768px) {
    .main-header {
        font-size: 2rem;
    }
    
    .stats-box, .query-input-container, .results-section {
        padding: 1rem;
        margin: 0.5rem 0;
    }
}

/* Smooth transitions only for specific elements */
.metric-container {
    transition: all 0.3s ease;
}

/* Override Streamlit's default responsive behavior */
.element-container {
    width: 100% !important;
}

/* Ensure consistent spacing */
.stColumns {
    gap: 1rem;
}
</style>
""", unsafe_allow_html=True)

# Initialize session state
if "query_history" not in st.session_state:
    st.session_state.query_history = []
if "db_connected" not in st.session_state:
    st.session_state.db_connected = None

# Header
st.markdown('<div class="header-section">', unsafe_allow_html=True)
st.markdown('<h1 class="main-header">🧠 AI-Powered SQL Assistant</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="subtitle">Transform natural language into SQL queries and get instant insights from the Chinook database</h3>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

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

# Main content area with stable container
with st.container():
    st.markdown('<div class="main-content-wrapper">', unsafe_allow_html=True)
    
    # Fixed-width columns to prevent shifting
    col1, col2 = st.columns([3, 1], gap="large")
    
    with col1:
        st.markdown('<div class="query-input-container">', unsafe_allow_html=True)
        
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
            height=120,
            key="main_query_input"
        )
        
        # Execute button
        execute_query = st.button(
            "🚀 Execute Query", 
            type="primary", 
            use_container_width=True,
            disabled=not nl_query.strip()
        )
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="stats-box">', unsafe_allow_html=True)
        
        # Statistics section
        st.markdown("**📊 Statistics**")
        st.metric("Total Queries", len(st.session_state.query_history))
        
        if st.session_state.query_history:
            avg_score = sum(entry.get('score', 0) for entry in st.session_state.query_history) / len(st.session_state.query_history)
            st.metric("Success Rate", f"{avg_score:.1%}")
            
            # Recent activity
            if len(st.session_state.query_history) > 0:
                last_query = st.session_state.query_history[-1]
                st.metric("Last Query", f"{last_query.get('execution_time', 0):.2f}s")
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('</div>', unsafe_allow_html=True)

# Query execution
if execute_query and nl_query:
    try:
        start_time = datetime.now()
        
        with st.spinner("🤖 Generating SQL query..."):
            # Generate SQL using enhanced module
            sql = prompt_to_sql(nl_query)
            
        # Results container
        with st.container():
            st.markdown('<div class="results-section">', unsafe_allow_html=True)
            
            # Display generated SQL
            st.subheader("📝 Generated SQL Query")
            st.code(sql, language="sql")
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Execute query
        with st.spinner("⚡ Executing query..."):
            try:
                headers, rows = run_query(sql, timeout=Config.QUERY_TIMEOUT)
                execution_time = (datetime.now() - start_time).total_seconds()
                
                # Create DataFrame
                if headers and rows:
                    df = pd.DataFrame(rows, columns=headers)
                    
                    # Results container
                    with st.container():
                        st.markdown('<div class="results-section">', unsafe_allow_html=True)
                        
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
                        
                        st.markdown('</div>', unsafe_allow_html=True)
                    
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
