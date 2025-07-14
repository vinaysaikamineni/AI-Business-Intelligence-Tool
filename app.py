import streamlit as st
import pandas as pd
import logging
import json
from datetime import datetime
from db_runner import run_query, test_connection, DatabaseError
from generate_sql import prompt_to_sql, SQLGenerationError
from statistical_analysis import StatisticalAnalyzer
from config import Config, CHINOOK_SCHEMA

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Streamlit configuration
st.set_page_config(
    page_title="AI-Powered Business Intelligence Tool",
    page_icon="📊",
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
st.markdown('<h1 class="main-header">📊 AI-Powered Business Intelligence Tool</h1>', unsafe_allow_html=True)
st.markdown('<h3 class="subtitle">Transform natural language into SQL queries and get instant business insights</h3>', unsafe_allow_html=True)
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

# Add Statistical Analysis Tab
st.markdown("---")
st.markdown("## 📈 Statistical Analysis")

# Create tabs for different analysis types
tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Descriptive Stats", "🔗 Correlations", "📈 Regression", "👥 Segmentation", "🚨 Anomalies"])

# Initialize analyzer
analyzer = StatisticalAnalyzer()

# Sample query to get data for analysis
if 'analysis_data' not in st.session_state:
    st.session_state.analysis_data = None

# Data selection for analysis
with st.expander("🔍 Select Data for Analysis", expanded=True):
    analysis_options = [
        "Customer Purchase Analysis",
        "Track Sales Analysis", 
        "Invoice Data Analysis",
        "Custom Query Results"
    ]
    
    selected_analysis = st.selectbox("Choose analysis dataset:", analysis_options)
    
    if st.button("📥 Load Data for Analysis"):
        with st.spinner("Loading data for statistical analysis..."):
            try:
                if selected_analysis == "Customer Purchase Analysis":
                    query = """
                    SELECT 
                        c.customer_id,
                        c.country,
                        COUNT(i.invoice_id) as total_orders,
                        SUM(i.total) as total_spent,
                        AVG(i.total) as avg_order_value,
                        MAX(i.invoice_date) as last_purchase
                    FROM customer c
                    LEFT JOIN invoice i ON c.customer_id = i.customer_id
                    GROUP BY c.customer_id, c.country
                    HAVING COUNT(i.invoice_id) > 0
                    """
                elif selected_analysis == "Track Sales Analysis":
                    query = """
                    SELECT 
                        t.track_id,
                        t.name as track_name,
                        g.name as genre,
                        t.unit_price,
                        COUNT(il.track_id) as times_purchased,
                        SUM(il.quantity) as total_quantity,
                        SUM(il.unit_price * il.quantity) as total_revenue
                    FROM track t
                    JOIN genre g ON t.genre_id = g.genre_id
                    LEFT JOIN invoice_line il ON t.track_id = il.track_id
                    GROUP BY t.track_id, t.name, g.name, t.unit_price
                    HAVING COUNT(il.track_id) > 0
                    LIMIT 500
                    """
                elif selected_analysis == "Invoice Data Analysis":
                    query = """
                    SELECT 
                        i.invoice_id,
                        i.customer_id,
                        i.total,
                        COUNT(il.invoice_line_id) as line_items,
                        AVG(il.unit_price) as avg_item_price,
                        SUM(il.quantity) as total_quantity
                    FROM invoice i
                    JOIN invoice_line il ON i.invoice_id = il.invoice_id
                    GROUP BY i.invoice_id, i.customer_id, i.total
                    """
                else:  # Custom Query Results
                    if st.session_state.query_history and len(st.session_state.query_history) > 0:
                        query = st.session_state.query_history[-1]['query']
                    else:
                        st.warning("No previous query results available. Please run a query first.")
                        query = None
                
                if query:
                    headers, rows = run_query(query)
                    if headers and rows:
                        st.session_state.analysis_data = pd.DataFrame(rows, columns=headers)
                        st.success(f"✅ Data loaded successfully! {len(st.session_state.analysis_data)} rows, {len(st.session_state.analysis_data.columns)} columns")
                        
                        # Show data preview
                        st.subheader("📋 Data Preview")
                        st.dataframe(st.session_state.analysis_data.head(), use_container_width=True)
                    else:
                        st.error("No data returned from query")
                        
            except Exception as e:
                st.error(f"Error loading data: {e}")

# Statistical Analysis Tabs
if st.session_state.analysis_data is not None:
    df = st.session_state.analysis_data
    
    with tab1:  # Descriptive Statistics
        st.subheader("📊 Descriptive Statistics")
        
        if st.button("🔍 Generate Descriptive Statistics"):
            with st.spinner("Calculating descriptive statistics..."):
                results = analyzer.descriptive_statistics(df)
                
                if 'error' not in results:
                    # Summary statistics
                    st.markdown("### 📈 Data Summary")
                    summary = results['summary']
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("Total Rows", summary['total_rows'])
                    with col2:
                        st.metric("Total Columns", summary['total_columns'])
                    with col3:
                        st.metric("Numeric Columns", summary['numeric_columns'])
                    with col4:
                        st.metric("Missing Values", summary['missing_values'])
                    
                    # Numeric statistics
                    if results['numeric_statistics']:
                        st.markdown("### 🔢 Numeric Statistics")
                        numeric_df = pd.DataFrame(results['numeric_statistics']).round(3)
                        st.dataframe(numeric_df, use_container_width=True)
                    
                    # Categorical statistics
                    if results['categorical_statistics']:
                        st.markdown("### 📝 Categorical Statistics")
                        for col, stats in results['categorical_statistics'].items():
                            with st.expander(f"📊 {col} Analysis"):
                                col1, col2 = st.columns(2)
                                with col1:
                                    st.metric("Unique Values", stats['unique_values'])
                                    st.metric("Most Frequent", stats['most_frequent'])
                                with col2:
                                    st.metric("Mode Frequency", stats['mode_frequency'])
                                
                                # Value distribution
                                if stats['value_distribution']:
                                    st.bar_chart(pd.Series(stats['value_distribution']))
                    
                    # Distribution analysis
                    if results['distribution_analysis']:
                        st.markdown("### 📊 Distribution Analysis")
                        for col, analysis in results['distribution_analysis'].items():
                            normality = analysis['normality_test']
                            status = "✅ Normal" if normality['is_normal'] else "❌ Not Normal"
                            st.write(f"**{col}**: {status} (p-value: {normality['p_value']:.4f})")
                else:
                    st.error(f"Error in analysis: {results['error']}")
    
    with tab2:  # Correlation Analysis
        st.subheader("🔗 Correlation Analysis")
        
        if st.button("🔍 Generate Correlation Analysis"):
            with st.spinner("Calculating correlations..."):
                results = analyzer.correlation_analysis(df)
                
                if 'error' not in results:
                    # Strong correlations
                    if results['strong_correlations']:
                        st.markdown("### 🎯 Strong Correlations")
                        for corr in results['strong_correlations']:
                            strength_emoji = "🔴" if corr['strength'] == 'Strong' else "🟡"
                            direction = "↗️" if corr['direction'] == 'Positive' else "↘️"
                            st.write(f"{strength_emoji} {direction} **{corr['variable1']}** ↔ **{corr['variable2']}**: {corr['pearson_correlation']:.3f}")
                    
                    # Correlation matrix
                    st.markdown("### 📊 Correlation Matrix")
                    corr_matrix = pd.DataFrame(results['pearson_correlation_matrix'])
                    st.dataframe(corr_matrix.round(3), use_container_width=True)
                    
                    # Summary
                    summary = results['correlation_summary']
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Total Correlations", summary['total_pairs'])
                    with col2:
                        st.metric("Strong Positive", summary['strong_positive'])
                    with col3:
                        st.metric("Strong Negative", summary['strong_negative'])
                else:
                    st.error(f"Error in analysis: {results['error']}")
    
    with tab3:  # Regression Analysis
        st.subheader("📈 Regression Analysis")
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            target_col = st.selectbox("Select target variable:", numeric_cols)
            feature_cols = st.multiselect(
                "Select feature variables:", 
                [col for col in numeric_cols if col != target_col],
                default=[col for col in numeric_cols if col != target_col][:3]
            )
            
            if feature_cols and st.button("🔍 Run Regression Analysis"):
                with st.spinner("Running regression analysis..."):
                    results = analyzer.regression_analysis(df, target_col, feature_cols)
                    
                    if 'error' not in results:
                        # Model performance
                        st.markdown("### 📊 Model Performance")
                        perf = results['model_performance']
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("R²", f"{perf['r_squared']:.3f}")
                        with col2:
                            st.metric("Adj. R²", f"{perf['adjusted_r_squared']:.3f}")
                        with col3:
                            st.metric("RMSE", f"{perf['rmse']:.3f}")
                        with col4:
                            st.metric("MAE", f"{perf['mean_absolute_error']:.3f}")
                        
                        # Coefficients
                        st.markdown("### 🎯 Coefficients")
                        coef_df = pd.DataFrame({
                            'Feature': list(results['coefficients'].keys()),
                            'Coefficient': list(results['coefficients'].values()),
                            'Importance': list(results['feature_importance'].values())
                        })
                        st.dataframe(coef_df.round(4), use_container_width=True)
                        
                        # Statistical significance
                        sig = results['statistical_significance']
                        st.markdown(f"**F-statistic:** {sig['f_statistic']:.3f} (p-value: {sig['f_pvalue']:.4f})")
                        significance = "✅ Significant" if sig['f_pvalue'] < 0.05 else "❌ Not Significant"
                        st.write(f"**Model Significance:** {significance}")
                    else:
                        st.error(f"Error in analysis: {results['error']}")
        else:
            st.warning("Need at least 2 numeric columns for regression analysis.")
    
    with tab4:  # Customer Segmentation
        st.subheader("👥 Customer Segmentation")
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) >= 2:
            cluster_cols = st.multiselect(
                "Select features for clustering:", 
                numeric_cols,
                default=numeric_cols[:min(3, len(numeric_cols))]
            )
            n_clusters = st.slider("Number of clusters:", 2, 8, 4)
            
            if cluster_cols and st.button("🔍 Perform Segmentation"):
                with st.spinner("Performing customer segmentation..."):
                    results = analyzer.customer_segmentation(df, cluster_cols, n_clusters)
                    
                    if 'error' not in results:
                        # Summary
                        st.markdown("### 📊 Segmentation Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Customers", results['total_customers'])
                        with col2:
                            st.metric("Number of Clusters", results['number_of_clusters'])
                        with col3:
                            if results['silhouette_score']:
                                st.metric("Silhouette Score", f"{results['silhouette_score']:.3f}")
                        
                        # Cluster details
                        st.markdown("### 🎯 Cluster Analysis")
                        cluster_stats = results['cluster_statistics']
                        
                        for cluster_name, stats in cluster_stats.items():
                            with st.expander(f"📊 {cluster_name.replace('_', ' ').title()} ({stats['size']} customers, {stats['percentage']:.1f}%)"):
                                st.write(f"**Characteristics:** {stats['characteristics']}")
                                
                                # Means comparison
                                means_df = pd.DataFrame({
                                    'Feature': list(stats['means'].keys()),
                                    'Average Value': list(stats['means'].values())
                                })
                                st.dataframe(means_df.round(3), use_container_width=True)
                    else:
                        st.error(f"Error in analysis: {results['error']}")
        else:
            st.warning("Need at least 2 numeric columns for segmentation analysis.")
    
    with tab5:  # Anomaly Detection
        st.subheader("🚨 Anomaly Detection")
        
        numeric_cols = df.select_dtypes(include=['number']).columns.tolist()
        
        if len(numeric_cols) >= 1:
            anomaly_cols = st.multiselect(
                "Select features for anomaly detection:", 
                numeric_cols,
                default=numeric_cols
            )
            
            if anomaly_cols and st.button("🔍 Detect Anomalies"):
                with st.spinner("Detecting anomalies..."):
                    results = analyzer.anomaly_detection(df, anomaly_cols)
                    
                    if 'error' not in results:
                        # Summary
                        st.markdown("### 📊 Anomaly Summary")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("Total Data Points", results['total_data_points'])
                        with col2:
                            st.metric("Anomalies Detected", results['anomalies_detected'])
                        with col3:
                            st.metric("Anomaly Rate", f"{results['anomaly_percentage']:.1f}%")
                        
                        # Most anomalous point
                        if results['anomalies_detected'] > 0:
                            st.markdown("### 🎯 Most Anomalous Data Point")
                            most_anomalous = results['most_anomalous']
                            st.write(f"**Index:** {most_anomalous['index']}")
                            st.write(f"**Anomaly Score:** {most_anomalous['score']:.4f}")
                            
                            # Values
                            values_df = pd.DataFrame({
                                'Feature': list(most_anomalous['values'].keys()),
                                'Value': list(most_anomalous['values'].values())
                            })
                            st.dataframe(values_df.round(3), use_container_width=True)
                        
                        # Feature statistics comparison
                        if 'feature_statistics' in results:
                            st.markdown("### 📊 Normal vs Anomalous Data Comparison")
                            normal_stats = pd.DataFrame(results['feature_statistics']['normal_data'])
                            anomalous_stats = pd.DataFrame(results['feature_statistics']['anomalous_data'])
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**Normal Data Statistics**")
                                st.dataframe(normal_stats.round(3), use_container_width=True)
                            with col2:
                                st.markdown("**Anomalous Data Statistics**")
                                st.dataframe(anomalous_stats.round(3), use_container_width=True)
                    else:
                        st.error(f"Error in analysis: {results['error']}")
        else:
            st.warning("Need at least 1 numeric column for anomaly detection.")

    # Generate Insights
    st.markdown("---")
    st.markdown("## 💡 AI-Generated Insights")
    
    if st.button("🧠 Generate Statistical Insights"):
        with st.spinner("Generating insights from statistical analysis..."):
            # Run multiple analyses to get comprehensive insights
            desc_stats = analyzer.descriptive_statistics(df)
            corr_analysis = analyzer.correlation_analysis(df)
            
            # Combine results
            all_results = {**desc_stats, **corr_analysis}
            
            insights = analyzer.generate_insights(df, all_results)
            
            st.markdown("### 🎯 Key Insights")
            for i, insight in enumerate(insights, 1):
                st.write(f"**{i}.** {insight}")
else:
    st.info("📊 Load data using the section above to begin statistical analysis.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🤖 Powered by OpenAI GPT • Built with Streamlit • PostgreSQL Database • Statistical Analysis with Python</p>
    <p>💡 Tip: Ask about sales trends, customer analytics, revenue insights, or performance metrics!</p>
</div>
""", unsafe_allow_html=True)
