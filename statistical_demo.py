"""
🎯 Statistical Analysis Demo Script

This script demonstrates the comprehensive statistical analysis capabilities
of the AI Business Intelligence Tool, perfect for showcasing to potential
employers for data analyst positions.

Author: Vinay Sai Kamineni
"""

import pandas as pd
import numpy as np
from statistical_analysis import StatisticalAnalyzer
from datetime import datetime

def create_sample_customer_data():
    """Create realistic customer purchase data for analysis"""
    np.random.seed(42)
    
    # Customer segments with different characteristics
    n_customers = 1000
    customers = []
    
    for i in range(n_customers):
        # Customer segments: 
        # 1. High-value (20%) - spend more, buy frequently
        # 2. Regular (60%) - average spending and frequency
        # 3. Occasional (20%) - low spending, infrequent purchases
        
        if i < 200:  # High-value customers
            total_spent = np.random.normal(2500, 800, 1)[0]
            total_orders = np.random.normal(25, 8, 1)[0]
            last_purchase_days = np.random.randint(1, 30)
        elif i < 800:  # Regular customers
            total_spent = np.random.normal(800, 300, 1)[0]
            total_orders = np.random.normal(8, 3, 1)[0]
            last_purchase_days = np.random.randint(1, 90)
        else:  # Occasional customers
            total_spent = np.random.normal(200, 100, 1)[0]
            total_orders = np.random.normal(2, 1, 1)[0]
            last_purchase_days = np.random.randint(30, 365)
        
        # Ensure positive values
        total_spent = max(total_spent, 10)
        total_orders = max(int(total_orders), 1)
        avg_order_value = total_spent / total_orders
        
        # Country distribution
        countries = ['USA', 'Canada', 'Germany', 'UK', 'France', 'Brazil', 'Australia']
        country = np.random.choice(countries, p=[0.3, 0.15, 0.15, 0.1, 0.1, 0.1, 0.1])
        
        customers.append({
            'customer_id': i + 1,
            'country': country,
            'total_orders': int(total_orders),
            'total_spent': round(total_spent, 2),
            'avg_order_value': round(avg_order_value, 2),
            'days_since_last_purchase': last_purchase_days,
            'customer_lifetime_value': round(total_spent * 1.2, 2)
        })
    
    return pd.DataFrame(customers)

def create_sample_sales_data():
    """Create realistic sales/product data for analysis"""
    np.random.seed(123)
    
    # Product categories and their characteristics
    categories = {
        'Electronics': {'avg_price': 250, 'std_price': 100, 'volume_multiplier': 1.2},
        'Clothing': {'avg_price': 50, 'std_price': 20, 'volume_multiplier': 2.0},
        'Books': {'avg_price': 15, 'std_price': 8, 'volume_multiplier': 1.5},
        'Home & Garden': {'avg_price': 80, 'std_price': 40, 'volume_multiplier': 0.8},
        'Sports': {'avg_price': 120, 'std_price': 60, 'volume_multiplier': 1.0}
    }
    
    products = []
    product_id = 1
    
    for category, props in categories.items():
        # Create 200 products per category
        for i in range(200):
            price = max(np.random.normal(props['avg_price'], props['std_price']), 5)
            
            # Sales volume inversely correlated with price (somewhat)
            base_volume = np.random.normal(100, 30)
            price_effect = (props['avg_price'] - price) / props['avg_price'] * 50
            volume = max(int((base_volume + price_effect) * props['volume_multiplier']), 1)
            
            revenue = price * volume
            profit_margin = np.random.normal(0.25, 0.1)  # 25% average margin
            profit_margin = max(min(profit_margin, 0.6), 0.05)  # Between 5% and 60%
            
            products.append({
                'product_id': product_id,
                'category': category,
                'unit_price': round(price, 2),
                'units_sold': volume,
                'total_revenue': round(revenue, 2),
                'profit_margin': round(profit_margin, 3),
                'profit': round(revenue * profit_margin, 2),
                'customer_rating': round(np.random.normal(4.0, 0.8), 1),
                'return_rate': round(max(np.random.normal(0.05, 0.03), 0), 3)
            })
            product_id += 1
    
    return pd.DataFrame(products)

def run_comprehensive_analysis():
    """Run comprehensive statistical analysis and generate business insights"""
    
    print("🚀 AI Business Intelligence Tool - Statistical Analysis Demo")
    print("=" * 70)
    print(f"📅 Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"👤 Analyst: Vinay Sai Kamineni")
    print(f"🎯 Purpose: Data Analyst Portfolio Demonstration")
    print("=" * 70)
    
    # Initialize analyzer
    analyzer = StatisticalAnalyzer()
    
    # Create sample datasets
    print("\n📊 Generating Sample Datasets...")
    customer_data = create_sample_customer_data()
    sales_data = create_sample_sales_data()
    
    print(f"✅ Customer Dataset: {len(customer_data)} customers across {customer_data['country'].nunique()} countries")
    print(f"✅ Sales Dataset: {len(sales_data)} products across {sales_data['category'].nunique()} categories")
    
    # === CUSTOMER ANALYSIS ===
    print("\n" + "="*50)
    print("👥 CUSTOMER BEHAVIOR ANALYSIS")
    print("="*50)
    
    print("\n📈 1. DESCRIPTIVE STATISTICS")
    print("-" * 30)
    
    customer_stats = analyzer.descriptive_statistics(customer_data)
    if 'error' not in customer_stats:
        summary = customer_stats['summary']
        print(f"📊 Dataset Overview:")
        print(f"   • Total Customers: {summary['total_rows']:,}")
        print(f"   • Data Columns: {summary['total_columns']}")
        print(f"   • Missing Values: {summary['missing_values']}")
        
        # Key business metrics
        if customer_stats['numeric_statistics']:
            stats = customer_stats['numeric_statistics']
            print(f"\n💰 Revenue Insights:")
            print(f"   • Average Customer Value: ${stats['mean']['total_spent']:.2f}")
            print(f"   • Median Customer Value: ${stats['median']['total_spent']:.2f}")
            print(f"   • Customer Value Range: ${stats['min']['total_spent']:.2f} - ${stats['max']['total_spent']:.2f}")
            print(f"   • Average Orders per Customer: {stats['mean']['total_orders']:.1f}")
    
    print("\n🔗 2. CORRELATION ANALYSIS")
    print("-" * 30)
    
    customer_corr = analyzer.correlation_analysis(customer_data)
    if 'error' not in customer_corr and customer_corr['strong_correlations']:
        print("🎯 Strong Relationships Discovered:")
        for i, corr in enumerate(customer_corr['strong_correlations'][:3], 1):
            direction = "📈 Positive" if corr['pearson_correlation'] > 0 else "📉 Negative"
            print(f"   {i}. {corr['variable1']} ↔ {corr['variable2']}")
            print(f"      {direction} correlation: {corr['pearson_correlation']:.3f} ({corr['strength']})")
    else:
        print("   No strong correlations detected in customer data")
    
    print("\n📊 3. CUSTOMER SEGMENTATION")
    print("-" * 30)
    
    segmentation_features = ['total_spent', 'total_orders', 'avg_order_value']
    segmentation = analyzer.customer_segmentation(customer_data, segmentation_features, n_clusters=4)
    
    if 'error' not in segmentation:
        print(f"🎯 Identified {segmentation['number_of_clusters']} Customer Segments:")
        cluster_stats = segmentation['cluster_statistics']
        
        # Sort clusters by size
        sorted_clusters = sorted(cluster_stats.items(), key=lambda x: x[1]['size'], reverse=True)
        
        for i, (cluster_name, stats) in enumerate(sorted_clusters, 1):
            avg_spent = stats['means']['total_spent']
            avg_orders = stats['means']['total_orders']
            size_pct = stats['percentage']
            
            # Classify segment
            if avg_spent > 1500:
                segment_type = "🌟 Premium"
            elif avg_spent > 600:
                segment_type = "💎 Standard"
            else:
                segment_type = "🎯 Budget"
                
            print(f"   {i}. {segment_type} Segment ({size_pct:.1f}% of customers)")
            print(f"      • Average Spending: ${avg_spent:.2f}")
            print(f"      • Average Orders: {avg_orders:.1f}")
            print(f"      • Customer Count: {stats['size']:,}")
    
    print("\n🚨 4. ANOMALY DETECTION")
    print("-" * 30)
    
    anomalies = analyzer.anomaly_detection(customer_data, ['total_spent', 'total_orders'])
    if 'error' not in anomalies:
        anomaly_rate = anomalies['anomaly_percentage']
        print(f"🔍 Anomaly Detection Results:")
        print(f"   • Anomaly Rate: {anomaly_rate:.1f}%")
        print(f"   • Anomalies Found: {anomalies['anomalies_detected']} customers")
        
        if anomalies['anomalies_detected'] > 0:
            most_anomalous = anomalies['most_anomalous']
            print(f"   • Most Unusual Customer ID: {most_anomalous['index']}")
            print(f"   • Anomaly Score: {most_anomalous['score']:.4f}")
    
    # === SALES ANALYSIS ===
    print("\n" + "="*50)
    print("💰 SALES & PRODUCT ANALYSIS")
    print("="*50)
    
    print("\n📈 1. REVENUE REGRESSION ANALYSIS")
    print("-" * 30)
    
    # Predict revenue based on price and volume
    regression = analyzer.regression_analysis(
        sales_data, 
        target_col='total_revenue',
        feature_cols=['unit_price', 'units_sold', 'profit_margin']
    )
    
    if 'error' not in regression:
        r2 = regression['model_performance']['r_squared']
        print(f"📊 Revenue Prediction Model:")
        print(f"   • Model Accuracy (R²): {r2:.3f} ({r2*100:.1f}% of variance explained)")
        
        print(f"\n🎯 Key Factors Affecting Revenue:")
        coefficients = regression['coefficients']
        for feature, coef in coefficients.items():
            impact = "📈 Positive" if coef > 0 else "📉 Negative"
            print(f"   • {feature}: {impact} impact ({coef:.2f})")
        
        # Statistical significance
        f_pvalue = regression['statistical_significance']['f_pvalue']
        significance = "✅ Statistically Significant" if f_pvalue < 0.05 else "❌ Not Significant"
        print(f"\n   {significance} (p-value: {f_pvalue:.4f})")
    
    print("\n🧪 2. HYPOTHESIS TESTING")
    print("-" * 30)
    
    # Test if Electronics category has significantly higher revenue than others
    sales_data['is_electronics'] = (sales_data['category'] == 'Electronics').astype(str)
    ttest_result = analyzer.hypothesis_testing(
        sales_data,
        test_type='t_test',
        column='total_revenue',
        group_column='is_electronics'
    )
    
    if 'error' not in ttest_result:
        group1_mean = ttest_result.get('group1_mean', 0)
        group2_mean = ttest_result.get('group2_mean', 0)
        electronics_mean = group1_mean if group1_mean > group2_mean else group2_mean
        other_mean = group2_mean if group1_mean > group2_mean else group1_mean
        p_value = ttest_result['p_value']
        significant = ttest_result['significant']
        
        result_text = "✅ Significantly different" if significant else "❌ No significant difference"
        print(f"📊 Electronics vs Other Categories Revenue Test:")
        print(f"   • Electronics Average Revenue: ${electronics_mean:.2f}")
        print(f"   • Other Categories Average: ${other_mean:.2f}")
        print(f"   • Statistical Result: {result_text} (p-value: {p_value:.4f})")
    
    # === INSIGHTS GENERATION ===
    print("\n" + "="*50)
    print("💡 AI-GENERATED BUSINESS INSIGHTS")
    print("="*50)
    
    # Generate insights for customer data
    customer_insights = analyzer.generate_insights(customer_data, {
        **customer_stats,
        **customer_corr,
        **segmentation
    })
    
    print("\n👥 Customer Insights:")
    for i, insight in enumerate(customer_insights[:3], 1):
        print(f"   {i}. {insight}")
    
    # Generate insights for sales data
    sales_combined_results = {
        **analyzer.descriptive_statistics(sales_data),
        **analyzer.correlation_analysis(sales_data)
    }
    
    sales_insights = analyzer.generate_insights(sales_data, sales_combined_results)
    
    print("\n💰 Sales Insights:")
    for i, insight in enumerate(sales_insights[:3], 1):
        print(f"   {i}. {insight}")
    
    # === SUMMARY & RECOMMENDATIONS ===
    print("\n" + "="*50)
    print("🎯 EXECUTIVE SUMMARY & RECOMMENDATIONS")
    print("="*50)
    
    print("\n📊 Key Findings:")
    
    # Customer findings
    if 'error' not in customer_stats:
        avg_customer_value = customer_stats['numeric_statistics']['mean']['total_spent']
        print(f"   • Customer base of {len(customer_data):,} with avg. value ${avg_customer_value:.2f}")
    
    if 'error' not in segmentation:
        largest_segment = max(segmentation['cluster_statistics'].items(), key=lambda x: x[1]['size'])
        print(f"   • Largest customer segment represents {largest_segment[1]['percentage']:.1f}% of base")
    
    # Sales findings
    if 'error' not in regression:
        print(f"   • Revenue prediction model explains {regression['model_performance']['r_squared']*100:.1f}% of variance")
    
    print("\n💼 Business Recommendations:")
    print("   1. 🎯 Focus marketing efforts on high-value customer segments")
    print("   2. 📊 Investigate anomalous customers for potential fraud or VIP opportunities")
    print("   3. 💰 Optimize pricing strategy based on revenue model insights")
    print("   4. 🔄 Develop retention programs for occasional customers")
    print("   5. 📈 Leverage strong correlations for cross-selling opportunities")
    
    print("\n" + "="*70)
    print("✅ Analysis Complete - Demonstrating Advanced Statistical Capabilities")
    print("🎓 Skills Showcased: Python, Statistical Analysis, Machine Learning, Business Intelligence")
    print("📧 Contact: vinaysaikamineni@email.com")
    print("="*70)

if __name__ == "__main__":
    run_comprehensive_analysis()
