"""
🧪 Test Suite for Statistical Analysis Module

Comprehensive tests for all statistical analysis functionality including:
- Descriptive Statistics
- Correlation Analysis
- Regression Analysis
- Hypothesis Testing
- Customer Segmentation
- Anomaly Detection

Author: Vinay Sai Kamineni
"""

import pytest
import pandas as pd
import numpy as np
from statistical_analysis import StatisticalAnalyzer
import warnings

warnings.filterwarnings('ignore')

class TestStatisticalAnalyzer:
    """Test class for StatisticalAnalyzer"""
    
    @pytest.fixture
    def analyzer(self):
        """Create StatisticalAnalyzer instance for testing"""
        return StatisticalAnalyzer()
    
    @pytest.fixture
    def sample_data(self):
        """Create sample data for testing"""
        np.random.seed(42)
        data = {
            'sales': np.random.normal(1000, 200, 100),
            'customers': np.random.normal(50, 10, 100),
            'price': np.random.normal(25, 5, 100),
            'category': np.random.choice(['A', 'B', 'C'], 100),
            'region': np.random.choice(['North', 'South'], 100)
        }
        return pd.DataFrame(data)
    
    @pytest.fixture
    def chinook_like_data(self):
        """Create Chinook-like sample data for testing"""
        np.random.seed(42)
        data = {
            'total': np.random.uniform(5, 50, 200),
            'quantity': np.random.randint(1, 10, 200),
            'unit_price': np.random.uniform(0.99, 2.99, 200),
            'customer_id': np.random.randint(1, 60, 200),
            'genre': np.random.choice(['Rock', 'Jazz', 'Classical', 'Pop'], 200),
            'country': np.random.choice(['USA', 'Canada', 'Germany', 'UK'], 200)
        }
        return pd.DataFrame(data)
    
    def test_descriptive_statistics_basic(self, analyzer, sample_data):
        """Test basic descriptive statistics functionality"""
        result = analyzer.descriptive_statistics(sample_data)
        
        assert 'summary' in result
        assert 'numeric_statistics' in result
        assert 'categorical_statistics' in result
        assert 'distribution_analysis' in result
        
        # Check summary statistics
        assert result['summary']['total_rows'] == 100
        assert result['summary']['total_columns'] == 5
        assert result['summary']['numeric_columns'] == 3
        assert result['summary']['categorical_columns'] == 2
        
        # Check numeric statistics
        assert 'mean' in result['numeric_statistics']
        assert 'std' in result['numeric_statistics']
        assert 'min' in result['numeric_statistics']
        assert 'max' in result['numeric_statistics']
        
        # Check categorical statistics
        assert 'category' in result['categorical_statistics']
        assert 'region' in result['categorical_statistics']
    
    def test_descriptive_statistics_empty_data(self, analyzer):
        """Test descriptive statistics with empty data"""
        empty_df = pd.DataFrame()
        result = analyzer.descriptive_statistics(empty_df)
        
        assert result['summary']['total_rows'] == 0
        assert result['summary']['total_columns'] == 0
    
    def test_correlation_analysis_basic(self, analyzer, sample_data):
        """Test basic correlation analysis"""
        result = analyzer.correlation_analysis(sample_data)
        
        assert 'pearson_correlation_matrix' in result
        assert 'spearman_correlation_matrix' in result
        assert 'strong_correlations' in result
        assert 'correlation_summary' in result
        
        # Check correlation matrix structure
        pearson_matrix = result['pearson_correlation_matrix']
        assert 'sales' in pearson_matrix
        assert 'customers' in pearson_matrix
        assert 'price' in pearson_matrix
    
    def test_correlation_analysis_insufficient_columns(self, analyzer):
        """Test correlation analysis with insufficient numeric columns"""
        df = pd.DataFrame({'category': ['A', 'B', 'C']})
        result = analyzer.correlation_analysis(df)
        
        assert 'error' in result
        assert 'Need at least 2 numeric columns' in result['error']
    
    def test_regression_analysis_basic(self, analyzer, sample_data):
        """Test basic regression analysis"""
        result = analyzer.regression_analysis(
            sample_data, 
            target_col='sales', 
            feature_cols=['customers', 'price']
        )
        
        assert 'model_performance' in result
        assert 'coefficients' in result
        assert 'intercept' in result
        assert 'statistical_significance' in result
        
        # Check performance metrics
        assert 'r_squared' in result['model_performance']
        assert 'rmse' in result['model_performance']
        
        # Check coefficients
        assert 'customers' in result['coefficients']
        assert 'price' in result['coefficients']
    
    def test_regression_analysis_missing_target(self, analyzer, sample_data):
        """Test regression analysis with missing target column"""
        result = analyzer.regression_analysis(
            sample_data, 
            target_col='nonexistent_column'
        )
        
        assert 'error' in result
        assert 'not found in data' in result['error']
    
    def test_hypothesis_testing_t_test(self, analyzer, sample_data):
        """Test t-test functionality"""
        # One-sample t-test
        result = analyzer.hypothesis_testing(
            sample_data, 
            test_type='t_test', 
            column='sales',
            test_value=1000
        )
        
        assert result['test_type'] == 'One-sample t-test'
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'significant' in result
        
        # Two-sample t-test
        result2 = analyzer.hypothesis_testing(
            sample_data,
            test_type='t_test',
            column='sales',
            group_column='region'
        )
        
        assert result2['test_type'] == 'Two-sample t-test'
        assert 'group1_mean' in result2
        assert 'group2_mean' in result2
    
    def test_hypothesis_testing_chi_square(self, analyzer, sample_data):
        """Test chi-square test functionality"""
        result = analyzer.hypothesis_testing(
            sample_data,
            test_type='chi_square',
            col1='category',
            col2='region'
        )
        
        assert result['test_type'] == 'Chi-square test of independence'
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'degrees_of_freedom' in result
        assert 'contingency_table' in result
    
    def test_hypothesis_testing_anova(self, analyzer, sample_data):
        """Test ANOVA test functionality"""
        result = analyzer.hypothesis_testing(
            sample_data,
            test_type='anova',
            value_column='sales',
            group_column='category'
        )
        
        assert result['test_type'] == 'One-way ANOVA'
        assert 'statistic' in result
        assert 'p_value' in result
        assert 'number_of_groups' in result
        assert 'group_means' in result
    
    def test_customer_segmentation_basic(self, analyzer, chinook_like_data):
        """Test customer segmentation functionality"""
        result = analyzer.customer_segmentation(
            chinook_like_data,
            feature_cols=['total', 'quantity', 'unit_price'],
            n_clusters=3
        )
        
        assert 'number_of_clusters' in result
        assert 'total_customers' in result
        assert 'cluster_statistics' in result
        assert 'cluster_centers' in result
        
        # Check cluster statistics
        cluster_stats = result['cluster_statistics']
        assert len(cluster_stats) == 3
        
        for cluster_name, stats in cluster_stats.items():
            assert 'size' in stats
            assert 'percentage' in stats
            assert 'means' in stats
    
    def test_customer_segmentation_insufficient_data(self, analyzer):
        """Test customer segmentation with insufficient data"""
        small_df = pd.DataFrame({
            'feature1': [1, 2],
            'feature2': [3, 4]
        })
        
        result = analyzer.customer_segmentation(
            small_df,
            feature_cols=['feature1', 'feature2'],
            n_clusters=4
        )
        
        assert 'error' in result
        assert 'Need at least 4 data points' in result['error']
    
    def test_anomaly_detection_basic(self, analyzer, sample_data):
        """Test anomaly detection functionality"""
        result = analyzer.anomaly_detection(
            sample_data,
            feature_cols=['sales', 'customers', 'price']
        )
        
        assert 'total_data_points' in result
        assert 'anomalies_detected' in result
        assert 'anomaly_percentage' in result
        assert 'most_anomalous' in result
        assert 'feature_statistics' in result
        
        # Check anomaly percentage is reasonable
        assert 0 <= result['anomaly_percentage'] <= 100
    
    def test_anomaly_detection_insufficient_data(self, analyzer):
        """Test anomaly detection with insufficient data"""
        small_df = pd.DataFrame({
            'feature1': [1, 2, 3],
            'feature2': [4, 5, 6]
        })
        
        result = analyzer.anomaly_detection(small_df)
        
        assert 'error' in result
        assert 'Need at least 10 data points' in result['error']
    
    def test_generate_insights_basic(self, analyzer, sample_data):
        """Test insights generation functionality"""
        # First run some analyses
        desc_stats = analyzer.descriptive_statistics(sample_data)
        corr_analysis = analyzer.correlation_analysis(sample_data)
        
        # Combine results
        analysis_results = {**desc_stats, **corr_analysis}
        
        insights = analyzer.generate_insights(sample_data, analysis_results)
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        assert all(isinstance(insight, str) for insight in insights)
    
    def test_generate_insights_empty_results(self, analyzer, sample_data):
        """Test insights generation with empty analysis results"""
        insights = analyzer.generate_insights(sample_data, {})
        
        assert isinstance(insights, list)
        assert len(insights) > 0
        assert "Analysis completed successfully" in insights[0]
    
    def test_integration_full_analysis_workflow(self, analyzer, chinook_like_data):
        """Test complete analysis workflow integration"""
        # Run comprehensive analysis
        desc_stats = analyzer.descriptive_statistics(chinook_like_data)
        corr_analysis = analyzer.correlation_analysis(chinook_like_data)
        regression = analyzer.regression_analysis(
            chinook_like_data, 
            target_col='total',
            feature_cols=['quantity', 'unit_price']
        )
        segmentation = analyzer.customer_segmentation(
            chinook_like_data,
            feature_cols=['total', 'quantity'],
            n_clusters=3
        )
        anomalies = analyzer.anomaly_detection(chinook_like_data)
        
        # Verify all analyses completed successfully
        assert 'error' not in desc_stats
        assert 'error' not in corr_analysis
        assert 'error' not in regression
        assert 'error' not in segmentation
        assert 'error' not in anomalies
        
        # Generate comprehensive insights
        all_results = {
            **desc_stats,
            **corr_analysis,
            **regression,
            **segmentation,
            **anomalies
        }
        
        insights = analyzer.generate_insights(chinook_like_data, all_results)
        
        assert len(insights) > 0
        assert all(isinstance(insight, str) for insight in insights)

if __name__ == "__main__":
    # Run tests
    pytest.main([__file__, "-v"])
