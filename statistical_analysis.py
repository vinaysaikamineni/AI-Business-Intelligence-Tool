"""
📊 Statistical Analysis Module for AI Business Intelligence Tool

This module provides comprehensive statistical analysis capabilities including:
- Descriptive Statistics
- Correlation Analysis  
- Regression Analysis
- Time Series Analysis
- Hypothesis Testing
- Customer Segmentation
- Anomaly Detection
- Statistical Insights Generation

Author: Vinay Sai Kamineni
"""

import pandas as pd
import numpy as np
import scipy.stats as stats
from scipy.stats import chi2_contingency, pearsonr, spearmanr
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.ensemble import IsolationForest
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller
import warnings
from typing import Dict, List, Tuple, Optional, Any
import logging

warnings.filterwarnings('ignore')

class StatisticalAnalyzer:
    """
    Comprehensive statistical analysis class for business intelligence
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.scaler = StandardScaler()
        
    def descriptive_statistics(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Calculate comprehensive descriptive statistics
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing descriptive statistics
        """
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            categorical_cols = df.select_dtypes(include=['object', 'category']).columns
            
            results = {
                'summary': {
                    'total_rows': len(df),
                    'total_columns': len(df.columns),
                    'numeric_columns': len(numeric_cols),
                    'categorical_columns': len(categorical_cols),
                    'missing_values': df.isnull().sum().sum()
                },
                'numeric_statistics': {},
                'categorical_statistics': {},
                'distribution_analysis': {}
            }
            
            # Numeric statistics
            if len(numeric_cols) > 0:
                numeric_df = df[numeric_cols]
                results['numeric_statistics'] = {
                    'mean': numeric_df.mean().to_dict(),
                    'median': numeric_df.median().to_dict(),
                    'std': numeric_df.std().to_dict(),
                    'min': numeric_df.min().to_dict(),
                    'max': numeric_df.max().to_dict(),
                    'quartiles': {
                        'q1': numeric_df.quantile(0.25).to_dict(),
                        'q3': numeric_df.quantile(0.75).to_dict()
                    },
                    'skewness': numeric_df.skew().to_dict(),
                    'kurtosis': numeric_df.kurtosis().to_dict()
                }
                
                # Distribution analysis
                for col in numeric_cols:
                    if df[col].notna().sum() > 0:
                        # Normality test
                        stat, p_value = stats.normaltest(df[col].dropna())
                        results['distribution_analysis'][col] = {
                            'normality_test': {
                                'statistic': float(stat),
                                'p_value': float(p_value),
                                'is_normal': p_value > 0.05
                            }
                        }
            
            # Categorical statistics
            if len(categorical_cols) > 0:
                for col in categorical_cols:
                    value_counts = df[col].value_counts()
                    results['categorical_statistics'][col] = {
                        'unique_values': df[col].nunique(),
                        'most_frequent': value_counts.index[0] if len(value_counts) > 0 else None,
                        'mode_frequency': int(value_counts.iloc[0]) if len(value_counts) > 0 else 0,
                        'value_distribution': value_counts.head(10).to_dict()
                    }
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in descriptive statistics: {str(e)}")
            return {'error': str(e)}
    
    def correlation_analysis(self, df: pd.DataFrame) -> Dict[str, Any]:
        """
        Perform comprehensive correlation analysis
        
        Args:
            df: DataFrame to analyze
            
        Returns:
            Dictionary containing correlation analysis results
        """
        try:
            numeric_cols = df.select_dtypes(include=[np.number]).columns
            
            if len(numeric_cols) < 2:
                return {'error': 'Need at least 2 numeric columns for correlation analysis'}
            
            numeric_df = df[numeric_cols].dropna()
            
            # Pearson correlation
            pearson_corr = numeric_df.corr(method='pearson')
            
            # Spearman correlation  
            spearman_corr = numeric_df.corr(method='spearman')
            
            # Find strong correlations
            strong_correlations = []
            for i in range(len(pearson_corr.columns)):
                for j in range(i+1, len(pearson_corr.columns)):
                    col1, col2 = pearson_corr.columns[i], pearson_corr.columns[j]
                    pearson_val = pearson_corr.iloc[i, j]
                    spearman_val = spearman_corr.iloc[i, j]
                    
                    if abs(pearson_val) > 0.5:  # Strong correlation threshold
                        strong_correlations.append({
                            'variable1': col1,
                            'variable2': col2,
                            'pearson_correlation': float(pearson_val),
                            'spearman_correlation': float(spearman_val),
                            'strength': 'Strong' if abs(pearson_val) > 0.7 else 'Moderate',
                            'direction': 'Positive' if pearson_val > 0 else 'Negative'
                        })
            
            return {
                'pearson_correlation_matrix': pearson_corr.to_dict(),
                'spearman_correlation_matrix': spearman_corr.to_dict(),
                'strong_correlations': strong_correlations,
                'correlation_summary': {
                    'total_pairs': len(strong_correlations),
                    'strong_positive': len([c for c in strong_correlations if c['pearson_correlation'] > 0.7]),
                    'strong_negative': len([c for c in strong_correlations if c['pearson_correlation'] < -0.7])
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in correlation analysis: {str(e)}")
            return {'error': str(e)}
    
    def regression_analysis(self, df: pd.DataFrame, target_col: str, feature_cols: List[str] = None) -> Dict[str, Any]:
        """
        Perform regression analysis
        
        Args:
            df: DataFrame to analyze
            target_col: Target variable for regression
            feature_cols: List of feature columns (if None, uses all numeric columns)
            
        Returns:
            Dictionary containing regression analysis results
        """
        try:
            if target_col not in df.columns:
                return {'error': f'Target column {target_col} not found in data'}
            
            if feature_cols is None:
                feature_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                              if col != target_col]
            
            if len(feature_cols) == 0:
                return {'error': 'No feature columns found for regression'}
            
            # Prepare data
            clean_df = df[feature_cols + [target_col]].dropna()
            X = clean_df[feature_cols]
            y = clean_df[target_col]
            
            if len(clean_df) < 10:
                return {'error': 'Insufficient data points for regression analysis'}
            
            # Simple Linear Regression (sklearn)
            lr = LinearRegression()
            lr.fit(X, y)
            y_pred = lr.predict(X)
            
            # Statistical regression (statsmodels)
            X_stats = sm.add_constant(X)
            model = sm.OLS(y, X_stats).fit()
            
            # Calculate metrics
            r2 = r2_score(y, y_pred)
            rmse = np.sqrt(mean_squared_error(y, y_pred))
            
            return {
                'model_performance': {
                    'r_squared': float(r2),
                    'adjusted_r_squared': float(model.rsquared_adj),
                    'rmse': float(rmse),
                    'mean_absolute_error': float(np.mean(np.abs(y - y_pred)))
                },
                'coefficients': {
                    feature_cols[i]: float(lr.coef_[i]) for i in range(len(feature_cols))
                },
                'intercept': float(lr.intercept_),
                'statistical_significance': {
                    'f_statistic': float(model.fvalue),
                    'f_pvalue': float(model.f_pvalue),
                    'coefficient_pvalues': model.pvalues.to_dict()
                },
                'feature_importance': {
                    col: abs(float(lr.coef_[i])) for i, col in enumerate(feature_cols)
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in regression analysis: {str(e)}")
            return {'error': str(e)}
    
    def hypothesis_testing(self, df: pd.DataFrame, test_type: str, **kwargs) -> Dict[str, Any]:
        """
        Perform various hypothesis tests
        
        Args:
            df: DataFrame to analyze
            test_type: Type of test ('t_test', 'chi_square', 'anova')
            **kwargs: Additional parameters for specific tests
            
        Returns:
            Dictionary containing test results
        """
        try:
            if test_type == 't_test':
                return self._t_test(df, **kwargs)
            elif test_type == 'chi_square':
                return self._chi_square_test(df, **kwargs)
            elif test_type == 'anova':
                return self._anova_test(df, **kwargs)
            else:
                return {'error': f'Unknown test type: {test_type}'}
                
        except Exception as e:
            self.logger.error(f"Error in hypothesis testing: {str(e)}")
            return {'error': str(e)}
    
    def _t_test(self, df: pd.DataFrame, column: str, test_value: float = None, 
                group_column: str = None) -> Dict[str, Any]:
        """One-sample or two-sample t-test"""
        if group_column:
            # Two-sample t-test
            groups = df[group_column].unique()
            if len(groups) != 2:
                return {'error': 'Two-sample t-test requires exactly 2 groups'}
            
            group1 = df[df[group_column] == groups[0]][column].dropna()
            group2 = df[df[group_column] == groups[1]][column].dropna()
            
            stat, p_value = stats.ttest_ind(group1, group2)
            
            return {
                'test_type': 'Two-sample t-test',
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'group1_mean': float(group1.mean()),
                'group2_mean': float(group2.mean()),
                'group1_size': len(group1),
                'group2_size': len(group2)
            }
        else:
            # One-sample t-test
            if test_value is None:
                test_value = df[column].mean()
            
            data = df[column].dropna()
            stat, p_value = stats.ttest_1samp(data, test_value)
            
            return {
                'test_type': 'One-sample t-test',
                'statistic': float(stat),
                'p_value': float(p_value),
                'significant': p_value < 0.05,
                'sample_mean': float(data.mean()),
                'test_value': float(test_value),
                'sample_size': len(data)
            }
    
    def _chi_square_test(self, df: pd.DataFrame, col1: str, col2: str) -> Dict[str, Any]:
        """Chi-square test of independence"""
        contingency_table = pd.crosstab(df[col1], df[col2])
        stat, p_value, dof, expected = chi2_contingency(contingency_table)
        
        return {
            'test_type': 'Chi-square test of independence',
            'statistic': float(stat),
            'p_value': float(p_value),
            'degrees_of_freedom': int(dof),
            'significant': p_value < 0.05,
            'contingency_table': contingency_table.to_dict(),
            'cramers_v': np.sqrt(stat / (contingency_table.sum().sum() * (min(contingency_table.shape) - 1)))
        }
    
    def _anova_test(self, df: pd.DataFrame, value_column: str, group_column: str) -> Dict[str, Any]:
        """One-way ANOVA test"""
        groups = [group[value_column].dropna() for name, group in df.groupby(group_column)]
        
        if len(groups) < 2:
            return {'error': 'ANOVA requires at least 2 groups'}
        
        stat, p_value = stats.f_oneway(*groups)
        
        return {
            'test_type': 'One-way ANOVA',
            'statistic': float(stat),
            'p_value': float(p_value),
            'significant': p_value < 0.05,
            'number_of_groups': len(groups),
            'group_means': {name: float(group[value_column].mean()) 
                          for name, group in df.groupby(group_column)}
        }
    
    def customer_segmentation(self, df: pd.DataFrame, feature_cols: List[str], 
                            n_clusters: int = 4) -> Dict[str, Any]:
        """
        Perform customer segmentation using K-means clustering
        
        Args:
            df: DataFrame with customer data
            feature_cols: Columns to use for clustering
            n_clusters: Number of clusters
            
        Returns:
            Dictionary containing segmentation results
        """
        try:
            # Prepare data
            cluster_df = df[feature_cols].dropna()
            
            if len(cluster_df) < n_clusters:
                return {'error': f'Need at least {n_clusters} data points for {n_clusters} clusters'}
            
            # Scale features
            scaled_features = self.scaler.fit_transform(cluster_df)
            
            # Perform clustering
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(scaled_features)
            
            # Add clusters to original data
            result_df = cluster_df.copy()
            result_df['cluster'] = clusters
            
            # Calculate cluster statistics
            cluster_stats = {}
            for i in range(n_clusters):
                cluster_data = result_df[result_df['cluster'] == i]
                cluster_stats[f'cluster_{i}'] = {
                    'size': len(cluster_data),
                    'percentage': len(cluster_data) / len(result_df) * 100,
                    'means': cluster_data[feature_cols].mean().to_dict(),
                    'characteristics': self._describe_cluster(cluster_data, feature_cols)
                }
            
            return {
                'number_of_clusters': n_clusters,
                'total_customers': len(result_df),
                'cluster_centers': kmeans.cluster_centers_.tolist(),
                'cluster_statistics': cluster_stats,
                'inertia': float(kmeans.inertia_),
                'silhouette_score': self._calculate_silhouette_score(scaled_features, clusters)
            }
            
        except Exception as e:
            self.logger.error(f"Error in customer segmentation: {str(e)}")
            return {'error': str(e)}
    
    def _describe_cluster(self, cluster_data: pd.DataFrame, feature_cols: List[str]) -> str:
        """Generate natural language description of cluster characteristics"""
        descriptions = []
        
        for col in feature_cols:
            mean_val = cluster_data[col].mean()
            overall_mean = cluster_data[col].mean()  # This should be calculated from full dataset
            
            if mean_val > overall_mean * 1.2:
                descriptions.append(f"High {col}")
            elif mean_val < overall_mean * 0.8:
                descriptions.append(f"Low {col}")
            else:
                descriptions.append(f"Average {col}")
        
        return ", ".join(descriptions)
    
    def _calculate_silhouette_score(self, X, labels):
        """Calculate silhouette score for clustering"""
        try:
            from sklearn.metrics import silhouette_score
            return float(silhouette_score(X, labels))
        except:
            return None
    
    def anomaly_detection(self, df: pd.DataFrame, feature_cols: List[str] = None) -> Dict[str, Any]:
        """
        Detect anomalies in the data using Isolation Forest
        
        Args:
            df: DataFrame to analyze
            feature_cols: Columns to use for anomaly detection
            
        Returns:
            Dictionary containing anomaly detection results
        """
        try:
            if feature_cols is None:
                feature_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            
            analysis_df = df[feature_cols].dropna()
            
            if len(analysis_df) < 10:
                return {'error': 'Need at least 10 data points for anomaly detection'}
            
            # Fit Isolation Forest
            iso_forest = IsolationForest(contamination=0.1, random_state=42)
            anomaly_labels = iso_forest.fit_predict(analysis_df)
            anomaly_scores = iso_forest.decision_function(analysis_df)
            
            # Identify anomalies
            anomalies = analysis_df[anomaly_labels == -1].copy()
            anomalies['anomaly_score'] = anomaly_scores[anomaly_labels == -1]
            
            return {
                'total_data_points': len(analysis_df),
                'anomalies_detected': len(anomalies),
                'anomaly_percentage': len(anomalies) / len(analysis_df) * 100,
                'anomaly_indices': anomalies.index.tolist(),
                'most_anomalous': {
                    'index': int(anomalies['anomaly_score'].idxmin()),
                    'score': float(anomalies['anomaly_score'].min()),
                    'values': anomalies.loc[anomalies['anomaly_score'].idxmin(), feature_cols].to_dict()
                },
                'feature_statistics': {
                    'normal_data': analysis_df[anomaly_labels == 1][feature_cols].describe().to_dict(),
                    'anomalous_data': anomalies[feature_cols].describe().to_dict()
                }
            }
            
        except Exception as e:
            self.logger.error(f"Error in anomaly detection: {str(e)}")
            return {'error': str(e)}
    
    def generate_insights(self, df: pd.DataFrame, analysis_results: Dict[str, Any]) -> List[str]:
        """
        Generate business insights based on statistical analysis results
        
        Args:
            df: Original DataFrame
            analysis_results: Results from various statistical analyses
            
        Returns:
            List of insights in natural language
        """
        insights = []
        
        try:
            # Descriptive statistics insights
            if 'numeric_statistics' in analysis_results:
                stats_data = analysis_results['numeric_statistics']
                for col, mean_val in stats_data.get('mean', {}).items():
                    std_val = stats_data.get('std', {}).get(col, 0)
                    if std_val > 0:
                        cv = std_val / mean_val
                        if cv > 1:
                            insights.append(f"'{col}' shows high variability (CV: {cv:.2f}), indicating diverse values across the dataset")
                        elif cv < 0.1:
                            insights.append(f"'{col}' shows low variability (CV: {cv:.2f}), indicating consistent values")
            
            # Correlation insights
            if 'strong_correlations' in analysis_results:
                correlations = analysis_results['strong_correlations']
                for corr in correlations[:3]:  # Top 3 correlations
                    direction = "positively" if corr['pearson_correlation'] > 0 else "negatively"
                    strength = corr['strength'].lower()
                    insights.append(f"'{corr['variable1']}' and '{corr['variable2']}' are {strength} {direction} correlated (r={corr['pearson_correlation']:.3f})")
            
            # Cluster insights
            if 'cluster_statistics' in analysis_results:
                cluster_stats = analysis_results['cluster_statistics']
                largest_cluster = max(cluster_stats.items(), key=lambda x: x[1]['size'])
                insights.append(f"Customer segmentation reveals {len(cluster_stats)} distinct groups, with {largest_cluster[0]} being the largest ({largest_cluster[1]['percentage']:.1f}% of customers)")
            
            # Anomaly insights
            if 'anomaly_percentage' in analysis_results:
                anomaly_pct = analysis_results['anomaly_percentage']
                if anomaly_pct > 5:
                    insights.append(f"High anomaly rate detected ({anomaly_pct:.1f}%), suggesting potential data quality issues or unique business cases")
                elif anomaly_pct < 1:
                    insights.append(f"Low anomaly rate ({anomaly_pct:.1f}%) indicates consistent data patterns")
                    
            if len(insights) == 0:
                insights.append("Analysis completed successfully. Consider exploring specific relationships between variables for deeper insights.")
                
        except Exception as e:
            self.logger.error(f"Error generating insights: {str(e)}")
            insights.append("Unable to generate automated insights due to data structure complexity.")
        
        return insights
