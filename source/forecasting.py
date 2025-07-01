import pandas as pd
import numpy as np
from prophet import Prophet
from statsmodels.tsa.arima.model import ARIMA
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler
from scipy import stats
import snowflake.connector
import os
import pickle
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class AdvancedForecaster:
    def __init__(self, snowflake_conn):
        self.conn = snowflake_conn
        self.models = {}
        self.forecasts = {}
        self.ensemble_forecasts = {}
        self.anomalies = {}
        self.correlations = {}
        
    def get_series_data(self, series_id):
        cursor = self.conn.cursor()
        
        query = """
        SELECT 
            f.DATE_KEY as ds,
            f.RAW_VALUE as y,
            c.CATEGORY_NAME
        FROM FACT_ECONOMIC_DATA f
        JOIN DIM_CATEGORY c ON f.CATEGORY_KEY = c.CATEGORY_KEY
        WHERE f.SERIES_ID = %s
        ORDER BY f.DATE_KEY
        """
        
        cursor.execute(query, (series_id,))
        results = cursor.fetchall()
        
        df = pd.DataFrame(results, columns=['ds', 'y', 'category_name'])
        df['ds'] = pd.to_datetime(df['ds'])
        
        return df
    
    def get_all_series_data(self):
        """Get all economic data for correlation analysis"""
        cursor = self.conn.cursor()
        
        query = """
        SELECT 
            f.DATE_KEY as ds,
            f.SERIES_ID,
            f.RAW_VALUE as value,
            c.CATEGORY_NAME
        FROM FACT_ECONOMIC_DATA f
        JOIN DIM_CATEGORY c ON f.CATEGORY_KEY = c.CATEGORY_KEY
        ORDER BY f.DATE_KEY, f.SERIES_ID
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        df = pd.DataFrame(results, columns=['ds', 'series_id', 'value', 'category_name'])
        df['ds'] = pd.to_datetime(df['ds'])
        
        # Pivot to get series as columns
        pivot_df = df.pivot(index='ds', columns='series_id', values='value')
        
        return pivot_df
    
    def detect_anomalies(self, series_id, contamination=0.1):
        """Detect anomalies using Isolation Forest"""
        print(f"Detecting anomalies for {series_id}...")
        
        df = self.get_series_data(series_id)
        
        if len(df) < 24:
            print(f"Insufficient data for anomaly detection: {len(df)} points")
            return None
            
        # Prepare features for anomaly detection
        df['value_lag1'] = df['y'].shift(1)
        df['value_lag3'] = df['y'].shift(3)
        df['rolling_mean_6'] = df['y'].rolling(window=6, min_periods=3).mean()
        df['rolling_std_6'] = df['y'].rolling(window=6, min_periods=3).std()
        df['yoy_change'] = df['y'].pct_change(periods=12) * 100
        
        # Drop rows with NaN values
        features_df = df[['y', 'value_lag1', 'value_lag3', 'rolling_mean_6', 'rolling_std_6', 'yoy_change']].dropna()
        
        if len(features_df) < 12:
            print(f"Insufficient clean data for anomaly detection")
            return None
        
        # Standardize features
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features_df)
        
        # Fit Isolation Forest
        iso_forest = IsolationForest(contamination=contamination, random_state=42, n_estimators=100)
        anomaly_labels = iso_forest.fit_predict(features_scaled)
        anomaly_scores = iso_forest.score_samples(features_scaled)
        
        # Create anomaly results
        anomaly_results = df.loc[features_df.index].copy()
        anomaly_results['anomaly'] = anomaly_labels == -1
        anomaly_results['anomaly_score'] = anomaly_scores
        
        # Store results
        self.anomalies[series_id] = anomaly_results
        
        anomaly_count = sum(anomaly_labels == -1)
        print(f"Found {anomaly_count} anomalies out of {len(features_df)} points ({anomaly_count/len(features_df)*100:.1f}%)")
        
        return anomaly_results
    
    def calculate_correlations(self):
        """Calculate cross-series correlations"""
        print("Calculating cross-series correlations...")
        
        all_data = self.get_all_series_data()
        
        if all_data.empty:
            print("No data available for correlation analysis")
            return None
        
        # Calculate correlation matrix
        correlation_matrix = all_data.corr()
        
        # Find strongest correlations (excluding self-correlations)
        correlations = []
        for i in range(len(correlation_matrix.columns)):
            for j in range(i+1, len(correlation_matrix.columns)):
                series1 = correlation_matrix.columns[i]
                series2 = correlation_matrix.columns[j]
                corr_value = correlation_matrix.iloc[i, j]
                
                if not pd.isna(corr_value):
                    correlations.append({
                        'series1': series1,
                        'series2': series2,
                        'correlation': corr_value,
                        'abs_correlation': abs(corr_value)
                    })
        
        # Sort by absolute correlation strength
        correlations_df = pd.DataFrame(correlations)
        correlations_df = correlations_df.sort_values('abs_correlation', ascending=False)
        
        self.correlations = {
            'matrix': correlation_matrix,
            'top_correlations': correlations_df.head(10)
        }
        
        print("Top 5 correlations:")
        for _, row in correlations_df.head(5).iterrows():
            print(f"  {row['series1']} ↔ {row['series2']}: {row['correlation']:.3f}")
        
        return self.correlations
    
    def train_ensemble_model(self, series_id, periods=6):
        """Train ensemble model combining Prophet and ARIMA"""
        print(f"Training ensemble model for {series_id}...")
        
        df = self.get_series_data(series_id)
        
        if len(df) < 24:
            print(f"Insufficient data for ensemble model: {len(df)} points")
            return None
        
        # Train Prophet model
        prophet_model = Prophet(
            yearly_seasonality=True,
            weekly_seasonality=False,
            daily_seasonality=False,
            seasonality_mode='additive',
            changepoint_prior_scale=0.05,
            interval_width=0.80
        )
        
        prophet_model.fit(df[['ds', 'y']])
        future = prophet_model.make_future_dataframe(periods=periods, freq='M')
        prophet_forecast = prophet_model.predict(future)
        
        # Train ARIMA model
        ts = df.set_index('ds')['y']
        
        # Auto-select ARIMA parameters
        best_aic = float('inf')
        best_order = None
        arima_model = None
        
        for p in range(3):
            for d in range(2):
                for q in range(3):
                    try:
                        model = ARIMA(ts, order=(p, d, q))
                        fitted_model = model.fit()
                        if fitted_model.aic < best_aic:
                            best_aic = fitted_model.aic
                            best_order = (p, d, q)
                            arima_model = fitted_model
                    except:
                        continue
        
        if arima_model is None:
            print(f"Failed to fit ARIMA model for {series_id}")
            return None
        
        # Generate ARIMA forecast
        arima_forecast_values = arima_model.forecast(steps=periods)
        arima_conf_int = arima_model.get_forecast(steps=periods).conf_int()
        
        # Create ensemble forecast
        last_date = ts.index[-1]
        future_dates = pd.date_range(start=last_date + pd.DateOffset(months=1), 
                                   periods=periods, freq='M')
        
        # Get Prophet future predictions only
        prophet_future = prophet_forecast[prophet_forecast['ds'] > last_date].head(periods)
        
        # Ensemble weights (can be optimized based on historical performance)
        prophet_weight = 0.6
        arima_weight = 0.4
        
        ensemble_forecast = []
        for i in range(periods):
            prophet_val = prophet_future.iloc[i]['yhat']
            arima_val = arima_forecast_values.iloc[i]
            
            ensemble_val = prophet_weight * prophet_val + arima_weight * arima_val
            ensemble_forecast.append(ensemble_val)
        
        # Calculate ensemble confidence intervals
        ensemble_lower = []
        ensemble_upper = []
        for i in range(periods):
            prophet_lower = prophet_future.iloc[i]['yhat_lower']
            prophet_upper = prophet_future.iloc[i]['yhat_upper']
            arima_lower = arima_conf_int.iloc[i, 0]
            arima_upper = arima_conf_int.iloc[i, 1]
            
            # Weighted average of confidence intervals
            ens_lower = prophet_weight * prophet_lower + arima_weight * arima_lower
            ens_upper = prophet_weight * prophet_upper + arima_weight * arima_upper
            
            ensemble_lower.append(ens_lower)
            ensemble_upper.append(ens_upper)
        
        ensemble_df = pd.DataFrame({
            'ds': future_dates,
            'yhat': ensemble_forecast,
            'yhat_lower': ensemble_lower,
            'yhat_upper': ensemble_upper,
            'prophet_forecast': prophet_future['yhat'].values,
            'arima_forecast': arima_forecast_values.values
        })
        
        # Store ensemble results
        self.ensemble_forecasts[series_id] = {
            'forecast': ensemble_df,
            'prophet_weight': prophet_weight,
            'arima_weight': arima_weight,
            'arima_order': best_order,
            'training_points': len(df)
        }
        
        print(f"Ensemble model trained successfully for {series_id}")
        print(f"  Prophet weight: {prophet_weight}, ARIMA weight: {arima_weight}")
        print(f"  ARIMA order: {best_order}")
        
        return ensemble_df
    
    def get_advanced_summary(self, series_id):
        """Get comprehensive analysis summary for a series"""
        summary = {
            'series_id': series_id,
            'analysis_date': datetime.now()
        }
        
        # Basic data info
        df = self.get_series_data(series_id)
        summary['data_points'] = len(df)
        summary['date_range'] = [df['ds'].min(), df['ds'].max()]
        summary['latest_value'] = df['y'].iloc[-1]
        
        # Ensemble forecast
        if series_id in self.ensemble_forecasts:
            ensemble = self.ensemble_forecasts[series_id]
            forecast_df = ensemble['forecast']
            summary['ensemble_forecast'] = {
                'next_3_months': forecast_df['yhat'].head(3).tolist(),
                'forecast_dates': forecast_df['ds'].head(3).dt.strftime('%Y-%m').tolist(),
                'confidence_lower': forecast_df['yhat_lower'].head(3).tolist(),
                'confidence_upper': forecast_df['yhat_upper'].head(3).tolist(),
                'prophet_component': forecast_df['prophet_forecast'].head(3).tolist(),
                'arima_component': forecast_df['arima_forecast'].head(3).tolist()
            }
        
        # Anomaly detection results
        if series_id in self.anomalies:
            anomaly_df = self.anomalies[series_id]
            recent_anomalies = anomaly_df[anomaly_df['anomaly'] == True].tail(5)
            summary['anomalies'] = {
                'total_anomalies': sum(anomaly_df['anomaly']),
                'anomaly_rate': sum(anomaly_df['anomaly']) / len(anomaly_df) * 100,
                'recent_anomalies': recent_anomalies[['ds', 'y']].to_dict('records') if len(recent_anomalies) > 0 else []
            }
        
        # Correlation insights
        if hasattr(self, 'correlations') and self.correlations:
            top_corr = self.correlations['top_correlations']
            related_series = top_corr[
                (top_corr['series1'] == series_id) | (top_corr['series2'] == series_id)
            ].head(3)
            
            correlations_list = []
            for _, row in related_series.iterrows():
                other_series = row['series2'] if row['series1'] == series_id else row['series1']
                correlations_list.append({
                    'related_series': other_series,
                    'correlation': row['correlation']
                })
            
            summary['correlations'] = correlations_list
        
        return summary
    
    def save_advanced_models(self, filepath='models/advanced_consumer_models.pkl'):
        """Save all advanced models and analysis results"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        with open(filepath, 'wb') as f:
            pickle.dump({
                'ensemble_forecasts': self.ensemble_forecasts,
                'anomalies': self.anomalies,
                'correlations': self.correlations,
                'saved_date': datetime.now()
            }, f)
        
        print(f"Advanced models saved to {filepath}")

def run_advanced_analysis():
    from dotenv import load_dotenv
    load_dotenv()
    
    # Connect to Snowflake
    mfa_token = input("Enter your MFA code: ")
    conn = snowflake.connector.connect(
        user=os.getenv('SNOWFLAKE_USER'),
        password=os.getenv('SNOWFLAKE_PASSWORD'),
        account=os.getenv('SNOWFLAKE_ACCOUNT'),
        passcode=mfa_token,
        warehouse='COMPUTE_WH',
        database='CONSUMER_SPENDING_DB',
        schema='ANALYTICS'
    )
    
    # Initialize advanced forecaster
    forecaster = AdvancedForecaster(conn)
    
    # Series to analyze
    series_list = ['PAYEMS', 'UNRATE', 'CPIAUCSL', 'DSPIC96']
    
    print("=== ADVANCED FORECASTING ANALYSIS ===\n")
    
    # 1. Calculate correlations across all series
    correlations = forecaster.calculate_correlations()
    
    print("\n" + "="*50)
    
    # 2. For each series, run full analysis
    for series_id in series_list:
        print(f"\n--- Advanced Analysis for {series_id} ---")
        
        # Anomaly detection
        anomalies = forecaster.detect_anomalies(series_id)
        
        # Ensemble forecasting
        ensemble_forecast = forecaster.train_ensemble_model(series_id)
        
        # Print summary
        if ensemble_forecast is not None:
            summary = forecaster.get_advanced_summary(series_id)
            print(f"\nSummary for {series_id}:")
            print(f"  Data points: {summary['data_points']}")
            print(f"  Latest value: {summary['latest_value']:.2f}")
            
            if 'ensemble_forecast' in summary:
                ens_forecast = summary['ensemble_forecast']
                print(f"  Next 3 months forecast: {[f'{x:.2f}' for x in ens_forecast['next_3_months']]}")
            
            if 'anomalies' in summary:
                anom = summary['anomalies']
                print(f"  Anomaly rate: {anom['anomaly_rate']:.1f}%")
            
            if 'correlations' in summary and summary['correlations']:
                print("  Top correlations:")
                for corr in summary['correlations'][:2]:
                    print(f"    {corr['related_series']}: {corr['correlation']:.3f}")
    
    # 3. Save all results
    forecaster.save_advanced_models()
    
    print("\n=== ADVANCED ANALYSIS COMPLETE ===")
    print("Features implemented:")
    print("✓ Ensemble forecasting (Prophet + ARIMA)")
    print("✓ Anomaly detection (Isolation Forest)")
    print("✓ Cross-series correlation analysis")
    print("✓ Advanced model persistence")
    
    conn.close()

if __name__ == "__main__":
    run_advanced_analysis()