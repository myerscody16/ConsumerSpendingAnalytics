import pandas as pd
import numpy as np
import snowflake.connector
import pickle
import os
from datetime import datetime, timedelta
import json
from dotenv import load_dotenv
import warnings
warnings.filterwarnings('ignore')

class EconomicRAGSystem:
    def __init__(self, snowflake_conn):
        self.conn = snowflake_conn
        self.knowledge_base = {}
        self.models = {}
        self.series_metadata = {
            'PAYEMS': {
                'name': 'Total Nonfarm Payrolls',
                'unit': 'thousands of jobs',
                'category': 'Employment',
                'description': 'Total number of people employed in the US economy, excluding farm workers'
            },
            'UNRATE': {
                'name': 'Unemployment Rate',
                'unit': 'percent',
                'category': 'Employment',
                'description': 'Percentage of labor force that is unemployed and actively seeking work'
            },
            'CPIAUCSL': {
                'name': 'Consumer Price Index',
                'unit': 'index (1982-84=100)',
                'category': 'Inflation',
                'description': 'Measure of average change in prices paid by consumers for goods and services'
            },
            'DSPIC96': {
                'name': 'Real Disposable Personal Income',
                'unit': 'billions of chained 2012 dollars',
                'category': 'Income',
                'description': 'Personal income after taxes, adjusted for inflation'
            },
            'PCEC96': {
                'name': 'Personal Consumption Expenditures',
                'unit': 'billions of chained 2012 dollars',
                'category': 'Spending',
                'description': 'Total consumer spending on goods and services'
            },
            'RSAFS': {
                'name': 'Retail Sales',
                'unit': 'millions of dollars',
                'category': 'Retail',
                'description': 'Total retail trade sales excluding food services'
            },
            'ECOMSA': {
                'name': 'E-commerce Sales',
                'unit': 'millions of dollars',
                'category': 'Retail',
                'description': 'E-commerce retail sales as percent of total retail sales'
            }
        }
        
    def load_models(self, filepath='models/advanced_consumer_models.pkl'):
        try:
            with open(filepath, 'rb') as f:
                self.models = pickle.load(f)
            print("Advanced models loaded successfully")
        except FileNotFoundError:
            print(f"Advanced models file not found at {filepath}")
            self.models = {}
    
    def build_knowledge_base(self):
        print("Building economic knowledge base...")
        
        knowledge = {
            'current_data': self._get_current_data(),
            'forecasts': self._get_forecast_summaries(),
            'trends': self._analyze_trends(),
            'correlations': self._get_correlation_insights(),
            'anomalies': self._get_anomaly_insights(),
            'economic_context': self._generate_economic_context()
        }
        
        self.knowledge_base = knowledge
        print("Knowledge base built successfully")
        
    def _get_current_data(self):
        cursor = self.conn.cursor()
        
        query = """
        SELECT 
            f.SERIES_ID,
            c.CATEGORY_NAME,
            f.RAW_VALUE,
            f.DATE_KEY
        FROM FACT_ECONOMIC_DATA f
        JOIN DIM_CATEGORY c ON f.CATEGORY_KEY = c.CATEGORY_KEY
        WHERE f.DATE_KEY IN (
            SELECT MAX(DATE_KEY) 
            FROM FACT_ECONOMIC_DATA f2 
            WHERE f2.SERIES_ID = f.SERIES_ID
        )
        ORDER BY f.SERIES_ID
        """
        
        cursor.execute(query)
        results = cursor.fetchall()
        
        current_data = {}
        for series_id, category_name, value, date in results:
            current_data[series_id] = {
                'value': value,
                'date': date.strftime('%Y-%m-%d'),
                'category': category_name,
                'metadata': self.series_metadata.get(series_id, {})
            }
        
        return current_data
    
    def _get_forecast_summaries(self):
        forecasts = {}
        
        if 'ensemble_forecasts' in self.models:
            for series_id, ensemble_data in self.models['ensemble_forecasts'].items():
                forecast_df = ensemble_data['forecast']
                forecasts[series_id] = {
                    'next_3_months': forecast_df['yhat'].head(3).tolist(),
                    'confidence_intervals': list(zip(
                        forecast_df['yhat_lower'].head(3).tolist(),
                        forecast_df['yhat_upper'].head(3).tolist()
                    )),
                    'forecast_dates': forecast_df['ds'].head(3).dt.strftime('%Y-%m').tolist(),
                    'prophet_component': forecast_df['prophet_forecast'].head(3).tolist(),
                    'arima_component': forecast_df['arima_forecast'].head(3).tolist()
                }
        
        return forecasts
    
    def _analyze_trends(self):
        trends = {}
        
        for series_id in ['PAYEMS', 'UNRATE', 'CPIAUCSL', 'DSPIC96']:
            cursor = self.conn.cursor()
            
            query = """
            SELECT DATE_KEY, RAW_VALUE
            FROM FACT_ECONOMIC_DATA
            WHERE SERIES_ID = %s
            ORDER BY DATE_KEY DESC
            LIMIT 12
            """
            
            cursor.execute(query, (series_id,))
            results = cursor.fetchall()
            
            if len(results) >= 12:
                values = [float(row[1]) for row in reversed(results)]
                
                recent_avg = np.mean(values[-3:])
                earlier_avg = np.mean(values[:3])
                trend_direction = "increasing" if recent_avg > earlier_avg else "decreasing"
                trend_magnitude = abs((recent_avg - earlier_avg) / earlier_avg * 100)
                
                volatility = np.std(values) / np.mean(values) * 100
                
                trends[series_id] = {
                    'direction': trend_direction,
                    'magnitude_percent': trend_magnitude,
                    'volatility_percent': volatility,
                    'recent_average': recent_avg,
                    'year_ago_average': earlier_avg
                }
        
        return trends
    
    def _get_correlation_insights(self):
        correlations = {}
        
        if 'correlations' in self.models and self.models['correlations']:
            top_corr = self.models['correlations']['top_correlations']
            
            strong_correlations = []
            moderate_correlations = []
            
            for _, row in top_corr.head(10).iterrows():
                corr_data = {
                    'series1': row['series1'],
                    'series2': row['series2'],
                    'correlation': row['correlation'],
                    'relationship': 'positive' if row['correlation'] > 0 else 'negative'
                }
                
                if abs(row['correlation']) > 0.8:
                    strong_correlations.append(corr_data)
                elif abs(row['correlation']) > 0.5:
                    moderate_correlations.append(corr_data)
            
            correlations = {
                'strong': strong_correlations,
                'moderate': moderate_correlations
            }
        
        return correlations
    
    def _get_anomaly_insights(self):
        anomaly_insights = {}
        
        if 'anomalies' in self.models:
            for series_id, anomaly_data in self.models['anomalies'].items():
                recent_anomalies = anomaly_data[anomaly_data['anomaly'] == True].tail(5)
                total_anomalies = sum(anomaly_data['anomaly'])
                anomaly_rate = total_anomalies / len(anomaly_data) * 100
                
                anomaly_insights[series_id] = {
                    'total_anomalies': int(total_anomalies),
                    'anomaly_rate_percent': anomaly_rate,
                    'recent_anomaly_dates': recent_anomalies['ds'].dt.strftime('%Y-%m').tolist() if len(recent_anomalies) > 0 else [],
                    'recent_anomaly_values': recent_anomalies['y'].tolist() if len(recent_anomalies) > 0 else []
                }
        
        return anomaly_insights
    
    def _generate_economic_context(self):
        context = {
            'employment_outlook': self._interpret_employment(),
            'inflation_outlook': self._interpret_inflation(),
            'key_risks': self._identify_risks(),
            'business_implications': self._generate_business_implications()
        }
        
        return context
    
    def _interpret_employment(self):
        employment_context = {}
        
        try:
            if ('forecasts' in self.knowledge_base and 
                'PAYEMS' in self.knowledge_base['forecasts'] and
                'current_data' in self.knowledge_base and
                'PAYEMS' in self.knowledge_base['current_data']):
                
                payems_forecast = self.knowledge_base['forecasts']['PAYEMS']
                current_jobs = self.knowledge_base['current_data']['PAYEMS']['value']
                forecasted_jobs = payems_forecast['next_3_months'][0]
                
                change = ((forecasted_jobs - current_jobs) / current_jobs) * 100
                
                employment_context = {
                    'current_jobs_thousands': current_jobs,
                    'forecasted_change_percent': change,
                    'outlook': 'declining' if change < -0.1 else 'stable' if abs(change) < 0.1 else 'growing'
                }
        except Exception as e:
            print(f"Warning: Could not interpret employment data: {e}")
            employment_context = {'outlook': 'unknown'}
        
        return employment_context
    
    def _interpret_inflation(self):
        inflation_context = {}
        
        try:
            if ('forecasts' in self.knowledge_base and 
                'CPIAUCSL' in self.knowledge_base['forecasts'] and
                'current_data' in self.knowledge_base and
                'CPIAUCSL' in self.knowledge_base['current_data']):
                
                cpi_forecast = self.knowledge_base['forecasts']['CPIAUCSL']
                current_cpi = self.knowledge_base['current_data']['CPIAUCSL']['value']
                forecasted_cpi = cpi_forecast['next_3_months'][0]
                
                inflation_rate = ((forecasted_cpi - current_cpi) / current_cpi) * 100 * 12
                
                inflation_context = {
                    'current_cpi': current_cpi,
                    'forecasted_inflation_rate_annual': inflation_rate,
                    'outlook': 'deflationary' if inflation_rate < -1 else 'low_inflation' if inflation_rate < 2 else 'moderate_inflation' if inflation_rate < 4 else 'high_inflation'
                }
        except Exception as e:
            print(f"Warning: Could not interpret inflation data: {e}")
            inflation_context = {'outlook': 'unknown'}
        
        return inflation_context
    
    def _identify_risks(self):
        risks = []
        
        if 'anomalies' in self.knowledge_base:
            for series_id, anomaly_data in self.knowledge_base['anomalies'].items():
                if anomaly_data['anomaly_rate_percent'] > 15:
                    risks.append(f"High volatility detected in {self.series_metadata.get(series_id, {}).get('name', series_id)}")
        
        if 'trends' in self.knowledge_base:
            for series_id, trend_data in self.knowledge_base['trends'].items():
                if series_id == 'UNRATE' and trend_data['direction'] == 'increasing':
                    risks.append("Rising unemployment trend detected")
                elif series_id == 'CPIAUCSL' and trend_data['magnitude_percent'] > 5:
                    risks.append("Significant inflation changes detected")
        
        return risks
    
    def _generate_business_implications(self):
        implications = []
        
        emp_context = self.knowledge_base.get('economic_context', {}).get('employment_outlook', {})
        if emp_context.get('outlook') == 'declining':
            implications.append("Potential reduction in consumer spending power due to employment decline")
        
        inf_context = self.knowledge_base.get('economic_context', {}).get('inflation_outlook', {})
        if inf_context.get('outlook') == 'deflationary':
            implications.append("Deflationary environment may impact pricing strategies")
        elif inf_context.get('outlook') == 'high_inflation':
            implications.append("High inflation may reduce consumer purchasing power")
        
        return implications
    
    def query(self, question):
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['employment', 'jobs', 'payroll', 'unemployment']):
            return self._answer_employment_question(question)
        elif any(word in question_lower for word in ['inflation', 'cpi', 'price', 'cost']):
            return self._answer_inflation_question(question)
        elif any(word in question_lower for word in ['forecast', 'predict', 'future', 'next']):
            return self._answer_forecast_question(question)
        elif any(word in question_lower for word in ['correlation', 'relationship', 'connected']):
            return self._answer_correlation_question(question)
        elif any(word in question_lower for word in ['anomaly', 'unusual', 'outlier', 'spike']):
            return self._answer_anomaly_question(question)
        elif any(word in question_lower for word in ['trend', 'direction', 'change']):
            return self._answer_trend_question(question)
        else:
            return self._answer_general_question(question)
    
    def _answer_employment_question(self, question):
        current_jobs = self.knowledge_base['current_data']['PAYEMS']['value']
        emp_outlook = self.knowledge_base['economic_context']['employment_outlook']
        
        response = {
            'question': question,
            'answer': f"Current employment is at {current_jobs:,.0f} thousand jobs. " +
                     f"The employment outlook is {emp_outlook.get('outlook', 'stable')} with a " +
                     f"forecasted change of {emp_outlook.get('forecasted_change_percent', 0):.2f}% over the next 3 months.",
            'data_sources': ['PAYEMS'],
            'key_metrics': {
                'current_jobs_thousands': current_jobs,
                'forecast_change_percent': emp_outlook.get('forecasted_change_percent', 0)
            }
        }
        
        return response
    
    def _answer_inflation_question(self, question):
        current_cpi = self.knowledge_base['current_data']['CPIAUCSL']['value']
        inf_outlook = self.knowledge_base['economic_context']['inflation_outlook']
        
        response = {
            'question': question,
            'answer': f"Current Consumer Price Index is {current_cpi:.2f}. " +
                     f"The inflation outlook is {inf_outlook.get('outlook', 'moderate')} with an " +
                     f"annualized inflation rate forecast of {inf_outlook.get('forecasted_inflation_rate_annual', 0):.2f}%.",
            'data_sources': ['CPIAUCSL'],
            'key_metrics': {
                'current_cpi': current_cpi,
                'forecasted_inflation_annual': inf_outlook.get('forecasted_inflation_rate_annual', 0)
            }
        }
        
        return response
    
    def _answer_forecast_question(self, question):
        forecasts = self.knowledge_base['forecasts']
        
        forecast_summary = []
        for series_id, forecast_data in forecasts.items():
            series_name = self.series_metadata.get(series_id, {}).get('name', series_id)
            next_value = forecast_data['next_3_months'][0]
            forecast_summary.append(f"{series_name}: {next_value:.2f}")
        
        response = {
            'question': question,
            'answer': f"Here are the 3-month forecasts: {'; '.join(forecast_summary)}",
            'data_sources': list(forecasts.keys()),
            'forecasts': forecasts
        }
        
        return response
    
    def _answer_correlation_question(self, question):
        correlations = self.knowledge_base['correlations']
        
        strong_corr_text = []
        for corr in correlations.get('strong', [])[:3]:
            strong_corr_text.append(f"{corr['series1']} and {corr['series2']} ({corr['correlation']:.3f})")
        
        response = {
            'question': question,
            'answer': f"Strongest correlations found: {'; '.join(strong_corr_text)}",
            'data_sources': ['correlation_analysis'],
            'correlations': correlations
        }
        
        return response
    
    def _answer_anomaly_question(self, question):
        anomalies = self.knowledge_base['anomalies']
        
        anomaly_summary = []
        for series_id, anomaly_data in anomalies.items():
            series_name = self.series_metadata.get(series_id, {}).get('name', series_id)
            rate = anomaly_data['anomaly_rate_percent']
            anomaly_summary.append(f"{series_name}: {rate:.1f}% anomaly rate")
        
        response = {
            'question': question,
            'answer': f"Anomaly detection results: {'; '.join(anomaly_summary)}",
            'data_sources': ['anomaly_detection'],
            'anomalies': anomalies
        }
        
        return response
    
    def _answer_trend_question(self, question):
        trends = self.knowledge_base['trends']
        
        trend_summary = []
        for series_id, trend_data in trends.items():
            series_name = self.series_metadata.get(series_id, {}).get('name', series_id)
            direction = trend_data['direction']
            magnitude = trend_data['magnitude_percent']
            trend_summary.append(f"{series_name}: {direction} ({magnitude:.1f}% change)")
        
        response = {
            'question': question,
            'answer': f"Current trends: {'; '.join(trend_summary)}",
            'data_sources': ['trend_analysis'],
            'trends': trends
        }
        
        return response
    
    def _answer_general_question(self, question):
        response = {
            'question': question,
            'answer': "I can help you analyze employment, inflation, forecasts, correlations, anomalies, and trends in economic data. " +
                     "Try asking about specific topics like 'What is the employment forecast?' or 'Are there any anomalies in inflation data?'",
            'data_sources': ['general'],
            'suggestions': [
                "What is the current employment situation?",
                "How is inflation trending?",
                "What are the 3-month forecasts?",
                "Which economic indicators are most correlated?",
                "Are there any unusual patterns in the data?"
            ]
        }
        
        return response

def run_rag_demo():
    load_dotenv()
    
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
    
    rag = EconomicRAGSystem(conn)
    rag.load_models()
    rag.build_knowledge_base()
    
    print("\n🤖 Economic Intelligence RAG System Ready!")
    print("Ask me questions about economic data, forecasts, trends, and correlations.")
    print("Type 'quit' to exit.\n")
    
    sample_questions = [
        "What is the current employment situation?",
        "How is inflation trending?",
        "What are the 3-month forecasts?",
        "Which economic indicators are strongly correlated?",
        "Are there any anomalies in the unemployment data?"
    ]
    
    while True:
        question = input("\n🔍 Your question: ").strip()
        
        if question.lower() in ['quit', 'exit', 'q']:
            break
        
        if question.lower() == 'help':
            print("\nSample questions you can ask:")
            for i, q in enumerate(sample_questions, 1):
                print(f"{i}. {q}")
            continue
        
        if not question:
            continue
        
        try:
            response = rag.query(question)
            print(f"\n💡 Answer: {response['answer']}")
            
            if 'key_metrics' in response:
                print(f"📊 Key Metrics: {response['key_metrics']}")
                
        except Exception as e:
            print(f"❌ Error processing question: {e}")
    
    conn.close()
    print("\n👋 Thanks for using the Economic Intelligence RAG System!")

if __name__ == "__main__":
    run_rag_demo()