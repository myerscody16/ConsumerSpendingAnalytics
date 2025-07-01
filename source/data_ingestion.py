import requests
import pandas as pd
import yaml
import os
from datetime import datetime, timedelta

class FREDDataClient:
    def __init__(self, api_key):
        self.api_key = api_key
        self.base_url = "https://api.stlouisfed.org/fred"
        
    def get_series_data(self, series_id, start_date=None):
        if start_date is None:
            start_date = (datetime.now() - timedelta(days=2555)).strftime('%Y-%m-%d')  # 7 years
            
        url = f"{self.base_url}/series/observations"
        params = {
            'series_id': series_id,
            'api_key': self.api_key,
            'file_type': 'json',
            'start_date': start_date
        }
        
        response = requests.get(url, params=params)
        
        if response.status_code != 200:
            print(f"API Error for {series_id}:")
            print(f"Status Code: {response.status_code}")
            print(f"Response: {response.text}")
            return pd.DataFrame()
            
        data = response.json()
        observations = data.get('observations', [])
        
        if not observations:
            print(f"No data returned for {series_id}")
            return pd.DataFrame()
        
        df = pd.DataFrame(observations)
        print(f"Columns for {series_id}: {df.columns.tolist()}")
        print(f"First few rows: {df.head()}")
        
        df['date'] = pd.to_datetime(df['date'])
        df['value'] = pd.to_numeric(df['value'], errors='coerce')
        df = df.dropna(subset=['value'])
        df['series_id'] = series_id
        
        return df[['date', 'value', 'series_id']]

def load_fred_config():
    with open('config/fred_series.yml', 'r') as f:
        return yaml.safe_load(f)

def fetch_all_series():
    api_key = os.getenv('FRED_API_KEY')
    print(f"API Key loaded: {'Yes' if api_key else 'No'}")
    print(f"API Key starts with: {api_key[:10] if api_key else 'None'}...")
    
    if not api_key:
        print("ERROR: FRED_API_KEY not found in environment variables")
        return pd.DataFrame()
    
    client = FREDDataClient(api_key)
    config = load_fred_config()
    
    all_data = []
    for series_id, description in config['series'].items():
        print(f"Fetching {series_id}: {description}")
        df = client.get_series_data(series_id)
        df['series_name'] = description
        all_data.append(df)
    
    combined_df = pd.concat(all_data, ignore_index=True)
    
    # Save to CSV
    os.makedirs('data/raw', exist_ok=True)
    filename = f"data/raw/fred_data_{datetime.now().strftime('%Y%m%d')}.csv"
    combined_df.to_csv(filename, index=False)
    print(f"Saved data to {filename}")
    
    return combined_df

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    df = fetch_all_series()
    print(f"Total records fetched: {len(df)}")
    print(df.head())