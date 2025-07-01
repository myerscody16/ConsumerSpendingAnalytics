import snowflake.connector
import pandas as pd
import os
from datetime import datetime
from dotenv import load_dotenv

class SnowflakeLoader:
    def __init__(self):
        load_dotenv()
        
        # Get MFA token from user input
        mfa_token = input("Enter your MFA code from authenticator app: ")
        
        self.conn = snowflake.connector.connect(
            user=os.getenv('SNOWFLAKE_USER'),
            password=os.getenv('SNOWFLAKE_PASSWORD'),
            account=os.getenv('SNOWFLAKE_ACCOUNT'),
            passcode=mfa_token,
            warehouse='COMPUTE_WH',
            database='CONSUMER_SPENDING_DB',
            schema='ANALYTICS'
        )
        print("Connected to Snowflake!")
        
    def load_csv_to_staging(self, csv_file_path):
        cursor = self.conn.cursor()
        
        # Clear staging table
        cursor.execute("TRUNCATE TABLE STAGE_FRED_RAW")
        
        # Use Snowflake's PUT command to upload file to internal stage
        put_sql = f"PUT file://{csv_file_path} @%STAGE_FRED_RAW"
        cursor.execute(put_sql)
        
        print(f"File uploaded to Snowflake stage")
        
        # Use COPY INTO to load data from stage to table
        copy_sql = """
        COPY INTO STAGE_FRED_RAW (DATE_COL, VALUE_COL, SERIES_ID, SERIES_NAME)
        FROM @%STAGE_FRED_RAW
        FILE_FORMAT = (
            TYPE = 'CSV'
            FIELD_DELIMITER = ','
            SKIP_HEADER = 1
            FIELD_OPTIONALLY_ENCLOSED_BY = '"'
            NULL_IF = ('NULL', 'null', '')
            EMPTY_FIELD_AS_NULL = TRUE
        )
        ON_ERROR = 'CONTINUE'
        """
        
        result = cursor.execute(copy_sql)
        
        # Get load results
        load_result = cursor.fetchall()
        print(f"Load results: {load_result}")
        
        self.conn.commit()
        print("Data successfully loaded to staging table")
        
    def stage_to_fact(self):
        cursor = self.conn.cursor()
        
        print("Moving data from staging to fact table...")
        
        merge_sql = """
            MERGE INTO FACT_ECONOMIC_DATA f
            USING (
                SELECT 
                    DATE_COL as DATE_KEY,
                    SERIES_ID as CATEGORY_KEY,
                    VALUE_COL as RAW_VALUE,
                    SERIES_ID
                FROM STAGE_FRED_RAW
                WHERE VALUE_COL IS NOT NULL
            ) s
            ON f.DATE_KEY = s.DATE_KEY AND f.CATEGORY_KEY = s.CATEGORY_KEY
            WHEN MATCHED THEN 
                UPDATE SET 
                    RAW_VALUE = s.RAW_VALUE,
                    LOAD_TIMESTAMP = CURRENT_TIMESTAMP()
            WHEN NOT MATCHED THEN
                INSERT (DATE_KEY, CATEGORY_KEY, RAW_VALUE, SERIES_ID)
                VALUES (s.DATE_KEY, s.CATEGORY_KEY, s.RAW_VALUE, s.SERIES_ID)
        """
        
        cursor.execute(merge_sql)
        self.conn.commit()
        print("Data successfully moved to fact table")
        
    def get_data_summary(self):
        cursor = self.conn.cursor()
        
        cursor.execute("""
            SELECT 
                c.CATEGORY_NAME,
                COUNT(*) as RECORD_COUNT,
                MIN(f.DATE_KEY) as EARLIEST_DATE,
                MAX(f.DATE_KEY) as LATEST_DATE
            FROM FACT_ECONOMIC_DATA f
            JOIN DIM_CATEGORY c ON f.CATEGORY_KEY = c.CATEGORY_KEY
            GROUP BY c.CATEGORY_NAME
            ORDER BY c.CATEGORY_NAME
        """)
        
        results = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]
        df = pd.DataFrame(results, columns=columns)
        return df
    
    def close(self):
        self.conn.close()

def load_latest_fred_data():
    # Find the most recent FRED data file
    data_dir = 'data/raw'
    fred_files = [f for f in os.listdir(data_dir) if f.startswith('fred_data_')]
    
    if not fred_files:
        print("No FRED data files found. Run data_ingestion.py first.")
        return
        
    latest_file = max(fred_files)
    csv_path = os.path.abspath(os.path.join(data_dir, latest_file))  # Use absolute path
    
    print(f"Loading data from {latest_file}")
    print(f"Full path: {csv_path}")
    
    loader = SnowflakeLoader()
    loader.load_csv_to_staging(csv_path)
    loader.stage_to_fact()
    
    summary = loader.get_data_summary()
    print("\nData Summary:")
    print(summary)
    
    loader.close()

if __name__ == "__main__":
    load_latest_fred_data()