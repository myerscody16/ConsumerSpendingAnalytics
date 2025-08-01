-- Create database and schema
CREATE DATABASE IF NOT EXISTS CONSUMER_SPENDING_DB;
USE DATABASE CONSUMER_SPENDING_DB;
CREATE SCHEMA IF NOT EXISTS ANALYTICS;
USE SCHEMA ANALYTICS;

-- Date dimension table
CREATE OR REPLACE TABLE DIM_DATE (
    DATE_KEY DATE PRIMARY KEY,
    YEAR INT,
    MONTH INT,
    QUARTER INT,
    MONTH_NAME STRING,
    QUARTER_NAME STRING
);

-- Category dimension table  
CREATE OR REPLACE TABLE DIM_CATEGORY (
    CATEGORY_KEY STRING PRIMARY KEY,
    CATEGORY_NAME STRING,
    FRED_SERIES_ID STRING,
    CATEGORY_TYPE STRING  -- 'CORE_SPENDING', 'RETAIL_CHANNEL', 'ECONOMIC_CONTEXT'
);

-- Main fact table for economic data
CREATE OR REPLACE TABLE FACT_ECONOMIC_DATA (
    DATE_KEY DATE,
    CATEGORY_KEY STRING,
    RAW_VALUE FLOAT,
    SERIES_ID STRING,
    LOAD_TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP(),
    PRIMARY KEY (DATE_KEY, CATEGORY_KEY)
);

-- Staging table for raw FRED data loads
CREATE OR REPLACE TABLE STAGE_FRED_RAW (
    DATE_COL DATE,
    VALUE_COL FLOAT,
    SERIES_ID STRING,
    SERIES_NAME STRING,
    LOAD_TIMESTAMP TIMESTAMP DEFAULT CURRENT_TIMESTAMP()
);

-- Insert date dimension data (last 10 years)
INSERT INTO DIM_DATE
SELECT 
    date_val as DATE_KEY,
    YEAR(date_val) as YEAR,
    MONTH(date_val) as MONTH,
    QUARTER(date_val) as QUARTER,
    MONTHNAME(date_val) as MONTH_NAME,
    'Q' || QUARTER(date_val) as QUARTER_NAME
FROM (
    SELECT DATEADD(day, seq4(), '2015-01-01') as date_val
    FROM TABLE(GENERATOR(rowcount => 3650))  -- 10 years of days
)
WHERE date_val <= CURRENT_DATE();

-- Insert category dimension data
INSERT INTO DIM_CATEGORY VALUES
('PCEC96', 'Personal Consumption Expenditures', 'PCEC96', 'CORE_SPENDING'),
('DGDSRC1A027NBEA', 'Durable Goods Spending', 'DGDSRC1A027NBEA', 'CORE_SPENDING'),
('NDGSRC1A027NBEA', 'Nondurable Goods Spending', 'NDGSRC1A027NBEA', 'CORE_SPENDING'),
('PCESVC96', 'Services Spending', 'PCESVC96', 'CORE_SPENDING'),
('RSAFS', 'Retail Sales', 'RSAFS', 'RETAIL_CHANNEL'),
('ECOMSA', 'E-commerce Sales', 'ECOMSA', 'RETAIL_CHANNEL'),
('RSFSDP', 'Restaurant Sales', 'RSFSDP', 'RETAIL_CHANNEL'),
('MVLOAS', 'Motor Vehicle Sales', 'MVLOAS', 'RETAIL_CHANNEL'),
('CPIAUCSL', 'Consumer Price Index', 'CPIAUCSL', 'ECONOMIC_CONTEXT'),
('UNRATE', 'Unemployment Rate', 'UNRATE', 'ECONOMIC_CONTEXT'),
('PAYEMS', 'Total Nonfarm Payrolls', 'PAYEMS', 'ECONOMIC_CONTEXT'),
('DSPIC96', 'Real Disposable Personal Income', 'DSPIC96', 'ECONOMIC_CONTEXT');