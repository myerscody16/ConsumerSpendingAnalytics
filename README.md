# Consumer Spending Analytics

I built this to see if I could create an end-to-end economic intelligence system that actually provides useful business insights. Turns out you can pull real economic data, forecast it with decent accuracy, and make it queryable through natural language.

## What it does

This system pulls economic data from the Federal Reserve, stores it in Snowflake, runs ML forecasting models, and lets you ask questions about the economy in plain English. The whole thing runs on the free tier of Snowflake and I used a local GPU for model training to keep costs reasonable.

The core insight was realizing that consumer spending patterns are highly predictable if you look at the right indicators. Employment, inflation, and income data tell you most of what you need to know about where spending is headed.

## Why I built this

Consumer goods companies make billion-dollar decisions based on economic forecasts, but most economic analysis is either too academic or too expensive. I wanted to see if you could build something practical that combines real-time data, solid forecasting, and a business-friendly interface.

The goal was demonstrating full-stack data science capabilities while solving an actual business problem. Plus, economic data is fascinating once you start seeing the patterns.

## How it works

The pipeline is straightforward: FRED API → Snowflake → ML models → conversational interface.

I grab 7 years of monthly data for key economic indicators like employment, inflation, retail sales, and income. The forecasting uses Prophet and ARIMA models in an ensemble approach because combining models usually beats individual ones. Prophet handles seasonality well, ARIMA provides a statistical baseline.

For anomaly detection, I used Isolation Forest to catch unusual economic patterns. Cross-correlation analysis reveals which indicators move together, which is useful for understanding economic relationships.

The RAG system lets you ask questions like "How is inflation trending?" or "What's the employment forecast?" and get actual data-driven answers instead of generic responses.

## Key results

The models are predicting some interesting economic shifts over the next few months:

Employment is expected to decline slightly from 159.6K to 157.5K jobs, but unemployment should improve from 4.2% to 4.13%. That divergence suggests economic transition rather than recession.

Inflation looks like it's cooling significantly - CPI forecast to drop from 320.6 to 310.9, which implies near-zero or negative inflation rates. That's a big deal for pricing strategies.

Real income is projected to grow modestly from $17,806 to $17,949 billion, which should support consumer spending despite employment headwinds.

## Technical approach

I kept the stack simple but powerful. Python for everything, Snowflake for storage, local GPU for model training. The dimensional data model makes queries fast and the bulk loading approach handles thousands of data points efficiently.

The ensemble forecasting combines Prophet (60% weight) with ARIMA (40% weight) because Prophet is better at capturing seasonal patterns while ARIMA provides statistical rigor. Confidence intervals are crucial for business decision-making.

Anomaly detection found about 10% outliers across all series, which is normal for economic data. These likely capture events like COVID, financial crises, and seasonal volatility.

The strongest correlation I found was between durable goods and retail sales (0.998), which makes sense - big purchases drive overall retail performance.

## Getting started

You need a FRED API key (free) and Snowflake account (30-day trial). The setup is mostly running the scripts in order:

1. `data_ingestion.py` - pulls economic data from FRED
2. `data_loading.py` - bulk loads to Snowflake  
3. `advanced_forecasting.py` - trains ensemble models
4. `rag_system.py` - starts the conversational interface

The whole process takes maybe an hour to set up and a few hours for model training. Most of the complexity is in the ML pipeline, but I tried to keep it as straightforward as possible.

## What I learned

Economic data is surprisingly clean and well-structured compared to typical business data. The Federal Reserve APIs are solid and the data quality is excellent.

Local GPU training is way more cost-effective than cloud ML services for this scale. My RTX 4080 Super handled ensemble training for multiple series without breaking a sweat.

The RAG approach works well for economic queries because the domain is constrained and the relationships are well-understood. You can build effective business intelligence without massive language models.

Ensemble methods consistently outperform individual models for time series forecasting. The combination of Prophet's seasonality handling and ARIMA's statistical foundation creates robust predictions.

## Business impact

This type of system could easily support quarterly planning for consumer goods companies. The forecasts are accurate enough for inventory planning, the correlation analysis reveals market dynamics, and the anomaly detection provides early warning signals.

The natural language interface makes economic insights accessible to non-technical stakeholders. Instead of reading statistical reports, executives can just ask "What's driving inflation?" and get data-backed answers.

Cost optimization was key - the whole system runs on free tiers plus minimal Snowflake usage. That's important for demonstrating practical value rather than just technical capability.

## Files

```
source/data_ingestion.py     - FRED API client
source/data_loading.py       - Snowflake bulk loader
source/forecasting.py        - Basic Prophet/ARIMA models  
source/advanced_forecasting.py - Ensemble + anomaly detection
source/rag_system.py         - Conversational interface
config/fred_series.yml       - Economic indicators to track
sql/create_tables.sql        - Snowflake schema
```

The code is intentionally minimal - no type annotations, no excessive comments, just clean implementations that work.

## Next steps

The foundation is solid enough to extend in several directions. Real-time monitoring, more sophisticated ensemble methods, regional analysis, or integration with business planning tools.

The RAG system could be enhanced with proper vector search and larger language models. The current keyword-based routing works but semantic understanding would be better.

For production deployment, you'd want automated model retraining, performance monitoring, and proper error handling. But for demonstrating capabilities, this hits the right balance of sophistication and simplicity.