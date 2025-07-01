# Time Series Forecasting Strategy & Model Selection

## Business Objective

Provide 3-6 month forecasts for consumer spending categories to enable:
- **Inventory Planning**: Anticipate demand shifts across product categories
- **Marketing Budget Allocation**: Time campaigns with spending trend predictions
- **Risk Management**: Early warning system for economic downturns
- **Strategic Planning**: Data-driven investment decisions

## Data Characteristics & Challenges

### Economic Time Series Properties
- **Seasonality**: Consumer spending has strong seasonal patterns (holidays, back-to-school)
- **Trend**: Long-term growth patterns influenced by population and economic growth
- **Volatility**: Economic shocks (COVID, recessions) create structural breaks
- **Correlation**: Categories influence each other (unemployment affects all spending)
- **Frequency**: Monthly data provides good signal without overwhelming noise

### Dataset Constraints
- **Limited History**: 7 years of data (2017-2024) 
- **External Shocks**: COVID period (2020-2021) creates unusual patterns
- **Sample Size**: ~84 observations per series (monthly data)
- **Missing Values**: Some FRED series have data gaps

## Model Selection Strategy

### Primary Models (Implementation Priority)

**1. Prophet (Meta's Forecasting Tool)**
- **Why**: Handles seasonality and holidays automatically
- **Strengths**: Robust to missing data, interpretable components
- **Business Value**: Can incorporate known events (Black Friday, tax season)
- **Training Speed**: Fast on single CPU, perfect for local GPU setup
- **Use Case**: Primary forecasting engine for all spending categories

**2. ARIMA/SARIMA (Classical Time Series)**
- **Why**: Industry standard baseline, well-understood by business stakeholders
- **Strengths**: Good for stationary series, established statistical foundation
- **Business Value**: Provides confidence intervals and statistical significance
- **Training Speed**: Very fast, good for quick iterations
- **Use Case**: Benchmark model and backup for Prophet

**3. Exponential Smoothing (Holt-Winters)**
- **Why**: Simple, interpretable trend and seasonal decomposition
- **Strengths**: Handles level, trend, and seasonality explicitly
- **Business Value**: Easy to explain to non-technical stakeholders
- **Training Speed**: Extremely fast
- **Use Case**: Baseline model and sanity check for complex models

### Model Selection Rationale

**Prophet as Primary Choice**:
- **Seasonal Handling**: Consumer spending is highly seasonal - Prophet excels here
- **Holiday Effects**: Can model Black Friday, Christmas, tax refund seasons
- **Robustness**: Handles COVID-period outliers better than classical methods
- **Interpretability**: Decomposed trend, seasonal, and holiday components
- **Business Integration**: Easy to add business knowledge (promotional calendars)

**Classical Models as Benchmarks**:
- **Statistical Rigor**: ARIMA provides formal statistical testing
- **Simplicity**: Exponential smoothing is easiest to explain and tune
- **Validation**: Multiple model comparison increases forecast confidence
- **Fallback**: If Prophet fails on any series, classical methods provide backup

## Training Strategy

### Local GPU Implementation
- **Hardware**: RTX 4080 Super for Prophet hyperparameter tuning
- **Cost Optimization**: Avoid expensive cloud ML compute ($50-200/hour)
- **Flexibility**: Experiment with model variants without usage limits
- **Speed**: Parallel training across multiple series

### Cross-Validation Approach
- **Time Series CV**: Walk-forward validation respecting temporal order
- **Horizon**: Test 1, 3, and 6-month forecast accuracy
- **Metrics**: MAPE (Mean Absolute Percentage Error) for business interpretability
- **Validation Period**: Use last 12 months for testing, train on earlier data

### Model Configuration

**Prophet Parameters**:
```python
yearly_seasonality=True     # Capture annual patterns
weekly_seasonality=False    # Monthly data doesn't need weekly
daily_seasonality=False     # Not relevant for monthly data
seasonality_mode='additive' # Start simple, test multiplicative
changepoint_prior_scale=0.05 # Conservative - avoid overfitting
holidays=us_holidays       # Include major shopping periods
```

**ARIMA Selection**:
- Auto-ARIMA for automated (p,d,q) selection
- Seasonal components (P,D,Q) for 12-month cycles
- AIC/BIC criteria for model selection

## Feature Engineering

### Derived Metrics
- **Real vs Nominal**: Inflation-adjusted spending using CPI
- **Growth Rates**: Month-over-month and year-over-year changes
- **Moving Averages**: 3-month and 12-month rolling averages
- **Seasonal Indices**: Deviation from trend for each month

### External Regressors (Advanced)
- **Economic Context**: Use unemployment rate as external regressor
- **Cross-Category**: Durable goods spending as predictor for services
- **Leading Indicators**: Employment data to predict spending with lag

## Forecast Horizon Strategy

### 3-Month Primary Horizon
- **Business Planning**: Quarterly budget cycles
- **Accuracy**: Most reliable forecast window for economic data
- **Actionability**: Sufficient lead time for operational decisions

### 6-Month Extended Horizon
- **Strategic Planning**: Semi-annual business reviews
- **Uncertainty Bands**: Wider confidence intervals for longer horizon
- **Scenario Analysis**: Best/worst case planning scenarios

## Model Evaluation Framework

### Accuracy Metrics
- **MAPE**: Primary metric - percentage error easy for business interpretation
- **RMSE**: Root mean squared error for penalty on large misses
- **MAE**: Mean absolute error for robust performance assessment

### Business Validation
- **Directional Accuracy**: Did we predict the right trend direction?
- **Magnitude Assessment**: Are forecast magnitudes reasonable?
- **Seasonal Validity**: Do seasonal patterns match business knowledge?

## Risk Management & Limitations

### Model Limitations
- **Structural Breaks**: Major economic shocks not in training data
- **Small Sample**: Limited history may miss long-term cycles
- **Correlation Assumptions**: Economic relationships may change over time

### Mitigation Strategies
- **Ensemble Forecasting**: Combine multiple model predictions
- **Confidence Intervals**: Provide uncertainty ranges, not point estimates
- **Regular Retraining**: Monthly model updates with new data
- **Human Oversight**: Business expert review of all forecasts

### Business Communication
- **Uncertainty Communication**: Always present ranges, not single numbers
- **Model Limitations**: Clearly communicate what models can/cannot predict
- **Update Frequency**: Set expectations for forecast refresh cycles

## Implementation Priorities

### Phase 1 (MVP - Day 2)
- Prophet models for 4 core spending categories
- 3-month forecasts with confidence intervals
- Basic accuracy evaluation

### Phase 2 (Enhancement - Future)
- ARIMA benchmarks for all categories
- Cross-category correlation modeling
- External regressor integration
- Automated model selection

### Phase 3 (Advanced - Future)
- Ensemble methods combining multiple approaches
- Real-time model performance monitoring
- Automated retraining pipelines
- Advanced uncertainty quantification

## Success Criteria

### Technical Metrics
- **MAPE < 15%**: Industry-standard accuracy for economic forecasting
- **Directional Accuracy > 70%**: Correctly predict spending direction
- **Seasonal R² > 0.8**: Capture seasonal patterns effectively

### Business Value
- **Actionable Insights**: Forecasts lead to specific business decisions
- **Stakeholder Adoption**: Business teams regularly use predictions
- **Decision Support**: Models influence quarterly planning processes

This approach balances **statistical rigor** with **practical business value**, leveraging proven methods while staying within resource constraints.