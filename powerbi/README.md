# Power BI Dashboard Guide

Use `scripts/export_powerbi.py` to convert RetailPulse outputs into CSV files that Power BI can import.

## Export Data

```bash
cd /Users/onlymec/RetailPulse
source .venv/bin/activate
python scripts/export_powerbi.py
```

The CSV files are written to:

```text
powerbi_exports/
```

## Recommended Power BI Tables

- `transactions.csv` – main transaction fact table
- `daily_sales.csv` – daily sales trend table
- `customer_summary.csv` – customer-level churn/value table
- `customer_segments.csv` – RFM + KMeans/DBSCAN segmentation table
- `inventory_recommendations.csv` – SKU reorder recommendations
- `forecast_30_days.csv` – 30-day demand forecast
- `model_metrics.csv` – MAPE, MAE, ROC-AUC, precision
- `country_sales.csv` – country revenue summary
- `product_category_sales.csv` – product category revenue summary

## Suggested Report Pages

1. Executive Overview
2. Sales Analytics
3. Demand Forecasting
4. Customer Segmentation
5. Churn Prediction
6. Inventory Optimization
7. Model Metrics

## Useful Measures

```DAX
Total Sales = SUM(transactions[TotalAmount])
Total Quantity = SUM(transactions[Quantity])
Orders = DISTINCTCOUNT(transactions[Invoice])
Customers = DISTINCTCOUNT(transactions[Customer ID])
AOV = DIVIDE([Total Sales], [Orders])
Churn Rate = AVERAGE(customer_summary[ChurnFlag])
Forecast MAPE = CALCULATE(MAX(model_metrics[Value]), model_metrics[Metric] = "MAPE")
Churn ROC AUC = CALCULATE(MAX(model_metrics[Value]), model_metrics[Metric] = "ROC-AUC")
```

