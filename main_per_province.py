import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import os
import re


# Create folder to save plots
os.makedirs("province_forecasts", exist_ok=True)

# Step 1: Load CSV
df = pd.read_csv("ODP-TR-Work-IMP-PT_NOC4.csv", sep='\t')

# Step 2: Clean TOTAL column
df['TOTAL'] = pd.to_numeric(df['TOTAL'].replace('--', pd.NA)).fillna(0)

# Step 3: Create datetime column
df['DATE'] = pd.to_datetime(df['EN_YEAR'].astype(str) + '-' + df['EN_MONTH'].astype(str) + '-01')

# Step 4: Get list of provinces
provinces = df['EN_PROVINCE_TERRITORY'].unique()

# Step 5: Loop through each province
for province in provinces:
    print(f"Forecasting for {province}...")
    df_prov = df[df['EN_PROVINCE_TERRITORY'] == province]
    
    # Aggregate monthly totals
    monthly_totals = df_prov.groupby('DATE')['TOTAL'].sum().reset_index()
    data = monthly_totals.rename(columns={'DATE': 'ds', 'TOTAL': 'y'})
    
    # Fit Prophet
    model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    model.fit(data)
    
    # Forecast
    future = model.make_future_dataframe(periods=36, freq='M')
    forecast = model.predict(future)
    
    # Clean province name for filename
    safe_province = re.sub(r'[\\/:"*?<>|]', '_', province)
    
    # Save CSV
    forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv(f"province_forecasts/forecast_{safe_province}.csv", index=False)
    
    # Plot forecast
    fig = model.plot(forecast)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], color='black', marker='o', linestyle='None', label='Actual data'),
        Line2D([0], [0], color='blue', lw=2, label='Predicted (yhat)'),
        Patch(facecolor='lightblue', alpha=0.4, label='Confidence interval')
    ]
    plt.legend(handles=legend_elements)
    plt.title(f"Forecast of Total Work Permits in {province}")
    plt.xlabel("Date")
    plt.ylabel("Total Permits")
    
    # Save plot
    plt.savefig(f"province_forecasts/forecast_{safe_province}.png")
    plt.close()

print("All province forecasts completed! Check the 'province_forecasts' folder.")
