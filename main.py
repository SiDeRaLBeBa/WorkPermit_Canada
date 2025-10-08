import pandas as pd
from prophet import Prophet
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


# Step 1: Load CSV

df = pd.read_csv("ODP-TR-Work-IMP-PT_NOC4.csv", sep='\t')


# Step 2: Clean TOTAL column

df['TOTAL'] = pd.to_numeric(df['TOTAL'].replace('--', pd.NA))
# Fill NaN with 0
df['TOTAL'] = df['TOTAL'].fillna(0)


# Step 3: Create datetime column

df['DATE'] = pd.to_datetime(df['EN_YEAR'].astype(str) + '-' + df['EN_MONTH'].astype(str) + '-01')


# Step 4: Aggregate total permits per month

monthly_totals = df.groupby('DATE')['TOTAL'].sum().reset_index()


# Step 5: Prepare data for Prophet

data = monthly_totals.rename(columns={'DATE': 'ds', 'TOTAL': 'y'})


# Step 6: Fit Prophet model

model = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
model.fit(data)


# Step 7: Forecast next 36 months

future = model.make_future_dataframe(periods=36, freq='M')
forecast = model.predict(future)


# Step 8: Save forecast to CSV

forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].to_csv("forecast.csv", index=False)


# Step 9: Plot forecast with legend

fig = model.plot(forecast)

# Legend elements

legend_elements = [
    Line2D([0], [0], color='black', marker='o', linestyle='None', label='Actual data'),
    Line2D([0], [0], color='blue', lw=2, label='Predicted (yhat)'),
    Patch(facecolor='lightblue', alpha=0.4, label='Confidence interval')
]

plt.legend(handles=legend_elements)
plt.title("Forecast of Total Work Permits in Canada")
plt.xlabel("Date")
plt.ylabel("Total Permits")
plt.savefig("forecast_plot.png")
plt.show()