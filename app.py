import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet.plot import plot_plotly
from prophet import Prophet
import os

st.title("Canada Work Permits Forecast")

# Load combined forecast
forecast_file = "forecast.csv"
forecast = pd.read_csv(forecast_file)

# Select province
province = st.selectbox("Select a province", forecast['Province'].unique())

df_prov = forecast[forecast['Province'] == province]

# Plot forecast
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(df_prov['ds'], df_prov['yhat'], color='blue', label='Predicted (yhat)')
ax.fill_between(df_prov['ds'], df_prov['yhat_lower'], df_prov['yhat_upper'], 
                color='lightblue', alpha=0.4, label='Confidence interval')
ax.scatter(df_prov['ds'], df_prov['yhat'], color='black', s=10, label='Forecast points')

ax.set_title(f"Forecast of Total Work Permits in {province}")
ax.set_xlabel("Date")
ax.set_ylabel("Total Permits")
ax.legend()
st.pyplot(fig)
