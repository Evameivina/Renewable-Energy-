# dashboard.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="Renewable Energy Dashboard", layout="wide")
st.title("Interactive Renewable Energy Dashboard (2000-2023)")

# ---------------------------
# Load Data
# ---------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/Evameivina/Renewable-Energy-/refs/heads/main/global_renewable_energy_production.csv"
    df = pd.read_csv(url)
    df['Year'] = df['Year'].astype(int)
    energy_cols = ['SolarEnergy','WindEnergy','HydroEnergy','OtherRenewableEnergy','TotalRenewableEnergy']
    df[energy_cols] = df[energy_cols].astype(float)
    df.fillna(0, inplace=True)
    
    # Perbaiki TotalEnergy
    df['CheckTotal'] = df[['SolarEnergy','WindEnergy','HydroEnergy','OtherRenewableEnergy']].sum(axis=1)
    inconsistent = df[df['CheckTotal'] != df['TotalRenewableEnergy']]
    df.loc[inconsistent.index, 'TotalRenewableEnergy'] = df.loc[inconsistent.index, ['SolarEnergy','WindEnergy','HydroEnergy','OtherRenewableEnergy']].sum(axis=1)
    return df

df = load_data()
energy_cols = ['SolarEnergy','WindEnergy','HydroEnergy','OtherRenewableEnergy','TotalRenewableEnergy']

# ---------------------------
# Sidebar - Filter Interaktif
# ---------------------------
st.sidebar.header("Filter Options")
selected_countries = st.sidebar.multiselect(
    "Select Countries",
    df['Country'].unique(),
    default=['USA','China']
)
selected_energy = st.sidebar.selectbox(
    "Select Energy Type",
    energy_cols
)
year_range = st.sidebar.slider(
    "Select Year Range",
    int(df['Year'].min()),
    int(df['Year'].max()),
    (2000, 2023)
)

filtered_df = df[
    (df['Country'].isin(selected_countries)) &
    (df['Year'] >= year_range[0]) &
    (df['Year'] <= year_range[1])
]

# ---------------------------
# Dataset & Statistik
# ---------------------------
st.subheader("Filtered Dataset")
st.dataframe(filtered_df.reset_index(drop=True))

st.subheader("Descriptive Statistics")
st.write(filtered_df[energy_cols].describe())

# ---------------------------
# Visualization - Trend Line
# ---------------------------
st.subheader(f"{selected_energy} Trend")
fig, ax = plt.subplots(figsize=(12,6))
for country in selected_countries:
    country_data = filtered_df[filtered_df['Country']==country]
    sns.lineplot(x='Year', y=selected_energy, data=country_data, marker='o', label=country, ax=ax)
ax.set_ylabel("Energy Production (GWh)")
ax.set_title(f"{selected_energy} Production ({year_range[0]}-{year_range[1]})")
ax.grid(True)
ax.legend()
st.pyplot(fig)

# ---------------------------
# Visualization - Total Energy Comparison
# ---------------------------
st.subheader(f"Total {selected_energy} per Country ({year_range[0]}-{year_range[1]})")
total_energy = filtered_df.groupby('Country')[selected_energy].sum().sort_values(ascending=False)
fig2, ax2 = plt.subplots(figsize=(12,5))
total_energy.plot(kind='bar', color='skyblue', ax=ax2)
ax2.set_ylabel("Energy Production (GWh)")
ax2.set_title(f"Total {selected_energy} per Country")
ax2.grid(True)
st.pyplot(fig2)

# ---------------------------
# Forecasting - Linear Regression
# ---------------------------
st.subheader(f"Forecast {selected_energy} (Global)")

global_energy = df.groupby('Year')[selected_energy].sum().reset_index()
X = global_energy[['Year']]
y = global_energy[selected_energy]
model = LinearRegression()
model.fit(X, y)

future_years = pd.DataFrame({'Year':[2024,2025,2026,2027,2028]})
pred = model.predict(future_years)
forecast_df = pd.DataFrame({'Year': future_years['Year'], f'Predicted_{selected_energy}': pred})

st.dataframe(forecast_df)

# Plot actual + forecast
fig3, ax3 = plt.subplots(figsize=(12,6))
sns.lineplot(x='Year', y=selected_energy, data=global_energy, marker='o', label='Actual', ax=ax3)
sns.lineplot(x='Year', y=f'Predicted_{selected_energy}', data=forecast_df, marker='o', label='Forecast', ax=ax3)
ax3.set_ylabel("Energy Production (GWh)")
ax3.set_title(f"Global {selected_energy} Forecast")
ax3.grid(True)
ax3.legend()
st.pyplot(fig3)

# ---------------------------
# Correlation Heatmap
# ---------------------------
st.subheader("Correlation Between Energy Types")
fig4, ax4 = plt.subplots(figsize=(8,5))
sns.heatmap(df[energy_cols].corr(), annot=True, cmap="coolwarm", ax=ax4)
st.pyplot(fig4)

st.markdown("Dashboard ini interaktif: pilih negara, jenis energi, dan range tahun untuk melihat tren, perbandingan total, dan prediksi energi terbarukan global.")

