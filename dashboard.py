# app.py

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
import numpy as np

# =========================
# 1. Title & description
# =========================
st.set_page_config(page_title="Renewable Energy Dashboard", layout="wide")
st.title("Renewable Energy Dashboard 🌱")
st.markdown("""
Dashboard interaktif untuk memantau produksi & konsumsi energi serta mengevaluasi investasi terhadap dampak lingkungan.
""")

# =========================
# 2. Load dataset
# =========================
url = "https://github.com/Evameivina/Renewable-Energy-/blob/main/energy_dataset_.csv"
df = pd.read_csv(url)

# Hapus duplikat
df = df.drop_duplicates()

# Bersihkan kolom & handle missing
df.columns = [col.strip() for col in df.columns]
df.fillna(0, inplace=True)

# Mapping kode numerik ke kategori
energy_map = {1: "Solar", 2: "Wind", 3: "Hydroelectric", 4: "Geothermal",
              5: "Biomass", 6: "Tidal", 7: "Wave"}
grid_map = {1: "Fully Integrated", 2: "Partially Integrated",
            3: "Minimal Integration", 4: "Isolated Microgrid"}
funding_map = {1: "Government", 2: "Private", 3: "Public-Private Partnership"}

df['Type_of_Renewable_Energy'] = df['Type_of_Renewable_Energy'].map(energy_map)
df['Grid_Integration_Level'] = df['Grid_Integration_Level'].map(grid_map)
df['Funding_Sources'] = df['Funding_Sources'].map(funding_map)

# Fokus kolom
fokus_cols = ['Type_of_Renewable_Energy', 
              'Energy_Production_MWh', 'Energy_Consumption_MWh',
              'Initial_Investment_USD', 'GHG_Emission_Reduction_tCO2e',
              'Air_Pollution_Reduction_Index', 'Funding_Sources']
df = df[fokus_cols]

# =========================
# 3. Sidebar - Filters
# =========================
st.sidebar.header("Filters")
energy_options = st.sidebar.multiselect(
    "Pilih jenis energi:",
    options=df['Type_of_Renewable_Energy'].unique(),
    default=df['Type_of_Renewable_Energy'].unique()
)

funding_options = st.sidebar.multiselect(
    "Pilih sumber pendanaan:",
    options=df['Funding_Sources'].unique(),
    default=df['Funding_Sources'].unique()
)

filtered_df = df[(df['Type_of_Renewable_Energy'].isin(energy_options)) &
                 (df['Funding_Sources'].isin(funding_options))]

# =========================
# 4. Fokus 1: Tren Produksi & Konsumsi Energi
# =========================
st.subheader("Tren Produksi & Konsumsi Energi")
st.markdown("Line chart produksi & konsumsi energi per jenis energi.")

plt.figure(figsize=(12,6))
for energy in filtered_df['Type_of_Renewable_Energy'].unique():
    temp = filtered_df[filtered_df['Type_of_Renewable_Energy'] == energy]
    plt.plot(temp['Energy_Production_MWh'].values, label=f"{energy} Production")
    plt.plot(temp['Energy_Consumption_MWh'].values, label=f"{energy} Consumption")

plt.xlabel("Record Index")
plt.ylabel("MWh")
plt.title("Energy Production vs Consumption")
plt.legend()
st.pyplot(plt)

# Forecasting sederhana untuk satu jenis energi (misal Solar)
if "Solar" in filtered_df['Type_of_Renewable_Energy'].unique():
    st.markdown("Prediksi produksi Solar 5 periode ke depan:")
    forecast_df = filtered_df[filtered_df['Type_of_Renewable_Energy']=="Solar"]
    X = np.array(range(len(forecast_df))).reshape(-1,1)
    y = forecast_df['Energy_Production_MWh'].values
    model = LinearRegression()
    model.fit(X, y)
    future_index = np.array(range(len(X), len(X)+5)).reshape(-1,1)
    y_pred_future = model.predict(future_index)
    st.write(y_pred_future)

# =========================
# 5. Fokus 2: Investasi vs Dampak Lingkungan
# =========================
st.subheader("Investasi vs Dampak Lingkungan")
st.markdown("Scatter plot investasi terhadap pengurangan emisi dan polusi udara.")

plt.figure(figsize=(10,6))
sns.scatterplot(data=filtered_df,
                x='Initial_Investment_USD',
                y='GHG_Emission_Reduction_tCO2e',
                hue='Type_of_Renewable_Energy',
                style='Funding_Sources',
                s=100)
plt.xlabel("Initial Investment (USD)")
plt.ylabel("GHG Emission Reduction (tCO2e)")
plt.title("Investment vs GHG Emission Reduction")
st.pyplot(plt)

plt.figure(figsize=(10,6))
sns.scatterplot(data=filtered_df,
                x='Initial_Investment_USD',
                y='Air_Pollution_Reduction_Index',
                hue='Type_of_Renewable_Energy',
                style='Funding_Sources',
                s=100)
plt.xlabel("Initial Investment (USD)")
plt.ylabel("Air Pollution Reduction Index")
plt.title("Investment vs Air Pollution Reduction")
st.pyplot(plt)

