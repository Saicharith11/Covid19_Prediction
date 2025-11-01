# -------------------------------------------------------------
# 🌍 COVID-19 SEIR + Machine Learning Dashboard
# -------------------------------------------------------------
# Author: Karthik
# Run using: streamlit run app.py
# -------------------------------------------------------------

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Streamlit Page Setup
st.set_page_config(page_title="COVID-19 SEIR & ML Forecast Dashboard", page_icon="🧬", layout="wide")
st.title("🌍 COVID-19 SEIR Simulation + Machine Learning Forecast")
st.write("This dashboard combines epidemiological modeling (SEIR) with Machine Learning for real-time COVID-19 case analysis.")

# -------------------------------------------------------------
# 1️⃣ Load Live COVID-19 Data (Our World in Data)
# -------------------------------------------------------------
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/owid/covid-19-data/master/public/data/owid-covid-data.csv"
    data = pd.read_csv(url)
    data['date'] = pd.to_datetime(data['date'])
    return data

data = load_data()
st.sidebar.header("⚙️ Settings")

# Country Selection
countries = sorted(data['location'].dropna().unique())
country = st.sidebar.selectbox("Select Country", countries, index=countries.index("India"))

country_data = data[data['location'] == country].copy()
country_data['new_cases'] = country_data['new_cases'].fillna(0)
country_data['total_cases'] = country_data['total_cases'].fillna(method='ffill').fillna(0)

# -------------------------------------------------------------
# 2️⃣ Display Real-Time Data
# -------------------------------------------------------------
st.subheader(f"📊 COVID-19 Data for {country}")

col1, col2 = st.columns(2)

with col1:
    st.write("**Daily New Cases**")
    st.line_chart(country_data.set_index('date')['new_cases'])

with col2:
    st.write("**Cumulative Total Cases**")
    st.line_chart(country_data.set_index('date')['total_cases'])

# -------------------------------------------------------------
# 3️⃣ SEIR Model Parameters
# -------------------------------------------------------------
st.sidebar.subheader("🧬 SEIR Model Controls")

beta = st.sidebar.slider("Infection Rate (β)", 0.1, 1.0, 0.5, 0.05)
sigma = st.sidebar.slider("Incubation Rate (σ = 1/incubation period)", 0.05, 1.0, 1/5.2, 0.05)
gamma = st.sidebar.slider("Recovery Rate (γ)", 0.05, 1.0, 0.2, 0.05)
days = st.sidebar.slider("Simulation Days", 30, 300, 180, 10)

# -------------------------------------------------------------
# 4️⃣ SEIR Model Definition
# -------------------------------------------------------------
def seir_equations(y, t, N, beta, sigma, gamma):
    S, E, I, R = y
    dSdt = -beta * S * I / N
    dEdt = beta * S * I / N - sigma * E
    dIdt = sigma * E - gamma * I
    dRdt = gamma * I
    return dSdt, dEdt, dIdt, dRdt

def run_seir_model(N, E0, I0, R0, beta, sigma, gamma, days):
    S0 = N - E0 - I0 - R0
    y0 = S0, E0, I0, R0
    t = np.linspace(0, days, days)
    ret = integrate.odeint(seir_equations, y0, t, args=(N, beta, sigma, gamma))
    S, E, I, R = ret.T
    return t, S, E, I, R

# -------------------------------------------------------------
# 5️⃣ Run SEIR Simulation
# -------------------------------------------------------------
N = 1e7  # Assume 10 million population
E0, I0, R0 = 10, 5, 0
t, S, E, I, R = run_seir_model(N, E0, I0, R0, beta, sigma, gamma, days)

# -------------------------------------------------------------
# 6️⃣ Plot SEIR Model Results
# -------------------------------------------------------------
st.subheader("🧬 SEIR Model Simulation")

fig, ax = plt.subplots(figsize=(9,5))
ax.plot(t, S/N, 'b', label='Susceptible')
ax.plot(t, E/N, 'y', label='Exposed')
ax.plot(t, I/N, 'r', label='Infected')
ax.plot(t, R/N, 'g', label='Recovered')
ax.set_xlabel("Days")
ax.set_ylabel("Fraction of Population")
ax.set_title("SEIR Model Dynamics")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.info("Adjust the sliders on the left to simulate lockdown or vaccination scenarios.")

# -------------------------------------------------------------
# 🤖 STEP 7 – Machine Learning Forecast (Linear Regression / XGBoost)
# -------------------------------------------------------------
st.subheader(f"🤖 Machine Learning Case Forecast for {country} (Next 14 Days)")

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from xgboost import XGBRegressor
from datetime import datetime

# Sidebar choice for model type
model_choice = st.sidebar.selectbox(
    "Select Forecasting Model",
    ("Linear Regression", "XGBoost (Advanced)")
)

# Prepare ML data
ml_data = country_data.tail(120).reset_index(drop=True)
ml_data['Day'] = np.arange(len(ml_data))
X = ml_data[['Day']]
y = ml_data['total_cases']

# Train selected model
if model_choice == "Linear Regression":
    model = LinearRegression()
    model.fit(X, y)
else:
    model = XGBRegressor(
        n_estimators=300,
        learning_rate=0.1,
        max_depth=5,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42
    )
    model.fit(X, y)

# Predict next 14 days
future_days = np.arange(len(ml_data), len(ml_data) + 14).reshape(-1, 1)
predictions = model.predict(future_days)

# Align predictions to today's date
today = pd.Timestamp(datetime.today().date())
future_dates = pd.date_range(start=today + pd.Timedelta(days=1), periods=14)
pred_df = pd.DataFrame({
    'Date': future_dates,
    'Predicted Total Cases': predictions
})

# Evaluate model
y_pred_train = model.predict(X)
rmse = np.sqrt(mean_squared_error(y, y_pred_train))

# Plot Forecast
fig2, ax2 = plt.subplots(figsize=(9,5))
ax2.plot(ml_data['date'], y, label='Actual Cases', color='blue')
ax2.plot(pred_df['Date'], pred_df['Predicted Total Cases'], 'r--', label=f'Predicted (Next 14 Days) – {model_choice}')
ax2.set_xlabel("Date")
ax2.set_ylabel("Total Cases")
ax2.set_title(f"Predicted COVID-19 Cases for {country} using {model_choice}")
ax2.legend()
ax2.grid(True)
st.pyplot(fig2)

# Display results
st.success(f"✅ {model_choice} RMSE (training): {rmse:.2f}")
st.dataframe(pred_df)

st.caption("💡 Use the sidebar to switch between Linear Regression and XGBoost for comparison.")
