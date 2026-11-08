import pickle
from datetime import date, timedelta

import pandas as pd
import requests
import streamlit as st

# ---- Settings ----------------------------------------------------------
CITIES = {
    "Pittsburgh": (40.44, -79.99),
    "New York": (40.71, -74.01),
    "Chicago": (41.88, -87.63),
    "Miami": (25.76, -80.19),
    "Los Angeles": (34.05, -118.24),
}
LOW_TEMP, HIGH_TEMP = 50, 100        # the temperatures (F) the model learned from
LOW_PRECIP, HIGH_PRECIP = 0, 1.5     # the daily precipitation (inches) the model learned from
LOW_SALES, HIGH_SALES = 200, 300     # below LOW_SALES is a slow day, above HIGH_SALES is a busy one
URL = "https://api.open-meteo.com/v1/forecast"
# ------------------------------------------------------------------------

# Load the trained model (we trained it once, in Colab, and saved it to a file)
with open("model_rain.pkl", "rb") as f:
    model = pickle.load(f)

st.set_page_config(page_title="Ice Cream Sales Predictor", page_icon="🍦")
st.title("🍦 Ice Cream Sales Predictor")
st.caption("Pick a city and a date. The app gets the weather forecast (temperature and rain) "
           "and predicts how many cones you will sell.")

city = st.selectbox("City", list(CITIES))
selected_date = st.date_input(
    "Date",
    value=date.today() + timedelta(days=1),
    min_value=date.today(),
    max_value=date.today() + timedelta(days=16),    # the forecast reaches about 16 days ahead
)

if st.button("Predict sales", type="primary"):
    # 1. Ask the weather API for the forecast
    latitude, longitude = CITIES[city]
    response = requests.get(
        URL,
        params={
            "latitude": latitude,
            "longitude": longitude,
            "daily": "temperature_2m_max,precipitation_sum",
            "temperature_unit": "fahrenheit",
            "precipitation_unit": "inch",       # the model was trained on inches
            "timezone": "auto",
            "start_date": str(selected_date),
            "end_date": str(selected_date),
        },
        timeout=30,
    )
    response.raise_for_status()
    daily = response.json()["daily"]
    temperature = daily["temperature_2m_max"][0]
    precipitation = daily["precipitation_sum"][0]

    # 2. Give the model the same inputs it was trained on (same names, same order), then predict
    features = pd.DataFrame({"temperature": [temperature], "precipitation": [precipitation]})
    prediction = max(0, model.predict(features)[0])

    # 3. Show the result
    col1, col2, col3 = st.columns(3)
    col1.metric("Forecast high", f"{temperature:.0f} °F")
    col2.metric("Forecast rain", f"{precipitation:.2f} in")
    col3.metric("Predicted sales", f"{prediction:.0f} cones")

    if temperature < LOW_TEMP or temperature > HIGH_TEMP:
        st.warning(f"This temperature is outside the range the model learned from ({LOW_TEMP} to {HIGH_TEMP} °F). "
                   "Treat the number with caution.")

    if precipitation < LOW_PRECIP or precipitation > HIGH_PRECIP:
        st.warning(f"This amount of rain is outside the range the model learned from "
                   f"({LOW_PRECIP} to {HIGH_PRECIP} inches). Treat the number with caution.")

    if prediction > HIGH_SALES:
        st.info("Busy day: make sure you are fully stocked.")
    elif prediction > LOW_SALES:
        st.info("A normal day is expected.")
    else:
        st.info("A slow day: consider a promotion, or order less.")
