import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Title and Introduction
st.title("Hydration Tracker")
st.subheader("Monitor your daily water intake and stay hydrated!")

# Daily Water Intake Monitoring
st.header("Hydration Tracking")
water_intake = st.number_input("Enter your water intake (ml)", min_value=0, step=100)
st.write(f"Today's water intake: {water_intake} ml")

# Customized Reminders
st.header("Customized Reminders")
activity_level = st.selectbox("Select your activity level:", ["Low", "Moderate", "High"])


# Environmental Factors
st.header("Environmental Factors")
temperature = st.slider("Current temperature (°C):", min_value=-10, max_value=50)
humidity = st.slider("Current humidity (%):", min_value=0, max_value=100)

st.write(f"Temperature: {temperature}°C, Humidity: {humidity}%")

# Hydration Recommendations
st.header("Hydration Recommendations")

def calculate_recommendation(temp, humidity, activity):
    base_intake = 2000  # in ml
    if activity == "High":
        base_intake += 500
    if temp > 30:
        base_intake += 300
    if humidity > 70:
        base_intake += 200
    return base_intake

recommended_intake = calculate_recommendation(temperature, humidity, activity_level)
st.write(f"Recommended daily water intake: {recommended_intake} ml")

# Hydration Tips
st.header("Hydration Tips")
if temperature > 30:
    st.write("It's hot outside. Make sure to drink extra water.")
if humidity > 70:
    st.write("High humidity can increase your water needs.")

# Set and Track Goals
st.header("Hydration Goals")
goal = st.number_input("Set your daily water intake goal (ml):", min_value=500, value=2000)

if water_intake >= goal:
    st.write("Congratulations! You've met your hydration goal for today.")
else:
    st.write("Keep going! You're on your way to meet your goal.")




# Display images related to hydration

st.image("https://www.ukbusinesssupplies.co.uk/cdn/shop/articles/stay-hydrated_500x.jpg?v=1660566417", caption="Stay Hydrated!")


