import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Title of the app
st.title("Calorie Calculator")

# Sidebar for user input
st.sidebar.header("User Information")

# User inputs
age = st.sidebar.number_input("Age", min_value=0, max_value=120, value=25)
weight = st.sidebar.number_input("Weight (kg)", min_value=0.0, value=70.0, format="%.1f")
height = st.sidebar.number_input("Height (cm)", min_value=0, max_value=250, value=170)
gender = st.sidebar.radio("Gender", ("Male", "Female"))
activity_level = st.sidebar.selectbox("Activity Level", ["Sedentary", "Lightly active", "Moderately active", "Very active", "Extra active"])

# Goal setting
st.sidebar.header("Goals")
weight_goal = st.sidebar.number_input("Goal Weight (kg)", min_value=weight - 20, value=weight, format="%.1f")
weight_change_rate = st.sidebar.selectbox("Weight Change Rate", ["Maintain", "Lose 0.5 kg/week", "Lose 1 kg/week", "Gain 0.5 kg/week", "Gain 1 kg/week"])

# Exercise Calculator
st.sidebar.header("Exercise Calculator")
exercise = st.sidebar.selectbox("Type of Exercise", ["Running", "Cycling", "Swimming", "Walking"])
exercise_duration = st.sidebar.number_input("Duration (minutes)", min_value=0, value=30)

# BMR Calculation
def calculate_bmr(weight, height, age, gender):
    if gender == "Male":
        return 10 * weight + 6.25 * height - 5 * age + 5
    else:
        return 10 * weight + 6.25 * height - 5 * age - 161

# TDEE Calculation
def calculate_tdee(bmr, activity_level):
    activity_multipliers = {
        "Sedentary": 1.2,
        "Lightly active": 1.375,
        "Moderately active": 1.55,
        "Very active": 1.725,
        "Extra active": 1.9
    }
    return bmr * activity_multipliers[activity_level]

# Macronutrient Distribution
def calculate_macros(tdee):
    protein = tdee * 0.3 / 4
    fats = tdee * 0.25 / 9
    carbs = tdee * 0.45 / 4
    return protein, fats, carbs

# Calculate calories burned based on exercise type
def calculate_calories_burned(exercise, duration):
    exercise_calories = {
        "Running": 10,  # calories per minute
        "Cycling": 8,   # calories per minute
        "Swimming": 12, # calories per minute
        "Walking": 5    # calories per minute
    }
    return exercise_calories.get(exercise, 0) * duration

# Perform calculations
bmr = calculate_bmr(weight, height, age, gender)
tdee = calculate_tdee(bmr, activity_level)
protein, fats, carbs = calculate_macros(tdee)
calories_burned = calculate_calories_burned(exercise, exercise_duration)

# Display results
st.header("Results")

st.subheader("Basal Metabolic Rate (BMR)")
st.write(f"Your BMR is {bmr:.2f} calories/day.")

st.subheader("Total Daily Energy Expenditure (TDEE)")
st.write(f"Your TDEE is {tdee:.2f} calories/day.")

st.subheader("Macronutrient Distribution")
st.write(f"Protein: {protein:.2f} grams/day")
st.write(f"Fats: {fats:.2f} grams/day")
st.write(f"Carbohydrates: {carbs:.2f} grams/day")

# Display macronutrient distribution as a pie chart
st.subheader("Macronutrient Distribution Chart")
labels = ['Protein', 'Fats', 'Carbohydrates']
sizes = [protein * 4, fats * 9, carbs * 4]
colors = ['#ff9999','#66b3ff','#99ff99']

fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140)
ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
st.pyplot(fig)

# Goal setting
st.subheader("Weight Goal Progress")
weight_change_needed = weight - weight_goal
st.write(f"You need to {'gain' if weight_change_needed < 0 else 'lose'} {abs(weight_change_needed):.2f} kg to reach your goal weight.")

# Calculate calories needed to change weight
calories_needed = 7700 * weight_change_needed  # Approx. 7700 calories per kg of body weight
st.write(f"To achieve this goal, you need to {'increase' if weight_change_needed < 0 else 'decrease'} your daily intake by {abs(calories_needed):.2f} calories.")

# Exercise details
st.subheader("Calories Burned from Exercise")
st.write(f"Calories burned from {exercise} for {exercise_duration} minutes: {calories_burned:.2f} calories.")

st.write("Adjust your caloric intake and exercise routine based on your goals.")
