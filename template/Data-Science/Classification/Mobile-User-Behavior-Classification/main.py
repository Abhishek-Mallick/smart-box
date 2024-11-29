import os
import pickle
import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model

model_folder = './model'
model_filename = 'user_behavior_model.keras'
model_path = os.path.join(model_folder, model_filename)
model = load_model(model_path)
print(model_path)
scaler_filename = 'scaler.pkl'
scaler_path = os.path.join(model_folder, scaler_filename)
print(scaler_path)
file = open(scaler_path, 'rb')
scaler = pickle.load(file)

st.set_page_config(
    page_title = "Mobile User Behavior",
)
st.markdown("## :blue[Mobile User Behavior Classification]")
st.image('https://img.freepik.com/premium-photo/horizontal-banner-template-with-multicultural-people-group-using-smartphone-university-college-backyard-millennial-friends-addicted-by-mobile-smart-phone-warm-vivid-filter_646109-1847.jpg?w=1380')
st.markdown("---")

sample_data = {
    "User ID": [1, 2, 3, 4, 5],
    "Device Model": ["Google Pixel 5", "OnePlus 9", "Xiaomi Mi 11", "Google Pixel 5", "iPhone 12"],
    "Operating System": ["Android", "Android", "Android", "Android", "iOS"],
    "App Usage Time (min/day)": [393, 268, 154, 239, 187],
    "Screen On Time (hours/day)": [6.4, 4.7, 4.0, 4.8, 4.3],
    "Battery Drain (mAh/day)": [1872, 1331, 761, 1676, 1367],
    "Number of Apps Installed": [67, 42, 32, 56, 58],
    "Data Usage (MB/day)": [1122, 944, 322, 871, 988],
    "Age": [40, 47, 42, 20, 31],
    "Gender": ["Male", "Female", "Male", "Male", "Female"],
    "User Behavior Class": [4, 3, 2, 3, 3]
}
st.text("Sample data")
st.dataframe(sample_data)

def preprocess_input_data(data):
    categorical_cols = ['Device Model', 'Operating System', 'Gender']
    
    data_encoded = pd.get_dummies(data, columns = categorical_cols, drop_first = True)

    expected_columns = ['App Usage Time (min/day)',
                    'Screen On Time (hours/day)',
                    'Battery Drain (mAh/day)',
                    'Number of Apps Installed',
                    'Data Usage (MB/day)',
                    'Age',
                    'Device Model_OnePlus 9',
                    'Device Model_Samsung Galaxy S21',
                    'Device Model_Xiaomi Mi 11',
                    'Device Model_iPhone 12',
                    #  'Device Model_Google Pixel 5',  
                    'Operating System_iOS',
                    'Gender_Male']

    data_encoded = data_encoded.reindex(columns = expected_columns, fill_value = 0)
    print(data_encoded.columns)

    scaled_data = scaler.transform(data_encoded)
    return scaled_data

def predict_behavior(input_data):
    input_data_processed = preprocess_input_data(input_data)
    # print(input_data_processed)
    input_data_scaled = input_data_processed.reshape(1, -1)
    # print(input_data_scaled)
    prediction = model.predict(input_data_scaled)
    predicted_class = np.argmax(prediction, axis = 1) + 1  # Adding 1 to match class labels (1-5)
    return predicted_class

def show_heatbar(predicted_class):
    if predicted_class[0] == 1:
        progress_color = 'green'
    elif predicted_class[0] == 2:
        progress_color = 'lightgreen'
    elif predicted_class[0] == 3:
        progress_color = 'yellow'
    elif predicted_class[0] == 4:
        progress_color = 'orange'
    else:
        progress_color = 'red'

    # Displaying a colored progress bar
    st.write(f"Predicted User Behavior Class: {predicted_class[0]}")
    progress = max((predicted_class[0] - 1) / 4, 0.1)
    st.markdown(
    f"""
    <div style="width: 100%; background-color: #262730; border-radius: 10px;">
        <div style="width: {progress * 100}%; height: 7px; border-radius: 10px; background-color: {progress_color};"></div>
    </div>
    """, unsafe_allow_html = True
)

def main():
    # User input fields
    st.markdown('### User Input Features')
    operating_system = st.selectbox("Operating System", ["Android", "iOS"])

    if operating_system == 'Android':
        device_model = st.selectbox("Device Model", ["Xiaomi Mi 11", "Google Pixel 5", "OnePlus 9", "Samsung Galaxy S21"])
    else:
        device_model = st.selectbox("Device Model", ["iPhone 12"])

    # device_model = st.selectbox("Device Model", ["Xiaomi Mi 11", "iPhone 12", "Google Pixel 5", "OnePlus 9", "Samsung Galaxy S21"])
    app_usage_time = st.number_input("App Usage Time (min/day)", min_value = 0)
    screen_on_time = st.number_input("Screen On Time (hours/day)", min_value = 0.0)
    battery_drain = st.number_input("Battery Drain (mAh/day)", min_value = 0)
    num_apps_installed = st.number_input("Number of Apps Installed", min_value = 0)
    data_usage = st.number_input("Data Usage (MB/day)", min_value = 0)
    age = st.number_input("Age", min_value = 0)
    gender = st.selectbox("Gender", ["Male", "Female"])

    # Create a DataFrame for input data
    input_data = pd.DataFrame({
        'Device Model': [device_model],
        'Operating System': [operating_system],
        'App Usage Time (min/day)': [app_usage_time],
        'Screen On Time (hours/day)': [screen_on_time],
        'Battery Drain (mAh/day)': [battery_drain],
        'Number of Apps Installed': [num_apps_installed],
        'Data Usage (MB/day)': [data_usage],
        'Age': [age],
        'Gender': [gender]
    })

    # Predict the behavior class when the button is pressed
    if st.button('Predict User Behavior Class'):
        predicted_class = predict_behavior(input_data)
        show_heatbar(predicted_class)

if __name__ == "__main__":
    main()