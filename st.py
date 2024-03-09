import streamlit as st
import pandas as pd
import sklearn
import pickle

# Set the background image and text color
background_image = """
<style>
[data-testid="stAppViewContainer"] > .main {
    background-image: url("https://media.istockphoto.com/id/1792119618/photo/green-tea-tree-leaves-camellia-sinensis-in-organic-farm-sunlight-fresh-young-tender-bud.jpg?s=1024x1024&w=is&k=20&c=G-YJnIy8SP8S0H_goXhNvwQXG0kc4cSyIBrGkUP4Ums=");
    background-size: 100vw 100vh;
    background-position: center;  
    background-repeat: no-repeat;
    color: #FFFFFF; /* Set text color to blue */
}
</style>
"""

st.markdown(background_image, unsafe_allow_html=True)



# Title and description
st.title('Agriculture Product Optimization')
st.write('Welcome to the Agriculture Product Optimization app. Select the parameters below to get predictions and visualizations.')



# Load the pre-trained logistic regression model
with open('logistic_regression_model.pkl', 'rb') as f:
    logistic_regression_model = pickle.load(f)
# Sidebar options
selected_option = st.selectbox('Select Option', ['Predict'])

if selected_option == 'Predict':
    st.subheader('Predict')
    # Allow user to input parameters for prediction
    n = st.number_input('Enter value for N',min_value=0.0)
    p = st.number_input('Enter value for P',min_value=0.0)
    k = st.number_input('Enter value for K',min_value=0.0)
    temperature = st.number_input('Enter value for Temperature')
    humidity = st.number_input('Enter value for Humidity',min_value=0.0, max_value=100.0)
    ph = st.number_input('Enter value for pH', min_value=0.0, max_value=14.0)
    rainfall = st.number_input('Enter value for Rainfall',min_value=0.0)
    
    # Make prediction using the logistic regression model
    prediction = logistic_regression_model.predict([[n, p, k, temperature, humidity, ph, rainfall]])
    
    # Display the prediction result
    st.write('Predicted Crop:', prediction[0])
