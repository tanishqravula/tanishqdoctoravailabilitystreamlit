"""This modules contains data about prediction page"""

# Import necessary modules
import streamlit as st

# Import necessary functions from web_functions
from web_functions import predict
import pandas as pd
import numpy as np
from web_functions import load_data


def app(df, X, Y):
    """This function create the prediction page"""

    # Add title to the page
    st.title("Prediction Page")

    # Add a brief description
    st.markdown(
        """
            <p style="font-size:25px">
                <b style="color:green">Appointment Availability</b> Checker.
            </p>
        """, unsafe_allow_html=True)
    
    # Take feature input from the user
    # Add a subheader
    st.subheader("Select Values:")

    # Take input of features from the user.
    PatientId = st.slider("PatientId", int(df["PatientId"].min()), int(df["PatientId"].max()))
    AppointmentID = st.slider("AppointmentID", int(df["AppointmentID"].min()), int(df["AppointmentID"].max()))
    Gender = st.slider("Gender", "M","F")
    ScheduledDay = st.slider("ScheduledDay", float(df["ScheduledDay"].min()), float(df["ScheduledDay"].max()))
    AppointmentDay= st.slider("AppointmentDay", float(df["AppointmentDay"].min()), float(df["AppointmentDay"].max()))
    Age = st.slider("Age", float(df["Age"].min()), float(df["Age"].max()))
    Neighbourhood = st.text_input("Neighbourhood")
    Scholarship= st.slider("Scholarship", float(df["Scholarship"].min()), float(df["Scholarship"].max()))
    Hipertension= st.slider("Hipertension", float(df["Hipertension"].min()), float(df["Hipertension"].max()))
    Diabetes= st.slider("Diabetes", float(df["Diabetes"].min()), float(df["Diabetes"].max()))
    Alcoholism= st.slider("Alcoholism", float(df["Alcoholism"].min()), float(df["Alcoholism"].max()))
    Handcap= st.slider("Handcap", float(df["Handcap"].min()), float(df["Handcap"].max()))
    SMS_received= st.slider("SMS_received", float(df["SMS_received"].min()), float(df["SMS_received"].max()))

    # Create a list to store all the features
    features = [PatientId ,AppointmentID ,Gender,ScheduledDay,AppointmentDay,Age , Neighbourhood,Scholarship,Hipertension,Diabetes,Alcoholism,Handcap,SMS_received]

    # Create a button to predict
    if st.button("Predict"):
        # Get prediction and model score
        prediction, score = predict(X, Y, features)
        st.info("Checking Appointment Availability...")

        # Print the output according to the prediction
        if (prediction == 'No'):
            st.success("Appointment availabilty met")
        else:
            st.error("Appointment schedule not available")

        # Print teh score of the model 
        st.write("The model used is trusted by doctor and has an accuracy of ", (score*100),"%")
