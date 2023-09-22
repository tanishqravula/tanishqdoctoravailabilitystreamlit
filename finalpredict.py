import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv('KaggleV2-May-2016.csv')
    return data

data = load_data()

# Data preprocessing
data['ScheduledDay'] = pd.to_datetime(data['ScheduledDay'])
data['AppointmentDay'] = pd.to_datetime(data['AppointmentDay'])
data['No-show'] = data['No-show'].map({'No': 0, 'Yes': 1})

# Sidebar with dataset preview
st.sidebar.header('Dataset Preview')
st.sidebar.write(data.head())

# Sidebar for model parameters
st.sidebar.header('Model Parameters')
test_size = st.sidebar.slider('Test Size', 0.1, 0.5, 0.3, 0.05)
random_state = st.sidebar.number_input('Random State', 0, 1000, 42)

# Preprocess the data
def preprocess_data(data):
    # Extract relevant features
    data['ScheduledHour'] = data['ScheduledDay'].dt.hour
    data['ScheduledDOW'] = data['ScheduledDay'].dt.dayofweek
    data['AppointmentDOW'] = data['AppointmentDay'].dt.dayofweek
    data['LeadTime'] = (data['AppointmentDay'] - data['ScheduledDay']).dt.days

    # Select features and target variable
    X = data[['Gender', 'Age', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received', 'ScheduledHour', 'ScheduledDOW', 'AppointmentDOW', 'LeadTime']]
    y = data['No-show']

    return X, y

X, y = preprocess_data(data)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Train a Random Forest Classifier
model = RandomForestClassifier(random_state=random_state)
model.fit(X_train, y_train)

# Model prediction
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)
classification_rep = classification_report(y_test, y_pred)

# Streamlit UI
st.title('No-Show Appointment Prediction')

# Display model parameters
st.header('Model Parameters')
st.write(f'Test Size: {test_size}')
st.write(f'Random State: {random_state}')

# Display model evaluation results
st.header('Model Evaluation')
st.write(f'Accuracy: {accuracy:.2f}')
st.subheader('Classification Report:')
st.text(classification_rep)

# Prediction form
st.header('Make a Prediction')
st.subheader('Enter patient information:')
gender = st.radio('Gender', ['F', 'M'])
age = st.number_input('Age', min_value=0, max_value=100)
scholarship = st.checkbox('Scholarship')
hipertension = st.checkbox('Hipertension')
diabetes = st.checkbox('Diabetes')
alcoholism = st.checkbox('Alcoholism')
handcap = st.checkbox('Handicap')
sms_received = st.checkbox('SMS Received')
scheduled_hour = st.slider('Scheduled Hour', 0, 23, 12)
scheduled_dow = st.slider('Scheduled Day of Week', 0, 6, 2)
appointment_dow = st.slider('Appointment Day of Week', 0, 6, 2)
lead_time = st.number_input('Lead Time (days)', min_value=0)

# Preprocess user input
user_data = pd.DataFrame({'Gender': [gender],
                          'Age': [age],
                          'Scholarship': [scholarship],
                          'Hipertension': [hipertension],
                          'Diabetes': [diabetes],
                          'Alcoholism': [alcoholism],
                          'Handcap': [handcap],
                          'SMS_received': [sms_received],
                          'ScheduledHour': [scheduled_hour],
                          'ScheduledDOW': [scheduled_dow],
                          'AppointmentDOW': [appointment_dow],
                          'LeadTime': [lead_time]})

# Model prediction
if st.button('Predict'):
    prediction = model.predict(user_data)[0]
    if prediction == 0:
        result = 'No-show (1)'
    else:
        result = 'Show-up (0)'
    st.subheader(f'Predicted Outcome: {result}')

# Note: This is a simplified example. In practice, you may want to fine-tune the model, perform feature engineering, and handle missing data more rigorously.
