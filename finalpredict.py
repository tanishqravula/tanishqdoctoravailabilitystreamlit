import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

# Load the dataset
@st.cache
def load_data():
    data = pd.read_csv('KaggleV2-May-2016.csv')
    return data

data = load_data()

# Sidebar with data preview
st.sidebar.header('Data Preview')
st.sidebar.write(data.head())

# Sidebar for model parameters
st.sidebar.header('Model Parameters')
test_size = st.sidebar.slider('Test Size', 0.1, 0.5, 0.3)
random_state = st.sidebar.number_input('Random State', 0, 1000, 42)

# Preprocess the data
def preprocess_data(data):
    # Encode categorical variables
    label_encoder = LabelEncoder()
    data['Gender'] = label_encoder.fit_transform(data['Gender'])
    data['Neighbourhood'] = label_encoder.fit_transform(data['Neighbourhood'])

    # Convert date columns to datetime objects
    data['ScheduledDay'] = pd.to_datetime(data['ScheduledDay'])
    data['AppointmentDay'] = pd.to_datetime(data['AppointmentDay'])

    # Extract features from datetime columns
    data['ScheduledYear'] = data['ScheduledDay'].dt.year
    data['ScheduledMonth'] = data['ScheduledDay'].dt.month
    data['ScheduledDay'] = data['ScheduledDay'].dt.day
    data['AppointmentYear'] = data['AppointmentDay'].dt.year
    data['AppointmentMonth'] = data['AppointmentDay'].dt.month
    data['AppointmentDay'] = data['AppointmentDay'].dt.day

    return data

data = preprocess_data(data)

# Split the data into train and test sets
X = data[['Gender', 'ScheduledDay', 'AppointmentDay', 'Age', 'Neighbourhood', 'Scholarship', 'Hipertension', 'Diabetes', 'Alcoholism', 'Handcap', 'SMS_received']]
y = data['No-show']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

# Train a logistic regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Model prediction
y_pred = model.predict(X_test)

# Model evaluation
accuracy = accuracy_score(y_test, y_pred)

# Streamlit UI
st.title('Medical Appointment No-Show Prediction')

# Display the dataset
st.header('Dataset')
st.write(data)

# Display model parameters
st.header('Model Parameters')
st.write(f'Test Size: {test_size}')
st.write(f'Random State: {random_state}')

# Display model evaluation results
st.header('Model Evaluation')
st.write(f'Accuracy: {accuracy:.2f}')

# Prediction form
st.header('Make a Prediction')
st.subheader('Enter patient information:')
gender = st.selectbox('Gender', ['F', 'M'])
scheduled_day = st.date_input('Scheduled Day')
appointment_day = st.date_input('Appointment Day')
age = st.number_input('Age', min_value=0)
neighbourhood = st.text_input('Neighbourhood')
scholarship = st.checkbox('Scholarship')
hipertension = st.checkbox('Hipertension')
diabetes = st.checkbox('Diabetes')
alcoholism = st.checkbox('Alcoholism')
handcap = st.checkbox('Handicap')
sms_received = st.checkbox('SMS Received')

# Preprocess user input
user_data = pd.DataFrame({'Gender': [gender],
                          'ScheduledDay': [scheduled_day],
                          'AppointmentDay': [appointment_day],
                          'Age': [age],
                          'Neighbourhood': [neighbourhood],
                          'Scholarship': [scholarship],
                          'Hipertension': [hipertension],
                          'Diabetes': [diabetes],
                          'Alcoholism': [alcoholism],
                          'Handcap': [handcap],
                          'SMS_received': [sms_received]})

user_data = preprocess_data(user_data)

# Predict using the trained model
if st.button('Predict'):
    prediction = model.predict(user_data)[0]
    st.subheader(f'Predicted No-Show: {prediction}')

# Note: This is a simplified example. In practice, you might want to use a more complex model and further data preprocessing.
