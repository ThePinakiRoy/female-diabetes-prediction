import streamlit as st
import pickle
import numpy as np

st.set_page_config(
    page_title="Ex-stream-ly Cool App",
    page_icon="🧊")
model = pickle.load(open("model.pkl", "rb"))
scaler = pickle.load(open("scaler.pkl", "rb"))

st.title("Female Diabetes Prediction")


col1, col2, col3 = st.columns(3)
with col1:
   pregnancies = st.number_input("Enter the number of Pregnancies")

with col2:
   glucose = st.number_input("Enter Glucose level")

with col3:
   bloodPressure = st.number_input("Enter Blood Pressure")

col4, col5, col6 = st.columns(3)
with col4:
   skinThickness = st.number_input("Enter Skin Thickness")

with col5:
   insulin = st.number_input("Enter Insulin level")

with col6:
   BMI = st.number_input("Enter BMI")

col7, col8 = st.columns(2)
with col7:
   diabetesPedigreeFunction = st.number_input("Enter the Diabetes Pedigree Function Value")

with col8:
   age = st.number_input("Enter your Age")



	    			

if st.button("Predict"):
    test_input = (pregnancies,glucose,bloodPressure,skinThickness,insulin,BMI,diabetesPedigreeFunction,age)    
    # changing to np array
    test_data_np = np.asarray(test_input)
    # reshape to predict
    test_data_reshape = test_data_np.reshape(1, -1)
    test_std_data = scaler.transform(test_data_reshape)
    pred = model.predict(test_std_data)
    if pred[0] == 0:    
        st.header("Non Diabetic")
    else:
        st.header("Diabetic")
