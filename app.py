import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,OneHotEncoder,LabelEncoder
import pandas as pd
import pickle

#loading the model
model = tf.keras.models.load_model("model.h5")

#loading the encoders and scaler
with open("scaler.pkl",'rb') as file:
    scaler = pickle.load(file)

with open("ohe_geography.pkl",'rb') as file:
    ohe_geo = pickle.load(file)

with open("label_encoder_gender.pkl","rb") as file:
    label_encoder_gen = pickle.load(file)


st.title("Customer Churn Prediction")

geography = st.selectbox("Geography",ohe_geo.categories_[0])
gender = st.selectbox("Gender",label_encoder_gen.classes_)
age = st.select_slider("Age",options=list(range(18, 93)))
balance =st.number_input("Balance")
credit_score = st.number_input("Credit Score")
estimated_salary = st.number_input("Estimated Salary")
tenure = st.select_slider("Tenure",options=list(range(0,10)))
num_of_products = st.selectbox("Number of Products",options=list(range(1,4)))
has_cr_card = st.selectbox("Has Credit Card",[0,1])
is_active_member = st.selectbox("Is Active Memeber",[0,1])


input_data = pd.DataFrame({
    "CreditScore":[credit_score],
    "Gender" : [label_encoder_gen.transform([gender])[0]],
    "Age": [age],
    "Tenure":[tenure],
    "Balance":[balance],
    "NumOfProducts":[num_of_products],
    "HasCrCard":[has_cr_card],
    "IsActiveMember":[is_active_member],
    "EstimatedSalary":[estimated_salary]
})

geo_encoded = ohe_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded,columns = ohe_geo.get_feature_names_out())

input_data = pd.concat([input_data.reset_index(drop=True),geo_encoded_df],axis=1)

input_data_sc = scaler.transform(input_data)

prediction = model.predict(input_data_sc)
prediction_proba = prediction[0][0]

st.write(prediction_proba)
if prediction_proba > 0.5:
    st.write("The customer is likely to churn")

else:
    st.write("The Customer is not likely to churn")
