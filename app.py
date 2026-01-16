import streamlit as st
import pickle
import pandas as pd
import tensorflow as tf
# from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pickle

model = tf.keras.model.load_model('model.h5')

with open('label_encoder_gender.pkl','wb') as file:
    label_encoder_gender=pickle.load(file)
with open('onehot_encoder_geo.pkl','wb') as file:
    onehot_encoder_geo=pickle.load(file)
with open('scaler.pkl','rb') as file:
    scaler=pickel.load(file)
st.title('customer churn prediction')



geography=st.selectbox('Geography',onehot_encoder_geo.categories[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number Of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])