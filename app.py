import streamlit as st
import pickle
model=tf.keras.models.load_model('model.h5')
with open('le_gender.pkl','rb') as f:
  le_gender=pickle.load(f)
with open('ohe.pkl','rb') as f:
  ohe=pickle.load(f)
with open('scaler.pkl','rb') as f:
  scaler=pickle.load(f)

st.title('Customer Churn Prediction')
geography=st.selectbox('Geography',ohe.categories_[0])
gender=st.selectbox('Gender',le_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit Score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

ip_data=pd.DataFrame({
    'creditscore':[credit_score],
    'Gender':[le_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'balance':[balance],
    'numofproducts':[num_of_products],
    'hasCrcard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'estimatedsalary':[estimated_salary]
})
geo_ohe=ohe.transform([[geography]]).toarray()
geo_ohe_df=pd.DataFrame(geo_ohe,columns=ohe.get_feature_names_out(['Geography']))
ip_data=pd.concat([ip_data.reset_index(drop=True),geo_ohe_df],axis=1)
ip_data_scaled=scaler.transform(ip_data)
prediction=model.predict(ip_data_scaled)
prediction_proba=prediction[0][0]

if st.button('Predict'):
  if prediction_proba>0.5:
    st.write('Churned')
  else:
    st.write('Not Churned')
