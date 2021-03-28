import streamlit as st
import pandas as pd
import numpy as np
from pycaret.classification import load_model, predict_model
model=load_model('final_model')
def predict(model,input_df):
  predictions_df=predict_model(estimator=model,data=input_df)
  predictions=predictions_df['Label'][0]
  return predictions
def run():
    from PIL import Image
    image=Image.open('E:\\bank-agencies (1).jpg')
    image_office=Image.open('E:\\download.jpg')
    st.image(image,use_column_width=False)
  add_selectbox=st.sidebar.selectbox(
      "How would like to get the predictions?",
      ('Realtime','Batch'))
  st.sidebar.info('This app predicts the defaulter nature of an applicant')
  st.title('Bank defaulter')
  if add_selectbox=='Realtime':
    age=st.number_input('age', min_value=0, max_value=100, value=0)
    ed=st.selectbox('ed',['1','2','3','4','5'])
    employ=st.number_input('employ', min_value=0, max_value=100, value=0)
    address=st.number_input('address',min_value=0, max_value=100, value=0)
    income=st.number_input('income', min_value=0, max_value=1000, value=0)
    debtinc=st.number_input('debtinc',min_value=0, max_value=1000, value=0)
    creddebt=st.number_input('creddebt',min_value=0, max_value=1000, value=0)
    othdebt=st.number_input('othdebt',min_value=0, max_value=1000, value=0)
    output=''
    input_dict={'age':age,'ed':ed,'employ':employ,'address':address,'income':income,
                'debtinc':debtinc,'creddebt':creddebt,'othdebt':othdebt}
    input_df=pd.DataFrame([input_dict])
    if st.button("predict"):
      output=predict(model=model,input_df=input_df)
      output=str(output)
    st.success('the output is{}'.format(output))
  if add_selectbox=='Batch':
     file_upload=st.file_uploader("Upload the csv file", type=['csv'])
     if file_upload is not None:
       data=pd.read_csv(file_upload)
       predictions=predict_model(estimator=model, data=data)
       st.write(predictions)

run()
