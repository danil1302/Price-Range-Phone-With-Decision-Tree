import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from pandas_profiling import ProfileReport
from streamlit_pandas_profiling import st_profile_report
import seaborn as sns
import pickle

#Import Model
klf_gini = pickle.load(open('DecisionTree.pkl','rb'))

#Load Datasets
data = pd.read_excel('Price Range Phone Dataset.xlsx')

st.title('Aplikasi Kisaran Prediksi Harga Telepon')

html_layout1 = """
<br>
<div style="background-color:#FF9BD2 ; padding:2px">
<h2 style="color:white;text-align:center;font-size:35px"><b>Prediction Phone</b></h2>
</div>
<br>
<br>
"""
st.markdown(html_layout1,unsafe_allow_html=True)
activities = ['Decision Tree','Model Lain']
option = st.sidebar.selectbox('Pilihan mu ?',activities)
st.sidebar.header('Data Pasien')

if st.checkbox("Tentang Dataset"):
    html_layout2 ="""
    <br>
    <p>Ini adalah Dataset Price Range Phone</p>
    """
    st.markdown(html_layout2,unsafe_allow_html=True)
    st.subheader('Dataset')
    st.write(data.head(10))
    st.subheader('Describe dataset')
    st.write(data.describe())

sns.set_style('darkgrid')

if st.checkbox('EDa'):
    pr =ProfileReport(data,explorative=True)
    st.header('**Input Dataframe**')
    st.write(data)
    st.write('---')
    st.header('**Profiling Report**')
    st_profile_report(pr)

#train test split
X = data.drop('price_range',axis=1)
y = data['price_range']
X_train, X_test,y_train,y_test = train_test_split(X,y,test_size=0.20,random_state=42)

#Training Data
if st.checkbox('Train-Test Dataset'):
    st.subheader('X_train')
    st.write(X_train.head())
    st.write(X_train.shape)
    st.subheader("y_train")
    st.write(y_train.head())
    st.write(y_train.shape)
    st.subheader('X_test')
    st.write(X_test.shape)
    st.subheader('y_test')
    st.write(y_test.head())
    st.write(y_test.shape)

def user_report():
    battery_power = st.sidebar.slider('Baterry Power',0,2000,876)
    blue = st.sidebar.slider('Blue',0,3,1)
    clock_speed = st.sidebar.slider('Clock Speed',0.5,3.0,0.7)
    dual_sim = st.sidebar.slider('Dual SIM',0,3,1)
    fc = st.sidebar.slider('Front Camera',0,20,5)
    four_g = st.sidebar.slider('4G',0,3,1)
    int_memory = st.sidebar.slider('Internal Memory', 0,60,27)
    m_dep = st.sidebar.slider('Mobile Departement',0.1,1.0,0.4)
    mobile_wt = st.sidebar.slider('Mobile WT',0,200,45)
    n_cores = st.sidebar.slider('Cores',0,10,4)
    pc = st.sidebar.slider('PC',0,25,12)
    px_height = st.sidebar.slider('Pixel Height',0,1000,37)
    px_width = st.sidebar.slider('Pixel Width',0,1000,120)
    ram = st.sidebar.slider('RAM',0,4000,678)
    sc_h = st.sidebar.slider('Screen Height', 0,20,12)
    sc_w = st.sidebar.slider('Screen Width',0,20,11)
    talk_time = st.sidebar.slider('Talk Time',0,30,14)
    three_g = st.sidebar.slider('3G',0,3,1)
    touch_screen = st.sidebar.slider('Touch Screen',0,3,1)
    wifi = st.sidebar.slider('Wi-Fi',0,3,1)
    
    user_report_data = {
       'battery_power':battery_power,
        'blue':blue,
        'clock_speed':clock_speed,
        'dual_sim':dual_sim,
        'fc':fc,
        'four_g':four_g,
        'int_memory':int_memory,
        'm_dep':m_dep,
        'mobile_wt':mobile_wt,
        'n_cores':n_cores,
        'pc':pc,
        'px_height':px_height,
        'px_width':px_width,
        'ram':ram,
        'sc_h':sc_h,
        'sc_w':sc_w,
        'talk_time':talk_time,
        'three_g':three_g,
        'touch_screen':touch_screen,
        'wifi':wifi
    }
    report_data = pd.DataFrame(user_report_data,index=[0])
    return report_data

#Data Phone
user_data = user_report()
st.subheader('Data Phone')
st.write(user_data)

user_result = klf_gini.predict(user_data)
dectree_score = accuracy_score(y_test,klf_gini.predict(X_test))

#output
st.subheader('Hasilnya adalah : ')
output=''
if user_result[0]==0:
    output='Bagus'
else:
    output ='Sangat Bagus'
st.title(output)
st.subheader('Model yang digunakan : \n'+option)
st.subheader('Accuracy : ')
st.write(str(dectree_score*100)+'%')