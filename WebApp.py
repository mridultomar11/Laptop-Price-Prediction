#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
# import seaborn as sns

import streamlit as st
from sklearn import *
import pickle


# In[ ]:


dt = pickle.load(open('DTRegressor.pkl','rb'))
rf = pickle.load(open('RFRegressor.pkl','rb'))
gb = pickle.load(open('GBRegressor.pkl','rb'))
ad = pickle.load(open('AdaBoost.pkl','rb'))


# In[ ]:


# def load_object(name):
#     pickle_obj = open(f"{name}.pck","rb")
#     obj = pickle.load(pickle_obj)
#     return obj

# lb = load_object("LabelEncoder")


# In[ ]:


st.title("Laptop Price Predicition")
html_temp = """
    <div style="background-color:darkblue ;padding:10px">
    <h2 style="color:white;text-align:center;">Price Predicting Models</h2>
    </div>
"""
st.markdown(html_temp, unsafe_allow_html=True)

activities = ['DecisionTree','RandomForest','GradientBoost','AdaBoost']
option = st.sidebar.selectbox('Which model would you like to use?',activities)
st.subheader(option)

l = ['LENOVO', 'ASUS', 'MI', 'HP', 'APPLE', 'DELL', 'MSI', 'INFINIX',
     'REALME', 'ACER', 'SAMSUNG', 'GIGABYTE', 'MICROSOFT', 'SONY',
     'PRIMEBOOK', 'AVITA', 'NOKIA', 'LG']
brand = st.selectbox('Select Brand Name', l)

l1 = [ 8.,  4., 16., 32.]
ram = st.selectbox('Select Ram Capacity', l1)

l2 = ['DDR4', 'DDR5', 'LPDDR5', 'LPDDR4X', 'Unified', 'LPDDR4', 'GDDR6', 'GDDR5', 'LPDDR3', 'DDR3']
ram_type = st.selectbox('Select Ram Type', l2)

l3 = [ 256.,  512., 1024.,    0., 2048.,  128., 4096.,    8.]
ssd_capacity = st.selectbox('Select SSD Capacity', l3)

l4 = [   0., 1024.,  512., 2048.]
hdd_capacity = st.selectbox('Select HDD Capacity', l4)

l5 = [ 0., 32., 64.]
emmc_storage = st.selectbox('Select EMMC Capacity', l5)

l6 = [39.62, 35.56, 33.78, 43.94, 34.54, 29.46, 45.72, 40.89, 41.15,
       40.64, 43.18, 38.  , 36.83, 34.04, 38.86, 36.07, 38.1 , 35.  ,
       42.16, 34.29, 35.81, 33.02]
display_size = st.selectbox('Select Display Size(in cm)', l6)

l7 = ['Intel Celeron Dual Core Processor',
       'Intel Core i5 Processor (10th Gen)',
       'Intel Core i3 Processor (11th Gen)',
       'Intel Core i5 Processor (11th Gen)', 'Apple M1 Processor',
       'AMD Ryzen 5 Hexa Core Processor',
       'Intel Core i5 Processor (12th Gen)',
       'AMD Athlon Dual Core Processor',
       'AMD Ryzen 7 Octa Core Processor',
       'AMD Ryzen 5 Quad Core Processor',
       'AMD Ryzen 3 Quad Core Processor',
       'Intel Core i9 Processor (12th Gen)',
       'Intel Core i7 Processor (12th Gen)', 'Apple M2 Processor',
       'Intel Core i7 Processor (11th Gen)', 'Apple M2 Pro Processor',
       'Intel Core i7 Processor (13th Gen)',
       'Intel Core i3 Processor (12th Gen)',
       'Intel Core i9 Processor (11th Gen)',
       'Intel Core i9 Processor (13th Gen)',
       'AMD Ryzen 3 Dual Core Processor', 'Apple M1 Max Processor',
       'Apple M2 Max Processor', 'AMD Dual Core Processor',
       'Intel Pentium Silver Processor',
       'Intel Core i3 Processor (10th Gen)',
       'MediaTek MediaTek Kompanio 500 Processor',
       'Intel Core i3 Processor (13th Gen)',
       'AMD Ryzen 9 Octa Core Processor',
       'Intel Pentium Quad Core Processor',
       'Intel Core i5 Processor (13th Gen)',
       'Intel Celeron Quad Core Processor',
       'AMD Ryzen 7 Hexa Core Processor', 'Apple M1 Pro Processor',
       'Intel Core i7 Processor (10th Gen)',
       'Intel Core i7 Processor (7th Gen)',
       'AMD Ryzen 7 Quad Core Processor',
       'Intel Core i9 Processor (8th Gen)',
       'Intel Core i9 Processor (10th Gen)',
       'Intel Core i7 Processor (8th Gen)',
       'Intel Core i7 Processor (6th Gen)',
       'AMD Ryzen 5 Dual Core Processor',
       'Intel Core i5 Processor (8th Gen)',
       'Intel Core i5 Processor (9th Gen)',
       'Intel Core i3 Processor (7th Gen)',
       'MediaTek MediaTek MT8788 Processor',
       'Intel Core i5 Processor (7th Gen)',
       'Intel Core i5 Processor (4th Gen)']
processor = st.selectbox('Processor', l7)

ratings = st.slider('Choose its rating', 0.0, 4.8)

encoding_dict_processor = {'Intel Celeron Dual Core Processor': 1, 'Intel Core i5 Processor (10th Gen)': 2, 'Intel Core i3 Processor (11th Gen)': 3, 'Intel Core i5 Processor (11th Gen)': 4, 'Apple M1 Processor': 5, 'AMD Ryzen 5 Hexa Core Processor': 6, 'Intel Core i5 Processor (12th Gen)': 7, 'AMD Athlon Dual Core Processor': 8, 'AMD Ryzen 7 Octa Core Processor': 9, 'AMD Ryzen 5 Quad Core Processor': 10, 'AMD Ryzen 3 Quad Core Processor': 11, 'Intel Core i9 Processor (12th Gen)': 12, 'Intel Core i7 Processor (12th Gen)': 13, 'Apple M2 Processor': 14, 'Intel Core i7 Processor (11th Gen)': 15, 'Apple M2 Pro Processor': 16, 'Intel Core i7 Processor (13th Gen)': 17, 'Intel Core i3 Processor (12th Gen)': 18, 'Intel Core i9 Processor (11th Gen)': 19, 'Intel Core i9 Processor (13th Gen)': 20, 'AMD Ryzen 3 Dual Core Processor': 21, 'Apple M1 Max Processor': 22, 'Apple M2 Max Processor': 23, 'AMD Dual Core Processor': 24, 'Intel Pentium Silver Processor': 25, 'Intel Core i3 Processor (10th Gen)': 26, 'MediaTek MediaTek Kompanio 500 Processor': 27, 'Intel Core i3 Processor (13th Gen)': 28, 'AMD Ryzen 9 Octa Core Processor': 29, 'Intel Pentium Quad Core Processor': 30, 'Intel Core i5 Processor (13th Gen)': 31, 'Intel Celeron Quad Core Processor': 32, 'AMD Ryzen 7 Hexa Core Processor': 33, 'Apple M1 Pro Processor': 34, 'Intel Core i7 Processor (10th Gen)': 35, 'Intel Core i7 Processor (7th Gen)': 36, 'AMD Ryzen 7 Quad Core Processor': 37, 'Intel Core i9 Processor (8th Gen)': 38, 'Intel Core i9 Processor (10th Gen)': 39, 'Intel Core i7 Processor (8th Gen)': 40, 'Intel Core i7 Processor (6th Gen)': 41, 'AMD Ryzen 5 Dual Core Processor': 42, 'Intel Core i5 Processor (8th Gen)': 43, 'Intel Core i5 Processor (9th Gen)': 44, 'Intel Core i3 Processor (7th Gen)': 45, 'MediaTek MediaTek MT8788 Processor': 46, 'Intel Core i5 Processor (7th Gen)': 47, 'Intel Core i5 Processor (4th Gen)': 48}
encodind_dict_brand = {'LENOVO': 1, 'ASUS': 2, 'MI': 3, 'HP': 4, 'APPLE': 5, 'DELL': 6, 'MSI': 7, 'INFINIX': 8, 'REALME': 9, 'ACER': 10, 'SAMSUNG': 11, 'GIGABYTE': 12, 'MICROSOFT': 13, 'SONY': 14, 'PRIMEBOOK': 15, 'AVITA': 16, 'NOKIA': 17, 'LG': 18}
encoding_dict_ramtype = {'DDR4': 1, 'DDR5': 2, 'LPDDR5': 3, 'LPDDR4X': 4, 'Unified': 5, 'LPDDR4': 6, 'GDDR6': 7, 'GDDR5': 8, 'LPDDR3': 9, 'DDR3': 10}

a = encoding_dict_processor.get(processor)
b = encodind_dict_brand.get(brand)
c = encoding_dict_ramtype.get(ram_type)

inputs = [[ratings ,a , b, ssd_capacity, 
          ram, c, display_size,
          emmc_storage, hdd_capacity]]

# inputs = Inputs.reshape([1,9])

if st.button('Predict'):
    if option=='DecisonTree':
        st.success(np.exp(dt.predict(inputs)))
    elif option=='RandomForest':
        st.success(np.exp(rf.predict(inputs)))
    elif option=='GradientBoost':
        st.success(np.exp(gb.predict(inputs)))
    else:
        st.success(np.exp(ad.predict(inputs)))



# strealit run iris_webapp.py
# Ctrl + C 


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




