#!/usr/bin/env python
# coding: utf-8

import tensorflow as tf
import numpy as np
import math
import streamlit as st




x1_data=[]
y1_data=[]
theta1_data=[]
theta2_data=[]




model1 = tf.keras.models.load_model('parameters_model1.h5')



model2 = tf.keras.models.load_model('parameters_model2.h5')


t =np.zeros(2)
t[0]=1
t[1] =2
c = tf.convert_to_tensor(t , dtype =float)
d =tf.reshape(c, (-1,2)) 
f = model1.predict(d)
f


a =math.pi/180

def x_data(theta_1,theta_2):
  theta1 = a*theta_1
  theta2 = a*theta_2
  x =  math.cos(theta1) + math.cos(theta1 + theta2)
  return float(x)

def y_data(theta_1,theta_2):
  theta1 = a*theta_1
  theta2 = a*theta_2
  y = math.sin(theta1) + math.sin(theta1 + theta2)
  return float(y)



st.title("MARS Open Project")
st.header("Modelling the inverse kinematics of a 2-link manipulator using deep neural networks.")

st.image('image2.jpeg',width=1000)

nav = st.sidebar.radio("Navigation",["Prediction","Contribute"])
if nav =="Prediction":
    st.header("input constraints -: (x,y)")
    st.latex(r''' x^2 +y^2 <=4''')
    st.header("predict angles")
    valx = st.number_input("Enter x-coordinate", -2.0000 ,2.00000, step = 0.0001 ,format= '%.3f')
    valy = st.number_input("Enter y-coordinate", -2.00000,2.00000, step = 0.0001, format= '%.3f')
    input1_1 = np.zeros(2)
    input1_1[0]=valx
    input1_1[1]=valy
    input1= tf.convert_to_tensor(input1_1 , dtype =np.float32)
    input1 = tf.reshape(input1 ,(-1,2))
   
    pred = model1.predict(input1)
    pred2 =model2.predict(input1)
    if st.button("Predict"):
      if (valx**2 + valy**2 <= 4):
        st.success("hello")
        st.success(f"Your predicted outputs are theta1 = {pred.item()} and theta2 = {pred2.item()}")
        st.success(f'Actual end affector positions x = {x_data(pred,pred2)} , y = {y_data(pred ,pred2)}')
      else :
        st.success('Please give correct inputs, inputs out of range')  

if nav =="Contribute":
    st.header("contribute to our dataset")
    x = st.number_input("x-coordinate " , -2.000 , 2.000)
    y = st.number_input("y-coordinate" , -2.000, 2.000)
    theta1 = st.number_input("theta1 ",0.00 , 360.00)
    theta2 = st.number_input("theta2",0.00, 180.00)
    if st.button("submit"):
        x1_data.append(x)
        y1_data.append(y)
        theta1_data.append(theta1)
        theta2_data.append(theta2)
        st.success("submitted")
        st.success("thanks for your contribution, your contribution is valuable to us")

st.markdown(""" made with :heart: by Aditya Raj and Rajib

""",True)







