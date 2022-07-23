#!/usr/bin/env python
# coding: utf-8

# ### IMPORTING REQUIRED LIBRARIES AND MODULES :- 

# In[58]:


import tensorflow as tf
import numpy as np

import math
import csv
from tensorflow.keras.models import Sequential as seq
from tensorflow.keras.layers import Dense as den
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
from datetime import datetime
tf.random.set_seed(42)


# ### CREATING DATASET(GENERATING ANGLES)

# In[59]:


'''
a1=[]
b1=[]
x11=-0.25

p = -1
for i in range(1440):
  x11 =x11+0.25
  y11=-0.25
  for j in range (720):
    y11 =y11+0.25
    a1.append(x11)
    b1.append(y11)

x =len(a1)
s = np.zeros((x,2), dtype = np.float32)
for i in range(x):
  p = p +1
  
  s[i][0] = a1[p]
  s[i][1] = b1[p]
  


fields = [ 'THETA1' ,'Theta2']

filename = '/content/angles_larger_data.csv'
with open(filename, 'w') as csvfile: 
  csvwriter = csv.writer(csvfile)
  csvwriter.writerow(fields)
  csvwriter.writerows(s)
'''


# In[60]:


angle = np.loadtxt('angles_larger_data.csv', delimiter=',', dtype=np.float32 , skiprows=1)


# In[61]:


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

x_1 = np.zeros(1036800)
y_1 =np.zeros(1036800)

s = np.zeros((1036800,5), dtype = np.float32)

for k in range(1036800):
  
    x_1[k] = x_data(angle[k][0],angle[k][1])
    
    y_1[k] = y_data(angle[k][0],angle[k][1])

p = -1
for i in range(1036800):
  p = p +1
  s[i][0] = p
  s[i][1] = x_1[p]
  s[i][2] = y_1[p]
  s[i][3] = angle[p][0]
  s[i][4] = angle[p][1]


# In[62]:


s =tf.constant(s)
data1 = s
train_input = data1[:836800,1:3]
train_output1 = data1[:836800 ,3:4 ]
train_output2 = data1[:836800 ,4: ]
val_input = data1[836800:936800,1:3]
val_output1 = data1[836800:936800 ,3:4 ]
val_output2 = data1[836800:936800 ,4: ]
test_input = data1[936800:,1:3]
test_output1 = data1[936800: ,3:4 ]
test_output2 = data1[936800: ,4: ]
test_input_short = data1[836870:836880,1:3]
test_output_short1 = data1[836870:836880, 3:4]
test_output_short2 = data1[836870:836880, 4:]


# ### Defining Neural Net for theta1

# In[87]:


model1 = seq([den(2 ,activation='relu'),
              den(16,activation ='relu'),
              den(32 ,activation ='relu'),
              den(48 ,activation ='relu'),
              den(32 ,activation ='relu'),
              den(16 ,activation ='relu'),
              den(1,activation ='relu')])
model1.compile(optimizer ='adam', loss ='mae')


# ### Training and validatating  model1

# In[88]:


model1.fit(train_input, train_output1,
          validation_data=(val_input,val_output1) ,epochs= 250 )


# #### Evaluating the performance of model1

# In[89]:


model1.evaluate(test_input, test_output1)


# #### predicting and comparing with a small data

# In[75]:


test = model1.predict(test_input_short)
print(test)
print(test_output_short1)


# ### Training and validatating  model2

# In[76]:


model2 = seq([den(2 ,activation='relu'),
              den(16,activation ='relu'),
              den(32 ,activation ='relu'),
              den(32 ,activation ='relu'),
              den(32 ,activation ='relu'),
              den(16 ,activation ='relu'),
              den(1,activation ='relu')])
model2.compile(optimizer ='adam', loss ='mae')


# #### Training and validating model2

# In[53]:


model2.fit(train_input, train_output2,
          validation_data=(val_input,val_output2) ,epochs= 120)


# #### evaluating the model2

# In[54]:


model2.evaluate(test_input, test_output2)


# #### predicting the outcomes of model2 over a small data
# 

# In[55]:


test = model2.predict(test_input_short)
print(test)
print(test_output_short2)


# 
# ### saving the models

# In[158]:


model1.save('parameters_model1.h5')
model2.save('parameters_model2.h5')


# In[ ]:




