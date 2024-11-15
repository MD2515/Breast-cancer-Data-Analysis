#!/usr/bin/env python
# coding: utf-8

# # SupportVectorMachine

# In[1]:


import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


# In[2]:


data = pd.read_csv('wdbc.csv', names=['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst', 'Unnamed'])


# In[3]:


data.head()


# In[4]:


data.drop(["Unnamed","id"],axis=1, inplace=True)


# In[5]:


data.head()


# In[6]:


M = data[data.diagnosis=="M"]
B = data[data.diagnosis=="B"]


# In[7]:


plt.scatter(M.radius_mean,M.texture_mean,color="red",label="malignant") 
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="benign")
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()


# In[8]:


data.diagnosis = [1 if each=="M" else 0 for each in data.diagnosis] 

y = data.diagnosis.values 
x_data = data.drop(["diagnosis"],axis=1) 


# In[9]:


x = (x_data - np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[10]:


x.head()


# In[11]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=1)


# In[12]:


from sklearn.svm import SVC
svc= SVC(random_state=42)
svc.fit(x_train,y_train)


# In[13]:


svc.score(x_test,y_test)


# In[ ]:





# # Decision Tree Classification

# In[15]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


# In[16]:


data = pd.read_csv('wdbc.csv', names=['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst', 'Unnamed'])
data.head()


# In[17]:


data.drop(["Unnamed","id"],axis=1,inplace=True)


# In[18]:


data.head()


# In[19]:


data.describe().T


# In[20]:


M = df[df["diagnosis"]=="M"]
B = df[df["diagnosis"]=="B"]


# In[21]:


plt.scatter(M.radius_mean,M.texture_mean, color = "red", label="malignant")
plt.scatter(B.radius_mean,B.texture_mean, color = "green", label="benign")
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()


# In[24]:


data.diagnosis = [1 if each == "M" else 0 for each in data.diagnosis]

X = data.iloc[:,1:].values
y = data.diagnosis.values


# In[25]:


X = ((X - np.min(X))/(np.max(X)-np.min(X)))


# In[26]:


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


# In[27]:


dtc = DecisionTreeClassifier()
dtc.fit(X_train,y_train)
prediction = dtc.predict(X_test)


# In[28]:


print(f"{dtc} , score: {dtc.score(X_test,y_test)}")


# In[ ]:





# # NaiveBayes

# In[29]:


import pandas as pd
import numpy as np 
import matplotlib.pyplot as plt


# In[30]:


data = pd.read_csv('wdbc.csv', names=['id', 'diagnosis', 'radius_mean', 'texture_mean', 'perimeter_mean', 'area_mean', 'smoothness_mean', 'compactness_mean', 'concavity_mean', 'concave_points_mean', 'symmetry_mean', 'fractal_dimension_mean', 'radius_se', 'texture_se', 'perimeter_se', 'area_se', 'smoothness_se', 'compactness_se', 'concavity_se', 'concave points_se', 'symmetry_se', 'fractal_dimension_se', 'radius_worst', 'texture_worst', 'perimeter_worst', 'area_worst', 'smoothness_worst', 'compactness_worst', 'concavity_worst', 'concave points_worst', 'symmetry_worst', 'fractal_dimension_worst', 'Unnamed'])

data.head()
data.drop(["id","Unnamed"],axis=1,inplace=True)


# In[31]:


data.tail()


# In[32]:


M = data[data.diagnosis == "M"] 
B = data[data.diagnosis == "B"]


# In[33]:


plt.scatter(M.radius_mean,M.texture_mean,color="red",label="malignant")
plt.scatter(B.radius_mean,B.texture_mean,color="green",label="benign")
plt.xlabel("radius_mean")
plt.ylabel("texture_mean")
plt.legend()
plt.show()


# In[34]:


data.diagnosis = [1 if each =="M" else 0 for each in data.diagnosis]

x_data= data.drop(["diagnosis"],axis=1)
y= data.diagnosis.values


# In[35]:


x=(x_data-np.min(x_data))/(np.max(x_data)-np.min(x_data))


# In[36]:


from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=1)


# In[37]:


from sklearn.naive_bayes import GaussianNB
nb = GaussianNB()
nb.fit(x_train,y_train)


# In[38]:


print("accuracy of svm algorithm: ",nb.score(x_test,y_test))


# In[ ]:




