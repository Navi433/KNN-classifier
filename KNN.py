#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
get_ipython().run_line_magic('matplotlib', 'inline')


# In[25]:


df=pd.read_csv('C:\\Users\Admin\Desktop\\ClassifiedData',index_col=0)


# In[27]:


df.head()


# In[30]:


df.isnull().sum()


# In[55]:


X=df.iloc[:,0:10]
y=df.iloc[:,10]


# In[33]:


sns.countplot(df['TARGET CLASS'])


# In[34]:


from sklearn.preprocessing import StandardScaler


# In[35]:


sc=StandardScaler()


# In[62]:


#X=sc.fit_transform(X)


# In[63]:


#X


# In[66]:


scaledfeatures=sc.fit_transform(df.drop(['TARGET CLASS'],axis=1))


# In[67]:


scaledfeatures


# In[72]:


dffeatures=pd.DataFrame(scaledfeatures,columns=df.columns[:-1])


# In[73]:


dffeatures


# In[74]:


from sklearn.model_selection import train_test_split


# In[75]:


X_train, X_test, y_train, y_test = train_test_split(scaledfeatures, df['TARGET CLASS'], test_size=0.30)


# In[78]:


from sklearn.neighbors import KNeighborsClassifier


# In[79]:


knn= KNeighborsClassifier(n_neighbors=1)


# In[80]:


knn.fit(X_train,y_train)


# In[82]:


ypred=knn.predict(X_test)


# In[112]:


ypred


# In[89]:


from sklearn.metrics import confusion_matrix,classification_report


# In[85]:


confusion_matrix(y_test,ypred)


# In[86]:


from sklearn.metrics import accuracy_score


# In[87]:


acc=accuracy_score(y_test,ypred)


# In[88]:


acc


# In[90]:


print(classification_report(y_test,ypred))


# In[102]:


error_rate = []

# Calculating error for K values between 1 and 40
for i in range(1, 40):
    knn = KNeighborsClassifier(n_neighbors=i)
    knn.fit(X_train, y_train)
    pred_i = knn.predict(X_test)
    error_rate.append(np.mean(pred_i != y_test))


# In[110]:


error_rate


# In[103]:


plt.figure(figsize=(12, 6))
plt.plot(range(1, 40), error_rate, color='red', linestyle='dashed', marker='o',
         markerfacecolor='blue', markersize=10)
plt.title('Error Rate K Value')
plt.xlabel('K Value')
plt.ylabel('Mean Error')


# In[104]:


knn= KNeighborsClassifier(n_neighbors=23)
knn.fit(X_train,y_train)


# In[105]:


ypred2=knn.predict(X_test)


# In[106]:


ypred2


# In[107]:


confusion_matrix(y_test,ypred2)


# In[108]:


acc=accuracy_score(y_test,ypred2)


# In[109]:


acc


# In[ ]:




