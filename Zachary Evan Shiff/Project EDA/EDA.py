#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np


# In[ ]:





# In[3]:


df = pd.read_csv('diabetes_data_upload.csv')


# In[4]:


df.head(10)


# In[5]:


df[df['class'] == 'Negative'].head(10)


# In[6]:


df[df['class'] == 'Positive'].head(10)


# In[7]:


df.columns


# In[8]:


#researching different columns and the symptoms of diabetes, what makes a person diabetic
Polyuria = your body makes more pee than normal. Adults usually make about 3 liters of urine per day. But with polyuria, you could make up to 15 liters per day

Polydipsia = Abnormally great thirst because of Polyuria

Visual Blurring = Visual symptom that makes it harder to see clear and it can

Partial Parasis = a condition in which some muscle movment is harder to control and acurse due to nerve damage from multiple ocasions. 

alopecia = Hair loss targeting the head a face. anyone can have alopecia and effects everyone. Could be at any age but most common between 10-30. Notmally happens in patches. 

delayed healing = Wound healing can be delayed by systemic factors that bear little or no direct relation to the location of the wound itself. Seems to be common in Positive patients and uncommon in Negative patients.


# In[ ]:





# In[9]:


results = []
for col in df.columns:
    missing_or_not = df[col].isnull().values.any()
    print(col + ":" + str(missing_or_not))


# In[10]:


#create a function that takes in either yes or no as an argument, yes and no are going to be in a string format, if yes then return 1, if no return 0
def convert(x):
    if x == 'Yes':
        return 1
    else:
        return 0
convert('Yes')


# In[11]:


import plotly.express as px #importing plotly


# In[12]:


df = pd.read_csv('diabetes_data_upload.csv')


# In[13]:


df_class = df['class'].value_counts().to_frame().reset_index()


# In[14]:


df_2 = df['class'].value_counts().to_frame().reset_index()
df_2


# In[15]:


df_2


# In[ ]:





# In[16]:


fig = px.pie(df_2, values = 'class', names = 'index', title = "Positive vs. Negative")
fig.show()


# In[17]:



df_positive = df[df['class'] == 'Positive']


# In[18]:


df_2 = df_positive['Gender'].value_counts().to_frame().reset_index()
df_2


# In[19]:



fig = px.bar(df_2, x="index", y="Gender", color="index", title="Positive Male and Female")
fig.show()


# In[20]:


df_negative = df[df['class'] == 'Negative']


# In[21]:


df_3 = df_negative['Gender'].value_counts().to_frame().reset_index()
df_3


# In[22]:


fig = px.bar(df_3, x="index", y="Gender", color="index", title="Negative Male and Female")
fig.show()


# In[23]:


def convert(x):
    if x == 'Yes'or x == 'Male' or x == 'Positive':
        return 1
    else:
        return 0


# In[24]:


for col in df.columns:
    if col != 'Age':
        df[col] = df[col].apply(convert)


# In[25]:


df.head()


# In[26]:


Drop_Columns = ['class']
x = df.drop(Drop_Columns,axis=1)
y = df[['class']]


# In[27]:


x


# In[28]:


y


# In[29]:


df_imshow = df.groupby(by ='Gender').mean().reset_index()
df_imshow


# In[ ]:





# In[30]:


Drop_Columns = ['class', 'Age', 'weakness', 'Genital thrush', 'Itching', 'delayed healing', 'muscle stiffness', 'Obesity']
x = df.drop(Drop_Columns,axis=1)
y = df['class']


# In[31]:


x


# In[32]:


x.corrwith(y)


# In[33]:


corr = df.corr()
corr['class']


# In[34]:


fig = px.imshow(corr, labels=dict(x = "Features", y = "Features" , color = "Correlation"))
fig.update_xaxes(side="top")
fig.show()


# In[43]:


fig = px.bar(corr, x = 'class', color = 'class', title = "Correlation Between Features and Class")
fig.write_html("file.html")
fig.show()


# In[40]:


# train test split


# In[41]:


import sklearn
from sklearn.model_selection import train_test_split


# In[42]:


x_train, x_test ,y_train, y_test = train_test_split(x, y)


# In[43]:


print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)


# In[44]:


#Machine Learning Model


# In[ ]:





# In[45]:


from sklearn.preprocessing import StandardScaler


# In[46]:


ss = StandardScaler()
x_train = ss.fit_transform(x_train)
x_test = ss.transform(x_test)


# In[ ]:





# In[47]:


# Random forest: forest classifiers have to be fitted with two arrays: a sparse or dense array X of shape, and an array Y of shape
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
RFC =  RandomForestClassifier
parameters = {'n_estimators' : [10, 8, 12],'max_features':('sqrt', 'log2', None)}


# In[48]:


clf = GridSearchCV(RandomForestClassifier(), parameters)
clf.fit(x_train, y_train)


# In[49]:


clf.best_params_


# In[50]:


y_hat = clf.predict(x_test)


# In[51]:


y_hat


# In[52]:


np.array(y_test)


# In[53]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.heatmap(confusion_matrix(y_test, y_hat), annot=True, fmt='g',)


# In[54]:


#MSE
total_squared_error = (np.sum((y_test - y_hat)**2)) #get the sum of all the errors (error = what we want (y_test) - what we predicted (y_hat))
mean_squared_error = total_squared_error/len(y_test) #divide this by how many rows/observations we have 
print(mean_squared_error)


# In[55]:


#Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_hat)


# In[56]:


#Recall
from sklearn.metrics import recall_score
recall_score(y_test, y_hat)


# In[57]:


#Precision
from sklearn.metrics import precision_score
precision_score(y_test, y_hat)


# In[58]:


#Decision tress: Decision Trees are a non-parametric supervised learning method used for classification and regression. The model can predict the value of a target variable by learning simple rules. The more data, the more complex the decision rules and the fitter the model.
from sklearn import tree
from sklearn.model_selection import GridSearchCV
parameters = {'criterion' :('gini', 'entropy', 'log_loss'), 'splitter' :('best', 'random')}


# In[59]:


clf = GridSearchCV(tree.DecisionTreeClassifier(), parameters)
clf.fit(x_train, y_train)


# In[60]:


clf.best_params_


# In[61]:


y_hat = clf.predict(x_test)


# In[62]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.heatmap(confusion_matrix(y_test, y_hat), annot=True, fmt='g',)


# In[63]:


#MSE
total_squared_error = (np.sum((y_test - y_hat)**2)) #get the sum of all the errors (error = what we want (y_test) - what we predicted (y_hat))
mean_squared_error = total_squared_error/len(y_test) #divide this by how many rows/observations we have 
print(mean_squared_error)


# In[64]:


#Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_hat)


# In[65]:


#Recall
from sklearn.metrics import recall_score
recall_score(y_test, y_hat)


# In[66]:


#Precision
from sklearn.metrics import precision_score
precision_score(y_test, y_hat)


# In[67]:


from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import GaussianNB
parameters = {'var_smoothing' : [10, 6, 8], 'priors': [None]}


# In[68]:


#naive bayes: Naive Bayes methods are a set of supervised learning algorithms based on applying Bayes’ theorem.
gnb = GridSearchCV(GaussianNB(),parameters)
gnb.fit(x_train, y_train)


# In[69]:


gnb.best_params_


# In[ ]:





# In[70]:


y_hat = gnb.predict(x_test)


# In[71]:


from sklearn.metrics import confusion_matrix
import seaborn as sns
sns.heatmap(confusion_matrix(y_test, y_hat), annot=True, fmt='g',)


# In[72]:


#MSE
total_squared_error = (np.sum((y_test - y_hat)**2)) #get the sum of all the errors (error = what we want (y_test) - what we predicted (y_hat))
mean_squared_error = total_squared_error/len(y_test) #divide this by how many rows/observations we have 
print(mean_squared_error)


# In[73]:


#Accuracy
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_hat)


# In[74]:


#Recall
from sklearn.metrics import recall_score
recall_score(y_test, y_hat)


# In[75]:


#Precision
from sklearn.metrics import precision_score
precision_score(y_test, y_hat)


# In[76]:


import pandas as pd
import numpy as np


# In[77]:


Drop_Columns = [ 'Age', 'weakness', 'Genital thrush', 'Itching', 'delayed healing', 'muscle stiffness', 'Obesity']
X2 = df.drop(Drop_Columns,axis=1)


# In[78]:


X2


# In[79]:


X2.columns


# In[80]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots


# In[ ]:





# In[ ]:





# In[116]:


import plotly.graph_objects as go
from plotly.subplots import make_subplots

fig = make_subplots(rows=3, cols=3)

fig.append_trace(go.Histogram(name="Gender", x=X2['Gender'], y=X2['class']),row=1, col=1)
fig.append_trace(go.Histogram(name='Irritability', x=df['Irritability'], y=df['class']),row=1, col=2)
fig.append_trace(go.Histogram(name = 'partial paresis', x=df['partial paresis'], y=df['class']),row=1, col=3)
fig.append_trace(go.Histogram(name = 'sudden weight loss', x=df['sudden weight loss'], y=df['class']), row=2, col=1)
fig.append_trace(go.Histogram(name = 'Polyphagia', x=df['Polyphagia'], y=df['class']),row=2, col=2)
fig.append_trace(go.Histogram(name = 'Polyuria', x=df['Polyuria'], y=df['class']),row=2, col=3)
fig.append_trace(go.Histogram(name = 'Polydipsia', x=df['Polydipsia'], y=df['class']),row=3, col=1)
fig.append_trace(go.Histogram(name = 'visual blurring', x=df['visual blurring'], y=df['class']), row=3, col=2)
fig.append_trace(go.Histogram(name = 'Alopecia', x=df['Alopecia'], y=df['class']),row=3, col=3)


fig.update_layout(legend_title_text='Features', title_text='Positive or Negative Features', title_x=0.5)
fig.show()


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




