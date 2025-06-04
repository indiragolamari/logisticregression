#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, roc_curve
from sklearn.preprocessing import LabelEncoder


# In[3]:


train_data = pd.read_csv('Titanic_train.csv')
test_data = pd.read_csv('Titanic_test.csv')



# In[ ]:





# In[ ]:






# In[6]:


print("\nTrain Data Head:\n", train_data.head())
print("\nTrain Data Info:\n")
train_data.info()
print("\nTrain Data Describe:\n", train_data.describe())


# In[8]:


print("\nMissing Values in Train Data:\n", train_data.isnull().sum())


# In[7]:


train_data.hist(bins=20, figsize=(15,10))
plt.show()


# In[8]:


numeric_cols = train_data.select_dtypes(include=['int64', 'float64'])
for col in numeric_cols:
    sns.boxplot(x=train_data[col])
    plt.title(col)
    plt.show()



# In[9]:


sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()


# In[19]:


train_data['Age'] = train_data['Age'].fillna(train_data['Age'].mean())
test_data['Age'] = test_data['Age'].fillna(test_data['Age'].mean())
train_data['Embarked'] = train_data['Embarked'].fillna(train_data['Embarked'].mode()[0])
test_data['Embarked'] = test_data['Embarked'].fillna(test_data['Embarked'].mode()[0])


# In[20]:


train_data.drop('Cabin', axis=1, inplace=True)
test_data.drop('Cabin', axis=1, inplace=True)


# In[ ]:


le = LabelEncoder()
for col in ['Sex', 'Embarked']:
    train_data[col] = le.fit_transform(train_data[col])
    test_data[col] = le.transform(test_data[col])


# In[ ]:


train_data.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)
test_data.drop(['Name', 'Ticket', 'PassengerId'], axis=1, inplace=True)


# In[ ]:


X = train_data.drop('Survived', axis=1)
y = train_data['Survived']


# In[ ]:


X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)


# In[ ]:


model = LogisticRegression(max_iter=1000)
model.fit(X_train, y_train)


# In[ ]:


y_pred = model.predict(X_val)
y_pred_proba = model.predict_proba(X_val)[:,1]

print("\nModel Evaluation Metrics:")
print("Accuracy :", accuracy_score(y_val, y_pred))
print("Precision:", precision_score(y_val, y_pred))
print("Recall   :", recall_score(y_val, y_pred))
print("F1 Score :", f1_score(y_val, y_pred))
print("ROC-AUC  :", roc_auc_score(y_val, y_pred_proba))


# In[ ]:


fpr, tpr, thresholds = roc_curve(y_val, y_pred_proba)
plt.plot(fpr, tpr, label="ROC curve")
plt.plot([0,1],[0,1],'k--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()



# In[ ]:


coefficients = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': model.coef_[0]
}).sort_values(by='Coefficient', ascending=False)

print("\nFeature Importance Based on Coefficients:\n", coefficients)


# In[ ]:


with open('logistic_model.pkl', 'wb') as file:
    pickle.dump(model, file)

print("\nModel Saved as 'logistic_model.pkl'.")



# In[25]:


import streamlit as st
import pickle
import numpy as np
import warnings
warnings.filterwarnings("ignore")


# In[26]:


model = pickle.load(open('logistic_model.pkl', 'rb'))


# In[27]:


st.title("🚢 Titanic Survival Prediction App")

st.write("Fill the details below to predict if a passenger would survive:")


# In[29]:


Pclass = st.selectbox('Passenger Class (1 = 1st, 2 = 2nd, 3 = 3rd)', [1,2,3])
Sex = st.selectbox('Sex (0 = Female, 1 = Male)', [0,1])
Age = st.slider('Age', 0, 100, 25)
SibSp = st.number_input('Number of Siblings/Spouses aboard', min_value=0)
Parch = st.number_input('Number of Parents/Children aboard', min_value=0)
Fare = st.slider('Fare', 0, 600, 50)
Embarked = st.selectbox('Port of Embarkation (0=Cherbourg, 1=Queenstown, 2=Southampton)', [0,1,2])


# In[30]:


if st.button('Predict'):
    input_data = np.array([[Pclass, Sex, Age, SibSp, Parch, Fare, Embarked]])
    prediction = model.predict(input_data)
    st.write('🎯 **Prediction:** ', '**Survived**' if prediction[0]==1 else '**Did not Survive**')


# # 1. Difference between Precision and Recall:
# # Precision = True Positives / (True Positives + False Positives)
# # Recall    = True Positives / (True Positives + False Negatives)
# # Precision focuses on quality (minimize false alarms), Recall focuses on completeness (catch all positives).
# 
# # 2. What is Cross-validation and why it is important?
# # Cross-validation is a technique where we divide data into 'k' parts and train/test the model 'k' times.
# # It checks model performance on unseen data and prevents overfitting.
# 
# 

# In[ ]:





# In[ ]:




