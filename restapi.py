#FAST API
#Model Deployment
import pandas as pd
import numpy as np
import os
os.chdir(r"D:\Data_Science\Bank_Note_Auth")
print(os.getcwd())
df=pd.read_csv("BankNote_Authentication.csv")
print(df.head())
print(df.size)
print(df.shape)
print(df.describe())
print(df.info())
print(df.isnull().sum())
X=df.iloc[:,0:4]
y=df.iloc[:,4]
print(X.head())
print(y.head())
#Train test split
from sklearn.model_selection import train_test_split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=0)
#RF classifier
from sklearn.ensemble import RandomForestClassifier
classifier=RandomForestClassifier()
classifier.fit(X_train,y_train)
#Prediction
y_pred=classifier.predict(X_test)
#Check accuracy
from sklearn.metrics import accuracy_score
score=accuracy_score(y_test,y_pred)
print(score) #0.9902912621359223
#Prediction with new data
print(classifier.predict([[2,3,4,1]])) #[0]
#Create pickle file
import pickle
pickle_out=open("classifier.pkl","wb")
pickle.dump(classifier,pickle_out)
pickle_out.close()