import pandas as pd
import numpy as np
import seaborn as sns
import pickle

data=pd.read_csv("C:\\Users\\saira\\OneDrive\\Desktop\\ipbl\\Others\\Heart-Disease-Prediction-Deployment-master\\heart.csv")

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

y=data['heartdisease']
x=data.drop('heartdisease',axis=1)

x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.3,random_state=42)
from sklearn.ensemble import RandomForestClassifier
rfmodel3=RandomForestClassifier()
rfmodel3.fit(x_train,y_train)
y_pred=rfmodel3.predict(x_test)
rfacc3=accuracy_score(y_test,y_pred)
cm=confusion_matrix(y_test, y_pred)
cr=classification_report(y_test, y_pred)

filename='heart.pkl'
with open(filename, 'wb') as file:
    pickle.dump(rfmodel3, file)