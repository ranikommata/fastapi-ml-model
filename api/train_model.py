import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import joblib

data={
    'math':[78,65,78,45,30,60,80,40,20,50],
    'science':[20,68,45,82,63,81,20,40,60,90],
    'English':[30,70,45,65,98,42,30,30,40,60],
    'Result':['Fail','Pass','Pass','Pass','Fail','Pass','Fail','Fail','Fail','Pass']
}
df=pd.DataFrame(data)
df['Result']=df['Result'].map({'Pass':1,'Fail':0})
#train the model
x=df[['math','science','English']]
y=df['Result']
#print(df)
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
#print(x_train)  #cd api #python train_model.py(to run in terminal)
#70% i/p  30%i/p,70%o/p   30%o/p
#print(x_test)
#call model
model=LogisticRegression()
model.fit(x_train,y_train)
#from sklearn.metrics import accuracy_score
#res=model.predict(x_test)
#print(res['score'])
#print("Accuracy",accuracy_score(y_test,res))
#new_student=pd.DataFrame([[60,40,30]],columns=['math','science','English'])
#predict=model.predict(new_student)
#print(predict[0])
joblib.dump(model,'model.pkl')
