import pandas as pd
from sklearn.linear_model import LinearRegression

data=pd.read_csv("dataset.csv")

x=data.iloc[:,:3].values   # tüm satılar ve il 3 sütün
y=data.iloc[:,3:].values

lr=LinearRegression()

lr.fit(x,y)

pred=lr.predict(x)

pred_list=[]
print(pred)

for i in pred:
    b=0
    ik=0
    if i[0]>0.5:
        ik=1
    else:
        ik=0
    pred_list.append([b,ik])
print(pred_list)
#print(y)
# şimdi bu aldığımız değerleri göz takibi.py ye uygulayacağız