# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder

#reading from csv data file into dataframe
df=pd.read_csv('C:\\Users\\lenovo\\Downloads\\data.csv')

#information regarding data attributes
print(df.info())

#diffrent data description and visualization functions

df.hist(column='Survived')

df.boxplot(column='Fare')

mage=df['Age'].mean()
meage=df['Age'].median()

'''get median for df age column where age<75 just an example'''
median = df.loc[df['Age']<75, 'Age'].median()


dfa=df['Age'][df['Age'].notna()].count()

uniquesurvived=df['Survived'].nunique()
survivedvalues=df['Survived'].value_counts()

pclassvalues=df['Pclass'].value_counts()

#outlier detection
q1=df['Fare'].quantile(0.25)
q3=df['Fare'].quantile(0.75)
q2=df['Fare'].quantile(0.5)
iqr=q3-q1

#removing values that are outliers
df=df[((df.Fare>=q1-1.5*iqr) & (df.Fare<=q3+1.5*iqr))]
print(df)

# explore it for nominal to numerical conversion
enc = OneHotEncoder(handle_unknown='ignore')


dftest=pd.read_csv('C:\\Users\\umair\\Downloads\\test.csv')
#coying one column into data frame
dfp=pd.DataFrame()
print(len(dftest))
dfp['PassengerId']=dftest['PassengerId'][0:len(dftest)]



#counting outliers through loop
outliers=0
for v in dfa:
    if (v<q1-1.5*iqr or v>q3+1.5*iqr):
        outliers+=1

#missing value detection
print(df[df.notna()])
print(df.isnull())

#counting missing value 
print(df.isnull().sum())
print(df['Cabin'].isnull().sum())


#filling missing values with mean
df['Age'].fillna(df['Age'].mean(),inplace=True)

'''filling na/missing values with median'''
df.fillna(median,inplace=True)

'''setting all the rows to not available where age>75'''
df.loc[df.Age > 75, 'Age'] = np.nan


#deleting a column
del(df['Fare'])


#nothing significant to pre-processing only creating data and wrting to csv file
temp=[]
for i in range(len( dftest)):
    d=dftest.loc[i,:]
    if d['Sex']=='female' and (d['Pclass']==1 or d['Pclass']==2):
        temp.append(1)
    else:
        temp.append(0)
dfp['Survived']=temp
dfp.to_csv('C:\\Users\\umair\\Downloads\\predicted.csv')
