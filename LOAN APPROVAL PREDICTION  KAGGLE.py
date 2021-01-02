#!/usr/bin/env python
# coding: utf-8

# In[854]:


import os
os.chdir("C:\\Users\\Shankii\\Desktop\\kaglle\\loan_prediction")


# In[855]:


import pandas as pd
import numpy as np
import math
import quandl
import scipy
from scipy import stats
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.anova import anova_lm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from scipy.optimize import minimize
from scipy.stats import chi2
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import binarize
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
import category_encoders as ce
from sklearn.ensemble import RandomForestClassifier
from sklearn import tree
from sklearn.naive_bayes import GaussianNB


# In[856]:


idtrain=pd.read_csv("train_u6lujuX_CVtuZ9i.csv")
idtrain.head()


# In[857]:


idtrain.drop('Loan_ID',axis=1,inplace=True)
idtrain.info()


# In[858]:


idtrain.describe(include='O')
### will show the columns which are object types

## so in gender col
## male has appeared 489 times


# In[859]:


idtrain.describe()

## credit_history seems to have just 2 values
## 0 and 1

## since only 2 values
## lets convert that into categorical cols


# In[860]:


idtrain['Credit_History']=idtrain['Credit_History'].astype('O')

idtrain.describe(include='O')


# In[ ]:





# In[861]:


## lets see if any duplicated values are there or not in the dataset

idtrain.duplicated().any()


## no duplicated values


# In[ ]:





# In[862]:


## lets fill the na values in the dataset

idtrain.isnull().sum()

## loan amount with the mean of the values
## credit history with maximum apperaing entry


# In[863]:


idtrain['LoanAmount']=idtrain['LoanAmount'].fillna(idtrain['LoanAmount'].mean())

idtrain['Credit_History']=idtrain['Credit_History'].fillna(idtrain['Credit_History'].median())


# In[864]:


## lets fill the NA values the value coming above them

idtrain=idtrain.fillna(method='bfill')

## lets not remove the null values from data
## since it may lead to removal of valuation info from dataset

idtrain.isnull().sum()

## all the NA values are been taken care of


# In[ ]:





# In[865]:


## lets see the data using various visualizations
## loan status vs gender

sns.countplot(idtrain['Gender'],hue=idtrain['Loan_Status'])


## using crosstab table, lets see the tabular computation of gender vs loan approval status

pd.crosstab(index=idtrain['Gender'],columns=idtrain['Loan_Status'])


## from the table
## males have higher chances of getting the loan approved


# In[ ]:





# In[866]:


## Education vs loan approval

sns.countplot(idtrain['Education'],hue=idtrain['Loan_Status'])


## using crosstab table, lets see the tabular computation of education vs loan approval status

pd.crosstab(index=idtrain['Education'],columns=[idtrain['Loan_Status'],idtrain['Gender']])



## from table
## graduates with gender as male have more chances of getting loan


# In[ ]:





# In[867]:


## mode of employment vs loan approval

sns.countplot(idtrain['Self_Employed'],hue=idtrain['Loan_Status'])


## using crosstab table, lets see the tabular computation of self_employed vs loan approval status

pd.crosstab(index=idtrain['Self_Employed'],columns=idtrain['Loan_Status'])



## not self employed has more chances of getting the loan


# In[ ]:





# In[868]:


## property vs loan
sns.countplot(idtrain['Property_Area'],hue=idtrain['Loan_Status'])


## using crosstab table, lets see the tabular computation of property_area vs loan approval status

pd.crosstab(index=idtrain['Property_Area'],columns=idtrain['Loan_Status'])


## semiurban has most chances of getting it


# In[869]:


k=idtrain.columns

for i in k:
    print(idtrain[i].value_counts(),'\n\n')


# In[870]:


## replace
## gender : male as 1 : female as 0
## loan_status : Y as 1 : N as 0
## married : yes as 1 ; no as 0
## education : graduate as 1 ; not graduate as 0
## SElf_employed : yes as 1 ; no as 0
## propert : urban as 2 ; rural as 0; semiurban as 1


## dependents as object type
## so we have to convert it into numeric so that 3+ can be written as 3


## dependents : 0 as 0, 1 as 1, 2 as 2 , 3+ as 3


idtrain['Gender']=idtrain['Gender'].map({'Male':1,'Female':0})
idtrain['Loan_Status']=idtrain['Loan_Status'].map({'Y':1,'N':0})
idtrain['Married']=idtrain['Married'].map({'Yes':1,'No':0})
idtrain['Education']=idtrain['Education'].map({'Graduate':1,'Not Graduate':0})
idtrain['Self_Employed']=idtrain['Self_Employed'].map({'Yes':1,'No':0})
idtrain['Property_Area']=idtrain['Property_Area'].map({'Urban':2,'Rural':0,'Semiurban':1})
idtrain['Dependents']=idtrain['Dependents'].map({'0':0,'1':1,'2':2,'3+':3})


# In[871]:


## lets see whether the above code has worked properly or not

k=idtrain.columns

for i in k:
    print(idtrain[i].value_counts(),'\n\n')
    
## it worked just fine    


# In[872]:


idtrain.info()
## all cols to int or float type
## no null values

## now we can go with variohs algos


# In[ ]:





# # HEATMAP

# In[873]:


## lets go first with finding out the relations between the various variables with loan approval

plt.figure(figsize=(16,15))
sns.heatmap(idtrain.corr(),annot=True)
plt.title('heatmap for seeing the corelation between the variables')


# In[874]:


## from heatmap:
## loan status is heavily dependent on credit history


# 

# In[875]:


x=idtrain.drop('Loan_Status',axis=1)
y=idtrain['Loan_Status']

xtrain,xtest,ytrain,ytest=train_test_split(x,y,test_size=0.3,random_state=22)


# # LOGISTIC REGRESSION

# In[876]:


## since we have to only predict 1 and 0
## we can use logistic reg

lg=LogisticRegression()
lg.fit(xtrain,ytrain)
ypred1=lg.predict(xtest)

print('accuracy is ', accuracy_score(ypred1,ytest))


# In[ ]:





# # SUPPORT VECTOR MACHINE

# In[877]:


model=SVC()
model.fit(xtrain,ytrain)
ypred2=model.predict(xtest)

print('accuracy is ', accuracy_score(ypred2,ytest))


# In[ ]:





# # DECISION TREE

# In[878]:


model=DecisionTreeClassifier()
model.fit(xtrain,ytrain)
ypred3=model.predict(xtest)

print('accuracy is ', accuracy_score(ypred3,ytest))


# In[ ]:





# # KNN

# In[879]:


model=KNeighborsClassifier()
model.fit(xtrain,ytrain)
ypred4=model.predict(xtest)

print('accuracy is ', accuracy_score(ypred4,ytest))


# In[ ]:





# In[880]:


## as of now:
## logistic regression seems to be better model
## but lets improve our model now


# In[ ]:





# In[881]:


## lets improve the model using feature engineering


# In[ ]:





# In[882]:


## lets go first with finding out the relations between the various variables with loan approval

plt.figure(figsize=(16,15))
sns.heatmap(idtrain.corr(),annot=True)
plt.title('heatmap for seeing the corelation between the variables')


# In[883]:


xtrain['col1']=xtrain['CoapplicantIncome'] + xtrain['ApplicantIncome']
xtrain['col2']=xtrain['LoanAmount']*xtrain['Loan_Amount_Term']


# In[884]:


## lets drop the cols used above

xtrain.drop(['CoapplicantIncome','ApplicantIncome','LoanAmount','Loan_Amount_Term'],axis=1,inplace=True)


# In[885]:


newdata=pd.concat([xtrain,ytrain],axis=1)


# In[886]:


## lets go first with finding out the relations between the various variables with loan approval

plt.figure(figsize=(16,15))
sns.heatmap(newdata.corr(),annot=True)
plt.title('heatmap for seeing the corelation between the variables')


# In[887]:


xtest['col1']=xtest['CoapplicantIncome'] + xtest['ApplicantIncome']
xtest['col2']=xtest['LoanAmount']*xtest['Loan_Amount_Term']

## lets drop the cols used above

xtest.drop(['CoapplicantIncome','ApplicantIncome','LoanAmount','Loan_Amount_Term'],axis=1,inplace=True)


# # LOGISTIC REGRESSION

# In[888]:


## since we have to only predict 1 and 0
## we can use logistic reg

lg=LogisticRegression()
lg.fit(xtrain,ytrain)
ypred1=lg.predict(xtest)

print('accuracy is ', accuracy_score(ypred1,ytest))


# # SUPPORT VECTOR MACHINE

# In[889]:


model=SVC()
model.fit(xtrain,ytrain)
ypred2=model.predict(xtest)

print('accuracy is ', accuracy_score(ypred2,ytest))


# # DECISION TREE 

# In[890]:


model=DecisionTreeClassifier()
model.fit(xtrain,ytrain)
ypred3=model.predict(xtest)

print('accuracy is ', accuracy_score(ypred3,ytest))


# # KNN

# In[891]:


model=KNeighborsClassifier()
model.fit(xtrain,ytrain)
ypred4=model.predict(xtest)

print('accuracy is ', accuracy_score(ypred4,ytest))


# In[ ]:





# In[892]:


from scipy.stats import norm

fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.distplot(xtrain['col1'],ax=ax[0],fit=norm)
sns.distplot(xtrain['col2'],fit=norm)


## both the columns doesnt follow normal distribution
## lets make it logarithmic


# In[893]:


xtrain['col1']=np.log(xtrain['col1'])
xtrain['col2']=np.log(xtrain['col2'])
xtest['col1']=np.log(xtest['col1'])
xtest['col2']=np.log(xtest['col2'])


# In[894]:



fig, ax = plt.subplots(1,2,figsize=(20,5))

sns.distplot(xtrain['col1'],ax=ax[0],fit=norm)
sns.distplot(xtrain['col2'],fit=norm)

## NOW ALMOST FOLLOWING THE NORMAL CURVE


# # LOGISTIC REGRESSION

# In[895]:


## since we have to only predict 1 and 0
## we can use logistic reg

lg=LogisticRegression()
lg.fit(xtrain,ytrain)
ypred1=lg.predict(xtest)

print('accuracy is ', accuracy_score(ypred1,ytest))


## model has become comparitively good


# # SUPPORT VECTOR MACHINE

# In[896]:


model=SVC()
model.fit(xtrain,ytrain)
ypred2=model.predict(xtest)

print('accuracy is ', accuracy_score(ypred2,ytest))


# # DECISION TREE

# In[897]:


model=DecisionTreeClassifier()
model.fit(xtrain,ytrain)
ypred3=model.predict(xtest)

print('accuracy is ', accuracy_score(ypred3,ytest))


# # KNN

# In[898]:


model=KNeighborsClassifier()
model.fit(xtrain,ytrain)
ypred4=model.predict(xtest)

print('accuracy is ', accuracy_score(ypred4,ytest))


# In[ ]:





# In[899]:


newdata.describe()

## see in some cols lke col1 and col2
## mean and max valuess are tooooo high

## so it means outliers are there

## so lets remove the outliers


# In[900]:


sns.boxplot(xtrain['col1'])

## has outliers


# In[901]:


sns.boxplot(xtrain['col2'])

## has outliers


# In[ ]:





# # removing outliers

# In[902]:


threshold=0.02

### the lesser the threshold, more the values get remove
## go on checking the diff values of the threshold and see which value suites best
col2_outlierstrain=xtrain['col2']
col2_outlierstest=xtest['col2']

q25train=np.percentile(col2_outlierstrain,25)
q75train=np.percentile(col2_outlierstrain,75)

iqr= q75train - q25train

cutoff= iqr * threshold

lowercut= q25train - cutoff
uppercut = q75train + cutoff

## lowercut and uppercut are like boundries for the extreme points for the values
## beyond which values will be deleted

outliers=[x for x in col2_outlierstrain if x<lowercut or x>uppercut]
data_outliers=pd.concat([xtrain,ytrain],axis=1)

data_outlierstrain = data_outliers.drop(data_outliers[(data_outliers['col2'] > uppercut) | (data_outliers['col2'] < lowercut)].index)


# In[903]:


sns.boxplot(xtrain['col2'])

## outliers are there for col2


# In[904]:


## same for col1
threshold=0.1

col1_outlierstrain=xtrain['col1']

q25train=np.percentile(col1_outlierstrain,25)
q75train=np.percentile(col1_outlierstrain,75)

iqr=q75train - q25train

cutoff= iqr * threshold

lowercut= q25train - cutoff
uppercut = q75train + cutoff

## lowercut and uppercut are like boundries for the extreme points for the values
## beyond which values will be deleted

outliers=[x for x in col1_outlierstrain if x<lowercut or x>uppercut]
data_outliers=pd.concat([xtrain,ytrain],axis=1)

data_outlierstrain = data_outliers.drop(data_outliers[(data_outliers['col1'] > uppercut) | (data_outliers['col2'] < lowercut)].index)


# In[905]:


sns.boxplot(xtrain['col1'])

## almost perfect fit


# In[906]:


## same for cols in xtest 
## same for col1

threshold=0.002
col1_outlierstest=xtest['col1']

q25test=np.percentile(col1_outlierstest,25)
q75test=np.percentile(col1_outlierstest,75)

iqr=q75test - q25test

cutoff= iqr * threshold

lowercut= q25test - cutoff
uppercut = q75test + cutoff

## lowercut and uppercut are like boundries for the extreme points for the values
## beyond which values will be deleted

outliers=[x for x in col1_outlierstest if x<lowercut or x>uppercut]
data_outliers=pd.concat([xtest,ytest],axis=1)

data_outlierstest = data_outliers.drop(data_outliers[(data_outliers['col1'] > uppercut) | (data_outliers['col1'] < lowercut)].index)


# In[907]:


sns.boxplot(xtest['col1'])

## outliers are there for col2


# In[908]:


## same for cols in xtest 
## same for col2

threshold=0.1
col2_outlierstest=xtest['col2']

q25test=np.percentile(col2_outlierstest,25)
q75test=np.percentile(col2_outlierstest,75)

iqr=q75test - q25test

cutoff= iqr * threshold

lowercut= q25test - cutoff
uppercut = q75test + cutoff

## lowercut and uppercut are like boundries for the extreme points for the values
## beyond which values will be deleted

outliers=[x for x in col1_outlierstest if x<lowercut or x>uppercut]
data_outliers=pd.concat([xtest,ytest],axis=1)

data_outlierstest = data_outliers.drop(data_outliers[(data_outliers['col2'] > uppercut) | (data_outliers['col2'] < lowercut)].index)


# In[909]:


sns.boxplot(xtest['col2'])

## perfect fit for this column


# In[ ]:





# In[910]:


## almost all the outliers are being removed
## lets move on using various algos


# In[ ]:





# In[911]:


xtrain=data_outlierstrain.drop('Loan_Status',axis=1)
ytrain=data_outlierstrain['Loan_Status']

xtest=data_outlierstest.drop('Loan_Status',axis=1)
ytest=data_outlierstest['Loan_Status']


# # LOGISTIC REGRESSION

# In[912]:


## since we have to only predict 1 and 0
## we can use logistic reg

lg=LogisticRegression()
lg.fit(xtrain,ytrain)
ypred1=lg.predict(xtest)

print('accuracy is ', accuracy_score(ypred1,ytest))


## model has become comparitively good


# # SUPPORT VECTOR MACHINE

# In[913]:


model=SVC()
model.fit(xtrain,ytrain)
ypred2=model.predict(xtest)

print('accuracy is ', accuracy_score(ypred2,ytest))


## model improved after removing outliers


# # DECISION TREE

# In[914]:


model=DecisionTreeClassifier()
model.fit(xtrain,ytrain)
ypred3=model.predict(xtest)

print('accuracy is ', accuracy_score(ypred3,ytest))


## slightly model improved


# # KNN

# In[915]:


model=KNeighborsClassifier()
model.fit(xtrain,ytrain)
ypred4=model.predict(xtest)

print('accuracy is ', accuracy_score(ypred4,ytest))


## improved a lot


# # RANDOM FOREST

# In[916]:


model=RandomForestClassifier()
model.fit(xtrain,ytrain)
ypred4=model.predict(xtest)

print('accuracy is ', accuracy_score(ypred4,ytest))


# In[ ]:





# # LOGISTIC REGRESSION IS THE BEST ONE FOR PREDICTION LOAN STATUS APPROVAL

# In[ ]:





# In[917]:


data_corr = pd.concat([xtrain, ytrain], axis=1)
corr = data_corr.corr()
plt.figure(figsize=(10,7))
sns.heatmap(corr, annot=True);


# In[918]:


## just credit history seems to be important one
## rest all varibales are just hanging in there in the dataset


# In[919]:


## lets see just by removing self employed  and see the model prediction

## but gemder and married have some strong relationship
## so lets remove one of them also


# In[920]:


xtrain.drop(['Self_Employed','Gender'],axis=1,inplace=True)
xtest.drop(['Self_Employed','Gender'],axis=1,inplace=True)


# # LOGISTIC REGRESSION

# In[921]:


## since we have to only predict 1 and 0
## we can use logistic reg

lg=LogisticRegression()
lg.fit(xtrain,ytrain)
ypred1=lg.predict(xtest)

print('accuracy is ', accuracy_score(ypred1,ytest))


# # SUPPORT VECTOR MACHINE

# In[922]:


model=SVC()
model.fit(xtrain,ytrain)
ypred2=model.predict(xtest)

print('accuracy is ', accuracy_score(ypred2,ytest))


# # DECISION TREE

# In[923]:


model=DecisionTreeClassifier()
model.fit(xtrain,ytrain)
ypred3=model.predict(xtest)

print('accuracy is ', accuracy_score(ypred3,ytest))



# # KNN

# In[924]:


model=KNeighborsClassifier()
model.fit(xtrain,ytrain)
ypred4=model.predict(xtest)

print('accuracy is ', accuracy_score(ypred4,ytest))



# # RANDOM FOREST

# In[925]:


model=RandomForestClassifier()
model.fit(xtrain,ytrain)
ypred4=model.predict(xtest)

print('accuracy is ', accuracy_score(ypred4,ytest))

## model improved


# In[ ]:





# In[926]:


## if u remove col2 and dependents also
## model accuracy is detrriorating


# In[ ]:





# # follow all the above steps on test sample also and predict the answer

# In[ ]:





# In[ ]:





# In[ ]:




