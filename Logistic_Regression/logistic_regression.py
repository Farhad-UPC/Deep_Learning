# -*- coding: utf-8 -*-
# Based on "Building A Logistic Regression in Python, Step by Step"
# Programmer: Farhad-UPC
from __future__ import print_function, division
# update your version of future --- >  sudo pip install -U future
import numpy as np 
import pandas as pd 
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
#from sklearn.cross_validation import train_test_split  
#train_test_split is now in model_selection
from sklearn.model_selection import train_test_split   #Split arrays or matrices into random train and test subsets
import seaborn as sns
import matplotlib.pyplot as plt 


#In logistic regression, the dependent variable is a binary variable that contains data coded as 1 (yes, success, etc.) or 0 (no, failure, etc.). 
#In other words, the logistic regression model predicts P(Y=1) as a function of X.
#Dataset resource : http://archive.ics.uci.edu/ml/index.php, It includes 41,188 records (customers) and 21 fields.

url = "https://raw.githubusercontent.com/Farhad-UPC/Deep_Learning/master/Logistic_Regression/bank.csv"
col_separator = ','
bank_data = pd.read_csv(url, header = 0, sep = col_separator)
print (bank_data.head (10))
#Sometimes csv file has null values, which are later displayed as NaN in Data Frame. 
#Pandas dropna() method allows the user to analyze and drop Rows/Columns with Null values in different ways.
bank_data = bank_data.dropna()
print (bank_data.shape)
print (list (bank_data.columns))
# y — has the client subscribed a term deposit? (binary: “1”, means “Yes”, “0” means “No”)
# Pre-processing  --->  The education column of the dataset has many categories and we need to reduce the categories for a better modelling.
# While analyzing the data, many times the user wants to see the unique values in a particular column, which can be done using Pandas unique() function.
print ('\n', bank_data["education"].unique())
# numpy.where(condition[, x, y]) ----> Return elements chosen from x or y depending on condition.
bank_data['education']=np.where(bank_data['education'] =='basic.9y', 'Basic', bank_data['education'])
bank_data['education']=np.where(bank_data['education'] =='basic.6y', 'Basic', bank_data['education'])
bank_data['education']=np.where(bank_data['education'] =='basic.4y', 'Basic', bank_data['education'])
print ('\n', bank_data["education"].unique(), '\n')


#percentage of no subscription or subscription
count_no_sub = len(bank_data[bank_data['y']==0])
count_sub = len(bank_data[bank_data['y']==1])
pct_of_no_sub = count_no_sub/(count_no_sub+count_sub)
print("percentage of no subscription is", pct_of_no_sub*100)
pct_of_sub = count_sub/(count_no_sub+count_sub) 
print("percentage of subscription", pct_of_sub*100, '\n\n', "Our classes are imbalanced, and the ratio of no-subscription to subscription instances is 89:11")

# Data exploration ---> The number of of no subscription or subscription

print (bank_data['y'].value_counts(), '\n')
sns.countplot(x = 'y', data = bank_data, palette = 'hls')    # Show the counts of observations in each categorical bin using bars
plt.show()

# More exploration and observation step ----> calculate categorical means to get a more detailed sense of our data

print (bank_data.groupby('y').mean(), '\n')
print (bank_data.groupby('job').mean(), '\n')
print (bank_data.groupby('marital').mean(), '\n')
print (bank_data.groupby('education').mean(), '\n')

# Visualizations step for better comparison 

#%matplotlib inline
pd.crosstab(bank_data.job,bank_data.y).plot(kind='bar')
plt.title('Purchase Frequency for Job Title')
plt.xlabel('Job')
plt.ylabel('Frequency of Purchase')
plt.savefig('purchase_fre_job')
plt.show() # the result shows that job title can be a good predictor of the outcome variable.

table=pd.crosstab(bank_data.marital,bank_data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Marital Status vs Purchase')
plt.xlabel('Marital Status')
plt.ylabel('Proportion of Customers')
plt.savefig('mariral_vs_pur_stack')
plt.show()   # The marital status does not seem a strong predictor for the outcome variable.

table=pd.crosstab(bank_data.education,bank_data.y)
table.div(table.sum(1).astype(float), axis=0).plot(kind='bar', stacked=True)
plt.title('Stacked Bar Chart of Education vs Purchase')
plt.xlabel('Education')
plt.ylabel('Proportion of Customers')
plt.savefig('edu_vs_pur_stack')
plt.show() # Education seems a good predictor of the outcome variable.

pd.crosstab(bank_data.day_of_week,bank_data.y).plot(kind='bar')
plt.title('Purchase Frequency for Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_dayofweek_bar')
plt.show()   # Day of week may not be a good predictor of the outcome.

pd.crosstab(bank_data.month,bank_data.y).plot(kind='bar')
plt.title('Purchase Frequency for Month')
plt.xlabel('Month')
plt.ylabel('Frequency of Purchase')
plt.savefig('pur_fre_month_bar') 
plt.show()  # Month might be a good predictor of the outcome variable.

bank_data.age.hist()
plt.title('Histogram of Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.savefig('hist_age')
plt.show()  # Most of the customers of the bank in this dataset are in the age range of 30–40.

#In statistics and econometrics, particularly in regression analysis,
#a dummy variable is one that takes the value 0 or 1 to indicate the 
#absence or presence of some categorical effect that may be expected to shift the outcome.
#Dummy variables are used as devices to sort data into mutually exclusive categories. 
cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
for var in cat_vars:
    cat_list='var'+'_'+var
    cat_list = pd.get_dummies(bank_data[var], prefix=var)  # Convert categorical variable into dummy/indicator variables.
    data1=bank_data.join(cat_list)
    bank_data=data1

cat_vars=['job','marital','education','default','housing','loan','contact','month','day_of_week','poutcome']
data_vars= bank_data.columns.values.tolist() #converting to list 
to_keep=[i for i in data_vars if i not in cat_vars]
data_final=bank_data[to_keep]
print (data_final.columns.values)
print (bank_data.columns.values)
print (data_final.head (10))
print (bank_data.head (10))

#To balance the data, you have several options. The first is to simply gather more data. While this is always preferable,
#it is often not possible. In this case, you can try resampling the data, either by under-sampling your majority class
#or over-sampling your minority class. Over-sampling consists of either sampling each member of the minority class with replacement,
#or creating synthetic members by randomly sampling from the feature set. This is what SMOTE — Synthetic Minority Over-sampling Technique — does.

X = data_final.loc[:, data_final.columns != 'y']
y = data_final.loc[:, data_final.columns == 'y']
# doesn´t support python 2.7
from imblearn.over_sampling import SMOTE
os = SMOTE(random_state=0)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
columns = X_train.columns
os_data_X,os_data_y=os.fit_sample(X_train, y_train)
os_data_X = pd.DataFrame(data=os_data_X,columns=columns )
os_data_y= pd.DataFrame(data=os_data_y,columns=['y'])
# we can Check the numbers of our data
print("length of oversampled data is ",len(os_data_X))
print("Number of no subscription in oversampled data",len(os_data_y[os_data_y['y']==0]))
print("Number of subscription",len(os_data_y[os_data_y['y']==1]))
print("Proportion of no subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==0])/len(os_data_X))
print("Proportion of subscription data in oversampled data is ",len(os_data_y[os_data_y['y']==1])/len(os_data_X))

#Recursive Feature Elimination (RFE) is based on the idea to repeatedly construct a model and 
#choose either the best or worst performing feature, setting the feature aside and
#then repeating the process with the rest of the features. This process is
#applied until all features in the dataset are exhausted. The goal of RFE 
#is to select features by recursively considering smaller and smaller sets of features.

data_final_vars=data_final.columns.values.tolist()
y=['y']
X=[i for i in data_final_vars if i not in y]
from sklearn.feature_selection import RFE
from sklearn.linear_model import LogisticRegression
logreg = LogisticRegression()
rfe = RFE(logreg, 20)
rfe = rfe.fit(os_data_X, os_data_y.values.ravel())
print(rfe.support_)
print(rfe.ranking_)

cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 'default_no', 'default_unknown', 
      'contact_cellular', 'contact_telephone', 'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 
      'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"] 
X=os_data_X[cols]
y=os_data_y['y']

# Implementing the model
import statsmodels.api as sm
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())


#The p-values for most of the variables are smaller than 0.05, except four variables, therefore, we will remove them.
cols=['euribor3m', 'job_blue-collar', 'job_housemaid', 'marital_unknown', 'education_illiterate', 
      'month_apr', 'month_aug', 'month_dec', 'month_jul', 'month_jun', 'month_mar', 
      'month_may', 'month_nov', 'month_oct', "poutcome_failure", "poutcome_success"] 
X=os_data_X[cols]
y=os_data_y['y']
logit_model=sm.Logit(y,X)
result=logit_model.fit()
print(result.summary2())

#Logistic Regression Model Fitting
#test date = 30%
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)
logreg = LogisticRegression()
logreg.fit(X_train, y_train)

y_pred = logreg.predict(X_test)
print('Accuracy of logistic regression classifier on test set:{:.2f}'.format(logreg.score(X_test, y_test)))
#Accuracy of logistic regression classifier on test set: 0.74

#Confusion Matrix
from sklearn.metrics import confusion_matrix
confusion_matrix = confusion_matrix(y_test, y_pred)
print(confusion_matrix)
from sklearn.metrics import classification_report
print(classification_report(y_test, y_pred))
#Interpretation: Of the entire test set, 74% of the promoted term deposit were 
#the term deposit that the customers liked.
#Of the entire test set, 74% of the customer’s preferred term deposits that were promoted.

#The receiver operating characteristic (ROC) curve is another common tool used with binary classifiers. 
#The dotted line represents the ROC curve of a purely random classifier;
#a good classifier stays as far away from that line as possible (toward the top-left corner).
from sklearn.metrics import roc_auc_score
from sklearn.metrics import roc_curve
logit_roc_auc = roc_auc_score(y_test, logreg.predict(X_test))
fpr, tpr, thresholds = roc_curve(y_test, logreg.predict_proba(X_test)[:,1])
plt.figure()
plt.plot(fpr, tpr, label='Logistic Regression (area = %0.2f)' % logit_roc_auc)
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Log_ROC')
plt.show()
