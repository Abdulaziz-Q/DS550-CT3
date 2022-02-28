# importing needed packages
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

#importing datasets
df = pd.read_csv('transfusion.csv')

# check if null values exists
df.info()

# descriptive statistics of the dataset
df.describe()

# exclude the target column from predictors columns
df_predictors = df.drop(columns='D')

# spilt the dataset into test and train sets. 80% test 20% train
X_train, X_test, y_train, y_test = train_test_split(df_predictors, df['D'], test_size=0.8, random_state=9)
# fitting Gaussian Naive Bayes to the dataset
gnb = GaussianNB()
gnb = gnb.fit(X_train, y_train)

# validate the model

#Perform classification on an array of test vectors X.
y_pred = gnb.predict(X_test)
print(y_pred)

# Return log-probability estimates for the test vector X.
print(gnb.predict_log_proba(X_test))

# Return probability estimates for the test vector X.
print(gnb.predict_proba(X_test))

# Return the mean accuracy on the given test data and labels.
print(gnb.score(X_test,y_test))