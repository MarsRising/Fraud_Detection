#!/usr/bin/env python
# coding: utf-8

# In[697]:


#libs
import pandas as pd
import numpy as np
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, RocCurveDisplay, precision_recall_curve, auc
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from statsmodels.tools.tools import add_constant

from geopy.geocoders import Nominatim
import time
from geopy.exc import GeocoderTimedOut
import re
from imblearn.under_sampling import RandomUnderSampler
from sklearn.feature_selection import SelectFromModel

print("libs download")


# In[698]:


df1 = pd.read_csv("Fraudulent_E-Commerce_Transaction_Data.csv")
df2 = pd.read_csv("Fraudulent_E-Commerce_Transaction_Data_2.csv")
df1.info()
df2.info()


# In[699]:


#check for dupes to understand if the data is all seperate
duplicate1=df1.duplicated().sum()
duplicate2=df2.duplicated().sum()
print(f"Duplicates in df1: {duplicate1}")
print(f"Duplicates in df2: {duplicate2}")
duplicate2


# In[700]:


#Combine dfs
df=pd.concat([df1, df2], ignore_index=True)
df.info()


# In[701]:


#check for dupes
duplicate=df.duplicated().sum()
print(f"Duplicates in df: {duplicate}")


# In[702]:


fraud_count = df['Is Fraudulent'].value_counts()
print(fraud_count)


# In[703]:


#drop the transactionid as it has no use
df.drop(['Transaction ID', 'Customer ID'], axis=1, inplace=True)


# In[704]:


#check for unique values
for column in df.columns:
    print(f"Unique values in '{column}':")
    print(df[column].unique())
    print("\n")


# In[705]:


#ensure transaction date is in right format
df['Transaction Date'] = pd.to_datetime(df['Transaction Date'])
df.info()


# In[706]:


#EDA
#Univariate analysis
#BoxPlots to see layout

#set up numerical columns for loop
numericals=df.select_dtypes(include=['float64', 'int64']).columns

#Loop through numerical columns and create boxplots for each
for col in numericals:
    plt.figure(figsize=(8,6))
    sns.boxplot(x=df[col])
    plt.title(f'Box Plot of {col}')
    plt.show()


# In[707]:


#check examples of impossible ages
df_invalid_age = df[df['Customer Age'] <= 15]
print(df_invalid_age.head())


# In[708]:


#check how much data it is tied to, to ensure its okay to drop these
fraud_count = df_invalid_age['Is Fraudulent'].value_counts()
print(fraud_count)


# In[709]:


#drop any unrealistic age as we will maintain enough data
df = df[df['Customer Age'] >= 15]
df.info()


# In[710]:


#set up numerical columns for loop
numericals=df.select_dtypes(include=['float64', 'int64']).columns

#Loop through numerical columns and create boxplots for each
for col in numericals:
    plt.figure(figsize=(8,6))
    sns.boxplot(x=df[col])
    plt.title(f'Box Plot of {col}')
    plt.show()


# In[711]:


#EDA for Categoricals
#Get categorical columns 
categoricals=df.select_dtypes(include=['object']).columns

#A lot of data, chek the top categories for display
top_n=10

#loop for statistical summary
for col in categoricals:
    print(f"Summary for '{col}':")
    print(f"Unique values in '{col}': {df[col].nunique()}")
    print(f"Most frequent value (mode) for '{col}': {df[col].mode()[0]}")
    
    # Display value counts for the top N categories
    print(f"Value counts for the top {top_n} categories in '{col}':")
    print(df[col].value_counts().head(top_n))


# In[712]:


#Large amount of unique IP addresses
#Usually would keep this, but it could lead to overfitting
#Drop IP
df = df.drop(columns=['IP Address'])

#init geocoder
geolocator = Nominatim(user_agent="address_parser")

#Large amount of different addresses. CState may provide more detail
#Function to extract only state
def extract_zip(address):
    if isinstance(address, str):
        match = re.search(r'\b\d{5}(-\d{4})?\b', address)
        if match:
            return match.group(0)
    return 'Unknown'

df['Shipping Address'] = df['Shipping Address'].apply(extract_zip)
df['Billing Address'] = df['Billing Address'].apply(extract_zip)

print(df['Shipping Address'])
print(df['Billing Address'])


# In[713]:


#Get categorical columns 
categoricals=df.select_dtypes(include=['object']).columns
#exclude 'Customer ID'
categoricals = [col for col in categoricals if col != 'Customer ID']
#loop for statistical summary
for col in categoricals:
    print(f"Summary for '{col}':")
    print(f"Unique values in '{col}': {df[col].nunique()}")
    print(f"Most frequent value (mode) for '{col}': {df[col].mode()[0]}")
    
    # Display value counts for the top N categories
    print(f"Value counts for the top {top_n} categories in '{col}':")
    print(df[col].value_counts().head(top_n))


# In[714]:


#LASTLY, UNIVARIATE FOR DATE
print(df['Transaction Date'].describe())


# In[715]:


#BIVARIATE ANALYSIS, COMPARING SPECIFICALLY THE TARGET OF 'IS FRAUD'
for feature in categoricals:
    #Limit to 10 categories
    top_n = 10
    top_categories = df[feature].value_counts().head(top_n).index
    df_filtered = df[df[feature].isin(top_categories)]

    # Group by feature and Is Fraudulent to get counts
    counts = df_filtered.groupby([feature, 'Is Fraudulent']).size().unstack().fillna(0)
    
    plt.figure(figsize=(8, 6))
    sns.countplot(data=df_filtered, x=feature, hue='Is Fraudulent')
    plt.title(f'{feature} vs Fraudulent Transactions')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()


# In[716]:


#BIVARIATE FOR NUMERICALS
for col in numericals:
    plt.figure(figsize=(8, 6))
    sns.boxplot(x='Is Fraudulent', y=col, data=df)
    plt.title(f'{col} vs Fraudulent Transactions')
    plt.xlabel('Is Fraudulent')
    plt.ylabel(col)
    plt.show()


# In[717]:


#TOP DATES
top_dates = df['Transaction Date'].value_counts().head(top_n).index

#filter
df_filtered = df[df['Transaction Date'].isin(top_dates)]

#BIVARIATE FOR DATE
plt.figure(figsize=(12, 6))
sns.countplot(x='Transaction Date', hue='Is Fraudulent', data=df_filtered)
plt.title(f'Fraudulent Transactions by Transaction Date (Top {top_n} Dates)')
plt.xlabel('Transaction Date')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# In[718]:


df['day_of_week'] = df['Transaction Date'].dt.dayofweek
df['hour'] = df['Transaction Date'].dt.hour
df['month'] = df['Transaction Date'].dt.month 
df=df.drop('Transaction Date', axis=1)
print(df.head())


# In[719]:


def bivariate_analysis_for_date(df, feature):
    plt.figure(figsize=(12, 6))
    sns.countplot(x=feature, hue='Is Fraudulent', data=df)
    plt.title(f'Fraudulent Transactions by {feature}')
    plt.xlabel(feature)
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    plt.show()

#Bivariate analysis for DATE
features_to_analyze = ['day_of_week', 'hour', 'month']

for feature in features_to_analyze:
    bivariate_analysis_for_date(df, feature)


# In[720]:


#LABEL ENCODING FOR CATEGORICALS
le = LabelEncoder()
ordinals = [col for col in categoricals if col not in ['Payment Method', 'Product Category', 'Device Used']]
#Loop each categorical
for col in ordinals:
    df[col] = le.fit_transform(df[col])

print(df.head())


# In[721]:


# Apply one-hot encoding for categorical columns that should be one-hot encoded
categoricals = ['Payment Method', 'Product Category', 'Device Used']

#ONE HOT ENCODING
df = pd.get_dummies(df, columns=categoricals, drop_first=True)
print(df.head())


# In[722]:


# Convert all Boolean columns (True/False) to 1/0
df = df.map(lambda x: 1 if x is True else (0 if x is False else x))

print(df.head())


# In[723]:


#check count
fraud_count = df_invalid_age['Is Fraudulent'].value_counts()
print(fraud_count)


# In[724]:


# Set X and y
X = df.drop('Is Fraudulent', axis=1)
y = df['Is Fraudulent']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print('data split')


# In[725]:


# Initialize RandomUnderSampler
under_sampler = RandomUnderSampler(sampling_strategy='auto', random_state=42)

# Apply Random Under-sampling to balance the dataset
X_train_under, y_train_under = under_sampler.fit_resample(X_train, y_train)

# Check class distribution after undersampling
print("Class distribution after Random Undersampling:")
print(y_train_under.value_counts())


# In[726]:


#XGBoost classifier model
xgb_model = xgb.XGBClassifier(objective='binary:logistic', 
                              eval_metric='logloss', 
                              max_depth=6, 
                              learning_rate=0.1, 
                              n_estimators=100,
                              subsample=0.8, 
                              colsample_bytree=0.8, 
                              scale_pos_weight=10)
#Train model
xgb_model.fit(X_train_under, y_train_under)

# Make predictions on the test set
y_pred = xgb_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))


# In[727]:


xgb_model = xgb.XGBClassifier(objective='binary:logistic')

#Hyperparameter
param_dist = {
    'max_depth': [3, 6, 9, 12],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [50, 100, 200],
    'subsample': [0.6, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.8, 1.0]
}

# Select features based on importance
#selector = SelectFromModel(xgb_model, threshold="mean", importance_getter="auto")
#X_train_selected = selector.fit_transform(X_train_under, y_train_under)
#X_test_selected = selector.transform(X_test)

#RandomizedSearchCV
random_search = RandomizedSearchCV(xgb_model, param_dist, n_iter=10, scoring='recall', cv=3, verbose=2, random_state=42)
random_search.fit(X_train_under, y_train_under)

#Best parameters and best score
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

#evaluation
y_pred = random_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
auc_score = roc_auc_score(y_test, y_pred)
print(f"Area Under the ROC Curve (AUC): {auc_score:.4f}")


# In[728]:


#feature importances
xgb_model = random_search.best_estimator_

plt.figure(figsize=(10, 8))
xgb.plot_importance(xgb_model, importance_type='weight', max_num_features=10)
plt.title("Top 10 Features Feature Importance")
plt.show()


# In[729]:


#predicted probabilities for fraud
y_scores = random_search.predict_proba(X_test)[:, 1]

#precision recall curve setup
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', label=f'Precision-Recall curve (AUC = {auc(recall, precision):.2f})')
plt.fill_between(recall, precision, color='blue', alpha=0.2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# In[730]:


#RandomizedSearchCV
random_search = RandomizedSearchCV(xgb_model, param_dist, n_iter=10, scoring='precision', cv=3, verbose=2, random_state=42)
random_search.fit(X_train_under, y_train_under)

#Best parameters and best score
print("Best Parameters:", random_search.best_params_)
print("Best Score:", random_search.best_score_)

#evaluation
y_pred = random_search.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy Score: {accuracy:.4f}")
print("Classification Report:")
print(classification_report(y_test, y_pred))
auc_score = roc_auc_score(y_test, y_pred)
print(f"Area Under the ROC Curve (AUC): {auc_score:.4f}")


# In[731]:


#predicted probabilities for fraud
y_scores = random_search.predict_proba(X_test)[:, 1]
#set up for precision recall curve
precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

plt.figure(figsize=(8, 6))
plt.plot(recall, precision, color='blue', label=f'Precision-Recall curve (AUC = {auc(recall, precision):.2f})')
plt.fill_between(recall, precision, color='blue', alpha=0.2)
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc='best')
plt.grid(True)
plt.show()


# In[ ]:




