# -*- coding: utf-8 -*-
"""
Created on Fri Jul 22 09:20:03 2022

@author: eshan
"""
import os
import pickle
import datetime
import numpy as np
import pandas as pd
import seaborn as sns
import scipy.stats as ss
import matplotlib.pyplot as plt

from sklearn.impute import KNNImputer
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

from tensorflow.keras.callbacks import TensorBoard, EarlyStopping
from tensorflow.keras.utils import plot_model

from bank_customer_module import FeatureSelection, ModelDevelopment
from bank_customer_module import ModelEvaluation
# pd.set_option('display.max_columns', None)

#%% Constants
CSV_PATH = os.path.join(os.getcwd(),'dataset','Train.csv')
MODEL_PATH = os.path.join(os.getcwd(),'model','model.h5')
LOGS_PATH = os.path.join(os.getcwd(),'logs',datetime.datetime.now().
                         strftime('%Y%m%d-%H%M%S'))
MMS_PATH = os.path.join(os.getcwd(),'model','mms.pkl')
OHE_PATH = os.path.join(os.getcwd(),'model','ohe.pkl')

#%% Data Loading
df = pd.read_csv(CSV_PATH)

#%% Data inspection

df.info()

df.describe().T
df.isna().sum() #High number of NaN (25831) in days_since_prev_campaign_contact
df.duplicated().sum() #No duplicates
df['id'].duplicated().sum() #no duplicated id

cat_col = ['id','job_type','marital','education','default','housing_loan',
           'personal_loan','communication_type','day_of_month','month',
           'num_contacts_in_campaign','num_contacts_prev_campaign',
           'prev_campaign_outcome','term_deposit_subscribed']
con_col = list(df.drop(labels=cat_col, axis=1).columns)

# for i in cat_col:
#     plt.figure()
#     sns.countplot(df[i])
#     plt.show()
    
for i in con_col:
    plt.figure()
    sns.distplot(df[i])
    plt.show()
    
# df.boxplot() 
# df.boxplot(column='balance') #large range of values & outliers. Negative values present
df.boxplot(column='last_contact_duration') #outliers present

#%% Data Cleaning

col_remove = ['id','days_since_prev_campaign_contact']
 
df_clean = df.drop(labels=col_remove, axis=1)
cat_col = [i for i in cat_col if i not in col_remove]
con_col = [i for i in con_col if i not in col_remove]

le = LabelEncoder()
for i in cat_col:
    temp = df_clean[i]
    temp[temp.notnull()] = le.fit_transform(temp[df_clean[i].notnull()])
    df_clean[i] = pd.to_numeric(df_clean[i],errors='coerce')
    PICKLE_SAVE_PATH = os.path.join(os.getcwd(),'encoder',i+'_encoder.pkl')
    pickle.dump(le, open(PICKLE_SAVE_PATH, 'wb'))
    
df_clean.info() #all columns are in int/float dtype

#Iterative imputer

# from sklearn.experimental import enable_iterative_imputer
# from sklearn.impute import IterativeImputer

# ii = IterativeImputer()
# df_ii = ii.fit_transform(df_clean)
# df_ii = pd.DataFrame(df_ii, index=None)
# df_ii.columns = df_clean.columns

# df_ii.describe().T
# df_ii.isna().sum()
# df_ii.duplicated().sum()

# II not suitable - negative values in num_contacts_in_campaign 

# KNN Imputation
knn_imputer = KNNImputer()
df_knn = knn_imputer.fit_transform(df_clean)
df_knn = pd.DataFrame(df_knn, index=None)
df_knn.columns = df_clean.columns
df_knn.isna().sum()
df_knn.info()
df_knn.describe().T

temp = df_knn[cat_col] #floats present in categorical data

for i in cat_col:
    df_knn[i] = np.floor(df_knn[i]).astype('int')
    
#%% Feature selection

# Target = term_deposit_subscribed (categorical)
y = df_knn['term_deposit_subscribed']

# Categorical vs Categorical - cramers v 

fs=FeatureSelection()
features = []

for i in cat_col:
    matrix = pd.crosstab(df_knn[i],y).to_numpy()
    if fs.cramers_corrected_stat(matrix) > 0.3:
        features.append(i)
        print(i)
        print(fs.cramers_corrected_stat(matrix))
        
# Continuous vs Categorical - Logistic Regression

for i in con_col:
    print(i)
    lr=LogisticRegression()
    lr.fit(np.expand_dims(df_knn[i],axis=-1),y)
    print(lr.score(np.expand_dims(df_knn[i],axis=-1),y))
    if lr.score(np.expand_dims(df_knn[i],axis=-1),y) > 0.7:
        features.append(i)

features.remove('term_deposit_subscribed') #remove target from list

#Features to use in model training: 'previous_campaign_outcome','customer_age', 
#'balance', 'last_contact_duration'

#%% Data preprocessing

X = df_knn[features]
y = df_knn['term_deposit_subscribed']

# MinMaxScaler
mms = MinMaxScaler()
X = mms.fit_transform(X)

with open(MMS_PATH,'wb')as file:
    pickle.dump(mms,file)

#OneHotEncoder
ohe = OneHotEncoder(sparse=False)
y = ohe.fit_transform(np.expand_dims(y, axis=-1))

pickle.dump(ohe,open(OHE_PATH,'wb'))

#Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                    test_size=0.3, 
                                                    random_state=123)

#%% Model Development

md = ModelDevelopment()

model = md.dl_model(X_train, y_train)

plot_model(model,to_file='model.png',show_shapes=True,show_layer_names=True)

model.compile(optimizer='adam',loss='categorical_crossentropy',metrics=['acc'])

#Callbacks
tensorboard_callback = TensorBoard(log_dir=LOGS_PATH, histogram_freq=1)
early_callback = EarlyStopping(monitor='loss',patience=5)

hist = model.fit(X_train,y_train,epochs=100,verbose=1,
                 validation_data=(X_test,y_test), 
                 callbacks=[tensorboard_callback,early_callback])

#%% Model Evaluation

#Plot validation
me = ModelEvaluation()
me.plot_hist_graph(hist)

y_pred = model.predict(X_test)

y_pred = np.argmax(y_pred,axis=1)
y_true = np.argmax(y_test,axis=1)

print(classification_report(y_true,y_pred))

conmx = confusion_matrix(y_true,y_pred)
disp = ConfusionMatrixDisplay(confusion_matrix=conmx)
disp.plot(cmap=plt.cm.Blues)
plt.show()

# low f1-score for category 1 - likely due to imbalanced target data 
# (Category 1 is less than 10% of dataset)

#%% Save Model

model.save(MODEL_PATH)

