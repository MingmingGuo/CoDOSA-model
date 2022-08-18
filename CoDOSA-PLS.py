#!/usr/bin/env python
# -*- coding:utf-8 -*-
#author: MingmingGuo
#If you use CoDOSA model in your research, please cite the following paper:M. M. Guo, W. Cui*, Theoretical investigation of metal-nonmetal co-decorated graphyne for electrocatalytic water splitting by DFT study and CoDOSA model. Submitted, 2021.


#imports
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cross_decomposition import PLSRegression
from sklearn import datasets 
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score

#read  csv data
chemisorption_data = pd.read_csv("CoDOSA-OER.csv")
print(chemisorption_data.info(),chemisorption_data.columns)
print(chemisorption_data.describe())


#train test split
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(chemisorption_data,test_size=0.2,random_state=10)

#correlation analysis to the property
corr_matrix = chemisorption_data.corr()
print('Input variables correlation analysis:')
print(corr_matrix['predict_property'].sort_values(ascending=False))
print('predict_property')

#drop alloy symbols from train set
train_set_numdata = train_set.drop('System',axis = 1)

#split input data and outputs
train_set_numdata_inputs = train_set_numdata.drop('predict_property',axis = 1)
train_set_trueE = train_set_numdata['predict_property'].copy()

#Normalizing data using Standard Scaler
from sklearn.preprocessing import StandardScaler
stdscaler = StandardScaler()
train_set_scaled = stdscaler.fit_transform(train_set_numdata_inputs)

#RMSE
from sklearn.metrics import mean_squared_error
deltaE_pred = lin_reg.predict(train_set_scaled)

def rmse(trueE,predE):
    mse = mean_squared_error(trueE,predE)
    rmse = np.sqrt(mse)
    print('RMSE: ',rmse)
    return rmse

#R^2
def r2(trueE,predE):
    SStot = np.sum((trueE-np.mean(trueE))**2)
    SSres = np.sum((trueE-predE)**2)
    r2=1-SSres/SStot
    print('R^2: ',r2)
    return r2

print('\nLin reg:')
rmse(train_set_trueE,deltaE_pred)
r2(train_set_trueE,deltaE_pred)

#cross_val_score evaluation
from sklearn.model_selection import cross_val_score
scores = cross_val_score(lin_reg,train_set_scaled,train_set_trueE,scoring = 'neg_mean_squared_error',cv=10)
lin_reg_rmse_scores = np.sqrt(-scores)

def display_scores(scores):
    print('Cross Validation scores:')
    print('Scores:',scores)
    print('Mean:',scores.mean())
    print('Std',scores.std())

print('\nLin reg cross val scores:')
pls_r2 = r2_score(train_set_trueE,deltaE_pred)
pls_rmse =np.sqrt(mean_squared_error(train_set_trueE,deltaE_pred))
print('R2=',pls_r2)
print('RMSE=',pls_rmse)
display_scores(lin_reg_rmse_scores)

#Regression model, parameter
PLS_reg_setup = PLSRegression(scale=True)
param_grid = {'n_components': range(1, 4)}

#GridSearchCV Automatic tuning parameter
gsearch = GridSearchCV(PLS_reg_setup, param_grid)

#Train the model on the training set
PLS_reg = gsearch.fit(train_set_scaled,train_set_trueE)

PLS_reg_predection = PLS_reg.predict(train_set_scaled)
print('\nPLS:')
pls_r2 = r2_score(train_set_trueE,PLS_reg_predection)
pls_rmse =np.sqrt(mean_squared_error(train_set_trueE,PLS_reg_predection))
print('R2=',pls_r2)
print('RMSE=',pls_rmse)

### Model evaluation on test set
print('\n\nMODEL EVALUATION ON TEST SET')
#preparing test set data
test_set_numdata = test_set.drop('System',axis = 1)
test_set_input_vals = test_set_numdata.drop('predict_property',axis = 1)
test_set_trueE = test_set['predict_property'].copy()

#scaling test set data (only transforming no fit)
test_set_input_scaled = stdscaler.transform(test_set_input_vals)

#PLS 
PLS_reg_test_predictions = PLS_reg.predict(test_set_input_scaled)
print('\nPLS test set predictions')
#PLS_rmse = rmse(test_set_trueE,PLS_reg_test_predictions)
#r2(test_set_trueE,PLS_reg_test_predictions)
pls_r2 = r2_score(test_set_trueE,PLS_reg_test_predection)
pls_rmse =np.sqrt(mean_squared_error(test_set_trueE,PLS_reg_test_predections))
print('R2=',pls_r2)
print('RMSE=',pls_rmse)

#The output data
#df = pd.DataFrame(test_set_trueE)
#df.to_csv('forest_test_set_trueE.csv')
#df = pd.DataFrame(forest_reg_test_predictions)
#df.to_csv('forest_reg_test_predictions.csv')

#plotting real values vs predicted
plt.scatter(PLS_reg_predection,train_set_trueE,label = 'Training set')
plt.scatter(PLS_reg_test_predictions,test_set_trueE, label = 'Test set')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('CoDOSA-PLS model')
plt.plot([0,5],[0,5])
plt.legend()
plt.ylim(0,5)
plt.xlim(0,5)
plt.show()







