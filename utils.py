# -*- coding: utf-8 -*-
"""
Created on Tue Mar 16 01:55:34 2021

@author: vishal
"""

import pandas as pd
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import cross_val_score,GridSearchCV

def model_scoring(model,model_name,x_train,y_train,x_val,y_val,Model_perf):
   
    temp = pd.DataFrame()
    
    temp['Model'] = [str(model_name)]
    temp['Train score'] = [cross_val_score(model,x_train,y_train,cv=5).mean()]
    temp['Validation score'] = [cross_val_score(model,x_val,y_val,cv=5).mean()]
    y_pred = model.predict_proba(x_val)[:,1]
    temp['roc_auc_Score'] = [roc_auc_score(y_val,y_pred)]
    
    Model_perf = pd.concat([Model_perf,temp],axis=0)
    return Model_perf


def model_building(estimator,param_grid,x_train,y_train,x_val,y_val,algo,Model_perf):
    model_grid = GridSearchCV(estimator=estimator,param_grid=param_grid,cv=5)
    model_grid.fit(x_train,y_train)
    best_model = model_grid.best_estimator_
    
    Model_perf1 = model_scoring(best_model,algo,x_train,y_train,x_val,y_val,Model_perf)
    return Model_perf1,best_model

def make_prediction(model,xtest,xtest_id,filename):
    file='Submission/submission'+'_'+ str(filename) + '.csv'
    
    test_pred = np.round(model.predict_proba(xtest)[:,1],2)
    submission = pd.DataFrame({'id':xtest_id, 'target':test_pred})
    submission.to_csv(file,index=False)
    