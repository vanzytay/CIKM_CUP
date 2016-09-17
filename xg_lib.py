import xgboost as xgb
from xgboost.sklearn import XGBClassifier
import matplotlib.pylab as plt
from sklearn import cross_validation, metrics   #Additional scklearn functions
from sklearn.grid_search import GridSearchCV   #Perforing grid search
import pandas as pd
import numpy as np
import matplotlib.pylab as plt
from matplotlib.pylab import rcParams
import operator
rcParams['figure.figsize'] = 12, 4

'''
Utility Library for XG Boost 
-----------------------------
Source =>
https://www.analyticsvidhya.com/blog/2016/03/complete-guide-parameter-tuning-xgboost-with-codes-python/
'''

def modelfit(alg, X, Y,useTrainCV=True, cv_folds=5, early_stopping_rounds=50, feature_index=None):
    if useTrainCV:
        xgb_param = alg.get_xgb_params()
        xgtrain = xgb.DMatrix(X, label=Y)
        cvresult = xgb.cv(xgb_param, xgtrain, num_boost_round=alg.get_params()['n_estimators'], nfold=cv_folds,
            metrics='auc', early_stopping_rounds=early_stopping_rounds, verbose_eval=True)
        alg.set_params(n_estimators=cvresult.shape[0])
    
    #Fit the algorithm on the data
    alg.fit(X, Y, eval_metric='auc')
        
    #Predict training set:
    dtrain_predictions = alg.predict(X)
    dtrain_predprob = alg.predict_proba(X)[:,1]
        
    #Print model report:
    print "\nModel Report"
    print "Accuracy : %.4g" % metrics.accuracy_score(Y, dtrain_predictions)
    print "AUC Score (Train): %f" % metrics.roc_auc_score(Y, dtrain_predprob)
    importance = alg.booster().get_fscore()
    importance = sorted(importance.items(), key=operator.itemgetter(1),reverse=True)
    for s in importance:
    	if(feature_index is not None):
    		_id = int(s[0][1:])
    		print('%s:%d',feature_index[_id],s[1])
    feat_imp = pd.Series(alg.booster().get_fscore()).sort_values(ascending=False)
    feat_imp.plot(kind='bar', title='Feature Importances')
    plt.ylabel('Feature Importance Score')
    plt.show()
