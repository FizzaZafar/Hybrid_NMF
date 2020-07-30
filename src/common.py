
import pandas as pd
import numpy as np
from sklearn.ensemble import StackingRegressor, GradientBoostingRegressor

import sklearn.linear_model as sl
models = { "LinearRegression": sl.LinearRegression(),
           "LassoCV" : sl.LassoCV(),
           "ElasticNetCV" : sl.ElasticNetCV(alphas=np.linspace(0.038,0.1,100)),
           "StackingRegressor" : StackingRegressor([('SGD',sl.SGDRegressor()),('GBR', GradientBoostingRegressor())],verbose=1),
           "RidgeCV" : sl.RidgeCV(),
           "SGDRegressor" : sl.SGDRegressor(),
           "Perceptron" : sl.Perceptron()
}

def read_data(train_file, test_file): 
    full_data = pd.read_csv(train_file)
    data_sub = pd.read_csv(test_file)
    full_data["User"] = full_data["User"]-1
    full_data["Movie"] = full_data["Movie"]-1
    data_sub["User"] = data_sub["User"]-1
    data_sub["Movie"] = data_sub["Movie"]-1
    return full_data, data_sub