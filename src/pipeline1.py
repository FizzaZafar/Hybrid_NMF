import pandas as pd
import numpy as np
import math
from scipy import linalg
from numpy import dot
import surprise
from surprise import NMF
import logging
from sklearn.model_selection import KFold
from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from common import read_data

DATA_TRAIN = "../data/data_train_clean.csv"
SAMPLE_SUBMISSION = "../data/sampleSubmission_clean.csv"

def do_nmf(data_raw,impute_params):
    data = data_raw.pivot(index="User", columns="Movie", values="Prediction").to_numpy()
    reader = surprise.Reader(rating_scale=(1, 5))
    dataset = surprise.Dataset.load_from_df(data_raw[["User","Movie","Prediction"]], reader)
    trainset = dataset.build_full_trainset()

    algo = NMF(n_factors=impute_params["FACTORS"], n_epochs=impute_params["EPOCHS"], verbose=True)
    algo.fit(trainset)

    testset = trainset.build_anti_testset()
    predictions = algo.test(testset)
    predictions = pd.DataFrame(predictions)

    predictions.rename(columns={"uid":"User","iid":"Movie","est":"Prediction"},inplace=True)
    predictions = predictions[["User","Movie","Prediction"]]

    data = pd.concat([data_raw,predictions],ignore_index=True)
    data = data.pivot(index="User", columns="Movie", values="Prediction").to_numpy()
    return data

def regress(data):
    logging.info("Start Fitting Regressors")
    all_trees = []
    mse = []
    preds=data.copy()
    for i in range(1000):
        if i%50==0:
            logging.info("iteration "+str(i))
            logging.info('Mean mse for last 50 iterations: '+str(np.mean(mse[-50:])))
        pca = PCA(n_components=40)
        data_red = pca.fit_transform(np.delete(data,i,axis=1))
        clf = LassoCV(normalize=True) 
        clf = clf.fit(data_red, data[:,i]) #i-th movie tree
        pred = clf.predict(data_red)
        preds[:,i]=pred
        mse.append(mean_squared_error(data[:,i],pred))
        all_trees.append(clf)
    return preds


def do(impute_params):
    logging.info("In pipeline1.do")
    data_raw,data_sub = read_data(DATA_TRAIN,SAMPLE_SUBMISSION)
    dense_data = do_nmf(data_raw,impute_params)
    preds = regress(dense_data)
    np.savez_compressed("../results/imputed_preds.npz",preds)
    logging.info('return from pipeline1.do')