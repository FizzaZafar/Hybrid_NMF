import pandas as pd
import numpy as np
import math
from scipy import linalg
from numpy import dot
import surprise
from surprise import NMF
from surprise.model_selection import GridSearchCV
import json
import logging
import common
import pipeline2

DATA_TRAIN = "data/data_train_clean.csv"
SAMPLE_SUBMISSION = "data/sampleSubmission_clean.csv"
def impute(grid_search_impute_nmf):
    data_raw = pd.read_csv(DATA_TRAIN)
    data_sub = pd.read_csv(SAMPLE_SUBMISSION)

    param_grid = {'n_epochs': grid_search_impute_nmf["EPOCHS"],
                'n_factors':grid_search_impute_nmf["FACTORS"]}

    gs = GridSearchCV(NMF, param_grid, measures=['rmse'], cv=5, n_jobs=-1,
                    joblib_verbose=100)
    reader = surprise.Reader(rating_scale=(1, 5))
    dataset = surprise.Dataset.load_from_df(data_raw[["User","Movie","Prediction"]], reader)
    logging.info("Starting gridsearch for imputation")
    gs.fit(dataset)
    logging.info("Done gridsearch for imputation")
    df_res = pd.DataFrame(gs.cv_results)
    df_res.to_csv("results/gs_impute_results.csv", index=False)

def split(full_data,fraction):
  test_raw = full_data.sample(frac = fraction) [["User","Movie","Prediction"]]
  train_raw = pd.merge(full_data,test_raw, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)[["User","Movie","Prediction"]]
  return test_raw, train_raw

def pipe2(grid_search_pipelines):
    full_data, data_sub = common.read_data(DATA_TRAIN, SAMPLE_SUBMISSION)
    results = []
    for user_cluster in grid_search_pipelines["NO_USER_CLUSTERS"]:
        for item_cluster in grid_search_pipelines["NO_ITEM_CLUSTERS"]:
            for global_nmf_k in grid_search_pipelines["GLOBAL_NMF_K"]:
                for local_u_nmf_k in grid_search_pipelines["LOCAL_U_NMF_K"]:
                    for local_i_nmf_k in grid_search_pipelines["LOCAL_I_NMF_K"]:
                        for global_nmf_epochs in grid_search_pipelines["GLOBAL_NMF_EPOCHS"]:
                            for local_u_nmf_epochs in grid_search_pipelines["LOCAL_U_NMF_EPOCHS"]:
                                for local_i_nmf_epochs in grid_search_pipelines["LOCAL_I_NMF_EPOCHS"]:
                                    rmse = []
                                    params = {
                                        "NO_USER_CLUSTERS":user_cluster, 
                                        "NO_ITEM_CLUSTERS":item_cluster,
                                        "GLOBAL_NMF_K":global_nmf_k,
                                        "LOCAL_U_NMF_K":local_u_nmf_k,
                                        "LOCAL_I_NMF_K":local_i_nmf_k,
                                        "GLOBAL_NMF_EPOCHS":global_nmf_epochs,
                                        "LOCAL_U_NMF_EPOCHS":local_u_nmf_epochs,
                                        "LOCAL_I_NMF_EPOCHS":local_i_nmf_epochs
                                    }
                                    k=grid_search_pipelines["NO_FOLDS"]
                                    for i in range(0,k):
                                        test_raw, train_raw = split(full_data, k/100)
                                        train_raw.to_csv("data/train.csv",index=False)
                                        test_raw.to_csv("data/val.csv",index=False)
                                        rmse.append(pipeline2.do(params=params, gen_submission=False, validate=True))
                                    params["MEAN_RMSE"]=np.mean(rmse)
                                    results.append(params)
    f = open("cv.json","w")
    f.write(json.dumps(results))

                                    
