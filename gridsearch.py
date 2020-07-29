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

def impute(grid_search_impute_nmf):
    data_raw = pd.read_csv("data/data_train_clean.csv")
    data_sub = pd.read_csv("data/sampleSubmission_clean.csv")

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