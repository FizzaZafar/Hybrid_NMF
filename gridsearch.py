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
import pipelines

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

def pipelines(grid_search_pipelines):
    full_data, data_sub = pipelines.read_data(DATA_TRAIN, SAMPLE_SUBMISSION)
