
import pandas as pd
import numpy as np
import math
import surprise
from surprise import NMF
from surprise import BaselineOnly
from scipy import linalg
from numpy import dot
from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
import json
import logging
import argparse
import os
import common


DATA_TRAIN = "data/data_train_clean.csv"
SAMPLE_SUBMISSION = "data/sampleSubmission_clean.csv"

num_users = 10000
num_movies = 1000

def get_regressors(df_full,df_ref):
  df_res = df_full.merge(df_ref, how='inner',on=["User","Movie"]) #loc[preds_df["Id"].isin(Ids),["Id","Prediction"]]
  return df_res

def get_u_v(data_raw,params):
  reader = surprise.Reader(rating_scale=(1, 5))
  # The columns must correspond to user id, item id and ratings (in that order).
  dataset = surprise.Dataset.load_from_df(data_raw[["User","Movie","Prediction"]], reader)
  trainset = dataset.build_full_trainset()
  algo = NMF(n_factors=params["GLOBAL_NMF_K"], n_epochs=params["GLOBAL_NMF_EPOCHS"], verbose=True)
  algo.fit(trainset)

  U_red = algo.pu
  V_red = algo.qi
  return (U_red, V_red)

def get_clusters(data_raw,params):
  user_clusters = params["NO_USER_CLUSTERS"]
  kmeans_u = KMeans(n_clusters=user_clusters, random_state=0).fit(U_red)

  item_clusters = params["NO_ITEM_CLUSTERS"]
  kmeans_i = KMeans(n_clusters=item_clusters, random_state=0).fit(V_red)

  data_raw["user cluster"] = data_raw.apply(lambda x:kmeans_u.labels_[x["User"]],axis=1)
  data_raw["item cluster"] = data_raw.apply(lambda x:kmeans_i.labels_[x["Movie"]],axis=1)
  return (user_clusters,item_clusters,data_raw)

def get_all_u_m():
  all_users = [i for i in range(num_users)]
  all_movies = [i for i in range(num_movies)]
  index = pd.MultiIndex.from_product([all_users, all_movies], names = ["uid", "iid"])
  all_u_m = pd.DataFrame(index = index).reset_index()
  return all_u_m

def user_factorization(data_raw,user_clusters,params):
  n_factors = params["LOCAL_U_NMF_K"]
  user_df = pd.DataFrame()
  for i in range(user_clusters):
    u_i = data_raw[data_raw["user cluster"]==i]
    print(u_i.shape)
    reader = surprise.Reader(rating_scale=(1, 5))
    dataset = surprise.Dataset.load_from_df(u_i[["User","Movie","Prediction"]], reader)
    trainset = dataset.build_full_trainset()
    algo = NMF(n_factors=n_factors, n_epochs=params["LOCAL_U_NMF_EPOCHS"], verbose=True)
    algo.fit(trainset)
    #u_i.rename(columns={"User":"uid","Movie":"iid","Prediction":"est"},inplace=True)
    testset = trainset.build_testset()
    print("Getting training preds: ")
    preds = algo.test(testset)
    predictions_train = pd.DataFrame(preds)
    testset = trainset.build_anti_testset()
    print("Getting test preds: ")
    preds = algo.test(testset)
    predictions_rest = pd.DataFrame(preds)
    user_df = pd.concat([user_df, predictions_train, predictions_rest],ignore_index=False, copy=False)
  print("local fact users done")
  all_u_m = get_all_u_m()
  user_df = all_u_m.merge(user_df,how="left",on=["uid","iid"])
  user_df = user_df[["uid","iid","est"]]
  return user_df

def item_factorization(data_raw,item_clusters,user_df,params):
  n_factors = params["LOCAL_I_NMF_K"]
  item_df = pd.DataFrame()
  for i in range(item_clusters):
      i_i = data_raw[data_raw["item cluster"]==i]
      reader = surprise.Reader(rating_scale=(1, 5))
      dataset = surprise.Dataset.load_from_df(i_i[["User","Movie","Prediction"]], reader)
      trainset = dataset.build_full_trainset()
      algo = NMF(n_factors=n_factors, n_epochs=params["LOCAL_I_NMF_EPOCHS"], verbose=True)
      algo.fit(trainset)
      #i_i.rename(columns={"User":"uid","Movie":"iid","Prediction":"est"},inplace=True)
      testset = trainset.build_testset()
      print("Getting training preds: ")
      preds = algo.test(testset)
      predictions_train = pd.DataFrame(preds)
      testset = trainset.build_anti_testset()
      print("Getting test preds: ")
      preds = algo.test(testset)
      predictions_rest = pd.DataFrame(preds)
      item_df = pd.concat([item_df, predictions_train, predictions_rest],ignore_index=False, copy=False)
      print("local fact items done")
  item_df = user_df[["uid","iid"]].merge(item_df, how="left", on=["uid","iid"])
  item_df["est"].loc[item_df["est"].isnull()] = 0
  return item_df

def merge(data_raw,regressors_train,user_df,item_df):
  global_df_all = regressors_train
  merge1 = global_df_all.merge(data_raw,on=["User","Movie"],copy=False)
  merge1.rename(columns={"Prediction_x":"regressed","Prediction_y":"Prediction"},inplace=True)
  user_df = user_df[["uid","iid","est"]]
  item_df = item_df
  merge2 = user_df.merge(item_df,on=["uid","iid"],how="outer",copy=False)[["uid","iid","est"]]
  merge2.rename(columns={"est_x":"users","est_y":"items"},inplace=True)
  merge2["users"].loc[merge2["users"].isnull()] = 0
  merge2["items"].loc[merge2["items"].isnull()] = 0
  data_raw = merge1.merge(merge2,how="left",left_on=["User","Movie"],right_on=["uid","iid"],copy=False)
  data_raw=data_raw[["User","Movie","Prediction","regressed","users","items"]]
  data_raw = data_raw.iloc[:,~data_raw.columns.duplicated()]  
  return (merge2,data_raw)

def model(data_raw):
  model = ElasticNetCV(alphas=np.linspace(0.0083,0.1,100)).fit(data_raw[["regressed","items","users"]],data_raw[["Prediction"]].to_numpy().reshape((data_raw.shape[0],)))
  return model

def gen_submission(model,data_sub,regressors_test,merge2):
  global_df_all = regressors_test
  merge1 = global_df_all.merge(data_sub,on=["User","Movie"],copy=False)
  merge1.rename(columns={"Prediction_x":"regressed","Prediction_y":"Prediction"},inplace=True)
  data_sub = merge1.merge(merge2,how="left",left_on=["User","Movie"],right_on=["uid","iid"],copy=False)
  data_sub = data_sub.iloc[:,~data_sub.columns.duplicated()]
  data_sub["Prediction"] = model.predict(data_sub[["regressed","items","users"]]).clip(1,5)
  data_sub = data_sub.iloc[:,~data_sub.columns.duplicated()]
  # data_sub = data_sub[["User","Movie","Prediction"]]
  data_sub["User"] = data_sub["User"]+1
  data_sub["Movie"] = data_sub["Movie"]+1
  data_sub["Id"] = data_sub.apply(lambda x:"r"+str(int(x["User"]))+"_c"+str(int(x["Movie"])),axis=1)
  sub_str = "results/submission.csv"
  data_sub[["Id","Prediction"]].to_csv(sub_str,index=False)


def do(params,gen_submission):
  data_raw, data_sub = common.read_data(DATA_TRAIN,SAMPLE_SUBMISSION)
  preds_mat = np.load('results/imputed_preds.npz', allow_pickle=True)['arr_0']
  preds = pd.DataFrame(preds_mat).reset_index().melt('index')
  preds.rename(columns={"index":"User", "variable":"Movie", "value":"Prediction"},inplace=True)
  regressors_train = get_regressors(preds,full_data)
  U_red, V_red = get_u_v(data_raw,params)
  user_clusters, item_clusters, data_raw = get_clusters(data_raw, params)
  user_df = user_factorization(data_raw,user_clusters,params)
  item_df = item_factorization(data_raw,item_clusters,user_df,params)
  merge2,data_raw = merge(data_raw,regressors_train,user_df,item_df)
  model = model(data_raw)
  if gen_submission:
    regressors_test = get_regressors(preds,data_sub)
    gen_submission(model,data_sub,regressors_test,merge2)
  


