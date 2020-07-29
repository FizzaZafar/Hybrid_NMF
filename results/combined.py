import pandas as pd
import numpy as np
import math
import surprise
from surprise import NMF
from scipy import linalg
from numpy import dot
from sklearn.linear_model import LassoCV
from sklearn.neural_network import MLPRegressor
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error
from sklearn.cluster import KMeans
import json
import logging
import argparse

# NO_USER_CLUSTERS = 5 #Check --> increasing made it good
# NO_ITEM_CLUSTERS = 2 #Check --> increasing made it bad
# GLOBAL_NMF_K = 5 # --> decreasing made it good (given you don't do PCA)
# LOCAL_U_NMF_K = 20
# LOCAL_I_NMF_K = 20
# GLOBAL_NMF_EPOCHS = 15 # --> decreasing made it good
# LOCAL_U_NMF_EPOCHS = 8
# LOCAL_I_NMF_EPOCHS = 8

# NO_USER_CLUSTERS = [3,4,5,6,7] #Check --> increasing made it good
# NO_ITEM_CLUSTERS = [2,3] #Check --> increasing made it bad
# GLOBAL_NMF_K = [4,5,6] # --> decreasing made it good (given you don't do PCA)
# LOCAL_U_NMF_K = [10,20,30]
# LOCAL_I_NMF_K = [10,20,30]
# GLOBAL_NMF_EPOCHS = [5,10,15,20] #15 --> decreasing made it good
# LOCAL_U_NMF_EPOCHS = [5,8,10]#8
# LOCAL_I_NMF_EPOCHS = [5,8,10]#8


#Useless!
#USER_COMPONENTS = 8
#ITEM_COMPONENTS = 5
#K_FOLDS = 5

#For impute
N_FACTORS = 200 #change this back
N_EPOCHS = 200 #200 change this back
N_PCA = 40


def impute(train_raw):
  data_raw = train_raw[["User","Movie","Prediction"]]
  data = data_raw.pivot(index="User", columns="Movie", values="Prediction").to_numpy()
  reader = surprise.Reader(rating_scale=(1, 5))

  # The columns must correspond to user id, item id and ratings (in that order).
  dataset = surprise.Dataset.load_from_df(data_raw[["User","Movie","Prediction"]], reader)
  trainset = dataset.build_full_trainset()
  algo = NMF(n_factors=N_FACTORS, n_epochs=N_EPOCHS, verbose=True)
  algo.fit(trainset)
  print("Common NMF done")
  testset = trainset.build_anti_testset()
  predictions = algo.test(testset)
  predictions = pd.DataFrame(predictions)
  predictions.rename(columns={"uid":"User","iid":"Movie","est":"Prediction"},inplace=True)
  predictions = predictions[["User","Movie","Prediction"]]
  data = pd.concat([data_raw,predictions],ignore_index=True)
  data = data.pivot(index="User", columns="Movie", values="Prediction").to_numpy()
  preds=data.copy()
  print("Starting PCA")
  for i in range(1000):#change to 1000
    if i%50==0:
      print("iteration ",i)
    pca = PCA(n_components=N_PCA)
    data_red = pca.fit_transform(np.delete(data,i,axis=1))
    clf = LassoCV(normalize=True) #MLPRegressor(hidden_layer_sizes = (5), max_iter=10000) #, verbose=False, n_jobs=-1
    clf = clf.fit(data_red, data[:,i]) #i-th movie tree
    pred = clf.predict(data_red)
    preds[:,i]=pred

  preds_df = pd.DataFrame(preds).reset_index().melt('index')
  preds_df.rename(columns={"index":"User", "variable":"Movie", "value":"Prediction"},inplace=True) 
  return preds_df

def start(train_file, test_file): #pass it data_train_clean.csv
  full_data = pd.read_csv(train_file)
  data_sub = pd.read_csv(test_file)
  full_data["User"] = full_data["User"]-1
  full_data["Movie"] = full_data["Movie"]-1
  data_sub["User"] = data_sub["User"]-1
  data_sub["Movie"] = data_sub["Movie"]-1
  #imputed = impute(full_data)
  # test_raw = data_raw.sample(frac = 0.1) [["User","Movie","Prediction"]]
  # train_raw = pd.merge(data_raw,test_raw, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)[["User","Movie","Prediction"]]
  print("reeturn from start")
  return full_data, data_sub

def split(full_data):
  test_raw = full_data.sample(frac = 0.2) [["User","Movie","Prediction"]]
  train_raw = pd.merge(full_data,test_raw, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)[["User","Movie","Prediction"]]
  return test_raw, train_raw

def run(data_raw,data_val,data_sub,regressors_train,regressors_val,regressors_test,params,k_fold):
  """# Load Data"""
  #data_raw = pd.read_csv("train_raw.csv")[["User","Movie","Prediction"]]
  num_users = 10000
  num_movies = 1000
  all_users = [i for i in range(num_users)]
  all_movies = [i for i in range(num_movies)]
  index = pd.MultiIndex.from_product([all_users, all_movies], names = ["uid", "iid"])
  all_u_m = pd.DataFrame(index = index).reset_index()

  """# Global NMF"""
  reader = surprise.Reader(rating_scale=(1, 5))
  # The columns must correspond to user id, item id and ratings (in that order).
  dataset = surprise.Dataset.load_from_df(data_raw[["User","Movie","Prediction"]], reader)
  trainset = dataset.build_full_trainset()
  algo = NMF(n_factors=params["GLOBAL_NMF_K"], n_epochs=params["GLOBAL_NMF_EPOCHS"], verbose=True)
  #cross_validate(algo,dataset,measures=['rmse'],cv=5,return_train_measures=True,verbose=True)
  algo.fit(trainset)
  print("global NMF done")
  """# Alternate to PCA"""

  U_red = algo.pu
  V_red = algo.qi

  # User Clustering
  user_clusters = params["NO_USER_CLUSTERS"]
  kmeans_u = KMeans(n_clusters=user_clusters, random_state=0).fit(U_red)

  item_clusters = params["NO_ITEM_CLUSTERS"]
  kmeans_i = KMeans(n_clusters=item_clusters, random_state=0).fit(V_red)

  data_raw["user cluster"] = data_raw.apply(lambda x:kmeans_u.labels_[x["User"]],axis=1)

  data_raw["item cluster"] = data_raw.apply(lambda x:kmeans_i.labels_[x["Movie"]],axis=1)

  """# Local Matrix Factorization"""

  n_factors = params["LOCAL_U_NMF_K"]
  global_df = pd.DataFrame()
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
    global_df = pd.concat([global_df, predictions_train, predictions_rest],ignore_index=False, copy=False)
  print("local fact users done")
  global_df = all_u_m.merge(global_df,how="left",on=["uid","iid"])

  global_df = global_df[["uid","iid","est"]]

  n_factors = params["LOCAL_I_NMF_K"]
  global_df_i = pd.DataFrame()
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
    global_df_i = pd.concat([global_df_i, predictions_train, predictions_rest],ignore_index=False, copy=False)
  print("local fact items done")
  global_df_i = global_df[["uid","iid"]].merge(global_df_i, how="left", on=["uid","iid"])

  global_df_i["est"].loc[global_df_i["est"].isnull()] = 0

  global_df_all = regressors_train
  # global_df_all["User"] = global_df_all["User"]-1
  # global_df_all["Movie"] = global_df_all["Movie"]-1

  merge1 = global_df_all.merge(data_raw,on=["User","Movie"],copy=False)

  merge1.rename(columns={"Prediction_x":"regressed","Prediction_y":"Prediction"},inplace=True)

  global_df = global_df[["uid","iid","est"]]

  global_df_i = global_df_i[["uid","iid","est"]]

  merge2 = global_df.merge(global_df_i,on=["uid","iid"],how="outer",copy=False)

  merge2.rename(columns={"est_x":"users","est_y":"items"},inplace=True)

  merge2["users"].loc[merge2["users"].isnull()] = 0
  merge2["items"].loc[merge2["items"].isnull()] = 0

  data_raw = merge1.merge(merge2,how="left",left_on=["User","Movie"],right_on=["uid","iid"],copy=False)

  data_raw=data_raw[["User","Movie","Prediction","regressed","users","items"]]
  data_raw = data_raw.iloc[:,~data_raw.columns.duplicated()]
  """# Models"""
  model = LassoCV(normalize=True).fit(data_raw[["regressed","items","users"]],data_raw[["Prediction"]].to_numpy().reshape((data_raw.shape[0],)))
  print("model done")
  """# Get Predictions for Hold Out Set"""
  data_val = data_val[["User","Movie","Prediction"]]

  global_df_all = regressors_val

  merge1 = global_df_all.merge(data_val,on=["User","Movie"],copy=False)

  merge1.rename(columns={"Prediction_x":"regressed","Prediction_y":"Prediction"},inplace=True)

  data_val = merge1.merge(merge2,how="left",left_on=["User","Movie"],right_on=["uid","iid"],copy=False)

  data_val["Prediction_new"] = model.predict(data_val[["regressed","items","users"]]).clip(1,5)
  data_val = data_val.iloc[:,~data_val.columns.duplicated()]
  rmse = mean_squared_error(data_val["Prediction_new"],data_val[["Prediction"]])
  # """# Get Predictions for Test Set"""

  # global_df_all = regressors_test

  # merge1 = global_df_all.merge(data_sub,on=["User","Movie"],copy=False)

  # merge1.rename(columns={"Prediction_x":"regressed","Prediction_y":"Prediction"},inplace=True)

  # data_sub = merge1.merge(merge2,how="left",left_on=["User","Movie"],right_on=["uid","iid"],copy=False)

  # data_sub["Prediction"] = model.predict(data_sub[["regressed","items","users"]]).clip(1,5)

  # sub_str = "submission" + "_" + str(k_fold) + ".csv"
  # data_sub[["User","Movie","Prediction"]].to_csv(sub_str,index=False)

  return rmse

def getRegressors(df_full,df_ref):
  df_res = df_full.merge(df_ref, how='inner',on=["User","Movie"]) #loc[preds_df["Id"].isin(Ids),["Id","Prediction"]]
  return df_res

# NO_USER_CLUSTERS = 5 #Check --> increasing made it good
# NO_ITEM_CLUSTERS = 2 #Check --> increasing made it bad
# GLOBAL_NMF_K = 5 # --> decreasing made it good (given you don't do PCA)
# LOCAL_U_NMF_K = 20
# LOCAL_I_NMF_K = 20
# GLOBAL_NMF_EPOCHS = 15 # --> decreasing made it good
# LOCAL_U_NMF_EPOCHS = 8
# LOCAL_I_NMF_EPOCHS = 8
def main(args):
  print("start")
  logging.basicConfig(filename='log'+args+".log", filemode='w', level=logging.INFO)
  full_data, data_sub = start("../data_train_clean.csv","../sampleSubmission_clean.csv")
  preds_mat = np.load('../preds.npz', allow_pickle=True)['pred']
  print("Read preds")
  preds = pd.DataFrame(preds_mat)
  preds.rename(columns={0:"User", 1:"Movie", 2:"Prediction"},inplace=True)
  print(preds.shape)
  f = open("../data/data_"+args+".txt","r")
  dat = f.read()
  print("1")
  all_params = json.loads(dat)
  test_raw_1 = pd.read_csv("../test1.csv")
  train_raw_1 = pd.read_csv("../train1.csv")
  test_raw_2 = pd.read_csv("../test2.csv")
  train_raw_2 = pd.read_csv("../train2.csv")
  print("2")
  for params in all_params:
    rmse = []
    regressors_train = getRegressors(preds,train_raw_1)
    regressors_val = getRegressors(preds,test_raw_1)
    regressors_test = getRegressors(preds,data_sub)
    print("3")
    rmse.append(run(train_raw_1,test_raw_1,data_sub,regressors_train,regressors_val,regressors_test,params,1))
    print("4")
    regressors_train = getRegressors(preds,train_raw_2)
    regressors_val = getRegressors(preds,test_raw_2)
    regressors_test = getRegressors(preds,data_sub)
    print("5")
    rmse.append(run(train_raw_2,test_raw_2,data_sub,regressors_train,regressors_val,regressors_test,params,2))
    print("6")
    params["MEAN_RMSE"]=np.mean(rmse)
    logging.info(params)

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument("-c",default="0")
  args = parser.parse_args()
  main(args.c)
