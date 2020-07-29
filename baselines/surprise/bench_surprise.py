import pandas as pd
import numpy as np
import math
import surprise
import json

def split(full_data):
  test_raw = full_data.sample(frac = 0.1) [["User","Movie","Prediction"]]
  train_raw = pd.merge(full_data,test_raw, indicator=True, how='outer').query('_merge=="left_only"').drop('_merge', axis=1)[["User","Movie","Prediction"]]
  return test_raw, train_raw

def start(train_file, test_file): #pass it data_train_clean.csv
  full_data = pd.read_csv(train_file)
  data_sub = pd.read_csv(test_file)
  full_data["User"] = full_data["User"]-1
  full_data["Movie"] = full_data["Movie"]-1
  data_sub["User"] = data_sub["User"]-1
  data_sub["Movie"] = data_sub["Movie"]-1
  return full_data, data_sub

def main():
  algos = {
    
    "NMF": {
      "func": surprise.NMF,
      "params": {"n_factors":200,
        "n_epochs":200, 
        "verbose":True, 
        "biased":True
      }
    },
    "SVDpp": {
      "func": surprise.SVDpp,
      "params": {}
    },
    "KNN": {
      "func": surprise.KNNBaseline,
      "params": {}
    },
    "Baseline": {
      "func": surprise.BaselineOnly,
      "params": {}
    },
    'Random': {
    "func": surprise.NormalPredictor,
    "params":{}
    }
  }
  full_data, data_sub = start("../data/data_train_clean.csv.csv","../data/sampleSubmission_clean.csv")
  num_folds = 5
  summary = "summary.txt" #JsonLines file
  f = open(summary,"w+")
  for func in algos:
    print(func)
    rmse = []
    for i in range(num_folds):
      test_raw, train_raw = split(full_data)
      reader = surprise.Reader(rating_scale=(1, 5))
      # The columns must correspond to user id, item id and ratings (in that order).
      dataset = surprise.Dataset.load_from_df(train_raw[["User","Movie","Prediction"]], reader)
      trainset = dataset.build_full_trainset()
      algo = algos[func]["func"](**(algos[func]["params"]))
      algo.fit(trainset)
      test_dataset = surprise.Dataset.load_from_df(test_raw[["User","Movie","Prediction"]], reader)
      trainset_test = test_dataset.build_full_trainset()
      testset = trainset_test.build_testset()
      predictions = algo.test(testset)
      rmse.append(surprise.accuracy.rmse(predictions))
    dat = {"algo": func,
           "params": algos[func]["params"],
           "rmse": np.mean(rmse)}
    f.write(json.dumps(dat)+"\n")
  f.close()

if __name__ == '__main__':
  main()