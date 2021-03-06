# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
import math
from scipy import linalg
from fastai.collab import *
import tensorflow as tf
from fastai.callbacks import *

data_raw = pd.read_csv("../../data/data_train_clean.csv")
data_raw.shape

data_sub = pd.read_csv("../../data/sampleSubmission_clean.csv")
data_sub.shape

data = CollabDataBunch.from_df(data_raw, seed=42, valid_pct=0.2, user_name='User', item_name='Movie',
                               rating_name='Prediction', test=data_sub)

data.show_batch()

learn = collab_learner(data, use_nn = True, emb_szs={'User':30,'Movie':80} , layers=[200,128,64,16], y_range=(1, 5))
learn.callback_fns.append(CSVLogger)

learn.lr_find() # find learning rate
learn.recorder.plot(suggestion=True) # plot learning rate graph
min_grad_lr = learn.recorder.min_grad_lr

learn.fit_one_cycle(5,0.0001)
learn.fit_one_cycle(2,min_grad_lr)
