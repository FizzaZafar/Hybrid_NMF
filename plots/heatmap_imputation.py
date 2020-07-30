import pandas as pd
import numpy as np
import math
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mp
import ast

df_global_nmf=pd.read_csv("../results/gs_impute_results.csv")
piv = pd.pivot_table(df_global_nmf.drop(df_global_nmf[df_global_nmf["param_n_factors"]>300].index), values="mean_test_rmse",index="param_n_epochs", columns="param_n_factors")
fig,ax = plt.subplots(1,1,figsize=(10,6))
sns.heatmap(piv, cmap=sns.cubehelix_palette(8), ax=ax)
ax.set(xlabel="Factors", ylabel="Epochs")
ax.invert_yaxis()
plt.tight_layout()
fig.savefig("figs/Global_params.pdf", dpi=1000)
plt.show()