import pandas as pd
import numpy as np
import math
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mp
import ast

all_files=["gs_impute_results.csv", "gs_impute_results_1.csv", "gs_impute_results_2.csv", "gs_impute_results_3.csv", "gs_impute_results_4.csv", "gs_impute_results_5.csv","gs_impute_results_6.csv"]
df_global_nmf = pd.concat([pd.read_csv("../results/"+fn) for fn in all_files], ignore_index=True)
print(df_global_nmf.shape)
piv = pd.pivot_table(df_global_nmf, values="mean_test_rmse",index="param_n_epochs", columns="param_n_factors")
fig,ax = plt.subplots(1,1,figsize=(10,6))
#sns.scatterplot(x="param_n_epochs",y="param_n_factors",hue="mean_test_rmse",data=df_global_nmf)
sns.heatmap(piv, ax=ax, cmap="YlGnBu")
ax.set(xlabel="Epochs", ylabel="Factors")
ax.invert_yaxis()
plt.tight_layout()
fig.savefig("figs/Global_params.pdf", dpi=1000)
plt.show()