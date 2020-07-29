import pandas as pd
import numpy as np
import math
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mp
import ast

with open("../results/submissions.json","r") as f:
  data = json.load(f)

df_sub = pd.DataFrame(data)
df_sub_2 = df_sub.sort_values(by="MEAN_RMSE",ascending=False).reset_index(drop=True)
data=df_sub_2
x = data.index
mp.rcParams["font.size"]=15
fig,ax1 = plt.subplots(1,1,figsize=(8,6))
ax2 = ax1.twinx()
sns.lineplot(x=x,y="MEAN_RMSE", data=data, ax=ax1, label="Validation")
sns.lineplot(x=x,y="publicScore", data=data, ax=ax2, label="Test", color="green")
ax2.set_ylim(bottom=0.95)
ax1.set_ylabel("Validation RMSE")
ax2.set_ylabel("Public RMSE")
handles1, labels1 = ax1.get_legend_handles_labels()
handles2, labels2 = ax2.get_legend_handles_labels()
plt.legend(handles1+handles2, labels1+labels2)
plt.tight_layout()
fig.savefig("figs/Local_params_valid_test.pdf",dpi=1000)
plt.show()