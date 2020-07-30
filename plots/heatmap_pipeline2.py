import pandas as pd
import numpy as np
import math
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mp
import ast

fn= "../results/cv.json"
with open(fn, "r") as f:
    data = ast.literal_eval(f.read())

df = pd.DataFrame(data)
df['LOCAL_U'] = list(zip(df["NO_USER_CLUSTERS"], df["LOCAL_U_NMF_K"], df["LOCAL_U_NMF_EPOCHS"]))
df['LOCAL_I'] = list(zip(df["NO_ITEM_CLUSTERS"], df["LOCAL_I_NMF_K"], df["LOCAL_I_NMF_EPOCHS"]))
piv = pd.pivot_table(df, values="MEAN_RMSE",index="LOCAL_U", columns="LOCAL_I")
fig,ax = plt.subplots(1,1,figsize=(10,6))
sns.heatmap(piv, cmap="YlGnBu", ax=ax)
ax.set(xlabel="(C,K,n) Items", ylabel="(C,K,n) Users")
plt.tight_layout()
fig.savefig("figs/Local_params.pdf", dpi=1000)
plt.show()

