import pandas as pd
import numpy as np
import math
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mp
import ast

df_baseline = pd.read_csv("../results/baselines.csv")
fig,ax1 = plt.subplots(1,1,figsize=(8,6))
sns.barplot(x="Algorithm",y="RMSE", hue="Type", data=df_baseline, ax=ax1)
#sns.barplot(x="Algorithm",y="Test RMSE", data=df_baseline, ax=ax2)
#plt.hist(x=[df_baseline["Test RMSE"], df_baseline["Validation RMSE"]],color=['r','b'], alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig("figs/Baselines.pdf", dpi=600)
plt.show()