import pandas as pd
import numpy as np
import math
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mp
import ast

df_baseline = pd.read_csv("../results/models.csv")
fig,ax1 = plt.subplots(1,1, figsize=(10,6))
g = sns.barplot(x="Algorithm",y="RMSE", hue="Type", data=df_baseline, ax=ax1)
for p in g.patches:
	g.annotate(format(p.get_height(), '.5f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points', fontsize=6)
#sns.barplot(x="Algorithm",y="Test RMSE", data=df_baseline, ax=ax2)
#plt.hist(x=[df_baseline["Test RMSE"], df_baseline["Validation RMSE"]],color=['r','b'], alpha=0.5)
plt.xticks(rotation=45)
plt.tight_layout()
fig.savefig("figs/Models.pdf", dpi=600)
plt.show()