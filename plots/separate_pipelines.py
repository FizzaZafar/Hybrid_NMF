import pandas as pd
import numpy as np
import math
import json
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib as mp
import ast

mp.rcParams["font.size"]=12

df_pipelines = pd.read_csv("../results/pipelines.csv")
fig,ax1 = plt.subplots(1,1)
locs = {-0.4:-0.1, 0.6000000000000001:0.4, 1.6:0.9, -2.7755575615628914e-17:0, 0.9999999999999999:0.5, 2.0:1}

g = sns.barplot(x="Algorithm",y="RMSE", hue="Type", data=df_pipelines, ax=ax1)
centers=[0,1,2,0,1,2]
for i,p in enumerate(g.patches):
	p.set_width(0.1)
	x,y = p.get_xy()
	print(i,x,y)
	p.set_xy((locs[x],y))
	g.annotate(format(p.get_height(), '.4f'), (p.get_x() + p.get_width() / 2., p.get_height()), ha = 'center', va = 'center', xytext = (0, 10), textcoords = 'offset points', fontsize=10)

labs = ax1.get_xticklabels()
ax1.set_xticks([0,0.5,1])
ax1.set_xlim(left=-0.2, right=1.2)
ax1.set_ylim(bottom=0, top=1.2)	
ax1.set_xticklabels(["Pipeline1", "Pipeline2", "Combined"])
#sns.barplot(x="Algorithm",y="Test RMSE", data=df_baseline, ax=ax2)
#plt.hist(x=[df_baseline["Test RMSE"], df_baseline["Validation RMSE"]],color=['r','b'], alpha=0.5)
ax1.legend(loc="lower right")
plt.tight_layout()
fig.savefig("figs/Pipelines.pdf", dpi=600)
plt.show()