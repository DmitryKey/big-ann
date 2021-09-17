import pandas as pd
import numpy as np
df = pd.read_table('../bucket_distribution_100M.tsv',sep='\s+',usecols=[4],names=['size'])
sizes = np.log(df['size'])
df1 = pd.DataFrame(sizes)
plt = df1.plot.hist(bins=1000)
plt.get_figure().savefig("dists.png")
