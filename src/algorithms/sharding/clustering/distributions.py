import pandas as pd
import numpy as np
#df = pd.read_table('../bucket_distribution_100M.tsv',sep='\s+',usecols=[4],names=['size'])
df = pd.read_csv('../bucket_distribution_config_bigann_large.csv')
sizes = np.log(df['points'])
df1 = pd.DataFrame(sizes)
plt = df1.plot.hist(bins=1000)
plt.get_figure().savefig("dists_large_1000.png")
