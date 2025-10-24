import pandas as pd




df = pd.read_parquet('/Users/emullen98/Library/CloudStorage/Box-Box/storm/threshold_wk/per_cloud/thr_wk_2012-12-30--03-15-16--375/cr.part00002.parquet')


print(df.columns)
print(df['threshold'])