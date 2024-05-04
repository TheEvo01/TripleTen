import pandas as pd

data = pd.read_csv('/datasets/visits_eng.csv', sep='\t')
name_stat_median = data['time_spent'].mean()
data_sorted = data.sort_values(by='time_spent', ascending=False)
data_sorted['time_spent'].plot(kind='bar')