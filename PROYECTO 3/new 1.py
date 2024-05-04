import pandas as pd

data = pd.read_csv('/datasets/visits_eng.csv', sep='\t')
total_visits = data.shape[0]
print('Número de visitas:', total_visits) 
total_stations = len(data['id'].unique())
print('Número de gasolineras:', total_stations)
print(data["date_time"].min(), data["date_time"].max()) 
total_days = 7
station_visits_per_day = total_visits / total_stations / total_days
print('Número de visitas por gasolinera por día:', station_visits_per_day)