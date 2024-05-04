print(df.query('(To != "Barcelona") and (Airline != "S7")'))

print(df.query('Has_luggage == False and Airline not in ["S7", "Rossiya"]'))

print(df.query('Travel_time_from < @max_time and Airline in ["Belavia", "S7", "Rossiya"]'))

print(data.sort_values(by='time_spent', ascending=False).head(10))

NEW------------------------------------------------------------------
sample = data.query('id == "3c1e4c52"')

print(len(sample))

---------------------------------------------------------------------
data.hist('time_spent', bins=100, range=(0, 1500))
---------------------------------------------------------------------

---------------------------------------------------------------------
data.hist('time_spent', bins=100, range=(0, 1500))
plt.show()
sample.hist('time_spent', bins=100, range=(0, 1500))
plt.show()
---------------------------------------------------------------------

-----------------------------------------------------------------------------------------
Segmentar los datos para usar los de tiempo inferior a 1000 segundos (unos 16 minutos)

data.query('time_spent < 1000')
-----------------------------------------------------------------------------------------

FORMATEO DE COLUMNA DATE TIME************************************************************
data['date_time'] = pd.to_datetime(data['date_time'], format='%Y%m%dT%H%M%S')
-----------------------------------------------------------------------------------------

para almacenar el dia de la semana-------------------------------------------------------

data['weekday'] = data['date_time'].dt.weekday

print(data['weekday'].value_counts())
-----------------------------------------------------------------------------------------



