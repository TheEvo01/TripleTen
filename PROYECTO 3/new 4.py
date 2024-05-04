GRAFICA MUESTRA PATRON EN HORA DE USO DE GASOLINERAS---------------------------------------------------
import pandas as pd

data = pd.read_csv('/datasets/visits_eng.csv', sep='\t')
data['date_time'] = pd.to_datetime(
    data['date_time'], format='%Y-%m-%dT%H:%M:%S'
)
data['local_time'] = data['date_time'] - pd.Timedelta(hours=7)

sample = data.query('id == "3c1e4c52"')

sample.plot(x='local_time', y='time_spent', style='o', ylim=(0, 1000), grid=True, figsize=(12, 6))
--------------------------------------------------------------------------------------------------------

NAVAJA OCCAM---------------------------------------------------------------------------------------------
import pandas as pd

data = pd.read_csv("/datasets/visits_eng.csv", sep="\t")
data['local_time'] = pd.to_datetime(
    data['date_time'], format='%Y-%m-%dT%H:%M:%S'
) - pd.Timedelta(hours=7)

data['date_hour'] = data['local_time'].dt.round('1H')

(data
     .query('id == "3c1e4c52"')
     .pivot_table(index='date_hour', values='time_spent', aggfunc='count')  # Usa data['date_hour'] en lugar de ('date_hour').dt.hour
     .plot(grid=True, figsize=(12, 5))
)
------------------------------------------------------------------------------------------------------------

COMPROBAR VIAJES RAPIDOS---------------------------------------------------------------------

import pandas as pd

data = pd.read_csv("/datasets/visits_eng.csv", sep="\t")
data['local_time'] = pd.to_datetime(
    data['date_time'], format='%Y-%m-%dT%H:%M:%S'
) - pd.Timedelta(hours=7)

def determine_too_fast(time_spent):
    if time_spent < 60:
        return True
    else:
        return False
    
data['too_fast'] = data['time_spent'].apply(determine_too_fast)

print(data.head())
----------------------------------------------------------------------------------------------

CALCULAR EL PORCENTAJE DE VISITAS RAPIDAS-----------------------------------------------------

import pandas as pd

data = pd.read_csv('/datasets/visits_eng.csv', sep='\t')
data['too_fast'] = data['time_spent'] < 60

too_fast_stat = data.pivot_table(index='id', values='too_fast')

print(too_fast_stat.head())
-----------------------------------------------------------------------------------------------

GRAFICAR EL HISTOGRAMA DE LOS RESULTADOS-------------------------------------------------------

import pandas as pd

data = pd.read_csv('/datasets/visits_eng.csv', sep='\t')
data['too_fast'] = data['time_spent'] < 60
too_fast_stat = data.pivot_table(index='id', values='too_fast')

too_fast_stat.hist(bins=30)
-------------------------------------------------------------------------------------------------

CONTRASTAR LAS VISITAS CORTAS CON LAS VISITAS LARGAS---------------------------------------------

import pandas as pd

data = pd.read_csv('/datasets/visits_eng.csv', sep='\t')
data['too_fast'] = data['time_spent'] < 60
too_fast_stat = data.pivot_table(index='id', values='too_fast')

too_fast_stat.hist(bins=30)

def determine_too_slow(time_spent):
    if time_spent > 1000:
        return True
    else:
        return False
    
data['too_slow'] = data['time_spent'].apply(determine_too_slow)

(data
     .pivot_table(index='id', values='too_slow')
     .hist(bins=30)
)
----------------------------------------------------------------------------------------------------


USAMOS PARA VER CUALES GASOLINERAS TENIAN MAYOR INCIDENCIA DE VISITAS RAPIDAS------------------------

print(too_fast_stat.sort_values('too_fast', ascending=False).head())
-----------------------------------------------------------------------------------------------------

EVALUAR LOS DATOS CON EL METODO DESCRIBE-------------------------------------------------------------

data.query('id == "792b6ded"').describe() 
-----------------------------------------------------------------------------------------------------


GUARDA EL DIA DE LA SEMANA EN UNA COLUMNA SEPARADA---------------------------------------------------

df['weekday'] = df['date'].dt.weekday
-----------------------------------------------------------------------------------------------------