QUITAR LA DATA IRRELEVANTE (VISITAS SUPER RAPIDAS A GASOLINERAS)------------------------------

import pandas as pd

data = pd.read_csv('/datasets/visits_eng.csv', sep='\t')

# filtra las visitas excepcionalmente rápidas y lentas y las gasolineras
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')

good_ids = too_fast_stat.query('too_fast < 0.5')

good_data = data.query('id in @good_ids.index')

print(len(good_data) / len(data))
-----------------------------------------------------------------------------------------------

VISITAS MENORES A 60seg O MAYORES A 1000seg SE EXCLUYEN----------------------------------------
import pandas as pd

data = pd.read_csv('/datasets/visits_eng.csv', sep='\t')

# filtra las visitas excepcionalmente rápidas y lentas y las gasolineras
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index')
good_data = good_data.query('60 <= time_spent <= 1000')

print(len(good_data))
-----------------------------------------------------------------------------------------------

HISTOGRAMA LIMPIO(MEDIANAS)--------------------------------------------------------------------

import pandas as pd

data = pd.read_csv('/datasets/visits_eng.csv', sep='\t')

# filtra las visitas excepcionalmente rápidas y lentas y las gasolineras
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index and 60 <= time_spent <= 1000')

# agrega datos por gasolinera individual y por cadenas
station_stat = data.pivot_table(
    index='id', values='time_spent', aggfunc='median'
)

good_stations_stat = good_data.pivot_table(index='id', values='time_spent', aggfunc='median')

good_stations_stat.hist(bins=50)
------------------------------------------------------------------------------------------------

ORDEN ASCENDENTE DE TIEMPO DESPUES DE LA MEDIANA, ORDENADO POR GASOLINERA-----------------------

import pandas as pd

data = pd.read_csv('/datasets/visits_eng.csv', sep='\t')

# filtra las visitas excepcionalmente rápidas y lentas y las gasolineras
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index and 60 <= time_spent <= 1000')

# agrega datos por cada gasolinera y por cadenas
station_stat = data.pivot_table(
    index='id', values='time_spent', aggfunc='median'
)
good_station_stat = good_data.pivot_table(
    index='id', values='time_spent', aggfunc='median'
)

good_stat = good_data.pivot_table(
    index='name', values='time_spent', aggfunc='median'
)

print(good_stat.sort_values('time_spent'))
--------------------------------------------------------------------------------------------------

MEDIANA DE LAS VISITAS POR CADENA-----------------------------------------------------------------

import pandas as pd

data = pd.read_csv('/datasets/visits_eng.csv', sep='\t')

# filtra visitas excesivamente rápidas y lentas y las gasolineras
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index and 60 <= time_spent <= 1000')

# agrega datos por cada gasolinera y por cadenas
station_stat = data.pivot_table(
    index='id', values='time_spent', aggfunc='median'
)
good_station_stat = good_data.pivot_table(
    index='id', values='time_spent', aggfunc='median'
)

stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(
    index='name', values='time_spent', aggfunc='median'
)

stat['good_time_spent'] = good_stat['time_spent']

print(stat)
---------------------------------------------------------------------------------------------------

DISTRIBUCION DE VISITAS POR CADENA-----------------------------------------------------------------

import pandas as pd

data = pd.read_csv('/datasets/visits_eng.csv', sep='\t')
data['local_time'] = pd.to_datetime(
    data['date_time'], format='%Y-%m-%dT%H:%M:%S'
) - pd.Timedelta(hours=7)
data['date_hour'] = data['local_time'].dt.round('1H')

# filtra visitas excesivamente rápidas y lentas y las gasolineras
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index and 60 <= time_spent <= 1000')

# agrega datos por cada gasolinera y por cadenas
station_stat = data.pivot_table(
    index='id', values='time_spent', aggfunc='median'
)
good_station_stat = good_data.pivot_table(
    index='id', values='time_spent', aggfunc='median'
)

stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(
    index='name', values='time_spent', aggfunc='median'
)
stat['good_time_spent'] = good_stat['time_spent']

id_name = good_data.pivot_table(
    index='id', values='name', aggfunc=['first', 'count']
)

print(id_name.head())
------------------------------------------------------------------------------------------------------

RENOMBRAR LAS COLUMNAS PARA DAR LOS TOQUES FINALES----------------------------------------------------

import pandas as pd

data = pd.read_csv('/datasets/visits_eng.csv', sep='\t')
data['local_time'] = pd.to_datetime(
    data['date_time'], format='%Y-%m-%dT%H:%M:%S'
) - pd.Timedelta(hours=7)
data['date_hour'] = data['local_time'].dt.round('1H')

# filtra visitas excesivamente rápidas y lentas y las gasolineras
data['too_fast'] = data['time_spent'] < 60
data['too_slow'] = data['time_spent'] > 1000
too_fast_stat = data.pivot_table(index='id', values='too_fast')
good_ids = too_fast_stat.query('too_fast < 0.5')
good_data = data.query('id in @good_ids.index and 60 <= time_spent <= 1000')

# agrega datos por cada gasolinera y por cadenas
station_stat = data.pivot_table(
    index='id', values='time_spent', aggfunc='median'
)
good_station_stat = good_data.pivot_table(
    index='id', values='time_spent', aggfunc='median'
)

stat = data.pivot_table(index='name', values='time_spent')
good_stat = good_data.pivot_table(
    index='name', values='time_spent', aggfunc='median'
)
stat['good_time_spent'] = good_stat['time_spent']

id_name = good_data.pivot_table(
    index='id', values='name', aggfunc=['first', 'count']
)

id_name.columns = ['name', 'count']

print(id_name.head())
-------------------------------------------------------------------------------------------------------







