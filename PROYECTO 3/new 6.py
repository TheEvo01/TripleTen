CONTINUACION.........................................................................

# Crea la variable station_stat_full combinando id_name y good_stations_stat
id_name.columns = ['name', 'count']
station_stat_full = id_name.join(good_station_stat)

# Imprime las primeras cinco líneas para visualizar la tabla nueva
print(station_stat_full.head())
-------------------------------------------------------------------------------------

# Crea el histograma del recuento de visitas entre las gasolineras (30 contenedores)
station_stat_full.hist('count', bins=30)

# Crea el histograma del recuento de visitas entre las gasolineras (rango de 0 a 300 visitas)
station_stat_full.hist('count', bins=30, range=(0, 300))

---------------------------------------------------------------------------------------------

PARA SEGMENGAR Y CALCULOS ESTADISTICOS, LA MEDIANA DE LA MEDIANA POR GASOLINERA, PARA CONSEGUIR EL VALOR POR CADENA-------------

# Utiliza station_stat_full.query().pivot_table() para segmentar y calcular las estadísticas
good_stat2 = station_stat_full.query('count > 30').pivot_table(index='name', values='time_spent', aggfunc=['median', 'count'])

# Establece el atributo columns de good_stat2
good_stat2.columns = ['median_time', 'stations']

# Imprime las primeras 5 filas de good_stat2
print(good_stat2.head())

--------------------------------------------------------------------------------------------------------------------------------

3RA ESTIMACION AHORA PODEMOS COMPARAR TODOS LOS RESULTADOS----------------------------------------------------------------------
# Utiliza station_stat_full.query().pivot_table() para segmentar y calcular las estadísticas
good_stat2 = station_stat_full.query('count > 30').pivot_table(index='name', values='time_spent', aggfunc=['median', 'count'])

# Establece el atributo columns de good_stat2
good_stat2.columns = ['median_time', 'stations']

# Añade las dos columnas en good_stat2 a la tabla stat y guarda la tabla resultante como final_stat
final_stat = stat.join(good_stat2)

# Imprime todo final_stat y compara tus estimaciones
print(final_stat)

--------------------------------------------------------------------------------------------------------------------------------

RELACION ENTRE VISITAS Y TIEMPO MEDIANO DE VISITA-------------------------------------------------------------------------------

# Crea un diagrama de dispersión que muestre la relación entre el número de visitas y el tiempo mediano de visita
station_stat_full.plot(kind='scatter', x='count', y='time_spent', alpha=0.7, figsize=(10, 6), grid=True, title="Relación entre el Número de Visitas y el Tiempo Mediano de Visita")

--------------------------------------------------------------------------------------------------------------------------------

CALCULA LA CORRELACION QUE HAY ENTRE CANTIDAD DE VISITAS Y TIEMPO DE REPOSTAJE, USANDO LA CORRELACION DE PEARSON-----------------

print(station_stat_full['count'].corr(station_stat_full['time_spent']))

---------------------------------------------------------------------------------------------------------------------------------

CREA UNA MATRIZ DE DISPERCION USANDO UNA NUEVA VARIABLE-------------------------------------------------------------------------
# Crea la variable station_stat_multi
station_stat_multi = data.pivot_table(index='id', values=['time_spent', 'too_fast', 'too_slow'])

# Imprime la matriz de correlación
print(station_stat_multi.corr())

# Crea la matriz del gráfico de dispersión utilizando scatter_matrix de pandas.plotting
pd.plotting.scatter_matrix(station_stat_multi, figsize=(9, 9))

---------------------------------------------------------------------------------------------------------------------------------

MATRIZ DE DISPERCION LIMPIA------------------------------------------------------------------------------------------------------
# Copia las estimaciones de los tiempos de visitas medianos desde good_station_stat a station_stat_multi
station_stat_multi['good_time_spent'] = good_station_stat['time_spent']

# Imprime la matriz de correlación
print(station_stat_multi.corr())

# Crea la matriz del gráfico de dispersión
scatter_matrix = pd.plotting.scatter_matrix(station_stat_multi, figsize=(9, 9))

---------------------------------------------------------------------------------------------------------------------------------

# Calcula los resultados de las cadenas de los resultados de las gasolineras
# pero no el promedio de visitas a todas las gasolineras de una cadena
good_name_stat2 = station_stat_full.query('count > 30').pivot_table(
    index='name', values='time_spent', aggfunc=['median', 'count']
)
good_name_stat2.columns = ['median_time', 'stations']

# Une los cálculos a name_stat
final_stat = name_stat.join(good_name_stat2)

# Ordena la tabla final_stat en orden ascendente de median_time
final_stat_sorted = final_stat.sort_values(by='median_time')

# Crea un gráfico de barras de median_time
bar_plot = final_stat_sorted['median_time'].plot(kind='bar', figsize=(10, 5))

---------------------------------------------------------------------------------------------------------------------------------

ELIMINAMOS LAS CADENAS SIN VALOR-------------------------------------------------------------------------------------------------

# agregar el tiempo de visita de las cadenas en función de los resultados de cada gasolinera,
# en vez del tiempo medio de visita de todas las gasolineras de la cadena
good_name_stat2 = station_stat_full.query('count > 30').pivot_table(
    index='name', values='time_spent', aggfunc=['median', 'count']
)
good_name_stat2.columns = ['median_time', 'stations']
final_stat = name_stat.join(good_name_stat2)

# Elimina los valores NaN de la columna median_time en final_stat,
# ordena final_stat en orden ascendente de median_time,
# y crea un gráfico de barras de median_time
(final_stat
     .dropna(subset=['median_time'])
     .sort_values(by='median_time')
     .plot(y='median_time', kind='bar', figsize=(10, 5), grid=True)
)

---------------------------------------------------------------------------------------------------------------------------------

VER LA DISTRIBUCION DEL NUMERO DE GASOLINERAS POR CADENA-------------------------------------------------------------------------

(final_stat['stations']
     .plot(kind='hist', bins=100, figsize=(10, 5))
)

---------------------------------------------------------------------------------------------------------------------------------

SOLO CADENAS GRANDES-------------------------------------------------------------------------------------------------------------

big_nets_stat = final_stat[final_stat['stations'] > 10]
print(big_nets_stat)

---------------------------------------------------------------------------------------------------------------------------------

GRANDES CADENAS Y LAS DEMAS SE LLAMARAN OTHERS-----------------------------------------------------------------------------------
# agregar el tiempo de visita de las cadenas en función de los resultados de cada gasolinera,
# en vez del tiempo medio de visita de todas las gasolineras de la cadena
good_name_stat2 = station_stat_full.query('count > 30').pivot_table(
    index='name', values='time_spent', aggfunc=['median', 'count']
)
good_name_stat2.columns = ['median_time', 'stations']
final_stat = name_stat.join(good_name_stat2)

big_nets_stat = final_stat.query('stations > 10')

# Agrega una nueva columna group_name a station_stat_full
station_stat_full['group_name'] = station_stat_full.apply(
    lambda row: row['name'] if row['name'] in big_nets_stat.index else 'Others', axis=1
)

# Imprime las primeras cinco filas de station_stat_full
print(station_stat_full.head())

---------------------------------------------------------------------------------------------------------------------------------

# Crea la variable stat_grouped que repite el cálculo de good_name_stat2 pero agrupa en group_name
stat_grouped = station_stat_full.query('count > 30').pivot_table(index='group_name', values='time_spent', aggfunc=['median', 'count'])

# Cambia el nombre de las columnas en stat_grouped a time_spent y count
stat_grouped.columns = ['time_spent', 'count']

# Ordena stat_grouped en orden ascendente de time_spent
stat_grouped.sort_values('time_spent', inplace=True)

# Imprime stat_grouped
print(stat_grouped)

----------------------------------------------------------------------------------------------------------------------------------

CREAR GRAFICO FINAL PARA VER LOS RESULTADOS---------------------------------------------------------------------------------------

# Crear un gráfico circular (pie chart) usando solo pandas
stat_grouped.plot(y='count', kind='pie', figsize=(8, 8));

----------------------------------------------------------------------------------------------------------------------------------

COLUMNA GROUP_NAME (PRUEBA FINAL)-------------------------------------------------------------------------------------------------

# Agregar la columna group_name a good_data
good_data['group_name'] = good_data['name'].where(
    good_data['name'].isin(big_nets_stat.index), 'Others'
)

# Mostrar las primeras cinco filas de good_data
print(good_data.head())

----------------------------------------------------------------------------------------------------------------------------------

CONTRASTAR HISTOGRAMAS------------------------------------------------------------------------------------------------------------

# Agregar la columna group_name a good_data
good_data['group_name'] = good_data['name'].where(
    good_data['name'].isin(big_nets_stat.index), 'Others'
)

# Utilizar el bucle for para agrupar good_data por group_name y crear histogramas
for name, group_data in good_data.groupby('group_name'):
    group_data['time_spent'].hist(bins=50)
    
----------------------------------------------------------------------------------------------------------------------------------

CREA UN BUCLE PARA EXAMINAR LA DISTRIBUCION DE LOS TIEMPOS POR CADENA-------------------------------------------------------------

# Agregar la columna group_name a good_data
good_data['group_name'] = good_data['name'].where(
    good_data['name'].isin(big_nets_stat.index), 'Others'
)

# Utilizar el bucle for para agrupar good_data por group_name y crear histogramas
for name, group_data in good_data.groupby('group_name'):
    group_data.hist('time_spent', bins=50)
    
----------------------------------------------------------------------------------------------------------------------------------

# Crear histogramas separados por cadena de gasolineras
for name, group_data in good_data.groupby('group_name'):
    group_data.plot(y='time_spent', title=name, kind='hist', bins=50)
    
----------------------------------------------------------------------------------------------------------------------------------