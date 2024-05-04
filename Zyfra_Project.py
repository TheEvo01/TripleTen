#!/usr/bin/env python
# coding: utf-8

# # Descripcion
# 
# Prepara un prototipo de un modelo de machine learning para Zyfra. La empresa desarrolla soluciones de eficiencia para la industria pesada.
# El modelo debe predecir la cantidad de oro extraído del mineral de oro. Dispones de los datos de extracción y purificación.
# El modelo ayudará a optimizar la producción y a eliminar los parámetros no rentables.
# La siguiente lección trata sobre el proceso de depuración del mineral. Te tocará seleccionar la información importante para el desarrollo del modelo. 

# <div class="alert alert-block alert-info">
# 
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
# El proyecto estara enfocado en encontrar el modelo de prediccion que mejor se ajuste a los requerimientos de `Zyfra` para depurar el proceso de purificacion del mineral de oro y asi mejorar el rendimiento de la extraccion del mineral.
# </div>

# # Inicializacion

# In[30]:


# Cargar todas las librerías

import pandas as pd
from sklearn.metrics import make_scorer
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score
from sklearn.metrics import make_scorer
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt


# # Preprocesamiento de los Datos

# ## Carga de los Datos

# In[2]:


# Carga el archivo de datos en un DataFrame

df_training = pd.read_csv('/datasets/gold_recovery_train.csv')
df_test  = pd.read_csv('/datasets/gold_recovery_test.csv')
df_full = pd.read_csv('/datasets/gold_recovery_full.csv')


# ## Exploracion de los Datos

# In[3]:


# Acceder a la columna 'rougher.output.recovery' del conjunto de entrenamiento para obtener los valores de recuperación

df_training['rougher.output.recovery']


# In[4]:


# imprime la información general/resumida sobre el DataFrame

df_training.head()


# <div class="alert alert-block alert-info">
# 
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
# Aca hemos cargado las librerias necesarias para el proyecto, cargamos el dataframe que usaremos y definimos el tipo de informacion que contiene el dataframe a ser usado.
# </div>

# In[5]:


# Obtener información sobre los datos

print(df_training.info())
print(df_test.info())
print(df_full.info())


# ## Corregir los Datos

# In[6]:


# Llenar los valores faltantes en los conjuntos de entrenamiento y prueba, utilizando el método de propagación hacia adelante (forward fill)

df_training = df_training.fillna(method = 'ffill')
df_test = df_test.fillna(method = 'ffill')


# In[7]:


# columnas faltantes en el conjunto de prueba en comparación con el conjunto de entrenamiento

missed_test_columns = set(list(df_training.columns.values))-set(list(df_test.columns.values))
missed_test_columns


# In[8]:


# Identificar columnas faltantes en df_test
missed_columns = set(df_full.columns) - set(df_test.columns)

# Llenar las columnas faltantes en df_test con los datos correspondientes de df_full
for column in missed_columns:
    if column != 'date':  # Excluir la columna 'date'
        df_test[column] = df_full[column]

# Verificar que todas las columnas se han agregado correctamente
df_test.info()


# In[9]:


# Rellenar los valores faltantes con la mediana de cada columna
df_test_filled = df_test.fillna(df_test.median())

# Verificar que ya no hay valores faltantes
print(df_test_filled.info())


# <div class="alert alert-block alert-info">
# 
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
# En esta seccion nos dedicamos a corregirlos, abordamos los hallazgos con algunas soluciones convenientes para poder tener una data lo mas fiel y precisa posible, corregimos la disparidad en los valores de los `DF` usando el metodo de propagacion hacia delante `ffill`, como pudimos observar nos topamos con que el `df_test` tenia una falta de columnas algo grave, que nos impediria hacer la investigacion, lo que decidimos hacer fue ir a la fuente, digase el df_full, copiar las columnas necesarias para tener identicos los `DF` y trabajar de ahi en adelante en eliminar valores faltantes y demas, luego que teniamos el `df_test` con las columnas deseadas, procedimos a rellenar los valores faltantes con la mediana, ya que teniendo valores ausentes no ibamos a poder usar de manera satisfactoria este `DF`, al rellenar con la mediana evitamos tener un impacto demasiado significativo en el resultado de la investigacion.
# </div>

# ## Enriquecer los Datos

# In[10]:


# Convertir la columna 'date' a tipo datetime en los tres DataFrames
df_training['date'] = pd.to_datetime(df_training['date'])
df_test_filled['date'] = pd.to_datetime(df_test_filled['date'])
df_full['date'] = pd.to_datetime(df_full['date'])

# Verificar el cambio de tipo de datos
print(df_training.info())
print(df_test_filled.info())
print(df_full.info())


# <div class="alert alert-block alert-info">
# 
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
# En esta pequeña seccion del preprocesamiento de los datos nos dedicamos a simplemente cambiar el tipo de variable de la columna date para un tipo de dato mas util en caso de que se deba utilizar mas adelante, por eso volvimos esta variable a tipo `datetime`, asi en caso de ser necesario podremos utilizarla en la investigacion.
# </div>

# # Analizar los Datos

# In[12]:


# vamos a evaluar la data que tenemos usando el metodo `describe`

print(df_training.describe())
print(df_test_filled.describe())


# In[13]:


# Definir la función para visualizar la relación y calcular la correlación
def visualize_relationship(feature, target, data, dataset_name):
    plt.scatter(data[feature], data[target], alpha=0.5)
    plt.xlabel(feature)
    plt.ylabel(target)
    plt.title(f'Relación entre {feature} y {target} en el dataset {dataset_name}')
    plt.show()

def calculate_pearson_correlation(data, target):
    correlation = data.corr()[target].sort_values(ascending=False)
    return correlation

# Crear el gráfico y calcular la correlación para df_training
for feature in df_training.columns[1:]:  # Empezamos desde 1 para omitir la columna 'date'
    visualize_relationship(feature, 'rougher.output.recovery', df_training, "df_training")
    correlation = calculate_pearson_correlation(df_training, 'rougher.output.recovery')
    print(f"\nCorrelación de Pearson para '{feature}' en df_training:")
    print(correlation)


# In[14]:


# Crear el gráfico y calcular la correlación para df_test_filled
for feature in df_test_filled.columns[1:]:  # Empezamos desde 1 para omitir la columna 'date'
    visualize_relationship(feature, 'rougher.output.recovery', df_test_filled, "df_test_filled")
    correlation = calculate_pearson_correlation(df_test_filled, 'rougher.output.recovery')
    print(f"\nCorrelación de Pearson para '{feature}' en df_test_filled:")
    print(correlation)


# In[15]:


def handle_outliers(df):
    for column in df.columns:
        if column != 'date':
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            
            outliers_mask = (df[column] < lower_bound) | (df[column] > upper_bound)
            
            # Rellenar los valores atípicos con la mediana
            median = df[column].median()
            df.loc[outliers_mask, column] = median
            
    return df

# Aplicar el manejo de valores atípicos a los tres DataFrames
df_training = handle_outliers(df_training)
df_test_filled = handle_outliers(df_test_filled)
df_full = handle_outliers(df_full)


# In[16]:


# Verificar duplicados en df_training
duplicates_training = df_training.duplicated().sum()
print("Número de filas duplicadas en df_training:", duplicates_training)

# Verificar duplicados en df_test_filled
duplicates_test = df_test_filled.duplicated().sum()
print("Número de filas duplicadas en df_test_filled:", duplicates_test)


# <div class="alert alert-block alert-info">
# 
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
# Revisamos si habia datos atipicos y al observar que hay muchos valores anormales en la distribucion total tomamos la decision de abordar el tema utilizando el rango intercuartilico ya que probamos dejando estos valores y la precision del modelo mermo de manera significativa, lo que nos lleva a la obligacion de tener que lidiar con estos valores atipicos para poder tener mayor precision en el modelo y al abordar el tema, esto nos permitio manejar la la falta de precision adecuadamente, tambien revisamos si habia datos duplicados y no los hay, asi que podemos seguir adelante.
# </div>

# # Construye el Modelo

# ## Calculo de Recuperacion del Mineral

# In[17]:


def recovery_calculation(row):
    numerator = row['rougher.output.concentrate_au'] * (row['rougher.input.feed_au'] - row['rougher.output.tail_au'])
    denominator = row['rougher.input.feed_au'] * (row['rougher.output.concentrate_au'] - row['rougher.output.tail_au'])
    
    # Manejo de casos donde el denominador es cero
    if denominator == 0:
        recovery = 0
    else:
        recovery = numerator / denominator * 100
    
    # Manejo de casos donde la recuperación calculada es extremadamente alta
    max_recovery = 1000  # Define un límite superior razonable para la recuperación
    if recovery > max_recovery:
        recovery = max_recovery
    
    return recovery


# In[18]:


# Calcular la recuperación para cada fila del conjunto de entrenamiento

df_training['recovery_calculated'] = df_training.apply(recovery_calculation, axis=1)


# In[19]:


# Calcular el Error Absoluto Medio (MAE) entre las recuperaciones calculadas y las recuperaciones reales
mae = (df_training['recovery_calculated'] - df_training['rougher.output.recovery']).abs().mean()
print("Error Absoluto Medio (MAE):", mae)


# <div class="alert alert-block alert-info">
# 
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
# Aca hemos Creado la funcion que calculara cuanto mineral se podra recuperar y la ponemos a prueba, como se puede observar el resultado es muy pequeño, lo que nos deja en evidencia la efectividad y confiabilidad del procedimiento de calculo que usamos, aparte de que la calidad de los datos tambien queda en evidencia, significa que tenemos unos calculos bastante exactos como para confiar en ellos ya que el resultado se acerca bastante a cero.
# </div>

# ## Creacion de la Funcion sMAPE

# In[20]:


# características (features) del conjunto de prueba

features = df_test_filled.columns.values
features


# In[21]:


# Definir las etiquetas (targets) que se utilizarán para el entrenamiento del modelo, que son las recuperaciones de la etapa de flotación inicial y final

targets = ['rougher.output.recovery', 'final.output.recovery']


# In[22]:


# Preparar los datos de entrenamiento dividiendo las características (features) y las etiquetas (targets), y renombrar las columnas de las etiquetas

X_train = df_training[features].reset_index(drop = True)
y_train = df_training[targets].reset_index(drop = True)

y_train.columns = [0,1]


# In[23]:


# Mostrar las etiquetas (targets) preparadas para el entrenamiento del modelo

y_train


# In[24]:


# Eliminar la columna 'date' de las características (features) del conjunto de entrenamiento

X_train = X_train.drop(['date'], axis = 1)


# In[25]:


# Definir funciones para calcular el sMAPE y un sMAPE ponderado para su uso como métrica de evaluación del modelo

def compute_smape(y, y_pred):
    n = len(y)
    real = abs(y)
    pred = abs(y_pred)
    diff = abs(y - y_pred)
    smape =  (1/n)*np.sum(diff /((real + pred)/2))*100
    return smape

def smape_ponderado(y, y_pred):
    
    y_rougher = y.iloc[:,0]
    y_pred_rougher = y_pred[:,0]
    
    y_final = y.iloc[:,1]
    y_pred_final = y_pred[:,1]
    
    smape_rougher = compute_smape(y_rougher, y_pred_rougher)
    smape_final = compute_smape(y_final, y_pred_final)
    
    return (0.25*smape_rougher + 0.75*smape_final)


# In[26]:


# Crear un objeto 'smape_scorer' para utilizar la métrica sMAPE como puntuación de validación cruzada

smape_scorer = make_scorer(smape_ponderado)


# <div class="alert alert-block alert-info">
# 
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
# Creamos la funcion `compute_smape`y `smape_ponderado` con las cuales se estara enfrentando el modelo para ser evaluado previo su entrenamiento.
# </div>

# ## Entrenamiento y Escogencia del Modelo

# In[31]:


# Entrenar un modelo Random Forest utilizando las características (features) y las etiquetas (targets) del conjunto de entrenamiento
# rf_model = RandomForestRegressor(random_state=42)
# rf_model.fit(X_train, y_train)

# Realizar la validación cruzada del modelo Random Forest utilizando la métrica sMAPE ponderada
# rf_scores = cross_val_score(rf_model, X_train, y_train, scoring=smape_scorer, cv=5)

# Calcular y mostrar el puntaje sMAPE promedio y los puntajes sMAPE para cada iteración de la validación cruzada del modelo Random Forest
# rf_final_score = rf_scores.mean()

# print('Puntajes sMAPE para cada iteración:', rf_scores)
# print('Modelo Random Forest | sMAPE = {:.20f}'.format(rf_final_score))


# In[32]:


# 1. Entrenar un modelo de regresión Lasso utilizando las características (features) y las etiquetas (targets) del conjunto de entrenamiento
lasso_model = Lasso(random_state=42)
lasso_model.fit(X_train, y_train)

# 2. Realizar la validación cruzada del modelo de regresión Lasso utilizando la métrica sMAPE ponderada
lasso_scores = cross_val_score(lasso_model, X_train, y_train, scoring=smape_scorer, cv=5)

# 3. Calcular y mostrar el puntaje sMAPE promedio y los puntajes sMAPE para cada iteración de la validación cruzada del modelo de regresión Lasso
lasso_final_score = lasso_scores.mean()

print('Puntajes sMAPE para cada iteración:', lasso_scores)
print('Modelo de Regresión Lasso | sMAPE = {:.20f}'.format(lasso_final_score))


# In[33]:


# 1. Entrenar un modelo de regresión Ridge utilizando las características (features) y las etiquetas (targets) del conjunto de entrenamiento
ridge_model = Ridge(random_state=42)
ridge_model.fit(X_train, y_train)

# 2. Realizar la validación cruzada del modelo de regresión Ridge utilizando la métrica sMAPE ponderada
ridge_scores = cross_val_score(ridge_model, X_train, y_train, scoring=smape_scorer, cv=5)

# 3. Calcular y mostrar el puntaje sMAPE promedio y los puntajes sMAPE para cada iteración de la validación cruzada del modelo de regresión Ridge
ridge_final_score = ridge_scores.mean()

print('Puntajes sMAPE para cada iteración:', ridge_scores)
print('Modelo de Regresión Ridge | sMAPE = {:.20f}'.format(ridge_final_score))


# In[34]:


# 1. Entrenar un modelo de regresión lineal utilizando las características (features) y las etiquetas (targets) del conjunto de entrenamiento
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)

# 2. Realizar la validación cruzada del modelo de regresión lineal utilizando la métrica sMAPE ponderada
lr_scores = cross_val_score(lr_model, X_train, y_train, scoring=smape_scorer, cv=5)

# 3. Calcular y mostrar el puntaje sMAPE promedio y los puntajes sMAPE para cada iteración de la validación cruzada del modelo de regresión lineal
lr_final_score = lr_scores.mean()

print('Puntajes sMAPE para cada iteración:', lr_scores)
print('Modelo de Regresión Lineal | sMAPE = {:.20f}'.format(lr_final_score))


# <div class="alert alert-block alert-info">
# 
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
# Aca enfrentamos los modelos a las funciones creadas con anterioridad para evaluar su efectividad, tenemos unos hallazgos de lo mas interesantes, podemos ver que el modelo `Random Forest` practicamente es prohibitivo por el tiempo que toma para arrojar un resultado, aparte de que su precision no es la mejor, asi que por ello esta descartado, luego vemos los resultados del modelo `Regression Lasso`, este modelo tarda menos en ofrecernos un resultado, pero su precision es aun peor que el `Random Forest`, asi que aunque sea mas rapido en responder, esta descartado por su falta de precision, en comparacion con el anterior, despues vemos el modelo `Regression Ridge`, este es superior a los anteriores en ambas cosas, velocidad de entrega y precision, lo cual lo pone como candidato interesante en la evaluacion, por ultimo vemos el modelo `Linear Regression`, aun mejor que los 3 anteriores, es un modelo que nos provee el resultado en poco tiempo y tiene una precision sumamente alta, lo cual nos decanta por este al buscar un modelo que cumpla con ser preciso en sus calculos y que no sea lento, asi que nos quedaremos con este como mejor modelo de la evaluacion, lo cual nos lleva a pasar a la parte de conclusiones.
# </div>

# # Conclusiones

# <div class="alert alert-block alert-info">
# 
# <b>Respuesta del estudiante</b> <a class="tocSkip"></a>
# Como pudimos observar, tuvimos varios retos para poder completar de manera satisfactoria la tarea requerida, podriamos empezar mencionando el hecho de que la data en el `df_test` estaba incompleta con columnas faltantes, aparte de que en el `df_training tambien habia valores faltantes`, al abordar estos inconvenientes primero para el `df_training` usando el metodo `ffill` pudimos solventar parte del inconveniente en la data, no obstante para solventar el problema que tenia la data en el `df_test`, tuve que hacer algo mas que usar el metodo `ffill`, usamos el `df_full` como guia, primero para completar las columnas faltantes en el `df_test` y luego creamos un nuevo `DF` el cual llamamos `df_test_filled` en el cual aplicamos la `mediana`, para completar la data faltante en cada una de estas columnas, porque la mediana?, pues simple, porque es un metodo el cual nos permite incidir lo menos posible en el resultado final de la evaluacion, ya que deseamos que la evaluacion sea lo mas veras posible, luego de abordar esta complicacion, pasamos a cambiar el tipo de variable de la columna `date` aunque al final no la usamos, pero en caso de que nos fuera de utilidad en la evaluacion, ya esta columna estaria preparada para su uso, revisamos si habia datos duplicados y no los habia, asi que continuamos la travesia en nuestra tarea, ahora para hacer un analisis a profundidad de los datos procedimos a describir ambos `DF` que estariamos usando, para ello usamos el metodo `describe` y viendo que habia ciertas peculiaridades en los resultados, decidi graficar la relacion de las columnas de ambos `DF`, al ver los resultados de los graficos, podemos observar que nuestra data tiene muchos valores anormales en la distribucion total, por lo cual hubo que lidiar con este problema de las anomalias en la data, de que manera?, simple, escogi utilizar una funcion para corregir estos datos utilizando la mediana de ellos, despues de haber sido calculado el rango intercuartilico, para poder someter la correccion, solo a los datos que la necesitaban, luego para estar seguros de que la data estaba corregida correctamente procedimos a crear nuestra funcion `sMAPE`, la cual nos ayudaria a simplificar el proceso de entrenamiento y evaluacion de los diferentes modelos que probamos y asi dar con cual seria el mas indicado para usar en este caso en especifico, luego de esto entonces pasamos a la parte de entrenamiento y evaluacion de los diferentes modelos, dada la lentitud del modelo `Random Forest Regression`, dejamos bloqueado como comentario y dejamos el resultado que nos arrojo por aca: `Puntajes sMAPE para cada iteración: [0.16748125 0.16835748 0.14708783 0.1234615  0.16430958]
# Modelo Random Forest | sMAPE = 0.15413952831308916358`, como de todos modos aparte de ser lento es uno de los menos precisos, no fue escogido como mejor modelo en la evaluacion, al final de nuestra evaluacion, pudimos observar que el modelo `Linear Regression` si cumplia con las caracteristicas que estabamos buscando en esta evaluacion, precision y rapidez en la entrega de resultados, asi que como tenemos un modelo que cumple, podemos decir que los resultados fueron mas que satisfactorios.
# </div>
