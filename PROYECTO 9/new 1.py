Comprueba la hipótesis de que el importe promedio de compra ha aumentado. 

import pandas as pd
from scipy import stats as st

# Datos de las muestras
sample_before = pd.Series([
    436, 397, 433, 412, 367, 353, 440, 375, 414, 
    410, 434, 356, 377, 403, 434, 377, 437, 383,
    388, 412, 350, 392, 354, 362, 392, 441, 371, 
    350, 364, 449, 413, 401, 382, 445, 366, 435,
    442, 413, 386, 390, 350, 364, 418, 369, 369, 
    368, 429, 388, 397, 393, 373, 438, 385, 365,
    447, 408, 379, 411, 358, 368, 442, 366, 431,
    400, 449, 422, 423, 427, 361, 354])

sample_after = pd.Series([
    439, 518, 452, 505, 493, 470, 498, 442, 497, 
    423, 524, 442, 459, 452, 463, 488, 497, 500,
    476, 501, 456, 425, 438, 435, 516, 453, 505, 
    441, 477, 469, 497, 502, 442, 449, 465, 429,
    442, 472, 466, 431, 490, 475, 447, 435, 482, 
    434, 525, 510, 494, 493, 495, 499, 455, 464,
    509, 432, 476, 438, 512, 423, 428, 499, 492, 
    493, 467, 493, 468, 420, 513, 427])

# Imprimir los valores medios anteriores y posteriores
print("La media de antes:", sample_before.mean())
print("La media de después:", sample_after.mean())

# Nivel crítico de significación
alpha = 0.05

# Realizar la prueba t de Student
results = st.ttest_ind(sample_before, sample_after)

# Obtener el valor p
pvalue = results.pvalue / 2

print('p-value: ', pvalue)

# Comprobar la hipótesis
if pvalue < alpha:
    print("La hipótesis nula se rechaza, a saber, es probable que el importe promedio de las compras aumente")
else:
    print("La hipótesis nula no se rechaza, a saber, es poco probable que el importe medio de las compras aumente")
***********************************************************************************************************
Construye un intervalo de confianza del 95 % para el importe promedio de compra en KicksYouCanPayRentWith
después de la implementación de la mascota de las zapatillas.

import pandas as pd
from scipy import stats as st

sample = pd.Series([
    439, 518, 452, 505, 493, 470, 498, 442, 497, 
    423, 524, 442, 459, 452, 463, 488, 497, 500,
    476, 501, 456, 425, 438, 435, 516, 453, 505, 
    441, 477, 469, 497, 502, 442, 449, 465, 429,
    442, 472, 466, 431, 490, 475, 447, 435, 482, 
    434, 525, 510, 494, 493, 495, 499, 455, 464,
    509, 432, 476, 438, 512, 423, 428, 499, 492, 
    493, 467, 493, 468, 420, 513, 427])

print('Media:', sample.mean())

# Construir intervalo de confianza del 95%
confidence_interval = st.t.interval(0.95, len(sample)-1, loc=sample.mean(), scale=st.sem(sample))

print('Intervalo de confianza del 95 %:', confidence_interval)
***********************************************************************************************************
Utilizando la técnica del bootstrapping, crea 10 submuestras y un cuantil del 99 % para cada una de ellas.

import pandas as pd
import numpy as np

data = pd.Series([
    10.7 ,  9.58,  7.74,  8.3 , 11.82,  9.74, 10.18,  8.43,  8.71,
     6.84,  9.26, 11.61, 11.08,  8.94,  8.44, 10.41,  9.36, 10.85,
    10.41,  8.37,  8.99, 10.17,  7.78, 10.79, 10.61, 10.87,  7.43,
     8.44,  9.44,  8.26,  7.98, 11.27, 11.61,  9.84, 12.47,  7.8 ,
    10.54,  8.99,  7.33,  8.55,  8.06, 10.62, 10.41,  9.29,  9.98,
     9.46,  9.99,  8.62, 11.34, 11.21, 15.19, 20.85, 19.15, 19.01,
    15.24, 16.66, 17.62, 18.22, 17.2 , 15.76, 16.89, 15.22, 18.7 ,
    14.84, 14.88, 19.41, 18.54, 17.85, 18.31, 13.68, 18.46, 13.99,
    16.38, 16.88, 17.82, 15.17, 15.16, 18.15, 15.08, 15.91, 16.82,
    16.85, 18.04, 17.51, 18.44, 15.33, 16.07, 17.22, 15.9 , 18.03,
    17.26, 17.6 , 16.77, 17.45, 13.73, 14.95, 15.57, 19.19, 14.39,
    15.76])

state = np.random.RandomState(12345)

for i in range(10):
    subsample = data.sample(frac=1, replace=True, random_state=state)
    print(subsample.quantile(0.99))
***********************************************************************************************************
Utilizando la técnica del bootstrapping, encuentra el intervalo de confianza del 90 % para el cuantil del 99 %.
Guarda su valor más bajo en la variable lower, y el valor más alto en la variable upper. 
Imprímelos (en el precódigo).
Llama a la función quantile() dos veces: primero, para obtener el cuantil del 99 % de cada submuestra, 
y luego, para obtener el intervalo de confianza.

import pandas as pd
import numpy as np

data = pd.Series([
    10.7 ,  9.58,  7.74,  8.3 , 11.82,  9.74, 10.18,  8.43,  8.71,
     6.84,  9.26, 11.61, 11.08,  8.94,  8.44, 10.41,  9.36, 10.85,
    10.41,  8.37,  8.99, 10.17,  7.78, 10.79, 10.61, 10.87,  7.43,
     8.44,  9.44,  8.26,  7.98, 11.27, 11.61,  9.84, 12.47,  7.8 ,
    10.54,  8.99,  7.33,  8.55,  8.06, 10.62, 10.41,  9.29,  9.98,
     9.46,  9.99,  8.62, 11.34, 11.21, 15.19, 20.85, 19.15, 19.01,
    15.24, 16.66, 17.62, 18.22, 17.2 , 15.76, 16.89, 15.22, 18.7 ,
    14.84, 14.88, 19.41, 18.54, 17.85, 18.31, 13.68, 18.46, 13.99,
    16.38, 16.88, 17.82, 15.17, 15.16, 18.15, 15.08, 15.91, 16.82,
    16.85, 18.04, 17.51, 18.44, 15.33, 16.07, 17.22, 15.9 , 18.03,
    17.26, 17.6 , 16.77, 17.45, 13.73, 14.95, 15.57, 19.19, 14.39,
    15.76])

state = np.random.RandomState(12345)

# Guarda los valores del cuantil del 99 % en la variable de valores
values = []
for i in range(1000):
    subsample = data.sample(frac=1, replace=True, random_state=state)
    quantile_99 = subsample.quantile(0.99)
    values.append(quantile_99)

# Cambia el tipo por comodidad
values = pd.Series(values)

# Calcula el intervalo de confianza del 90%
lower = values.quantile(0.05)
upper = values.quantile(0.95)

# Imprime los valores lower y upper
print(lower)
print(upper)
***********************************************************************************************************
Analiza las dos muestras y comprueba la hipótesis que dice que el importe promedio de compra ha aumentado. 

import pandas as pd
import numpy as np

# datos del grupo de control A
samples_A = pd.Series([
    98.24, 97.77, 95.56, 99.49, 101.4, 105.35, 95.83, 93.02,
    101.37, 95.66, 98.34, 100.75, 104.93, 97., 95.46, 100.03,
    102.34, 98.23, 97.05, 97.76, 98.63, 98.82, 99.51, 99.31,
    98.58, 96.84, 93.71, 101.38, 100.6, 103.68, 104.78, 101.51,
    100.89, 102.27, 99.87, 94.83, 95.95, 105.2, 97., 95.54,
    98.38, 99.81, 103.34, 101.14, 102.19, 94.77, 94.74, 99.56,
    102., 100.95, 102.19, 103.75, 103.65, 95.07, 103.53, 100.42,
    98.09, 94.86, 101.47, 103.07, 100.15, 100.32, 100.89, 101.23,
    95.95, 103.69, 100.09, 96.28, 96.11, 97.63, 99.45, 100.81,
    102.18, 94.92, 98.89, 101.48, 101.29, 94.43, 101.55, 95.85,
    100.16, 97.49, 105.17, 104.83, 101.9, 100.56, 104.91, 94.17,
    103.48, 100.55, 102.66, 100.62, 96.93, 102.67, 101.27, 98.56,
    102.41, 100.69, 99.67, 100.99])

# datos del grupo experimental B
samples_B = pd.Series([
    101.67, 102.27, 97.01, 103.46, 100.76, 101.19, 99.11, 97.59,
    101.01, 101.45, 94.8, 101.55, 96.38, 99.03, 102.83, 97.32,
    98.25, 97.17, 101.1, 102.57, 104.59, 105.63, 98.93, 103.87,
    98.48, 101.14, 102.24, 98.55, 105.61, 100.06, 99., 102.53,
    101.56, 102.68, 103.26, 96.62, 99.48, 107.6, 99.87, 103.58,
    105.05, 105.69, 94.52, 99.51, 99.81, 99.44, 97.35, 102.97,
    99.77, 99.59, 102.12, 104.29, 98.31, 98.83, 96.83, 99.2,
    97.88, 102.34, 102.04, 99.88, 99.69, 103.43, 100.71, 92.71,
    99.99, 99.39, 99.19, 99.29, 100.34, 101.08, 100.29, 93.83,
    103.63, 98.88, 105.36, 101.82, 100.86, 100.75, 99.4, 95.37,
    107.96, 97.69, 102.17, 99.41, 98.97, 97.96, 98.31, 97.09,
    103.92, 100.98, 102.76, 98.24, 97., 98.99, 103.54, 99.72,
    101.62, 100.62, 102.79, 104.19])

# diferencia real entre las medias de los grupos
AB_difference = samples_B.mean() - samples_A.mean()
print("Diferencia entre los importes promedios de compra:", AB_difference)

alpha = 0.05

# Concatena las muestras
state = np.random.RandomState(12345)

bootstrap_samples = 1000
count = 0
for i in range(bootstrap_samples):
    # calcula cuántas veces excederá la diferencia entre las medias
    # el valor actual, siempre que la hipótesis nula sea cierta
    united_samples = pd.concat([samples_A, samples_B])
    subsample = united_samples.sample(frac=1, replace=True, random_state=state)
    
    subsample_A = subsample[:len(samples_A)]
    subsample_B = subsample[len(samples_A):]
    bootstrap_difference = subsample_B.mean() - subsample_A.mean()
    
    if bootstrap_difference >= AB_difference:
        count += 1

pvalue = 1. * count / bootstrap_samples
print('p-value =', pvalue)

if pvalue < alpha:
    print("La hipótesis nula se rechaza, a saber, es probable que el importe promedio de las compras aumente")
else:
    print("La hipótesis nula no se rechaza, a saber, es poco probable que el importe medio de las compras aumente")
***********************************************************************************************************
  Escribe la función revenue() que calcula y devuelve el valor de los ingresos.
  
  import pandas as pd

def revenue(target, probabilities, count):
    # Ordena las probabilidades de mayor a menor
    probs_sorted = probabilities.sort_values(ascending=False)
    
    # Selecciona las respuestas basándose en las probabilidades ordenadas
    selected = target[probs_sorted.index][:count]
    
    # Calcula los ingresos multiplicando el número de estudiantes que asisten por 10
    return 10 * selected.sum()

# Ejemplo de uso
target = pd.Series([1, 1, 0, 0, 1, 0])
probab = pd.Series([0.2, 0.9, 0.8, 0.3, 0.5, 0.1])

res = revenue(target, probab, 3)

print(res)
***********************************************************************************************************
Para encontrar el cuantil de ingresos del 1 %, realiza el proceso de bootstrapping con 1000 repeticiones.

import pandas as pd
import numpy as np

# Abre los archivos
# toma el índice “0” para convertir los datos a pd-Series
target = pd.read_csv('/datasets/eng_target.csv')['0']
probabilities = pd.read_csv('/datasets/eng_probabilites.csv')['0']

def revenue(target, probabilities, count):
    probs_sorted = probabilities.sort_values(ascending=False)
    selected = target[probs_sorted.index][:count]
    return 10 * selected.sum()

state = np.random.RandomState(12345)

values = []
for i in range(1000):
    target_subsample = target.sample(n=25, replace=True, random_state=state)
    probs_subsample = probabilities[target_subsample.index]
    
    values.append(revenue(target_subsample, probs_subsample, 10))

values = pd.Series(values)
lower = values.quantile(0.01)

# Calcula el ingreso promedio
mean = values.mean()

# Imprime los resultados
print("Ingresos promedio:", mean)
print("Cuantil del 1 %:", lower)
***********************************************************************************************************
Realiza el voto mayoritario para el conjunto de datos. Almacena el objetivo en la variable objetivo. 

import pandas as pd

# Lee el conjunto de datos
data = pd.read_csv('/datasets/heart_labeled.csv')

# Realiza el voto mayoritario
target = []
for i in range(data.shape[0]):
    labels = data.loc[i, ['label_1', 'label_2', 'label_3']]
    true_label = int(labels.mean() > 0.5)
    target.append(true_label)

# Agrega la columna 'target' al conjunto de datos
data['target'] = target

# Muestra las cinco primeras filas del conjunto de datos resultante
print(data.head())
***********************************************************************************************************
Completa el código del bucle para obtener una validación cruzada en tres bloques del mismo tamaño.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('/datasets/heart.csv')
features = data.drop(['target'], axis=1)
target = data['target']

scores = []

# Establece el tamaño del bloque si solo hay tres de ellos
sample_size = int(len(data) / 3)

for i in range(0, len(data), sample_size):
    # Crea las matrices valid_indexes y train_indexes
    valid_indexes = list(range(i, i + sample_size))
    train_indexes = list(range(0, i)) + list(range(i + sample_size, len(data)))

    # Divide las características y el objetivo en muestras
    features_train, target_train = features.iloc[train_indexes], target.iloc[train_indexes]
    features_valid, target_valid = features.iloc[valid_indexes], target.iloc[valid_indexes]

    # Entrena el modelo
    model = DecisionTreeClassifier(random_state=0)
    model = model.fit(features_train, target_train)

    # Evalúa la calidad del modelo
    score = model.score(features_valid, target_valid)

    scores.append(score)

# Calcula la calidad media del modelo
final_score = sum(scores) / len(scores)
print('Valor de calidad promedio del modelo:', final_score)
***********************************************************************************************************
Calcula el puntaje de evaluación promedio usando el método de validación cruzada y guárdalo en 
la variable final_score.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

data = pd.read_csv('/datasets/heart.csv')
features = data.drop(['target'], axis=1)
target = data['target']

model = DecisionTreeClassifier(random_state=0)

# Calcula las puntuaciones llamando a la función cross_val_score con cinco bloques
scores = cross_val_score(model, features, target, cv=5)

# Calcula la puntuación promedio del modelo
final_score = scores.mean()

# Muestra en pantalla el valor final_score
print('Puntuación media de la evaluación del modelo:', final_score)
***********************************************************************************************************

# Dividir datos para la Región 0
train_data_0, val_data_0 = train_test_split(data_0, test_size=0.25, random_state=42)

# Seleccionar características y objetivo para entrenamiento
X_train_0 = train_data_0[['f0', 'f1', 'f2']]
y_train_0 = train_data_0['product']

# Seleccionar características y objetivo para validación
X_val_0 = val_data_0[['f0', 'f1', 'f2']]
y_val_0 = val_data_0['product']



# Inicializar modelo
model_0 = LinearRegression()

# Entrenar modelo para la Región 0
model_0.fit(X_train_0, y_train_0)

# Hacer predicciones para el conjunto de validación de la Región 0
predictions_0 = model_0.predict(X_val_0)

# Crear DataFrame con predicciones y respuestas correctas para la Región 0
results_0 = pd.DataFrame({'Predictions': predictions_0, 'Actual': y_val_0})




# Calcular el volumen medio de reservas predicho y RMSE del modelo para la Región 0
mean_predicted_volume_0 = results_0['Predictions'].mean()
rmse_0 = np.sqrt(mean_squared_error(y_val_0, predictions_0))

# Mostrar resultados para la Región 0
print(f"Volumen medio de reservas predicho para la Región 0: {mean_predicted_volume_0}")
print(f"RMSE del modelo para la Región 0: {rmse_0}")