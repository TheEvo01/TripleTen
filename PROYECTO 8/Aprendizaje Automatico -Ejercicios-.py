Carga de datos. Muestra en la pantalla los primeros diez elementos.

import pandas as pd

# Cargar los datos desde el archivo CSV
data = pd.read_csv('/datasets/travel_insurance_us.csv')

# Mostrar los primeros diez elementos del DataFrame
print(data.head(10))
*************************************************************************************
Divide los datos en dos conjuntos. Muestra en pantalla los tamaños de las tablas.

import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('/datasets/travel_insurance_us.csv')

# Dividir los datos en conjuntos de entrenamiento y validación
train, valid = train_test_split(data, test_size=0.25, random_state=12345)

# Definir las características y el objetivo para el conjunto de entrenamiento
features_train = train.drop('Claim', axis=1)
target_train = train['Claim']

# Definir las características y el objetivo para el conjunto de validación
features_valid = valid.drop('Claim', axis=1)
target_valid = valid['Claim']

# Mostrar los tamaños de las tablas de características para entrenamiento y validación
print(features_train.shape)
print(features_valid.shape)
*************************************************************************************
Verifica los tipos de características almacenadas en la tabla. Muéstralos.

import pandas as pd

data = pd.read_csv('/datasets/travel_insurance_us.csv')

# Verificar los tipos de datos en la tabla
print(data.dtypes)

# Mostrar los primeros cinco valores de la columna "Gender"
print(data['Gender'].head())
*************************************************************************************
Podemos mirar los valores de la columna Gender usando OHE. Llama a pd.get_dummies()

import pandas as pd

data = pd.read_csv('/datasets/travel_insurance_us.csv')

print(pd.get_dummies(data ['Gender']).head())
**************************************************************************************
Programa la variableGender con OHE. Llama a pd.get_dummies() con el argumento 
drop_first para evitar la trampa dummy. 

import pandas as pd

data = pd.read_csv('/datasets/travel_insurance_us.csv')

# Codificar la variable 'Gender' utilizando One-Hot Encoding (OHE) y evitar la trampa de variables ficticias
print(pd.get_dummies(data ['Gender'], drop_first=True).head())
***************************************************************************************
Programa todo el DataFrame con One-Hot. Llama a pd.get_dummies() con el argumento 
drop_first. Almacena la tabla en la variable data_ohe.

import pandas as pd

data = pd.read_csv('/datasets/travel_insurance_us.csv')

data_ohe = pd.get_dummies(data, drop_first=True)

print(data_ohe.head(3))
***************************************************************************************
Divide los datos de origen en dos conjuntos utilizando la proporción de 75:25 (%):

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('/datasets/travel_insurance_us.csv')

data_ohe = pd.get_dummies(data, drop_first=True)
target = data_ohe['Claim']
features = data_ohe.drop('Claim', axis=1)

# Dividir los datos en conjuntos de entrenamiento y validación
features_train, features_valid, target_train, target_valid = train_test_split(features, target, test_size=0.25, random_state=12345)

# Entrenar una regresión logística
model = LogisticRegression(solver='liblinear', random_state=12345)
model.fit(features_train, target_train)

print('¡Entrenado!')
*****************************************************************************************
Transforma las funciones utilizando codificación de etiquetas. Importa 
OrdinalEncoder desde el módulo sklearn.preprocessing.

import pandas as pd
from sklearn.preprocessing import OrdinalEncoder

data = pd.read_csv('/datasets/travel_insurance_us.csv')

# Initialize the OrdinalEncoder
encoder = OrdinalEncoder()

# Fit and transform the data using OrdinalEncoder
data_ordinal = pd.DataFrame(encoder.fit_transform(data), columns=data.columns)

# Display the first five rows of the transformed DataFrame
print(data_ordinal.head())
*******************************************************************************************
Utiliza los datos transformados para entrenar un árbol de decisión.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import OrdinalEncoder

data = pd.read_csv('/datasets/travel_insurance_us.csv')

encoder = OrdinalEncoder()
data_ordinal = pd.DataFrame(encoder.fit_transform(data), columns=data.columns)

target = data_ordinal['Claim']
features = data_ordinal.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

# Crear y entrenar un modelo DecisionTreeClassifier
model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)

# Mostrar un mensaje para confirmar que el entrenamiento ha finalizado
print('¡Entrenado!')
*********************************************************************************************
Estandariza las características numéricas.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('/datasets/travel_insurance_us.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

numeric = ['Duration', 'Net Sales', 'Commission (in value)', 'Age']

# Instanciar y ajustar el StandardScaler a los datos de entrenamiento
scaler = StandardScaler()
scaler.fit(features_train[numeric])

# Transformar el conjunto de entrenamiento
features_train[numeric] = scaler.transform(features_train[numeric])

# Transformar el conjunto de validación
features_valid[numeric] = scaler.transform(features_valid[numeric])

# Mostrar las primeras cinco filas del conjunto de entrenamiento
print(features_train.head())
************************************************************************************************
Entrena el modelo de árbol de decisión calculando el valor de exactitud en el conjunto de validación. 

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)

predicted_valid = model.predict(features_valid)
accuracy_valid = accuracy_score(target_valid, predicted_valid)

print(accuracy_valid)
************************************************************************************************
Para contar clases en la característica objetivo, utiliza el método value_counts().

import pandas as pd

data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

# Calcular las frecuencias relativas de las clases en la característica objetivo
class_frequency = data['Claim'].value_counts(normalize=True)
print(class_frequency)

# Mostrar las frecuencias relativas en un diagrama de barras
class_frequency.plot(kind='bar')
*************************************************************************************************
Analiza las frecuencias de clase de las predicciones del árbol de decisión (la variable 
predicted_valid).

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)

predicted_valid = pd.Series(model.predict(features_valid))

# Analizar las frecuencias de clase de las predicciones del árbol de decisión
class_frequency = predicted_valid.value_counts(normalize=True)
print(class_frequency)

# Mostrar las frecuencias de clase en un diagrama de barras
class_frequency.plot(kind='bar')
***************************************************************************************************
Crea un modelo constante: predice la clase "0" para cualquier observación. Guarda sus 
predicciones en la variable target_pred_constant.

import pandas as pd
from sklearn.metrics import accuracy_score

data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)

# Crear un modelo constante que predice la clase "0" para todas las observaciones
target_pred_constant = pd.Series([0] * len(target))

# Calcular y mostrar la precisión del modelo constante
print(accuracy_score(target, target_pred_constant))
****************************************************************************************************
Aquí hay un ejemplo de predicciones frente a respuestas correctas. Cuenta el número de respuestas 
Verdadero Positivo (VP).

import pandas as pd

target = pd.Series([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
predictions = pd.Series([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1])

# Contar el número de Verdaderos Positivos (VP)
vp_count = ((target == 1) & (predictions == 1)).sum()

print(vp_count)
****************************************************************************************************
Cuenta el número de respuestas Verdadero Negativo (VN) tal como lo hiciste en el ejercicio anterior.

import pandas as pd

target = pd.Series([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
predictions = pd.Series([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1])

# Contar el número de Verdaderos Negativos (VN)
vn_count = ((target == 0) & (predictions == 0)).sum()

print(vn_count)
*****************************************************************************************************
Encuentra el número de respuestas Falso Positivo (FP) de la misma manera que encontraste las 
respuestas VN en la tarea anterior.

import pandas as pd

target = pd.Series([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
predictions = pd.Series([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1])

# Contar el número de Falsos Positivos (FP)
fp_count = ((target == 0) & (predictions == 1)).sum()

print(fp_count)
******************************************************************************************************
Cuenta el número de respuestas Falso Negativo (FN). Muestra los resultados en la pantalla.

import pandas as pd

target = pd.Series([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
predictions = pd.Series([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1])

# Contar el número de Falsos Negativos (FN)
fn_count = ((target == 1) & (predictions == 0)).sum()

print(fn_count)
******************************************************************************************************
Calcula la matriz de confusión utilizando la función confusion_matrix().

import pandas as pd
from sklearn.metrics import confusion_matrix

target = pd.Series([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
predictions = pd.Series([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1])

# Calcular la matriz de confusión
conf_matrix = confusion_matrix(target, predictions)

# Mostrar la matriz de confusión
print(conf_matrix)
******************************************************************************************************
Calcula una matriz de confusión para el árbol de decisión y llama a la función confusion_matrix().

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)

# Cálculo de la matriz de confusión
conf_matrix = confusion_matrix(target_valid, predicted_valid)

# Mostrar la matriz de confusión en pantalla
print(conf_matrix)
*******************************************************************************************************
En el módulo sklearn.metrics, encuentra la función que se encarga de calcular recall.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)

# Cálculo del recall
recall = recall_score(target_valid, predicted_valid)

# Mostrar el recall en pantalla
print(recall)
*******************************************************************************************************
En el módulo sklearn.metrics, encuentra la función que calcula la precisión.

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split

data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)

# Calcular la precisión del modelo en el conjunto de validación
precision = precision_score(target_valid, predicted_valid)

# Mostrar la precisión en pantalla
print(precision)
*******************************************************************************************************
Calcula lo siguiente:
precisión, usando la función precision_score()
recall, usando la función recall_score()
Valor F1, utilizando la fórmula de la lección.

import pandas as pd
from sklearn.metrics import precision_score, recall_score

target = pd.Series([1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0, 0, 0, 1])
predictions = pd.Series([1, 1, 0, 0, 1, 1, 1, 0, 0, 1, 0, 1, 0, 1])

precision = precision_score(target, predictions)
recall = recall_score(target, predictions)
f1 = 2 * precision * recall / (precision + recall)

print('Recall:', recall)
print('Precisión:', precision)
print('Puntuación F1', f1)
********************************************************************************************************
En el módulo sklearn.metrics, busca la función que calcula el valor F1

import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score

data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

model = DecisionTreeClassifier(random_state=12345)
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)

# Calcular el valor F1
f1 = f1_score(target_valid, predicted_valid)

# Mostrar el valor F1 en pantalla
print(f1)
********************************************************************************************************
Equilibra los pesos de clase pasando el valor de argumento adecuado para class_weight.

import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

model = LogisticRegression(random_state=12345, class_weight='balanced', solver='liblinear')
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)
print('F1:', f1_score(target_valid, predicted_valid))
********************************************************************************************************
Hemos dividido la muestra de entrenamiento en observaciones negativas y positivas.
Declara cuatro variables y almacena las tablas:

import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

# Almacenar las observaciones con respuesta "0"
features_zeros = features_train[target_train == 0]
target_zeros = target_train[target_train == 0]

# Almacenar las observaciones con respuesta "1"
features_ones = features_train[target_train == 1]
target_ones = target_train[target_train == 1]

print(features_zeros.shape)
print(features_ones.shape)
print(target_zeros.shape)
print(target_ones.shape)
**********************************************************************************************************
Duplica las observaciones de clase positivas y combínalas con las  negativas.
Utiliza la función pd.concat() para concatenar las tablas.

import pandas as pd
from sklearn.model_selection import train_test_split

data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

features_zeros = features_train[target_train == 0]
features_ones = features_train[target_train == 1]
target_zeros = target_train[target_train == 0]
target_ones = target_train[target_train == 1]

repeat = 10
features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

print(features_upsampled.shape)
print(target_upsampled.shape)
**********************************************************************************************************
Mezcla los datos. Usa la función shuffle() en las variables features_upsampled y targets_upsampled. 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

    features_upsampled, target_upsampled = shuffle(features_upsampled, target_upsampled, random_state=12345)
    
    return features_upsampled, target_upsampled 

features_upsampled, target_upsampled = upsample(features_train, target_train, 10)

print(features_upsampled.shape)
print(target_upsampled.shape)
***********************************************************************************************************
Entrena el modelo LogisticRegression con los nuevos datos. Encuentra el valor F1.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

def upsample(features, target, repeat):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_upsampled = pd.concat([features_zeros] + [features_ones] * repeat)
    target_upsampled = pd.concat([target_zeros] + [target_ones] * repeat)

    features_upsampled, target_upsampled = shuffle(
        features_upsampled, target_upsampled, random_state=12345
    )

    return features_upsampled, target_upsampled

features_upsampled, target_upsampled = upsample(
    features_train, target_train, 10
)

logistic_regression = LogisticRegression(solver='liblinear', random_state=12345)
logistic_regression.fit(features_upsampled, target_upsampled)

predicted_valid = logistic_regression.predict(features_valid)
f1 = f1_score(target_valid, predicted_valid)

print('F1:', f1_score(target_valid, predicted_valid))
***********************************************************************************************************
Para realizar el submuestreo, escribe una función downsample() y pásale tres argumentos:

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_downsampled = pd.concat(
        [features_zeros.sample(frac=fraction, random_state=12345)] + [features_ones]
    )
    target_downsampled = pd.concat(
        [target_zeros.sample(frac=fraction, random_state=12345)] + [target_ones]
    )
    
    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345
    )
    
    return features_downsampled, target_downsampled


features_downsampled, target_downsampled = downsample(
    features_train, target_train, 0.1
)

print(features_downsampled.shape)
print(target_downsampled.shape)
***********************************************************************************************************
Entrena el modelo LogisticRegression con los nuevos datos. Encuentra el valor F1.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

def downsample(features, target, fraction):
    features_zeros = features[target == 0]
    features_ones = features[target == 1]
    target_zeros = target[target == 0]
    target_ones = target[target == 1]

    features_downsampled = pd.concat(
        [features_zeros.sample(frac=fraction, random_state=12345)]
        + [features_ones]
    )
    target_downsampled = pd.concat(
        [target_zeros.sample(frac=fraction, random_state=12345)]
        + [target_ones]
    )

    features_downsampled, target_downsampled = shuffle(
        features_downsampled, target_downsampled, random_state=12345
    )

    return features_downsampled, target_downsampled

features_downsampled, target_downsampled = downsample(
    features_train, target_train, 0.1
)

# Entrenar el modelo Logistic Regression con los nuevos datos
model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_downsampled, target_downsampled)

# Predecir en el conjunto de validación
predicted_valid = model.predict(features_valid)

print('F1:', f1_score(target_valid, predicted_valid))
***********************************************************************************************************
Encuentra las probabilidades de clase para la muestra de validación.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)

# Obtener las probabilidades de clase para la muestra de validación
probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

print(probabilities_one_valid[:5])
***********************************************************************************************************
Pasa por los valores de umbral de 0 a 0.3 en intervalos de 0.02. Encuentra precisión y recall 
para cada valor del umbral.

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score

data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)
probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

for threshold in np.arange(0, 0.3, 0.02):
    predicted_valid = probabilities_one_valid > threshold
    precision = precision_score(target_valid, predicted_valid)
    recall = recall_score(target_valid, predicted_valid)

    print(
        'Threshold = {:.2f} | Precision = {:.3f}, Recall = {:.3f}'.format(
            threshold, precision, recall
        )
    )
***********************************************************************************************************
Como obtener una Curva de Precision y Recall (Curva PR)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_recall_curve
from sklearn.linear_model import LogisticRegression

data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)

probabilities_valid = model.predict_proba(features_valid)
precision, recall, thresholds = precision_recall_curve(
    target_valid, probabilities_valid[:, 1]
)

plt.figure(figsize=(6, 6))
plt.step(recall, precision, where='post')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.ylim([0.0, 1.05])
plt.xlim([0.0, 1.0])
plt.title('Precision-Recall Curve')
plt.show() 
***********************************************************************************************************
Haz una curva ROC para la regresión logística y trázala en la gráfica.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve

data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)

probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

fpr, tpr, thresholds = roc_curve(target_valid, probabilities_one_valid)

plt.figure()

# Curva ROC para la regresión logística
plt.plot(fpr, tpr)

# Curva ROC para modelo aleatorio (parece una línea recta)
plt.plot([0, 1], [0, 1], linestyle='--')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('Tasa de falsos positivos')
plt.ylabel('Tasa de verdaderos positivos')
plt.title('Curva ROC')
plt.show()
***********************************************************************************************************
Calcula el AUC-ROC para la regresión logística.

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score

data = pd.read_csv('/datasets/travel_insurance_us_preprocessed.csv')

target = data['Claim']
features = data.drop('Claim', axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

model = LogisticRegression(random_state=12345, solver='liblinear')
model.fit(features_train, target_train)

probabilities_valid = model.predict_proba(features_valid)
probabilities_one_valid = probabilities_valid[:, 1]

auc_roc = roc_auc_score(target_valid, probabilities_one_valid)
print(auc_roc)
***********************************************************************************************************
Carga los datos de flights.csv

import pandas as pd

data = pd.read_csv('/datasets/flights.csv')

# Muestra el tamaño de la tabla
print(data.shape)

# Muestra las primeras cinco filas de la tabla
print(data.head())
***********************************************************************************************************
Codifica la característica categórica utilizando OHE. Estandariza las características numéricas.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

data = pd.read_csv('/datasets/flights.csv')

# Codificar las características categóricas usando OHE
data_ohe = pd.get_dummies(data, drop_first=True)

target = data_ohe['Arrival Delay']
features = data_ohe.drop(['Arrival Delay'], axis=1)

# Separar características y objetivo en conjuntos de entrenamiento y validación
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

# Lista de características numéricas
numeric = ['Day', 'Day Of Week', 'Origin Airport Delay Rate',
           'Destination Airport Delay Rate', 'Scheduled Time', 'Distance',
           'Scheduled Departure Hour', 'Scheduled Departure Minute']

# Estandarizar las características numéricas
scaler = StandardScaler()
scaler.fit(features_train[numeric])

features_train[numeric] = scaler.transform(features_train[numeric])
features_valid[numeric] = scaler.transform(features_valid[numeric])  # Transforma el conjunto de validación

# Mostrar los tamaños de la tabla
print(features_train.shape)
print(features_valid.shape)
***********************************************************************************************************
Carga los datos de /datasets/flights_preprocessed.csv. Declara la variable predicted_valid.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('/datasets/flights_preprocessed.csv')

target = data['Arrival Delay']
features = data.drop(['Arrival Delay'], axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

model = LinearRegression()
model.fit(features_train, target_train)

predicted_valid = model.predict(features_valid)
mse = mean_squared_error(target_valid, predicted_valid)

print("MSE =", mse)
print("RMSE =", mse ** 0.5)
***********************************************************************************************************
Encuentra los valores de ECM y RECM para el modelo constante: esto predice el valor objetivo 
medio para cada observación.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

data = pd.read_csv('/datasets/flights_preprocessed.csv')

target = data['Arrival Delay']
features = data.drop(['Arrival Delay'], axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

model = LinearRegression()
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)
mse = mean_squared_error(target_valid, predicted_valid)

print('Linear Regression')
print('MSE =', mse)
print('RMSE =', mse ** 0.5)

# Calculating MSE and RMSE for the constant model (predicting the mean value)
predicted_valid_constant = pd.Series(target_train.mean(), index=target_valid.index)
mse_constant = mean_squared_error(target_valid, predicted_valid_constant)

print('Mean')
print('MSE =', mse_constant)
print('RMSE =', mse_constant ** 0.5)
***********************************************************************************************************
Calcula el valor R2 para la regresión lineal. Busca la función adecuada en la documentación de 
sklearn.metrics.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

data = pd.read_csv('/datasets/flights_preprocessed.csv')

target = data['Arrival Delay']
features = data.drop(['Arrival Delay'], axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

model = LinearRegression()
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)

r2 = r2_score(target_valid, predicted_valid)
print('R2 =', r2)
***********************************************************************************************************
Codifica la función mae() según la fórmula. Esta función las respuestas y predicciones correctas y 
devuelve el valor de error absoluto medio.

import pandas as pd

def mae(target, predictions):
    error = 0
    for i in range(target.shape[0]):
        error += abs(target.iloc[i] - predictions.iloc[i])
    return error / target.shape[0]

target = pd.Series([-0.5, 2.1, 1.5, 0.3])
predictions = pd.Series([-0.6, 1.7, 1.6, 0.2])

print(mae(target, predictions))
***********************************************************************************************************
Calcula EAM para la regresión lineal. Encuentra la función apropiada en la documentación de sklearn. 
Impórtala. Muéstrala en la pantalla.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('/datasets/flights_preprocessed.csv')

target = data['Arrival Delay']
features = data.drop(['Arrival Delay'], axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

model = LinearRegression()
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)

mae = mean_absolute_error(target_valid, predicted_valid)
print(mae)
***********************************************************************************************************
Calcula la métrica EAM para la mediana objetivo. El cálculo EAM para la regresión lineal se 
encuentra en el precódigo.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

data = pd.read_csv('/datasets/flights_preprocessed.csv')

target = data['Arrival Delay']
features = data.drop(['Arrival Delay'], axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

model = LinearRegression()
model.fit(features_train, target_train)
predicted_valid = model.predict(features_valid)

print('Linear Regression')
print(mean_absolute_error(target_valid, predicted_valid))
print()

predicted_valid = pd.Series(target_train.median(), index=target_valid.index)
print('Median')
print(mean_absolute_error(target_valid, predicted_valid))
**********************************************************************************************************
Construye un modelo con un valor EAM menor o igual a 26.2.
En la quinta lección de este capítulo ya aprendimos que RandomForestRegressor es una buena alternativa 
para la regresión lineal.

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

data = pd.read_csv("/datasets/flights_preprocessed.csv")

target = data['Arrival Delay']
features = data.drop(['Arrival Delay'], axis=1)
features_train, features_valid, target_train, target_valid = train_test_split(
    features, target, test_size=0.25, random_state=12345
)

# Construye un modelo RandomForestRegressor
model = RandomForestRegressor(n_estimators=50, max_depth=8, random_state=12345)
model.fit(features_train, target_train)
predictions_train = model.predict(features_train)
predictions_valid = model.predict(features_valid)

print("Configuración del modelo actual lograda:")
print(
    "Valor EAM en un conjunto de entrenamiento: ",
    mean_absolute_error(target_train, predictions_train),
)
print(
    "Valor EAM en un conjunto de validación: ",
    mean_absolute_error(target_valid, predictions_valid),
)
**********************************************************************************************************





