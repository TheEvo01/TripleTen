Encuentra el valor esperado y la varianza usando la fórmula de la lección**************************************************************

import numpy as np

x_probs = {
    '-4': 0.05,
    '-2': 0.25,
    '0': 0.1,
    '1': 0.1,
    '5': 0.1,
    '7': 0.05,
    '15': 0.35,
}

# Calculate the expected value (E[X])
expectation = sum(int(x_i) * x_probs[x_i] for x_i in x_probs)

# Calculate (E[X])^2
square_of_expectation = expectation ** 2

# Calculate E[X^2]
expectation_of_squares = sum(int(x_i) ** 2 * x_probs[x_i] for x_i in x_probs)

# Calculate the variance (Var[X])
variance = expectation_of_squares - square_of_expectation

print('Expected value:', expectation)
print('Variance:', variance)

que significa esto? la variable aleatoria que estas viendo tiende hacia el valor esperado. De promedio, 
los habitantes de N disfrutan de temperaturas calidas de 5°C (41°F) en abril.

Y el hecho de que la varianza sea 55 no es particularmente significativo. Ahora es el momento de tomar las raices y multiplicarlas por 3.

La raiz cuadrada de 55 es 7 y algo. Luego, de acuerdo con la regla de las tres sigmas, las temperaturas de abril en N (99% del tiempo) 
caeran dentro del intervalo (5-7x3; 5+7*3). Entonces, el clima en N no es malo en abril. Las temperaturas oscilan entre -16°C y +26°C
(-3°F y +79°F).
*****************************************************************************************************************************************

import numpy as np

# La probabilidad de que su signo pertenezca a cualquiera de los 4 elementos es 1/4.
# Necesitamos sumar las probabilidades de 2 signos elementales (fuego y tierra) para obtener la probabilidad
# de una pitón que pesa 3 kg. Para los demás elementos, la probabilidad se mantiene en 1/4.
# creación de diccionario y código de cálculo
# Necesitamos sumar las probabilidades de 2 signos elementales (fuego y tierra) para obtener la probabilidad
# de una pitón que pesa 3 kg.
weight_probs = {
    '2': 0.25,
    '3': 0.5,
    '5': 0.25
}

# Calculate the expected value (E[X])
expectation = sum(int(weight) * weight_probs[weight] for weight in weight_probs)

# Calculate (E[X])^2
square_of_expectation = expectation ** 2

# Calculate E[X^2]
expectation_of_squares = sum(int(weight) ** 2 * weight_probs[weight] for weight in weight_probs)

# Calculate the variance (Var[X])
variance = expectation_of_squares - square_of_expectation

print('Expected value:', expectation)
print('Variance:', variance)
*****************************************************************************************************************************************

¿Cuál es la probabilidad de que de cinco usuarios diferentes, los tres primeros hagan clic en el cartel y los otros dos no?

a = 0.88 #  probabilidad de que un usuario haga clic en el cartel
b = 0.12 # probabilidad de que un usuario haga clic en otro lugar
prob = a * a * a * b * b
print (prob)

0.0098131968
******************************************************************************************************************************************

para calcular los exitos o perdidas (puede ser las buenas y malas, o las ojos azules y verdes, solo hay que cambiar los valores, pero los
calculos se hacen en este mismo orden, por ejemplo ojos azules 75% y verdes 25%, solo cambiamos los valores:

   Piton Buena y Mala                          Ojos Azules y Verdes
0 éxitos: 0.9 x 0.9 = 0.81              -   0.75 * 0.75 = 0.5625 * 100 = 56.25%
1 éxito: (0.9 x 0.1) x 2 = 0.18         -   0.75 * 0.25 * 2 = 0.375 * 100 = 37.5%
2 éxitos: 0.1 x 0.1 = 0.01              -   0.25 * 0.25 = 0.0625 * 100 = 6.25%

*******************************************************************************************************************************************
Algunos días, las pitones en el vivero de pitones son alimentadas con manzanas y otros días con peras.

from math import factorial

# Número de formas de obtener k días de n días:
# C(n, k) =  **n! / ( k! * (n-k)! )

# Number of days in the week
n_days = 7

# Number of days with pears
k_pears = 3

# Calculate the number of different diets
n_diets = factorial(n_days) // (factorial(k_pears) * factorial(n_days - k_pears))

# Convert the result to a floating-point number
n_diets = float(n_diets)

print(n_diets)
*******************************************************************************************************************************************

Probabilidad para los examenes donde la probabilidad de suspender cada examen es un 15%

from matplotlib import pyplot as plt
from math import factorial

n_exams = 6 # agrega tu código aquí: ¿Cuántos exámenes necesita aprobar?
failure_rate = 0.15 # agrega tu código aquí: ¿cuál es la probabilidad de que suspenda un examen?

distr = [] # agrega tu código aquí: crea una variable para el valor de distribución

for k in range(0,n_exams+1):
    choose = factorial(n_exams) / (factorial(k) * factorial(n_exams - k))
    prob_pass = choose * failure_rate ** k * (1 - failure_rate) ** (n_exams - k)
    distr.append(prob_pass) 
    
# crea un histograma de la distribución de probabilidad
plt.bar(range(0, n_exams + 1), distr)
*******************************************************************************************************************************************

tu empresa esta organizando un evento importante y esta buscando al menos 6 socios de medios para publicidad

from matplotlib import pyplot as plt
from math import factorial

p = 1/5 # agrega tu código aquí: ¿cuál es la probabilidad de redactar un contrato? 
n = 30  # agrega tu código aquí: ¿con cuántas empresas negociaremos?

distr = [] # agrega tu código aquí: crea una variable para el valor de distribución

for k in range(0 , n + 1):
      # Calculate the probability of getting k "yes" responses
    prob_k_yes = (factorial(n) / (factorial(k) * factorial(n - k))) * (p ** k) * ((1 - p) ** (n - k))
    distr.append(prob_k_yes)
    
# agrega tu código aquí: crea un histograma de la distribución de probabilidad
plt.bar(range(0, n + 1), distr)
plt.xlabel("Number of Media Representatives Saying Yes")
plt.ylabel("Probability")
plt.title("Probability Distribution of Media Representatives Saying Yes")
plt.show()
*******************************************************************************************************************************************

Probabilidad de visitas a un sitio web

from scipy import stats as st

mu = 100500 # coloca tu código aquí: ¿cuál es la media de la distribución?
sigma = 3500 # coloca tu código aquí: ¿cuál es la desviación estándar de la distribución?

bonus_threshold = 111000 # coloca tu código aquí: ¿dónde se encuentra el umbral para el bono?
penalty_threshold = 92000 # coloca tu código aquí: ¿dónde se encuentra el umbral para la penalización?

p_bonus = p_bonus = 1 - st.norm.cdf(bonus_threshold, loc=mu, scale=sigma) # coloca tu código aquí: calcula la probabilidad de lograr el bono
p_penalty = st.norm.cdf(penalty_threshold, loc=mu, scale=sigma) # coloca tu código aquí: calcula la probabilidad de obtener la penalización

print('Probabilidad de bonificación:', p_bonus)
print('Probabilidad de penalización:', p_penalty)
*******************************************************************************************************************************************

calcular la cantidad de articulos que necesita comprar la tienda virtual para cumplir con los clientes

from scipy import stats as st

mu = 420 # coloca tu código aquí: ¿cuál es la media de la distribución?
sigma = 65 # coloca tu código aquí: ¿cuál es la desviación estándar de la distribución?
prob = 0.9 # coloca tu código aquí: ¿cuál es la probabilidad requerida de vender todos los productos?

n_shipment = st.norm(mu, sigma).ppf(1 - prob) # coloca tu código aquí: ¿cuántos artículos se deben pedir?

print('Necesitan pedir artículos:', int(n_shipment))
*******************************************************************************************************************************************

calcular cuantos clientes estaran satisfechos con el precio de la entrega de la empresa

from scipy import stats as st

mu = 24 # coloca tu código aquí: ¿cuál es la media de la distribución?
sigma = 3.20 # coloca tu código aquí: ¿cuál es la desviación estándar de la distribución?
threshold = 0.75 # coloca tu código aquí: ¿qué porcentaje de pedidos debería costar más del doble del costo de envío?

max_delivery_price = st.norm.ppf(1 - threshold, loc=mu, scale=sigma) # coloca tu código aquí: el costo máximo de envío

print('Coste máximo para la entrega del mensajero:', max_delivery_price)
*******************************************************************************************************************************************

expectativas de visibilidad mediante correo electronico

from scipy import stats as st
import math as mt

binom_p = 0.4  # Probability of a user opening the newsletter
threshold = 9000  # Expected number of users opening the newsletter
binom_n = 23000  # Total number of people the newsletter is sent to

mu = binom_n * binom_p
sigma = mt.sqrt(binom_n * binom_p * (1 - binom_p))

# Calculate the probability of meeting the customer's expectations using the cumulative distribution function (CDF)
p_threshold = 1 - st.norm(mu, sigma).cdf(threshold)

print(p_threshold)
******************************************************************************************************************************************

saber la cantidad estimada disponible de patinetes en las estaciones de distribucion

from scipy import stats as st
import pandas as pd

scooters = pd.Series([15, 31, 10, 21, 21, 32, 30, 25, 21,
28, 25, 32, 38, 18, 33, 24, 26, 40, 24, 37, 20, 36, 28, 38,
24, 35, 33, 21, 29, 26, 13, 25, 34, 38, 23, 37, 31, 28, 32,
24, 25, 13, 38, 34, 48, 19, 20, 22, 38, 28, 31, 18, 21, 24,
31, 21, 28, 29, 33, 40, 26, 33, 33,  6, 27, 24, 17, 28,  7,
33, 25, 25, 29, 19, 30, 29, 22, 15, 28, 36, 25, 36, 25, 29,
33, 19, 32, 32, 28, 26, 18, 48, 15, 27, 27, 27,  0, 28, 39,
27, 25, 39, 28, 22, 33, 30, 35, 19, 20, 18, 31, 44, 20, 18,
17, 28, 17, 44, 40, 33,])

optimal_value = 30 # escribe tu código aquí

alpha = 0.05 # escribe tu código aquí
results = st.ttest_1samp(scooters, popmean=optimal_value) # escribe tu código aquí

print('p-value: ', results.pvalue)

if results.pvalue < alpha:
    print('Rechazamos la hipótesis nula')
else:
    print("No rechazamos la hipótesis nula")
*******************************************************************************************************************************************

calcular la ganancia hipotetica diaria en un periodo de tiempo 

from scipy import stats as st
import numpy as np
import pandas as pd

revenue = pd.Series([727, 678, 685, 669, 661, 705, 701, 717, 
                     655,643, 660, 709, 701, 681, 716, 655, 
                     716, 695, 684, 687, 669,647, 721, 681, 
                     674, 641, 704, 717, 656, 725, 684, 665])

interested_value = 800 # ¿Cuánto prometió Robby Tobbinson?

alpha = 0.05 # indica el nivel de significación estadística

results = st.ttest_1samp(revenue, popmean=interested_value) # utiliza el método st.ttest_1samp

print('p-value:', results.pvalue / 2) # imprime el valor p para una prueba unilateral

if results.pvalue / 2 < alpha and np.mean(revenue) < interested_value: 
    # compara el valor obtenido y el nivel crítico de significación estadística
    # y verifica si la media muestral está en el lado correcto del interested_value
    print("Rechazamos la hipótesis nula: los ingresos fueron considerablemente menores que 800 dólares")
else:
    print("No podemos rechazar la hipótesis nula: los ingresos no fueron considerablemente menores")
********************************************************************************************************************************************

prueba la hipotesis de que ambos grupos de usuarios pasan la misma cantidad de tiempo en el sitio web

from scipy import stats as st
import numpy as np

# Tiempo pasado en el sitio web por usuarios con un nombre de usuario y contraseña
time_on_site_logpass = [368, 113, 328, 447, 1, 156, 335, 233,
                       308, 181, 271, 239, 411, 293, 303,
                       206, 196, 203, 311, 205, 297, 529,
                       373, 217, 416, 206, 1, 128, 16, 214]

# Tiempo pasado en el sitio web por los usuarios que inician sesión a través de las redes sociales
time_on_site_social  = [451, 182, 469, 546, 396, 630, 206,
                        130, 45, 569, 434, 321, 374, 149,
                        721, 350, 347, 446, 406, 365, 203,
                        405, 631, 545, 584, 248, 171, 309,
                        338, 505]

# Nivel de significancia (alfa)
alpha = 0.05

# Prueba de hipótesis para muestras independientes usando t-test
results = st.ttest_ind(time_on_site_logpass, time_on_site_social)

print('p-value:', results.pvalue)

if results.pvalue < alpha:
    print("Rechazamos la hipótesis nula: Los grupos pasan diferentes cantidades de tiempo en el sitio web")
else:
    print("No rechazamos la hipótesis nula: Los grupos pasan la misma cantidad de tiempo en el sitio web")
********************************************************************************************************************************************

probar la hipotesis de que los usuarios cambian el tiempo de uso dependiendo la estacion del año

from scipy import stats as st
import numpy as np

pages_per_session_autumn = [7.1, 7.3, 9.8, 7.3, 6.4, 10.5, 8.7, 
                            17.5, 3.3, 15.5, 16.2, 0.4, 8.3, 
                            8.1, 3.0, 6.1, 4.4, 18.8, 14.7, 16.4, 
                            13.6, 4.4, 7.4, 12.4, 3.9, 13.6, 
                            8.8, 8.1, 13.6, 12.2]
pages_per_session_summer = [12.1, 24.3, 6.4, 19.9, 19.7, 12.5, 17.6, 
                            5.0, 22.4, 13.5, 10.8, 23.4, 9.4, 3.7, 
                            2.5, 19.8, 4.8, 29.0, 1.7, 28.6, 16.7, 
                            14.2, 10.6, 18.2, 14.7, 23.8, 15.9, 16.2, 
                            12.1, 14.5]

alpha = 0.05 # tu código: establece un nivel crítico de significación estadística

results = st.ttest_ind(pages_per_session_autumn, pages_per_session_summer, equal_var=False) # tu código: prueba la hipótesis de que las medias de las dos poblaciones independientes son iguales
print('p-value:', results.pvalue) # tu código: imprime el valor p obtenido

if results.pvalue < alpha: # su código: compara los valores p obtenidos con el nivel de significación estadística
    print("Rechazamos la hipótesis nula")
else:
    print("No rechazamos la hipótesis nula")
*********************************************************************************************************************************************

Prueba la hipótesis de que el tiempo que pasan allí cambió (aumentó o disminuyó) después del rediseño

from scipy import stats as st
import numpy as np

time_before = [1732, 1301, 1540, 2247, 1632, 1550, 754, 1946, 1889, 
          2748, 1349, 1648, 1665, 2416, 1470, 1681, 1868, 1629, 
          1271, 1633, 2131, 942, 1599, 1127, 2200, 661, 1207, 
          1737, 2410, 1486]

time_after = [955, 2577, 360, 139, 1618, 990, 644, 1796, 1487, 949, 472, 
         1906, 1758, 1258, 2554, 612, 309, 1864, 1294, 1487, 1164, 1559, 
         491, 2286, 1270, 2069, 1553, 1629, 1704, 1623]

alpha = 0.05 # tu código: establece un nivel crítico de significación estadística

results = st.ttest_rel(time_before, time_after) # tu código: realiza la prueba y calcula el valor p

print('p-value:', results.pvalue) # tu código: imprime el valor p obtenido

if results.pvalue < alpha: # tu código: compara el valor p con el nivel de la significación estadística
    print("Rechazamos la hipótesis nula")
else:
    print("No rechazamos la hipótesis nula")
*********************************************************************************************************************************************

Prueba la hipótesis de que los jugadores empezaron a usar más balas después de que se introdujo la nueva característica.

from scipy import stats as st
import numpy as np
import pandas as pd

bullets_before = [821, 1164, 598, 854, 455, 1220, 161, 1400, 479, 215, 
          564, 159, 920, 173, 276, 444, 273, 711, 291, 880, 
          892, 712, 16, 476, 498, 9, 1251, 938, 389, 513]

bullets_after = [904, 220, 676, 459, 299, 659, 1698, 1120, 514, 1086, 1499, 
         1262, 829, 476, 1149, 996, 1247, 1117, 1324, 532, 1458, 898, 
         1837, 455, 1667, 898, 474, 558, 639, 1012]

print('media antes:', pd.Series(bullets_before).mean())
print('media después:', pd.Series(bullets_after).mean())

alpha = 0.05 # tu código: establece un nivel crítico de significación estadística

results = st.ttest_rel(
    bullets_before, 
    bullets_after)

print('p-value:', results.pvalue / 2) # tu código: imprime el valor p obtenido

if results.pvalue < alpha: # tu código: compara el valor p con la significación estadística
    print("Rechazamos la hipótesis nula")
else:
    print("No rechazamos la hipótesis nula")
*********************************************************************************************************************************************

