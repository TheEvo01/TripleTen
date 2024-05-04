-expresion regular-

import requests
import re

URL = 'https://tripleten-com.github.io/simple-shop_es/'
req_text = requests.get(URL).text
print(re.findall('[A-ü ]*Mantequilla[A-ü ]*', req_text))


crear un objeto beautifulsoup que guarde todo el contenido de una pagina HTML

import requests  # Importa la librería para enviar solicitudes al servidor
from bs4 import BeautifulSoup  # Importa la librería para analizar la página web

URL = 'https://tripleten-com.github.io/simple-shop_es/'
req = requests.get(URL)
soup = BeautifulSoup(req.text, 'lxml')
print(soup)
**************************************************************************************************************
OBTENER EL NOMBRE DE LOS PRODUCTOS Y AGREGARLOS A UNA LISTA

import requests # Importa la librería para enviar solicitudes al servidor
from bs4 import BeautifulSoup # Importa la librería para analizar la página web

URL = 'https://tripleten-com.github.io/simple-shop_es/'
req = requests.get(URL)  # solicitud GET
soup = BeautifulSoup(req.text, 'lxml')
name_products = []

for row in soup.find_all(
    'p', attrs={'class': 't754__title t-name t-name_md js-product-name'}
):
    name_products.append(row.text.strip('\n '))
print(name_products)
---------------------------------------------------------------------------------------------------------------
OBTENER LOS PRECIOS DE LOS PRODUCTOS Y AGREGARLOS A UNA LISTA

import requests # Importa la librería para enviar solicitudes al servidor
from bs4 import BeautifulSoup # Importa la librería para analizar la página web

URL = 'https://tripleten-com.github.io/simple-shop_es/'
req = requests.get(URL)  # solicitud GET
soup = BeautifulSoup(req.text, 'lxml')

name_products = []  # Lista donde se almacenan los nombres de los productos
for row in soup.find_all(
    'p', attrs={'class': 't754__title t-name t-name_md js-product-name'}
):
    name_products.append(row.text)
price = [] 
for row in soup.find_all(
    'p', attrs={'class': 't754__price-value js-product-price'}
):
    price.append(row.text)
print(price)
----------------------------------------------------------------------------------------------------------------
AGREGAR LOS DATOS A UN DATAFRAME VACIO

import pandas as pd
import requests  # Importa la librería para enviar solicitudes al servidor
from bs4 import BeautifulSoup  # Importa la librería para analizar la página web

URL = 'https://tripleten-com.github.io/simple-shop_es/'
req = requests.get(URL)  # solicitud GET
soup = BeautifulSoup(req.text, 'lxml')

name_products = []  # Lista donde se almacenan los nombres de los productos
for row in soup.find_all(
        'p', attrs={'class': 't754__title t-name t-name_md js-product-name'}
):
    name_products.append(row.text)
price = []  # Lista donde se almacenan los precios de los productos
for row in soup.find_all(
        'p', attrs={'class': 't754__price-value js-product-price'}
):
    price.append(row.text)

# Crear un DataFrame y llenarlo con los datos
products_data = pd.DataFrame({
    'name': name_products,
    'price': price
})

# Mostrar las primeras 5 filas del DataFrame
print(products_data.head())
-------------------------------------------------------------------------------------------------------------------
SOLICITUD DE API - OBTENER EL PRONOSTICO DE CINCO DIAS PARA LA ZONA DE MOSCU, OMITIENDO INFORMACION DETALLADA

import requests

BASE_URL = "https://weather.data-analyst.praktikum-services.ru/v1/forecast" 
# URL para el método get()

params = { # diccionario con parámetros de solicitud
    "city" : "Moscow",
    "limit": 5 
}

response = requests.get(BASE_URL, params=params)
print(response.text)
----------------------------------------------------------------------------------------------------------------------
NO INCLUYAS EL PRONOSTICO POR HORA, SIMPLEMENTE PASA A ESTA CLAVE DE DICCIONARIO EL VALOR "FALSE"

import requests

BASE_URL = "https://weather.data-analyst.praktikum-services.ru/v1/forecast" 
# URL para el método get() 

params = {
    "city": "Moscow",
    "limit": 5,
    "hours": False,
    "extra": True
}

response = requests.get(BASE_URL, params=params)

print(response.text)
-----------------------------------------------------------------------------------------------------------------------
En la lección anterior, analizaste datos de un sitio web meteorológico, pero fue difícil distinguir las claves 
individuales en el diccionario porque había mucho texto. Por ejemplo, aquí tienes la llave info, con datos de la 
ciudad del pronóstico. Procesa el contenido de texto de la respuesta (response.text) con el método json.loads() y 
guárdalo en la variable response_parsed. Recupera los datos con la clave 'info' y muestra los resultados.

import requests
import json

BASE_URL = "https://weather.data-analyst.praktikum-services.ru/v1/forecast"

params = {
    "city" : "Moscow"
}

response = requests.get(BASE_URL, params=params)

# Procesar el contenido de texto de la respuesta
response_parsed = json.loads(response.text)

# Mostrar los datos correspondientes a la clave 'info'
print(response_parsed['info'])
------------------------------------------------------------------------------------------------------------------------
RECUPERA LOS DATOS SOBRE EL CLIMA ACTUAL DE LA RESPUESTA

import requests
import json

BASE_URL = "https://weather.data-analyst.praktikum-services.ru/v1/forecast"

params = {
    "city" : "Moscow"
}

response = requests.get(BASE_URL, params=params)

response_parsed = json.loads(response.text)

print(response_parsed['fact'])
------------------------------------------------------------------------------------------------------------------------
OBTEN DATOS SOBRE EL CLIMA ACTUAL Y VUELVE A COLOCARLOS EN FORMATO JSON

import requests
import json

BASE_URL = "https://weather.data-analyst.praktikum-services.ru/v1/forecast"

params = {
    "city" : "Moscow"
}

response = requests.get(BASE_URL, params=params)

# Procesar el contenido de texto de la respuesta
response_parsed = json.loads(response.text)

# Obtener datos sobre el clima actual
fact_weather = response_parsed['fact']

# Convertir a JSON y mostrar
print(json.dumps(fact_weather))
--------------------------------------------------------------------------------------------------------------------------