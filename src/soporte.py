#Importamos todas las librerías que vamos a necesitar para ejecutar las funciones que vamos a utilizar.
#*Si nos aparece en amarilllo no signifca que haya errores sino que el kernel no lo entiende.*

import pandas as pd
import numpy as np
import pandas as pd
import numpy as np
from scipy.stats import shapiro, levene
from scipy.stats import ttest_ind
from scipy.stats import mannwhitneyu
from scipy.stats import chi2_contingency
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from IPython.display import display

#Función que explora los datos: 

def exploracion(dataframe):
    '''
    Realiza una exploración inicial de un DataFrame de pandas proporcionando información 
    sobre los datos nulos, duplicados, tipos de datos, valores únicos y estadísticas descriptivas.

    Esta función imprime un resumen con:
    - El número de filas y columnas del DataFrame.
    - El porcentaje de valores nulos y no nulos por columna.
    - El tipo de dato de cada columna.
    - El número de valores únicos en cada columna.
    - El número total de registros duplicados y su porcentaje respecto al total de filas.
    - Las columnas que contienen datos nulos y las que no contienen datos nulos.
    - Estadísticas descriptivas para las columnas numéricas y categóricas (de tipo 'object').

    Parámetros:
    ----------
    df : pandas.DataFrame
        El DataFrame que se desea explorar.

    Retorna:
    --------
    df_info : pandas.DataFrame
        Un DataFrame con la siguiente información para cada columna:
        - '% nulos': Porcentaje de valores nulos.
        - '% no_nulos': Porcentaje de valores no nulos.
        - 'tipo_dato': Tipo de dato de la columna.
        - 'num_valores_unicos': Número de valores únicos en la columna.'''
    
    df_info = pd.DataFrame()

    df_info["% nulos"] = round(dataframe.isna().sum()/dataframe.shape[0]*100, 2).astype(str)+"%" # type: ignore
    df_info["% no_nulos"] = round(dataframe.notna().sum()/dataframe.shape[0]*100, 2).astype(str)+"%" # type: ignore
    df_info["tipo_dato"] = dataframe.dtypes # type: ignore
    df_info["num_valores_unicos"] = dataframe.nunique() # type: ignore

    print(f"""El DataFrame tiene {dataframe.shape[0]} filas y {dataframe.shape[1]} columnas. # type: ignore # type: ignore # type: ignore # type: ignore # type: ignore # type: ignore # type: ignore # type: ignore # type: ignore # type: ignore
    Tiene {dataframe.duplicated().sum()} datos duplicados, lo que supone un porcentaje de {round(dataframe.duplicated().sum()/dataframe.shape[0], 2)}% de los datos.

    Hay {len(list(df_info[(df_info["% nulos"] != "0.0%")].index))} columnas con datos nulos, y son: 
    {list(df_info[(df_info["% nulos"] != "0.0%")].index)}

    y sin nulos hay {len(list(df_info[(df_info["% nulos"] == "0.0%")].index))} columnas y son: 
    {list(df_info[(df_info["% nulos"] == "0.0%")].index)}

    A continuación tenemos un detalle sobre los datos nulos y los tipos y número de datos:""")

    display(df_info.head())

    print("Principales estadísticos de las columnas categóricas:")

    try:

        display(dataframe.describe(include="object").T)

    except: print("No existen variables categóricas")

    print("Principales estadísticos de las columnas numéricas:")

    display(dataframe.describe(exclude="object").T)

    return df_info

# Función de limpieza de negativos: 

def limpiar_negativos(df, col="Salary"):
    """
    Limpia los valores negativos de una columna especificada.

    Parámetros:  
    - df (pd.DataFrame): DataFrame a procesar.
    - col (str): Nombre de la columna donde eliminar valores negativos.

    Retorna: 
    - pd.DataFrame: DataFrame sin filas con valores negativos en la columna especificada.
    """
    # Eliminar filas con valores negativos
    df_limpio = df[df[col] >= 0]
    
    return df_limpio

# Función que testea la normalidad de los datos:

#Función que testea la normalidad de los datos:
def normalidad_dos_columnas(dataframe, columna1, columna2):
    """
    Evalúa la normalidad de dos columnas de datos de un DataFrame utilizando la prueba de Shapiro-Wilk.
    """
    for columna in [columna1, columna2]:
        if not pd.api.types.is_numeric_dtype(dataframe[columna]):
            print(f"La columna {columna} no es numérica y no puede evaluarse para normalidad.")
            continue

        stat, p_value = shapiro(dataframe[columna])
        if p_value > 0.05:
            print(f"Para la columna {columna}, los datos siguen una distribución normal.")
        else:
            print(f"Para la columna {columna}, los datos no siguen una distribución normal.")


# Función para calcular el test Mann-Whitney y ver si hay diferencias entre los grupos de estudio:

def test_man_whitney(dataframe, columnas_metricas, grupo_control, grupo_test, columna_grupos = "campaign_name"):
    """
    Realiza la prueba de Mann-Whitney U para comparar las medianas de las métricas entre dos grupos en un DataFrame dado.
    Parámetros:
    - dataframe (DataFrame): El DataFrame que contiene los datos.
    - columnas_metricas (list): Una lista de nombres de columnas que representan las métricas a comparar entre los grupos.
    - grupo_control (str): El nombre del grupo de control en la columna especificada por columna_grupos.
    - grupo_test (str): El nombre del grupo de test en la columna especificada por columna_grupos.
    - columna_grupos (str): El nombre de la columna que contiene la información de los grupos. Por defecto, "campaign_name".
    Returns
    No devuelve nada directamente, pero imprime en la consola si las medianas son diferentes o iguales para cada métrica.
    Se utiliza la prueba de Mann-Whitney U para evaluar si hay diferencias significativas entre los grupos.
    """
    # filtramos el DataFrame para quedarnos solo con los datos de control
    control = dataframe[dataframe[columna_grupos] == grupo_control]
    # filtramos el DataFrame para quedarnos solo con los datos de control
    test = dataframe[dataframe[columna_grupos] == grupo_test]
    # iteramos por las columnas de las metricas para ver si para cada una de ellas hay diferencias entre los grupos
    for metrica in columnas_metricas:
        # filtrams el conjunto de datos para quedarnos solo con la columna de la metrica que nos interesa
        metrica_control = control[metrica]
        metrica_test = test[metrica]
        # aplicamos el estadístico
        u_statistic, p_value = stats.mannwhitneyu(metrica_control, metrica_test)
        if p_value < 0.05:
            print(f"Para la métrica {metrica}, las medianas son diferentes.")
        else:
            print(f"Para la métrica {metrica}, las medianas son iguales.")
## llamamos a la función
test_man_whitney(df, metricas, "Control Campaign", "Test Campaign" )


#Alternativa al test Mann Whitnney: 

from scipy.stats import mannwhitneyu

# Dividir en dos grupos por nivel educativo
grupo_1 = ["High School or Below", "College", "Bachelor"]  # Menor nivel educativo
grupo_2 = ["Master", "Doctor"]  # Mayor nivel educativo

# Filtrar los datos según los grupos combinados
datos_grupo_1 = datos_filtrados[datos_filtrados['Education'].isin(grupo_1)]['Flights Booked']
datos_grupo_2 = datos_filtrados[datos_filtrados['Education'].isin(grupo_2)]['Flights Booked']

# Realizar el test Mann-Whitney U
u_stat, p_value = mannwhitneyu(datos_grupo_1, datos_grupo_2, alternative='two-sided')

# Mostrar resultados
if p_value < 0.05:
    print(f"Las medianas entre los dos grupos combinados son significativamente diferentes (p-valor = {p_value:.4f}).")
else:
    print(f"No hay diferencias significativas entre las medianas de los dos grupos combinados (p-valor = {p_value:.4f}).")

print(f"U-Statistic: {u_stat}")