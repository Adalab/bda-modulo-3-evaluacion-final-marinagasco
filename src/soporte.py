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

    display(dataframe.describe(include="O").T)

    print("Principales estadísticos de las columnas numéricas:")

    display(dataframe.describe(exclude="O").T)

    return df_info

# Función que testea la normalidad de los datos:

def normalidad(dataframe, columna):
    """
    Evalúa la normalidad de una columna de datos de un DataFrame utilizando la prueba de Shapiro-Wilk.
    Parámetros:
        dataframe (DataFrame): El DataFrame que contiene los datos.
        columna (str): El nombre de la columna en el DataFrame que se va a evaluar para la normalidad.
    Returns:
        None: Imprime un mensaje indicando si los datos siguen o no una distribución normal.
    """
    statistic, p_value = stats.shapiro(dataframe[columna])
    if p_value > 0.05:
        print(f"Para la columna {columna} los datos siguen una distribución normal.")
    else:
        print(f"Para la columna {columna} los datos no siguen una distribución normal.")

#Función que comprueba la homegeneidad: 

def homogeneidad (dataframe, columna, columna_metrica):
    """
    Evalúa la homogeneidad de las varianzas entre grupos para una métrica específica en un DataFrame dado.
    Parámetros:
    - dataframe (DataFrame): El DataFrame que contiene los datos.
    - columna (str): El nombre de la columna que se utilizará para dividir los datos en grupos.
    - columna_metrica (str): El nombre de la columna que se utilizará para evaluar la homogeneidad de las varianzas.
    Returns:
    No devuelve nada directamente, pero imprime en la consola si las varianzas son homogéneas o no entre los grupos.
    Se utiliza la prueba de Levene para evaluar la homogeneidad de las varianzas. Si el valor p resultante es mayor que 0.05,
    se concluye que las varianzas son homogéneas; de lo contrario, se concluye que las varianzas no son homogéneas.
    """
    # lo primero que tenemos que hacer es crear tantos conjuntos de datos para cada una de las categorías que tenemos, Control Campaign y Test Campaign
    valores_evaluar = []
    for valor in dataframe[columna].unique():
        valores_evaluar.append(dataframe[dataframe[columna]== valor][columna_metrica])
    statistic, p_value = stats.levene(*valores_evaluar)
    if p_value > 0.05:
        print(f"Para la métrica {columna_metrica} las varianzas son homogéneas entre grupos.")
    else:
        print(f"Para la métrica {columna_metrica}, las varianzas no son homogéneas entre grupos.")

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


# Función para extraer info y ver categorías y subcategorías:


# Función para calcular el intervalo de confianza en un 95% de confiabilidad: