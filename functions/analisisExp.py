import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def analizar_dataset(df):
    """
    Realiza un análisis exploratorio inicial de un DataFrame
    
    Parámetros:
    df (pandas.DataFrame): DataFrame a analizar
    
    Retorna:
    dict: Diccionario con información del análisis
    """
    analisis = {}
    
    # 1. Información general
    analisis['info_general'] = {
        'num_filas': len(df),
        'num_columnas': len(df.columns),
        'memoria_uso': df.memory_usage().sum() / 1024**2,  # En MB
        'columnas': df.columns.tolist()
    }
    
    # 2. Análisis de tipos de datos
    analisis['tipos_datos'] = df.dtypes.value_counts().to_dict()
    analisis['tipos_por_columna'] = df.dtypes.to_dict()
    
    # 3. Análisis de valores nulos
    nulos = df.isnull().sum()
    porcentaje_nulos = (nulos / len(df)) * 100
    analisis['valores_nulos'] = {
        'total_nulos': nulos.to_dict(),
        'porcentaje_nulos': porcentaje_nulos.to_dict()
    }
    
    # 4. Análisis de duplicados
    analisis['duplicados'] = {
        'filas_duplicadas': df.duplicated().sum(),
        'porcentaje_duplicados': (df.duplicated().sum() / len(df)) * 100
    }
    
    # 5. Estadísticas básicas para columnas numéricas
    columnas_numericas = df.select_dtypes(include=[np.number]).columns
    if len(columnas_numericas) > 0:
        analisis['estadisticas_numericas'] = df[columnas_numericas].describe().to_dict()
        
        # Detectar valores atípicos
        outliers_info = {}
        for columna in columnas_numericas:
            Q1 = df[columna].quantile(0.25)
            Q3 = df[columna].quantile(0.75)
            IQR = Q3 - Q1
            outliers = ((df[columna] < (Q1 - 1.5 * IQR)) | (df[columna] > (Q3 + 1.5 * IQR))).sum()
            outliers_info[columna] = {
                'num_outliers': outliers,
                'porcentaje_outliers': (outliers / len(df)) * 100
            }
        analisis['outliers'] = outliers_info
    
    # 6. Análisis de columnas categóricas
    columnas_categoricas = df.select_dtypes(include=['object']).columns
    if len(columnas_categoricas) > 0:
        cat_info = {}
        for columna in columnas_categoricas:
            valores_unicos = df[columna].nunique()
            cat_info[columna] = {
                'valores_unicos': valores_unicos,
                'valores_mas_comunes': df[columna].value_counts().head(5).to_dict()
            }
        analisis['analisis_categoricas'] = cat_info
    
    return analisis

def imprimir_reporte(analisis):
    """
    Imprime un reporte formateado del análisis
    """
    print("=== REPORTE DE ANÁLISIS EXPLORATORIO ===\n")
    
    print("INFORMACIÓN GENERAL:")
    print(f"- Número de filas: {analisis['info_general']['num_filas']:,}")
    print(f"- Número de columnas: {analisis['info_general']['num_columnas']}")
    print(f"- Uso de memoria: {analisis['info_general']['memoria_uso']:.2f} MB\n")
    
    print("VALORES NULOS:")
    for col, nulos in analisis['valores_nulos']['porcentaje_nulos'].items():
        if nulos > 0:
            print(f"- {col}: {nulos:.2f}% ({analisis['valores_nulos']['total_nulos'][col]} valores)")
    
    print(f"\nDUPLICADOS:")
    print(f"- Filas duplicadas: {analisis['duplicados']['filas_duplicadas']} ({analisis['duplicados']['porcentaje_duplicados']:.2f}%)")
    
    if 'outliers' in analisis:
        print("\nVALORES ATÍPICOS:")
        for col, info in analisis['outliers'].items():
            if info['num_outliers'] > 0:
                print(f"- {col}: {info['num_outliers']} outliers ({info['porcentaje_outliers']:.2f}%)")
    
    if 'analisis_categoricas' in analisis:
        print("\nVARIABLES CATEGÓRICAS:")
        for col, info in analisis['analisis_categoricas'].items():
            print(f"- {col}: {info['valores_unicos']} valores únicos")