import pandas as pd
import numpy as np

# Cargar el dataset
df = pd.read_excel('../data/raw/BD_IPSA_1940.xlsx')

print('VERIFICACIÓN DE VARIABLES CORRECTAS')
print('='*50)

# Verificar las columnas TCH y sacarosa
if 'TCH' in df.columns and 'sacarosa' in df.columns:
    print('Columnas encontradas: TCH y sacarosa')
    print(f'TCH - Rango: {df["TCH"].min():.2f} - {df["TCH"].max():.2f}')
    print(f'TCH - Media: {df["TCH"].mean():.2f}')
    print(f'Sacarosa - Rango: {df["sacarosa"].min():.2f} - {df["sacarosa"].max():.2f}')
    print(f'Sacarosa - Media: {df["sacarosa"].mean():.2f}')
    
    # Crear dataset limpio
    df_clean = df[['TCH', 'sacarosa']].copy()
    df_clean = df_clean.dropna()
    df_clean.columns = ['TCH', 'Sacarosa_Porcentaje']
    
    print(f'\nDataset limpio: {df_clean.shape}')
    print('Estadísticas descriptivas:')
    print(df_clean.describe())
else:
    print('Error: Columnas TCH o sacarosa no encontradas')
    print('Columnas disponibles:', df.columns.tolist())

