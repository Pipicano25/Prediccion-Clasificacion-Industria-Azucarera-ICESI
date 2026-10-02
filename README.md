# Prediccion-Clasificacion-Industria-Azucarera-ICESI

Proyecto académico de la Universidad ICESI que analiza datos de la industria azucarera para predecir y clasificar dos variables de desempeño agrícola: **TCH** (Toneladas de Caña por Hectárea) y **%Sac.Caña** (porcentaje de sacarosa), categorizándolas en niveles Bajo, Medio y Alto mediante técnicas de machine learning.

## Tecnologías

- Python 3
- pandas, NumPy, Matplotlib, Seaborn
- scikit-learn (Regresión Logística, Random Forest, K-Means)
- ReportLab (generación de informes PDF)
- Jupyter Notebook

## Instalación

1. Clonar el repositorio.
2. Crear un entorno virtual e instalar las dependencias de análisis de datos y machine learning (`pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, `reportlab`, `openpyxl`).
3. Los datos de entrada se encuentran en `data/raw/` (archivos `.xlsx`).

## Temas, tecnologías y notebooks

Cada notebook enlaza a su archivo en GitHub.

| Tema | Tecnologías | Notebooks |
|---|---|---|
| Predicción TCH/sacarosa: clasificación | scikit-learn, seaborn | [clasificacion.ipynb](https://github.com/Pipicano25/Prediccion-Clasificacion-Industria-Azucarera-ICESI/blob/main/scripts/clasificacion.ipynb) |
| Predicción TCH/sacarosa: regresión | scikit-learn, pandas | [exploracion_datos.ipynb](https://github.com/Pipicano25/Prediccion-Clasificacion-Industria-Azucarera-ICESI/blob/main/scripts/Regresion/exploracion_datos.ipynb), [exploracion_datos_2.ipynb](https://github.com/Pipicano25/Prediccion-Clasificacion-Industria-Azucarera-ICESI/blob/main/scripts/Regresion/exploracion_datos_2.ipynb), [regresion_tch_sac_cania.ipynb](https://github.com/Pipicano25/Prediccion-Clasificacion-Industria-Azucarera-ICESI/blob/main/scripts/Regresion/regresion_tch_sac_cania.ipynb) |
| Pipeline paso a paso (EDA + regresión) | scikit-learn, pandas | [Paso_a_Paso_Providencia.ipynb](https://github.com/Pipicano25/Prediccion-Clasificacion-Industria-Azucarera-ICESI/blob/main/example/Paso_a_Paso_Providencia.ipynb) |

## Uso

- `scripts/Regresion/exploracion_datos.ipynb` y `exploracion_datos_2.ipynb` — análisis exploratorio de los datos.
- `scripts/Regresion/regresion_tch_sac_cania.ipynb` — regresión sobre TCH y %Sac.Caña.
- `scripts/clasificacion.ipynb` — clasificación en categorías con modelos supervisados y K-Means.
- `scripts/verificar_datos.py` — verifica las columnas TCH y sacarosa del dataset.
- `scripts/generar_informe_ejecutivo.py` — genera un informe ejecutivo en PDF con resultados y conclusiones.
- `example/Paso_a_Paso_Providencia.ipynb` — notebook de ejemplo paso a paso.

## Estructura

- `data/raw/` — datasets en Excel (`BD_IPSA_1940.xlsx`, `HISTORICO_SUERTES.xlsx`).
- `scripts/` — notebooks y scripts de análisis, clasificación y reporte.
- `example/` — notebook de ejemplo.

## Licencia

MIT
