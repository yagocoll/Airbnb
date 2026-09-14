# FairNight

Recomendador de precios para anuncios de Airbnb: describe tu alojamiento y te dice a qué precio lo pondría
el mercado, comparándolo con miles de anuncios reales, explicando qué variables suben o bajan ese precio, y
comparándolo con anuncios reales parecidos al tuyo.

Trabajo Fin de Máster (Universidad Complutense de Madrid, 2026).

**App en vivo:** https://yagocoll-fair-night-airbnb-price-predictor.streamlit.app/

## Qué hace

- Predice el precio de un anuncio a partir de sus características (ubicación exacta, tipo de alojamiento,
  capacidad, condiciones de reserva, historial del anfitrión...).
- Explica esa predicción con SHAP: qué variables la suben, cuáles la bajan, y cuánto pesa cada una.
- Compara el anuncio con un conjunto de anuncios reales parecidos, no solo con el precio predicho.
- Avisa cuándo la predicción es menos fiable (alojamientos muy grandes, distritos con poca muestra,
  anuncios sin reseñas todavía).
- Incluye un mapa interactivo del mercado por barrio (precio, ocupación, densidad de anuncios, valoración).

## Ciudades y resultados

Un modelo independiente por ciudad, entrenado solo con datos reales de esa zona (sin aplicar la fórmula de
una ciudad a otra). Métricas en el conjunto de test (nunca visto durante el entrenamiento):

| Ciudad    | Anuncios | RMSE  | MAE   | R² (€) | R² (log) | MAPE  |
|-----------|---------:|------:|------:|-------:|---------:|------:|
| Barcelona |   13.258 |  141€ |   51€ |   0,79 |     0,89 | 20,6% |
| Madrid    |   18.949 |  150€ |   45€ |   0,40 |     0,77 | 24,6% |
| Valencia  |    7.234 |  110€ |   40€ |   0,37 |     0,76 | 24,1% |
| Málaga    |    8.742 |  155€ |   52€ |   0,57 |     0,78 | 22,3% |
| Sevilla   |    7.640 |  273€ |   56€ |   0,17 |     0,60 | 26,9% |
| Mallorca  |   11.044 |  606€ |  212€ |   0,54 |     0,78 | 30,6% |
| Euskadi   |    5.670 |  162€ |   71€ |   0,59 |     0,77 | 27,1% |
| **Total** | **72.537** |     |       |        |          |       |

El R² en euros depende de la dispersión de precios de cada mercado (una ciudad con villas de lujo o una cola
de precio muy larga tendrá un R² en euros más bajo aunque el modelo esté igual de bien ajustado); el R² en
escala logarítmica es la comparación justa entre ciudades.

En las 7 ciudades, tras comparar Ridge, Random Forest, Extra Trees e Hist Gradient Boosting, el modelo
ganador es siempre **Hist Gradient Boosting** (`max_leaf_nodes=63, max_iter=300, max_depth=10,
learning_rate=0.05` en Barcelona, ajustado de forma independiente por ciudad), sobre un vector de 71
features por anuncio.

## Estructura del repositorio

```
Airbnb/
  app/                        aplicación Streamlit
    app.py                    interfaz y orquestación (formulario, predicción, comparables, mapa)
    features.py                reconstruye el vector de features que espera cada price_model.pkl
    comparables.py             selección del conjunto de anuncios reales comparables
    requirements.txt
    assets/<ciudad>/           geojson de distritos, importancia de variables, fotos, logo
  notebooks/<ciudad>/          01_eda, 02_feature_engineering, 03_model_baseline,
                                04_model_training, 05_model_evaluation (una carpeta por ciudad)
  models/<ciudad>/             price_model.pkl + model_metadata.json (features, hiperparámetros, métricas)
  data/
    raw/<ciudad>/               listings.csv.gz de Inside Airbnb (no versionado)
    processed/<ciudad>/         csv limpios y con features, generados por los notebooks
```

## Cómo funciona el pipeline

Cada ciudad sigue el mismo proceso, notebook a notebook:

1. **`01_eda.ipynb`**: carga del dataset completo de Inside Airbnb (90 columnas), limpieza, tratamiento de
   nulos y duplicados, análisis univariante y bivariante, correlaciones, geografía y *amenities*.
2. **`02_feature_engineering.ipynb`**: construcción de las features finales a partir de las columnas de
   partida (codificación de categóricas, variables derivadas de host/reseñas/geografía, *log1p* sobre
   variables muy asimétricas).
3. **`03_model_baseline.ipynb`**: baselines (media, mediana, regresión lineal) y *split* train/test.
4. **`04_model_training.ipynb`**: comparativa de cuatro familias de modelos y ajuste de hiperparámetros con
   `RandomizedSearchCV`.
5. **`05_model_evaluation.ipynb`**: evaluación final en test, importancia de variables y SHAP.

Un hallazgo consistente en las 7 ciudades: la **estancia mínima** (`minimum_nights`) es la variable con más
peso del modelo en todas ellas, porque separa dos mercados distintos que conviven bajo el mismo anuncio de
Airbnb: uno turístico (estancias cortas, compite con hoteles) y otro de temporada (estancias largas, compite
con el alquiler residencial), con estructuras de precio muy distintas.

## La aplicación

| Página | Qué muestra |
|---|---|
| Inicio | Resumen de la zona y acceso directo a las tres secciones principales |
| Formulario | Características del anuncio, con mapa clicable para la ubicación exacta |
| Predicción | Precio recomendado, desglose SHAP ("por qué este precio") y simulador *what-if* |
| Comparar | El anuncio frente a un conjunto de anuncios reales parecidos, con veredicto y mapa |
| Modelo | Cuánto acierta el modelo, dónde es más y menos fiable (sesgo por distrito y por tramo de precio) |
| Variables | Catálogo de todas las variables de entrada del modelo |
| Análisis | Importancia de variables, curvas de dependencia parcial, SHAP e interacción entre variables |
| Explora el mercado | Mapa interactivo por barrio (precio, ocupación, densidad, valoración) |

## Cómo ejecutar en local

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r app/requirements.txt
streamlit run app/app.py
```

La app necesita los modelos (`models/<ciudad>/`) y los datos procesados (`data/processed/<ciudad>/`), ya
incluidos en el repositorio. Los datos crudos (`data/raw/`) no están versionados por peso; solo hacen falta
para regenerar el pipeline desde cero.

## Datos

Los datos de origen son de [Inside Airbnb](https://insideairbnb.com), un proyecto independiente de datos
abiertos que publica capturas periódicas de los anuncios activos en Airbnb. Cada ciudad usa su propio
snapshot (`listings.csv.gz` y `neighbourhoods.geojson`).

## Autor

Yago Coll Crespo, Trabajo Fin de Máster, Universidad Complutense de Madrid, 2026.
