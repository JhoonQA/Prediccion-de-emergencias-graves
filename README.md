# Emergencias graves en el Perú INDECI - SINPAD 2021–2026
**Predicción de emergencias graves en el Perú (INDECI – SINPAD 2021–2026)**
Random Forest (bagging) vs XGBoost (boosting), con el Árbol de Decisión como línea base y LightGBM como extra.

## Archivos
| Archivo | Qué es |
|---|---|
| `proyecto_indeci.py` | Programa principal: limpieza → gráficos → entrenamiento de los 4 modelos → evaluación (fases KDD). |
| `app_streamlit.py` | App web explicativa 12 secciones con gráficos interactivos y explicaciones de cada uno y simulador. |
| `analisis_temporal.py` | Validación año por año (backtesting 2023–2026) que usa la sección «Mapa y rendimiento» de la app. |
| `data/geo/peru_departamental.geojson` | Límites de los 25 departamentos para el mapa (repositorio público juaneladio/peru-geojson). |
| `ecuaciones.py` | Dibuja las ecuaciones del informe como imágenes. |
| `data/` | Datos originales del INDECI (Plataforma Nacional de Datos Abiertos). |
| `figuras/` | Todas las figuras (PNG, 200 dpi), útiles también para las diapositivas. |
| `resultados/` | Métricas, hiperparámetros, partición y modelos entrenados. |

## Cómo ejecutarlo
```bash
pip install pandas scikit-learn xgboost lightgbm matplotlib seaborn python-docx joblib streamlit plotly
python proyecto_indeci.py
python analisis_temporal.py
python -m streamlit run app_streamlit.py   # app web explicativa
```

## Resultados (prueba = año 2025, nunca visto al entrenar)
| Modelo | F1 | AUC | Recall |
|---|---|---|---|
| Árbol de Decisión (base) | 0.585 | 0.717 | 63.6 % |
| **Random Forest** | **0.642** | **0.778** | **72.1 %** |
| XGBoost | 0.619 | 0.768 | 65.3 % |
| LightGBM (extra) | 0.624 | 0.769 | 67.9 % |
