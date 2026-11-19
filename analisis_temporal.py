
import os
import time

import joblib
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "resultados")

paq = joblib.load(os.path.join(RES, "modelos_entrenados.joblib"))
est = pd.read_pickle(os.path.join(RES, "datos_estudio.pkl"))
est = est.loc[:, ~est.columns.duplicated()]
FEAT = paq["features"]

filas = []
for anio in [2023, 2024, 2025, 2026]:
    tr, te = est[est["ANIO"] < anio], est[est["ANIO"] == anio]
    for nombre, modelo in paq["modelos"].items():
        t0 = time.time()
        m = clone(modelo).fit(tr[FEAT], tr["GRAVE"])
        p = m.predict_proba(te[FEAT])[:, 1]
        pred = (p >= paq["umbrales"][nombre]).astype(int)
        filas.append({"Año de prueba": anio, "Entrenado con": f"2021–{anio - 1}", "Modelo": nombre,
                      "N entrenamiento": len(tr), "N prueba": len(te), "% graves": te["GRAVE"].mean(),
                      "F1": f1_score(te["GRAVE"], pred), "AUC-ROC": roc_auc_score(te["GRAVE"], p),
                      "Recall": recall_score(te["GRAVE"], pred), "Precisión": precision_score(te["GRAVE"], pred)})
        print(f"{anio} {nombre:18s} F1={filas[-1]['F1']:.3f} AUC={filas[-1]['AUC-ROC']:.3f} ({time.time() - t0:.1f}s)")

pd.DataFrame(filas).to_csv(os.path.join(RES, "backtest_anual.csv"), index=False, encoding="utf-8-sig")
print("Guardado resultados/backtest_anual.csv")
