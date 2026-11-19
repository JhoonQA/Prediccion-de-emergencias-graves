
import os
import time
import unicodedata
import warnings

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, roc_curve, confusion_matrix, classification_report)
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

warnings.filterwarnings("ignore")

BASE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(BASE, "data")
FIG = os.path.join(BASE, "figuras")
RES = os.path.join(BASE, "resultados")
os.makedirs(FIG, exist_ok=True)
os.makedirs(RES, exist_ok=True)

SEED = 42
ANIO_INICIO = 2021          # periodo de estudio: 2021 - 2026 (I semestre)
ANIO_TEST = 2025            # validación temporal: entrenar 2021-2024, probar 2025-2026

# Paleta coherente para todo el informe
C_RF, C_XGB, C_DT, C_LGB = "#2E7D32", "#C62828", "#1565C0", "#EF6C00"
C_GRAVE, C_LEVE = "#C62828", "#90A4AE"
COLORES = {"Árbol de Decisión": C_DT, "Random Forest": C_RF, "XGBoost": C_XGB, "LightGBM": C_LGB}

plt.rcParams.update({
    "figure.dpi": 110, "savefig.dpi": 200, "savefig.bbox": "tight",
    "font.size": 10, "axes.titlesize": 12, "axes.titleweight": "bold",
    "axes.spines.top": False, "axes.spines.right": False,
})

MESES = ["ENERO", "FEBRERO", "MARZO", "ABRIL", "MAYO", "JUNIO", "JULIO", "AGOSTO",
         "SETIEMBRE", "OCTUBRE", "NOVIEMBRE", "DICIEMBRE"]


def guardar(nombre):
    plt.savefig(os.path.join(FIG, nombre))
    plt.close()
    print(f"   [figura] figuras/{nombre}")


def normalizar(txt):
    """Mayúsculas, sin tildes y sin espacios sobrantes."""
    if pd.isna(txt):
        return "SIN DATO"
    txt = unicodedata.normalize("NFKD", str(txt)).encode("ascii", "ignore").decode()
    return " ".join(txt.upper().split())


# Unificación de nombres de peligro (el SINPAD cambió su catálogo en el tiempo)
MAPA_PELIGRO = {
    "LLUVIAS INTENSAS": "LLUVIA INTENSA",
    "INCENDIOS URBANOS": "INCENDIO URBANO",
    "INCENDIO URB. E INDUST.": "INCENDIO URBANO",
    "INCENDIOS FORESTALES": "INCENDIO FORESTAL",
    "HUAICO": "HUAYCO",
    "DERRUMBE CERROS": "DERRUMBE",
    "DERRUMBE DE CERRO": "DERRUMBE",
    "DERRUMBE DE ROCAS": "DERRUMBE",
    "TORMENTAS ELECTRICAS": "TORMENTA ELECTRICA",
    "GRANIZADAS": "GRANIZADA",
    "DESCENSO DE TEMPERATURA": "BAJAS TEMPERATURAS",
    "HELADA": "BAJAS TEMPERATURAS",
    "FRIAJE": "BAJAS TEMPERATURAS",
    "NEVADA": "BAJAS TEMPERATURAS",
    "DEFICIT HIDRICO": "SEQUIA",
    "ZOTROS": "OTROS",
}

GRUPO_PELIGRO = {
    "LLUVIA INTENSA": "Hidrometeorológico", "INUNDACION": "Hidrometeorológico",
    "VIENTOS FUERTES": "Hidrometeorológico", "TORMENTA ELECTRICA": "Hidrometeorológico",
    "GRANIZADA": "Hidrometeorológico", "SEQUIA": "Hidrometeorológico",
    "MAREJADA": "Hidrometeorológico", "OLEAJE ANOMALO": "Hidrometeorológico",
    "BAJAS TEMPERATURAS": "Temperaturas extremas",
    "DESLIZAMIENTO": "Geodinámico", "DERRUMBE": "Geodinámico", "HUAYCO": "Geodinámico",
    "EROSION": "Geodinámico", "ALUD": "Geodinámico", "SISMO": "Geodinámico",
    "ACTIVIDAD VOLCANICA": "Geodinámico",
    "INCENDIO URBANO": "Incendios", "INCENDIO FORESTAL": "Incendios",
}


def limpiar_peligro(p):
    p = normalizar(p)
    if p.startswith("INUNDACION"):
        return "INUNDACION"
    if p.startswith("COLAPSO DE ESTRUCTURA"):
        return "COLAPSO DE ESTRUCTURA"
    return MAPA_PELIGRO.get(p, p)


# FASE 1 - SELECCIÓN DE DATOS
def cargar_datos():
    print("\n=== FASE 1: SELECCIÓN DE DATOS ===")
    f1 = os.path.join(DATA, "BD_2003-2025_EMERGENCIAS.csv")
    f2 = os.path.join(DATA, "BD_2026_Semestre_I.csv")
    a = pd.read_csv(f1, sep=";", encoding="latin-1", dtype=str)
    b = pd.read_csv(f2, sep=";", encoding="latin-1", dtype=str)
    # El archivo 2026 trae los mismos 49 campos con nombres ligeramente distintos:
    # se alinean por posición usando los nombres del archivo histórico.
    b.columns = a.columns
    # Formatos de fecha mixtos: DD/MM/AAAA (2003-2024) y MM/DD/AA (2025 y 2026), como indica el diccionario
    def fechas(s):
        f = pd.to_datetime(s, format="%d/%m/%Y", errors="coerce")
        return f.fillna(pd.to_datetime(s, format="%m/%d/%y", errors="coerce"))
    a["FECHA_DT"] = fechas(a["FECHA  DE LA EMER"])
    b["FECHA_DT"] = fechas(b["FECHA  DE LA EMER"])
    print(f"   Fechas válidas: {pd.concat([a, b])['FECHA_DT'].notna().mean():.2%}")
    df = pd.concat([a, b], ignore_index=True)
    print(f"   Registros totales 2003-2026: {len(df):,}  |  columnas: {df.shape[1]}")

    df = df.rename(columns={
        "CODIGO DE EMERGENCIA-SINPAD": "CODIGO", "FECHA  DE LA EMER": "FECHA", "AÑO": "ANIO",
        "COD. DISTRITO": "UBIGEO", "DPTO.": "DEPARTAMENTO", "PROV.": "PROVINCIA", "DIST.": "DISTRITO",
        "REGIÓN NATURAL": "REGION",
    })
    return df


# FASE 2 - PREPROCESAMIENTO Y LIMPIEZA
COLS_DANIO = ["FALLECIDOS", "DESAPARECIDOS", "LESIONADOS", "DAMNIFICADOS", "AFECTADOS",
              "VIVIENDAS DESTRUIDAS", "VIVIENDAS AFECTADAS"]


def preprocesar(df):
    print("\n=== FASE 2: PREPROCESAMIENTO ===")
    n0 = len(df)
    df["ANIO"] = pd.to_numeric(df["ANIO"], errors="coerce")
    df = df.dropna(subset=["ANIO"]).copy()
    df["ANIO"] = df["ANIO"].astype(int)

    # columnas numéricas (decimales con coma en algunos registros)
    num_cols = [c for c in df.columns[11:] if c != "FECHA_DT"]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c].astype(str).str.replace(",", ".", regex=False), errors="coerce")
    nulos_num = int(df[num_cols].isna().sum().sum())
    df[num_cols] = df[num_cols].fillna(0)

    for c in ["DEPARTAMENTO", "PROVINCIA", "DISTRITO", "REGION", "TIPO DE PELIGRO", "MES"]:
        df[c] = df[c].map(normalizar)
    df["TIPO DE PELIGRO"] = df["TIPO DE PELIGRO"].replace({"INDUCIDOS POR LA ACCION HUMANA": "ACCION HUMANA"})
    df["PELIGRO_RAW"] = df["PELIGRO"]
    df["PELIGRO"] = df["PELIGRO"].map(limpiar_peligro)
    df["UBIGEO"] = df["UBIGEO"].astype(str).str.strip().str.zfill(6)
    df["MES_NUM"] = df["MES"].map({m: i + 1 for i, m in enumerate(MESES)})
    df = df.dropna(subset=["MES_NUM"])
    df["MES_NUM"] = df["MES_NUM"].astype(int)

    dup = int(df.duplicated(subset=["CODIGO", "UBIGEO", "ANIO", "PELIGRO"]).sum())
    df = df.drop_duplicates(subset=["CODIGO", "UBIGEO", "ANIO", "PELIGRO"])

    # valores negativos imposibles en conteos de daños
    neg = int((df[COLS_DANIO] < 0).sum().sum())
    df[COLS_DANIO] = df[COLS_DANIO].clip(lower=0)

    calidad = pd.DataFrame({
        "Control": ["Registros iniciales", "Años inválidos eliminados", "Celdas numéricas nulas (→0)",
                    "Registros duplicados eliminados", "Valores negativos corregidos", "Registros finales"],
        "Valor": [n0, n0 - len(df) - dup, nulos_num, dup, neg, len(df)],
    })
    calidad.to_csv(os.path.join(RES, "calidad_datos.csv"), index=False, encoding="utf-8-sig")
    print(calidad.to_string(index=False))
    return df



# FASE 3 - TRANSFORMACIÓN (variable objetivo + variables predictoras)

def transformar(df):
    print("\n=== FASE 3: TRANSFORMACIÓN ===")
    # Variable objetivo: emergencia GRAVE = deja pérdidas humanas, damnificados
    # (personas que perdieron su vivienda/medios de vida) o viviendas destruidas.
    df["GRAVE"] = ((df["FALLECIDOS"] + df["DESAPARECIDOS"] > 0) |
                   (df["DAMNIFICADOS"] > 0) |
                   (df["VIVIENDAS DESTRUIDAS"] > 0)).astype(int)

    df["PROV_KEY"] = df["DEPARTAMENTO"] + "|" + df["PROVINCIA"]

    def memoria(claves, prefijo, suav=10):
        g = df.groupby(claves + ["ANIO"])["GRAVE"].agg(["size", "sum"]).reset_index().sort_values("ANIO")
        g[["n_acum", "g_acum"]] = g.groupby(claves)[["size", "sum"]].cumsum().values
        g["n_prev"] = g["n_acum"] - g["size"]          # acumulado hasta el año anterior
        g["g_prev"] = g["g_acum"] - g["sum"]
        g["n_ult"] = g.groupby(claves)["size"].shift(1)  # solo el año inmediatamente anterior
        g["g_ult"] = g.groupby(claves)["sum"].shift(1)
        tasa_glob = (df.groupby("ANIO")["GRAVE"].sum().cumsum().shift(1) /
                     df.groupby("ANIO").size().cumsum().shift(1))
        g["glob"] = g["ANIO"].map(tasa_glob).fillna(0.4)
        # tasa suavizada (bayesiana) para distritos con pocas emergencias previas
        g[f"{prefijo}_TASA_GRAVE"] = (g["g_prev"] + suav * g["glob"]) / (g["n_prev"] + suav)
        g[f"{prefijo}_N_PREV"] = g["n_prev"]
        g[f"{prefijo}_TASA_ULT_ANIO"] = ((g["g_ult"].fillna(0) + suav * g["glob"]) /
                                         (g["n_ult"].fillna(0) + suav))
        cols = [f"{prefijo}_TASA_GRAVE", f"{prefijo}_N_PREV", f"{prefijo}_TASA_ULT_ANIO"]
        return df.merge(g[claves + ["ANIO"] + cols], on=claves + ["ANIO"], how="left")

    n_antes = len(df)
    df = memoria(["UBIGEO"], "DIST")
    df = memoria(["PROV_KEY"], "PROV")
    df = memoria(["UBIGEO", "PELIGRO"], "DISTPEL")
    assert len(df) == n_antes

    # --- INTENSIDAD DEL EVENTO: emergencias reportadas en el mismo departamento en los 7 días
    def ventana(clave, dias):
        res = pd.Series(0, index=df.index, dtype=float)
        ok = df["FECHA_DT"].notna()
        for _, g in df[ok].groupby(clave):
            t = g["FECHA_DT"].values.astype("datetime64[D]").astype(np.int64)
            orden = np.argsort(t)
            ts = t[orden]
            izq = np.searchsorted(ts, ts - dias, side="left")
            der = np.searchsorted(ts, ts, side="right")
            cnt = np.empty_like(ts, dtype=float)
            cnt[orden] = der - izq - 1
            res.loc[g.index] = cnt
        return res

    df["DEP_EMERG_7D"] = ventana("DEPARTAMENTO", 7)
    df["DIST_EMERG_30D"] = ventana("UBIGEO", 30)

    est = df[df["ANIO"] >= ANIO_INICIO].copy()   # 2021-2026: periodo de estudio

    # --- Variables temporales
    est["TEMPORADA_LLUVIAS"] = est["MES_NUM"].isin([12, 1, 2, 3, 4]).astype(int)
    est["MES_SIN"] = np.sin(2 * np.pi * est["MES_NUM"] / 12)
    est["MES_COS"] = np.cos(2 * np.pi * est["MES_NUM"] / 12)

    # --- Peligros poco frecuentes se agrupan en OTROS
    frec = est["PELIGRO"].value_counts()
    raros = frec[frec < 50].index
    est.loc[est["PELIGRO"].isin(raros), "PELIGRO"] = "OTROS"
    est["GRUPO_PELIGRO"] = est["PELIGRO"].map(GRUPO_PELIGRO).fillna("Otros / acción humana")

    # --- Codificación ordinal de categóricas (los árboles no necesitan one-hot)
    categoricas = ["DEPARTAMENTO", "REGION", "PELIGRO", "TIPO DE PELIGRO", "GRUPO_PELIGRO"]
    codigos = {}
    for c in categoricas:
        est[c + "_COD"] = est[c].astype("category").cat.codes
        codigos[c] = dict(enumerate(est[c].astype("category").cat.categories))

    est["UBIGEO_NUM"] = est["UBIGEO"].astype(int)
    features = ["PELIGRO_COD", "GRUPO_PELIGRO_COD", "TIPO DE PELIGRO_COD", "REGION_COD", "DEPARTAMENTO_COD",
                "UBIGEO_NUM", "MES_NUM", "MES_SIN", "MES_COS", "TEMPORADA_LLUVIAS",
                "DIST_TASA_GRAVE", "DIST_N_PREV", "DIST_TASA_ULT_ANIO", "PROV_TASA_GRAVE", "PROV_TASA_ULT_ANIO",
                "DISTPEL_TASA_GRAVE", "DISTPEL_N_PREV", "DEP_EMERG_7D", "DIST_EMERG_30D"]
    nombres = {"PELIGRO_COD": "Peligro", "GRUPO_PELIGRO_COD": "Grupo de peligro",
               "TIPO DE PELIGRO_COD": "Origen (natural/humano)", "REGION_COD": "Región natural",
               "DEPARTAMENTO_COD": "Departamento", "UBIGEO_NUM": "Código UBIGEO (ubicación)",
               "MES_NUM": "Mes", "MES_SIN": "Mes (seno)",
               "MES_COS": "Mes (coseno)", "TEMPORADA_LLUVIAS": "Temporada de lluvias",
               "DIST_TASA_GRAVE": "Tasa hist. gravedad distrito",
               "DIST_N_PREV": "N° hist. emergencias distrito",
               "DIST_TASA_ULT_ANIO": "Tasa gravedad distrito (año previo)",
               "PROV_TASA_GRAVE": "Tasa hist. gravedad provincia",
               "PROV_TASA_ULT_ANIO": "Tasa gravedad provincia (año previo)",
               "DISTPEL_TASA_GRAVE": "Tasa hist. gravedad del peligro en distrito",
               "DISTPEL_N_PREV": "N° hist. mismo peligro en distrito",
               "DEP_EMERG_7D": "Emergencias en el dpto. (7 días)",
               "DIST_EMERG_30D": "Emergencias en el distrito (30 días)"}

    print(f"   Periodo de estudio 2021-2026: {len(est):,} emergencias")
    print(f"   Emergencias GRAVES: {est['GRAVE'].sum():,} ({est['GRAVE'].mean():.1%})")
    print(f"   Variables predictoras: {len(features)}")
    print("   (Las columnas de daños NO se usan como predictoras: definen la clase -> evitaría fuga de datos)")
    pd.DataFrame({"Variable": [nombres[f] for f in features], "Columna": features}).to_csv(
        os.path.join(RES, "variables_predictoras.csv"), index=False, encoding="utf-8-sig")
    return df, est, features, nombres



# ANÁLISIS EXPLORATORIO (figuras para el informe)
def figura_kdd():
    fig, ax = plt.subplots(figsize=(11, 2.6))
    ax.axis("off")
    fases = ["Selección", "Preprocesamiento", "Transformación", "Minería de datos", "Evaluación"]
    desc = ["SINPAD-INDECI\n2021–2026", "Nulos, duplicados,\nnormalización", "Variable GRAVE,\nmemoria histórica",
            "Árbol · RF ·\nXGBoost · LightGBM", "F1, AUC, matriz\nde confusión"]
    cols = ["#1565C0", "#00838F", "#2E7D32", "#EF6C00", "#C62828"]
    for i, (f, d, c) in enumerate(zip(fases, desc, cols)):
        x = i * 2.2
        ax.add_patch(plt.Rectangle((x, 0.3), 1.8, 1.2, color=c, alpha=0.9, ec="none"))
        ax.text(x + 0.9, 1.15, f, ha="center", va="center", color="white", fontweight="bold", fontsize=9)
        ax.text(x + 0.9, 0.65, d, ha="center", va="center", color="white", fontsize=8)
        if i < 4:
            ax.annotate("", xy=(x + 2.15, 0.9), xytext=(x + 1.85, 0.9),
                        arrowprops=dict(arrowstyle="-|>", color="#455A64", lw=2))
    ax.annotate("", xy=(0.9, 0.28), xytext=(9.7, 0.28),
                arrowprops=dict(arrowstyle="-|>", color="#90A4AE", lw=1.2, ls="--", connectionstyle="arc3,rad=-0.12"))
    ax.text(5.3, -0.45, "proceso iterativo (retroalimentación)", ha="center", color="#78909C", fontsize=8, style="italic")
    ax.set_xlim(-0.1, 10.9)
    ax.set_ylim(-0.7, 1.6)
    guardar("fig01_proceso_kdd.png")


def eda(df, est):
    print("\n=== ANÁLISIS EXPLORATORIO ===")
    figura_kdd()

    # Fig 2: evolución anual 2003-2026
    anual = df.groupby("ANIO").agg(total=("GRAVE", "size"), graves=("GRAVE", "sum"))
    fig, ax = plt.subplots(figsize=(10, 4))
    colores = ["#B0BEC5" if a < ANIO_INICIO else "#1565C0" for a in anual.index]
    ax.bar(anual.index, anual["total"], color=colores, label="Total de emergencias")
    ax.bar(anual.index, anual["graves"], color=C_GRAVE, alpha=0.85, width=0.5, label="Emergencias graves")
    ax.axvline(ANIO_INICIO - 0.5, color="#37474F", ls="--", lw=1)
    ax.text(ANIO_INICIO - 0.4, anual["total"].max() * 0.95, "Periodo de estudio →", fontsize=9, color="#37474F")
    ax.text(2003, anual["total"].max() * 0.95, "Memoria histórica (2003–2020)", fontsize=9, color="#78909C")
    ax.set_title("Emergencias registradas en el SINPAD por año")
    ax.set_xlabel("Año (2026: solo I semestre)")
    ax.set_ylabel("N° de emergencias")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.88), frameon=False)
    guardar("fig02_emergencias_por_anio.png")

    # Fig 3: distribución de la clase objetivo
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.8), gridspec_kw={"width_ratios": [1, 1.6]})
    vc = est["GRAVE"].value_counts().sort_index()
    axes[0].pie(vc, colors=[C_LEVE, C_GRAVE], autopct="%1.1f%%", pctdistance=0.78,
                startangle=90, wedgeprops=dict(width=0.45, edgecolor="white"),
                textprops={"fontsize": 10, "color": "white", "fontweight": "bold"})
    axes[0].text(0, 0, f"{len(est):,}\nemergencias", ha="center", va="center", fontsize=9)
    axes[0].legend(["No grave", "Grave"], loc="lower center", bbox_to_anchor=(0.5, -0.12), ncol=2, frameon=False)
    axes[0].set_title("Clase objetivo (2021–2026)")
    pa = est.groupby("ANIO")["GRAVE"].agg(["size", "mean"])
    axes[1].bar(pa.index.astype(str), pa["size"], color="#90CAF9")
    ax2 = axes[1].twinx()
    ax2.plot(pa.index.astype(str), pa["mean"] * 100, "o-", color=C_GRAVE, lw=2)
    ax2.set_ylabel("% graves", color=C_GRAVE)
    ax2.set_ylim(0, 100)
    ax2.spines["right"].set_visible(True)
    axes[1].set_ylabel("N° de emergencias")
    axes[1].set_title("Emergencias y % de graves por año")
    guardar("fig03_clase_objetivo.png")

    # Fig 4: peligros más frecuentes y su gravedad
    top = est.groupby("PELIGRO")["GRAVE"].agg(["size", "mean"]).sort_values("size").tail(15)
    fig, ax = plt.subplots(figsize=(9, 5.5))
    cmap = plt.cm.Reds
    bars = ax.barh(top.index, top["size"], color=cmap(0.25 + 0.75 * top["mean"]))
    for bar, (n, m) in zip(bars, top.values):
        ax.text(bar.get_width() + top["size"].max() * 0.01, bar.get_y() + bar.get_height() / 2,
                f"{int(n):,}  ({m:.0%} graves)", va="center", fontsize=8)
    ax.set_xlim(0, top["size"].max() * 1.3)
    ax.set_title("Peligros más frecuentes (2021–2026) y proporción de emergencias graves")
    ax.set_xlabel("N° de emergencias")
    guardar("fig04_peligros_frecuentes.png")

    # Fig 5: mapa de calor departamento x mes
    piv = est.pivot_table(index="DEPARTAMENTO", columns="MES_NUM", values="GRAVE", aggfunc="size", fill_value=0)
    piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]
    piv.columns = [m[:3].title() for m in MESES]
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(piv, cmap="YlOrRd", linewidths=0.3, ax=ax, cbar_kws={"label": "N° de emergencias"})
    ax.set_title("Emergencias por departamento y mes (2021–2026)")
    ax.set_xlabel("Mes")
    ax.set_ylabel("")
    guardar("fig05_mapa_calor_departamento_mes.png")

    # Fig 6: gravedad por región natural y por grupo de peligro
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    r = est.groupby("REGION")["GRAVE"].mean().sort_values() * 100
    axes[0].barh(r.index, r.values, color="#5C6BC0")
    for i, v in enumerate(r.values):
        axes[0].text(v + 1, i, f"{v:.1f}%", va="center")
    axes[0].set_title("% de emergencias graves por región natural")
    axes[0].set_xlim(0, 100)
    g = est.groupby("GRUPO_PELIGRO")["GRAVE"].mean().sort_values() * 100
    axes[1].barh(g.index, g.values, color="#EF6C00")
    for i, v in enumerate(g.values):
        axes[1].text(v + 1, i, f"{v:.1f}%", va="center")
    axes[1].set_title("% de emergencias graves por grupo de peligro")
    axes[1].set_xlim(0, 100)
    guardar("fig06_gravedad_region_grupo.png")

    # Fig 7: estacionalidad
    m = est.groupby("MES_NUM")["GRAVE"].agg(["size", "mean"])
    fig, ax = plt.subplots(figsize=(10, 3.8))
    ax.bar([x[:3].title() for x in MESES], m["size"], color=["#1E88E5" if i in [12, 1, 2, 3, 4] else "#B0BEC5" for i in m.index])
    ax.set_ylabel("N° de emergencias")
    ax2 = ax.twinx()
    ax2.plot([x[:3].title() for x in MESES], m["mean"] * 100, "o-", color=C_GRAVE, lw=2, label="% graves")
    ax2.set_ylim(0, 100)
    ax2.set_ylabel("% graves", color=C_GRAVE)
    ax2.spines["right"].set_visible(True)
    ax.set_title("Estacionalidad: emergencias por mes (azul = temporada de lluvias dic–abr)")
    guardar("fig07_estacionalidad.png")

    # Fig 8: correlación entre variables numéricas de daños
    corr_cols = ["FALLECIDOS", "LESIONADOS", "DAMNIFICADOS", "AFECTADOS", "VIVIENDAS DESTRUIDAS",
                 "VIVIENDAS AFECTADAS", "HAS CULTIVO AFECTADO", "GRAVE"]
    corr = est[corr_cols].corr(method="spearman")
    fig, ax = plt.subplots(figsize=(8, 6.5))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, vmin=-1, vmax=1, square=True,
                ax=ax, annot_kws={"size": 8}, cbar_kws={"shrink": 0.8})
    ax.set_title("Correlación de Spearman entre variables de daño y la clase GRAVE")
    guardar("fig08_correlacion_danios.png")



# TEORÍA: Gini vs Entropía (figura explicativa)
def figura_impureza():
    p = np.linspace(0.001, 0.999, 300)
    gini = 1 - p ** 2 - (1 - p) ** 2
    ent = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    err = 1 - np.maximum(p, 1 - p)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(p, ent, lw=2.5, color="#6A1B9A", label="Entropía  H = −Σ pᵢ log₂ pᵢ")
    ax.plot(p, gini, lw=2.5, color=C_DT, label="Gini  G = 1 − Σ pᵢ²")
    ax.plot(p, err, lw=1.5, ls="--", color="#78909C", label="Error de clasificación")
    ax.axvline(0.5, color="#B0BEC5", lw=1, ls=":")
    ax.annotate("máxima impureza\n(50% grave / 50% no grave)", xy=(0.5, 1), xytext=(0.62, 0.85),
                arrowprops=dict(arrowstyle="->"), fontsize=8)
    ax.annotate("nodo puro", xy=(0.02, 0.02), xytext=(0.08, 0.3), arrowprops=dict(arrowstyle="->"), fontsize=8)
    ax.set_xlabel("p = proporción de emergencias graves en el nodo")
    ax.set_ylabel("Impureza")
    ax.set_title("Medidas de impureza para la división de nodos")
    ax.legend(frameon=False)
    guardar("fig09_gini_vs_entropia.png")



# FASE 4 - MINERÍA DE DATOS
def metricas(nombre, modelo, X, y, t_entr, umbral=0.5):
    proba = modelo.predict_proba(X)[:, 1]
    pred = (proba >= umbral).astype(int)
    return {"Modelo": nombre, "Accuracy": accuracy_score(y, pred), "Precisión": precision_score(y, pred),
            "Recall": recall_score(y, pred), "F1": f1_score(y, pred), "AUC-ROC": roc_auc_score(y, proba),
            "Umbral": umbral, "Tiempo entrenamiento (s)": t_entr}, proba, pred


def umbral_optimo(modelo, X, y, cv):
    """Umbral que maximiza F1 usando predicciones fuera de pliegue SOLO del entrenamiento."""
    from sklearn.model_selection import cross_val_predict
    oof = cross_val_predict(modelo, X, y, cv=cv, method="predict_proba")[:, 1]
    grid = np.arange(0.20, 0.71, 0.01)
    f1s = [f1_score(y, (oof >= u).astype(int)) for u in grid]
    return float(grid[int(np.argmax(f1s))]), grid, f1s


def entrenar(est, features, nombres):
    print("\n=== FASE 4: MINERÍA DE DATOS ===")
    tr = est[est["ANIO"] < ANIO_TEST]
    te = est[est["ANIO"] == ANIO_TEST]
    t26 = est[est["ANIO"] > ANIO_TEST]          # 2026-I: validación fuera de tiempo (cambio de registro)
    X_tr, y_tr, X_te, y_te = tr[features], tr["GRAVE"], te[features], te["GRAVE"]
    print(f"   Entrenamiento 2021-{ANIO_TEST-1}: {len(tr):,}  ({y_tr.mean():.1%} graves)")
    print(f"   Prueba {ANIO_TEST}:             {len(te):,}  ({y_te.mean():.1%} graves)")
    print(f"   Validación 2026-I:        {len(t26):,}  ({t26['GRAVE'].mean():.1%} graves)")
    pd.DataFrame({"Conjunto": [f"Entrenamiento (2021–{ANIO_TEST-1})", f"Prueba ({ANIO_TEST})",
                               "Validación fuera de tiempo (2026 – I sem.)"],
                  "Registros": [len(tr), len(te), len(t26)],
                  "% graves": [y_tr.mean() * 100, y_te.mean() * 100, t26["GRAVE"].mean() * 100]}
                 ).to_csv(os.path.join(RES, "particion.csv"), index=False, encoding="utf-8-sig")

    cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=SEED)
    espacios = {
        "Árbol de Decisión": (DecisionTreeClassifier(random_state=SEED),
                              {"criterion": ["gini", "entropy"], "max_depth": [4, 6, 8, 10, 12, 15, None],
                               "min_samples_leaf": [1, 5, 10, 20, 50, 100]}),
        "Random Forest": (RandomForestClassifier(random_state=SEED, n_jobs=-1),
                          {"n_estimators": [200, 300, 500], "max_depth": [10, 15, 20, None],
                           "min_samples_leaf": [1, 3, 5, 10], "max_features": ["sqrt", 0.5],
                           "criterion": ["gini", "entropy"]}),
        "XGBoost": (XGBClassifier(random_state=SEED, n_jobs=-1, eval_metric="logloss", tree_method="hist"),
                    {"n_estimators": [300, 500, 800], "learning_rate": [0.03, 0.05, 0.1],
                     "max_depth": [4, 6, 8], "subsample": [0.8, 1.0], "colsample_bytree": [0.7, 0.9, 1.0],
                     "min_child_weight": [1, 5, 10]}),
        "LightGBM": (LGBMClassifier(random_state=SEED, n_jobs=-1, verbose=-1),
                     {"n_estimators": [300, 500, 800], "learning_rate": [0.03, 0.05, 0.1],
                      "num_leaves": [15, 31, 63], "min_child_samples": [10, 20, 50],
                      "subsample": [0.8, 1.0], "subsample_freq": [1], "colsample_bytree": [0.7, 0.9, 1.0]}),
    }

    resultados, modelos, probas, preds, mejores = [], {}, {}, {}, []
    umbrales, curvas_umbral = {}, {}
    for nombre, (base, grid) in espacios.items():
        print(f"\n   >> {nombre}: búsqueda de hiperparámetros (RandomizedSearchCV, 3-fold)...")
        busq = RandomizedSearchCV(base, grid, n_iter=15, scoring="f1", cv=cv, random_state=SEED, n_jobs=1)
        busq.fit(X_tr, y_tr)
        mejor = busq.best_estimator_
        u, grid_u, f1_u = umbral_optimo(mejor, X_tr, y_tr, cv)
        curvas_umbral[nombre] = (grid_u, f1_u, u)
        t0 = time.time()
        mejor.fit(X_tr, y_tr)                       # reentrenamiento para medir tiempo
        t = time.time() - t0
        fila, proba, pred = metricas(nombre, mejor, X_te, y_te, t, umbral=u)
        fila["F1 (validación cruzada)"] = busq.best_score_
        f05 = f1_score(y_te, (proba >= 0.5).astype(int))
        fila["F1 con umbral 0.5"] = f05
        resultados.append(fila)
        modelos[nombre], probas[nombre], preds[nombre] = mejor, proba, pred
        umbrales[nombre] = u
        mejores.append({"Modelo": nombre, "Mejores hiperparámetros": str(busq.best_params_)})
        print(f"      mejores parámetros: {busq.best_params_}")
        print(f"      umbral óptimo = {u:.2f} | F1 prueba = {fila['F1']:.4f} (umbral 0.5: {f05:.4f}) | "
              f"AUC = {fila['AUC-ROC']:.4f} | tiempo = {t:.1f}s")

    res = pd.DataFrame(resultados)
    res.to_csv(os.path.join(RES, "metricas_modelos.csv"), index=False, encoding="utf-8-sig")
    # Validación fuera de tiempo 2026-I
    r26 = [metricas(n, m, t26[features], t26["GRAVE"], 0, umbral=umbrales[n])[0] for n, m in modelos.items()]
    # Figura: selección del umbral óptimo (F1 fuera de pliegue en entrenamiento)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    for n, (g, f, u) in curvas_umbral.items():
        ax.plot(g, f, lw=2, color=COLORES[n], label=f"{n} (umbral = {u:.2f})")
        ax.plot(u, max(f), "o", color=COLORES[n])
    ax.axvline(0.5, ls=":", color="#90A4AE")
    ax.text(0.505, ax.get_ylim()[0] + 0.01, "umbral por defecto 0.5", fontsize=8, color="#78909C")
    ax.set_xlabel("Umbral de decisión  (grave si P ≥ umbral)")
    ax.set_ylabel("F1 (validación cruzada, entrenamiento)")
    ax.set_title("Selección del umbral de decisión que maximiza F1")
    ax.legend(frameon=False, fontsize=8)
    guardar("fig20_umbral_optimo.png")
    pd.DataFrame(r26).drop(columns=["Tiempo entrenamiento (s)"]).to_csv(
        os.path.join(RES, "metricas_2026.csv"), index=False, encoding="utf-8-sig")
    pd.DataFrame(mejores).to_csv(os.path.join(RES, "hiperparametros.csv"), index=False, encoding="utf-8-sig")
    with open(os.path.join(RES, "reportes_clasificacion.txt"), "w", encoding="utf-8") as f:
        for n in modelos:
            f.write(f"===== {n} =====\n")
            f.write(classification_report(y_te, preds[n], target_names=["No grave", "Grave"], digits=4))
            f.write("\n")
    # Guardar modelos y casos de prueba para el simulador de la exposición
    import joblib
    cols_desc = ["CODIGO", "FECHA", "ANIO", "MES", "DEPARTAMENTO", "PROVINCIA", "DISTRITO", "PELIGRO", "REGION",
                 "FALLECIDOS", "DESAPARECIDOS", "DAMNIFICADOS", "AFECTADOS", "VIVIENDAS DESTRUIDAS", "GRAVE"]
    joblib.dump({"modelos": modelos, "umbrales": umbrales, "features": features, "nombres": nombres,
                 "casos": te[cols_desc + features].reset_index(drop=True)},
                os.path.join(RES, "modelos_entrenados.joblib"), compress=3)
    return modelos, probas, preds, res, (X_tr, y_tr, X_te, y_te)


# Figuras específicas de cada algoritmo
def figuras_arbol(X_tr, y_tr, features, nombres):
    # Árbol pequeño (profundidad 3) SOLO para visualizar la estructura de datos
    dt = DecisionTreeClassifier(max_depth=3, criterion="gini", min_samples_leaf=50, random_state=SEED)
    dt.fit(X_tr, y_tr)
    fig, ax = plt.subplots(figsize=(18, 8))
    plot_tree(dt, feature_names=[nombres[f] for f in features], class_names=["No grave", "Grave"],
              filled=True, rounded=True, impurity=True, proportion=False, fontsize=9, ax=ax)
    ax.set_title("Estructura del Árbol de Decisión (profundidad 3, criterio Gini)", fontsize=14)
    guardar("fig10_arbol_decision.png")

    # Profundidad vs F1: sobreajuste del árbol único
    from sklearn.model_selection import cross_val_score
    prof = list(range(1, 26, 2))
    f1_tr, f1_cv = [], []
    for d in prof:
        m = DecisionTreeClassifier(max_depth=d, random_state=SEED).fit(X_tr, y_tr)
        f1_tr.append(f1_score(y_tr, m.predict(X_tr)))
        f1_cv.append(cross_val_score(DecisionTreeClassifier(max_depth=d, random_state=SEED), X_tr, y_tr,
                                     cv=3, scoring="f1").mean())
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(prof, f1_tr, "o-", color="#90A4AE", label="Entrenamiento")
    ax.plot(prof, f1_cv, "o-", color=C_DT, label="Validación cruzada")
    best = prof[int(np.argmax(f1_cv))]
    ax.axvline(best, ls="--", color=C_DT, lw=1)
    ax.fill_between(prof, f1_cv, f1_tr, color="#FFCDD2", alpha=0.5, label="Brecha = sobreajuste")
    ax.set_xlabel("Profundidad máxima del árbol")
    ax.set_ylabel("F1")
    ax.set_title("Un solo árbol se sobreajusta al crecer")
    ax.legend(frameon=False)
    guardar("fig11_arbol_sobreajuste.png")


def figuras_bagging_boosting():
    """Diagrama conceptual: Random Forest (paralelo) vs XGBoost (secuencial)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.6))
    for ax in axes:
        ax.axis("off")
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)

    def arbol(ax, x, y, c, s=0.55):
        ax.plot([x, x - s, x, x + s], [y, y - 1, y, y - 1], color=c, lw=1.5)
        ax.plot([x - s, x - s * 1.5, x - s, x - s * 0.5], [y - 1, y - 2, y - 1, y - 2], color=c, lw=1.2)
        ax.plot([x + s, x + s * 0.5, x + s, x + s * 1.5], [y - 1, y - 2, y - 1, y - 2], color=c, lw=1.2)
        for px, py in [(x, y), (x - s, y - 1), (x + s, y - 1)]:
            ax.add_patch(plt.Circle((px, py), 0.18, color=c))
        for px in [x - s * 1.5, x - s * 0.5, x + s * 0.5, x + s * 1.5]:
            ax.add_patch(plt.Rectangle((px - 0.15, y - 2.2), 0.3, 0.3, color=c, alpha=0.6))

    a = axes[0]
    a.set_title("Random Forest  (bagging: árboles en PARALELO)", color=C_RF)
    a.add_patch(plt.Rectangle((3.5, 8.6), 3, 1, color="#ECEFF1"))
    a.text(5, 9.1, "Datos de entrenamiento", ha="center", va="center", fontsize=9)
    for i, x in enumerate([1.5, 4, 6.5, 9]):
        a.annotate("", xy=(x, 7.3), xytext=(5, 8.6), arrowprops=dict(arrowstyle="->", color="#90A4AE"))
        a.text(x, 7.6, f"muestra bootstrap {i+1}" if i < 3 else "muestra B", ha="center", fontsize=7)
        arbol(a, x, 6.8, C_RF)
        a.annotate("", xy=(5, 2.6), xytext=(x, 4.4), arrowprops=dict(arrowstyle="->", color="#90A4AE"))
    a.text(7.8, 5.8, "···", fontsize=20, color=C_RF)
    a.add_patch(plt.Rectangle((2.3, 1.4), 5.4, 1.2, color=C_RF))
    a.text(5, 2.0, "VOTACIÓN por mayoría", ha="center", va="center", color="white", fontweight="bold")
    a.text(5, 0.6, "Cada árbol es independiente → reduce la VARIANZA", ha="center", fontsize=8, style="italic")

    b = axes[1]
    b.set_title("XGBoost  (boosting: árboles en SECUENCIA)", color=C_XGB)
    xs = [1.0, 3.6, 6.2, 8.8]
    for i, x in enumerate(xs):
        arbol(b, x, 7.5, C_XGB, s=0.5)
        b.text(x, 8.2, f"árbol {i+1}" if i < 3 else "árbol M", ha="center", fontsize=8)
        if i < 3:
            b.annotate("", xy=(xs[i + 1] - 0.9, 6.5), xytext=(x + 0.9, 6.5),
                       arrowprops=dict(arrowstyle="-|>", color=C_XGB, lw=1.8))
            b.text((x + xs[i + 1]) / 2, 6.9, "errores\n(residuos)", ha="center", fontsize=6.5, color=C_XGB)
        b.annotate("", xy=(5, 2.6), xytext=(x, 5.2), arrowprops=dict(arrowstyle="->", color="#90A4AE"))
    b.add_patch(plt.Rectangle((1.0, 1.4), 8.0, 1.2, color=C_XGB))
    b.text(5, 2.0, "SUMA:  F(x) = f₁(x) + η·f₂(x) + … + η·f_M(x)", ha="center", va="center", color="white",
           fontweight="bold", fontsize=9)
    b.text(5, 0.6, "Cada árbol corrige al anterior → reduce el SESGO", ha="center", fontsize=8, style="italic")
    guardar("fig12_bagging_vs_boosting.png")


def figuras_rf_xgb(modelos, X_tr, y_tr, X_te, y_te):
    # Random Forest: cómo mejora al agregar árboles (error OOB)
    rf_best = modelos["Random Forest"].get_params()
    n_list = [1, 5, 10, 25, 50, 100, 200, 300, 500]
    oob = []
    for n in n_list:
        p = {k: v for k, v in rf_best.items() if k not in ("n_estimators", "oob_score", "warm_start", "bootstrap")}
        m = RandomForestClassifier(**p, n_estimators=max(n, 2), oob_score=True, bootstrap=True)
        m.fit(X_tr, y_tr)
        oob.append(1 - m.oob_score_)
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(n_list, oob, "o-", color=C_RF, lw=2)
    ax.set_xscale("log")
    ax.set_xlabel("N° de árboles en el bosque (escala log)")
    ax.set_ylabel("Error OOB (out-of-bag)")
    ax.set_title("Random Forest: más árboles → menor error (y luego se estabiliza)")
    for n, e in zip(n_list, oob):
        if n in (1, 10, 100, 500):
            ax.annotate(f"{e:.3f}", (n, e), textcoords="offset points", xytext=(0, 8), ha="center", fontsize=8)
    guardar("fig13_rf_error_oob.png")
    oob_res = dict(zip(n_list, oob))

    # XGBoost: curva de aprendizaje por iteración
    xgb_p = modelos["XGBoost"].get_params()
    m = XGBClassifier(**{**xgb_p, "eval_metric": "logloss"})
    m.fit(X_tr, y_tr, eval_set=[(X_tr, y_tr), (X_te, y_te)], verbose=False)
    ev = m.evals_result()
    fig, ax = plt.subplots(figsize=(7.5, 4))
    ax.plot(ev["validation_0"]["logloss"], color="#90A4AE", lw=2, label="Entrenamiento")
    ax.plot(ev["validation_1"]["logloss"], color=C_XGB, lw=2, label="Prueba")
    ax.set_xlabel("Iteración (n° de árboles agregados)")
    ax.set_ylabel("Log-loss")
    te_loss = np.array(ev["validation_1"]["logloss"])
    it = int(te_loss.argmin())
    ax.axvline(it, ls="--", color="#455A64", lw=1)
    ax.annotate(f"mínimo en prueba\n(árbol {it})", xy=(it, te_loss[it]), xytext=(it + 120, te_loss[it] - 0.08),
                arrowprops=dict(arrowstyle="->"), fontsize=8)
    ax.set_title("XGBoost: la pérdida de entrenamiento baja siempre; la de prueba, solo al inicio")
    ax.legend(frameon=False)
    guardar("fig14_xgb_curva_aprendizaje.png")
    return oob_res


def figuras_evaluacion(modelos, probas, preds, res, y_te, features, nombres):
    print("\n=== FASE 5: EVALUACIÓN ===")
    orden = ["Árbol de Decisión", "Random Forest", "XGBoost", "LightGBM"]

    # Matrices de confusión
    fig, axes = plt.subplots(1, 4, figsize=(17, 4))
    for ax, n in zip(axes, orden):
        cm = confusion_matrix(y_te, preds[n])
        cmn = cm / cm.sum(axis=1, keepdims=True)
        etiquetas = np.array([[f"{cm[i, j]:,}\n({cmn[i, j]:.1%})" for j in range(2)] for i in range(2)])
        sns.heatmap(cmn, annot=etiquetas, fmt="", cmap=sns.light_palette(COLORES[n], as_cmap=True),
                    cbar=False, ax=ax, xticklabels=["No grave", "Grave"], yticklabels=["No grave", "Grave"],
                    vmin=0, vmax=1, annot_kws={"size": 10})
        ax.set_title(n, color=COLORES[n])
        ax.set_xlabel("Predicción")
        ax.set_ylabel("Real")
    guardar("fig15_matrices_confusion.png")

    # Curvas ROC
    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    for n in orden:
        fpr, tpr, _ = roc_curve(y_te, probas[n])
        ax.plot(fpr, tpr, lw=2.2, color=COLORES[n], label=f"{n} (AUC = {roc_auc_score(y_te, probas[n]):.3f})")
    ax.plot([0, 1], [0, 1], "--", color="#B0BEC5", label="Azar (AUC = 0.5)")
    ax.set_xlabel("Tasa de falsos positivos")
    ax.set_ylabel("Tasa de verdaderos positivos (Recall)")
    ax.set_title("Curvas ROC en el conjunto de prueba (2025)")
    ax.legend(loc="lower right", frameon=False, fontsize=9)
    guardar("fig16_curvas_roc.png")

    # Comparación de métricas
    met = ["Accuracy", "Precisión", "Recall", "F1", "AUC-ROC"]
    r = res.set_index("Modelo").loc[orden]
    fig, ax = plt.subplots(figsize=(11, 4.5))
    w = 0.2
    x = np.arange(len(met))
    for i, n in enumerate(orden):
        vals = r.loc[n, met].values.astype(float)
        bars = ax.bar(x + (i - 1.5) * w, vals, w, color=COLORES[n], label=n)
        for b, v in zip(bars, vals):
            ax.text(b.get_x() + b.get_width() / 2, v + 0.005, f"{v:.3f}", ha="center", fontsize=6.5, rotation=90)
    ax.set_xticks(x)
    ax.set_xticklabels(met)
    lo = max(0, r[met].values.min() - 0.1)
    ax.set_ylim(lo, 1.0)
    ax.set_title("Comparación de métricas en el conjunto de prueba")
    ax.legend(ncol=4, frameon=False, loc="upper center", bbox_to_anchor=(0.5, -0.08))
    guardar("fig17_comparacion_metricas.png")

    # Importancia de variables: RF vs XGBoost
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, n in zip(axes, ["Random Forest", "XGBoost"]):
        imp = pd.Series(modelos[n].feature_importances_, index=[nombres[f] for f in features]).sort_values()
        imp = imp / imp.sum()
        ax.barh(imp.index, imp.values, color=COLORES[n])
        for i, v in enumerate(imp.values):
            ax.text(v + 0.003, i, f"{v:.1%}", va="center", fontsize=8)
        ax.set_title(f"Importancia de variables – {n}", color=COLORES[n])
        ax.set_xlim(0, imp.max() * 1.4)
    fig.subplots_adjust(wspace=0.75)
    guardar("fig18_importancia_variables.png")

    # Tiempo de entrenamiento
    fig, ax = plt.subplots(figsize=(7, 3.2))
    t = r["Tiempo entrenamiento (s)"]
    ax.barh(t.index, t.values, color=[COLORES[n] for n in t.index])
    for i, v in enumerate(t.values):
        ax.text(v, i, f" {v:.2f} s", va="center")
    ax.set_xlabel("Segundos")
    ax.set_title("Tiempo de entrenamiento (mejores hiperparámetros)")
    ax.invert_yaxis()
    guardar("fig19_tiempo_entrenamiento.png")


# ===========================================================================
def main():
    t0 = time.time()
    df = cargar_datos()
    df = preprocesar(df)
    df, est, features, nombres = transformar(df)
    eda(df, est)
    figura_impureza()
    figuras_bagging_boosting()
    modelos, probas, preds, res, (X_tr, y_tr, X_te, y_te) = entrenar(est, features, nombres)
    figuras_arbol(X_tr, y_tr, features, nombres)
    oob = figuras_rf_xgb(modelos, X_tr, y_tr, X_te, y_te)
    figuras_evaluacion(modelos, probas, preds, res, y_te, features, nombres)

    # Datos numéricos que cita el informe
    pa = est.groupby("ANIO")["GRAVE"].mean()
    vd = est.groupby("ANIO")["VIVIENDAS DESTRUIDAS"].apply(lambda s: (s > 0).mean())
    pel = est.groupby("PELIGRO")["GRAVE"].agg(["size", "mean"])
    raw_est = df[df["ANIO"] >= ANIO_INICIO]["PELIGRO_RAW"]
    extra = {
        "n_total": len(df), "n_estudio": len(est), "pct_grave": f"{est['GRAVE'].mean() * 100:.1f} %",
        "n_features": len(features), "n_2021": int((est["ANIO"] == 2021).sum()),
        "n_2025": int((est["ANIO"] == 2025).sum()), "n_peligros": est["PELIGRO"].nunique(),
        "n_peligros_orig": raw_est.nunique(),
        "pct_min_21_25": f"{pa.loc[2021:2025].min() * 100:.1f} %", "pct_max_21_25": f"{pa.loc[2021:2025].max() * 100:.1f} %",
        "pct_2026": f"{pa.loc[2026] * 100:.1f} %", "vd_2025": f"{vd.loc[2025] * 100:.1f} %",
        "vd_2026": f"{vd.loc[2026] * 100:.1f} %", "n_lluvia": int(pel.loc["LLUVIA INTENSA", "size"]),
        "pct_incendio": f"{pel.loc['INCENDIO URBANO', 'mean'] * 100:.0f} %",
        "oob_1": f"{oob[1]:.3f}", "oob_500": f"{oob[500]:.3f}", "oob_100": f"{oob[100]:.3f}",
        "n_distritos": est["UBIGEO"].nunique(),
    }
    pd.DataFrame({"clave": list(extra), "valor": list(extra.values())}).to_csv(
        os.path.join(RES, "datos_informe.csv"), index=False, encoding="utf-8-sig")

    # Resumen para el informe
    resumen = os.path.join(RES, "resumen.txt")
    with open(resumen, "w", encoding="utf-8") as f:
        f.write(f"Registros 2003-2026: {len(df):,}\n")
        f.write(f"Registros periodo de estudio 2021-2026: {len(est):,}\n")
        f.write(f"% graves (2021-2026): {est['GRAVE'].mean():.2%}\n")
        f.write(f"Departamentos: {est['DEPARTAMENTO'].nunique()} | Distritos: {est['UBIGEO'].nunique()} | "
                f"Peligros: {est['PELIGRO'].nunique()}\n\n")
        f.write(res.round(4).to_string(index=False))
    print("\n" + res.round(4).to_string(index=False))
    print(f"\nListo en {time.time() - t0:.0f} s. Figuras en 'figuras/', tablas en 'resultados/'.")


if __name__ == "__main__":
    main()
