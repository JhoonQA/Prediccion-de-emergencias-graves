
import json
import os

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score, log_loss, precision_score, recall_score,
                             roc_auc_score, roc_curve)
from sklearn.tree import DecisionTreeClassifier, plot_tree

# ---------------------------------------------------------------- DATOS DEL EQUIPO (editar aquí)
AUTORES = "Quispe Iquise Jose Armando  -  Jhon Einer Quispe Apaza"
CURSO = "Estructuras de Datos Avanzadas"
UNIVERSIDAD = ""
DOCENTE = ""

BASE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(BASE, "resultados")
FIG = os.path.join(BASE, "figuras")

st.set_page_config(page_title="Emergencias graves en el Perú", layout="wide", initial_sidebar_state="expanded")

# ---------------------------------------------------------------- IDENTIDAD VISUAL
AZUL, AZUL_CLARO, GRAVE, GRIS, VERDE, AMBAR = "#1F3A5F", "#DCE6F1", "#B23A48", "#9AA8B6", "#2A7F62", "#C9932E"
TINTA, TINTA_SUAVE, BORDE, FONDO = "#1F2937", "#5B6675", "#E3E8EF", "#F4F6F9"
COL = {"Árbol de Decisión": "#8FA6BF", "Random Forest": AZUL, "XGBoost": "#C8553D", "LightGBM": AMBAR}
ORDEN = list(COL)
MESES = ["Ene", "Feb", "Mar", "Abr", "May", "Jun", "Jul", "Ago", "Set", "Oct", "Nov", "Dic"]
FUENTE = "IBM Plex Sans, Segoe UI, sans-serif"

pio.templates["proyecto"] = go.layout.Template(layout=dict(
    font=dict(family=FUENTE, size=13, color=TINTA),
    colorway=[AZUL, "#C8553D", AMBAR, "#8FA6BF", VERDE, GRAVE],
    paper_bgcolor="white", plot_bgcolor="white",
    xaxis=dict(showgrid=False, linecolor="#C9D2DD", ticks="outside", tickcolor="#C9D2DD", title_font=dict(size=12)),
    yaxis=dict(gridcolor="#EEF2F6", zeroline=False, linecolor="#C9D2DD", title_font=dict(size=12)),
    legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0, title_text="", font=dict(size=12)),
    margin=dict(l=10, r=10, t=40, b=10), hoverlabel=dict(font_family=FUENTE, bgcolor="white"),
    coloraxis=dict(colorbar=dict(thickness=12, outlinewidth=0)),
))
pio.templates.default = "plotly_white+proyecto"
ESC_AZUL = ["#EEF3F8", "#A9C0D9", "#5A7FA8", AZUL]
ESC_ROJO = ["#FBF0F1", "#E7AFB6", "#CB6A77", GRAVE]
ESC_F1 = [GRAVE, "#F1E3C8", VERDE]

if "pres" not in st.session_state:
    st.session_state.pres = False
TAM = 1.12 if st.session_state.pres else 1.0

st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@400;500;600;700&display=swap');
.stApp, .stApp p, .stApp li, .stApp label, .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp button,
.stApp input, .stApp table, .stApp td, .stApp th, .stApp div[data-testid="stMarkdownContainer"] {{
  font-family: 'IBM Plex Sans', 'Segoe UI', sans-serif !important; }}
.block-container {{ padding-top: 1.6rem; max-width: 1280px; }}
.stApp p, .stApp li {{ font-size: {1.0 * TAM}rem; line-height: 1.6; color: {TINTA}; }}
h1 {{ font-weight: 700 !important; color: {AZUL} !important; letter-spacing: -0.01em; font-size: {2.0 * TAM}rem !important; }}
h2, h3 {{ font-weight: 600 !important; color: {TINTA} !important; }}
h3 {{ font-size: {1.25 * TAM}rem !important; margin-top: 1.2rem !important; }}
section[data-testid="stSidebar"] {{ background: {FONDO}; border-right: 1px solid {BORDE}; }}
section[data-testid="stSidebar"] .stButton button {{ justify-content: flex-start; text-align: left; padding: 4px 10px;
  min-height: 0; font-size: 0.92rem; border-radius: 6px; }}
section[data-testid="stSidebar"] .stButton button p {{ font-size: 0.92rem; text-align: left; }}
section[data-testid="stSidebar"] .stButton button > div {{ justify-content: flex-start; width: 100%; }}
section[data-testid="stSidebar"] .stButton button[kind="tertiary"]:hover {{ background: {AZUL_CLARO}; }}
.stApp button[kind="primary"] p, .stApp button[kind="primary"] {{ color: white !important; }}
.kpi .v.txt {{ font-size: {1.2 * TAM}rem; word-break: keep-all; }}
.marca {{ font-weight: 700; color: {AZUL}; font-size: 1.15rem; line-height: 1.25; }}
.marca-sub {{ color: {TINTA_SUAVE}; font-size: 0.82rem; margin-bottom: 0.6rem; }}
.nav-bloque {{ font-size: 0.72rem; font-weight: 600; letter-spacing: 0.08em; text-transform: uppercase;
  color: {TINTA_SUAVE}; margin: 0.9rem 0 0.15rem 0.2rem; }}
.miga {{ font-size: 0.78rem; font-weight: 600; letter-spacing: 0.06em; text-transform: uppercase; color: {TINTA_SUAVE}; }}
.idea {{ border-left: 4px solid {AZUL}; background: {FONDO}; padding: 12px 18px; border-radius: 0 8px 8px 0;
  margin: 6px 0 18px 0; font-size: {1.05 * TAM}rem; color: {TINTA}; }}
.idea b {{ color: {AZUL}; }}
.kpi {{ border: 1px solid {BORDE}; border-radius: 10px; padding: 16px 18px; background: white; height: 100%; }}
.kpi .v {{ font-size: {1.9 * TAM}rem; font-weight: 700; color: {AZUL}; line-height: 1.1; }}
.kpi .e {{ font-size: 0.85rem; font-weight: 600; color: {TINTA}; margin-top: 6px; }}
.kpi .n {{ font-size: 0.8rem; color: {TINTA_SUAVE}; margin-top: 2px; }}
.kpi.grave .v {{ color: {GRAVE}; }}
.kpi.win {{ border: 2px solid {AZUL}; background: #F7F9FC; }}
.lectura {{ border: 1px solid {BORDE}; border-radius: 10px; padding: 14px 16px; background: white; font-size: 0.92rem; }}
.lectura h5 {{ font-size: 0.72rem; letter-spacing: 0.08em; text-transform: uppercase; color: {TINTA_SUAVE};
  margin: 0 0 4px 0; font-weight: 600; }}
.lectura p {{ margin: 0 0 12px 0; font-size: {0.92 * TAM}rem !important; color: {TINTA}; }}
.lectura .conc {{ background: {AZUL_CLARO}; border-radius: 8px; padding: 10px 12px; margin: 0; }}
.nota {{ border: 1px solid {BORDE}; border-radius: 10px; padding: 14px 16px; background: white; margin: 6px 0 14px 0; }}
.nota.alerta {{ border-left: 4px solid {GRAVE}; }}
.nota.ok {{ border-left: 4px solid {VERDE}; }}
.nota.info {{ border-left: 4px solid {AZUL}; }}
.nota .t {{ font-weight: 600; margin-bottom: 4px; color: {TINTA}; }}
.hero {{ background: {AZUL}; border-radius: 14px; padding: 34px 38px; color: white; margin-bottom: 22px; }}
.hero .k {{ font-size: 0.78rem; letter-spacing: 0.1em; text-transform: uppercase; opacity: 0.75; }}
.hero .t {{ font-size: {2.2 * TAM}rem; font-weight: 700; line-height: 1.2; margin: 8px 0 10px 0; }}
.hero .s {{ font-size: {1.05 * TAM}rem; opacity: 0.9; max-width: 820px; }}
.hero .a {{ font-size: 0.88rem; opacity: 0.8; margin-top: 18px; border-top: 1px solid rgba(255,255,255,0.25); padding-top: 12px; }}
.paso {{ border: 1px solid {BORDE}; border-top: 4px solid {AZUL}; border-radius: 10px; padding: 14px; background: white; height: 100%; }}
.paso .num {{ font-size: 0.75rem; font-weight: 700; color: {TINTA_SUAVE}; letter-spacing: 0.06em; }}
.paso .nom {{ font-weight: 600; font-size: 1.0rem; color: {AZUL}; margin: 4px 0 6px 0; }}
.paso .txt {{ font-size: 0.86rem; color: {TINTA}; line-height: 1.45; }}
.chip {{ display: inline-block; padding: 2px 10px; border-radius: 999px; font-size: 0.8rem; font-weight: 600; }}
.chip.grave {{ background: #F6E1E4; color: {GRAVE}; }}
.chip.nograve {{ background: #E4EFEA; color: {VERDE}; }}
.slide {{ border: 1px solid {BORDE}; border-radius: 14px; background: white; padding: 34px 40px 28px 40px;
  aspect-ratio: 16 / 9; display: flex; flex-direction: column; gap: 18px; box-shadow: 0 1px 2px rgba(16,24,40,0.04); }}
.slide-top {{ display: flex; justify-content: space-between; align-items: flex-end; border-bottom: 3px solid {AZUL};
  padding-bottom: 14px; gap: 24px; }}
.slide-k {{ font-size: 0.78rem; letter-spacing: 0.1em; text-transform: uppercase; color: {TINTA_SUAVE}; font-weight: 600; }}
.slide-t {{ font-size: {1.9 * TAM}rem; font-weight: 700; color: {AZUL}; line-height: 1.2; margin-top: 6px; }}
.slide-autores {{ font-size: 0.82rem; color: {TINTA_SUAVE}; text-align: right; white-space: nowrap; }}
.slide-cols {{ display: grid; grid-template-columns: 1.15fr 1fr 1.1fr; gap: 28px; flex: 1; }}
.slide-col {{ border-top: 1px solid {BORDE}; padding-top: 12px; }}
.slide-n {{ font-size: 0.75rem; font-weight: 700; letter-spacing: 0.06em; text-transform: uppercase; color: {GRAVE}; }}
.slide-h {{ font-size: {1.2 * TAM}rem; font-weight: 600; color: {TINTA}; margin: 4px 0 8px 0; }}
.slide-col p, .slide-col li {{ font-size: {0.95 * TAM}rem !important; line-height: 1.5; margin-bottom: 8px; }}
.slide-col ul {{ padding-left: 18px; margin: 0; }}
.slide-pie {{ font-size: {0.8 * TAM}rem !important; color: {TINTA_SUAVE} !important; }}
.slide-q {{ background: {AZUL}; color: white; border-radius: 10px; padding: 14px 20px; font-size: {1.08 * TAM}rem;
  font-weight: 500; line-height: 1.45; }}
.slide-q span {{ display: block; font-size: 0.72rem; letter-spacing: 0.1em; text-transform: uppercase; opacity: 0.75;
  margin-bottom: 4px; font-weight: 600; }}
.paso-arbol {{ font-family: 'IBM Plex Mono', Consolas, monospace; font-size: 0.86rem; padding: 6px 10px; border-left: 2px solid {BORDE};
  margin-left: 6px; color: {TINTA}; }}
div[data-testid="stProgress"] > div > div > div > div {{ background-color: {AZUL}; }}
</style>""", unsafe_allow_html=True)


# ---------------------------------------------------------------- COMPONENTES
def n(x):
    return f"{int(round(x)):,}".replace(",", " ")


def idea(texto):
    st.markdown(f'<div class="idea"><b>Idea clave.</b> {texto}</div>', unsafe_allow_html=True)


def kpi(col, valor, etiqueta, nota_txt="", grave=False, destacar=False):
    es_texto = any(ch.isalpha() for ch in str(valor))
    col.markdown(f'<div class="kpi{" grave" if grave else ""}{" win" if destacar else ""}"><div class="v{" txt" if es_texto else ""}">{valor}</div>'
                 f'<div class="e">{etiqueta}</div><div class="n">{nota_txt}</div></div>', unsafe_allow_html=True)


def nota(tipo, titulo, texto):
    st.markdown(f'<div class="nota {tipo}"><div class="t">{titulo}</div>{texto}</div>', unsafe_allow_html=True)


def lectura(col, que, como, conclusion):
    col.markdown(f'<div class="lectura"><h5>Qué muestra</h5><p>{que}</p><h5>Cómo leerlo</h5><p>{como}</p>'
                 f'<h5>Conclusión</h5><p class="conc">{conclusion}</p></div>', unsafe_allow_html=True)


def grafico(fig, que, como, conclusion, alto=None, proporcion=(2.3, 1)):
    fig.update_layout(legend_title_text="")
    if alto:
        fig.update_layout(height=alto)
    izq, der = st.columns(proporcion, gap="large")
    izq.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    lectura(der, que, como, conclusion)


# ---------------------------------------------------------------- DATOS
@st.cache_data
def cargar():
    est = pd.read_pickle(os.path.join(RES, "datos_estudio.pkl"))
    est = est.loc[:, ~est.columns.duplicated()]
    anual = pd.read_csv(os.path.join(RES, "anual_2003_2026.csv"))
    met = pd.read_csv(os.path.join(RES, "metricas_modelos.csv")).set_index("Modelo")
    met26 = pd.read_csv(os.path.join(RES, "metricas_2026.csv")).set_index("Modelo")
    hip = pd.read_csv(os.path.join(RES, "hiperparametros.csv")).set_index("Modelo")
    mapa = pd.read_csv(os.path.join(RES, "mapa_peligros.csv"))
    bt = pd.read_csv(os.path.join(RES, "backtest_anual.csv"))
    return est, anual, met, met26, hip, mapa, bt


@st.cache_resource
def cargar_modelos():
    return joblib.load(os.path.join(RES, "modelos_entrenados.joblib"))


est, anual, met, met26, hip, mapa_pel, bt = cargar()
paq = cargar_modelos()
MOD, UMB, FEAT, NOM = paq["modelos"], paq["umbrales"], paq["features"], paq["nombres"]
tr, te, t26 = est[est["ANIO"] <= 2024], est[est["ANIO"] == 2025], est[est["ANIO"] == 2026]


@st.cache_data
def probabilidades():
    return {m: MOD[m].predict_proba(te[FEAT])[:, 1] for m in ORDEN}


PROBA = probabilidades()
RF_F1, RF_AUC, RF_REC = met.loc["Random Forest", "F1"], met.loc["Random Forest", "AUC-ROC"], met.loc["Random Forest", "Recall"]
DT_F1, DT_REC = met.loc["Árbol de Decisión", "F1"], met.loc["Árbol de Decisión", "Recall"]


# PÁGINAS
def p_presentacion():
    an = anual.sort_values("ANIO")
    ancho, alto, x0, y0 = 460, 150, 30, 12
    mx = an["total"].max()
    paso = (ancho - x0 - 10) / len(an)
    barras = []
    for k, r in enumerate(an.itertuples()):
        h = r.total / mx * (alto - 40)
        hg = r.graves / mx * (alto - 40)
        x = x0 + k * paso
        color = "#8FA6BF" if r.ANIO >= 2021 else "#D5DCE4"
        barras.append(f'<rect x="{x:.1f}" y="{alto - 22 - h:.1f}" width="{paso * 0.72:.1f}" height="{h:.1f}" fill="{color}"/>')
        barras.append(f'<rect x="{x + paso * 0.18:.1f}" y="{alto - 22 - hg:.1f}" width="{paso * 0.36:.1f}" height="{hg:.1f}" '
                      f'fill="{GRAVE}"/>')
        if r.ANIO in (2003, 2010, 2015, 2021, 2025):
            barras.append(f'<text x="{x + paso * 0.36:.1f}" y="{alto - 6}" font-size="10" text-anchor="middle" '
                          f'fill="{TINTA_SUAVE}">{r.ANIO}</text>')
    i25 = int(np.where(an["ANIO"].values == 2025)[0][0])
    x25 = x0 + i25 * paso + paso * 0.36
    y25 = alto - 22 - an["total"].values[i25] / mx * (alto - 40)
    svg = (f'<svg viewBox="0 0 {ancho} {alto}" width="100%" role="img" aria-label="Emergencias por año">'
           f'<g font-family="IBM Plex Sans, sans-serif">{"".join(barras)}'
           f'<text x="{x25 - 8:.1f}" y="{y25 - 4:.1f}" font-size="11" font-weight="600" text-anchor="end" '
           f'fill="{AZUL}">16 401 en 2025</text></g></svg>')
    st.markdown(f"""
<div class="slide">
  <div class="slide-top">
    <div>
      <div class="slide-k">Proyecto final · {CURSO}</div>
      <div class="slide-t">Anticipar emergencias graves en el Perú con árboles de decisión</div>
    </div>
    <div class="slide-autores">{AUTORES}<br>{UNIVERSIDAD}</div>
  </div>
  <div class="slide-cols">
    <div class="slide-col">
      <div class="slide-n">01 · El problema</div>
      <div class="slide-h">Se decide antes de saber</div>
      <p>Cada emergencia llega al SINPAD sin saber cuánto daño dejará. La evaluación tarda <b>días</b>, pero la ayuda
      humanitaria se asigna en las <b>primeras horas</b> y no alcanza para todas.</p>
      {svg}
      <p class="slide-pie">Emergencias registradas por año; en rojo, las graves.</p>
    </div>
    <div class="slide-col">
      <div class="slide-n">02 · Por qué este proyecto</div>
      <div class="slide-h">Priorizar con datos</div>
      <p>Un modelo que estime, al momento del reporte, qué emergencias serán <b>graves</b> ayudaría a decidir a dónde
      enviar primero la ayuda.</p>
      <p>Además permite aplicar lo visto en el curso: el <b>árbol de decisión es un árbol binario</b>, y Random Forest y
      XGBoost son colecciones de árboles.</p>
    </div>
    <div class="slide-col">
      <div class="slide-n">03 · Por qué esta base de datos</div>
      <div class="slide-h">Oficial, reciente y completa</div>
      <ul>
        <li><b>Fuente oficial:</b> INDECI, en la Plataforma Nacional de Datos Abiertos.</li>
        <li><b>Volumen:</b> 151 491 emergencias, de 2003 a 2026.</li>
        <li><b>Actualizada:</b> octubre de 2026, con detalle por distrito.</li>
        <li><b>Registra los daños</b>, lo que permite definir qué es una emergencia grave.</li>
      </ul>
      <p class="slide-pie">Otras opciones evaluadas: el catálogo sísmico del IGP no tenía un problema de clasificación
      claro y los registros de dengue tenían pocas variables.</p>
    </div>
  </div>
  <div class="slide-q"><span>Pregunta de investigación</span>Con lo que se sabe al momento del reporte, ¿podemos anticipar si
  una emergencia dejará fallecidos, damnificados o viviendas destruidas?</div>
</div>""", unsafe_allow_html=True)


def p_inicio():
    st.markdown(f"""<div class="hero"><div class="k">Proyecto final · {CURSO}</div>
<div class="t">¿Podemos anticipar qué emergencias serán graves?</div>
<div class="s">Predicción de emergencias con fallecidos, damnificados o viviendas destruidas a partir de los registros
del SINPAD–INDECI (2021–2026), comparando Random Forest y XGBoost con un Árbol de Decisión como línea base.</div>
<div class="a">{AUTORES} &nbsp;·&nbsp; {UNIVERSIDAD} &nbsp;·&nbsp; {DOCENTE}</div></div>""", unsafe_allow_html=True)
    c = st.columns(4, gap="medium")
    kpi(c[0], n(len(est)), "Emergencias analizadas", "2021 a 2026 (I semestre)")
    kpi(c[1], f"{est['GRAVE'].mean():.1%}", "Fueron graves", "Fallecidos, damnificados o viviendas destruidas", grave=True)
    kpi(c[2], "Random Forest", "Mejor modelo", f"F1 = {RF_F1:.3f} · AUC = {RF_AUC:.3f}")
    kpi(c[3], f"{RF_REC:.0%}", "Graves detectadas en 2025", f"Frente a {DT_REC:.0%} con un solo árbol")
    st.markdown("### Cómo se desarrolló el proyecto")
    st.caption("Metodología KDD (Knowledge Discovery in Databases), en cinco fases.")
    pasos = [("01", "Selección", "151 491 emergencias oficiales del INDECI, 2003–2026."),
             ("02", "Preprocesamiento", "Fechas en dos formatos, 58 nombres de peligro unificados en 19."),
             ("03", "Transformación", "Clase GRAVE, 19 variables sin fuga de datos y memoria histórica."),
             ("04", "Minería de datos", "Árbol de Decisión, Random Forest, XGBoost y LightGBM."),
             ("05", "Evaluación", "Entrenar 2021–2024, probar en 2025 y validar año por año.")]
    for col, (num, nom, txt) in zip(st.columns(5, gap="small"), pasos):
        col.markdown(f'<div class="paso"><div class="num">FASE {num}</div><div class="nom">{nom}</div>'
                     f'<div class="txt">{txt}</div></div>', unsafe_allow_html=True)
    st.markdown("### Resultado principal")
    nota("ok", f"Random Forest detecta el {RF_REC:.0%} de las emergencias graves de un año que nunca vio",
         f"En la prueba con las {n(len(te))} emergencias de 2025 obtuvo F1 = {RF_F1:.3f} y AUC = {RF_AUC:.3f}, por encima "
         f"del árbol individual (F1 = {DT_F1:.3f}) y de XGBoost (F1 = {met.loc['XGBoost', 'F1']:.3f}). Use los botones "
         "<b>Siguiente</b> al final de cada página o el menú de la izquierda para recorrer el proyecto.")


def p_problema():
    idea("Las decisiones de respuesta se toman en las primeras horas, pero la evaluación de daños llega días después.")
    a, b = st.columns([1.25, 1], gap="large")
    with a:
        st.markdown("""
**Contexto.** El Perú está en el Cinturón de Fuego del Pacífico, tiene la cordillera de los Andes y recibe el impacto
del Fenómeno El Niño. Cada año hay lluvias intensas, huaycos, inundaciones, heladas, friajes e incendios.

**Quién atiende.** El **INDECI** (Instituto Nacional de Defensa Civil) registra cada emergencia en el **SINPAD**, el
Sistema de Información Nacional para la Respuesta y Rehabilitación.

**El problema.** La evaluación de daños llega **días después**, pero el envío de ayuda humanitaria, carpas o equipos
de rescate se decide **en las primeras horas**. Y los recursos no alcanzan para todas las emergencias.
""")
        nota("info", "Pregunta de investigación",
             "Con lo que se sabe al momento del reporte (qué peligro es, dónde, cuándo y qué historial tiene el "
             "distrito), <b>¿será una emergencia grave?</b>")
    with b:
        st.markdown("#### Lo que el modelo conoce y lo que no")
        c1, c2 = st.columns(2)
        c1.markdown(f'<div class="nota ok"><div class="t">Sí conoce</div>Tipo de peligro<br>Ubicación<br>Mes y '
                    f'temporada<br>Historial del distrito<br>Emergencias de esa semana</div>', unsafe_allow_html=True)
        c2.markdown(f'<div class="nota alerta"><div class="t">No conoce</div>Fallecidos<br>Damnificados<br>Viviendas '
                    f'destruidas<br>Evaluación de daños<br>(es el futuro)</div>', unsafe_allow_html=True)
    st.markdown("### Por qué árboles de decisión: la relación con Estructuras de Datos")
    a, b = st.columns([1, 1.15], gap="large")
    a.markdown("""
Un **árbol de decisión es un árbol binario**:

- cada **nodo interno** guarda una pregunta, por ejemplo *¿el peligro es un incendio?*;
- cada **rama** es una respuesta (sí o no);
- cada **hoja** guarda la decisión final: *grave* o *no grave*.

Clasificar una emergencia es **recorrer el árbol desde la raíz hasta una hoja**, igual que una búsqueda en un árbol
binario, con un costo proporcional a la profundidad. Random Forest y XGBoost son **colecciones de muchos árboles**.
""")
    b.markdown(f"""<svg viewBox="0 0 520 250" width="100%" role="img" aria-label="Árbol de decisión de ejemplo">
<g font-family="IBM Plex Sans, sans-serif" font-size="13" text-anchor="middle">
<line x1="260" y1="58" x2="140" y2="120" stroke="#9AA8B6"/><line x1="260" y1="58" x2="380" y2="120" stroke="#9AA8B6"/>
<line x1="140" y1="150" x2="80" y2="198" stroke="#9AA8B6"/><line x1="140" y1="150" x2="200" y2="198" stroke="#9AA8B6"/>
<rect x="160" y="18" width="200" height="40" rx="8" fill="{AZUL}"/><text x="260" y="43" fill="white" font-weight="600">¿Historial grave alto?</text>
<text x="185" y="92" fill="{TINTA_SUAVE}" font-size="12">no</text><text x="335" y="92" fill="{TINTA_SUAVE}" font-size="12">sí</text>
<rect x="55" y="120" width="170" height="34" rx="8" fill="white" stroke="{AZUL}"/><text x="140" y="142" fill="{TINTA}">¿Es un incendio?</text>
<rect x="310" y="120" width="140" height="34" rx="8" fill="#F6E1E4" stroke="{GRAVE}"/><text x="380" y="142" fill="{GRAVE}" font-weight="600">GRAVE</text>
<rect x="20" y="198" width="120" height="34" rx="8" fill="#E4EFEA" stroke="{VERDE}"/><text x="80" y="220" fill="{VERDE}" font-weight="600">NO GRAVE</text>
<rect x="150" y="198" width="100" height="34" rx="8" fill="#F6E1E4" stroke="{GRAVE}"/><text x="200" y="220" fill="{GRAVE}" font-weight="600">GRAVE</text>
<text x="470" y="42" fill="{TINTA_SUAVE}" font-size="11" text-anchor="start">raíz</text>
<text x="458" y="142" fill="{TINTA_SUAVE}" font-size="11" text-anchor="start">hoja</text>
<text x="60" y="142" fill="{TINTA_SUAVE}" font-size="11" text-anchor="end">nodo</text>
</g></svg>""", unsafe_allow_html=True)
    b.caption("Ejemplo simplificado. El árbol real elige las preguntas a partir de los datos.")


def p_datos():
    idea("Los datos son oficiales y recientes: el registro completo de emergencias del INDECI publicado como datos abiertos.")
    c = st.columns(4, gap="medium")
    kpi(c[0], "151 491", "Registros totales", "Emergencias de 2003 a 2026")
    kpi(c[1], "49", "Columnas", "Ubicación, peligro y 38 de daños")
    kpi(c[2], n(len(est)), "Periodo de estudio", "2021 a 2026 (I semestre)")
    kpi(c[3], n(est["UBIGEO"].nunique()), "Distritos", "De los 25 departamentos")
    st.markdown("")
    st.markdown("**Fuente:** Plataforma Nacional de Datos Abiertos del Perú, dataset *Emergencias y daños a nivel "
                "nacional por departamento* (INDECI). Última actualización: 30 de octubre de 2026.")
    st.markdown("### Emergencias registradas por año")
    an = anual.copy()
    fig = go.Figure()
    fig.add_bar(x=an["ANIO"], y=an["total"], name="Total de emergencias",
                marker_color=np.where(an["ANIO"] >= 2021, "#8FA6BF", "#D5DCE4"))
    fig.add_bar(x=an["ANIO"], y=an["graves"], name="Emergencias graves", marker_color=GRAVE, width=0.45)
    fig.update_layout(barmode="overlay", xaxis_title="Año (2026: solo I semestre)", yaxis_title="N° de emergencias")
    fig.add_vline(x=2020.5, line_dash="dot", line_color=TINTA_SUAVE)
    fig.add_annotation(x=2011.5, y=15500, text="Memoria histórica (2003–2020)", showarrow=False, font_color=TINTA_SUAVE)
    fig.add_annotation(x=2023.5, y=17800, text="Periodo de estudio", showarrow=False, font_color=AZUL)
    grafico(fig, "Cuántas emergencias registró el SINPAD cada año (barra ancha) y cuántas fueron graves (barra roja).",
            "Los años grises (2003–2020) no se usan para entrenar: sirven para calcular el historial de cada distrito. "
            "Los años en azul son el periodo de estudio.",
            "Las emergencias registradas se duplicaron desde 2021 (de 7 799 a 16 401 en 2025) por las lluvias de El "
            "Niño Costero y el ciclón Yaku. Hay muchos datos recientes para entrenar.", alto=400)
    st.markdown("### Qué contiene cada registro")
    a, b = st.columns([1.2, 1], gap="large")
    a.dataframe(pd.DataFrame({
        "Grupo de columnas": ["Identificación y fecha", "Ubicación", "Peligro", "Daños a personas", "Daños materiales"],
        "Ejemplos": ["Código SINPAD, fecha, año, mes", "UBIGEO, departamento, distrito, región natural",
                     "Peligro, origen (natural o humano)", "Fallecidos, desaparecidos, damnificados, afectados",
                     "Viviendas, colegios, puentes, carreteras, cultivos"],
        "Uso en el proyecto": ["Ordenar en el tiempo", "Predecir", "Predecir", "Definir la clase GRAVE", "Solo análisis"]}),
        hide_index=True, width="stretch")
    with b:
        st.markdown("**Cinco emergencias reales del dataset**")
        st.dataframe(est[["FECHA", "DISTRITO", "PELIGRO", "DAMNIFICADOS", "VIVIENDAS DESTRUIDAS", "GRAVE"]]
                     .sample(5, random_state=3), hide_index=True, width="stretch")


def p_limpieza():
    idea("Antes de entrenar se revisó la calidad de los datos y se corrigieron cuatro problemas que habrían distorsionado los resultados.")
    probs = [
        ("Fechas en dos formatos", "Hasta 2024 el formato es DD/MM/AAAA (20/01/2020). Desde 2025 es MM/DD/AA (05/19/25).",
         "Se lee con el primer formato y, si falla, con el segundo.", "100 % de fechas válidas. Sin la corrección se "
         "perdían todas las de 2025, el año de prueba."),
        ("Un peligro con varios nombres", "El catálogo del SINPAD cambió: «LLUVIA INTENSA» y «LLUVIAS INTENSAS», "
         "«HUAICO» y «HUAYCO».", "Normalización del texto y una tabla de equivalencias.", "58 nombres reducidos a 19 categorías."),
        ("Columnas con nombres distintos", "El archivo 2003–2025 dice «DPTO.» y el de 2026 «DEP».",
         "Alineación por posición según el diccionario de datos oficial.", "Un solo dataset con 49 columnas."),
        ("Celdas vacías en daños", "525 476 celdas vacías, casi todas en campos de aulas añadidos recientemente.",
         "Una celda vacía significa que no se registró daño: se reemplazó por 0.", "Sin duplicados ni valores negativos."),
    ]
    for fila in (probs[:2], probs[2:]):
        cols = st.columns(2, gap="large")
        for col, (t, p, s, r) in zip(cols, fila):
            with col.container(border=True):
                st.markdown(f"#### {t}")
                st.markdown(f"**Problema.** {p}\n\n**Solución.** {s}\n\n**Resultado.** {r}")
    with st.expander("Ver la tabla de equivalencias de peligros"):
        m = mapa_pel[mapa_pel["PELIGRO_RAW"].str.upper().str.strip() != mapa_pel["PELIGRO"]]
        st.dataframe(m.rename(columns={"PELIGRO_RAW": "Nombre original", "PELIGRO": "Nombre unificado"}),
                     hide_index=True, width="stretch", height=320)
    nota("info", "Por qué importa", "Un modelo es tan bueno como sus datos. El problema de las fechas, por ejemplo, "
         "habría dejado en cero las variables de intensidad del evento en todo 2025 y empeorado los resultados sin que "
         "nadie lo notara.")


def p_objetivo():
    idea("El modelo solo ve lo que se sabe en el momento del reporte; los daños son la respuesta, no una pista.")
    st.markdown("### Qué es una emergencia grave")
    a, b = st.columns([1.4, 1], gap="large")
    with a:
        st.latex(r"\text{GRAVE}=1 \iff (\text{fallecidos}+\text{desaparecidos}>0)\ \lor\ (\text{damnificados}>0)"
                 r"\ \lor\ (\text{viviendas destruidas}>0)")
        st.markdown("Son los tres criterios con los que el INDECI activa **ayuda humanitaria y albergue**. Un "
                    "**damnificado** perdió su vivienda o sus medios de vida; un **afectado** solo sufrió daños "
                    "parciales, y eso no cuenta como grave.")
    vc = est["GRAVE"].value_counts().rename({0: "No grave", 1: "Grave"})
    fig = go.Figure(go.Pie(values=vc.values, labels=vc.index, hole=0.62, sort=False,
                           marker=dict(colors=[GRIS, GRAVE], line=dict(color="white", width=2)),
                           textinfo="percent", textfont=dict(size=14, color="white")))
    fig.update_layout(height=250, margin=dict(t=10, b=10, l=10, r=10), showlegend=True,
                      annotations=[dict(text=f"{n(len(est))}<br>emergencias", showarrow=False, font_size=13)])
    b.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    st.markdown("### La trampa: la fuga de datos")
    a, b = st.columns([1.3, 1], gap="large")
    with a:
        st.markdown("Si el modelo pudiera ver las columnas de daños, obtendría casi **100 % de acierto**. Parece "
                    "perfecto, pero esas columnas **son la respuesta** y no se conocen al momento del reporte. Pulse el "
                    "botón para comprobarlo con los datos reales.")
        boton = st.button("Entrenar un árbol con fuga de datos", type="primary")
    with b:
        if boton:
            cols = ["FALLECIDOS", "DESAPARECIDOS", "DAMNIFICADOS", "VIVIENDAS DESTRUIDAS", "AFECTADOS"]
            m = DecisionTreeClassifier(max_depth=4, random_state=0).fit(tr[cols], tr["GRAVE"])
            p = m.predict(te[cols])
            c = st.columns(2)
            kpi(c[0], f"{accuracy_score(te['GRAVE'], p):.1%}", "Exactitud con fuga", "Prueba 2025", grave=True)
            kpi(c[1], f"{f1_score(te['GRAVE'], p):.3f}", "F1 con fuga", "Prueba 2025", grave=True)
            st.markdown("")
            nota("alerta", "Resultado engañoso", "El modelo solo aprendió la definición. Por eso se eliminaron las "
                 "<b>38 columnas de daños</b> del conjunto de predictores.")
    st.markdown("### Las 19 variables que sí usa el modelo")
    fam = [("Evento", "Peligro · grupo de peligro · origen natural o humano"),
           ("Territorio", "Región natural · departamento · código UBIGEO"),
           ("Tiempo", "Mes · mes como ciclo (seno y coseno) · temporada de lluvias"),
           ("Memoria histórica", "Tasa de gravedad del distrito, la provincia y el mismo peligro · emergencias previas"),
           ("Intensidad del evento", "Emergencias en el departamento (7 días) y en el distrito (30 días)")]
    for col, (t, v) in zip(st.columns(5, gap="small"), fam):
        col.markdown(f'<div class="paso"><div class="nom">{t}</div><div class="txt">{v}</div></div>', unsafe_allow_html=True)
    st.markdown("### La memoria histórica con un ejemplo real")
    top = te["UBIGEO"].value_counts().index[0]
    ej = te[te["UBIGEO"] == top].iloc[0]
    nota("info", f"Distrito de {ej['DISTRITO'].title()} ({ej['DEPARTAMENTO'].title()}), emergencia de 2025",
         f"Antes de 2025 este distrito acumulaba <b>{n(ej['DIST_N_PREV'])}</b> emergencias registradas desde 2003. Con "
         f"ellas se calcula su tasa histórica de gravedad: <b>{ej['DIST_TASA_GRAVE']:.0%}</b>. Ese número entra al modelo "
         "como una variable más. Solo se usan años anteriores, así que el modelo nunca ve el futuro. Un distrito con "
         "historial de emergencias graves suele tener viviendas más vulnerables.")


def p_exploracion():
    idea("Los peligros más frecuentes no son los más graves, y la gravedad cambia según la región y la época del año.")
    f1c, f2c = st.columns(2)
    dep = f1c.selectbox("Departamento", ["Todo el Perú"] + sorted(est["DEPARTAMENTO"].unique()))
    anios = f2c.slider("Años", 2021, 2026, (2021, 2026))
    d = est[(est["ANIO"] >= anios[0]) & (est["ANIO"] <= anios[1])]
    if dep != "Todo el Perú":
        d = d[d["DEPARTAMENTO"] == dep]
    st.caption(f"{n(len(d))} emergencias seleccionadas · {d['GRAVE'].mean():.1%} graves")

    st.markdown("### Qué peligros ocurren más y cuáles son más graves")
    g = d.groupby("PELIGRO")["GRAVE"].agg(cant="size", pct="mean").reset_index().sort_values("cant").tail(15)
    g["pct"] *= 100
    fig = px.bar(g, x="cant", y="PELIGRO", orientation="h", color="pct", color_continuous_scale=ESC_ROJO,
                 range_color=(0, 100), labels={"cant": "N° de emergencias", "PELIGRO": "", "pct": "% graves"},
                 text=g["pct"].round(0).astype(int).astype(str) + " %")
    fig.update_traces(textposition="outside", cliponaxis=False)
    grafico(fig, "Los 15 peligros más frecuentes. El largo de la barra es la cantidad y el color, el porcentaje de graves.",
            "Una barra larga es un peligro frecuente; una barra roja oscura, un peligro que suele ser grave.",
            "Las lluvias intensas son las más frecuentes, pero los incendios urbanos (76 % graves), los colapsos, la "
            "erosión y los sismos son los más graves porque destruyen viviendas. Las heladas casi nunca lo son (8 %).", alto=520)

    st.markdown("### Dónde y cuándo ocurren")
    piv = d.pivot_table(index="DEPARTAMENTO", columns="MES_NUM", values="GRAVE", aggfunc="size", fill_value=0)
    piv = piv.reindex(columns=range(1, 13), fill_value=0)
    piv = piv.loc[piv.sum(axis=1).sort_values(ascending=False).index]
    fig = px.imshow(piv.values, x=MESES, y=[x.title() for x in piv.index], color_continuous_scale=ESC_AZUL,
                    aspect="auto", labels=dict(color="Emergencias"))
    grafico(fig, "Cuántas emergencias hubo en cada departamento (fila) y mes (columna).",
            "Más oscuro, más emergencias. Los departamentos están ordenados de más a menos emergencias.",
            "Las emergencias se concentran de enero a marzo, en la temporada de lluvias, en la sierra: Cusco, "
            "Huancavelica, Ayacucho, Cajamarca y Apurímac.", alto=640)

    st.markdown("### Qué regiones y grupos de peligro son más graves")
    a, b = st.columns(2, gap="large")
    for col, campo, titulo in [(a, "REGION", "Por región natural"), (b, "GRUPO_PELIGRO", "Por grupo de peligro")]:
        r = (d.groupby(campo)["GRAVE"].mean() * 100).sort_values().reset_index()
        r[campo] = r[campo].str.title() if campo == "REGION" else r[campo]
        fig = px.bar(r, x="GRAVE", y=campo, orientation="h", text=r["GRAVE"].round(1).astype(str) + " %",
                     labels={"GRAVE": "% de emergencias graves", campo: ""}, range_x=[0, 100],
                     color_discrete_sequence=[AZUL])
        fig.update_traces(textposition="outside")
        fig.update_layout(height=290, title=dict(text=titulo, font_size=14))
        col.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    nota("info", "Lectura", "La costa y la selva baja (cerca del 60 % de graves) casi duplican a la sierra (33 %): en la "
         "sierra predominan las heladas, que afectan pero no destruyen viviendas. Los incendios son el grupo más grave.")

    st.markdown("### Estacionalidad")
    m = d.groupby("MES_NUM")["GRAVE"].agg(cant="size", pct="mean").reindex(range(1, 13)).fillna(0)
    fig = go.Figure()
    fig.add_bar(x=MESES, y=m["cant"], name="Emergencias",
                marker_color=["#5A7FA8" if i in (12, 1, 2, 3, 4) else "#D5DCE4" for i in range(1, 13)])
    fig.add_scatter(x=MESES, y=m["pct"] * 100, name="% graves", yaxis="y2", mode="lines+markers",
                    line=dict(color=GRAVE, width=3))
    fig.update_layout(yaxis2=dict(overlaying="y", side="right", range=[0, 100], title="% graves", showgrid=False),
                      yaxis_title="N° de emergencias")
    grafico(fig, "Emergencias por mes (barras) y porcentaje de graves (línea roja).",
            "Las barras azules son la temporada de lluvias (diciembre a abril).",
            "La cantidad de emergencias se dispara con las lluvias, pero el porcentaje de graves no sube tanto: muchas "
            "lluvias afectan sin destruir. En los meses secos aparecen incendios y heladas.", alto=380)


@st.cache_data
def curva_profundidad():
    filas = []
    for prof in [1, 2, 3, 4, 5, 6, 8, 10, 12, 15, 18, 22, 26]:
        m = DecisionTreeClassifier(max_depth=prof, random_state=42).fit(tr[FEAT], tr["GRAVE"])
        filas.append({"Profundidad": prof, "Entrenamiento": f1_score(tr["GRAVE"], m.predict(tr[FEAT])),
                      "Prueba 2025": f1_score(te["GRAVE"], m.predict(te[FEAT]))})
    return pd.DataFrame(filas)


def p_arbol():
    idea("Un árbol aprende preguntas que separan las emergencias graves de las que no lo son, pero si crece demasiado, memoriza.")
    st.markdown("### Cómo elige cada pregunta: la impureza")
    a, b = st.columns([1, 1.3], gap="large")
    with a:
        st.markdown("En cada nodo, el árbol prueba todas las preguntas posibles y se queda con la que deja los grupos "
                    "más **puros** (todo grave o todo no grave). La pureza se mide con el **índice de Gini** o la "
                    "**entropía**.")
        p = st.slider("Proporción de emergencias graves dentro de un nodo", 0.0, 1.0, 0.5, 0.01)
        gini = 1 - p ** 2 - (1 - p) ** 2
        ent = 0 if p in (0, 1) else -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
        c = st.columns(3)
        kpi(c[0], f"{gini:.3f}", "Gini", "1 − Σ pᵢ²")
        kpi(c[1], f"{ent:.3f}", "Entropía", "−Σ pᵢ log₂ pᵢ")
        kpi(c[2], "Puro" if gini < 0.05 else ("Mezclado" if gini > 0.45 else "Intermedio"), "Estado del nodo", "")
    xs = np.linspace(0.001, 0.999, 200)
    fig = go.Figure()
    fig.add_scatter(x=xs, y=1 - xs ** 2 - (1 - xs) ** 2, name="Gini", line=dict(color=AZUL, width=3))
    fig.add_scatter(x=xs, y=-(xs * np.log2(xs) + (1 - xs) * np.log2(1 - xs)), name="Entropía", line=dict(color=AMBAR, width=3))
    fig.add_vline(x=p, line_dash="dot", line_color=TINTA)
    fig.update_layout(height=330, xaxis_title="Proporción de graves en el nodo", yaxis_title="Impureza")
    b.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    nota("info", "Lectura", "Con 0 % o 100 % de graves el nodo es puro (impureza 0); con 50 % está totalmente mezclado. "
         "El árbol elige la pregunta que más reduce la impureza: la <b>ganancia de información</b>.")

    st.markdown("### El árbol del proyecto")
    c1, c2 = st.columns([1, 3])
    prof = c1.slider("Profundidad máxima", 1, 20, 3)
    crit = c1.radio("Criterio", ["gini", "entropy"], format_func=lambda x: "Gini" if x == "gini" else "Entropía")

    @st.cache_resource
    def arbol(prof, crit):
        return DecisionTreeClassifier(max_depth=prof, criterion=crit, min_samples_leaf=20,
                                      random_state=42).fit(tr[FEAT], tr["GRAVE"])

    m = arbol(prof, crit)
    f_tr = f1_score(tr["GRAVE"], m.predict(tr[FEAT]))
    f_te = f1_score(te["GRAVE"], (m.predict_proba(te[FEAT])[:, 1] >= 0.35).astype(int))
    with c2:
        c = st.columns(4)
        kpi(c[0], n(m.tree_.node_count), "Nodos", "Tamaño de la estructura")
        kpi(c[1], n(m.get_n_leaves()), "Hojas", "Decisiones finales")
        kpi(c[2], f"{f_tr:.3f}", "F1 entrenamiento", "Datos 2021–2024")
        kpi(c[3], f"{f_te:.3f}", "F1 prueba", "Datos de 2025", grave=f_te < f_tr - 0.15)
    if prof <= 4:
        import matplotlib.pyplot as plt
        plt.rcParams["font.family"] = "Segoe UI"
        figm, ax = plt.subplots(figsize=(22, 3.5 + 2 * prof))
        plot_tree(m, feature_names=[NOM[f] for f in FEAT], class_names=["No grave", "Grave"], filled=True,
                  rounded=True, fontsize=10, ax=ax, impurity=True)
        st.pyplot(figm)
        st.caption("Cada nodo muestra la pregunta, el Gini, el número de emergencias (samples), su reparto "
                   "[no graves, graves] y la decisión. Naranja: mayoría no grave; azul: mayoría grave.")
    else:
        nota("info", f"Con profundidad {prof} el árbol ya tiene {n(m.tree_.node_count)} nodos",
             "Es demasiado grande para dibujarlo. Baje la profundidad a 4 o menos para verlo.")

    st.markdown("### La debilidad del árbol único: el sobreajuste")
    cp = curva_profundidad()
    fig = go.Figure()
    fig.add_scatter(x=cp["Profundidad"], y=cp["Entrenamiento"], name="Entrenamiento (2021–2024)", mode="lines+markers",
                    line=dict(color=GRIS, width=3))
    fig.add_scatter(x=cp["Profundidad"], y=cp["Prueba 2025"], name="Prueba (2025)", mode="lines+markers",
                    line=dict(color=AZUL, width=3), fill="tonexty", fillcolor="rgba(178,58,72,0.08)")
    fig.update_layout(xaxis_title="Profundidad máxima del árbol", yaxis_title="F1")
    grafico(fig, "El F1 del árbol en los datos con que aprendió (gris) y en datos nuevos de 2025 (azul), según su profundidad.",
            "La zona sombreada entre ambas líneas es la brecha: cuanto más grande, más memoriza el árbol.",
            "Al crecer, el árbol acierta casi todo lo que ya vio, pero no mejora en datos nuevos: memoriza. Esta es la "
            "razón de usar Random Forest y XGBoost.", alto=380)


@st.cache_data
def curva_bosque():
    filas = []
    for k in [1, 3, 5, 10, 25, 50, 100, 200]:
        rf = RandomForestClassifier(n_estimators=k, min_samples_leaf=3, max_features="sqrt", n_jobs=-1,
                                    random_state=42).fit(tr[FEAT], tr["GRAVE"])
        filas.append({"Árboles": k, "AUC": roc_auc_score(te["GRAVE"], rf.predict_proba(te[FEAT])[:, 1])})
    return pd.DataFrame(filas)


@st.cache_data
def curva_xgb():
    from xgboost import XGBClassifier
    m = XGBClassifier(**MOD["XGBoost"].get_params())
    m.fit(tr[FEAT], tr["GRAVE"], eval_set=[(tr[FEAT], tr["GRAVE"]), (te[FEAT], te["GRAVE"])], verbose=False)
    ev = m.evals_result()
    return pd.DataFrame({"Iteración": range(len(ev["validation_0"]["logloss"])),
                         "Entrenamiento": ev["validation_0"]["logloss"], "Prueba 2025": ev["validation_1"]["logloss"]})


def p_ensambles():
    idea("Random Forest combina muchos árboles en paralelo y vota; XGBoost los combina en secuencia, corrigiendo errores.")
    a, b = st.columns(2, gap="large")
    with a.container(border=True):
        st.markdown(f"<h4 style='color:{AZUL}'>Random Forest · bagging</h4>", unsafe_allow_html=True)
        st.markdown("- Entrena **cientos de árboles en paralelo**, independientes entre sí.\n"
                    "- Cada árbol ve una **muestra aleatoria** de los datos y solo algunas variables por nodo.\n"
                    "- La decisión final es una **votación**.\n"
                    "- **Reduce la varianza:** los errores de cada árbol se cancelan.\n\n"
                    "*Analogía: consultar a 300 expertos por separado y quedarse con la mayoría.*")
    with b.container(border=True):
        st.markdown("<h4 style='color:#C8553D'>XGBoost · boosting</h4>", unsafe_allow_html=True)
        st.markdown("- Entrena **árboles en secuencia**.\n"
                    "- Cada árbol nuevo se concentra en **corregir los errores** de los anteriores.\n"
                    "- La decisión final es la **suma** de todos los árboles.\n"
                    "- **Reduce el sesgo**, pero puede memorizar si se entrena demasiado.\n\n"
                    "*Analogía: un alumno que repasa solo los ejercicios que falló.*")
    st.image(os.path.join(FIG, "fig12_bagging_vs_boosting.png"), width="stretch")

    st.markdown("### Cuántos árboles necesita el bosque")
    cb = curva_bosque()
    k = st.select_slider("Número de árboles", cb["Árboles"].tolist(), value=10)
    fila = cb[cb["Árboles"] == k].iloc[0]
    fig = go.Figure()
    fig.add_scatter(x=cb["Árboles"], y=cb["AUC"], mode="lines+markers", line=dict(color=AZUL, width=3), name="AUC")
    fig.add_scatter(x=[k], y=[fila["AUC"]], mode="markers", marker=dict(size=16, color=GRAVE), name="Selección")
    fig.update_layout(xaxis=dict(type="log", title="Número de árboles (escala logarítmica)"), yaxis_title="AUC en 2025",
                      showlegend=False)
    grafico(fig, f"La calidad del bosque (AUC en 2025) según cuántos árboles tiene. Con {k} árboles: AUC = {fila['AUC']:.3f}.",
            "Mueva el deslizador: con 1 árbol el bosque es un solo árbol de decisión. El punto rojo es su selección.",
            "El AUC mejora rápido hasta unos 50 a 100 árboles y luego se estabiliza: la «sabiduría de la multitud».", alto=360)

    st.markdown("### XGBoost: la curva de aprendizaje")
    cx = curva_xgb()
    imin = int(cx["Prueba 2025"].idxmin())
    fig = go.Figure()
    fig.add_scatter(x=cx["Iteración"], y=cx["Entrenamiento"], name="Entrenamiento", line=dict(color=GRIS, width=3))
    fig.add_scatter(x=cx["Iteración"], y=cx["Prueba 2025"], name="Prueba 2025", line=dict(color="#C8553D", width=3))
    fig.add_vline(x=imin, line_dash="dot", line_color=TINTA)
    fig.add_annotation(x=imin, y=cx["Prueba 2025"].max(), text=f"Mínimo en prueba: árbol {imin}", showarrow=False,
                       xanchor="left", xshift=6)
    fig.update_layout(xaxis_title="Número de árboles agregados", yaxis_title="Error (log-loss)")
    grafico(fig, "Cómo cambia el error de XGBoost al agregar árboles, en entrenamiento (gris) y en 2025 (rojo).",
            "Más bajo es mejor. Si la línea gris baja mientras la roja sube, el modelo está memorizando.",
            f"El error de entrenamiento siempre baja, pero el de 2025 solo baja hasta el árbol {imin} y luego sube: XGBoost "
            "aprendió particularidades de 2021–2024 (sobreajuste temporal). Por eso Random Forest generalizó mejor.", alto=380)
    with st.expander("Ver los hiperparámetros elegidos para cada modelo"):
        h = hip.copy()
        h["Mejores hiperparámetros"] = h["Mejores hiperparámetros"].str.replace("[{}']", "", regex=True)
        st.dataframe(h, width="stretch")
        st.caption("Búsqueda aleatoria de 15 combinaciones por modelo con validación cruzada de 3 particiones.")


def p_validacion():
    idea("Se entrena con el pasado y se evalúa en un año que el modelo nunca vio, como se usaría en la realidad.")
    st.markdown("### Partición temporal")
    tl = pd.DataFrame({"Conjunto": ["Memoria histórica", "Entrenamiento", "Prueba", "Validación extra"],
                       "Inicio": ["2003-01-01", "2021-01-01", "2025-01-01", "2026-01-01"],
                       "Fin": ["2020-12-31", "2024-12-31", "2025-12-31", "2026-06-30"],
                       "Detalle": ["Historial de cada distrito", "41 105 emergencias", "16 401 emergencias",
                                   "9 352 emergencias"]})
    fig = px.timeline(tl, x_start="Inicio", x_end="Fin", y="Conjunto", color="Conjunto",
                      color_discrete_sequence=["#D5DCE4", AZUL, GRAVE, AMBAR])
    fig.update_traces(text=tl["Detalle"], textposition="inside", insidetextanchor="middle")
    fig.update_yaxes(autorange="reversed", title="")
    fig.update_layout(height=250, showlegend=False, margin=dict(t=10))
    st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    a, b = st.columns(2, gap="large")
    with a:
        nota("alerta", "Partición aleatoria (la habitual)", "Mezcla emergencias de todos los años: el modelo podría ver "
             "emergencias de 2025 al entrenar. Es como dar el examen conociendo las respuestas.")
    with b:
        nota("ok", "Partición temporal (la de este proyecto)", "Entrena solo con 2021–2024 y se evalúa en 2025. Además, "
             "se repitió el experimento para 2023, 2024 y 2026 (ver sección Mapa y rendimiento).")
    nota("info", "Por qué 2026 se evalúa aparte", "En 2026 el porcentaje de graves saltó de 38 % a 69 % porque las "
         "emergencias con «viviendas destruidas» pasaron de 37 % a 63 %, mientras que las de damnificados se mantuvieron. "
         "Es un cambio en la forma de registrar, no un año más grave.")

    st.markdown("### Las métricas en palabras simples")
    st.dataframe(pd.DataFrame({
        "Métrica": ["Accuracy (exactitud)", "Precisión", "Recall (sensibilidad)", "F1", "AUC-ROC"],
        "Pregunta que responde": ["¿Qué porcentaje de todas las predicciones fue correcto?",
                                  "De las que el modelo llamó graves, ¿cuántas lo eran?",
                                  "De las emergencias realmente graves, ¿cuántas detectó?",
                                  "¿Qué tan equilibradas están la precisión y el recall?",
                                  "¿Qué tan bien ordena las emergencias de mayor a menor riesgo?"],
        "Importancia aquí": ["Engañosa si una clase es más común", "Mide las falsas alarmas",
                             "La más importante: no dejar graves sin atender", "Métrica principal del proyecto",
                             "0.5 es azar; 1 es perfecto"]}), hide_index=True, width="stretch")

    st.markdown("### Efecto del umbral de decisión")
    st.markdown("El modelo entrega una **probabilidad** (por ejemplo, 42 % de ser grave). El **umbral** decide desde qué "
                "probabilidad se considera grave.")
    c1, c2 = st.columns([1, 2.4], gap="large")
    mod = c1.selectbox("Modelo", ORDEN, index=1)
    u = c1.slider("Umbral", 0.05, 0.95, float(round(UMB[mod], 2)), 0.01)
    c1.caption(f"Umbral óptimo, elegido solo con datos de entrenamiento: {UMB[mod]:.2f}")
    y = te["GRAVE"].values
    pred = (PROBA[mod] >= u).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pred).ravel()
    with c2:
        c = st.columns(4)
        kpi(c[0], f"{precision_score(y, pred, zero_division=0):.3f}", "Precisión")
        kpi(c[1], f"{recall_score(y, pred):.3f}", "Recall")
        kpi(c[2], f"{f1_score(y, pred):.3f}", "F1")
        kpi(c[3], f"{accuracy_score(y, pred):.3f}", "Accuracy")
    a, b = st.columns([1, 1.2], gap="large")
    fig = go.Figure(go.Heatmap(z=[[tn, fp], [fn, tp]], x=["Predijo no grave", "Predijo grave"],
                               y=["Era no grave", "Era grave"], text=[[n(tn), n(fp)], [n(fn), n(tp)]],
                               texttemplate="%{text}", textfont=dict(size=18), colorscale=ESC_AZUL, showscale=False))
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(height=300, margin=dict(t=10))
    a.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
    b.markdown(f"""
**Con un umbral de {u:.2f}, en 2025:**

- Detectó **{n(tp)}** de las **{n(tp + fn)}** emergencias graves (**{tp / (tp + fn):.0%}**).
- Se le escaparon **{n(fn)}** emergencias graves: los falsos negativos, el error más costoso.
- Dio **{n(fp)}** falsas alarmas: emergencias no graves marcadas como graves.
- Acertó **{n(tn)}** emergencias no graves.
""")
    nota("info", "Pruebe esto", "Suba el umbral a 0.80: casi no hay falsas alarmas, pero se escapan muchas graves. Bájelo "
         "a 0.10: se detectan casi todas, pero todo parece grave. El umbral óptimo (alrededor de 0.33) equilibra ambos "
         "errores, y en gestión de riesgos es preferible una falsa alarma a una emergencia grave sin atender.")


def p_resultados():
    idea(f"Random Forest es el mejor modelo: F1 = {RF_F1:.3f} y AUC = {RF_AUC:.3f}, y supera al árbol único en "
         f"{(RF_F1 - DT_F1) * 100:.1f} puntos de F1.")
    c = st.columns(4, gap="medium")
    for col, m in zip(c, ORDEN):
        kpi(col, f"{met.loc[m, 'F1']:.3f}", m + (" · mejor modelo" if m == "Random Forest" else ""),
            f"F1 · AUC {met.loc[m, 'AUC-ROC']:.3f} · recall {met.loc[m, 'Recall']:.0%}", destacar=(m == "Random Forest"))
    st.markdown("")
    larga = met.loc[ORDEN, ["Precisión", "Recall", "F1", "AUC-ROC"]].reset_index().melt(
        id_vars="Modelo", var_name="Métrica", value_name="Valor")
    fig = px.bar(larga, x="Métrica", y="Valor", color="Modelo", barmode="group", text=larga["Valor"].round(3),
                 color_discrete_map=COL)
    fig.update_traces(textposition="outside", textfont_size=11)
    fig.update_layout(yaxis_range=[0.4, 0.86])
    grafico(fig, f"Cuatro métricas de cada modelo sobre las {n(len(te))} emergencias de 2025.",
            "Para cada métrica, la barra más alta es la mejor. La línea base es el Árbol de Decisión (celeste).",
            "Los tres ensambles superan al árbol individual. Random Forest (azul oscuro) gana en F1, recall y AUC.", alto=420)

    st.markdown("### ¿Es justa la comparación?")
    c = st.columns(4, gap="small")
    for col, (t, x) in zip(c, [("Mismos datos", "Las mismas 19 variables y las mismas emergencias para los cuatro modelos."),
                               ("Mismo ajuste", "15 combinaciones de hiperparámetros con validación cruzada para cada uno."),
                               ("Mismo umbral", "Cada modelo con su umbral óptimo, elegido solo con entrenamiento."),
                               ("Mismo examen", "Las 16 401 emergencias de 2025 y cuatro años de backtesting.")]):
        col.markdown(f'<div class="paso"><div class="nom">{t}</div><div class="txt">{x}</div></div>', unsafe_allow_html=True)
    st.markdown("")
    nota("info", "Por qué ganó Random Forest", f"En la validación cruzada dentro de 2021–2024, XGBoost iba ligeramente "
         f"adelante ({met.loc['XGBoost', 'F1 (validación cruzada)']:.3f} frente a "
         f"{met.loc['Random Forest', 'F1 (validación cruzada)']:.3f}). Al predecir 2025, XGBoost se sobreajustó a los años "
         "de entrenamiento, mientras que Random Forest, al promediar árboles independientes, fue más robusto ante los "
         "cambios entre años. Límite: no se aplicó una prueba estadística formal de significancia.")

    st.markdown("### Curvas ROC")
    fig = go.Figure()
    for m in ORDEN:
        fpr, tpr, _ = roc_curve(te["GRAVE"], PROBA[m])
        fig.add_scatter(x=fpr, y=tpr, name=f"{m} ({roc_auc_score(te['GRAVE'], PROBA[m]):.3f})",
                        line=dict(color=COL[m], width=3))
    fig.add_scatter(x=[0, 1], y=[0, 1], name="Azar (0.500)", line=dict(dash="dot", color=GRIS))
    fig.update_layout(xaxis_title="Falsas alarmas (tasa de falsos positivos)", yaxis_title="Graves detectadas (recall)")
    grafico(fig, "Para todos los umbrales posibles: graves detectadas (vertical) frente a falsas alarmas (horizontal).",
            "Una curva más cercana a la esquina superior izquierda es mejor. La diagonal punteada es adivinar al azar; el "
            "número entre paréntesis es el AUC.",
            "Los tres ensambles quedan por encima del árbol, y Random Forest arriba en casi todo el rango.", alto=440)

    st.markdown("### Qué variables usó más cada modelo")
    mm = st.radio("Modelo", ["Random Forest", "XGBoost"], horizontal=True)
    imp = pd.Series(MOD[mm].feature_importances_, index=[NOM[f] for f in FEAT])
    imp = (imp / imp.sum() * 100).sort_values()
    fig = px.bar(x=imp.values, y=imp.index, orientation="h", text=imp.round(1).astype(str) + " %",
                 color_discrete_sequence=[COL[mm]], labels={"x": "Importancia (%)", "y": ""})
    fig.update_traces(textposition="outside", cliponaxis=False)
    grafico(fig, "Cuánto aportó cada variable a las decisiones del modelo.",
            "Una barra más larga es una variable más usada para separar emergencias graves de no graves.",
            "Random Forest se apoya en la memoria histórica del distrito; XGBoost, en el tipo de peligro. Ambos coinciden "
            "en que el peligro y la vulnerabilidad histórica del distrito son lo más importante.", alto=580)

    st.markdown("### Validación extra: primer semestre de 2026")
    st.dataframe(met26.loc[ORDEN, ["Precisión", "Recall", "F1", "AUC-ROC"]].round(3), width="stretch")
    st.caption("Random Forest sigue siendo el mejor, pero el AUC baja a cerca de 0.68 por el cambio en el registro de "
               "viviendas destruidas. Un modelo en uso real debe vigilarse y reentrenarse cuando cambian los datos.")


@st.cache_data
def geo():
    with open(os.path.join(BASE, "data", "geo", "peru_departamental.geojson"), encoding="utf-8") as f:
        g = json.load(f)
    nombres = {ft["properties"]["FIRST_IDDP"]: ft["properties"]["NOMBDEP"] for ft in g["features"]}

    def area(r):
        return sum(x1 * y2 - x2 * y1 for (x1, y1), (x2, y2) in zip(r, r[1:] + r[:1])) / 2

    def orientar(poly):  # el anillo exterior debe ir en sentido horario para Plotly
        return [r if (area(r) < 0) == (i == 0) else r[::-1] for i, r in enumerate(poly)]

    for ft in g["features"]:
        gm = ft["geometry"]
        if gm["type"] == "Polygon":
            gm["coordinates"] = orientar(gm["coordinates"])
        elif gm["type"] == "MultiPolygon":
            gm["coordinates"] = [orientar(pl) for pl in gm["coordinates"]]
    return g, nombres


@st.cache_data
def por_departamento(modelo):
    _, nomdep = geo()
    e = est.assign(COD=est["UBIGEO"].str[:2])
    tot = e.groupby("COD").agg(emergencias=("GRAVE", "size"), pct_graves=("GRAVE", "mean"))
    t = te.assign(COD=te["UBIGEO"].str[:2], p=PROBA[modelo])
    t["pred"] = (t["p"] >= UMB[modelo]).astype(int)
    filas = []
    for cod, g in t.groupby("COD"):
        y, pr = g["GRAVE"], g["pred"]
        vp, fn = int(((y == 1) & (pr == 1)).sum()), int(((y == 1) & (pr == 0)).sum())
        fp, vn = int(((y == 0) & (pr == 1)).sum()), int(((y == 0) & (pr == 0)).sum())
        filas.append({"COD": cod, "n_2025": len(g), "graves_2025": int(y.sum()), "VP": vp, "FN": fn, "FP": fp,
                      "recall": vp / (vp + fn) if vp + fn else np.nan,
                      "precision": vp / (vp + fp) if vp + fp else np.nan,
                      "F1": f1_score(y, pr, zero_division=0) if y.sum() else np.nan})
    d = tot.join(pd.DataFrame(filas).set_index("COD"), how="left").reset_index()
    d["Departamento"] = d["COD"].map(nomdep).fillna(d["COD"]).str.title()
    return d


@st.cache_data
def mensual():
    datos = pd.concat([te, t26])
    filas = []
    for m in ORDEN:
        p = MOD[m].predict_proba(datos[FEAT])[:, 1]
        d = datos.assign(p=p, pred=(p >= UMB[m]).astype(int))
        for (a, mes), g in d.groupby(["ANIO", "MES_NUM"]):
            if g["GRAVE"].nunique() < 2:
                continue
            filas.append({"Modelo": m, "Fecha": pd.Timestamp(int(a), int(mes), 1), "N": len(g),
                          "F1": f1_score(g["GRAVE"], g["pred"]), "AUC-ROC": roc_auc_score(g["GRAVE"], g["p"]),
                          "Recall": recall_score(g["GRAVE"], g["pred"])})
    return pd.DataFrame(filas)


def p_mapa():
    idea("El modelo no rinde igual en todo el país ni en todas las épocas, y conocer dónde falla es parte del resultado.")
    GEO, _ = geo()
    modelo = st.selectbox("Modelo a analizar", ORDEN, index=1)
    dd = por_departamento(modelo)
    t1, t2, t3 = st.tabs(["Mapa del Perú", "Errores por departamento", "Evolución en el tiempo"])
    with t1:
        opciones = {"Emergencias registradas (2021–2026)": ("emergencias", ESC_AZUL, "Emergencias"),
                    "Porcentaje de emergencias graves": ("pct_graves", ESC_ROJO, "% graves"),
                    f"F1 del modelo en 2025": ("F1", ESC_F1, "F1"),
                    "Graves no detectadas en 2025": ("FN", ESC_ROJO, "Graves no detectadas"),
                    "Falsas alarmas en 2025": ("FP", ESC_AZUL, "Falsas alarmas")}
        a, b = st.columns([1.35, 1], gap="large")
        elegido = b.radio("Variable del mapa", list(opciones))
        col, escala, etiqueta = opciones[elegido]
        dm = dd.copy()
        dm["pct_graves"] *= 100
        fig = px.choropleth(dm, geojson=GEO, locations="COD", featureidkey="properties.FIRST_IDDP", color=col,
                            color_continuous_scale=escala, hover_name="Departamento",
                            hover_data={"COD": False, "emergencias": ":,", "pct_graves": ":.1f", "F1": ":.3f",
                                        "FN": True, "FP": True},
                            labels={col: etiqueta, "emergencias": "Emergencias 2021–2026", "pct_graves": "% graves",
                                    "FN": "Graves no detectadas", "FP": "Falsas alarmas"})
        fig.update_traces(marker_line_color="white", marker_line_width=0.8)
        fig.update_geos(visible=False, projection_type="mercator", lonaxis_range=[-81.8, -68.2],
                        lataxis_range=[-18.6, 0.2], bgcolor="rgba(0,0,0,0)")
        fig.update_layout(height=700, margin=dict(l=0, r=0, t=0, b=0),
                          coloraxis_colorbar=dict(title=etiqueta, len=0.5, x=0.9, thickness=12))
        a.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        top = dm.sort_values(col, ascending=(col == "F1")).head(5)
        with b:
            st.markdown(f"**{'Los 5 con peor F1' if col == 'F1' else 'Los 5 más altos'}**")
            st.dataframe(top[["Departamento", col]].rename(columns={col: etiqueta}).round(3), hide_index=True,
                         width="stretch")
            lectura(st, "Los 25 departamentos coloreados según la variable elegida.",
                    "Más intenso, valor más alto. Pase el mouse sobre un departamento para ver todos sus datos.",
                    "La mayoría de emergencias está en la sierra sur y centro, pero el porcentaje de graves es mayor en la "
                    "costa y la selva.")
    with t2:
        de = dd.dropna(subset=["F1"])
        de = de[de["n_2025"] >= 30].sort_values("F1")
        f1_nac = met.loc[modelo, "F1"]
        fig = px.bar(de, x="F1", y="Departamento", orientation="h", color="recall", color_continuous_scale=ESC_F1,
                     range_color=(0.3, 1), text=de["F1"].round(2),
                     hover_data={"n_2025": True, "graves_2025": True, "FN": True, "FP": True},
                     labels={"recall": "Recall", "n_2025": "Emergencias 2025", "graves_2025": "Graves reales",
                             "FN": "Graves no detectadas", "FP": "Falsas alarmas"})
        fig.add_vline(x=f1_nac, line_dash="dot", line_color=TINTA, annotation_text=f"Nacional {f1_nac:.3f}")
        peor, mejor = de.head(3), de.tail(3).iloc[::-1]
        grafico(fig, f"El F1 de {modelo} en cada departamento con al menos 30 emergencias en 2025. El color es el recall.",
                "A la derecha de la línea punteada, mejor que el promedio nacional; a la izquierda, peor.",
                f"Mejores: {', '.join(mejor['Departamento'])}. Peores: {', '.join(peor['Departamento'])}. Acierta más donde "
                "predominan los incendios urbanos con historial claro (Lima, Callao) y menos donde casi todo son lluvias "
                "intensas y pocas son graves (Cajamarca, Áncash), porque sin datos de precipitación no distingue qué "
                "lluvia destruirá viviendas.", alto=720)
        de2 = de.sort_values("graves_2025", ascending=False).head(12)
        larga = de2.melt(id_vars="Departamento", value_vars=["VP", "FN", "FP"], var_name="Tipo", value_name="Cantidad")
        larga["Tipo"] = larga["Tipo"].map({"VP": "Graves detectadas", "FN": "Graves no detectadas", "FP": "Falsas alarmas"})
        fig = px.bar(larga, x="Departamento", y="Cantidad", color="Tipo", barmode="group",
                     color_discrete_map={"Graves detectadas": AZUL, "Graves no detectadas": GRAVE, "Falsas alarmas": AMBAR})
        grafico(fig, "Para los 12 departamentos con más emergencias graves en 2025: aciertos y dos tipos de error.",
                "Lo ideal es una barra azul alta y barras roja y ámbar bajas.",
                "Los errores más costosos son los rojos: emergencias graves sin atender. Donde abundan, convendría bajar "
                "el umbral o incorporar datos locales.", alto=420)
    with t3:
        st.markdown("#### Validación año por año")
        st.markdown("El experimento se repitió cuatro veces, siempre entrenando con los años anteriores y probando en el "
                    "siguiente: 2023, 2024, 2025 y 2026.")
        metr = st.radio("Métrica", ["AUC-ROC", "F1", "Recall", "Precisión"], horizontal=True)
        fig = px.line(bt, x="Año de prueba", y=metr, color="Modelo", markers=True, color_discrete_map=COL,
                      hover_data=["Entrenado con", "N entrenamiento", "N prueba"])
        fig.update_traces(line=dict(width=3), marker=dict(size=10))
        fig.update_layout(xaxis=dict(dtick=1))
        grafico(fig, "El rendimiento de cada modelo en cuatro años que no vio al entrenar.",
                "Cada línea es un modelo; si una está siempre arriba, ese modelo es consistentemente mejor.",
                "El árbol único es el peor los cuatro años. Random Forest gana en 2023, 2025 y 2026, y en 2024 los "
                "ensambles casi empatan. El AUC mejora con más años de entrenamiento y cae en 2026 por el cambio de "
                "registro.", alto=400)
        st.markdown("#### Mes a mes")
        mm = mensual()
        sel = st.multiselect("Modelos", ORDEN, default=["Árbol de Decisión", "Random Forest", "XGBoost"])
        met_m = st.radio("Métrica mensual", ["AUC-ROC", "F1", "Recall"], horizontal=True, key="mm")
        fig = go.Figure()
        base = mm[mm["Modelo"] == "Random Forest"]
        fig.add_bar(x=base["Fecha"], y=base["N"], name="Emergencias del mes", marker_color="#E3E8EF", yaxis="y2")
        for m in sel:
            s = mm[mm["Modelo"] == m]
            fig.add_scatter(x=s["Fecha"], y=s[met_m], name=m, mode="lines+markers", line=dict(color=COL[m], width=3))
        fig.add_vrect(x0="2025-12-16", x1="2026-06-30", fillcolor="#F6E1E4", opacity=0.45, line_width=0,
                      annotation_text="2026", annotation_position="top left")
        fig.update_layout(yaxis=dict(title=met_m, overlaying="y2"),
                          yaxis2=dict(side="right", title="Emergencias", showgrid=False))
        grafico(fig, "El rendimiento mes a mes en 2025 y el primer semestre de 2026, con las emergencias del mes al fondo.",
                "Cada punto es un mes. Las barras grises usan el eje derecho. La zona rosada es 2026.",
                "El modelo acierta más en los meses secos (AUC de 0.87 a 0.91 en junio–agosto) que en la temporada de "
                "lluvias (0.72 en febrero–abril). En marzo, el 61 % de las emergencias son lluvias intensas, difíciles de "
                "distinguir sin datos de precipitación.", alto=420)


def p_simulador():
    idea("Una emergencia real de 2025, que los modelos nunca vieron: qué predice cada modelo y qué pasó realmente.")
    casos = paq["casos"]
    c1, c2, c3, c4 = st.columns([1, 1, 1, 0.8])
    pel = c1.selectbox("Peligro", ["Todos"] + sorted(casos["PELIGRO"].unique()))
    dep = c2.selectbox("Departamento", ["Todos"] + sorted(casos["DEPARTAMENTO"].unique()))
    real = c3.selectbox("Resultado real", ["Todos", "Grave", "No grave"])
    if "semilla" not in st.session_state:
        st.session_state.semilla = 0
    c4.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)
    if c4.button("Otra emergencia", type="primary", width="stretch"):
        st.session_state.semilla += 1
    d = casos
    if pel != "Todos":
        d = d[d["PELIGRO"] == pel]
    if dep != "Todos":
        d = d[d["DEPARTAMENTO"] == dep]
    if real != "Todos":
        d = d[d["GRAVE"] == (1 if real == "Grave" else 0)]
    if d.empty:
        st.warning("No hay emergencias con esos filtros.")
        return
    fila = d.sample(1, random_state=st.session_state.semilla).iloc[0]
    a, b = st.columns([1, 1.3], gap="large")
    with a.container(border=True):
        st.markdown(f"<div class='miga'>Emergencia SINPAD {fila['CODIGO']}</div>", unsafe_allow_html=True)
        st.markdown(f"#### {fila['DISTRITO'].title()}, {fila['PROVINCIA'].title()} ({fila['DEPARTAMENTO'].title()})")
        fecha = pd.to_datetime(fila["FECHA"], format="%m/%d/%y", errors="coerce")
        fecha_txt = fecha.strftime("%d/%m/%Y") if pd.notna(fecha) else fila["FECHA"]
        st.markdown(f"**Fecha:** {fecha_txt} &nbsp;·&nbsp; **Peligro:** {fila['PELIGRO'].capitalize()}")
        st.markdown("**Lo que el modelo sabía al momento del reporte**")
        st.dataframe(pd.DataFrame({"Variable": [NOM[f] for f in ["DIST_TASA_GRAVE", "DISTPEL_TASA_GRAVE", "DIST_N_PREV",
                                                                 "DEP_EMERG_7D"]],
                                   "Valor": [f"{fila['DIST_TASA_GRAVE']:.0%}", f"{fila['DISTPEL_TASA_GRAVE']:.0%}",
                                             n(fila["DIST_N_PREV"]), n(fila["DEP_EMERG_7D"])]}),
                     hide_index=True, width="stretch")
        with st.expander("Mostrar lo que realmente pasó"):
            chip = "grave" if fila["GRAVE"] == 1 else "nograve"
            st.markdown(f"Fallecidos: **{int(fila['FALLECIDOS'])}** · Desaparecidos: **{int(fila['DESAPARECIDOS'])}** · "
                        f"Damnificados: **{int(fila['DAMNIFICADOS'])}** · Afectados: **{int(fila['AFECTADOS'])}** · "
                        f"Viviendas destruidas: **{int(fila['VIVIENDAS DESTRUIDAS'])}**")
            st.markdown(f"<span class='chip {chip}'>{'GRAVE' if fila['GRAVE'] == 1 else 'NO GRAVE'}</span>",
                        unsafe_allow_html=True)
    X = fila[FEAT].to_frame().T.astype(float)
    probs = {m: MOD[m].predict_proba(X)[0, 1] for m in ORDEN}
    with b:
        fig = go.Figure()
        fig.add_bar(y=ORDEN, x=[probs[m] for m in ORDEN], orientation="h", marker_color=[COL[m] for m in ORDEN],
                    text=[f"{probs[m]:.0%}" for m in ORDEN], textposition="outside", cliponaxis=False, name="P(grave)")
        fig.add_scatter(y=ORDEN, x=[UMB[m] for m in ORDEN], mode="markers", name="Umbral del modelo",
                        marker=dict(symbol="line-ns", size=26, line=dict(width=3, color=TINTA)))
        fig.update_layout(height=280, xaxis=dict(range=[0, 1.1], tickformat=".0%", title="Probabilidad de emergencia grave"),
                          yaxis=dict(autorange="reversed"), margin=dict(t=30))
        st.plotly_chart(fig, width="stretch", config={"displayModeBar": False})
        filas = []
        for m in ORDEN:
            pred = int(probs[m] >= UMB[m])
            filas.append({"Modelo": m, "Predicción": "Grave" if pred else "No grave",
                          "Resultado": "Acierta" if pred == fila["GRAVE"] else "Falla"})
        st.dataframe(pd.DataFrame(filas), hide_index=True, width="stretch")
    st.markdown("### El recorrido dentro del Árbol de Decisión")
    arbol = MOD["Árbol de Decisión"].tree_
    x = X.values[0].astype(np.float32)
    nodo, pasos = 0, []
    while arbol.children_left[nodo] != -1:
        f, um = arbol.feature[nodo], arbol.threshold[nodo]
        izq = x[f] <= um
        pasos.append(f"Nodo {nodo}: ¿{NOM[FEAT[f]]} ≤ {um:.3f}? · valor {x[f]:.3f} · "
                     f"{'sí, baja a la izquierda' if izq else 'no, baja a la derecha'}")
        nodo = arbol.children_left[nodo] if izq else arbol.children_right[nodo]
    v = arbol.value[nodo][0]
    a, b = st.columns([1.6, 1], gap="large")
    with a:
        for i, p in enumerate(pasos[:10], 1):
            st.markdown(f"<div class='paso-arbol'>{i:>2}. {p}</div>", unsafe_allow_html=True)
        if len(pasos) > 10:
            st.markdown(f"<div class='paso-arbol'>… y {len(pasos) - 10} preguntas más</div>", unsafe_allow_html=True)
    with b:
        nota("ok", f"Hoja {nodo}, profundidad {len(pasos)}", f"De las emergencias de entrenamiento que llegaron a esta hoja, "
             f"el <b>{v[1] / v.sum():.0%}</b> fueron graves. Recorrer el árbol es como buscar en un árbol binario: en cada "
             "nodo se compara un valor y se baja a la izquierda o a la derecha, con costo O(profundidad).")


def p_conclusiones():
    idea("Es posible anticipar la gravedad de una emergencia, y combinar muchos árboles funciona mejor que uno solo.")
    conc = [("01", "Se puede anticipar", f"Con la información del momento del reporte, Random Forest detecta el "
             f"{RF_REC:.0%} de las emergencias graves de un año que nunca vio (AUC = {RF_AUC:.3f})."),
            ("02", "Muchos árboles superan a uno", f"El árbol único sobreajusta; los ensambles lo superan (F1 de "
             f"{DT_F1:.3f} a {RF_F1:.3f}) y el resultado se repite en los cuatro años evaluados."),
            ("03", "Bagging fue más robusto", "XGBoost se ajustó demasiado a 2021–2024; Random Forest, al promediar "
             "árboles independientes, resistió mejor los cambios entre años."),
            ("04", "Peligro e historial deciden", "Lo que más pesa es el tipo de peligro (incendios, colapsos y sismos "
             "destruyen viviendas) y la vulnerabilidad histórica del distrito.")]
    for fila in (conc[:2], conc[2:]):
        for col, (num, t, x) in zip(st.columns(2, gap="large"), fila):
            col.markdown(f'<div class="paso"><div class="num">CONCLUSIÓN {num}</div><div class="nom">{t}</div>'
                         f'<div class="txt">{x}</div></div>', unsafe_allow_html=True)
        st.markdown("")
    a, b = st.columns(2, gap="large")
    with a:
        nota("info", "Uso práctico", "Integrar el modelo al SINPAD como alerta temprana para priorizar la evaluación de "
             "daños y el envío de ayuda humanitaria en las primeras horas.")
    with b:
        nota("alerta", "Límites y trabajo futuro", "Falla más en la temporada de lluvias: agregar precipitación del "
             "SENAMHI sería la mejora más útil. También conviene monitorear cambios de registro como el de 2026.")
    st.markdown("### Preguntas frecuentes")
    faq = {
        "¿Por qué no se obtiene 90 % de acierto o más?":
            "Porque se predice antes de conocer los daños. Con las columnas de daño se llega a 100 %, pero sería trampa. "
            "Un AUC de 0.78 en un año futuro es un resultado sólido para un problema real.",
        "¿Por qué Random Forest y no XGBoost?":
            "En la validación cruzada XGBoost ganó, pero al predecir 2025 se sobreajustó: su error en prueba sube después "
            "de unos 90 árboles. Random Forest fue más robusto ante el cambio entre años.",
        "¿Por qué el umbral es 0.33 y no 0.5?":
            "Solo el 35 % del entrenamiento es grave; con 0.5 se escapan muchas graves. El umbral se eligió maximizando F1 "
            "solo con datos de entrenamiento.",
        "¿Qué relación tiene con Estructuras de Datos?":
            "El árbol de decisión es un árbol binario: se construye recursivamente, cada nodo guarda una condición y "
            "clasificar es recorrerlo de la raíz a una hoja en O(profundidad).",
        "¿Cómo se evitó que el modelo viera el futuro?":
            "Sin columnas de daño, memoria histórica solo con años anteriores, entrenamiento 2021–2024 y prueba 2025, e "
            "hiperparámetros y umbral elegidos solo con entrenamiento.",
        "¿Dónde y cuándo se equivoca más?":
            "En la temporada de lluvias y en departamentos andinos donde casi todo son lluvias intensas (Cajamarca, "
            "Áncash). Acierta más con incendios urbanos en Lima y Callao.",
        "¿El resultado es casualidad de un solo año?":
            "No. En el backtesting de 2023 a 2026 el árbol único fue el peor los cuatro años y Random Forest ganó en tres.",
    }
    for q, r in faq.items():
        with st.expander(q):
            st.write(r)

# NAVEGACIÓN
PAGINAS = [
    ("Apertura", "Presentación del proyecto", p_presentacion),
    ("Contexto", "Inicio", p_inicio),
    ("Contexto", "El problema", p_problema),
    ("Datos", "Los datos", p_datos),
    ("Datos", "Limpieza de datos", p_limpieza),
    ("Datos", "Variable objetivo y fuga de datos", p_objetivo),
    ("Datos", "Exploración de los datos", p_exploracion),
    ("Modelos", "Árbol de Decisión", p_arbol),
    ("Modelos", "Random Forest y XGBoost", p_ensambles),
    ("Evaluación", "Validación", p_validacion),
    ("Evaluación", "Resultados", p_resultados),
    ("Evaluación", "Mapa y rendimiento en el tiempo", p_mapa),
    ("Demostración y cierre", "Simulador", p_simulador),
    ("Demostración y cierre", "Conclusiones", p_conclusiones),
]
if "pag" not in st.session_state:
    st.session_state.pag = 0


def ir(i):
    st.session_state.pag = i


with st.sidebar:
    st.markdown(f'<div class="marca">Emergencias graves<br>en el Perú</div>'
                f'<div class="marca-sub">INDECI · SINPAD 2021–2026</div>', unsafe_allow_html=True)
    st.toggle("Modo presentación", key="pres", help="Agranda los textos para proyectar.")
    bloque_actual = None
    for i, (bloque, titulo, _) in enumerate(PAGINAS):
        if bloque != bloque_actual:
            st.markdown(f'<div class="nav-bloque">{bloque}</div>', unsafe_allow_html=True)
            bloque_actual = bloque
        st.button(titulo, key=f"nav{i}", on_click=ir, args=(i,), width="stretch",
                  type="primary" if i == st.session_state.pag else "tertiary")
    with st.expander("Glosario"):
        st.markdown("""
**Árbol de decisión.** Árbol binario de preguntas cuyas hojas dan la clase.
**Gini / entropía.** Miden qué tan mezclado está un nodo.
**Sobreajuste.** El modelo memoriza los datos de entrenamiento.
**Bagging.** Muchos modelos en paralelo que votan (Random Forest).
**Boosting.** Modelos en secuencia que corrigen errores (XGBoost).
**Recall.** Porcentaje de emergencias graves detectadas.
**F1.** Equilibrio entre precisión y recall.
**AUC.** Capacidad de ordenar por riesgo; 0.5 es azar.
**Fuga de datos.** Usar información que no se tendría al predecir.
**Umbral.** Probabilidad desde la que se considera grave.
""")

i = st.session_state.pag
bloque, titulo, funcion = PAGINAS[i]
st.markdown(f'<div class="miga">{bloque} &nbsp;·&nbsp; Paso {i + 1} de {len(PAGINAS)}</div>', unsafe_allow_html=True)
st.progress((i + 1) / len(PAGINAS))
if funcion not in (p_presentacion, p_inicio):
    st.title(titulo)
funcion()

st.markdown("---")
a, _, b = st.columns([1, 2, 1])
if i > 0:
    a.button(f"Anterior: {PAGINAS[i - 1][1]}", on_click=ir, args=(i - 1,), width="stretch", key="prev")
if i < len(PAGINAS) - 1:
    b.button(f"Siguiente: {PAGINAS[i + 1][1]}", on_click=ir, args=(i + 1,), width="stretch", type="primary", key="next")
