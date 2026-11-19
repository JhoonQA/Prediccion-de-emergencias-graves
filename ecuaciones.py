
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
plt.rcParams["mathtext.fontset"] = "cm"

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figuras", "ecuaciones")
os.makedirs(OUT, exist_ok=True)

ECUACIONES = {
    "eq01_grave": r"$y_i=\begin{cases}1 & \mathrm{si}\ (F_i+D_i>0)\ \lor\ (Dam_i>0)\ \lor\ (VD_i>0)\\ 0 & \mathrm{en\ otro\ caso}\end{cases}$",
    "eq02_gini": r"$G(t)=1-\sum_{k=1}^{K}p_k^{2}$",
    "eq03_entropia": r"$H(t)=-\sum_{k=1}^{K}p_k\,\log_{2}p_k$",
    "eq04_ganancia": r"$\Delta I(t,s)=I(t)-\frac{N_{izq}}{N_t}\,I(t_{izq})-\frac{N_{der}}{N_t}\,I(t_{der})$",
    "eq05_rf": r"$\hat{y}(x)=\mathrm{moda}\{\,h_1(x),\,h_2(x),\,\ldots,\,h_B(x)\,\}$",
    "eq06_xgb_modelo": r"$\hat{y}_i^{(m)}=\hat{y}_i^{(m-1)}+\eta\,f_m(x_i)$",
    "eq07_xgb_objetivo": r"$\mathcal{L}^{(m)}=\sum_{i=1}^{n}l\left(y_i,\hat{y}_i^{(m-1)}+f_m(x_i)\right)+\gamma T+\frac{1}{2}\lambda\sum_{j=1}^{T}w_j^{2}$",
    "eq08_metricas": r"$\mathrm{Precisión}=\frac{VP}{VP+FP}\qquad \mathrm{Recall}=\frac{VP}{VP+FN}\qquad F_1=2\cdot\frac{\mathrm{Precisión}\cdot\mathrm{Recall}}{\mathrm{Precisión}+\mathrm{Recall}}$",
}

ECUACIONES["eq01_grave"] = (r"$y_i=1\ \ \mathrm{si}\ \ (F_i+D_i>0)\ \vee\ (Dam_i>0)\ \vee\ (VD_i>0);"
                            r"\quad y_i=0\ \ \mathrm{en\ otro\ caso}$")


def render(nombre, tex, size=15):
    fig = plt.figure(figsize=(0.01, 0.01))
    fig.text(0, 0, tex, fontsize=size)
    fig.savefig(os.path.join(OUT, nombre + ".png"), dpi=250, bbox_inches="tight", pad_inches=0.05,
                transparent=False, facecolor="white")
    plt.close(fig)


if __name__ == "__main__":
    for k, v in ECUACIONES.items():
        render(k, v)
        print("ok", k)
