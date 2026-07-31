"""Figuras del informe. Estáticas, a PNG. Sin widgets."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

FIG = Path(__file__).resolve().parents[1] / "figures"
CSV = Path(__file__).resolve().parents[1] / "data" / "walkforward.csv"
NIVEL = 0.99
ACTO1 = ["historico", "normal", "mc_gbm", "mc_merton", "mc_merton_idio"]
ORDEN = ACTO1 + ["t_student", "cornish_fisher", "evt", "ewma", "fhs"]

# Un color por modelo. Los cinco del Acto 2 caían todos en el mismo verde de
# reserva, así que en las figuras 5 y 6 —las que sostienen las conclusiones—
# la mitad de los puntos era indistinguible.
COLOR = {"historico": "#555555", "normal": "#d62728", "mc_gbm": "#ff7f0e",
         "mc_merton": "#1f77b4", "mc_merton_idio": "#9467bd",
         "t_student": "#8c564b", "cornish_fisher": "#e377c2",
         "evt": "#17becf", "ewma": "#2ca02c", "fhs": "#bcbd22"}


def cargar():
    return pd.read_csv(CSV, parse_dates=["fecha"])


def fig1_excepciones_acumuladas(df, cartera="igual_peso", modelos=None):
    """Si un modelo está bien calibrado, su curva sigue la diagonal esperada.

    Por defecto solo los cinco del Acto 1: es la figura que acompaña ese
    argumento en el informe, y con los diez encima las curvas se solapan y
    dejan de leerse.
    """
    modelos = modelos or ACTO1
    fig, ax = plt.subplots(figsize=(11, 5.5))
    g = df[df.cartera == cartera]
    for m in modelos:
        s = g[g.modelo == m].sort_values("fecha")
        ax.plot(s.fecha, s.excepcion.cumsum(), lw=1.8, color=COLOR[m], label=m)
    n = g.fecha.nunique()
    fechas = np.sort(g.fecha.unique())
    ax.plot(fechas, np.arange(1, n + 1) * (1 - NIVEL), "k--", lw=1.5,
            label=f"esperado ({(1-NIVEL):.0%})")
    ax.set_title(f"Excepciones acumuladas al {NIVEL:.0%} — cartera {cartera}")
    ax.set_ylabel("excepciones acumuladas")
    ax.legend(frameon=False, ncol=3, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "1_excepciones_acumuladas.png", dpi=150)
    plt.close(fig)


def fig2_var_vs_realizado(df, cartera="igual_peso", ini="2019-06-01", fin="2020-12-31"):
    """El episodio que define el resultado: marzo de 2020."""
    fig, ax = plt.subplots(figsize=(11, 5.5))
    g = df[(df.cartera == cartera) & df.fecha.between(ini, fin)]
    base = g[g.modelo == "normal"].sort_values("fecha")
    ax.bar(base.fecha, -base.realizado, width=1.0, color="#cccccc",
           label="pérdida realizada")
    for m in ("normal", "mc_merton"):
        s = g[g.modelo == m].sort_values("fecha")
        ax.plot(s.fecha, s.VaR, lw=2, color=COLOR[m], label=f"VaR 99% — {m}")
        exc = s[s.excepcion == 1]
        ax.scatter(exc.fecha, -exc.realizado, s=22, color=COLOR[m], zorder=5,
                   edgecolor="white", linewidth=0.5)
    ax.set_title(f"VaR predicho vs pérdida realizada — {cartera} ({ini[:7]} a {fin[:7]})")
    ax.set_ylabel("pérdida diaria")
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "2_var_vs_realizado.png", dpi=150)
    plt.close(fig)


def fig3_razon_excepciones(df):
    """Razón observadas/esperadas. 1.0 es la calibración perfecta."""
    p = 1 - NIVEL
    t = (df.groupby(["cartera", "modelo"])
           .agg(exc=("excepcion", "sum"), n=("excepcion", "size")).reset_index())
    t["razon"] = t.exc / (t.n * p)
    piv = t.pivot(index="cartera", columns="modelo", values="razon")[ORDEN]

    fig, ax = plt.subplots(figsize=(11, 5))
    piv.plot(kind="bar", ax=ax, color=[COLOR[m] for m in ORDEN], width=0.78)
    ax.axhline(1.0, color="k", ls="--", lw=1.5)
    ax.set_title("Excepciones observadas / esperadas al 99%  (1.0 = perfecto)")
    ax.set_ylabel("razón")
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=0)
    ax.legend(frameon=False, ncol=5, fontsize=8)
    ax.grid(alpha=0.25, axis="y")
    fig.tight_layout()
    fig.savefig(FIG / "3_razon_excepciones.png", dpi=150)
    plt.close(fig)


def fig4_carteras(df):
    """Equity y underwater de las cuatro carteras out-of-sample."""
    r = (df.drop_duplicates(["fecha", "cartera"])
           .pivot(index="fecha", columns="cartera", values="realizado"))
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(11, 8), sharex=True,
                                 gridspec_kw={"height_ratios": [2, 1]})
    for c in r.columns:
        eq = np.exp(r[c].cumsum())
        a1.plot(eq.index, eq, lw=1.7, label=c)
        a2.fill_between(eq.index, eq / eq.cummax() - 1, 0, alpha=0.25)
    a1.set_title("Crecimiento de $1 out-of-sample (2011-2026)")
    a1.set_ylabel("valor")
    a1.legend(frameon=False, ncol=4, fontsize=9)
    a1.grid(alpha=0.25)
    a2.set_title("Drawdown")
    a2.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "4_carteras.png", dpi=150)
    plt.close(fig)


def fig5_frontera(t):
    """El intercambio: acertar la frecuencia vs acertar el momento.

    El punto ideal es (1, 1) abajo a la izquierda. Nadie está ahí.
    """
    fig, ax = plt.subplots(figsize=(9.5, 6.5))
    for m, row in t.iterrows():
        ax.scatter(row.razon, row.pers, s=170, color=COLOR.get(m, "#2ca02c"),
                   edgecolor="white", zorder=4)
        ax.annotate(m, (row.razon, row.pers), textcoords="offset points",
                    xytext=(9, 4), fontsize=9)
    ax.axvline(1.0, color="k", ls="--", lw=1.2, alpha=0.6)
    ax.axhline(1.0, color="k", ls=":", lw=1.2, alpha=0.6)
    ax.scatter([1], [1], marker="*", s=380, color="gold", edgecolor="k",
               zorder=5, label="ideal (1, 1)")
    ax.set_xlabel("razón de excepciones observadas/esperadas  (1.0 = frecuencia correcta)")
    ax.set_ylabel("persistencia π₁₁/π₀₁  (1.0 = sin agrupamiento)")
    ax.set_title("Ningún modelo acierta las dos cosas")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "5_frontera.png", dpi=150)
    plt.close(fig)


def fig6_incentivo(t):
    """El hallazgo incómodo: peor modelo, menos capital."""
    fig, ax = plt.subplots(figsize=(9.5, 6))
    ax.scatter(t.pasa, t.capital, s=170, c=[COLOR.get(m, "#2ca02c") for m in t.index],
               edgecolor="white", zorder=4)
    for m, row in t.iterrows():
        ax.annotate(m, (row.pasa, row.capital), textcoords="offset points",
                    xytext=(9, 3), fontsize=9)
    z = np.polyfit(t.pasa, t.capital, 1)
    xs = np.linspace(t.pasa.min(), t.pasa.max(), 10)
    ax.plot(xs, np.polyval(z, xs), "r--", lw=1.6,
            label=f"pendiente {z[0]:+.1f} mil USD por prueba superada")
    ax.set_xlabel("pruebas estadísticas superadas (de 16)")
    ax.set_ylabel("proxy de capital, miles de USD por $10M")
    ax.set_title("El incentivo invertido: los modelos que reprueban cuestan menos")
    # La figura aparece en el README: el descargo viaja con ella, no solo en el texto.
    fig.text(0.5, 0.005,
             "Proxy bajo el semáforo VaR histórico (Basilea II/2.5). No es capital regulatorio.",
             ha="center", fontsize=7.5, color="#666666")
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "6_incentivo.png", dpi=150)
    plt.close(fig)


def tabla_carteras(df) -> pd.DataFrame:
    r = (df.drop_duplicates(["fecha", "cartera"])
           .pivot(index="fecha", columns="cartera", values="realizado"))
    out = {}
    for c in r.columns:
        x = r[c].dropna()
        eq = np.exp(x.cumsum())
        var = -np.quantile(x, 1 - NIVEL)
        out[c] = {
            "ret_anual": x.mean() * 252,
            "vol_anual": x.std() * np.sqrt(252),
            "sharpe": (x.mean() * 252) / (x.std() * np.sqrt(252)),
            "VaR99": var,
            "CVaR99": -x[x <= -var].mean(),
            "peor_dia": -x.min(),
            "max_dd": -(eq / eq.cummax() - 1).min(),
        }
    return pd.DataFrame(out).T


if __name__ == "__main__":
    FIG.mkdir(exist_ok=True)
    df = cargar()
    fig1_excepciones_acumuladas(df)
    fig2_var_vs_realizado(df)
    fig3_razon_excepciones(df)
    fig4_carteras(df)
    print("Figuras en figures/\n")
    print(tabla_carteras(df).round(4).to_string())
