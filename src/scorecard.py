"""Model scorecard with declared criteria and bootstrap confidence intervals.

Point rankings hide how much of a gap is noise. With ~39 expected exceptions in
3,923 days, a ratio of 1.20 against 1.08 may well be the same model twice. This
module attaches an interval to every ratio and scores models on stated
criteria instead of narrative.

    python -m src.scorecard
"""

import numpy as np
import pandas as pd

from src import backtest as bt, basel

NIVEL = 0.99

# Declared criteria. Each is a hard threshold, stated up front, so the ranking
# cannot be tuned after seeing the results.
CRITERIOS = {
    "cobertura": "razón obs/esp dentro de [0.8, 1.2] y Kupiec no rechaza (p > 0.05)",
    "independencia": "Christoffersen de independencia no rechaza (p > 0.05)",
    "magnitud": "Acerbi-Székely no rechaza (p > 0.05)",
    "estabilidad": "el IC bootstrap al 95% de la razón contiene 1.0",
}


def ic_razon(exc: np.ndarray, nivel: float = NIVEL, n_boot: int = 5_000,
             rng=None) -> tuple[float, float, float]:
    """Bootstrap CI for the observed/expected exception ratio.

    Resampled in blocks of 20 days, not i.i.d.: exceptions cluster, and an
    i.i.d. bootstrap would break that dependence and report an interval far
    narrower than the truth.
    """
    rng = rng or np.random.default_rng(0)
    e = np.asarray(exc).astype(int)
    n, p, bloque = len(e), 1 - nivel, 20
    razones = np.empty(n_boot)
    n_bloques = n // bloque
    inicios = np.arange(n_bloques) * bloque
    for b in range(n_boot):
        sel = rng.choice(inicios, size=n_bloques, replace=True)
        muestra = np.concatenate([e[i:i + bloque] for i in sel])
        razones[b] = muestra.sum() / (len(muestra) * p)
    return (float(e.sum() / (n * p)), float(np.quantile(razones, 0.025)),
            float(np.quantile(razones, 0.975)))


def scorecard(df: pd.DataFrame, nivel: float = NIVEL, rng=None) -> pd.DataFrame:
    """One row per (portfolio, model) with the four criteria and the interval."""
    rng = rng or np.random.default_rng(0)
    p = 1 - nivel
    estado = df.get("estado_modelo", pd.Series("ok", index=df.index))
    filas = []

    for (c, m), g in df.assign(_e=estado).groupby(["cartera", "modelo"], sort=False):
        g = g[g._e == "ok"].sort_values("fecha")
        e = g.excepcion.values.astype(int)
        pof = bt.kupiec_pof(e, p)
        ind = bt.christoffersen_ind(e)
        az = basel.acerbi_szekely_z2(g.realizado.values, g.VaR.values, g.ES.values,
                                     nivel=nivel, n_boot=5_000, rng=rng)
        razon, lo, hi = ic_razon(e, nivel, rng=rng)

        cob = (0.8 <= razon <= 1.2) and pof["p_valor"] > 0.05
        indep = ind["p_valor"] > 0.05
        mag = az["p_valor"] > 0.05
        est = lo <= 1.0 <= hi

        filas.append({
            "cartera": c, "modelo": m, "razon": razon, "ic_bajo": lo, "ic_alto": hi,
            "cobertura": cob, "independencia": indep, "magnitud": mag,
            "estabilidad": est, "puntaje": int(cob) + int(indep) + int(mag) + int(est),
        })
    return pd.DataFrame(filas)


def resumen(sc: pd.DataFrame) -> pd.DataFrame:
    """Aggregate across portfolios. Max score is 4 criteria × 4 portfolios."""
    return (sc.groupby("modelo")
              .agg(razon=("razon", "mean"), ic_bajo=("ic_bajo", "mean"),
                   ic_alto=("ic_alto", "mean"), cobertura=("cobertura", "sum"),
                   independencia=("independencia", "sum"), magnitud=("magnitud", "sum"),
                   estabilidad=("estabilidad", "sum"), puntaje=("puntaje", "sum"))
              .sort_values("puntaje", ascending=False))


if __name__ == "__main__":
    from src import risk

    df = pd.read_csv("data/walkforward.csv", parse_dates=["fecha"])
    sc = scorecard(df)
    r = resumen(sc)
    r.to_csv("data/scorecard.csv")

    print("CRITERIOS DECLARADOS\n")
    for k, v in CRITERIOS.items():
        print(f"  {k:15s} {v}")

    print("\n\nSCORECARD — sumas sobre 4 carteras, máximo 16\n")
    print(r.round(3).to_string())

    print("\n\nSOLAPAMIENTO DE INTERVALOS (cartera min_var)\n")
    mv = sc[sc.cartera == "min_var"].set_index("modelo").loc[list(risk.REGISTRO)]
    for m, row in mv.iterrows():
        marca = "contiene 1.0" if row.ic_bajo <= 1 <= row.ic_alto else ""
        print(f"  {m:16s} {row.razon:5.2f}  IC95 [{row.ic_bajo:.2f}, {row.ic_alto:.2f}]  {marca}")
    print("\n  Los intervalos que se solapan no distinguen modelos: el ranking puntual "
          "\n  sugiere más precisión de la que 39 excepciones esperadas permiten.")
