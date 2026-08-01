"""Walk-forward y pruebas de cobertura del VaR.

DISEÑO TEMPORAL (Rutas 3 y 4 del plan) — el detalle que decide si el backtest
significa algo:

  · Ventana rodante de 1,000 días. Al cierre del día t se observa hasta t.
  · Los pesos se reoptimizan SOLO al cambiar de mes.
  · El VaR se recalcula TODOS los días, sobre los pesos del último rebalanceo.
  · Excepción: la pérdida realizada en t+1 supera el VaR predicho en t.

Reoptimizar a diario invalidaría la medición: el VaR mediría una cartera que
nunca existió un día completo.
"""

import zlib

import numpy as np
import pandas as pd
from scipy import stats

from src import merton, optimize as opt, risk

VENTANA = 1000
NIVEL = 0.99
N_ESC_OPT = 20_000


# ── Pruebas de cobertura ─────────────────────────────────────────────────────


def _ll(k, n, p):
    """Log-verosimilitud binomial, con 0·log(0) = 0."""
    out = 0.0
    if k:
        out += k * np.log(p)
    if n - k:
        out += (n - k) * np.log1p(-p)
    return out


def kupiec_pof(exc: np.ndarray, p: float = 1 - NIVEL) -> dict:
    """Proportion of Failures. H0: la tasa de excepciones es exactamente p.

    LR = −2·[ ll(x, n, p) − ll(x, n, x/n) ]  ~  χ²(1)

    Detecta si hay DEMASIADAS o DEMASIADO POCAS excepciones. No dice nada sobre
    cuándo ocurren: para eso está Christoffersen.
    """
    exc = np.asarray(exc).astype(int)
    n, x = len(exc), int(exc.sum())
    # Esquema constante pase lo que pase: una función que cambia sus claves
    # según la entrada revienta río abajo justo cuando algo ya salió mal.
    if n == 0:
        return {"n": 0, "x": 0, "tasa": np.nan, "esperadas": 0.0,
                "LR": np.nan, "p_valor": np.nan}
    p_hat = x / n
    lr = -2 * (_ll(x, n, p) - _ll(x, n, p_hat)) if 0 < p_hat < 1 else -2 * (_ll(x, n, p))
    return {"n": n, "x": x, "tasa": p_hat, "esperadas": n * p,
            "LR": float(lr), "p_valor": float(stats.chi2.sf(lr, 1))}


def christoffersen_ind(exc: np.ndarray) -> dict:
    """Independencia. H0: una excepción hoy no predice una excepción mañana.

    Contrasta una cadena de Markov de dos estados contra la hipótesis de que la
    probabilidad de excepción no depende del día anterior. Es la prueba que caza
    el apelmazamiento de excepciones en crisis — un modelo puede acertar la
    frecuencia anual y aun así fallar todas sus excepciones en la misma semana.
    """
    exc = np.asarray(exc).astype(int)
    a, b = exc[:-1], exc[1:]
    n00 = int(((a == 0) & (b == 0)).sum())
    n01 = int(((a == 0) & (b == 1)).sum())
    n10 = int(((a == 1) & (b == 0)).sum())
    n11 = int(((a == 1) & (b == 1)).sum())

    if (n01 + n11) == 0 or (n00 + n01) == 0 or (n10 + n11) == 0:
        return {"LR": 0.0, "p_valor": 1.0, "pi_01": np.nan, "pi_11": np.nan}

    pi01, pi11 = n01 / (n00 + n01), n11 / (n10 + n11)
    pi = (n01 + n11) / (n00 + n01 + n10 + n11)
    ll_h1 = _ll(n01, n00 + n01, pi01) + _ll(n11, n10 + n11, pi11)
    ll_h0 = _ll(n01 + n11, n00 + n01 + n10 + n11, pi)
    lr = -2 * (ll_h0 - ll_h1)
    return {"LR": float(lr), "p_valor": float(stats.chi2.sf(lr, 1)),
            "pi_01": pi01, "pi_11": pi11}


def christoffersen_cc(exc: np.ndarray, p: float = 1 - NIVEL) -> dict:
    """Cobertura condicional: frecuencia e independencia a la vez. χ²(2)."""
    lr = kupiec_pof(exc, p)["LR"] + christoffersen_ind(exc)["LR"]
    return {"LR": float(lr), "p_valor": float(stats.chi2.sf(lr, 2))}


# ── Carteras ─────────────────────────────────────────────────────────────────


def _escenarios_ventana(win: np.ndarray, rng, n_esc: int = N_ESC_OPT) -> np.ndarray:
    df = pd.DataFrame(win)
    params = {c: merton.calibrar(win[:, i]) for i, c in enumerate(df.columns)}
    return merton.escenarios(df, params, n_esc, rng)


CARTERAS = {
    "min_cvar": lambda win, rng: opt.min_cvar(_escenarios_ventana(win, rng))["w"],
    "min_var": lambda win, rng: opt.min_varianza(np.cov(win.T, ddof=1))["w"],
    "risk_parity": lambda win, rng: opt.risk_parity(np.cov(win.T, ddof=1))["w"],
    "igual_peso": lambda win, rng: np.full(win.shape[1], 1 / win.shape[1]),
}


# ── Walk-forward ─────────────────────────────────────────────────────────────


def _flujo(semilla: int, etiqueta: str, t: int) -> np.random.Generator:
    """Generador independiente por (etiqueta, día), estable entre corridas.

    Un generador COMPARTIDO haría que el resultado de un modelo dependiera de
    cuántos modelos hay en el registro y en qué orden consumen: añadir una
    especificación corría la secuencia de todas las demás. Con λ=0.05 eso movió
    a mc_merton de 38 excepciones a 41 sin tocar el modelo.

    La etiqueta entra por CRC32 de su nombre, no por su posición, y crc32 es
    estable entre procesos — a diferencia de hash() sobre cadenas, que numpy
    aceptaría pero que Python aleatoriza por proceso.

    El día entra en la clave y la cartera NO: así todas las carteras ven los
    mismos escenarios el mismo día, lo que convierte su comparación en números
    aleatorios comunes y le quita ruido.
    """
    return np.random.default_rng([semilla, zlib.crc32(etiqueta.encode()), t])


def walk_forward(rets: pd.DataFrame, ventana: int = VENTANA, nivel: float = NIVEL,
                 semilla: int = 0, verbose: bool = True,
                 carteras: list[str] | None = None,
                 modelos: list[str] | None = None,
                 lam: float | None = None) -> pd.DataFrame:
    """Una fila por (fecha, cartera, modelo) con VaR, ES y retorno realizado.

    `carteras`, `modelos` y `lam` permiten correr un subconjunto: los estudios de
    sensibilidad no necesitan el universo completo y con él serían inviables en
    una laptop.
    """
    idx, X, T = rets.index, rets.values, len(rets)
    carteras_sel = {k: CARTERAS[k] for k in (carteras or CARTERAS)}
    modelos_sel = {k: risk.REGISTRO[k] for k in (modelos or risk.REGISTRO)}

    # λ es un parámetro del modelo, no del bucle: se inyecta por partial para no
    # mutar estado global entre corridas de sensibilidad.
    if lam is not None:
        from functools import partial
        modelos_sel = {k: (partial(v, lam=lam) if k == "mc_merton" else v)
                       for k, v in modelos_sel.items()}

    filas, w_act, mes_prev, n_reb, n_fallos = [], {}, None, 0, 0
    for t in range(ventana - 1, T - 1):
        win = X[t - ventana + 1 : t + 1]
        mes = (idx[t].year, idx[t].month)
        if mes != mes_prev:
            for cn, cf in carteras_sel.items():
                w_act[cn] = cf(win, _flujo(semilla, f"cartera:{cn}", t))
            mes_prev, n_reb = mes, n_reb + 1
            if verbose and n_reb % 24 == 0:
                print(f"  ... {idx[t].date()}  ({n_reb} rebalanceos)", flush=True)

        for cn, w in w_act.items():
            realizado = float(X[t + 1] @ w)
            for mn, mf in modelos_sel.items():
                # La trazabilidad del modelo es parte del resultado: si un
                # modelo no puede calcularse, su fila queda marcada y con NaN,
                # nunca rellenada por otro modelo.
                try:
                    var, es = mf(win, w, nivel, _flujo(semilla, mn, t))
                    estado = "ok"
                except risk.CalibrationInfeasible as err:
                    var, es, estado = np.nan, np.nan, f"calibracion_infactible: {err}"
                    n_fallos += 1
                filas.append((idx[t + 1], cn, mn, var, es, realizado, estado))

    df = pd.DataFrame(filas, columns=["fecha", "cartera", "modelo", "VaR", "ES",
                                      "realizado", "estado_modelo"])
    # Una fila sin VaR no puede declarar excepción: queda NaN, no 0. Contarla
    # como "sin excepción" sesgaría a la baja el conteo del modelo que falló.
    df["excepcion"] = np.where(df["VaR"].isna(), np.nan,
                               (-df["realizado"] > df["VaR"]).astype(float))
    if verbose:
        print(f"  {n_reb} rebalanceos, {df.fecha.nunique():,} días evaluados")
        if n_fallos:
            print(f"  ⚠ {n_fallos} evaluaciones con calibración infactible "
                  f"(marcadas en estado_modelo, VaR=NaN)")
        else:
            print("  estado_modelo: 'ok' en todas las filas")
    return df


def veredictos(df: pd.DataFrame, nivel: float = NIVEL) -> pd.DataFrame:
    """Tabla maestra: una fila por (cartera, modelo) con las tres pruebas.

    Las filas con estado_modelo distinto de 'ok' se excluyen del conteo y se
    reportan en la columna `descartadas`: un veredicto calculado sobre filas
    que el modelo no pudo producir no significa nada.
    """
    p = 1 - nivel
    out = []
    estado = df.get("estado_modelo", pd.Series("ok", index=df.index))
    for (cn, mn), g in df.assign(_estado=estado).groupby(["cartera", "modelo"],
                                                         sort=False):
        g = g.sort_values("fecha")
        n_desc = int((g["_estado"] != "ok").sum())
        g = g[g["_estado"] == "ok"]
        e = g["excepcion"].values.astype(int)
        pof, ind, cc = kupiec_pof(e, p), christoffersen_ind(e), christoffersen_cc(e, p)
        out.append({
            "cartera": cn, "modelo": mn,
            "n": pof["n"], "exc": pof["x"], "esperadas": round(pof["esperadas"], 1),
            "razon": pof["x"] / pof["esperadas"] if pof["esperadas"] else np.nan,
            "p_kupiec": pof["p_valor"], "p_ind": ind["p_valor"], "p_cc": cc["p_valor"],
            "VaR_medio": g["VaR"].mean(), "descartadas": n_desc,
        })
    r = pd.DataFrame(out)
    for col, nombre in (("p_kupiec", "Kupiec"), ("p_ind", "Indep"), ("p_cc", "CC")):
        r[nombre] = np.where(r[col] > 0.05, "pasa", "REPRUEBA")
    return r


if __name__ == "__main__":
    from src import data

    rets = data.log_returns()
    print(f"Walk-forward: ventana {VENTANA}, nivel {NIVEL}, "
          f"{len(CARTERAS)} carteras × {len(risk.REGISTRO)} modelos\n")
    df = walk_forward(rets)
    df.to_csv("data/walkforward.csv", index=False)

    v = veredictos(df)
    print("\n\nVEREDICTOS\n")
    cols = ["cartera", "modelo", "exc", "esperadas", "razon", "Kupiec", "Indep", "CC"]
    print(v[cols].to_string(index=False, float_format=lambda x: f"{x:.2f}"))
