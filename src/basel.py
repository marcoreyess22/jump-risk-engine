"""Semáforo de Basilea y backtest de Expected Shortfall.

La pieza que traduce estadística en dinero: el marco de backtesting de Basilea
clasifica un modelo según sus excepciones en 250 días hábiles y le asigna un
multiplicador de capital. Reprobar el backtest no es un demérito académico —
encarece el capital regulatorio de forma directa y calculable.
"""

import numpy as np
import pandas as pd

DIAS_VENTANA = 250

# Excepciones en 250 días → (zona, multiplicador). Marco de backtesting de
# Basilea para VaR al 99%; el multiplicador escala el requerimiento de capital.
TABLA = [
    (4, "verde", 3.00),
    (5, "amarilla", 3.40),
    (6, "amarilla", 3.50),
    (7, "amarilla", 3.65),
    (8, "amarilla", 3.75),
    (9, "amarilla", 3.85),
]
ROJA = ("roja", 4.00)


def zona(n_excepciones: int) -> tuple[str, float]:
    """Clasificación de una ventana de 250 días."""
    for tope, z, mult in TABLA:
        if n_excepciones <= tope:
            return z, mult
    return ROJA


def ventanas_basilea(exc: np.ndarray, paso: int = DIAS_VENTANA) -> pd.DataFrame:
    """Trocea la serie de excepciones en ventanas disjuntas de 250 días.

    Disjuntas y no rodantes: es como supervisa un regulador, y evita contar la
    misma crisis quince veces.
    """
    exc = np.asarray(exc)
    n = len(exc) // paso
    filas = []
    for i in range(n):
        x = int(exc[i * paso:(i + 1) * paso].sum())
        z, m = zona(x)
        filas.append({"ventana": i, "excepciones": x, "zona": z, "multiplicador": m})
    return pd.DataFrame(filas)


def resumen_capital(exc: np.ndarray, paso: int = DIAS_VENTANA) -> dict:
    v = ventanas_basilea(exc, paso)
    if v.empty:
        return {}
    return {
        "ventanas": len(v),
        "verde": int((v.zona == "verde").sum()),
        "amarilla": int((v.zona == "amarilla").sum()),
        "roja": int((v.zona == "roja").sum()),
        "mult_medio": float(v.multiplicador.mean()),
        "mult_max": float(v.multiplicador.max()),
    }


# ── Backtest de Expected Shortfall ───────────────────────────────────────────


def acerbi_szekely_z2(realizado, var, es, nivel=0.99, n_boot=20_000, rng=None):
    """Test 2 de Acerbi-Székely (2014), incondicional.

        Z2 = 1 + (1/(N*(1-beta))) * sum_t [ X_t * 1{X_t < -VaR_t} / ES_t ]

    Bajo H0 (el ES está bien especificado) E[Z2] = 0. Z2 < 0 indica que el ES
    subestima la pérdida en la cola. El valor crítico no es analítico: se obtiene
    por simulación bajo la hipótesis nula, remuestreando qué días son excepción.

    Importa porque Basilea III (FRTB) trasladó la métrica de capital de VaR a ES,
    y un ES sin validar no sirve para nada.
    """
    rng = rng or np.random.default_rng(0)
    x, v, e = map(np.asarray, (realizado, var, es))
    N, p = len(x), 1 - nivel

    ind = x < -v
    z2 = 1 + (x[ind] / e[ind]).sum() / (N * p) if ind.any() else 1.0

    # DISTRIBUCIÓN NULA. Debe generarse desde el modelo, no desde los datos.
    #
    # Remuestrear los cocientes de cola OBSERVADOS —que fue la primera versión
    # de esto— contamina el nulo con la alternativa: si el ES está subestimado,
    # los cocientes observados son grandes y el nulo se desplaza junto con el
    # estadístico. Medido, esa versión rechazaba 4 de 100 con el ES un 50%
    # subestimado: sin potencia.
    #
    # Bajo H0 la pérdida dada excepción tiene media ES_t. Se modela el exceso
    # sobre el VaR como exponencial de media (ES_t − VaR_t): es la elección de
    # máxima entropía consistente con ese par (VaR, ES), y es un supuesto
    # declarado sobre la FORMA de la cola, no sobre su escala.
    # ES < VaR es un input incoherente y degeneraría el nulo en silencio
    # (exceso ≈ 0 ⇒ nulo sin varianza ⇒ el test deja de discriminar).
    # Ningún modelo del registro lo produce; se verifica por si se añade uno.
    if (e < v).any():
        i = int(np.argmax(e < v))
        raise ValueError(
            f"ES < VaR en la posición {i} ({e[i]:.6f} < {v[i]:.6f}): el ES promedia "
            "las pérdidas peores que el VaR y no puede ser menor. Revisa el signo del modelo."
        )

    exceso = e - v
    nulos = np.empty(n_boot)
    for b in range(n_boot):
        golpe = rng.random(N) < p
        if not golpe.any():
            nulos[b] = 1.0
            continue
        perdida = v[golpe] + rng.exponential(exceso[golpe])
        nulos[b] = 1 - (perdida / e[golpe]).sum() / (N * p)

    return {"Z2": float(z2), "p_valor": float((nulos <= z2).mean()),
            "excepciones": int(ind.sum())}


if __name__ == "__main__":
    df = pd.read_csv("data/walkforward.csv", parse_dates=["fecha"])
    rng = np.random.default_rng(0)

    filas = []
    for (c, m), g in df.groupby(["cartera", "modelo"], sort=False):
        g = g.sort_values("fecha")
        cap = resumen_capital(g.excepcion.values)
        az = acerbi_szekely_z2(g.realizado.values, g.VaR.values, g.ES.values, rng=rng)
        filas.append({"cartera": c, "modelo": m, **cap,
                      "Z2": az["Z2"], "p_ES": az["p_valor"]})

    r = pd.DataFrame(filas)
    r["ES_valido"] = np.where(r.p_ES > 0.05, "pasa", "REPRUEBA")
    print(r.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
