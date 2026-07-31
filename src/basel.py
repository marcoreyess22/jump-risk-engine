"""Historical Basel VaR traffic light — EDUCATIONAL SIMULATION.

⚠ SCOPE — read before citing any number from this module.

This is a teaching reconstruction of the *historical* Basel backtesting traffic
light for 1-day 99% VaR (Basel II / Basel 2.5 market risk framework). It is NOT
a regulatory capital calculation and it is NOT an FRTB implementation.

What the current framework actually does, and this project does not:

  · Under FRTB, the Internal Models Approach (IMA) computes capital from
    **Expected Shortfall at 97.5%**, not from VaR, with liquidity-horizon
    scaling and stressed-period calibration.
  · 1-day VaR backtesting survives in FRTB, but at *desk* level and paired with
    P&L attribution tests (PLA) that compare risk-theoretical against
    hypothetical P&L. Neither is implemented here.
  · Actual capital involves non-modellable risk factors, the standardised
    approach floor, and supervisory add-ons — all outside this project.

Consequently every monetary figure produced here is a **proxy**, named as such
in the code (`capital_proxy`), and exists only to compare specifications against
each other on a common yardstick. It does not estimate a real requirement for
any institution.

Primary sources:
  · MAR32 — Internal models approach: backtesting and P&L attribution
    https://www.bis.org/basel_framework/chapter/MAR/32.htm
  · BCBS d457 — Minimum capital requirements for market risk (FRTB, 2019)
    https://www.bis.org/bcbs/publ/d457.htm

References for the statistics implemented here:
  · Acerbi & Székely (2014), "Backtesting Expected Shortfall", Risk Magazine.
"""

import numpy as np
import pandas as pd

DIAS_VENTANA = 250

# Historical Basel II / 2.5 traffic light for 1-day 99% VaR: exceptions in a
# 250-business-day window map to a zone and a capital multiplier. Retained for
# the comparison exercise; see the module docstring on why this is a proxy.
TABLA = [
    (4, "verde", 3.00),
    (5, "amarilla", 3.40),
    (6, "amarilla", 3.50),
    (7, "amarilla", 3.65),
    (8, "amarilla", 3.75),
    (9, "amarilla", 3.85),
]
ROJA = ("roja", 4.00)

DESCARGO = (
    "Proxy ilustrativo basado en el semáforo histórico de VaR (Basilea II/2.5). "
    "No es un cálculo de capital regulatorio ni una implementación de FRTB, que "
    "usa Expected Shortfall al 97.5% para el IMA."
)


def zona(n_excepciones: int) -> tuple[str, float]:
    """Zone and multiplier for a 250-day window under the historical table."""
    for tope, z, mult in TABLA:
        if n_excepciones <= tope:
            return z, mult
    return ROJA


def ventanas_basilea(exc: np.ndarray, paso: int = DIAS_VENTANA) -> pd.DataFrame:
    """Split the exception series into disjoint 250-day windows.

    Disjoint rather than rolling: it mirrors how a supervisor reviews, and it
    avoids counting the same crisis fifteen times.
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
    """Zone distribution and mean multiplier. See DESCARGO for scope."""
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


def capital_proxy(exc: np.ndarray, var_medio: float, nocional: float = 10e6,
                  paso: int = DIAS_VENTANA) -> float:
    """Illustrative comparison yardstick: mean multiplier × mean VaR × notional.

    NOT a regulatory capital requirement. The historical framework scales a
    60-day average VaR by the multiplier; this collapses that to a mean and
    ignores the stressed-VaR add-on, the ES-based IMA, liquidity horizons and
    every supervisory overlay. It exists to rank specifications on one axis.
    """
    r = resumen_capital(exc, paso)
    return float(r["mult_medio"] * var_medio * nocional) if r else float("nan")


# ── Expected Shortfall backtest ──────────────────────────────────────────────


def acerbi_szekely_z2(realizado, var, es, nivel=0.99, n_boot=20_000, rng=None):
    """Acerbi & Székely (2014) Test 2, unconditional.

        Z2 = 1 + (1/(N*(1-beta))) * sum_t [ X_t * 1{X_t < -VaR_t} / ES_t ]

    Under H0 (correctly specified ES) E[Z2] = 0. Z2 < 0 means the ES
    underestimates the loss in the tail. The critical value is not analytic and
    is obtained by simulating under the null.

    This matters because FRTB moved the IMA capital metric from VaR to ES, and
    an unvalidated ES is worthless.
    """
    rng = rng or np.random.default_rng(0)
    x, v, e = map(np.asarray, (realizado, var, es))
    N, p = len(x), 1 - nivel

    ind = x < -v
    z2 = 1 + (x[ind] / e[ind]).sum() / (N * p) if ind.any() else 1.0

    # NULL DISTRIBUTION. It must be generated from the model, not the data.
    #
    # Resampling the OBSERVED tail ratios — the first version of this — taints
    # the null with the alternative: if ES is understated the observed ratios
    # are large and the null shifts along with the statistic. Measured, that
    # version rejected 4 out of 100 with ES understated by 50%: no power.
    #
    # Under H0 the loss given an exception has mean ES_t. The excess over VaR is
    # modelled as exponential with mean (ES_t − VaR_t): the maximum-entropy
    # choice consistent with that pair, a declared assumption on the SHAPE of
    # the tail, not on its scale.
    if (e < v).any():
        i = int(np.argmax(e < v))
        raise ValueError(
            f"ES < VaR at position {i} ({e[i]:.6f} < {v[i]:.6f}): ES averages the "
            "losses worse than VaR and cannot be smaller. Check the model's sign."
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
    print(f"\n{DESCARGO}\n")
    df = pd.read_csv("data/walkforward.csv", parse_dates=["fecha"])
    rng = np.random.default_rng(0)

    filas = []
    for (c, m), g in df.groupby(["cartera", "modelo"], sort=False):
        g = g.sort_values("fecha")
        cap = resumen_capital(g.excepcion.values)
        az = acerbi_szekely_z2(g.realizado.values, g.VaR.values, g.ES.values, rng=rng)
        filas.append({"cartera": c, "modelo": m, **cap,
                      "capital_proxy_k": capital_proxy(g.excepcion.values,
                                                       g.VaR.mean()) / 1e3,
                      "Z2": az["Z2"], "p_ES": az["p_valor"]})

    r = pd.DataFrame(filas)
    r["ES_valido"] = np.where(r.p_ES > 0.05, "pasa", "REPRUEBA")
    print(r.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"\ncapital_proxy_k está en miles de USD sobre $10M nocional. {DESCARGO}")
