"""Optimización de cartera: mín-CVaR, Markowitz y risk parity.

Solo posiciones largas y capital totalmente invertido (sum w = 1, w >= 0) en las
tres. Sin costos de transacción — declarado como limitación en el README.
"""

import cvxpy as cp
import numpy as np

BETA = 0.99


# ─────────────────────────────────────────────────────────────────────────────
# mín-CVaR — Rockafellar & Uryasev (2000)
#
# CVaR_β(w) = min_α [ α + 1/((1−β)S) · Σ_s (L_s(w) − α)^+ ]
#
# con L_s(w) = −w'r_s la pérdida del escenario s. El máximo con cero es convexo
# y lineal a trozos, así que el conjunto es un programa lineal una vez que se
# introducen las variables auxiliares u_s ≥ L_s − α, u_s ≥ 0 — cosa que cvxpy
# hace solo al ver cp.pos().
#
# Lo elegante: α no es un parámetro sino una variable, y en el óptimo α* es el
# VaR al nivel β. Minimizas CVaR y el VaR sale de regalo.
# ─────────────────────────────────────────────────────────────────────────────


def min_cvar(R: np.ndarray, beta: float = BETA, ret_min: float | None = None,
             solver: str | None = None) -> dict:
    """R: escenarios (S, n) de retornos. Devuelve pesos, α (= VaR) y CVaR."""
    S, n = R.shape
    w = cp.Variable(n)
    alpha = cp.Variable()

    perdidas = -R @ w
    cvar = alpha + cp.sum(cp.pos(perdidas - alpha)) / ((1 - beta) * S)

    restr = [cp.sum(w) == 1, w >= 0]
    if ret_min is not None:
        restr.append(R.mean(axis=0) @ w >= ret_min)

    prob = cp.Problem(cp.Minimize(cvar), restr)
    prob.solve(solver=solver)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"mín-CVaR no resolvió: {prob.status}")

    return {"w": np.asarray(w.value).ravel(), "alpha": float(alpha.value),
            "cvar": float(prob.value), "status": prob.status}


def min_varianza(Sigma: np.ndarray, mu: np.ndarray | None = None,
                 ret_min: float | None = None, solver: str | None = None) -> dict:
    """Markowitz. Sin ret_min es la cartera de mínima varianza global."""
    n = len(Sigma)
    w = cp.Variable(n)
    restr = [cp.sum(w) == 1, w >= 0]
    if ret_min is not None:
        if mu is None:
            raise ValueError("ret_min requiere mu")
        restr.append(mu @ w >= ret_min)

    prob = cp.Problem(cp.Minimize(cp.quad_form(w, cp.psd_wrap(Sigma))), restr)
    prob.solve(solver=solver)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"mín-varianza no resolvió: {prob.status}")

    return {"w": np.asarray(w.value).ravel(), "var": float(prob.value),
            "status": prob.status}


def risk_parity(Sigma: np.ndarray, solver: str | None = None) -> dict:
    """Contribución al riesgo igual para todos los activos.

    La formulación directa (igualar w_i·(Σw)_i) no es convexa. La de Spinu y
    Maillard sí, y su solución normalizada es la misma cartera:

        min  ½·w'Σw − (1/n)·Σ log(w_i),   w > 0     →     w / Σw

    El término logarítmico empuja los pesos lejos de cero y es lo que fuerza la
    igualdad de contribuciones en el óptimo.
    """
    n = len(Sigma)
    w = cp.Variable(n, pos=True)
    obj = 0.5 * cp.quad_form(w, cp.psd_wrap(Sigma)) - cp.sum(cp.log(w)) / n
    prob = cp.Problem(cp.Minimize(obj))
    prob.solve(solver=solver)
    if prob.status not in ("optimal", "optimal_inaccurate"):
        raise RuntimeError(f"risk parity no resolvió: {prob.status}")

    wv = np.asarray(w.value).ravel()
    return {"w": wv / wv.sum(), "status": prob.status}


def contribuciones_riesgo(w: np.ndarray, Sigma: np.ndarray) -> np.ndarray:
    """RC_i = w_i·(Σw)_i / σ_p. Suman a σ_p por la identidad de Euler."""
    sig = np.sqrt(w @ Sigma @ w)
    return w * (Sigma @ w) / sig


def metricas(w: np.ndarray, R: np.ndarray, beta: float = BETA) -> dict:
    """Perfil empírico de la cartera sobre los escenarios dados."""
    r = R @ w
    var = -np.quantile(r, 1 - beta)
    cola = r[r <= -var]
    return {
        "ret_medio": r.mean(),
        "vol": r.std(ddof=1),
        "VaR": var,
        "CVaR": -cola.mean() if len(cola) else np.nan,
    }


if __name__ == "__main__":
    import time

    import pandas as pd

    from src import data, merton

    rets = data.log_returns()
    params = {c: merton.calibrar(rets[c].values) for c in rets.columns}
    rng = np.random.default_rng(3)

    R = merton.escenarios(rets, params, 20_000, rng)
    Sigma = np.cov(rets.values.T, ddof=1)

    t0 = time.perf_counter()
    cv = min_cvar(R)
    t_cvar = time.perf_counter() - t0
    mv = min_varianza(Sigma)
    rp = risk_parity(Sigma)

    tabla = pd.DataFrame(
        {"min_CVaR": cv["w"], "min_var": mv["w"], "risk_parity": rp["w"]},
        index=rets.columns,
    )
    print(f"Pesos (20,000 escenarios Merton, β={BETA})\n")
    print((tabla * 100).round(2).to_string())

    print(f"\n\nPerfil sobre los mismos escenarios (diario):\n")
    perfil = pd.DataFrame({k: metricas(tabla[k].values, R) for k in tabla})
    print((perfil * 100).round(3).to_string())

    print(f"\n\nLP resuelto en {t_cvar:.2f}s | 180 rebalanceos ≈ {t_cvar*180/60:.1f} min")
