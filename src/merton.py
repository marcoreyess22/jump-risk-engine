"""Calibración y simulación del jump-diffusion de Merton.

Retorno logarítmico diario (Δt = 1 día, todo en unidades diarias):

    X = m + σ·Z + Σ_{i=1}^{N} Y_i ,    N ~ Poisson(λ),  Y_i ~ N(μ_J, σ_J²)

Cumulantes: la parte de difusión solo aporta a κ1 y κ2; la de saltos aporta a
todos vía κ_n = λ·E[Yⁿ].

    κ1 = m + λ·μ_J
    κ2 = σ² + λ·(μ_J² + σ_J²)
    κ3 = λ·(μ_J³ + 3·μ_J·σ_J²)
    κ4 = λ·(μ_J⁴ + 6·μ_J²·σ_J² + 3·σ_J⁴)

IDENTIFICACIÓN — la decisión de modelado de esta fase:
Son 5 parámetros (m, σ, λ, μ_J, σ_J) contra 4 momentos. λ NO queda identificada
por los momentos 2 a 4: distintos λ ajustan los mismos momentos igual de bien.
Usar κ6 la identificaría, pero un sexto momento muestral es ruido puro.

La salida es fijar λ, resolver (μ_J, σ_J) exactamente de κ3 y κ4, obtener σ² por
residuo de κ2, y reportar la sensibilidad a λ. Es una elección declarada, no un
resultado — y así se documenta en el README.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.optimize import least_squares

LAMBDA_DEFAULT = 0.05  # saltos/día ≈ 12.6/año
DIAS_ANIO = 252


@dataclass
class ParamsMerton:
    m: float        # deriva diaria de la parte de difusión
    sigma: float    # vol diaria de difusión
    lam: float      # intensidad de saltos (saltos/día)
    mu_j: float     # media del salto en log
    sigma_j: float  # dispersión del salto en log

    @property
    def var_saltos(self) -> float:
        return self.lam * (self.mu_j**2 + self.sigma_j**2)

    @property
    def frac_var_saltos(self) -> float:
        """Fracción de la varianza total explicada por saltos."""
        return self.var_saltos / (self.sigma**2 + self.var_saltos)

    def as_dict(self) -> dict:
        return {
            "m": self.m,
            "sigma": self.sigma,
            "lam": self.lam,
            "mu_j": self.mu_j,
            "sigma_j": self.sigma_j,
            "vol_dif_anual": self.sigma * np.sqrt(DIAS_ANIO),
            "saltos_anio": self.lam * DIAS_ANIO,
            "frac_var_saltos": self.frac_var_saltos,
        }


def momentos(r: np.ndarray) -> tuple[float, float, float, float]:
    """(media, varianza, skew, kurtosis de exceso) muestrales."""
    s = pd.Series(np.asarray(r).ravel())
    return s.mean(), s.var(ddof=1), s.skew(), s.kurtosis()


def _residuos(p, var, skew, exkurt, lam):
    """Desajuste en skew y kurtosis de exceso estandarizados.

    Se estandariza por la varianza empírica en vez de usar residuos relativos
    sobre κ3: TLT y EEM tienen skew ≈ 0 y un residuo relativo explotaría.
    """
    mu_j, sigma_j = p
    k3 = lam * (mu_j**3 + 3 * mu_j * sigma_j**2)
    k4 = lam * (mu_j**4 + 6 * mu_j**2 * sigma_j**2 + 3 * sigma_j**4)
    return [k3 / var**1.5 - skew, k4 / var**2 - exkurt]


def calibrar(r: np.ndarray, lam: float = LAMBDA_DEFAULT) -> ParamsMerton:
    """Método de momentos con λ fija. Levanta ValueError si λ es infactible."""
    mean, var, skew, exkurt = momentos(r)

    # Semilla analítica: si μ_J es chico frente a σ_J, entonces κ4 ≈ 3λσ_J⁴
    # y κ3 ≈ 3λμ_Jσ_J². Despejando ambas se obtiene un arranque muy cercano.
    sigma_j0 = max((exkurt * var**2 / (3 * lam)) ** 0.25, 1e-4)
    mu_j0 = skew * var**1.5 / (3 * lam * sigma_j0**2)

    sol = least_squares(
        _residuos,
        x0=[mu_j0, sigma_j0],
        args=(var, skew, exkurt, lam),
        bounds=([-0.5, 1e-5], [0.5, 0.5]),
    )
    mu_j, sigma_j = sol.x

    var_saltos = lam * (mu_j**2 + sigma_j**2)
    if var_saltos >= var:
        raise ValueError(
            f"λ={lam} infactible: los saltos explicarían {var_saltos/var:.1%} "
            "de la varianza y σ² saldría negativa. Baja λ."
        )

    return ParamsMerton(
        m=mean - lam * mu_j,
        sigma=np.sqrt(var - var_saltos),
        lam=lam,
        mu_j=mu_j,
        sigma_j=sigma_j,
    )


def calibrar_panel(rets: pd.DataFrame, lam: float = LAMBDA_DEFAULT) -> pd.DataFrame:
    return pd.DataFrame(
        {c: calibrar(rets[c].values, lam).as_dict() for c in rets.columns}
    ).T


def cumulantes_teoricos(p: ParamsMerton) -> tuple[float, float, float, float]:
    """(media, varianza, skew, kurtosis de exceso) del modelo, en forma cerrada.

    Permite validar la calibración sin ruido de simulación: si la calibración es
    correcta, esto reproduce los momentos empíricos a precisión de máquina.
    """
    k1 = p.m + p.lam * p.mu_j
    k2 = p.sigma**2 + p.lam * (p.mu_j**2 + p.sigma_j**2)
    k3 = p.lam * (p.mu_j**3 + 3 * p.mu_j * p.sigma_j**2)
    k4 = p.lam * (p.mu_j**4 + 6 * p.mu_j**2 * p.sigma_j**2 + 3 * p.sigma_j**4)
    return k1, k2, k3 / k2**1.5, k4 / k2**2


def simular(p: ParamsMerton, n_dias: int, n_sim: int, rng) -> np.ndarray:
    """Retornos log simulados, forma (n_dias, n_sim).

    La suma de N normales iid es normal con media N·μ_J y desviación √N·σ_J,
    así que el proceso de Poisson compuesto se genera sin ningún bucle.
    """
    forma = (n_dias, n_sim)
    difusion = p.m + p.sigma * rng.standard_normal(forma)
    n_saltos = rng.poisson(p.lam, forma)
    saltos = n_saltos * p.mu_j + np.sqrt(n_saltos) * p.sigma_j * rng.standard_normal(forma)
    return difusion + saltos


def simular_gbm(p: ParamsMerton, n_dias: int, n_sim: int, rng) -> np.ndarray:
    """Contrafactual: misma media y varianza total, sin saltos. Skew y kurtosis 0."""
    media = p.m + p.lam * p.mu_j
    vol = np.sqrt(p.sigma**2 + p.var_saltos)
    return media + vol * rng.standard_normal((n_dias, n_sim))


# ─────────────────────────────────────────────────────────────────────────────
# Día 4 — trayectorias de precio y escenarios conjuntos
#
# MEDIDA: todo lo que sigue vive bajo la medida real (P), calibrada a retornos
# históricos. NO es la medida neutral al riesgo (Q) del pricing de opciones: ahí
# la deriva se reemplaza por r y la esperanza descontada da el precio justo.
# Aquí queremos la distribución que el mundo realmente genera, porque el objeto
# del proyecto es el VaR, no el precio de un derivado. Confundirlas es el error
# clásico al pasar de un notebook de pricing a uno de riesgo.
# ─────────────────────────────────────────────────────────────────────────────


def esperanza_precio(p: ParamsMerton, S0: float, n_dias: int) -> float:
    """E[S_T] en forma cerrada.

    E[e^X] = exp(m + σ²/2 + λ(e^{μ_J + σ_J²/2} − 1))

    El término de saltos es la función generatriz de momentos del Poisson
    compuesto. Con λ = 0 colapsa a la esperanza lognormal de siempre.
    """
    k = np.exp(p.mu_j + 0.5 * p.sigma_j**2) - 1
    return S0 * np.exp(n_dias * (p.m + 0.5 * p.sigma**2 + p.lam * k))


def simular_precios(p: ParamsMerton, S0: float, n_dias: int, n_sim: int, rng,
                    saltos: bool = True) -> np.ndarray:
    """Trayectorias de precio, forma (n_dias + 1, n_sim), arrancando en S0."""
    sim = simular if saltos else simular_gbm
    r = sim(p, n_dias, n_sim, rng)
    precios = np.empty((n_dias + 1, n_sim))
    precios[0] = S0
    precios[1:] = S0 * np.exp(np.cumsum(r, axis=0))
    return precios


def _vectores(rets: pd.DataFrame, params: dict[str, ParamsMerton]):
    cols = list(rets.columns)
    ps = [params[c] for c in cols]
    lams = np.array([p.lam for p in ps])
    if not np.allclose(lams, lams[0]):
        raise ValueError("el salto sistémico exige la misma λ en todos los activos")
    return (
        np.array([p.m for p in ps]),
        np.array([p.mu_j for p in ps]),
        np.array([p.sigma_j for p in ps]),
        float(lams[0]),
    )


def cov_saltos(rets: pd.DataFrame, params: dict[str, ParamsMerton],
               sistemico: bool = True) -> np.ndarray:
    """Covarianza aportada por los saltos.

    SALTO SISTÉMICO (por defecto). Un único Poisson N ~ Poisson(λ) compartido por
    todos los activos; dado N, el salto del activo i es la suma de N normales de
    media μ_i y dispersión σ_i, correlacionadas entre activos por la matriz
    empírica C. Entonces:

        Cov(J_i, J_j) = E[N]·Cov(Y_i,Y_j) + Var(N)·μ_i·μ_j
                      = λ·(C_ij·σ_i·σ_j + μ_i·μ_j)

    Truco de identificación: como λ es la misma en todos los activos (la fijamos
    globalmente), compartir la cuenta de Poisson NO altera ninguna marginal —
    cada activo conserva exactamente su Merton calibrado. Se gana dependencia de
    cola sin recalibrar nada.

    Con sistemico=False los saltos son independientes y la covarianza es
    diagonal. Se conserva porque comparar ambos regímenes es un resultado: es
    la diferencia entre un mundo donde las crisis son idiosincrásicas y uno
    donde son comunes.
    """
    _, mu, sig, lam = _vectores(rets, params)
    if not sistemico:
        return np.diag(lam * (mu**2 + sig**2))
    C = np.corrcoef(rets.values.T)
    return lam * (C * np.outer(sig, sig) + np.outer(mu, mu))


def cov_difusion(rets: pd.DataFrame, params: dict[str, ParamsMerton],
                 sistemico: bool = True) -> np.ndarray:
    """Covarianza de la difusión, por diferencia: D = Σ_total − J.

    Con saltos independientes J es diagonal, y restarle una diagonal grande a una
    Σ dominada por un factor común la vuelve indefinida — con estos 8 activos,
    correlacionados hasta 0.95, D deja de ser PSD. Con saltos sistémicos J tiene
    los mismos elementos diagonales pero también fuera de la diagonal, así que lo
    que se resta tiene la forma de Σ y la resta sí es factible.
    """
    D = np.cov(rets.values.T, ddof=1) - cov_saltos(rets, params, sistemico)
    w, V = np.linalg.eigh(D)
    if w.min() < -1e-12:
        import warnings
        warnings.warn(
            f"D no es PSD (autovalor mínimo {w.min():.2e}); se recorta a 0. "
            "Con sistemico=False es esperable: baja λ o usa saltos sistémicos.",
            stacklevel=2,
        )
        D = V @ np.diag(np.clip(w, 0, None)) @ V.T
    return D


def escenarios(rets: pd.DataFrame, params: dict[str, ParamsMerton], n_esc: int,
               rng, saltos: bool = True, sistemico: bool = True,
               D: np.ndarray | None = None) -> np.ndarray:
    """Retornos conjuntos a 1 día, forma (n_esc, n_activos).

    Es la entrada del optimizador. Con saltos=False produce el contrafactual
    gaussiano: misma media y misma covarianza total, sin tercer ni cuarto
    cumulante.
    """
    n = rets.shape[1]
    m, mu, sig, lam = _vectores(rets, params)

    if not saltos:
        media = m + lam * mu
        L = np.linalg.cholesky(np.cov(rets.values.T, ddof=1))
        return media + rng.standard_normal((n_esc, n)) @ L.T

    D = cov_difusion(rets, params, sistemico) if D is None else D
    L = np.linalg.cholesky(D + 1e-14 * np.eye(n))
    difusion = m + rng.standard_normal((n_esc, n)) @ L.T

    if sistemico:
        # Una sola cuenta compartida; tamaños correlacionados por C.
        N = rng.poisson(lam, (n_esc, 1))
        Lc = np.linalg.cholesky(np.corrcoef(rets.values.T) + 1e-14 * np.eye(n))
        W = rng.standard_normal((n_esc, n)) @ Lc.T
    else:
        N = rng.poisson(lam, (n_esc, n))
        W = rng.standard_normal((n_esc, n))

    return difusion + N * mu + np.sqrt(N) * sig * W


def sensibilidad_lambda(r: np.ndarray, lams=(0.01, 0.02, 0.05, 0.1, 0.2)) -> pd.DataFrame:
    """Cómo cambian los parámetros al mover λ. Los momentos ajustan igual de bien
    en todo el rango: es exactamente lo que significa que λ no esté identificada."""
    filas = {}
    for lam in lams:
        try:
            filas[lam] = calibrar(r, lam).as_dict()
        except ValueError:
            filas[lam] = {}
    return pd.DataFrame(filas).T


if __name__ == "__main__":
    from src import data

    rets = data.log_returns()
    print(f"λ = {LAMBDA_DEFAULT}/día ({LAMBDA_DEFAULT * DIAS_ANIO:.1f} saltos/año)\n")

    tabla = calibrar_panel(rets)
    print(
        tabla[["vol_dif_anual", "mu_j", "sigma_j", "frac_var_saltos"]]
        .rename(columns={
            "vol_dif_anual": "vol_dif_anual",
            "mu_j": "salto_medio",
            "sigma_j": "salto_sd",
            "frac_var_saltos": "%var_saltos",
        })
        .round(4)
        .to_string()
    )

    print("\n\nSensibilidad a λ (SPY):")
    print(
        sensibilidad_lambda(rets["SPY"].values)[
            ["vol_dif_anual", "mu_j", "sigma_j", "frac_var_saltos"]
        ].round(4).to_string()
    )
