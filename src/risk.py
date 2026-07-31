"""Registro de modelos de VaR/ES.

CONVENCIÓN: VaR y ES se devuelven como PÉRDIDAS POSITIVAS. Un VaR de 0.015 son
1.5 puntos porcentuales de pérdida. Siempre ES >= VaR, porque ES promedia las
pérdidas peores que el VaR.

FIRMA ÚNICA — la decisión de arquitectura de la Ruta 7:

    modelo(rets, w, nivel, rng) -> (VaR, ES)

    rets  : (T, n) retornos de los activos en la ventana
    w     : (n,)   pesos de la cartera, fijos desde el último rebalanceo
    nivel : float  nivel de confianza, p. ej. 0.99
    rng   : generador; lo ignoran los modelos deterministas

Los modelos univariados agregan la cartera internamente con rets @ w. Los de
Monte Carlo necesitan la estructura conjunta — es justo el punto: el riesgo de
cola de una CARTERA depende de cómo saltan los activos JUNTOS, no de cómo salta
cada uno por su lado.

backtest.py itera sobre REGISTRO. Añadir una especificación es añadir una función
y una entrada al diccionario; el bucle de backtest no se toca. El Acto 2 vive de
esta propiedad.
"""

import numpy as np
from scipy import stats

from src import merton

N_ESC = 20_000


def _var_es_empirico(r: np.ndarray, nivel: float) -> tuple[float, float]:
    """VaR y ES a partir de una muestra de retornos. Base común de todo el módulo."""
    var = -np.quantile(r, 1 - nivel)
    cola = r[r <= -var]
    return float(var), float(-cola.mean()) if len(cola) else float(var)


# ── Especificaciones ─────────────────────────────────────────────────────────


def historico(rets, w, nivel, rng=None):
    """Simulación histórica. Sin supuestos de distribución, pero no puede
    producir una pérdida peor que la peor ya observada en la ventana."""
    return _var_es_empirico(rets @ w, nivel)


def normal(rets, w, nivel, rng=None):
    """Paramétrico gaussiano, en forma cerrada.

    Para L = -r con r ~ N(mu, sigma^2):
        VaR = -mu + sigma*z_beta
        ES  = -mu + sigma*phi(z_beta)/(1-beta)
    """
    r = rets @ w
    mu, sd = r.mean(), r.std(ddof=1)
    z = stats.norm.ppf(nivel)
    return float(-mu + sd * z), float(-mu + sd * stats.norm.pdf(z) / (1 - nivel))


def mc_gbm(rets, w, nivel, rng):
    """Monte Carlo gaussiano multivariado: media y covarianza de la ventana.

    Es el contrafactual central del proyecto — misma estructura de correlación
    que Merton, sin tercer ni cuarto cumulante.
    """
    mu = rets.mean(axis=0)
    L = np.linalg.cholesky(np.cov(rets.T, ddof=1) + 1e-14 * np.eye(rets.shape[1]))
    esc = mu + rng.standard_normal((N_ESC, rets.shape[1])) @ L.T
    return _var_es_empirico(esc @ w, nivel)


_CACHE_CALIB: dict = {}


def _calibrar_ventana(rets: np.ndarray, lam: float):
    """Calibración memoizada de las marginales de una ventana.

    La calibración depende solo de la ventana, NO de los pesos, pero el walk-
    forward la pide una vez por cada (cartera, modelo de Merton) — 64 veces al
    día donde bastan 8. La caché es de tamaño 1 porque el bucle avanza día a día
    y nunca vuelve a una ventana anterior.
    """
    clave = (rets.shape, hash(rets.tobytes()), lam)
    if clave not in _CACHE_CALIB:
        _CACHE_CALIB.clear()
        n = rets.shape[1]
        _CACHE_CALIB[clave] = {i: merton.calibrar(rets[:, i], lam) for i in range(n)}
    return _CACHE_CALIB[clave]


def mc_merton(rets, w, nivel, rng, sistemico=True, lam=merton.LAMBDA_DEFAULT):
    """Monte Carlo con jump-diffusion y salto sistémico.

    Calibra las marginales sobre la ventana, arma los escenarios conjuntos y
    agrega. Si la calibración resulta infactible para esta ventana (λ demasiado
    alta para su varianza), degrada a mc_gbm en vez de tumbar el backtest.
    """
    import pandas as pd

    try:
        params = _calibrar_ventana(rets, lam)
    except ValueError as err:
        # Sustituir un modelo por otro a mitad de un backtest corrompe la
        # comparación entre modelos. Nunca ocurre con este universo y λ=0.05
        # (verificado sobre todas las ventanas), pero si ocurriera tiene que
        # verse: un aviso, no una sustitución muda.
        import warnings
        warnings.warn(f"mc_merton degradó a mc_gbm en esta ventana: {err}", stacklevel=2)
        return mc_gbm(rets, w, nivel, rng)

    esc = merton.escenarios(pd.DataFrame(rets), params, N_ESC, rng, sistemico=sistemico)
    return _var_es_empirico(esc @ w, nivel)


def mc_merton_idio(rets, w, nivel, rng):
    """Merton con saltos independientes. Aísla cuánto aporta el salto sistémico."""
    return mc_merton(rets, w, nivel, rng, sistemico=False)


# ─────────────────────────────────────────────────────────────────────────────
# ACTO 2 — cinco especificaciones más.
#
# Las cinco son univariadas sobre la serie de la cartera. La línea que las
# divide, y la pregunta del Acto 2: EWMA y FHS tienen volatilidad CONDICIONAL
# (la de hoy depende de lo que pasó ayer); t-Student, Cornish-Fisher y EVT no.
# El Acto 1 mostró que ningún modelo incondicional pasa la prueba de
# independencia. Si condicionar por volatilidad es el remedio, solo EWMA y FHS
# deberían mejorar.
# ─────────────────────────────────────────────────────────────────────────────

LAMBDA_EWMA = 0.94  # RiskMetrics, dato diario


def _sigma_ewma(r: np.ndarray, lam: float = LAMBDA_EWMA) -> np.ndarray:
    """Serie de volatilidad condicional. sigma2_t usa datos hasta t-1."""
    s2 = np.empty(len(r))
    s2[0] = r.var(ddof=1)
    for t in range(1, len(r)):
        s2[t] = lam * s2[t - 1] + (1 - lam) * r[t - 1] ** 2
    return np.sqrt(s2)


def ewma(rets, w, nivel, rng=None, lam=LAMBDA_EWMA):
    """RiskMetrics: gaussiano con varianza que decae exponencialmente.

    Media cero, como es estándar en la industria a horizonte diario. La
    volatilidad de mañana se proyecta desde la de hoy y el choque de hoy.
    """
    r = rets @ w
    s2 = _sigma_ewma(r, lam)[-1] ** 2
    sd = np.sqrt(lam * s2 + (1 - lam) * r[-1] ** 2)  # proyección a t+1
    z = stats.norm.ppf(nivel)
    return float(sd * z), float(sd * stats.norm.pdf(z) / (1 - nivel))


def t_student(rets, w, nivel, rng=None):
    """Paramétrico t de Student, con nu ajustado por máxima verosimilitud.

    Mismo esqueleto que el normal pero con colas de ley de potencias. ES en
    forma cerrada:  ES = -loc + scale * f(t_q)/(1-beta) * (nu + t_q^2)/(nu - 1)
    """
    r = rets @ w
    nu, loc, scale = stats.t.fit(r)
    nu = max(nu, 2.05)  # ES infinito si nu <= 1; varianza infinita si nu <= 2
    tq = stats.t.ppf(1 - nivel, nu)
    var = -(loc + scale * tq)
    es = -loc + scale * stats.t.pdf(tq, nu) / (1 - nivel) * (nu + tq**2) / (nu - 1)
    return float(var), float(es)


def cornish_fisher(rets, w, nivel, rng=None):
    """Expansión de Cornish-Fisher: corrige el cuantil normal por skew y kurtosis.

    z_cf = z + (z^2-1)S/6 + (z^3-3z)K/24 - (2z^3-5z)S^2/36

    No hay forma cerrada del ES, así que se integra el cuantil sobre la cola.
    Es una expansión asintótica: con skew o kurtosis grandes puede dejar de ser
    monótona, y entonces el cuantil pierde sentido. Se reporta, no se esconde.
    """
    r = rets @ w
    mu, sd = r.mean(), r.std(ddof=1)
    # bias=False da los estimadores corregidos, idénticos a los de pandas.
    S = float(stats.skew(r, bias=False))
    K = float(stats.kurtosis(r, bias=False))

    def z_cf(p):
        z = stats.norm.ppf(p)
        return (z + (z**2 - 1) * S / 6 + (z**3 - 3 * z) * K / 24
                - (2 * z**3 - 5 * z) * S**2 / 36)

    var = -(mu + sd * z_cf(1 - nivel))
    ps = np.linspace(1e-6, 1 - nivel, 2000)
    es = -(mu + sd * np.trapezoid(z_cf(ps), ps) / (1 - nivel))
    return float(var), float(es)


def fhs(rets, w, nivel, rng=None, lam=LAMBDA_EWMA):
    """Filtered Historical Simulation.

    Combina lo mejor de dos mundos: la FORMA empírica de la distribución (sin
    suponer normalidad) y la ESCALA condicional de hoy (sin suponer que la
    volatilidad de 2008 aplica en 2017). Se estandarizan los retornos por su
    volatilidad EWMA y se reescalan los residuos por la volatilidad vigente.
    """
    r = rets @ w
    sd = _sigma_ewma(r, lam)
    z = r / sd
    sd_next = np.sqrt(lam * sd[-1] ** 2 + (1 - lam) * r[-1] ** 2)
    return _var_es_empirico(z * sd_next, nivel)


def evt(rets, w, nivel, rng=None, q_umbral=0.95):
    """Teoría de valores extremos: Pareto generalizada sobre los excesos.

    Solo modela la cola, que es lo único que importa para el VaR. Por el teorema
    de Pickands-Balkema-de Haan, los excesos sobre un umbral alto convergen a una
    GPD sea cual sea la distribución original.

    VaR = u + (sigma/xi)*[ ((n/Nu)*(1-beta))^(-xi) - 1 ]
    ES  = VaR/(1-xi) + (sigma - xi*u)/(1-xi)          para xi < 1

    El umbral se fija en el percentil 95 y se declara. Elegirlo "óptimamente" es
    un tema de tesis y no mejora el resultado.
    """
    L = -(rets @ w)
    n = len(L)
    u = np.quantile(L, q_umbral)
    exc = L[L > u] - u
    if len(exc) < 30:
        return _var_es_empirico(rets @ w, nivel)

    xi, _, beta_g = stats.genpareto.fit(exc, floc=0)
    nu_ratio = n / len(exc)

    if abs(xi) < 1e-6:
        var = u + beta_g * np.log(nu_ratio * (1 - nivel) ** -1)
    else:
        var = u + (beta_g / xi) * ((nu_ratio * (1 - nivel)) ** -xi - 1)

    if xi < 1:
        es = var / (1 - xi) + (beta_g - xi * u) / (1 - xi)
    else:
        # Con xi >= 1 la GPD tiene media infinita: el ES no existe. Nunca ocurre
        # con estos datos (xi observado entre -0.17 y +0.53), pero inventar un
        # múltiplo del VaR sería fabricar un número. Se recurre al ES empírico,
        # que es finito y observable.
        _, es = _var_es_empirico(rets @ w, nivel)
        es = max(es, var)
    return float(var), float(es)


REGISTRO = {
    # Acto 1 — incondicionales
    "historico": historico,
    "normal": normal,
    "mc_gbm": mc_gbm,
    "mc_merton": mc_merton,
    "mc_merton_idio": mc_merton_idio,
    # Acto 2 — sin volatilidad condicional
    "t_student": t_student,
    "cornish_fisher": cornish_fisher,
    "evt": evt,
    # Acto 2 — CON volatilidad condicional
    "ewma": ewma,
    "fhs": fhs,
}

CONDICIONALES = {"ewma", "fhs"}


def evaluar_todos(rets, w, nivel, rng, modelos=None) -> dict[str, tuple[float, float]]:
    return {k: REGISTRO[k](rets, w, nivel, rng) for k in (modelos or REGISTRO)}


if __name__ == "__main__":
    import time

    import pandas as pd

    from src import data, optimize as opt

    rets = data.log_returns()
    ventana = rets.values[-1000:]
    rng = np.random.default_rng(1)

    w = opt.min_varianza(np.cov(ventana.T, ddof=1))["w"]

    filas, tiempos = {}, {}
    for nombre in REGISTRO:
        t0 = time.perf_counter()
        var, es = REGISTRO[nombre](ventana, w, 0.99, rng)
        tiempos[nombre] = time.perf_counter() - t0
        filas[nombre] = {"VaR_99": var, "ES_99": es, "ES/VaR": es / var,
                         "ms": tiempos[nombre] * 1000}

    print("Ventana de 1,000 días, cartera de mínima varianza, nivel 99%\n")
    print(pd.DataFrame(filas).T.round(4).to_string())

    total_dia = sum(tiempos.values())
    print(f"\nCosto por día: {total_dia*1000:.0f} ms  →  "
          f"3,900 días ≈ {total_dia*3900/60:.1f} min de walk-forward")
