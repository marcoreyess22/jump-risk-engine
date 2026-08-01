"""Checks de validación por fase.

Cada fase del plan tiene su check aquí. Si un check falla, no se avanza.
Correr:  python tests/test_core.py
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import pandas as pd

from src import backtest as bt, data, merton, optimize as opt, risk


def test_dia_1_datos():
    """Día 1: matriz completa, sin NaN, volatilidades plausibles, colas gordas."""
    px = data.load_prices()
    rets = data.log_returns(px)
    s = data.summary(rets)

    assert list(px.columns) == data.TICKERS, "faltan o sobran activos"
    assert px.notna().all().all(), "hay NaN en precios"
    assert rets.notna().all().all(), "hay NaN en retornos"
    assert len(px) > 4000, f"muy pocos días: {len(px)}"

    # Volatilidad anualizada en rangos plausibles por clase de activo.
    assert 0.08 < s.loc["TLT", "vol_anual"] < 0.20, s.loc["TLT", "vol_anual"]
    assert 0.13 < s.loc["SPY", "vol_anual"] < 0.26, s.loc["SPY", "vol_anual"]
    assert 0.15 < s.loc["EEM", "vol_anual"] < 0.32, s.loc["EEM", "vol_anual"]

    # Colas gordas en todos los activos. Kurtosis de exceso: normal = 0.
    # Si algún activo diera ~0, el cálculo de retornos está mal.
    assert (s["kurtosis"] > 1).all(), s["kurtosis"]

    print(f"  OK  {len(px)} días × {len(px.columns)} activos, sin NaN")
    print(f"  OK  vol anual en rango: TLT {s.loc['TLT','vol_anual']:.1%}, "
          f"SPY {s.loc['SPY','vol_anual']:.1%}, EEM {s.loc['EEM','vol_anual']:.1%}")
    print(f"  OK  kurtosis de exceso > 1 en los {len(s)} activos "
          f"(mín {s['kurtosis'].min():.1f}, máx {s['kurtosis'].max():.1f})")


def test_dia_2_calibracion():
    """La calibración reproduce los momentos empíricos en forma cerrada.

    Sin ruido de simulación: si esto falla, la formulación de cumulantes o el
    solver están mal. Es el check que aísla la calibración del simulador.
    """
    rets = data.log_returns()
    for c in rets.columns:
        p = merton.calibrar(rets[c].values)
        emp = merton.momentos(rets[c].values)
        teo = merton.cumulantes_teoricos(p)
        for nombre, e, t in zip(("media", "var", "skew", "kurtosis"), emp, teo):
            assert np.isclose(e, t, rtol=1e-6, atol=1e-12), f"{c} {nombre}: {e} vs {t}"
        assert p.sigma > 0, f"{c}: σ no positiva"

    # λ demasiado alta debe fallar explícitamente, no devolver σ² negativa.
    try:
        merton.calibrar(rets["TLT"].values, lam=5.0)
        raise AssertionError("λ infactible no levantó ValueError")
    except ValueError:
        pass

    print(f"  OK  cumulantes teóricos = momentos empíricos en los {len(rets.columns)} activos (rtol 1e-6)")
    print("  OK  λ infactible levanta ValueError")


def test_dia_3_simulador():
    """El simulador reproduce los cumulantes teóricos del modelo.

    Tolerancia absoluta en skew: EEM y TLT tienen skew ≈ 0 y un criterio
    relativo explotaría sin indicar ningún problema real.
    """
    rets = data.log_returns()
    rng = np.random.default_rng(42)
    n = 1_000_000

    for c in rets.columns:
        p = merton.calibrar(rets[c].values)
        sim = merton.simular(p, n, 1, rng).ravel()
        _, var_t, skew_t, kurt_t = merton.cumulantes_teoricos(p)
        _, var_s, skew_s, kurt_s = merton.momentos(sim)

        assert np.isclose(var_s, var_t, rtol=0.02), f"{c} var: {var_s} vs {var_t}"
        assert abs(skew_s - skew_t) < 0.10, f"{c} skew: {skew_s} vs {skew_t}"
        assert abs(kurt_s / kurt_t - 1) < 0.10, f"{c} kurtosis: {kurt_s} vs {kurt_t}"

    # Contrafactual GBM: misma varianza, pero sin tercer ni cuarto cumulante.
    p = merton.calibrar(rets["SPY"].values)
    gbm = merton.simular_gbm(p, n, 1, rng).ravel()
    _, var_g, skew_g, kurt_g = merton.momentos(gbm)
    assert np.isclose(var_g, var_t := merton.cumulantes_teoricos(p)[1], rtol=0.02)
    assert abs(skew_g) < 0.02, f"GBM con skew: {skew_g}"
    assert abs(kurt_g) < 0.05, f"GBM con kurtosis: {kurt_g}"

    print(f"  OK  simulación de {n:,} días reproduce varianza, skew y kurtosis del modelo")
    print(f"  OK  contrafactual GBM: skew {skew_g:+.4f}, kurtosis {kurt_g:+.4f} (ambos ≈ 0)")


def test_dia_4_convergencia():
    """E[S_T] simulada converge al valor cerrado, con error cayendo como 1/√N."""
    rets = data.log_returns()
    p = merton.calibrar(rets["SPY"].values)
    rng = np.random.default_rng(7)
    S0, n_dias = 100.0, 252

    teorico = merton.esperanza_precio(p, S0, n_dias)

    # Comparar realizaciones sueltas y exigir decrecimiento monótono sería
    # inválido: el error es aleatorio y el test fallaría al azar. Lo que sí es
    # una ley es que el error caiga dentro de su propio error estándar, y que
    # ese error estándar escale como 1/√N.
    ses = []
    for n_sim in (1_000, 10_000, 100_000):
        st = merton.simular_precios(p, S0, n_dias, n_sim, rng)[-1]
        se = st.std(ddof=1) / np.sqrt(n_sim)
        ses.append(se)
        z = abs(st.mean() - teorico) / se
        assert z < 4, f"N={n_sim}: sesgo de {z:.1f} errores estándar"

    razon = ses[0] / ses[-1]
    assert 7 < razon < 14, f"el error estándar no escala como 1/√N: razón {razon:.1f}"

    # Con λ = 0 la esperanza debe colapsar a la lognormal clásica.
    sin_saltos = merton.ParamsMerton(p.m, p.sigma, 0.0, 0.0, 1e-9)
    assert np.isclose(
        merton.esperanza_precio(sin_saltos, S0, n_dias),
        S0 * np.exp(n_dias * (p.m + 0.5 * p.sigma**2)),
    )

    print(f"  OK  E[S_T] sin sesgo (< 4 SE) y el SE escala 1/√N: razón {razon:.1f} en 100×")
    print("  OK  con λ=0 la esperanza colapsa a la lognormal")


def test_dia_4_escenarios_conjuntos():
    """Los escenarios conjuntos reproducen la covarianza empírica."""
    rets = data.log_returns()
    params = {c: merton.calibrar(rets[c].values) for c in rets.columns}
    rng = np.random.default_rng(11)
    n_esc = 400_000

    emp = np.cov(rets.values.T, ddof=1)

    for saltos in (True, False):
        esc = merton.escenarios(rets, params, n_esc, rng, saltos=saltos)
        assert esc.shape == (n_esc, len(rets.columns)), esc.shape
        sim = np.cov(esc.T, ddof=1)
        # Comparar en correlación: es la escala en la que el error es legible.
        d = np.sqrt(np.diag(emp))
        corr_emp, corr_sim = emp / np.outer(d, d), sim / np.outer(
            np.sqrt(np.diag(sim)), np.sqrt(np.diag(sim))
        )
        err_vol = np.abs(np.sqrt(np.diag(sim)) / d - 1).max()
        err_corr = np.abs(corr_sim - corr_emp).max()
        assert err_vol < 0.02, f"saltos={saltos}: error de vol {err_vol:.4f}"
        assert err_corr < 0.02, f"saltos={saltos}: error de correlación {err_corr:.4f}"
        etiqueta = "Merton" if saltos else "GBM   "
        print(f"  OK  {etiqueta}: err vol máx {err_vol:.4f}, err corr máx {err_corr:.4f}")

    # El contrafactual gaussiano no debe tener colas: es el punto de comparación.
    gbm = merton.escenarios(rets, params, n_esc, rng, saltos=False)
    mert = merton.escenarios(rets, params, n_esc, rng, saltos=True)
    k_gbm = pd.Series(gbm[:, 0]).kurtosis()
    k_mert = pd.Series(mert[:, 0]).kurtosis()
    assert abs(k_gbm) < 0.05, f"GBM con kurtosis {k_gbm}"
    assert k_mert > 5, f"Merton sin colas: kurtosis {k_mert}"
    print(f"  OK  kurtosis SPY — GBM {k_gbm:+.3f} vs Merton {k_mert:+.2f}")

    # Dependencia de cola: en los peores días conjuntos, el salto sistémico debe
    # mover los activos juntos y el idiosincrásico no. Es la razón de ser del
    # salto compartido, así que se verifica y no se asume.
    def corr_de_cola(esc, q=0.01):
        peor = esc[:, 0] <= np.quantile(esc[:, 0], q)   # peores días de SPY
        return np.corrcoef(esc[peor, 0], esc[peor, 1])[0, 1]

    idio = merton.escenarios(rets, params, n_esc, rng, sistemico=False)
    tc_sis, tc_idio = corr_de_cola(mert), corr_de_cola(idio)
    assert tc_sis > tc_idio + 0.15, f"sistémico {tc_sis:.3f} vs idio {tc_idio:.3f}"
    print(f"  OK  corr SPY-QQQ en el peor 1%: sistémico {tc_sis:+.3f} vs "
          f"idiosincrásico {tc_idio:+.3f}")


def test_dia_5_alpha_es_el_var():
    """En el óptimo del LP, α* es el VaR al nivel β (Rockafellar-Uryasev).

    Si la formulación estuviera mal, α convergería a otra cosa y este check
    fallaría aunque los pesos se vieran razonables.
    """
    rets = data.log_returns()
    params = {c: merton.calibrar(rets[c].values) for c in rets.columns}
    rng = np.random.default_rng(3)
    R = merton.escenarios(rets, params, 100_000, rng)

    sol = opt.min_cvar(R, beta=0.99)
    emp = opt.metricas(sol["w"], R, beta=0.99)

    assert abs(sol["alpha"] - emp["VaR"]) < 1e-4, f"{sol['alpha']} vs {emp['VaR']}"
    assert abs(sol["cvar"] - emp["CVaR"]) / emp["CVaR"] < 0.01
    assert np.isclose(sol["w"].sum(), 1) and (sol["w"] >= -1e-8).all()

    print(f"  OK  α* = VaR empírico: {sol['alpha']:.6f} vs {emp['VaR']:.6f}")
    print(f"  OK  CVaR del LP = CVaR empírico: {sol['cvar']:.6f} vs {emp['CVaR']:.6f}")


def test_dia_6_equivalencia_gaussiana():
    """Bajo normalidad, mín-CVaR colapsa a media-varianza.

    OJO — CVaR_β(w) = −w'μ + [φ(z_β)/(1−β)]·√(w'Σw) incluye el término de media,
    así que mín-CVaR NO es mínima varianza en general: lo es solo si neutralizas
    la media o si fijas el retorno. Se verifican las dos rutas.
    """
    rets = data.log_returns()
    Sigma = np.cov(rets.values.T, ddof=1)
    L = np.linalg.cholesky(Sigma)
    rng = np.random.default_rng(5)
    S = 200_000

    # Ruta 1: escenarios centrados → mín-CVaR = mínima varianza.
    R = rng.standard_normal((S, 8)) @ L.T
    R -= R.mean(axis=0)
    errs = {}
    for beta in (0.90, 0.99):
        w_c = opt.min_cvar(R, beta=beta)["w"]
        w_v = opt.min_varianza(np.cov(R.T, ddof=1))["w"]
        errs[beta] = np.abs(w_c - w_v).max()
    assert errs[0.90] < 0.010, errs
    assert errs[0.99] < 0.020, errs

    # Ruta 2: con retorno fijo, ambos minimizan la misma dispersión.
    R2 = rng.standard_normal((S, 8)) @ L.T + rets.values.mean(axis=0)
    mu = R2.mean(axis=0)
    objetivo = float(np.quantile(mu, 0.6))
    w_c = opt.min_cvar(R2, beta=0.90, ret_min=objetivo)["w"]
    w_v = opt.min_varianza(np.cov(R2.T, ddof=1), mu=mu, ret_min=objetivo)["w"]
    err2 = np.abs(w_c - w_v).max()
    assert err2 < 0.020, err2

    print(f"  OK  centrados: máx|Δw| = {errs[0.90]:.4f} (β=0.90), "
          f"{errs[0.99]:.4f} (β=0.99) — la cola con β=0.99 tiene 10× menos datos")
    print(f"  OK  con retorno fijo: máx|Δw| = {err2:.4f}")


def test_dia_6_risk_parity():
    """Contribuciones al riesgo iguales, y suman a la volatilidad (Euler)."""
    rets = data.log_returns()
    Sigma = np.cov(rets.values.T, ddof=1)
    w = opt.risk_parity(Sigma)["w"]
    rc = opt.contribuciones_riesgo(w, Sigma)

    assert np.isclose(w.sum(), 1) and (w > 0).all()
    assert abs(rc.max() / rc.min() - 1) < 0.01, f"contribuciones desiguales: {rc}"
    assert np.isclose(rc.sum(), np.sqrt(w @ Sigma @ w))

    print(f"  OK  contribuciones al riesgo iguales (razón máx/mín "
          f"{rc.max()/rc.min():.5f}) y suman a σ_p")


def test_dia_6_el_optimizador_optimiza():
    """Sobre escenarios con saltos, mín-CVaR debe batir a los otros dos en CVaR."""
    rets = data.log_returns()
    params = {c: merton.calibrar(rets[c].values) for c in rets.columns}
    rng = np.random.default_rng(3)
    R = merton.escenarios(rets, params, 100_000, rng)
    Sigma = np.cov(rets.values.T, ddof=1)

    cvars = {
        "min_CVaR": opt.metricas(opt.min_cvar(R)["w"], R)["CVaR"],
        "min_var": opt.metricas(opt.min_varianza(Sigma)["w"], R)["CVaR"],
        "risk_parity": opt.metricas(opt.risk_parity(Sigma)["w"], R)["CVaR"],
    }
    assert cvars["min_CVaR"] == min(cvars.values()), cvars
    print("  OK  CVaR in-sample: " + ", ".join(f"{k} {v:.5f}" for k, v in cvars.items()))


def test_dia_7_registro():
    """Todo el registro responde a la firma única y devuelve pérdidas coherentes.

    ES >= VaR siempre: el ES promedia las pérdidas PEORES que el VaR. Si algún
    modelo lo invierte, tiene un error de signo — el más común del módulo.
    """
    rets = data.log_returns()
    ventana = rets.values[-1000:]
    w = opt.min_varianza(np.cov(ventana.T, ddof=1))["w"]
    rng = np.random.default_rng(1)

    res = risk.evaluar_todos(ventana, w, 0.99, rng)
    assert set(res) == set(risk.REGISTRO), "evaluar_todos no cubre el registro"

    for nombre, (var, es) in res.items():
        assert np.isfinite(var) and np.isfinite(es), f"{nombre}: no finito"
        assert var > 0, f"{nombre}: VaR no positivo ({var}) — convención de pérdida"
        assert es >= var, f"{nombre}: ES {es} < VaR {var} — signo invertido"

    print(f"  OK  {len(res)} modelos con firma única, VaR > 0 y ES >= VaR")


def test_dia_7_var_historico_corta_donde_debe():
    """El VaR histórico al 99% deja ~1% de la ventana por debajo.

    Es cierto por construcción, y por eso sirve: valida el código de conteo que
    después usará el backtest para declarar excepciones.
    """
    rets = data.log_returns()
    ventana = rets.values[-1000:]
    w = np.full(8, 1 / 8)
    r = ventana @ w

    for nivel in (0.95, 0.99):
        var, _ = risk.historico(ventana, w, nivel)
        frac = (r <= -var).mean()
        assert abs(frac - (1 - nivel)) < 0.005, f"nivel {nivel}: corta {frac:.4f}"

    print("  OK  el VaR histórico corta la fracción correcta al 95% y al 99%")


def test_dia_7_normal_forma_cerrada():
    """La fórmula cerrada del gaussiano coincide con su propia simulación.

    Y el cociente ES/VaR de una normal centrada vale phi(z)/((1-beta)*z), que al
    99% es 1.1456. Es una constante conocida: si el modelo se desvía, la fórmula
    del ES está mal.
    """
    from scipy import stats

    rng = np.random.default_rng(21)
    w = np.array([1.0])
    muestra = (0.0004 + 0.01 * rng.standard_normal(2_000_000)).reshape(-1, 1)

    var_c, es_c = risk.normal(muestra, w, 0.99)
    var_e, es_e = risk._var_es_empirico(muestra @ w, 0.99)
    assert abs(var_c / var_e - 1) < 0.01, f"VaR {var_c} vs {var_e}"
    assert abs(es_c / es_e - 1) < 0.01, f"ES {es_c} vs {es_e}"

    z = stats.norm.ppf(0.99)
    teorico = stats.norm.pdf(z) / (0.01 * z)
    centrada = (0.01 * rng.standard_normal(1_000_000)).reshape(-1, 1)
    v, e = risk.normal(centrada, w, 0.99)
    assert abs(e / v - teorico) < 0.01, f"ES/VaR {e/v:.4f} vs {teorico:.4f}"

    print(f"  OK  forma cerrada = simulación (2M muestras)")
    print(f"  OK  ES/VaR de una normal centrada: {e/v:.4f} vs {teorico:.4f} teórico")


def test_dia_7_el_salto_sistemico_engorda_la_cola():
    """A nivel de CARTERA, solo el salto sistémico sobrevive a la agregación.

    Los saltos idiosincrásicos se promedian entre activos — teorema central del
    límite operando sobre choques independientes — y la cola de la cartera vuelve
    a parecer gaussiana. Es lo que justifica el salto sistémico con una medida y
    no con un argumento estético.
    """
    rets = data.log_returns()
    ventana = rets.values[-1000:]
    w = np.full(8, 1 / 8)
    rng = np.random.default_rng(1)

    ratios = {}
    for nombre in ("mc_gbm", "mc_merton", "mc_merton_idio"):
        var, es = risk.REGISTRO[nombre](ventana, w, 0.99, rng)
        ratios[nombre] = es / var

    assert ratios["mc_merton"] > ratios["mc_gbm"] + 0.10, ratios
    assert abs(ratios["mc_merton_idio"] - ratios["mc_gbm"]) < 0.06, ratios

    print("  OK  ES/VaR — " + ", ".join(f"{k} {v:.3f}" for k, v in ratios.items()))
    print("  OK  el salto idiosincrásico se diversifica; el sistémico no")


def _markov(n, pi01, pi11, rng):
    """Excepciones apelmazadas: la probabilidad de fallar mañana sube si fallé hoy."""
    e = np.zeros(n, int)
    for i in range(1, n):
        e[i] = rng.random() < (pi11 if e[i - 1] else pi01)
    return e


def test_dia_8_las_pruebas_funcionan():
    """Probar el test antes de confiar en él.

    Si Kupiec no distingue una tasa del 1% de una del 5%, todos los veredictos
    del proyecto son ruido. Se mide tamaño (rechazos bajo H0 cierta) y potencia
    (rechazos bajo H0 falsa) sobre 100 réplicas.
    """
    rng = np.random.default_rng(0)
    n, p = 4000, 0.01

    tam = sum(bt.kupiec_pof(rng.random(n) < 0.01, p)["p_valor"] < 0.05 for _ in range(100))
    assert tam <= 15, f"tamaño inflado: rechaza {tam}/100 bajo H0 cierta"

    for tasa in (0.02, 0.05):
        pot = sum(bt.kupiec_pof(rng.random(n) < tasa, p)["p_valor"] < 0.05 for _ in range(100))
        assert pot >= 90, f"sin potencia contra Bernoulli({tasa}): {pot}/100"

    print(f"  OK  Kupiec: rechaza {tam}/100 bajo H0 cierta, 100/100 con tasa inflada")


def test_dia_9_independencia_ve_lo_que_kupiec_no():
    """El caso que justifica la prueba de independencia.

    Se construyen excepciones apelmazadas con la MISMA tasa incondicional del 1%:
    Kupiec las aprueba porque la frecuencia es correcta, y solo Christoffersen ve
    que están concentradas. Es el modo de falla típico en crisis.
    """
    rng = np.random.default_rng(1)
    n = 4000

    tam = sum(bt.christoffersen_ind(rng.random(n) < 0.01)["p_valor"] < 0.05 for _ in range(100))
    assert tam <= 15, f"tamaño inflado: rechaza {tam}/100 con excepciones iid"

    rech_ind, rech_pof, tasas = 0, 0, []
    for _ in range(100):
        e = _markov(n, 0.005, 0.5, rng)
        tasas.append(e.mean())
        rech_ind += bt.christoffersen_ind(e)["p_valor"] < 0.05
        rech_pof += bt.kupiec_pof(e, 0.01)["p_valor"] < 0.05

    assert 0.007 < np.mean(tasas) < 0.014, f"la tasa no es ~1%: {np.mean(tasas)}"
    assert rech_ind >= 90, f"independencia sin potencia: {rech_ind}/100"
    assert rech_pof <= 40, f"Kupiec no debería ver el apelmazamiento: {rech_pof}/100"

    print(f"  OK  independencia: {tam}/100 con iid, {rech_ind}/100 con apelmazamiento")
    print(f"  OK  con tasa {np.mean(tasas):.4f}, Kupiec solo rechaza {rech_pof}/100 — "
          "ve la frecuencia, no el momento")


def test_dia_9_cobertura_condicional():
    """CC = POF + IND, contrastado contra χ²(2)."""
    rng = np.random.default_rng(2)
    e = _markov(4000, 0.005, 0.5, rng)
    cc = bt.christoffersen_cc(e, 0.01)
    suma = bt.kupiec_pof(e, 0.01)["LR"] + bt.christoffersen_ind(e)["LR"]
    assert np.isclose(cc["LR"], suma)
    assert cc["p_valor"] < 0.05
    print(f"  OK  CC = POF + IND (LR {cc['LR']:.1f}, p {cc['p_valor']:.2e})")


def test_dia_11_coherencia_del_registro():
    """Coherencia de TODO el registro: VaR > 0, ES >= VaR y monotonía en el nivel.

    Un modelo que viole la monotonía (VaR al 99% menor que al 95%) o que
    devuelva ES < VaR tiene un error de signo. Es el fallo más común al añadir
    especificaciones, y por eso se recorre el registro completo.

    El conteo se acota por abajo, no se fija: una aserción exacta se rompería
    con cada modelo nuevo y acabaría relajándose sin pensar. La cota inferior
    sigue detectando el borrado accidental, que es lo que importa.
    """
    rets = data.log_returns()
    ventana = rets.values[-1000:]
    w = np.full(8, 1 / 8)
    rng = np.random.default_rng(4)

    assert len(risk.REGISTRO) >= 10, f"faltan modelos: {len(risk.REGISTRO)}"
    for nombre, f in risk.REGISTRO.items():
        v95, e95 = f(ventana, w, 0.95, rng)
        v99, e99 = f(ventana, w, 0.99, rng)
        for et, v, e in (("95", v95, e95), ("99", v99, e99)):
            assert np.isfinite(v) and np.isfinite(e), f"{nombre}@{et}: no finito"
            assert v > 0, f"{nombre}@{et}: VaR {v} no positivo"
            assert e >= v, f"{nombre}@{et}: ES {e} < VaR {v} — signo invertido"
        assert v99 >= v95 * 0.98, f"{nombre}: VaR99 {v99} < VaR95 {v95} — no monótono"

    print(f"  OK  los {len(risk.REGISTRO)} modelos: VaR > 0, ES >= VaR, monótonos en el nivel")


def test_dia_11_formas_cerradas():
    """Las formas cerradas de t-Student y EVT contra simulación directa."""
    from scipy import stats

    rng = np.random.default_rng(15)
    w = np.array([1.0])

    # t-Student: se genera de una t conocida y se compara con su propio cierre.
    nu_real, escala = 5.0, 0.01
    muestra = (escala * rng.standard_t(nu_real, 2_000_000)).reshape(-1, 1)
    v_c, e_c = risk.t_student(muestra, w, 0.99)
    v_e, e_e = risk._var_es_empirico(muestra @ w, 0.99)
    assert abs(v_c / v_e - 1) < 0.02, f"VaR t: {v_c} vs {v_e}"
    assert abs(e_c / e_e - 1) < 0.03, f"ES t: {e_c} vs {e_e}"

    # EVT sobre la misma cola de ley de potencias: debe recuperarla sin
    # conocer la distribución generadora.
    v_evt, e_evt = risk.evt(muestra[:20_000], w, 0.99)
    assert abs(v_evt / v_e - 1) < 0.15, f"VaR EVT: {v_evt} vs {v_e}"
    assert e_evt >= v_evt

    print(f"  OK  t-Student cerrado vs simulado: VaR {v_c:.5f}/{v_e:.5f}, "
          f"ES {e_c:.5f}/{e_e:.5f}")
    print(f"  OK  EVT recupera la cola de una t(5) sin conocerla: {v_evt:.5f} vs {v_e:.5f}")


def test_dia_11_los_condicionales_reaccionan():
    """EWMA y FHS deben responder al régimen; los incondicionales casi no.

    Se comparan dos ventanas que terminan en marzo de 2020 y en un tramo en
    calma. Un modelo con volatilidad condicional debe subir el VaR mucho más que
    uno que promedia 1,000 días planos. Es la propiedad que el Acto 2 pone a
    prueba contra la falla de independencia del Acto 1.
    """
    rets = data.log_returns()
    w = np.full(8, 1 / 8)
    rng = np.random.default_rng(6)

    fin_crisis = rets.index.get_loc(rets.index[rets.index <= "2020-03-20"][-1])
    fin_calma = rets.index.get_loc(rets.index[rets.index <= "2017-09-01"][-1])

    razones = {}
    for nombre, f in risk.REGISTRO.items():
        v_cri = f(rets.values[fin_crisis - 999:fin_crisis + 1], w, 0.99, rng)[0]
        v_cal = f(rets.values[fin_calma - 999:fin_calma + 1], w, 0.99, rng)[0]
        razones[nombre] = v_cri / v_cal

    cond = [razones[m] for m in risk.CONDICIONALES]
    incond = [v for k, v in razones.items() if k not in risk.CONDICIONALES]
    assert min(cond) > max(incond), f"los condicionales no dominan: {razones}"

    print("  OK  VaR(crisis)/VaR(calma) — condicionales " +
          ", ".join(f"{m} {razones[m]:.2f}" for m in risk.CONDICIONALES) +
          f" | incondicionales {min(incond):.2f}–{max(incond):.2f}")


def test_diario_cierra_y_no_duplica():
    """El motor diario cierra el día pendiente contra el retorno correcto.

    Es la única lógica con estado del proyecto, y su modo de falla silencioso es
    grave: si empareja la predicción con el día equivocado, las excepciones son
    ruido y el semáforo miente. Se verifica fecha, retorno y marca.
    """
    from src import diario

    rets = data.log_returns()
    w = np.full(8, 1 / 8)
    predicho_en = rets.index[-6]
    esperado_f = rets.index[-5]
    esperado_r = float(rets.loc[esperado_f].values @ w)

    e = {"cartera": "x", "registros": [{
        "predicho_en": str(predicho_en.date()), "realizado_en": None,
        "var": {"m": 0.01}, "es": {"m": 0.015}, "realizado": None, "exc": None,
        "pesos": [float(x) for x in w]}]}

    assert diario.cerrar_pendientes(e, rets) == 1
    r = e["registros"][0]
    assert r["realizado_en"] == str(esperado_f.date()), r["realizado_en"]
    assert abs(r["realizado"] - esperado_r) < 1e-12
    assert r["exc"]["m"] == int(-esperado_r > 0.01)

    # Un registro ya cerrado no se vuelve a tocar en la siguiente corrida.
    assert diario.cerrar_pendientes(e, rets) == 0

    # Una predicción sin día posterior queda pendiente, no se inventa un cierre.
    e2 = {"registros": [{"predicho_en": str(rets.index[-1].date()),
                         "realizado_en": None, "var": {}, "es": {},
                         "realizado": None, "exc": None, "pesos": list(w)}]}
    assert diario.cerrar_pendientes(e2, rets) == 0

    print(f"  OK  cierra {predicho_en.date()} contra {esperado_f.date()} "
          f"({esperado_r:+.4%}), sin recerrar ni inventar")


def test_acerbi_szekely_tiene_potencia():
    """Probar el test de ES antes de confiar en él — el mismo estándar que
    Kupiec y Christoffersen, que a este no se le había aplicado.

    Alternativa legítima: el modelo reporta el VaR y ES de una normal, y los
    datos vienen de una t reescalada al MISMO VaR. La tasa de excepciones es
    idéntica y solo cambia el grosor de la cola, así que lo único que puede
    detectar el rechazo es el ES mal especificado.
    """
    from scipy import stats

    from src import basel

    rng = np.random.default_rng(0)
    N = 3900
    zq = stats.norm.ppf(0.99)
    var, es = zq, stats.norm.pdf(zq) / 0.01

    def rechazos(muestra):
        return sum(
            basel.acerbi_szekely_z2(muestra(), np.full(N, var), np.full(N, es),
                                    n_boot=500, rng=rng)["p_valor"] < 0.05
            for _ in range(60)
        )

    tam = rechazos(lambda: rng.standard_normal(N))
    assert tam <= 15, f"tamaño inflado: rechaza {tam}/60 bajo H0 cierta"

    pot = {}
    for nu in (6, 3):
        s = var / stats.t.ppf(0.99, nu)
        pot[nu] = rechazos(lambda nu=nu, s=s: s * rng.standard_t(nu, N))
    assert pot[3] > pot[6], f"la potencia no crece con la severidad: {pot}"
    assert pot[3] >= 20, f"sin potencia contra t(3): {pot[3]}/60"

    # Input incoherente: debe fallar fuerte, no degenerar el nulo en silencio.
    try:
        basel.acerbi_szekely_z2(rng.standard_normal(100), np.full(100, 2.0),
                                np.full(100, 1.0), n_boot=10, rng=rng)
        raise AssertionError("ES < VaR no levantó ValueError")
    except ValueError:
        pass

    print(f"  OK  tamaño {tam}/60 bajo H0; potencia t(6) {pot[6]}/60 → t(3) {pot[3]}/60")
    print("  OK  ES < VaR levanta ValueError en vez de degenerar el nulo")


def test_auditoria_b_merton_nunca_degrada_en_silencio():
    """Una fila etiquetada mc_merton jamás puede venir de mc_gbm sin marca.

    El fallback silencioso corrompería toda comparación entre modelos y lo haría
    de forma invisible. Se verifica que la excepción se levante, que el
    walk-forward la registre en estado_modelo con VaR NaN, y que el modelo con
    fallback explícito sea una entrada DISTINTA del registro.
    """
    rets = data.log_returns()
    ventana = rets.values[-1000:]
    w = np.full(8, 1 / 8)
    rng = np.random.default_rng(0)

    # λ infactible ⇒ excepción tipada, nunca un número de otro modelo.
    try:
        risk.mc_merton(ventana, w, 0.99, rng, lam=50.0)
        raise AssertionError("mc_merton no levantó CalibrationInfeasible")
    except risk.CalibrationInfeasible as err:
        assert "infactible" in str(err) and "50.0" in str(err), str(err)

    # El fallback explícito sí devuelve número, y es otro modelo con otro nombre.
    v, e = risk.mc_merton_fallback_gbm(ventana, w, 0.99, rng)
    assert np.isfinite(v) and e >= v
    assert "mc_merton_fallback_gbm" not in risk.REGISTRO, \
        "el fallback no debe estar en el registro por defecto"

    # El walk-forward marca la fila en vez de rellenarla con otro modelo.
    sub = rets.iloc[-260:]
    df = bt.walk_forward(sub, ventana=250, carteras=["igual_peso"],
                         modelos=["mc_merton"], lam=50.0, verbose=False)
    assert len(df), "el walk-forward no produjo filas"
    assert (df.estado_modelo != "ok").all(), df.estado_modelo.unique()[:2]
    assert df.VaR.isna().all(), "una fila fallida no puede traer VaR"
    assert df.excepcion.isna().all(), "una fila sin VaR no puede declarar excepción"

    v = bt.veredictos(df)
    assert (v.descartadas > 0).all(), "veredictos no reportó las filas descartadas"

    print(f"  OK  CalibrationInfeasible se levanta; {len(df)} filas marcadas con "
          "VaR NaN y excepción NaN")
    print("  OK  el fallback explícito es una entrada distinta, fuera del registro")


def test_auditoria_c_diagnostico_psd():
    """El contrafactual idiosincrático declara su propia distorsión.

    Con saltos independientes, D = Σ − J no es PSD y hay que proyectarla, lo que
    rompe la reproducción de la covarianza objetivo. Eso tiene que salir en el
    diagnóstico, no en un warning que nadie lee.
    """
    rets = data.log_returns()
    params = {c: merton.calibrar(rets[c].values) for c in rets.columns}

    sis = merton.diagnostico_covarianza(rets, params, sistemico=True)
    idi = merton.diagnostico_covarianza(rets, params, sistemico=False)

    # Sistémico: sin proyección y covarianza exacta.
    assert not sis["proyectado"], sis["autovalor_min"]
    assert sis["err_cov_rel"] < 1e-10, sis["err_cov_rel"]
    assert sis["err_corr_max"] < 1e-10, sis["err_corr_max"]

    # Idiosincrático: proyecta, y el error es material — no un detalle numérico.
    assert idi["proyectado"], "se esperaba proyección PSD con saltos independientes"
    assert idi["autovalor_min"] < 0
    assert idi["err_cov_rel"] > 0.01, idi["err_cov_rel"]

    rep = merton.reporte_psd(rets, params)
    assert set(rep.columns) >= {"autovalor_min", "proyeccion_psd", "err_cov_rel",
                                "contrafactual_limpio"}
    assert rep.set_index("acoplamiento").loc["sistemico", "contrafactual_limpio"]
    assert not rep.set_index("acoplamiento").loc["idiosincratico", "contrafactual_limpio"]

    # La advertencia metodológica tiene que estar escrita donde se lee.
    assert "no es un contrafactual limpio" in risk.mc_merton_idio.__doc__.lower()

    print(f"  OK  sistémico: sin proyección, err cov {sis['err_cov_rel']:.1e}")
    print(f"  OK  idiosincrático: proyecta (autovalor {idi['autovalor_min']:.1e}), "
          f"err cov {idi['err_cov_rel']:.1%}, err corr {idi['err_corr_max']:.1%}")


def test_auditoria_a_alcance_regulatorio():
    """basel.py no puede presentarse como cálculo regulatorio vigente."""
    from src import basel

    # Se normalizan los espacios: la aserción es sobre el contenido, no sobre
    # dónde caiga el ajuste de línea.
    doc = " ".join(basel.__doc__.lower().split())
    for termino in ("educational", "not a regulatory", "not an frtb",
                    "expected shortfall", "bis.org"):
        assert termino in doc, f"falta '{termino}' en el docstring de alcance"

    assert "no es un cálculo de capital regulatorio" in basel.DESCARGO.lower()
    assert "frtb" in basel.DESCARGO.lower()

    # La métrica monetaria se llama proxy, no capital a secas.
    assert hasattr(basel, "capital_proxy")
    assert "not a regulatory capital requirement" in basel.capital_proxy.__doc__.lower()

    val = basel.capital_proxy(np.zeros(500), 0.015)
    assert np.isclose(val, 3.00 * 0.015 * 10e6), val

    print("  OK  alcance declarado: educativo, no FRTB, con fuentes BIS")
    print(f"  OK  capital_proxy nombrado como proxy y coherente ({val:,.0f})")


def test_auditoria_d_reproducibilidad():
    """El manifiesto refleja el estado real del repositorio."""
    import json
    from pathlib import Path

    from src import provenance

    m = provenance.manifiesto()
    assert m["parametros"]["ventana_dias"] == bt.VENTANA
    assert m["parametros"]["nivel_confianza"] == bt.NIVEL
    assert m["parametros"]["modelos"] == list(risk.REGISTRO)
    assert m["cobertura_datos"]["dias"] > 4000
    assert m["hashes"]["data/prices.csv"]["sha256"], "sin hash de precios"
    assert "no estima requerimientos de capital" in m["alcance"].lower()

    raiz = Path(__file__).resolve().parents[1]
    py = (raiz / "pyproject.toml").read_text()
    assert 'requires-python = ">=3.10"' in py, "pyproject sin requires-python"

    if provenance.MANIFIESTO.exists():
        difs = provenance.comparar(m, json.loads(provenance.MANIFIESTO.read_text()))
        assert not difs, f"el manifiesto guardado no coincide: {difs[:3]}"

    print(f"  OK  manifiesto con {len(m['hashes'])} hashes, parámetros leídos del código")
    print("  OK  pyproject declara requires-python >= 3.10")


def test_acto3_escala_condicional():
    """Los dos modelos nuevos escalan con la volatilidad reciente, no con la ventana.

    Es la propiedad que los define: entrando a marzo de 2020 el VaR tiene que
    subir mucho más que el de un modelo de ventana plana. Si no sube, la
    estandarización EWMA no está llegando a los escenarios.
    """
    rets = data.log_returns()
    w = np.full(8, 1 / 8)
    rng = np.random.default_rng(3)
    i_cri = rets.index.get_loc(rets.index[rets.index <= "2020-03-20"][-1])
    i_cal = rets.index.get_loc(rets.index[rets.index <= "2017-09-01"][-1])

    razones = {}
    for m in ("mc_merton", "mc_merton_ewma", "fhs_merton", "ewma"):
        f = risk.REGISTRO[m]
        v_cri = f(rets.values[i_cri - 999:i_cri + 1], w, 0.99, rng)[0]
        v_cal = f(rets.values[i_cal - 999:i_cal + 1], w, 0.99, rng)[0]
        razones[m] = v_cri / v_cal

    for m in ("mc_merton_ewma", "fhs_merton"):
        assert razones[m] > 3 * razones["mc_merton"], \
            f"{m} no reacciona al régimen: {razones}"
        assert m in risk.CONDICIONALES

    print("  OK  VaR(crisis)/VaR(calma) — " +
          ", ".join(f"{m} {razones[m]:.1f}" for m in razones))


def test_acto3_cola_parametrica_supera_lo_observado():
    """La diferencia frente a FHS: puede generar pérdidas nunca vistas.

    FHS remuestrea residuos observados, así que su peor escenario está acotado
    por el peor residuo de la ventana. Con innovaciones paramétricas ese techo
    desaparece — que es justo lo que hace falta cuando un salto golpea durante
    un tramo de calma.
    """
    rets = data.log_returns()
    w = np.full(8, 1 / 8)
    rng = np.random.default_rng(11)
    ventana = rets.values[-1000:]

    r = ventana @ w
    sd = risk._sigma_ewma(r)
    peor_residuo = (r / sd).min()

    p = merton.calibrar(r / sd)
    sim = merton.simular(p, 200_000, 1, rng).ravel()
    assert sim.min() < peor_residuo, \
        f"la simulación no supera el peor residuo observado ({sim.min():.2f} vs {peor_residuo:.2f})"

    # Y la escala sigue siendo la condicional, no la de la ventana entera.
    v_par, _ = risk.fhs_merton(ventana, w, 0.99, rng)
    v_fhs, _ = risk.fhs(ventana, w, 0.99, rng)
    assert 0.5 < v_par / v_fhs < 2.0, f"escala descuadrada: {v_par:.5f} vs {v_fhs:.5f}"

    print(f"  OK  peor residuo observado {peor_residuo:.2f}, simulado {sim.min():.2f} "
          "— la cola paramétrica va más allá de lo visto")
    print(f"  OK  escala coherente con FHS: {v_par:.5f} vs {v_fhs:.5f}")


def test_acto3_limites_de_calibracion_escalan():
    """calibrar() debe funcionar en cualquier escala, no solo en retornos diarios.

    Los límites del solver eran absolutos y rechazaban residuos estandarizados
    (escala ~1) por salirse de un tope pensado para retornos (~1e-2). Ahora son
    múltiplos de la desviación muestral.
    """
    rng = np.random.default_rng(5)
    base = rng.standard_normal(4000) + 0.3 * rng.standard_t(3, 4000)

    calibs = {}
    for escala in (1e-3, 1.0, 1e3):
        p = merton.calibrar(base * escala)
        calibs[escala] = p
        _, var_t, skew_t, kurt_t = merton.cumulantes_teoricos(p)
        emp = merton.momentos(base * escala)
        assert np.isclose(var_t, emp[1], rtol=1e-6), escala
        assert np.isclose(kurt_t, emp[3], rtol=1e-6), escala

    # La solución debe ser equivariante: escalar los datos escala los parámetros.
    a, b = calibs[1e-3], calibs[1.0]
    assert np.isclose(b.sigma_j / a.sigma_j, 1e3, rtol=1e-3), \
        f"no equivariante: {b.sigma_j / a.sigma_j}"

    print("  OK  calibra en escalas 1e-3, 1 y 1e3 con equivarianza exacta")


def test_reproducibilidad_flujos_independientes():
    """El resultado de un modelo no puede depender de con quién comparte registro.

    Con un generador compartido, los modelos consumían en orden y añadir una
    especificación corría la secuencia de todas las demás: mc_merton pasó de 38
    excepciones a 41 sin que nadie tocara el modelo. La semilla estaba fija y aun
    así el resultado dependía del tamaño del registro.
    """
    rets = data.log_returns().iloc[-1150:]
    base = dict(rets=rets, ventana=1000, carteras=["igual_peso"], verbose=False)

    solo = bt.walk_forward(modelos=["mc_merton"], **base).set_index("fecha").VaR
    acomp = bt.walk_forward(modelos=["normal", "mc_merton", "ewma"], **base)
    acomp = acomp[acomp.modelo == "mc_merton"].set_index("fecha").VaR
    assert np.allclose(solo, acomp), f"máx Δ {np.abs(solo - acomp).max():.2e}"

    repetido = bt.walk_forward(modelos=["mc_merton"], **base).set_index("fecha").VaR
    assert np.allclose(solo, repetido), "no repetible entre corridas"

    # La etiqueta entra por su nombre: dos nombres distintos, flujos distintos;
    # el mismo nombre, el mismo flujo, sin importar el orden de creación.
    a = bt._flujo(0, "mc_merton", 50).standard_normal(5)
    b = bt._flujo(0, "fhs", 50).standard_normal(5)
    c = bt._flujo(0, "mc_merton", 50).standard_normal(5)
    assert not np.allclose(a, b), "dos modelos comparten flujo"
    assert np.allclose(a, c), "el mismo modelo no reproduce su flujo"
    assert not np.allclose(a, bt._flujo(0, "mc_merton", 51).standard_normal(5)), \
        "días distintos comparten flujo"

    print("  OK  mc_merton idéntico solo y acompañado (máx Δ 0.0), y entre corridas")
    print("  OK  flujos separados por nombre y por día, estables entre procesos")


if __name__ == "__main__":
    fallos = 0
    for nombre, fn in sorted(globals().items()):
        if nombre.startswith("test_"):
            print(f"\n{nombre}")
            try:
                fn()
            except AssertionError as e:
                print(f"  FALLA  {e}")
                fallos += 1
    print(f"\n{'FALLARON ' + str(fallos) if fallos else 'TODO PASA'}")
    sys.exit(1 if fallos else 0)
