"""Corrida diaria: del backtest histórico a un motor con estado.

Lo que el backtest hace 3,923 veces de un tirón, esto lo hace una vez al día y
recuerda el resultado entre corridas. Es la única pieza que le faltaba al motor
para operar: estado persistente.

    python -m src.diario                 corrida normal
    python -m src.diario --sin-descarga  usa el caché, sin tocar la red
    python -m src.diario --sembrar       inicializa el historial desde el backtest
    python -m src.diario --cartera min_cvar --modelo mc_merton

Cada corrida hace cuatro cosas:
  1. Cierra el día anterior: compara lo realizado contra el VaR que se predijo
     y anota si hubo excepción.
  2. Reoptimiza los pesos si cambió el mes.
  3. Calibra sobre la ventana más reciente y predice el VaR de mañana.
  4. Reporta la zona del semáforo sobre las últimas 250 observaciones.

El estado vive en data/estado_diario.json. Bórralo para empezar de cero.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src import backtest as bt, basel, data, risk

ESTADO = Path(__file__).resolve().parents[1] / "data" / "estado_diario.json"
CARTERA_DEF = "min_var"
MODELO_DEF = "fhs"  # la recomendación del informe: única decente en ambos ejes


def cargar_estado(cartera: str, modelo: str) -> dict:
    if ESTADO.exists():
        e = json.loads(ESTADO.read_text())
        if e.get("cartera") == cartera:
            return e
        print(f"  aviso: el estado guardado es de '{e.get('cartera')}'; "
              f"se reinicia para '{cartera}'")
    return {"cartera": cartera, "modelo": modelo, "ultimo_rebalanceo": None,
            "pesos": None, "registros": []}


def guardar_estado(e: dict) -> None:
    ESTADO.parent.mkdir(parents=True, exist_ok=True)
    ESTADO.write_text(json.dumps(e, indent=1))


def sembrar_desde_backtest(e: dict, csv: Path) -> dict:
    """Arranca el historial con las excepciones ya medidas en el walk-forward.

    Sin esto, el semáforo necesita 250 días de operación real antes de decir
    algo. Con esto, arranca informado desde el primer día.
    """
    df = pd.read_csv(csv, parse_dates=["fecha"])
    df = df[df.cartera == e["cartera"]]
    if df.empty:
        raise SystemExit(f"el backtest no tiene la cartera '{e['cartera']}'")

    piv = df.pivot(index="fecha", columns="modelo", values="VaR").tail(400)
    pes = df.pivot(index="fecha", columns="modelo", values="ES").tail(400)
    real = df.drop_duplicates("fecha").set_index("fecha").realizado.tail(400)

    e["registros"] = [
        {"predicho_en": None, "realizado_en": str(f.date()),
         "var": {m: float(piv.loc[f, m]) for m in piv.columns},
         "es": {m: float(pes.loc[f, m]) for m in pes.columns},
         "realizado": float(real.loc[f]),
         "exc": {m: int(-real.loc[f] > piv.loc[f, m]) for m in piv.columns}}
        for f in piv.index
    ]
    print(f"  sembrado con {len(e['registros'])} días del walk-forward "
          f"({piv.index.min().date()} → {piv.index.max().date()})")
    return e


def cerrar_pendientes(e: dict, rets: pd.DataFrame) -> int:
    """Rellena los registros que ya tienen retorno realizado y marca excepciones."""
    cerrados = 0
    for r in e["registros"]:
        if r["realizado"] is not None or r["predicho_en"] is None:
            continue
        posteriores = rets.index[rets.index > pd.Timestamp(r["predicho_en"])]
        if not len(posteriores):
            continue
        f = posteriores[0]
        realizado = float(rets.loc[f].values @ np.array(r["pesos"]))
        r["realizado_en"] = str(f.date())
        r["realizado"] = realizado
        r["exc"] = {m: int(-realizado > v) for m, v in r["var"].items()}
        cerrados += 1
    return cerrados


def main() -> None:
    ap = argparse.ArgumentParser(description="Corrida diaria del motor de riesgo")
    ap.add_argument("--cartera", default=CARTERA_DEF, choices=list(bt.CARTERAS))
    ap.add_argument("--modelo", default=MODELO_DEF, choices=list(risk.REGISTRO))
    ap.add_argument("--sin-descarga", action="store_true",
                    help="usa el caché en vez de refrescar precios")
    ap.add_argument("--sembrar", action="store_true",
                    help="inicializa el historial desde data/walkforward.csv")
    a = ap.parse_args()

    print(f"\n{'═' * 62}\n  MOTOR DE RIESGO — corrida diaria\n{'═' * 62}\n")

    # ── 1. Datos
    if a.sin_descarga:
        print("  datos: caché local (--sin-descarga)")
    else:
        print("  datos: refrescando desde el proveedor...")
        data.load_prices(refresh=True)
    rets = data.log_returns()
    hoy = rets.index[-1]
    print(f"  último cierre disponible: {hoy.date()}  ({len(rets):,} días)\n")

    e = cargar_estado(a.cartera, a.modelo)
    if a.sembrar:
        e = sembrar_desde_backtest(e, ESTADO.parent / "walkforward.csv")

    ventana = rets.values[-bt.VENTANA:]
    rng = np.random.default_rng(0)

    # ── 2. Cierre del día anterior
    if e["pesos"] is not None:
        n = cerrar_pendientes(e, rets)
        ultimo = [r for r in e["registros"] if r["realizado"] is not None]
        if n and ultimo:
            r = ultimo[-1]
            hubo = r["exc"].get(a.modelo, 0)
            marca = "⚠ EXCEPCIÓN" if hubo else "sin excepción"
            print(f"  CIERRE DE {r['realizado_en']}")
            print(f"    realizado      {r['realizado']:+.2%}")
            print(f"    VaR predicho    {r['var'][a.modelo]:.2%}  ({a.modelo})   {marca}\n")

    # ── 3. Rebalanceo mensual
    mes = f"{hoy.year}-{hoy.month:02d}"
    if e["ultimo_rebalanceo"] != mes:
        print(f"  REBALANCEO — mes nuevo ({mes})")
        w = bt.CARTERAS[a.cartera](ventana, rng)
        e["pesos"], e["ultimo_rebalanceo"] = [float(x) for x in w], mes
        print("    " + "  ".join(f"{t} {p:.1%}"
                                 for t, p in zip(data.TICKERS, w) if p > 0.005) + "\n")
    else:
        print(f"  pesos vigentes desde {e['ultimo_rebalanceo']} (sin rebalanceo)\n")
    w = np.array(e["pesos"])

    # ── 4. Predicción para mañana
    print(f"  RIESGO PARA LA PRÓXIMA SESIÓN — cartera {a.cartera}, nivel 99%\n")
    print(f"    {'modelo':<16}{'VaR':>9}{'ES':>9}")
    vars_hoy, es_hoy = {}, {}
    for m, f in risk.REGISTRO.items():
        v, s = f(ventana, w, bt.NIVEL, rng)
        vars_hoy[m], es_hoy[m] = float(v), float(s)
        marca = "  ←" if m == a.modelo else ""
        print(f"    {m:<16}{v:>8.2%}{s:>9.2%}{marca}")

    # Idempotente: correrlo dos veces con los mismos datos reemplaza la
    # predicción del día en vez de duplicarla.
    e["registros"] = [r for r in e["registros"]
                      if r.get("predicho_en") != str(hoy.date())]
    e["registros"].append({
        "predicho_en": str(hoy.date()), "realizado_en": None,
        "var": vars_hoy, "es": es_hoy, "realizado": None, "exc": None,
        "pesos": [float(x) for x in w],
    })

    # ── 5. Semáforo
    cerrados = [r for r in e["registros"] if r["exc"]][-basel.DIAS_VENTANA:]
    print(f"\n  SEMÁFORO DE BASILEA — últimos {len(cerrados)} días registrados")
    if len(cerrados) < basel.DIAS_VENTANA:
        print(f"    (la ventana regulatoria son {basel.DIAS_VENTANA} días; "
              "usa --sembrar para arrancar informado)")
    if cerrados:
        print(f"\n    {'modelo':<16}{'exc':>5}{'zona':>11}{'mult':>7}")
        for m in risk.REGISTRO:
            x = sum(r["exc"].get(m, 0) for r in cerrados)
            escala = round(x * basel.DIAS_VENTANA / len(cerrados))
            z, mult = basel.zona(escala)
            icono = {"verde": "●", "amarilla": "●", "roja": "●"}[z]
            marca = "  ←" if m == a.modelo else ""
            print(f"    {m:<16}{x:>5}{icono + ' ' + z:>11}{mult:>7.2f}{marca}")
        print(f"\n    zona por conteo escalado a {basel.DIAS_VENTANA} días")

    guardar_estado(e)
    print(f"\n  estado guardado en {ESTADO.relative_to(ESTADO.parents[1])}"
          f"  ({len(e['registros'])} registros)\n")


if __name__ == "__main__":
    main()
