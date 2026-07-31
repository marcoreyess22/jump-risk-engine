"""Reproducibility manifest: hashes, cut-off dates, parameters, seeds, versions.

Answers one question: *is the table in the README the one this code produces
from this data?* Anyone can regenerate the manifest and compare hashes.

    python -m src.provenance            print the manifest
    python -m src.provenance --write    write it to data/manifest.json
    python -m src.provenance --check    compare against the stored manifest
"""

import argparse
import hashlib
import json
import platform
import sys
from importlib import metadata
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
MANIFIESTO = RAIZ / "data" / "manifest.json"

# Files whose content defines a result. Derived outputs are included when they
# exist; their absence is recorded rather than treated as an error, because the
# 15 MB walk-forward is deliberately not versioned.
ARCHIVOS = [
    "data/prices.csv",
    "data/walkforward.csv",
    "data/tabla_maestra.csv",
    "src/data.py",
    "src/merton.py",
    "src/optimize.py",
    "src/risk.py",
    "src/backtest.py",
    "src/basel.py",
    "src/plots.py",
    "tests/test_core.py",
]

PAQUETES = ["numpy", "pandas", "scipy", "matplotlib", "cvxpy", "yfinance"]


def sha256(p: Path) -> str:
    h = hashlib.sha256()
    with p.open("rb") as f:
        for bloque in iter(lambda: f.read(1 << 20), b""):
            h.update(bloque)
    return h.hexdigest()


def hashes() -> dict:
    out = {}
    for rel in ARCHIVOS:
        p = RAIZ / rel
        out[rel] = {"sha256": sha256(p), "bytes": p.stat().st_size} if p.exists() \
            else {"sha256": None, "bytes": None, "nota": "ausente (derivado, regenerable)"}
    return out


def parametros() -> dict:
    """Key parameters, read from the modules so they cannot drift from the code."""
    from src import backtest, basel, data, merton, risk

    return {
        "universo": data.TICKERS,
        "inicio_datos": data.START,
        "lambda_saltos_por_dia": merton.LAMBDA_DEFAULT,
        "lambda_saltos_por_anio": merton.LAMBDA_DEFAULT * merton.DIAS_ANIO,
        "ventana_dias": backtest.VENTANA,
        "nivel_confianza": backtest.NIVEL,
        "escenarios_var": risk.N_ESC,
        "escenarios_optimizador": backtest.N_ESC_OPT,
        "lambda_ewma": risk.LAMBDA_EWMA,
        "ventana_semaforo_dias": basel.DIAS_VENTANA,
        "semilla_walk_forward": 0,
        "modelos": list(risk.REGISTRO),
        "carteras": list(backtest.CARTERAS),
        "rebalanceo": "mensual",
        "recalculo_var": "diario",
    }


def cobertura_datos() -> dict:
    from src import data

    if not (RAIZ / "data" / "prices.csv").exists():
        return {"nota": "sin caché de precios"}
    px = data.load_prices()
    return {
        "primer_dia": str(px.index.min().date()),
        "ultimo_dia": str(px.index.max().date()),
        "dias": int(len(px)),
        "activos": list(px.columns),
    }


def entorno() -> dict:
    versiones = {}
    for p in PAQUETES:
        try:
            versiones[p] = metadata.version(p)
        except metadata.PackageNotFoundError:
            versiones[p] = None
    return {
        "python": platform.python_version(),
        "plataforma": platform.platform(),
        "paquetes": versiones,
    }


def manifiesto() -> dict:
    return {
        "proyecto": "jump-risk-engine",
        "alcance": (
            "Estudio comparativo de modelos VaR/ES fuera de muestra. "
            "No estima requerimientos de capital regulatorio."
        ),
        "cobertura_datos": cobertura_datos(),
        "parametros": parametros(),
        "entorno": entorno(),
        "hashes": hashes(),
    }


def comparar(actual: dict, guardado: dict) -> list[str]:
    difs = []
    for clave in ("parametros", "cobertura_datos"):
        for k, v in actual[clave].items():
            if guardado.get(clave, {}).get(k) != v:
                difs.append(f"{clave}.{k}: {guardado.get(clave, {}).get(k)!r} → {v!r}")
    for rel, h in actual["hashes"].items():
        g = guardado.get("hashes", {}).get(rel, {})
        if g.get("sha256") != h["sha256"]:
            difs.append(f"hash {rel}: {str(g.get('sha256'))[:12]} → {str(h['sha256'])[:12]}")
    return difs


def main() -> None:
    ap = argparse.ArgumentParser(description="Manifiesto de reproducibilidad")
    ap.add_argument("--write", action="store_true", help="escribe data/manifest.json")
    ap.add_argument("--check", action="store_true", help="compara contra el guardado")
    a = ap.parse_args()

    m = manifiesto()

    if a.check:
        if not MANIFIESTO.exists():
            print("no hay manifiesto guardado; corre --write primero")
            sys.exit(1)
        difs = comparar(m, json.loads(MANIFIESTO.read_text()))
        if difs:
            print(f"DIFIERE del manifiesto guardado ({len(difs)} diferencias):")
            for d in difs:
                print(f"  · {d}")
            sys.exit(1)
        print("COINCIDE con el manifiesto guardado")
        return

    texto = json.dumps(m, indent=2, ensure_ascii=False)
    if a.write:
        MANIFIESTO.parent.mkdir(parents=True, exist_ok=True)
        MANIFIESTO.write_text(texto + "\n")
        print(f"escrito en {MANIFIESTO.relative_to(RAIZ)}")
    print(texto)


if __name__ == "__main__":
    main()
