"""Ingesta y caché de precios.

Descarga una vez desde yfinance, guarda a CSV, y lee de disco en adelante.
El resto del proyecto nunca toca la red.
"""

from pathlib import Path

import numpy as np
import pandas as pd

TICKERS = ["SPY", "QQQ", "IWM", "EFA", "EEM", "TLT", "GLD", "DBC"]
START = "2007-01-01"
DIAS_ANIO = 252

CACHE = Path(__file__).resolve().parents[1] / "data" / "prices.csv"
# Caché del motor diario: fuera del control de versiones, para que refrescar
# datos no toque el ancla de reproducibilidad.
CACHE_VIVO = Path(__file__).resolve().parents[1] / "data" / "prices_live.csv"


def load_prices(refresh: bool = False, cache: Path | None = None) -> pd.DataFrame:
    """Precios de cierre ajustado. Usa el caché salvo que refresh=True.

    El caché se invalida solo si su universo no coincide con TICKERS. Sin esa
    comprobación, cambiar TICKERS devolvía el universo viejo en silencio y todo
    el pipeline corría sobre activos que nadie pidió.

    `cache` permite apuntar a otro archivo. Existe porque data/prices.csv es el
    ANCLA DE REPRODUCIBILIDAD: está versionado y su hash vive en el manifiesto,
    así que refrescarlo invalida en silencio todas las tablas publicadas. El
    proveedor además reajusta la historia entera al aplicar un dividendo, de
    modo que un refresco no añade un día: cambia los 4,900 anteriores.
    Los procesos que necesitan datos frescos usan su propia caché.
    """
    cache = cache or CACHE
    if cache.exists() and not refresh:
        px = pd.read_csv(cache, index_col=0, parse_dates=True)
        if list(px.columns) == TICKERS:
            return px
        print(f"  caché con universo distinto ({list(px.columns)}); redescargando")

    import yfinance as yf

    px = yf.download(TICKERS, start=START, auto_adjust=True, progress=False)["Close"]
    px = px[TICKERS].dropna()
    cache.parent.mkdir(parents=True, exist_ok=True)
    px.to_csv(cache)
    return px


def log_returns(prices: pd.DataFrame | None = None) -> pd.DataFrame:
    px = load_prices() if prices is None else prices
    return np.log(px / px.shift(1)).dropna()


def summary(rets: pd.DataFrame | None = None) -> pd.DataFrame:
    """Perfil por activo. `kurtosis` es de exceso (normal = 0)."""
    r = log_returns() if rets is None else rets
    return pd.DataFrame(
        {
            "ret_anual": r.mean() * DIAS_ANIO,
            "vol_anual": r.std() * np.sqrt(DIAS_ANIO),
            "skew": r.skew(),
            "kurtosis": r.kurtosis(),
        }
    )


if __name__ == "__main__":
    px = load_prices()
    print(f"Precios: {px.shape[0]} días × {px.shape[1]} activos")
    print(f"Rango:   {px.index.min().date()} → {px.index.max().date()}\n")
    print(summary().round(4).to_string())
