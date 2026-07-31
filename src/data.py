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


def load_prices(refresh: bool = False) -> pd.DataFrame:
    """Precios de cierre ajustado. Usa el caché salvo que refresh=True.

    El caché se invalida solo si su universo no coincide con TICKERS. Sin esa
    comprobación, cambiar TICKERS devolvía el universo viejo en silencio y todo
    el pipeline corría sobre activos que nadie pidió.
    """
    if CACHE.exists() and not refresh:
        px = pd.read_csv(CACHE, index_col=0, parse_dates=True)
        if list(px.columns) == TICKERS:
            return px
        print(f"  caché con universo distinto ({list(px.columns)}); redescargando")

    import yfinance as yf

    px = yf.download(TICKERS, start=START, auto_adjust=True, progress=False)["Close"]
    px = px[TICKERS].dropna()
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    px.to_csv(CACHE)
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
