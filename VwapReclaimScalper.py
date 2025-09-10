# --- VwapReclaimScalper.py ---
# Freqtrade 2025.x - SOL/USD 5m optimized (no external TA libs)

from __future__ import annotations

from datetime import timedelta, datetime
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy import (
    IStrategy,
    DecimalParameter,
    IntParameter,
    stoploss_from_open,
    merge_informative_pair,
)
from freqtrade.exchange import timeframe_to_minutes  # type: ignore
from freqtrade.persistence import Trade


class VwapReclaimScalper(IStrategy):
    """
    5m VWAP Reclaim Scalper (long-only, spot)

    Entry quality upgrades:
      - Higher TF filter: 1h uptrend (close > EMA50_1h).
      - Strong-body reclaim from below VWAP with on-bar VWAP retest.
      - Depth control (min prior depth below VWAP) + max extension above VWAP.
      - Micro-trend (EMA stack + positive EMA50 slope).
      - Volatility gate via ATR% of price.
      - Breakout confirm over recent high + chop filter (limit VWAP flips).
    """

    # -------- Core / timeframes --------
    timeframe = "5m"
    informative_timeframe = "1h"
    process_only_new_candles = True
    can_short = False

    # -------- Exits (keep tight first; don't hyperopt yet) --------
    minimal_roi = {"0": 0.007, "15": 0.004, "45": 0.001, "120": 0}
    stoploss = -0.006

    trailing_stop = True
    trailing_only_offset_is_reached = True
    trailing_stop_positive_offset = 0.006
    trailing_stop_positive = 0.003

    use_exit_signal = True
    exit_profit_only = False
    ignore_roi_if_entry_signal = False

    # Use ATR dynamic stop
    use_custom_stoploss = True

    # Need enough history for EMA50_1h (50*12=600) + buffers
    startup_candle_count: int = 650

    # -------- Hyperopt-friendly knobs (BUY space) --------
    pumpdump_threshold = DecimalParameter(0.010, 0.030, default=0.015, decimals=3, space="buy")
    vol_mult = DecimalParameter(1.05, 1.40, default=1.20, decimals=2, space="buy")
    slope_min = DecimalParameter(0.0000, 0.0020, default=0.0002, decimals=4, space="buy")

    # Prior depth below VWAP (prev close) and max extension above VWAP (current close)
    reclaim_pad = DecimalParameter(0.0010, 0.0040, default=0.0020, decimals=4, space="buy")   # >= 0.10–0.40%
    max_extension = DecimalParameter(0.0010, 0.0030, default=0.0015, decimals=4, space="buy") # <= 0.10–0.30%

    # Volatility gate (ATR% of price)
    atr_min_pct = DecimalParameter(0.0010, 0.0030, default=0.0015, decimals=4, space="buy")  # 0.10–0.30%
    atr_max_pct = DecimalParameter(0.0030, 0.0070, default=0.0050, decimals=4, space="buy")  # 0.30–0.70%

    # Candle quality & chop
    body_ratio_min = DecimalParameter(0.55, 0.85, default=0.65, decimals=2, space="buy")     # body / range
    chop_window = IntParameter(8, 14, default=10, space="buy")
    chop_max_flips = IntParameter(1, 3, default=2, space="buy")

    # ----------------------------- Protections -----------------------------
    @property
    def protections(self) -> List[Dict]:
        return [
            {"method": "StoplossGuard", "lookback_period_candles": 12, "trade_limit": 3, "stop_duration_candles": 24, "only_per_pair": True},
            {"method": "CooldownPeriod", "stop_duration_candles": 5},
            {"method": "MaxDrawdown", "lookback_period_candles": 12 * 24, "trade_limit": 1, "max_allowed_drawdown": 0.06, "stop_duration_candles": 12 * 12},
        ]

    # Make sure Freqtrade fetches informative data for all pairs
    def informative_pairs(self) -> List[tuple[str, str]]:
        return [(pair, self.informative_timeframe) for pair in self.dp.current_whitelist()]

    # ----------------------------- Helpers -----------------------------
    @staticmethod
    def _ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    @staticmethod
    def _atr_wilder(df: DataFrame, period: int = 14) -> pd.Series:
        h, l, c = df["high"], df["low"], df["close"]
        pc = c.shift(1)
        tr = pd.concat([h - l, (h - pc).abs(), (l - pc).abs()], axis=1).max(axis=1)
        return tr.ewm(alpha=1 / period, adjust=False).mean()

    @staticmethod
    def _anchored_vwap(df: DataFrame) -> pd.Series:
        tp = (df["high"] + df["low"] + df["close"]) / 3.0
        idx = df.index if isinstance(df.index, pd.DatetimeIndex) else pd.to_datetime(df["date"])
        dti = pd.DatetimeIndex(idx)
        # Robust day anchor regardless of tz-awareness
        try:
            day = dti.tz_localize(None).floor("D")
        except TypeError:
            day = dti.floor("D")
        tpv = tp * df["volume"]
        return tpv.groupby(day).cumsum() / df["volume"].groupby(day).cumsum().replace(0, np.nan)

    # ----------------------------- Indicators -----------------------------
    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        # 5m indicators
        df["ema9"] = self._ema(df["close"], 9)
        df["ema21"] = self._ema(df["close"], 21)
        df["ema50"] = self._ema(df["close"], 50)
        df["ema50_slope"] = (df["ema50"] / df["ema50"].shift(3) - 1.0)

        df["vol_ma20"] = df["volume"].rolling(20).mean()
        df["atr14"] = self._atr_wilder(df, period=14)
        df["atr_pct"] = df["atr14"] / df["close"]

        df["vwap"] = self._anchored_vwap(df)
        df["ret_15m_abs"] = (df["close"] / df["close"].shift(3) - 1.0).abs()

        # Candle quality
        df["body"] = (df["close"] - df["open"]).abs()
        df["range"] = (df["high"] - df["low"]).replace(0, np.nan)
        df["body_ratio"] = df["body"] / df["range"]
        df["green"] = df["close"] > df["open"]

        # Reclaim anatomy
        pc, pv = df["close"].shift(1), df["vwap"].shift(1)
        df["prev_below"] = pc < pv
        df["depth_pct"] = (pv - pc) / pv
        df["retest_ok"] = df["low"] <= df["vwap"]
        df["extension_pct"] = (df["close"] - df["vwap"]) / df["vwap"]

        # Chop (count flips of (close > vwap) over window)
        side = (df["close"] > df["vwap"]).astype(int)
        flips = side.diff().abs().fillna(0)
        df["flip_count"] = flips.rolling(self.chop_window.value).sum()

        # Small breakout confirm (over recent high, excluding current bar)
        df["recent_high"] = df["high"].shift(1).rolling(6).max()

        # 1h informative (EMA50 trend filter) — compute without suffix, merge, then use _1h
        inf = self.dp.get_pair_dataframe(pair=metadata["pair"], timeframe=self.informative_timeframe).copy()
        inf["ema50"] = self._ema(inf["close"], 50)   # <-- no suffix here

        df = merge_informative_pair(df, inf, self.timeframe, self.informative_timeframe, ffill=True)

        # After merge, informative cols are suffixed with "_1h"
        if "ema50_1h" not in df.columns and "ema50_1h_1h" in df.columns:
            df["ema50_1h"] = df["ema50_1h_1h"]

        df["trend_1h_up"] = (df["close_1h"] > df["ema50_1h"]).fillna(False)

        return df

    # ----------------------------- Entry / Exit -----------------------------
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["enter_long"] = 0

        depth_ok = df["prev_below"] & (df["depth_pct"] >= float(self.reclaim_pad.value))
        extension_ok = df["extension_pct"] <= float(self.max_extension.value)
        strong_body = df["green"] & (df["body_ratio"] >= float(self.body_ratio_min.value))
        retest = df["retest_ok"]

        vol_ok = df["volume"] > (df["vol_ma20"] * float(self.vol_mult.value))
        trend_ok = (df["ema9"] > df["ema21"]) & ((df["ema21"] > df["ema50"]) | (df["ema50_slope"] > float(self.slope_min.value)))
        volat_ok = (df["atr_pct"] >= float(self.atr_min_pct.value)) & (df["atr_pct"] <= float(self.atr_max_pct.value))
        sane_move = df["ret_15m_abs"] < float(self.pumpdump_threshold.value)
        chop_ok = df["flip_count"] <= int(self.chop_max_flips.value)
        breakout_ok = df["close"] > df["recent_high"]

        long_cond = (
            df["trend_1h_up"]
            & depth_ok & extension_ok & strong_body & retest
            & vol_ok & trend_ok & volat_ok & sane_move
            & chop_ok & breakout_ok
        )

        df.loc[long_cond, "enter_long"] = 1
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        df["exit_long"] = 0
        exit_long = (df["close"] < df["vwap"]) | (df["ema9"] < df["ema21"])
        df.loc[exit_long, "exit_long"] = 1
        return df

    # ----------------------------- Dynamic Stop & Timeouts -----------------------------
    def custom_stoploss(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        after_fill: bool = False,
        **kwargs,
    ) -> float | None:
        # Use analyzed DF (has our indicators)
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        if df is None or df.empty or current_rate <= 0:
            return self.stoploss

        if len(df) < self.startup_candle_count:
            return self.stoploss

        atr = df["atr14"].iloc[-1]
        if not np.isfinite(atr) or atr <= 0:
            return self.stoploss

        atr_pct = float(atr) / float(current_rate) * 0.8
        atr_pct = max(0.003, min(0.012, atr_pct))  # 0.3% .. 1.2%

        return stoploss_from_open(-atr_pct, current_profit)

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[str]:
        tf_min = timeframe_to_minutes(self.timeframe)
        age_in_candles = int((current_time - trade.open_date_utc).total_seconds() // (tf_min * 60))

        if age_in_candles >= 24 and current_profit < 0.012:  # ~120 minutes
            return "timeout_120m_low_profit"
        if age_in_candles >= 18 and current_profit < 0.003:  # ~90 minutes
            return "timeout_90m_low_profit"
        return None

    # ----------------------------- Plotting -----------------------------
    plot_config = {
        "main_plot": {
            "vwap": {"color": "orange", "fill_area": True, "fill_color": "rgba(255,165,0,0.1)"},
            "ema9": {"color": "blue"},
            "ema21": {"color": "purple"},
            "ema50": {"color": "teal"},
        },
        "subplots": {
            "ATR": {"atr14": {"color": "gray"}},
            "Volume": {"vol_ma20": {"color": "gray"}},
            "Buy Signals": {"enter_long": {"color": "green", "type": "indicator"}},
            "Exit Signals": {"exit_long": {"color": "red", "type": "indicator"}},
        },
    }