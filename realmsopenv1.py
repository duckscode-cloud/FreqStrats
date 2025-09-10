# --- realmsopenv1.py ---
from functools import reduce
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
from pandas import DataFrame

from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (
    IntParameter,
    DecimalParameter,
    BooleanParameter,
    merge_informative_pair,
)
from freqtrade.optimize.space import Categorical, Integer, SKDecimal  # nested HyperOpt spaces

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib

# Optional VWAP from `ta` – fallback if missing
try:
    from ta.volume import VolumeWeightedAveragePrice
    HAS_TA = True
except Exception:
    HAS_TA = False


# ---------- helpers ----------
def _s(df: pd.DataFrame, col: str, default=False) -> pd.Series:
    """
    Safe column accessor: return df[col] if present,
    else a Series(default) aligned to df.index.
    """
    if col in df.columns:
        return df[col]
    return pd.Series(default, index=df.index)


def supertrend(df: DataFrame, period: int = 10, multiplier: float = 3.0) -> DataFrame:
    atr = ta.ATR(df, timeperiod=period)
    hl2 = (df["high"] + df["low"]) / 2.0
    basic_ub = hl2 + multiplier * atr
    basic_lb = hl2 - multiplier * atr

    final_ub = basic_ub.copy()
    final_lb = basic_lb.copy()

    for i in range(1, len(df)):
        final_ub.iat[i] = (
            basic_ub.iat[i]
            if (basic_ub.iat[i] < final_ub.iat[i - 1]) or (df["close"].iat[i - 1] > final_ub.iat[i - 1])
            else final_ub.iat[i - 1]
        )
        final_lb.iat[i] = (
            basic_lb.iat[i]
            if (basic_lb.iat[i] > final_lb.iat[i - 1]) or (df["close"].iat[i - 1] < final_lb.iat[i - 1])
            else final_lb.iat[i - 1]
        )

    st = pd.DataFrame(index=df.index)
    st["basic_ub"] = basic_ub
    st["final_ub"] = final_ub
    st["basic_lb"] = basic_lb
    st["final_lb"] = final_lb
    return st


def ewo(series: pd.Series, fast: int = 5, slow: int = 35) -> pd.Series:
    ema_fast = ta.EMA(series, timeperiod=fast)
    ema_slow = ta.EMA(series, timeperiod=slow)
    return (ema_fast - ema_slow) / series * 100.0


# ----- robust CMF fallback -----
def _cmf_manual(df: DataFrame, n: int = 50) -> pd.Series:
    high, low, close, vol = df["high"], df["low"], df["close"], df["volume"]
    denom = (high - low).replace(0, np.nan)
    mfm = ((close - low) - (high - close)) / denom
    mfv = mfm * vol
    out = mfv.rolling(n, min_periods=n).sum() / vol.rolling(n, min_periods=n).sum()
    return out.fillna(0.0)


def _cmf(df: DataFrame, n: int = 50) -> pd.Series:
    if hasattr(qtpylib, "chaikin_money_flow"):
        return qtpylib.chaikin_money_flow(df, n=n)
    return _cmf_manual(df, n=n)


class realmsopenv1(IStrategy):
    """
    Multi-timeframe momentum / breakout + pump filters + liquidity/volatility guards,
    with BTC market regime and Supertrend oversight. Pairs are stake-quoted (e.g., USDT).

    Upgrades in this version:
      - Explicit regime_ok (1h trend + RSI + CTI + ATR% band)
      - ATR%-based dynamic stoploss (1%..6%) and position sizing
      - Maker-tilted entries via custom_entry_price()
      - Safe column access to avoid KeyError ('ispumping', etc.)
    """

    timeframe = "5m"
    informative_timeframe = "1h"
    startup_candle_count = 600  # ensure long lookbacks (e.g., r_480) have context

    # Processing / callbacks
    process_only_new_candles = True
    use_custom_stoploss = True

    # Risk (defaults; will be hyperopted)
    minimal_roi = {"0": 0.08, "60": 0.04, "240": 0.02, "720": 0}
    stoploss = -0.10

    trailing_stop = True
    trailing_stop_positive = 0.02
    trailing_stop_positive_offset = 0.055
    trailing_only_offset_is_reached = True

    # DCA-style retries (disabled - no adjust_trade_position implementation)
    position_adjustment_enable = False
    max_entry_position_adjustment = 0

    # ------------ Hyperoptable knobs (relaxed defaults to ensure signals) ------------
    rsi_buy        = IntParameter(45, 65, default=49, space="buy")
    rsi_fast_buy   = IntParameter(45, 65, default=50, space="buy")
    rsi72_buy      = IntParameter(48, 60, default=50, space="buy")
    rsi84_buy      = IntParameter(48, 60, default=50, space="buy")
    rsi112_buy     = IntParameter(48, 60, default=50, space="buy")
    rsi_1h_buy     = IntParameter(48, 60, default=50, space="buy")
    cti_1h_min     = DecimalParameter(-0.5, 0.5, default=-0.10, decimals=2, space="buy")

    # NEW: 1h ATR% regime band
    atrp_min_1h    = DecimalParameter(0.002, 0.050, default=0.003, decimals=3, space="buy")  # 0.2% .. 5%
    atrp_max_1h    = DecimalParameter(0.010, 0.080, default=0.030, decimals=3, space="buy")  # 1.0% .. 8%

    ispumping_thr          = DecimalParameter(0.02, 0.10, default=0.045, decimals=3, space="buy")
    isshortpumping_thr     = DecimalParameter(0.01, 0.05, default=0.025, decimals=3, space="buy")
    recentispumping_window = IntParameter(10, 60, default=30, space="buy")

    pct_change_min   = DecimalParameter(0.002, 0.02, default=0.003, decimals=4, space="buy")
    vol_mult_12      = DecimalParameter(0.8, 2.0, default=1.0, decimals=2, space="buy")
    vol_mult_24      = DecimalParameter(0.8, 2.0, default=1.0, decimals=2, space="buy")
    lookback_candles = IntParameter(20, 200, default=60, space="buy")
    r_480_max        = DecimalParameter(0.05, 0.50, default=0.40, decimals=2, space="buy")

    vwap_k        = DecimalParameter(1.0, 3.0, default=2.0, decimals=1, space="buy")
    bb_len        = IntParameter(18, 22, default=20, space="buy")
    bb_mult       = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space="buy")
    high_offset_2 = DecimalParameter(0.001, 0.02, default=0.002, decimals=3, space="buy")
    low_offset    = DecimalParameter(0.002, 0.03, default=0.01, decimals=3, space="buy")

    ema_vwap_min  = DecimalParameter(-0.03, 0.00, default=-0.02, decimals=3, space="buy")
    ema_vwap_max  = DecimalParameter(0.00, 0.02, default=0.015, decimals=3, space="buy")
    rel_price_max = DecimalParameter(0.00, 0.03, default=0.030, decimals=3, space="buy")

    ewo_high_thr  = DecimalParameter(2.0, 15.0, default=2.0, decimals=1, space="buy")
    ewo_low_thr   = DecimalParameter(-15.0, -2.0, default=-6.0, decimals=1, space="sell")

    btc_threshold = DecimalParameter(0.0, 0.02, default=0.000, decimals=3, space="buy")
    btc_diff_min  = DecimalParameter(-0.02, 0.02, default=-0.005, decimals=3, space="buy")

    adaptive = BooleanParameter(default=True, space="buy")

    # Toggles (hyperoptable)
    use_pump_filter  = BooleanParameter(default=True, space="buy")
    use_regime_guard = BooleanParameter(default=True, space="buy")
    use_btc_guard    = BooleanParameter(default=True, space="buy")
    use_close_15m    = BooleanParameter(default=True, space="buy")

    # -------- Protections (hyperoptable via --spaces protection) --------
    cooldown_candles       = IntParameter(1, 12, default=2, space="protection")
    use_stoploss_guard     = BooleanParameter(default=True, space="protection")
    slg_lookback_candles   = IntParameter(12, 96, default=24, space="protection")
    slg_trade_limit        = IntParameter(2, 8,  default=4,  space="protection")
    slg_stop_candles       = IntParameter(2, 24, default=4,  space="protection")
    slg_only_per_pair      = BooleanParameter(default=False, space="protection")

    use_maxdd              = BooleanParameter(default=True, space="protection")
    mdd_lookback_candles   = IntParameter(24, 288, default=96, space="protection")
    mdd_trade_limit        = IntParameter(10, 60, default=20, space="protection")
    mdd_stop_candles       = IntParameter(4, 48,  default=12, space="protection")
    mdd_allowed_drawdown   = DecimalParameter(0.05, 0.35, default=0.20, decimals=3, space="protection")

    # Optional debug (adds columns only; no logic change)
    DEBUG_DIAGNOSTICS = True

    plot_config = {
        "main_plot": {
            "ema_slow": {"color": "orange"},
            "vwap": {"color": "blue"},
            "vwap_upperband": {"color": "blue"},
            "bb_upperband2": {"color": "purple"},
            "final_ub": {"color": "red"},
        },
        "subplots": {
            "RSI": {"rsi": {"color": "green"}, "rsi_fast": {"color": "teal"}, "rsi_1h": {"color": "grey"}},
            "EWO": {"ewo": {"color": "brown"}},
            "Stretch": {"ema_vwap_diff_50": {"color": "red"}},
            "Vol": {"volume_mean_12": {"color": "grey"}, "volume_mean_24": {"color": "grey"}},
        },
    }

    # ---------- Nested HyperOpt spaces (ROI/SL/Trailing) ----------
    class HyperOpt:
        @staticmethod
        def roi_space():
            # 4-step ROI tuned for 5m breakout/squeeze behavior
            return [
                Integer(10, 180,  name="roi_t1"),
                Integer(20, 300,  name="roi_t2"),
                Integer(60, 900,  name="roi_t3"),
                SKDecimal(0.010, 0.070, decimals=3, name="roi_p1"),
                SKDecimal(0.005, 0.050, decimals=3, name="roi_p2"),
                SKDecimal(0.001, 0.030, decimals=3, name="roi_p3"),
            ]

        @staticmethod
        def generate_roi_table(params: dict) -> dict:
            t1 = int(params["roi_t1"])
            t2 = int(params["roi_t2"])
            t3 = int(params["roi_t3"])
            p1 = float(params["roi_p1"])
            p2 = float(params["roi_p2"])
            p3 = float(params["roi_p3"])
            return {
                0: p1 + p2 + p3,
                t1: p2 + p3,
                t1 + t2: p3,
                t1 + t2 + t3: 0.0,
            }

        @staticmethod
        def stoploss_space():
            # Wider space lets trend trades breathe
            return [SKDecimal(-0.200, -0.030, decimals=3, name="stoploss")]

        @staticmethod
        def trailing_space():
            # Ensure offset > positive by adding an extra delta term
            return [
                Categorical([True], name="trailing_stop"),
                SKDecimal(0.005, 0.050, decimals=3, name="trailing_stop_positive"),
                SKDecimal(0.005, 0.100, decimals=3, name="trailing_stop_positive_offset_p1"),
                Categorical([True, False], name="trailing_only_offset_is_reached"),
            ]

        @staticmethod
        def generate_trailing_params(params: dict) -> dict:
            pos = float(params["trailing_stop_positive"])
            return {
                "trailing_stop": True,
                "trailing_stop_positive": pos,
                "trailing_stop_positive_offset": pos + float(params["trailing_stop_positive_offset_p1"]),
                "trailing_only_offset_is_reached": params["trailing_only_offset_is_reached"],
            }

    # ---------- Protections property ----------
    @property
    def protections(self):
        prot = [
            {"method": "CooldownPeriod", "stop_duration_candles": int(self.cooldown_candles.value)}
        ]
        if self.use_stoploss_guard.value:
            prot.append({
                "method": "StoplossGuard",
                "lookback_period_candles": int(self.slg_lookback_candles.value),
                "trade_limit": int(self.slg_trade_limit.value),
                "stop_duration_candles": int(self.slg_stop_candles.value),
                "only_per_pair": bool(self.slg_only_per_pair.value),
            })
        if self.use_maxdd.value:
            prot.append({
                "method": "MaxDrawdown",
                "lookback_period_candles": int(self.mdd_lookback_candles.value),
                "trade_limit": int(self.mdd_trade_limit.value),
                "stop_duration_candles": int(self.mdd_stop_candles.value),
                "max_allowed_drawdown": float(self.mdd_allowed_drawdown.value),
            })
        return prot

    # ---------------------------- indicators/signals ----------------------------
    def informative_pairs(self) -> List[Tuple[str, str]]:
        pairs: List[Tuple[str, str]] = []
        if self.dp:
            for tf in [self.informative_timeframe, "15m"]:
                for p in self.dp.current_whitelist():
                    pairs.append((p, tf))
        pairs.append(("BTC/USDT", "5m"))
        pairs.append(("BTC/USDT", "1d"))
        return pairs

    def populate_indicators(self, df: DataFrame, metadata: Dict) -> DataFrame:
        pair = metadata["pair"]

        # ----- Base (5m) -----
        df["ema_slow"] = ta.EMA(df, timeperiod=100)
        df["rsi"] = ta.RSI(df, timeperiod=14)
        df["rsi_fast"] = ta.RSI(df, timeperiod=7)
        df["rsi_72"] = ta.RSI(df, timeperiod=72)
        df["rsi_84"] = ta.RSI(df, timeperiod=84)
        df["rsi_112"] = ta.RSI(df, timeperiod=112)

        ha = qtpylib.heikinashi(df)
        df["ha_high"] = ha["high"]

        df["volume_mean_12"] = df["volume"].rolling(12).mean()
        df["volume_mean_24"] = df["volume"].rolling(24).mean()

        # VWAP (+ fallback)
        if HAS_TA:
            vwap_obj = VolumeWeightedAveragePrice(
                high=df["high"], low=df["low"], close=df["close"], volume=df["volume"], window=50
            )
            df["vwap"] = vwap_obj.vwap
        else:
            tp = (df["high"] + df["low"] + df["close"]) / 3.0
            df["vwap"] = (tp * df["volume"]).rolling(50).sum() / df["volume"].rolling(50).sum()

        vstd = (df["close"] - df["vwap"]).rolling(50).std()
        df["vwap_upperband"] = df["vwap"] + self.vwap_k.value * vstd
        df["vwap_width"] = (df["vwap_upperband"] - (df["vwap"] - self.vwap_k.value * vstd)) / df["vwap"]

        df["ema_50"] = ta.EMA(df, timeperiod=50)
        df["ema_vwap_diff_50"] = (df["ema_50"] - df["vwap"]) / df["vwap"]
        df["relative_price"] = (df["close"] - df["vwap"]) / df["vwap"]

        bb = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=self.bb_len.value, stds=self.bb_mult.value)
        df["bb_upperband2"] = bb["upper"]

        st = supertrend(df, period=10, multiplier=3.0)
        df["basic_ub"] = st["basic_ub"]
        df["final_ub"] = st["final_ub"]
        df["basic_lb"] = st["basic_lb"]
        df["final_lb"] = st["final_lb"]

        df["ewo"] = ewo(df["close"], fast=5, slow=35)

        lb = self.lookback_candles.value
        df["pct_change_lb"] = (df["close"] / df["close"].shift(lb) - 1.0).fillna(0.0)
        df["r_480"] = (df["close"] / df["close"].shift(480) - 1.0).abs().fillna(0.0)

        # CMF + divergence helpers
        df["cmf"] = _cmf(df, n=50)
        df["high_roll"] = df["high"].rolling(50).max()
        df["cmf_roll"] = df["cmf"].rolling(50).max()
        df["cmf_div_slow"] = (df["high"] > df["high_roll"].shift(1)) & (df["cmf"] < df["cmf_roll"].shift(1))

        mom = ta.MOM(df, timeperiod=14)
        price_chg = df["close"] - df["close"].shift(14)
        df["momdiv_coh"] = (np.sign(mom) == np.sign(price_chg))
        df["momdiv_col"] = (np.sign(mom) != np.sign(price_chg)) & (mom.abs() < mom.abs().rolling(14).mean())

        # ----- Pump filters (5m) -----
        ret1 = (df["close"] / df["close"].shift(1) - 1.0).abs()
        ret3 = (df["close"] / df["close"].shift(3) - 1.0).abs()
        thr_pump = float(self.ispumping_thr.value)
        thr_short = float(self.isshortpumping_thr.value)
        win_recent = int(self.recentispumping_window.value)

        df["ispumping"] = (ret1 > thr_pump)
        df["isshortpumping"] = (ret3 > thr_short)
        df["ispumping_rolling"] = ret1.rolling(12, min_periods=1).max() > thr_pump
        df["recentispumping_rolling"] = ret1.rolling(win_recent, min_periods=1).max() > thr_pump

        # Normalize to boolean dtype
        for col in ["ispumping", "isshortpumping", "ispumping_rolling", "recentispumping_rolling",
                    "cmf_div_slow", "momdiv_coh", "momdiv_col"]:
            df[col] = _s(df, col, False).astype(bool)

        # ----- 15m informative (neutral names pre-merge) -----
        if self.dp:
            df15 = self.dp.get_pair_dataframe(pair=pair, timeframe="15m")
        else:
            df15 = self.resample_dataframe(df, timeframe="15m")
        df15 = df15[["date", "close"]]
        df = merge_informative_pair(df, df15, self.timeframe, "15m", ffill=True)
        if "close_15m" not in df.columns:
            cand = [c for c in df.columns if c.startswith("close") and c.endswith("_15m")]
            if cand:
                df["close_15m"] = df[cand[0]]

        # ----- 1h informative -----
        if self.dp:
            df1h = self.dp.get_pair_dataframe(pair=pair, timeframe=self.informative_timeframe)
        else:
            df1h = self.resample_dataframe(df, timeframe=self.informative_timeframe)

        df1h["ema_fast"] = ta.EMA(df1h, timeperiod=50)
        df1h["ema_slow"] = ta.EMA(df1h, timeperiod=200)
        df1h["uptrend"] = df1h["ema_fast"] > df1h["ema_slow"]
        df1h["rsi"] = ta.RSI(df1h, timeperiod=14)

        slope = ta.LINEARREG_SLOPE(df1h, timeperiod=40)
        df1h["cti_40"] = slope / (df1h["close"].rolling(40).std() + 1e-9)

        # NEW: 1h ATR% for volatility gating
        df1h["atr_1h"] = ta.ATR(df1h, timeperiod=14)
        df1h["atrp_1h"] = (df1h["atr_1h"] / df1h["close"]).clip(lower=1e-6)

        df = merge_informative_pair(
            df, df1h[["date", "uptrend", "rsi", "cti_40", "atrp_1h"]],
            self.timeframe, self.informative_timeframe, ffill=True,
        )
        # Normalize names in case suffixing differs
        for want, prefix in [("uptrend_1h", "uptrend"), ("rsi_1h", "rsi"),
                             ("cti_40_1h", "cti_40"), ("atrp_1h_1h", "atrp_1h")]:
            if want not in df.columns:
                alt = [c for c in df.columns if c.startswith(prefix) and c.endswith("_1h")]
                if alt:
                    df[want] = df[alt[0]]

        # Combined regime flag (trend + RSI + CTI + volatility band)
        df["vol_ok_1h"] = _s(df, "atrp_1h_1h").between(self.atrp_min_1h.value, self.atrp_max_1h.value)
        df["regime_ok"] = (
            (_s(df, "uptrend_1h").astype(bool) == True)
            & (_s(df, "rsi_1h") > self.rsi_1h_buy.value)
            & (_s(df, "cti_40_1h") > self.cti_1h_min.value)
            & (_s(df, "vol_ok_1h").astype(bool))
        ).astype(int)

        # ----- BTC regime -----
        if self.dp:
            btc5 = self.dp.get_pair_dataframe(pair="BTC/USDT", timeframe="5m")
            btc1d = self.dp.get_pair_dataframe(pair="BTC/USDT", timeframe="1d")
        else:
            btc5 = self.resample_dataframe(df, timeframe="5m")
            btc1d = self.resample_dataframe(df, timeframe="1d")

        btc5["btc_fast"] = ta.EMA(btc5, timeperiod=50)
        btc1d["btc_fast"] = ta.EMA(btc1d, timeperiod=50)
        df = merge_informative_pair(df, btc5[["date", "btc_fast"]], self.timeframe, "5m", ffill=True)
        df = merge_informative_pair(df, btc1d[["date", "btc_fast"]], self.timeframe, "1d", ffill=True)

        btc_ema_5m_col = "btc_fast_5m"
        btc_ema_1d_col = "btc_fast_1d"
        if btc_ema_5m_col not in df.columns or btc_ema_1d_col not in df.columns:
            cand_5m = [c for c in df.columns if c.startswith("btc_fast") and c.endswith("_5m")]
            cand_1d = [c for c in df.columns if c.startswith("btc_fast") and c.endswith("_1d")]
            if not cand_5m or not cand_1d:
                raise ValueError(f"BTC regime columns not found after merge. Columns: {list(df.columns)}")
            btc_ema_5m_col, btc_ema_1d_col = cand_5m[0], cand_1d[0]

        df["btc_5m_1d_diff"] = (df[btc_ema_5m_col] / df[btc_ema_1d_col] - 1.0).fillna(0.0)

        # Current pair vs its own 1d EMA50
        if self.dp:
            pair1d = self.dp.get_pair_dataframe(pair=pair, timeframe="1d")
        else:
            pair1d = self.resample_dataframe(df, timeframe="1d")
        pair1d["ema_fast"] = ta.EMA(pair1d, timeperiod=50)
        df = merge_informative_pair(df, pair1d[["date", "ema_fast"]], self.timeframe, "1d", ffill=True)
        df["_5m_1d_dif"] = (df["ema_50"] / df["ema_fast_1d"] - 1.0).fillna(0.0)

        # Convenience
        df["close_"] = df["close"]
        df["high_"] = df["high"]

        return df

    def populate_entry_trend(self, df: DataFrame, metadata: Dict) -> DataFrame:
        pair = metadata["pair"]
        stake_currency = self.config.get("stake_currency", "USDT")
        is_stake_currency = pair.endswith(f"/{stake_currency}") or pair.endswith(f"-{stake_currency}")

        vol_guard = (df["volume"] > (self.vol_mult_12.value * _s(df, "volume_mean_12"))) & (
            df["volume"] > (self.vol_mult_24.value * _s(df, "volume_mean_24"))
        )
        move_guard = _s(df, "pct_change_lb").abs() > self.pct_change_min.value
        extension_guard = _s(df, "r_480") < self.r_480_max.value

        # Pump filter (optional toggle) - SAFE
        pump_ok = (~_s(df, "ispumping", False)) & (~_s(df, "isshortpumping", False)) \
                  & (~_s(df, "ispumping_rolling", False)) & (~_s(df, "recentispumping_rolling", False))
        if not self.use_pump_filter.value:
            pump_ok = _s(df, "close_") > 0

        stretch_guard = (_s(df, "ema_vwap_diff_50") >= self.ema_vwap_min.value) & (_s(df, "ema_vwap_diff_50") <= self.ema_vwap_max.value)
        rel_price_guard = _s(df, "relative_price") <= self.rel_price_max.value

        # Regime (single flag)
        regime_guard = (_s(df, "regime_ok", 0).astype(int) == 1)
        if not self.use_regime_guard.value:
            regime_guard = _s(df, "close_") > 0

        # BTC regime
        btc_guard = (_s(df, "btc_5m_1d_diff") > self.btc_diff_min.value) & (_s(df, "btc_5m_1d_diff") > self.btc_threshold.value)
        if not self.use_btc_guard.value:
            btc_guard = _s(df, "close_") > 0

        rsi_align = (
            (_s(df, "rsi") > self.rsi_buy.value)
            & (_s(df, "rsi_fast") > self.rsi_fast_buy.value)
            & (_s(df, "rsi_72") > self.rsi72_buy.value)
            & (_s(df, "rsi_84") > self.rsi84_buy.value)
            & (_s(df, "rsi_112") > self.rsi112_buy.value)
        )
        ewo_ok = _s(df, "ewo") > self.ewo_high_thr.value

        upper_break = (
            (_s(df, "close_") > _s(df, "vwap_upperband"))
            | (_s(df, "close_") > _s(df, "bb_upperband2"))
            | (_s(df, "close_") > _s(df, "final_ub"))
            | (_s(df, "close_") > _s(df, "high_").rolling(self.lookback_candles.value).max() * (1 + self.high_offset_2.value))
        )
        close_15m_ok = _s(df, "close_15m") > _s(df, "close_15m").shift(1)
        if not self.use_close_15m.value:
            close_15m_ok = _s(df, "close_") > 0

        flow_ok = (~_s(df, "cmf_div_slow", False)) & (_s(df, "momdiv_coh", False)) & (~_s(df, "momdiv_col", False))

        # Adaptive gate (safe)
        if self.adaptive.value:
            vww = _s(df, "vwap_width")
            narrow = vww < vww.rolling(200).quantile(0.3)
            wide = vww > vww.rolling(200).quantile(0.7)
            rsi_tight = (_s(df, "rsi_fast") > (self.rsi_fast_buy.value + 2)) & (_s(df, "rsi") > (self.rsi_buy.value + 2))
            stretch_tight = _s(df, "relative_price") <= (self.rel_price_max.value * 0.8)
            rsi_relax = (_s(df, "rsi_fast") > (self.rsi_fast_buy.value - 2)) & (_s(df, "rsi") > (self.rsi_buy.value - 2))
            stretch_relax = _s(df, "relative_price") <= (self.rel_price_max.value * 1.15)
            adaptive_ok = (narrow & rsi_tight & stretch_tight) | (wide & rsi_relax & stretch_relax) | (~narrow & ~wide)
        else:
            adaptive_ok = _s(df, "close_") > 0

        # Combine
        conditions: List[pd.Series] = [
            is_stake_currency,
            df["volume"] > 0,
            vol_guard,
            move_guard,
            extension_guard,
            pump_ok,
            stretch_guard,
            rel_price_guard,
            regime_guard,
            btc_guard,
            rsi_align,
            ewo_ok,
            upper_break,
            close_15m_ok,
            flow_ok,
            adaptive_ok,
        ]

        if self.DEBUG_DIAGNOSTICS:
            df["g_is_stake_currency"] = bool(is_stake_currency)
            df["g_vol_guard"] = vol_guard.astype(int)
            df["g_move_guard"] = move_guard.astype(int)
            df["g_extension_guard"] = extension_guard.astype(int)
            df["g_pump_ok"] = pump_ok.astype(int)
            df["g_stretch_guard"] = stretch_guard.astype(int)
            df["g_rel_price_guard"] = rel_price_guard.astype(int)
            df["g_regime_ok"] = (_s(df, "regime_ok", 0).astype(int) == 1).astype(int)
            df["g_btc_guard"] = btc_guard.astype(int)
            df["g_rsi_align"] = rsi_align.astype(int)
            df["g_ewo_ok"] = ewo_ok.astype(int)
            df["g_upper_break"] = upper_break.astype(int)
            df["g_close_15m_ok"] = close_15m_ok.astype(int)
            df["g_flow_ok"] = flow_ok.astype(int)
            df["g_adaptive_ok"] = adaptive_ok.astype(int)
            all_mask = reduce(lambda a, b: a & b, [c if isinstance(c, pd.Series) else bool(c) for c in conditions])
            df["enter_all"] = all_mask.astype(int)

        if conditions:
            df.loc[reduce(lambda a, b: a & b, conditions), ["enter_long", "enter_tag"]] = (1, "upper_break_mtf_adaptive")
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: Dict) -> DataFrame:
        exit_cond = (
            (_s(df, "rsi_fast") < 50)
            | (_s(df, "ewo") < self.ewo_low_thr.value)
            | (_s(df, "close_") < _s(df, "vwap"))
            | (_s(df, "close_") < _s(df, "final_lb"))
            | (_s(df, "relative_price") < -self.low_offset.value)
        )
        df.loc[exit_cond, "exit_long"] = 1
        return df

    # ---- Cost & risk: dynamic stop via 1h ATR%, maker-tilted entry, and regime-aware sizing ----
    def custom_stoploss(self, pair: str, trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        row = df.iloc[-1] if len(df) else {}
        atrp = float(row.get("atrp_1h_1h", 0.01))  # default ~1%
        sl = max(0.01, min(0.06, 2.5 * atrp))     # 1% .. 6%
        return sl

    def custom_entry_price(self, pair: str, trade, current_time: datetime,
                           proposed_rate: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        return proposed_rate * 0.999  # gentle maker tilt

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float = 1.0, entry_tag: Optional[str] = None, side: str = "long", **kwargs) -> float:
        budget = float(kwargs.get("budget", max_stake))
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        row = df.iloc[-1] if len(df) else {}
        in_regime = int(row.get("regime_ok", 0)) == 1
        risk_pct = 0.005 if in_regime else 0.001  # 0.5% vs 0.1%
        atrp = float(row.get("atrp_1h_1h", 0.01))
        desired = (budget * risk_pct) / max(atrp, 1e-3)
        if min_stake:
            desired = max(min_stake, desired)
        return float(min(desired, max_stake))
