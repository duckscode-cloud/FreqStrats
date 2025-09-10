# --- realmsopen_breathe.py ---
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
from freqtrade.optimize.space import Categorical, Integer, SKDecimal

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
    """Safe accessor; returns a Series aligned to df.index when col missing."""
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

    return pd.DataFrame(
        {"basic_ub": basic_ub, "final_ub": final_ub, "basic_lb": basic_lb, "final_lb": final_lb},
        index=df.index,
    )


def ewo(series: pd.Series, fast: int = 5, slow: int = 35) -> pd.Series:
    ema_fast = ta.EMA(series, timeperiod=fast)
    ema_slow = ta.EMA(series, timeperiod=slow)
    return (ema_fast - ema_slow) / series * 100.0


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


class realmsopen_breathe(IStrategy):
    """
    5m breakout/momentum with relaxed gates, volatility/market regime guards,
    up-move-friendly pump filter, and realistic ROI/trailing for scalps.

    Key fixes vs prior version:
      - custom_stoploss now returns NEGATIVE values + BE raise
      - widened 1h ATR% band, simplified RSI/EWO gating
      - pump filter no longer blocks healthy UP moves
      - ROI/Trailing tuned so they actually arm on 5m winners
    """

    timeframe = "5m"
    informative_timeframe = "1h"
    startup_candle_count = 600

    process_only_new_candles = True
    use_custom_stoploss = True

    # Exits tuned for 5m scalps (reachable)
    minimal_roi = {"0": 0.007, "30": 0.004, "120": 0.002, "360": 0}
    stoploss = -0.10  # hard cap; dynamic SL used

    trailing_stop = True
    trailing_stop_positive = 0.0035
    trailing_stop_positive_offset = 0.009
    trailing_only_offset_is_reached = True

    position_adjustment_enable = False
    max_entry_position_adjustment = 0

    # ---------------- Hyperoptable knobs (wider, looser) ----------------
    rsi_buy      = IntParameter(40, 62, default=49, space="buy")      # floor for base RSI
    rsi_fast_buf = IntParameter(0, 6, default=2, space="buy")         # how much fast RSI must exceed base RSI

    # ATR% volatility band (1h)
    atrp_min_1h  = DecimalParameter(0.0015, 0.050, default=0.002, decimals=4, space="buy")  # 0.15% .. 5.0%
    atrp_max_1h  = DecimalParameter(0.020, 0.080, default=0.060, decimals=3, space="buy")   # 2.0% .. 8.0%

    # Volume / extension guards
    pct_change_min   = DecimalParameter(0.0015, 0.02, default=0.003, decimals=4, space="buy")
    vol_mult_12      = DecimalParameter(0.8, 2.0, default=1.0, decimals=2, space="buy")
    vol_mult_24      = DecimalParameter(0.8, 2.0, default=1.0, decimals=2, space="buy")
    lookback_candles = IntParameter(30, 100, default=40, space="buy")
    r_480_max        = DecimalParameter(0.40, 0.80, default=0.60, decimals=2, space="buy")

    # VWAP / BB bands
    vwap_k        = DecimalParameter(1.0, 3.0, default=2.0, decimals=1, space="buy")
    bb_len        = IntParameter(18, 22, default=20, space="buy")
    bb_mult       = DecimalParameter(1.5, 3.0, default=2.0, decimals=1, space="buy")
    rel_price_max = DecimalParameter(0.01, 0.05, default=0.03, decimals=3, space="buy")

    # EWO easier for 5m
    ewo_high_thr  = DecimalParameter(0.3, 3.0, default=0.8, decimals=1, space="buy")
    ewo_low_thr   = DecimalParameter(-12.0, -1.0, default=-6.0, decimals=1, space="sell")

    # BTC guard (can be toggled off by hyperopt)
    btc_threshold = DecimalParameter(0.0, 0.02, default=0.000, decimals=3, space="buy")
    btc_diff_min  = DecimalParameter(-0.02, 0.02, default=-0.005, decimals=3, space="buy")

    # Toggles
    use_regime_guard = BooleanParameter(default=True, space="buy")
    use_btc_guard    = BooleanParameter(default=True, space="buy")
    use_close_15m    = BooleanParameter(default=False, space="buy")  # relaxed by default

    # Protections (hyperoptable via --spaces protection)
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

    DEBUG_DIAGNOSTICS = True

    plot_config = {
        "main_plot": {
            "ema_slow": {"color": "orange"},
            "vwap": {"color": "blue"},
            "vwap_upperband": {"color": "blue"},
            "bb_upperband2": {"color": "purple"},
            "final_ub": {"color": "red"},
            "final_lb": {"color": "red"},
        },
        "subplots": {
            "RSI": {"rsi": {"color": "green"}, "rsi_fast": {"color": "teal"}, "rsi_1h": {"color": "grey"}},
            "EWO": {"ewo": {"color": "brown"}},
            "Vol": {"volume_mean_12": {"color": "grey"}, "volume_mean_24": {"color": "grey"}},
        },
    }

    # ---------- HyperOpt spaces (ROI/SL/Trailing) ----------
    class HyperOpt:
        @staticmethod
        def roi_space():
            return [
                Integer(10, 120, name="roi_t1"),
                Integer(20, 240, name="roi_t2"),
                Integer(60, 480, name="roi_t3"),
                SKDecimal(0.004, 0.020, decimals=3, name="roi_p1"),
                SKDecimal(0.002, 0.012, decimals=3, name="roi_p2"),
                SKDecimal(0.001, 0.006, decimals=3, name="roi_p3"),
            ]

        @staticmethod
        def generate_roi_table(params: dict) -> dict:
            t1, t2, t3 = int(params["roi_t1"]), int(params["roi_t2"]), int(params["roi_t3"])
            p1, p2, p3 = float(params["roi_p1"]), float(params["roi_p2"]), float(params["roi_p3"])
            return {0: p1 + p2 + p3, t1: p2 + p3, t1 + t2: p3, t1 + t2 + t3: 0.0}

        @staticmethod
        def stoploss_space():
            return [SKDecimal(-0.200, -0.030, decimals=3, name="stoploss")]

        @staticmethod
        def trailing_space():
            return [
                Categorical([True], name="trailing_stop"),
                SKDecimal(0.002, 0.010, decimals=4, name="trailing_stop_positive"),
                SKDecimal(0.006, 0.020, decimals=3, name="trailing_stop_positive_offset"),
                Categorical([True, False], name="trailing_only_offset_is_reached"),
            ]

        @staticmethod
        def generate_trailing_params(params: dict) -> dict:
            return {
                "trailing_stop": True,
                "trailing_stop_positive": float(params["trailing_stop_positive"]),
                "trailing_stop_positive_offset": float(params["trailing_stop_positive_offset"]),
                "trailing_only_offset_is_reached": params["trailing_only_offset_is_reached"],
            }

    # ---------- Protections ----------
    @property
    def protections(self):
        prot = [{"method": "CooldownPeriod", "stop_duration_candles": int(self.cooldown_candles.value)}]
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
        pairs += [("BTC/USDT", "5m"), ("BTC/USDT", "1d")]
        return pairs

    def populate_indicators(self, df: DataFrame, metadata: Dict) -> DataFrame:
        pair = metadata["pair"]

        # ----- Base (5m) -----
        df["ema_slow"] = ta.EMA(df, timeperiod=100)
        df["ema_50"] = ta.EMA(df, timeperiod=50)

        df["rsi"] = ta.RSI(df, timeperiod=14)
        df["rsi_fast"] = ta.RSI(df, timeperiod=7)

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
        df["relative_price"] = (df["close"] - df["vwap"]) / df["vwap"]

        bb = qtpylib.bollinger_bands(qtpylib.typical_price(df), window=self.bb_len.value, stds=self.bb_mult.value)
        df["bb_upperband2"] = bb["upper"]

        st = supertrend(df, period=10, multiplier=3.0)
        df["final_ub"] = st["final_ub"]
        df["final_lb"] = st["final_lb"]

        df["ewo"] = ewo(df["close"], fast=5, slow=35)

        lb = self.lookback_candles.value
        df["pct_change_lb"] = (df["close"] / df["close"].shift(lb) - 1.0).fillna(0.0)
        df["r_480"] = (df["close"] / df["close"].shift(480) - 1.0).abs().fillna(0.0)

        df["cmf"] = _cmf(df, n=50)

        # ----- Pump filter (UP-move friendly) -----
        ret1 = df["close"].pct_change(1).fillna(0.0)
        ret3 = df["close"].pct_change(3).fillna(0.0)
        # Only block big DOWN moves / pathological spikes; do not block up surges.
        df["pump_bad"] = (ret1 < -0.04) | (ret3 < -0.07)

        # ----- 15m informative -----
        if self.dp:
            df15 = self.dp.get_pair_dataframe(pair=pair, timeframe="15m")
        else:
            df15 = self.resample_dataframe(df, timeframe="15m")
        df15 = df15[["date", "close"]].rename(columns={"close": "close_15m"})
        df = merge_informative_pair(df, df15, self.timeframe, "15m", ffill=True)

        # ----- 1h informative (regime) -----
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

        df1h["atr_1h"] = ta.ATR(df1h, timeperiod=14)
        df1h["atrp_1h"] = (df1h["atr_1h"] / df1h["close"]).clip(lower=1e-6)

        df = merge_informative_pair(
            df, df1h[["date", "uptrend", "rsi", "cti_40", "atrp_1h"]],
            self.timeframe, self.informative_timeframe, ffill=True,
        )

        # Normalize informative names
        def _pick(col_prefix: str, suffix="_1h"):
            cands = [c for c in df.columns if c.startswith(col_prefix) and c.endswith(suffix)]
            return cands[0] if cands else None

        if "uptrend_1h" not in df.columns:
            alt = _pick("uptrend")
            if alt: df["uptrend_1h"] = df[alt]
        if "rsi_1h" not in df.columns:
            alt = _pick("rsi")
            if alt: df["rsi_1h"] = df[alt]
        if "cti_40_1h" not in df.columns:
            alt = _pick("cti_40")
            if alt: df["cti_40_1h"] = df[alt]
        if "atrp_1h_1h" not in df.columns:
            alt = _pick("atrp_1h")
            if alt: df["atrp_1h_1h"] = df[alt]

        # Combined regime flag (trend + RSI + volatility band)
        df["vol_ok_1h"] = _s(df, "atrp_1h_1h").between(self.atrp_min_1h.value, self.atrp_max_1h.value)
        df["regime_ok"] = (
            _s(df, "uptrend_1h").astype(bool)
            & (_s(df, "rsi_1h") > 50)
            & _s(df, "vol_ok_1h").astype(bool)
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

        # pick merged names
        btc_ema_5m_col = [c for c in df.columns if c.startswith("btc_fast") and c.endswith("_5m")]
        btc_ema_1d_col = [c for c in df.columns if c.startswith("btc_fast") and c.endswith("_1d")]
        if not btc_ema_5m_col or not btc_ema_1d_col:
            raise ValueError(f"BTC regime columns missing after merge.")
        df["btc_5m_1d_diff"] = (df[btc_ema_5m_col[0]] / df[btc_ema_1d_col[0]] - 1.0).fillna(0.0)

        # Convenience
        df["close_"] = df["close"]
        df["high_"] = df["high"]
        return df

    def populate_entry_trend(self, df: DataFrame, metadata: Dict) -> DataFrame:
        pair = metadata["pair"]
        stake_currency = self.config.get("stake_currency", "USDT")
        is_stake_currency = pair.endswith(f"/{stake_currency}") or pair.endswith(f"-{stake_currency}")

        # Volume & motion guards
        vol_guard = (df["volume"] > (self.vol_mult_12.value * _s(df, "volume_mean_12"))) & (
            df["volume"] > (self.vol_mult_24.value * _s(df, "volume_mean_24"))
        )
        move_guard = _s(df, "pct_change_lb").abs() > self.pct_change_min.value
        extension_guard = _s(df, "r_480") < self.r_480_max.value

        # Pump: block only ugly DOWN moves / pathological spikes
        pump_ok = ~_s(df, "pump_bad", False)

        # Stretch/relative price guards
        rel_price_guard = _s(df, "relative_price") <= self.rel_price_max.value

        # Regime guard
        regime_guard = (_s(df, "regime_ok", 0).astype(int) == 1)
        if not self.use_regime_guard.value:
            regime_guard = _s(df, "close_") > 0

        # BTC guard
        btc_guard = (_s(df, "btc_5m_1d_diff") > self.btc_diff_min.value) & (_s(df, "btc_5m_1d_diff") > self.btc_threshold.value)
        if not self.use_btc_guard.value:
            btc_guard = _s(df, "close_") > 0

        # RSI: momentum improving (fast above base by buffer) and base above floor
        rsi_align = (_s(df, "rsi_fast") > (_s(df, "rsi") + self.rsi_fast_buf.value)) & (_s(df, "rsi") > self.rsi_buy.value)

        # EWO easier on 5m
        ewo_ok = _s(df, "ewo") > self.ewo_high_thr.value

        # Breakout: either above VWAP upper band OR BB upper with buy-side flow (cmf>0)
        upper_break = (
            (_s(df, "close_") > _s(df, "vwap_upperband")) |
            ((_s(df, "close_") > _s(df, "bb_upperband2")) & (_s(df, "cmf") > 0))
        )

        # 15m trend optional
        close_15m_ok = _s(df, "close_") > 0
        if self.use_close_15m.value:
            close_15m_ok = _s(df, "close_15m") >= _s(df, "close_15m").shift(1)

        conditions: List[pd.Series] = [
            is_stake_currency,
            df["volume"] > 0,
            vol_guard,
            move_guard,
            extension_guard,
            pump_ok,
            rel_price_guard,
            regime_guard,
            btc_guard,
            rsi_align,
            ewo_ok,
            upper_break,
            close_15m_ok,
        ]

        if self.DEBUG_DIAGNOSTICS:
            df["g_is_stake_currency"] = bool(is_stake_currency)
            df["g_vol_guard"] = vol_guard.astype(int)
            df["g_move_guard"] = move_guard.astype(int)
            df["g_extension_guard"] = extension_guard.astype(int)
            df["g_pump_ok"] = pump_ok.astype(int)
            df["g_rel_price_guard"] = rel_price_guard.astype(int)
            df["g_regime_ok"] = (_s(df, "regime_ok", 0).astype(int) == 1).astype(int)
            df["g_btc_guard"] = btc_guard.astype(int)
            df["g_rsi_align"] = rsi_align.astype(int)
            df["g_ewo_ok"] = ewo_ok.astype(int)
            df["g_upper_break"] = upper_break.astype(int)
            df["g_close_15m_ok"] = close_15m_ok.astype(int)
            all_mask = reduce(lambda a, b: a & b, [c if isinstance(c, pd.Series) else bool(c) for c in conditions])
            df["enter_all"] = all_mask.astype(int)

        if conditions:
            df.loc[reduce(lambda a, b: a & b, conditions), ["enter_long", "enter_tag"]] = (1, "breath_breakout_mtf")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: Dict) -> DataFrame:
        # Exit when losing VWAP and momentum fades; keep winners if still above VWAP
        exit_cond = (
            ( (_s(df, "close_") < _s(df, "vwap")) & (_s(df, "rsi_fast") < 50) ) |
            (_s(df, "ewo") < self.ewo_low_thr.value) |
            (_s(df, "close_") < _s(df, "final_lb")) |
            (_s(df, "relative_price") < -0.01)
        )
        df.loc[exit_cond, "exit_long"] = 1
        return df

    # ---- Costs & risk: dynamic stop via 1h ATR%, BE raise, maker-tilted entry, ATR-normalized sizing ----
    def custom_stoploss(self, pair: str, trade, current_time: datetime,
                        current_rate: float, current_profit: float, **kwargs) -> float:
        # IMPORTANT: return NEGATIVE values
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        atrp = float(df.iloc[-1].get("atrp_1h_1h", 0.01)) if len(df) else 0.01

        # Base SL: 0.6%..3.5%, scaled by ATR%
        sl = max(0.006, min(0.035, 1.8 * atrp))

        # Raise to near-BE once modestly green to protect scalps
        if current_profit > 0.006:
            sl = min(sl, 0.0005)  # ~0.05%

        return -sl

    def custom_entry_price(self, pair: str, trade, current_time: datetime,
                           proposed_rate: float, entry_tag: Optional[str], side: str, **kwargs) -> float:
        # Gentle maker tilt; adjust if exchange maker/taker fees differ
        return proposed_rate * 0.999

    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, min_stake: Optional[float], max_stake: float,
                            leverage: float = 1.0, entry_tag: Optional[str] = None, side: str = "long", **kwargs) -> float:
        # ATR-normalized stake sizing: risk ~0.5% in-regime, ~0.1% out-of-regime (but we don't enter out-of-regime by default)
        budget = float(kwargs.get("budget", max_stake))
        df, _ = self.dp.get_analyzed_dataframe(pair=pair, timeframe=self.timeframe)
        in_regime = int(df.iloc[-1].get("regime_ok", 0)) == 1 if len(df) else True
        risk_pct = 0.005 if in_regime else 0.001
        atrp = float(df.iloc[-1].get("atrp_1h_1h", 0.01)) if len(df) else 0.01
        desired = (budget * risk_pct) / max(atrp, 1e-3)
        if min_stake:
            desired = max(min_stake, desired)
        return float(min(desired, max_stake))
