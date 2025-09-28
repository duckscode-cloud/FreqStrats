# --- kasuari_offset_ma.py ---
from functools import reduce
from typing import Optional, Union
from datetime import datetime, timedelta

import math
import logging
import numpy as np
import pandas as pd
from pandas import DataFrame

import talib.abstract as ta
import freqtrade.vendor.qtpylib.indicators as qtpylib  # (unused in signals, kept for parity)

from freqtrade.persistence import Trade
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import (
    DecimalParameter,
    IntParameter,
    BooleanParameter,
    CategoricalParameter,
    stoploss_from_open,
    stoploss_from_absolute,
)
from freqtrade.exchange import timeframe_to_minutes

logger = logging.getLogger(__name__)


def tv_wma(series: pd.Series, length: int = 9) -> pd.Series:
    """
    TradingView-style Weighted MA used by Hull MA.
    Expect a Pandas Series (e.g., close).
    """
    if length < 2:
        length = 2
    norm = 0.0
    s = 0.0
    # weights grow with (length - i) * length  (matches the original snippet's intent)
    for i in range(1, length - 1):
        weight = (length - i) * length
        norm += weight
        s = s + series.shift(i) * weight
    return (s / norm) if norm > 0 else series.copy() * 0.0


def tv_hma(df: DataFrame, length: int = 9, field: str = "close") -> pd.Series:
    """
    TradingView Hull MA constructed from tv_wma.
    """
    if length < 2:
        length = 2
    h = 2 * tv_wma(df[field], math.floor(length / 2)) - tv_wma(df[field], length)
    return tv_wma(h, math.floor(math.sqrt(length)))


class Kasuari_btc_template(IStrategy):
    """
    Offset-MA concept with HMA/DEMA/TEMA variants and RSI/MFI gating.
    Placeholders replaced with concrete RSI/MFI conditions.
    """

    INTERFACE_VERSION = 3

    def version(self) -> str:
        return "Kasuari-v1-btc-clean"

    # Leverage & ROI (kept as in your original logic)
    lev = 10
    roi = 0.0015
    minimal_roi = {"0": (roi * lev)}

    # --- Parameter blocks (kept) ---
    optimize_buy_hma1 = False
    buy_length_hma1 = IntParameter(1, 50, default=6, optimize=optimize_buy_hma1)
    buy_offset_hma1 = IntParameter(16, 20, default=20, optimize=optimize_buy_hma1)

    optimize_buy_hma2 = False
    buy_length_hma2 = IntParameter(1, 50, default=6, optimize=optimize_buy_hma2)
    buy_offset_hma2 = IntParameter(16, 20, default=20, optimize=optimize_buy_hma2)

    optimize_buy_hma3 = False
    buy_length_hma3 = IntParameter(1, 50, default=6, optimize=optimize_buy_hma3)
    buy_offset_hma3 = IntParameter(20, 24, default=20, optimize=optimize_buy_hma3)

    optimize_buy_hma4 = False
    buy_length_hma4 = IntParameter(1, 50, default=6, optimize=optimize_buy_hma4)
    buy_offset_hma4 = IntParameter(20, 24, default=20, optimize=optimize_buy_hma4)

    optimize_buy_dema1 = False
    buy_length_dema1 = IntParameter(1, 50, default=6, optimize=optimize_buy_dema1)
    buy_offset_dema1 = IntParameter(16, 20, default=20, optimize=optimize_buy_dema1)

    optimize_buy_dema2 = False
    buy_length_dema2 = IntParameter(1, 50, default=6, optimize=optimize_buy_dema2)
    buy_offset_dema2 = IntParameter(16, 20, default=20, optimize=optimize_buy_dema2)

    optimize_buy_dema3 = False
    buy_length_dema3 = IntParameter(1, 50, default=6, optimize=optimize_buy_dema3)
    buy_offset_dema3 = IntParameter(20, 24, default=20, optimize=optimize_buy_dema3)

    optimize_buy_dema4 = False
    buy_length_dema4 = IntParameter(1, 50, default=6, optimize=optimize_buy_dema4)
    buy_offset_dema4 = IntParameter(20, 24, default=20, optimize=optimize_buy_dema4)

    optimize_buy_tema1 = False
    buy_length_tema1 = IntParameter(1, 40, default=6, optimize=optimize_buy_tema1)
    buy_offset_tema1 = IntParameter(16, 20, default=20, optimize=optimize_buy_tema1)

    optimize_buy_tema2 = False
    buy_length_tema2 = IntParameter(1, 40, default=6, optimize=optimize_buy_tema2)
    buy_offset_tema2 = IntParameter(16, 20, default=20, optimize=optimize_buy_tema2)

    optimize_buy_tema3 = False
    buy_length_tema3 = IntParameter(1, 40, default=6, optimize=optimize_buy_tema3)
    buy_offset_tema3 = IntParameter(20, 24, default=20, optimize=optimize_buy_tema3)

    optimize_buy_tema4 = False
    buy_length_tema4 = IntParameter(1, 40, default=6, optimize=optimize_buy_tema4)
    buy_offset_tema4 = IntParameter(20, 24, default=20, optimize=optimize_buy_tema4)

    optimize_buy_rsi1 = False
    buy_rsi1 = IntParameter(1, 10, default=6, optimize=optimize_buy_rsi1)
    buy_rolling_rsi1 = IntParameter(2, 30, default=20, optimize=optimize_buy_rsi1)
    buy_diff_rsi1 = IntParameter(2, 10, default=6, optimize=optimize_buy_rsi1)

    optimize_buy_rsi2 = False
    buy_rsi2 = IntParameter(10, 19, default=16, optimize=optimize_buy_rsi2)
    buy_rolling_rsi2 = IntParameter(2, 30, default=20, optimize=optimize_buy_rsi2)
    buy_diff_rsi2 = IntParameter(2, 10, default=6, optimize=optimize_buy_rsi2)

    optimize_buy_rsi3 = False
    buy_rsi3 = IntParameter(1, 10, default=6, optimize=optimize_buy_rsi3)
    buy_rolling_rsi3 = IntParameter(2, 30, default=10, optimize=optimize_buy_rsi3)
    buy_diff_rsi3 = IntParameter(2, 10, default=6, optimize=optimize_buy_rsi3)

    optimize_buy_rsi4 = False
    buy_rsi4 = IntParameter(10, 19, default=16, optimize=optimize_buy_rsi4)
    buy_rolling_rsi4 = IntParameter(2, 30, default=20, optimize=optimize_buy_rsi4)
    buy_diff_rsi4 = IntParameter(1, 10, default=6, optimize=optimize_buy_rsi4)

    optimize_buy_mfi_1 = False
    buy_mfi_1 = IntParameter(1, 10, default=6, optimize=optimize_buy_mfi_1)
    buy_rolling_mfi_1 = IntParameter(1, 30, default=20, optimize=optimize_buy_mfi_1)
    buy_diff_mfi_1 = IntParameter(1, 10, default=6, optimize=optimize_buy_mfi_1)

    optimize_buy_mfi_2 = False
    buy_mfi_2 = IntParameter(10, 19, default=16, optimize=optimize_buy_mfi_2)
    buy_rolling_mfi_2 = IntParameter(1, 30, default=20, optimize=optimize_buy_mfi_2)
    buy_diff_mfi_2 = IntParameter(1, 10, default=6, optimize=optimize_buy_mfi_2)

    optimize_buy_mfi_3 = False
    buy_mfi_3 = IntParameter(1, 10, default=6, optimize=optimize_buy_mfi_3)
    buy_rolling_mfi_3 = IntParameter(1, 19, default=10, optimize=optimize_buy_mfi_3)
    buy_diff_mfi_3 = IntParameter(1, 10, default=6, optimize=optimize_buy_mfi_3)

    optimize_buy_mfi_4 = False
    buy_mfi_4 = IntParameter(10, 19, default=16, optimize=optimize_buy_mfi_4)
    buy_rolling_mfi_4 = IntParameter(12, 30, default=20, optimize=optimize_buy_mfi_4)
    buy_diff_mfi_4 = IntParameter(1, 10, default=6, optimize=optimize_buy_mfi_4)

    optimize_sell_ema1 = False
    sell_length_ema1 = IntParameter(1, 50, default=6, optimize=optimize_sell_ema1)
    sell_offset_ema1 = IntParameter(20, 24, default=20, optimize=optimize_sell_ema1)

    optimize_sell_ema2 = False
    sell_length_ema2 = IntParameter(1, 50, default=6, optimize=optimize_sell_ema2)
    sell_offset_ema2 = IntParameter(16, 20, default=20, optimize=optimize_sell_ema2)

    optimize_sell_ema3 = False
    sell_length_ema3 = IntParameter(1, 50, default=6, optimize=optimize_sell_ema3)
    sell_offset_ema3 = IntParameter(16, 20, default=20, optimize=optimize_sell_ema3)

    optimize_sell_ema4 = False
    sell_length_ema4 = IntParameter(1, 50, default=6, optimize=optimize_sell_ema4)
    sell_offset_ema4 = IntParameter(20, 24, default=20, optimize=optimize_sell_ema4)

    sell_clear_old_trade = IntParameter(11, 25, default=20, optimize=False)
    sell_clear_old_trade_profit = IntParameter(0, 5, default=0, optimize=False)

    # Risk settings (kept)
    stoploss = -0.99
    trailing_stop = False
    trailing_stop_positive = 0.01
    trailing_stop_positive_offset = 0.03
    trailing_only_offset_is_reached = True

    use_exit_signal = True
    exit_profit_only = False
    exit_profit_offset = 0.01
    ignore_roi_if_entry_signal = False

    timeframe = "15m"
    can_short = True
    process_only_new_candles = True
    startup_candle_count = 999

    # handy shortcut
    timeframe_minutes = timeframe_to_minutes(timeframe)

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        # Core oscillators
        dataframe["rsi"] = ta.RSI(dataframe, timeperiod=15)
        dataframe["rsi_45"] = ta.RSI(dataframe, timeperiod=45)

        dataframe["mfi"] = ta.MFI(dataframe, timeperiod=15)
        dataframe["mfi_45"] = ta.MFI(dataframe, timeperiod=45)

        # Stoch fast (only 'fastk' used in original)
        stoch_fast = ta.STOCHF(dataframe, 5, 3, 0, 3, 0)
        dataframe["fastk"] = stoch_fast["fastk"]

        # Data quality guard
        dataframe["live_data_ok"] = (
            dataframe["volume"].rolling(window=72, min_periods=72).min() > 0
        )

        # Precompute offsets when not optimizing (kept)
        if not self.optimize_buy_hma1:
            dataframe["hma_offset_buy1"] = (
                tv_hma(dataframe, int(5 * self.buy_length_hma1.value))
                * 0.05
                * self.buy_offset_hma1.value
            )
        if not self.optimize_buy_hma2:
            dataframe["hma_offset_buy2"] = (
                tv_hma(dataframe, int(5 * self.buy_length_hma2.value))
                * 0.05
                * self.buy_offset_hma2.value
            )
        if not self.optimize_buy_hma3:
            dataframe["hma_offset_buy3"] = (
                tv_hma(dataframe, int(5 * self.buy_length_hma3.value))
                * 0.05
                * self.buy_offset_hma3.value
            )
        if not self.optimize_buy_hma4:
            dataframe["hma_offset_buy4"] = (
                tv_hma(dataframe, int(5 * self.buy_length_hma4.value))
                * 0.05
                * self.buy_offset_hma4.value
            )

        if not self.optimize_buy_dema1:
            dataframe["dema_offset_buy1"] = (
                ta.DEMA(dataframe, int(5 * self.buy_length_dema1.value))
                * 0.05
                * self.buy_offset_dema1.value
            )
        if not self.optimize_buy_dema2:
            dataframe["dema_offset_buy2"] = (
                ta.DEMA(dataframe, int(5 * self.buy_length_dema2.value))
                * 0.05
                * self.buy_offset_dema2.value
            )
        if not self.optimize_buy_dema3:
            dataframe["dema_offset_buy3"] = (
                ta.DEMA(dataframe, int(5 * self.buy_length_dema3.value))
                * 0.05
                * self.buy_offset_dema3.value
            )
        if not self.optimize_buy_dema4:
            dataframe["dema_offset_buy4"] = (
                ta.DEMA(dataframe, int(5 * self.buy_length_dema4.value))
                * 0.05
                * self.buy_offset_dema4.value
            )

        if not self.optimize_buy_tema1:
            dataframe["tema_offset_buy1"] = (
                ta.TEMA(dataframe, int(5 * self.buy_length_tema1.value))
                * 0.05
                * self.buy_offset_tema1.value
            )
        if not self.optimize_buy_tema2:
            dataframe["tema_offset_buy2"] = (
                ta.TEMA(dataframe, int(5 * self.buy_length_tema2.value))
                * 0.05
                * self.buy_offset_tema2.value
            )
        if not self.optimize_buy_tema3:
            dataframe["tema_offset_buy3"] = (
                ta.TEMA(dataframe, int(5 * self.buy_length_tema3.value))
                * 0.05
                * self.buy_offset_tema3.value
            )
        if not self.optimize_buy_tema4:
            dataframe["tema_offset_buy4"] = (
                ta.TEMA(dataframe, int(5 * self.buy_length_tema4.value))
                * 0.05
                * self.buy_offset_tema4.value
            )

        # EMA columns for exits when not optimizing (kept)
        if not self.optimize_sell_ema1:
            col = f"ema_{int(5 * self.sell_length_ema1.value)}"
            if col not in dataframe.columns:
                dataframe[col] = ta.EMA(dataframe, int(5 * self.sell_length_ema1.value))
        if not self.optimize_sell_ema2:
            col = f"ema_{int(5 * self.sell_length_ema2.value)}"
            if col not in dataframe.columns:
                dataframe[col] = ta.EMA(dataframe, int(5 * self.sell_length_ema2.value))
        if not self.optimize_sell_ema3:
            col = f"ema_{int(5 * self.sell_length_ema3.value)}"
            if col not in dataframe.columns:
                dataframe[col] = ta.EMA(dataframe, int(5 * self.sell_length_ema3.value))
        if not self.optimize_sell_ema4:
            col = f"ema_{int(5 * self.sell_length_ema4.value)}"
            if col not in dataframe.columns:
                dataframe[col] = ta.EMA(dataframe, int(5 * self.sell_length_ema4.value))

        # Volume context (kept)
        vol_20_max = dataframe["volume"].rolling(window=20).max()
        vol_20_min = dataframe["volume"].rolling(window=20).min()
        roll_20 = (vol_20_max - dataframe["volume"]) / (vol_20_max - vol_20_min)
        dataframe["vol_base"] = roll_20.rolling(5).mean()
        dataframe["vol_20"] = roll_20

        vol_40_max = dataframe["volume"].rolling(window=40).max()
        vol_40_min = dataframe["volume"].rolling(window=40).min()
        roll_40 = (vol_40_max - dataframe["volume"]) / (vol_40_max - vol_40_min)
        dataframe["vol_40_base"] = roll_40.rolling(5).mean()
        dataframe["vol_40"] = roll_40

        return dataframe

    def populate_entry_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        conditions = []
        conditions_short = []

        # When optimizing, compute offsets here too (kept)
        if self.optimize_buy_hma1:
            dataframe["hma_offset_buy1"] = (
                tv_hma(dataframe, int(5 * self.buy_length_hma1.value))
                * 0.05
                * self.buy_offset_hma1.value
            )
        if self.optimize_buy_hma2:
            dataframe["hma_offset_buy2"] = (
                tv_hma(dataframe, int(5 * self.buy_length_hma2.value))
                * 0.05
                * self.buy_offset_hma2.value
            )
        if self.optimize_buy_hma3:
            dataframe["hma_offset_buy3"] = (
                tv_hma(dataframe, int(5 * self.buy_length_hma3.value))
                * 0.05
                * self.buy_offset_hma3.value
            )
        if self.optimize_buy_hma4:
            dataframe["hma_offset_buy4"] = (
                tv_hma(dataframe, int(5 * self.buy_length_hma4.value))
                * 0.05
                * self.buy_offset_hma4.value
            )

        if self.optimize_buy_dema1:
            dataframe["dema_offset_buy1"] = (
                ta.DEMA(dataframe, int(5 * self.buy_length_dema1.value))
                * 0.05
                * self.buy_offset_dema1.value
            )
        if self.optimize_buy_dema2:
            dataframe["dema_offset_buy2"] = (
                ta.DEMA(dataframe, int(5 * self.buy_length_dema2.value))
                * 0.05
                * self.buy_offset_dema2.value
            )
        if self.optimize_buy_dema3:
            dataframe["dema_offset_buy3"] = (
                ta.DEMA(dataframe, int(5 * self.buy_length_dema3.value))
                * 0.05
                * self.buy_offset_dema3.value
            )
        if self.optimize_buy_dema4:
            dataframe["dema_offset_buy4"] = (
                ta.DEMA(dataframe, int(5 * self.buy_length_dema4.value))
                * 0.05
                * self.buy_offset_dema4.value
            )

        if self.optimize_buy_tema1:
            dataframe["tema_offset_buy1"] = (
                ta.TEMA(dataframe, int(5 * self.buy_length_tema1.value))
                * 0.05
                * self.buy_offset_tema1.value
            )
        if self.optimize_buy_tema2:
            dataframe["tema_offset_buy2"] = (
                ta.TEMA(dataframe, int(5 * self.buy_length_tema2.value))
                * 0.05
                * self.buy_offset_tema2.value
            )
        if self.optimize_buy_tema3:
            dataframe["tema_offset_buy3"] = (
                ta.TEMA(dataframe, int(5 * self.buy_length_tema3.value))
                * 0.05
                * self.buy_offset_tema3.value
            )
        if self.optimize_buy_tema4:
            dataframe["tema_offset_buy4"] = (
                ta.TEMA(dataframe, int(5 * self.buy_length_tema4.value))
                * 0.05
                * self.buy_offset_tema4.value
            )

        # --- Replacements for placeholder RSI/MFI flags (concrete, sane gates) ---
        rsi = dataframe["rsi"]
        rsi_ma = dataframe["rsi_45"]

        # Long-leaning RSI gates
        rsi_1 = (rsi < 35) & (rsi < rsi_ma) & (rsi.diff() > 0)
        rsi_12 = (rsi < 45) & (rsi.diff() > 0)
        rsi_24 = rsi.rolling(2).mean().diff() > 0

        # Short-leaning RSI gates
        rsi_13 = (rsi > 55) & (rsi.diff() < 0)
        rsi_34 = rsi.rolling(3).mean().diff() < 0
        rsi_4 = (rsi > 65) & (rsi > rsi_ma) & (rsi.diff() < 0)

        mfi = dataframe["mfi"]
        mfi_ma = dataframe["mfi_45"]

        # Long-leaning MFI gates
        mfi_1 = (mfi < 35) & (mfi < mfi_ma) & (mfi.diff() > 0)
        mfi_3 = (mfi < 40) & (mfi.diff() > 0)

        # Short-leaning MFI gates
        mfi_2 = (mfi > 65) & (mfi > mfi_ma) & (mfi.diff() < 0)
        mfi_4 = (mfi > 60) & (mfi.diff() < 0)

        # ------------------------------------------------------------------------

        dataframe["enter_tag"] = ""
        dataframe["enter_long"] = 0
        dataframe["enter_short"] = 0

        add_check = dataframe["live_data_ok"]

        # === HMA blocks (kept) ===
        no_hma_1 = []
        buy_offset_hma1 = (dataframe["close"] < dataframe["hma_offset_buy1"]) & rsi_1
        dataframe.loc[buy_offset_hma1, "enter_tag"] += "l_h_1 "
        conditions.append(buy_offset_hma1 if not no_hma_1 else buy_offset_hma1 & (reduce(lambda x, y: x | y, no_hma_1) == False))

        no_hma_2 = []
        buy_offset_hma2 = (dataframe["close"] < dataframe["hma_offset_buy2"]) & rsi_12 & rsi_24
        dataframe.loc[buy_offset_hma2, "enter_tag"] += "l_h_2 "
        conditions.append(buy_offset_hma2 if not no_hma_2 else buy_offset_hma2 & (reduce(lambda x, y: x | y, no_hma_2) == False))

        no_hma_3 = []
        buy_offset_hma3 = (dataframe["close"] > dataframe["hma_offset_buy3"]) & rsi_13 & rsi_34
        dataframe.loc[buy_offset_hma3, "enter_tag"] += "s_h_3 "
        conditions_short.append(buy_offset_hma3 if not no_hma_3 else buy_offset_hma3 & (reduce(lambda x, y: x | y, no_hma_3) == False))

        no_hma_4 = []
        buy_offset_hma4 = (dataframe["close"] > dataframe["hma_offset_buy4"]) & rsi_4
        dataframe.loc[buy_offset_hma4, "enter_tag"] += "s_h_4 "
        conditions_short.append(buy_offset_hma4 if not no_hma_4 else buy_offset_hma4 & (reduce(lambda x, y: x | y, no_hma_4) == False))

        # === DEMA blocks (kept) ===
        no_dema_1 = []
        buy_offset_dema1 = ((dataframe["close"] < dataframe["dema_offset_buy1"]).rolling(2).min() > 0) & rsi_1
        dataframe.loc[buy_offset_dema1, "enter_tag"] += "l_d_1 "
        conditions.append(buy_offset_dema1 if not no_dema_1 else buy_offset_dema1 & (reduce(lambda x, y: x | y, no_dema_1) == False))

        no_dema_2 = []
        buy_offset_dema2 = ((dataframe["close"] < dataframe["dema_offset_buy2"]).rolling(3).min() > 0) & rsi_12 & rsi_24
        dataframe.loc[buy_offset_dema2, "enter_tag"] += "l_dema3_12 "
        conditions.append(buy_offset_dema2 if not no_dema_2 else buy_offset_dema2 & (reduce(lambda x, y: x | y, no_dema_2) == False))

        no_dema_3 = []
        buy_offset_dema3 = (dataframe["close"] > dataframe["dema_offset_buy3"]) & rsi_13 & rsi_34
        dataframe.loc[buy_offset_dema3, "enter_tag"] += "s_d_3 "
        conditions_short.append(buy_offset_dema3 if not no_dema_3 else buy_offset_dema3 & (reduce(lambda x, y: x | y, no_dema_3) == False))

        no_dema_4 = []
        buy_offset_dema4 = ((dataframe["close"] > dataframe["dema_offset_buy4"]).rolling(3).min() > 0) & rsi_4
        dataframe.loc[buy_offset_dema4, "enter_tag"] += "s_dema3_34 "
        conditions_short.append(buy_offset_dema4 if not no_dema_4 else buy_offset_dema4 & (reduce(lambda x, y: x | y, no_dema_4) == False))

        # === TEMA blocks (kept) ===
        no_tema_1 = []
        buy_offset_tema1 = (dataframe["close"] < dataframe["tema_offset_buy1"]) & rsi_1
        dataframe.loc[buy_offset_tema1, "enter_tag"] += "l_t_1 "
        conditions.append(buy_offset_tema1 if not no_tema_1 else buy_offset_tema1 & (reduce(lambda x, y: x | y, no_tema_1) == False))

        no_tema_2 = []
        buy_offset_tema2 = ((dataframe["close"] < dataframe["tema_offset_buy2"]).rolling(2).min() > 0) & rsi_12 & rsi_24
        dataframe.loc[buy_offset_tema2, "enter_tag"] += "l_t_2 "
        conditions.append(buy_offset_tema2 if not no_tema_2 else buy_offset_tema2 & (reduce(lambda x, y: x | y, no_tema_2) == False))

        no_tema_3 = []
        buy_offset_tema3 = (dataframe["close"] > dataframe["tema_offset_buy3"]) & rsi_13 & rsi_34
        dataframe.loc[buy_offset_tema3, "enter_tag"] += "s_t_3 "
        conditions_short.append(buy_offset_tema3 if not no_tema_3 else buy_offset_tema3 & (reduce(lambda x, y: x | y, no_tema_3) == False))

        no_tema_4 = []
        buy_offset_tema4 = (dataframe["close"] > dataframe["tema_offset_buy4"]) & rsi_4
        dataframe.loc[buy_offset_tema4, "enter_tag"] += "s_t_4 "
        conditions_short.append(buy_offset_tema4 if not no_tema_4 else buy_offset_tema4 & (reduce(lambda x, y: x | y, no_tema_4) == False))

        # === RSI-only blocks (kept) ===
        no_rsi_1 = []
        buy_offset_rsi1 = (
            (dataframe["rsi"] < (5 * self.buy_rsi1.value)).rolling(int(self.buy_rolling_rsi1.value)).min() > 0
        ) & ((dataframe["rsi"].shift() - dataframe["rsi"]) > (3 * self.buy_diff_rsi1.value))
        dataframe.loc[buy_offset_rsi1, "enter_tag"] += "l_r_1 "
        conditions.append(buy_offset_rsi1 if not no_rsi_1 else buy_offset_rsi1 & (reduce(lambda x, y: x | y, no_rsi_1) == False))

        no_rsi_2 = []
        buy_offset_rsi2 = (
            (dataframe["rsi"] > (5 * self.buy_rsi2.value)).rolling(int(self.buy_rolling_rsi2.value)).min() > 0
        ) & ((dataframe["rsi"] - dataframe["rsi"].shift()) > (3 * self.buy_diff_rsi2.value))
        dataframe.loc[buy_offset_rsi2, "enter_tag"] += "s_r_2 "
        conditions_short.append(buy_offset_rsi2 if not no_rsi_2 else buy_offset_rsi2 & (reduce(lambda x, y: x | y, no_rsi_2) == False))

        no_rsi_3 = []
        buy_offset_rsi3 = (
            (dataframe["rsi"] < (5 * self.buy_rsi3.value)).rolling(int(self.buy_rolling_rsi3.value)).min() > 0
        ) & ((dataframe["rsi"].shift() - dataframe["rsi"]) > (3 * self.buy_diff_rsi3.value))
        dataframe.loc[buy_offset_rsi3, "enter_tag"] += "l_r_3 "
        conditions.append(buy_offset_rsi3 if not no_rsi_3 else buy_offset_rsi3 & (reduce(lambda x, y: x | y, no_rsi_3) == False))

        no_rsi_4 = []
        buy_offset_rsi4 = (
            (dataframe["rsi"] > (5 * self.buy_rsi4.value)).rolling(int(self.buy_rolling_rsi4.value)).min() > 0
        ) & ((dataframe["rsi"] - dataframe["rsi"].shift()) > (3 * self.buy_diff_rsi4.value))
        dataframe.loc[buy_offset_rsi4, "enter_tag"] += "s_r_4 "
        conditions_short.append(buy_offset_rsi4 if not no_rsi_4 else buy_offset_rsi4 & (reduce(lambda x, y: x | y, no_rsi_4) == False))

        # === MFI-only blocks (kept) ===
        no_mfi_1 = []
        buy_offset_mfi_1 = (
            (dataframe["mfi"] < (5 * self.buy_mfi_1.value)).rolling(int(self.buy_rolling_mfi_1.value)).min() > 0
        ) & ((dataframe["mfi"].shift() - dataframe["mfi"]) > (3 * self.buy_diff_mfi_1.value))
        dataframe.loc[buy_offset_mfi_1, "enter_tag"] += "l_m_1 "
        conditions.append(buy_offset_mfi_1 if not no_mfi_1 else buy_offset_mfi_1 & (reduce(lambda x, y: x | y, no_mfi_1) == False))

        no_mfi_2 = []
        buy_offset_mfi_2 = (
            (dataframe["mfi"] > (5 * self.buy_mfi_2.value)).rolling(int(self.buy_rolling_mfi_2.value)).min() > 0
        ) & ((dataframe["mfi"] - dataframe["mfi"].shift()) > (3 * self.buy_diff_mfi_2.value))
        dataframe.loc[buy_offset_mfi_2, "enter_tag"] += "s_m_2 "
        conditions_short.append(buy_offset_mfi_2 if not no_mfi_2 else buy_offset_mfi_2 & (reduce(lambda x, y: x | y, no_mfi_2) == False))

        no_mfi_3 = []
        buy_offset_mfi_3 = (
            (dataframe["mfi"] < (5 * self.buy_mfi_3.value)).rolling(int(self.buy_rolling_mfi_3.value)).min() > 0
        ) & ((dataframe["mfi"].shift() - dataframe["mfi"]) > (3 * self.buy_diff_mfi_3.value))
        dataframe.loc[buy_offset_mfi_3, "enter_tag"] += "l_m_3 "
        conditions.append(buy_offset_mfi_3 if not no_mfi_3 else buy_offset_mfi_3 & (reduce(lambda x, y: x | y, no_mfi_3) == False))

        no_mfi_4 = []
        buy_offset_mfi_4 = (
            (dataframe["mfi"] > (5 * self.buy_mfi_4.value)).rolling(int(self.buy_rolling_mfi_4.value)).min() > 0
        ) & ((dataframe["mfi"] - dataframe["mfi"].shift()) > (3 * self.buy_diff_mfi_4.value))
        dataframe.loc[buy_offset_mfi_4, "enter_tag"] += "s_m_4 "
        conditions_short.append(buy_offset_mfi_4 if not no_mfi_4 else buy_offset_mfi_4 & (reduce(lambda x, y: x | y, no_mfi_4) == False))

        # Apply long/short aggregations (kept)
        if conditions:
            no_long = []
            dataframe.loc[(reduce(lambda x, y: x | y, conditions)) & (add_check), "enter_long"] = 1
            if no_long:
                dataframe.loc[(reduce(lambda x, y: x | y, no_long)), "enter_long"] = 0

        if conditions_short:
            no_short = []
            dataframe.loc[(reduce(lambda x, y: x | y, conditions_short)) & (add_check), "enter_short"] = 1
            if no_short:
                dataframe.loc[(reduce(lambda x, y: x | y, no_short)), "enter_short"] = 0

        return dataframe

    def populate_exit_trend(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        dataframe["exit_tag"] = ""
        conditions = []
        conditions_short = []

        add_check = dataframe["volume"] > 0

        # Ensure EMA columns exist when optimizing (kept)
        if self.optimize_sell_ema1:
            col = f"ema_{int(5 * self.sell_length_ema1.value)}"
            if col not in dataframe.columns:
                dataframe[col] = ta.EMA(dataframe, int(5 * self.sell_length_ema1.value))
        if self.optimize_sell_ema2:
            col = f"ema_{int(5 * self.sell_length_ema2.value)}"
            if col not in dataframe.columns:
                dataframe[col] = ta.EMA(dataframe, int(5 * self.sell_length_ema2.value))
        if self.optimize_sell_ema3:
            col = f"ema_{int(5 * self.sell_length_ema3.value)}"
            if col not in dataframe.columns:
                dataframe[col] = ta.EMA(dataframe, int(5 * self.sell_length_ema3.value))
        if self.optimize_sell_ema4:
            col = f"ema_{int(5 * self.sell_length_ema4.value)}"
            if col not in dataframe.columns:
                dataframe[col] = ta.EMA(dataframe, int(5 * self.sell_length_ema4.value))

        # Exit rules (kept)
        sell_ema_1 = dataframe["close"] > (
            dataframe[f"ema_{int(5 * self.sell_length_ema1.value)}"] * 0.05 * self.sell_offset_ema1.value
        )
        dataframe.loc[sell_ema_1, "exit_tag"] += "l_e_u "
        conditions.append(sell_ema_1)

        sell_ema_2 = dataframe["close"] < (
            dataframe[f"ema_{int(5 * self.sell_length_ema2.value)}"] * 0.05 * self.sell_offset_ema2.value
        )
        dataframe.loc[sell_ema_2, "exit_tag"] += "l_e_d "
        conditions.append(sell_ema_2)

        sell_ema_3 = dataframe["close"] < (
            dataframe[f"ema_{int(5 * self.sell_length_ema3.value)}"] * 0.05 * self.sell_offset_ema3.value
        )
        dataframe.loc[sell_ema_3, "exit_tag"] += "s_e_d "
        conditions_short.append(sell_ema_3)

        sell_ema_4 = dataframe["close"] > (
            dataframe[f"ema_{int(5 * self.sell_length_ema4.value)}"] * 0.05 * self.sell_offset_ema4.value
        )
        dataframe.loc[sell_ema_4, "exit_tag"] += "s_e_u "
        conditions_short.append(sell_ema_4)

        if conditions:
            dataframe.loc[(reduce(lambda x, y: x | y, conditions)) & (add_check), "exit_long"] = 1
        if conditions_short:
            dataframe.loc[(reduce(lambda x, y: x | y, conditions_short)) & (add_check), "exit_short"] = 1

        return dataframe

    def leverage(
        self,
        pair: str,
        current_time: datetime,
        current_rate: float,
        proposed_leverage: float,
        max_leverage: float,
        entry_tag: Optional[str],
        side: str,
        **kwargs,
    ) -> float:
        return self.lev

    def custom_exit(
        self,
        pair: str,
        trade: Trade,
        current_time: datetime,
        current_rate: float,
        current_profit: float,
        **kwargs,
    ) -> Optional[Union[str, bool]]:
        """
        Keep original logic: after N * timeframe minutes, if profit above -x%, force exit tag.
        """
        # Recompute profit from last candle (kept)
        if (current_time - timedelta(minutes=int(self.timeframe_minutes))) >= trade.open_date_utc:
            dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
            current_candle = dataframe.iloc[-1].squeeze()
            current_profit = trade.calc_profit_ratio(current_candle["close"])

        if current_time - timedelta(minutes=int(self.timeframe_minutes * self.sell_clear_old_trade.value)) >= trade.open_date_utc:
            if current_profit >= (-0.01 * self.sell_clear_old_trade_profit.value):
                return "sell_old_trade"
        return None
