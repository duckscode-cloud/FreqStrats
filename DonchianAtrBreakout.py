# DonchianAtrBreakout — Option A, tuned to reduce draws
from freqtrade.strategy.interface import IStrategy
from freqtrade.strategy import merge_informative_pair
from freqtrade.enums import RunMode
from pandas import DataFrame
import numpy as np
from datetime import datetime
from typing import Optional
from freqtrade.persistence import Trade

class DonchianAtrBreakout(IStrategy):
    INTERFACE_VERSION = 3

    timeframe = '5m'
    informative_timeframe = '1h'
    startup_candle_count = 200
    process_only_new_candles = True

    # ROI: keep small positive tail so turn into tiny wins
    minimal_roi = {
        "0":   0.016,   # was 0.018
        "30":  0.010,   # was 0.012
        "90":  0.004,   # was 0.006
        "180": 0.002    # was 0.0
    }

    # Built-in trailing: arm earlier, trail a bit tighter
    trailing_stop = True
    trailing_stop_positive = 0.004     # was 0.006
    trailing_stop_positive_offset = 0.009  # was 0.012
    trailing_only_offset_is_reached = True

    stoploss = -0.10
    use_custom_stoploss = False   # keep built-in trailing

    position_adjustment_enable = True
    max_entry_position_adjustment = 0

    # Signals (slightly more permissive defaults)
    dc_len = 14
    ema_fast = 50
    ema_slow = 200
    atr_len = 14
    vol_floor = 0.004      # leave as-is; consider 0.003 if you want more trades
    tp1_pct = 0.008

    # Risk model (kept)
    risk_per_trade = 0.008
    atr_mult_sl = 1.2

    # ---------- indicators ----------
    @staticmethod
    def _atr(df: DataFrame, n: int):
        prev_close = df['close'].shift(1)
        tr1 = (df['high'] - df['low']).abs()
        tr2 = (df['high'] - prev_close).abs()
        tr3 = (df['low'] - prev_close).abs()
        tr = np.maximum(tr1, np.maximum(tr2, tr3))
        return tr.ewm(alpha=1 / n, adjust=False, min_periods=n).mean()

    def informative_pairs(self):
        return [(p, self.informative_timeframe) for p in self.dp.current_whitelist()]

    def populate_indicators(self, df: DataFrame, metadata: dict) -> DataFrame:
        df['ema_fast'] = df['close'].ewm(span=self.ema_fast, adjust=False).mean()
        df['ema_slow'] = df['close'].ewm(span=self.ema_slow, adjust=False).mean()
        df['dc_high'] = df['high'].rolling(self.dc_len).max()
        df['dc_low']  = df['low'].rolling(self.dc_len).min()
        df['atr'] = self._atr(df, self.atr_len)
        df['atrp'] = df['atr'] / df['close']

        inf = self.dp.get_pair_dataframe(pair=metadata['pair'], timeframe=self.informative_timeframe)
        if inf is not None and not inf.empty:
            inf['ema_fast_1h'] = inf['close'].ewm(span=self.ema_fast, adjust=False).mean()
            inf['ema_slow_1h'] = inf['close'].ewm(span=self.ema_slow, adjust=False).mean()
            df = merge_informative_pair(df, inf[['ema_fast_1h','ema_slow_1h']],
                                        self.timeframe, self.informative_timeframe, ffill=True)
            df['htf_trend'] = (df['ema_fast_1h'] > df['ema_slow_1h']).astype(int)
        else:
            df['htf_trend'] = 1
        return df

    # ---------- entries/exits ----------
    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        long_cond = (
            (df['high'] > df['dc_high'].shift(1)) &      # intrabar breakout
            (df['ema_fast'] > df['ema_slow']) &          # 5m trend
            (df['htf_trend'] == 1) &                     # 1h trend
            (df['atrp'] >= self.vol_floor)               # volatility floor
        )
        df.loc[long_cond, 'enter_long'] = 1
        df.loc[long_cond, 'enter_tag'] = 'donchian_breakout'
        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        # exits via ROI + trailing + custom_exit below
        return df

    # ---------- partial take-profit ----------
    def adjust_trade_position(self, trade: Trade, current_time: datetime,
                              current_rate: float, current_profit: float, **kwargs) -> Optional[float]:
        if current_profit is not None and current_profit >= self.tp1_pct and trade.nr_of_successful_exits == 0:
            return -(trade.stake_amount * 0.50)
        return None

    # ---------- risk-based stake ----------
    def custom_stake_amount(self, pair: str, current_time: datetime, current_rate: float,
                            proposed_stake: float, **kwargs) -> float:
        try:
            df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
            atr = float(df.iloc[-1]['atr'])
        except Exception:
            return proposed_stake
        if current_rate <= 0 or atr <= 0:
            return proposed_stake
        stop_distance = self.atr_mult_sl * atr
        equity = float(self.wallets.get_total_stake_amount())
        risk_cash = equity * self.risk_per_trade
        stake = risk_cash * (current_rate / stop_distance)
        return float(max(0.0, min(stake, proposed_stake, equity)))

    # ---------- turn “draws” into small wins (custom_exit) ----------
    def custom_exit(self, pair: str, trade: Trade, current_time: datetime,
                    current_rate: float, current_profit: float, **kwargs):
        """
        If momentum dies, skim small profit instead of waiting to breakeven.
        Returns a tag to trigger market exit.
        """
        if current_profit is None:
            return None
        # trade age in minutes
        age = int((current_time - trade.open_date_utc).total_seconds() // 60)

        # 1) Time-decay skim: after 60–240 min, if small green but not strong enough to arm trail
        if 60 <= age <= 240 and 0.001 <= current_profit <= 0.004:
            return "time_skim"

        # 2) Momentum fade: price back under 5m EMA50 while green -> take the win
        try:
            df = self.dp.get_pair_dataframe(pair=pair, timeframe=self.timeframe)
            if df.iloc[-1]['close'] < df.iloc[-1]['ema_fast'] and current_profit > 0.001:
                return "ema_fade"
        except Exception:
            pass

        # 3) Hard timeout to avoid EOB breakeven: after 6h, accept very small profit if available
        if age >= 360 and current_profit > 0.0:
            return "timeout_skim"

        return None

    # live spread-guard unchanged (only applies in live/DR)
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, **kwargs) -> bool:
        if getattr(self.dp, "runmode", None) not in (RunMode.LIVE, RunMode.DRY_RUN):
            return True
        try:
            ob = self.dp.orderbook(pair, 1)
            bid = float(ob['bids'][0][0]); ask = float(ob['asks'][0][0])
            mid = (bid + ask) / 2.0
            spread = (ask - bid) / mid if mid > 0 else 0.0
            return spread <= 0.0007
        except Exception:
            return True
