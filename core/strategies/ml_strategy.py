"""ML strategy — LightGBM ensemble model for BTC 5-min binary prediction.

Uses a pre-trained LightGBM model with 25 features derived from:
- 5m BTC/USDT spot candles (MEXC via ccxt)
- 15m BTC/USDT futures candles (MEXC via ccxt)
- 1h BTC/USDT futures candles (MEXC via ccxt)
- Funding rate (MEXC futures via ccxt)
- CVD (MEXC futures REST API via httpx)
- Time-of-day features

The model outputs P(UP) — probability the next candle closes higher.
Threshold logic: prob >= threshold -> UP, prob <= (1-threshold) -> DOWN, else SKIP.
"""

from __future__ import annotations

import asyncio
import logging
import math
from datetime import datetime, timezone
from typing import Any

import httpx
import numpy as np
import pandas as pd

import config as cfg
from core.strategies.base import BaseStrategy
from polymarket.markets import get_next_slot_info, get_slot_prices

log = logging.getLogger(__name__)

# Feature order MUST match training exactly
FEATURE_COLS = [
    'body_ratio_n1', 'body_ratio_n2', 'body_ratio_n3',
    'upper_wick_n1', 'upper_wick_n2',
    'lower_wick_n1', 'lower_wick_n2',
    'volume_ratio_n1', 'volume_ratio_n2',
    'body_ratio_15m', 'dir_15m', 'volume_ratio_15m',
    'body_ratio_1h', 'dir_1h', 'ema9_slope_1h',
    'funding_rate', 'funding_zscore',
    'delta_ratio', 'cvd_delta', 'cvd_5', 'cvd_20', 'cvd_trend',
    'hour_sin', 'hour_cos', 'day_of_week',
]


class MLStrategy(BaseStrategy):
    """LightGBM ML model strategy for BTC 5-min binary options."""

    def __init__(self):
        import lightgbm as lgb
        import ccxt

        self.threshold = cfg.ML_THRESHOLD
        model_path = cfg.ML_MODEL_PATH
        log.info("MLStrategy: loading model from %s (threshold=%.3f)", model_path, self.threshold)
        self.model = lgb.Booster(model_file=model_path)
        log.info("MLStrategy: model loaded successfully")

        # Initialize MEXC exchanges — public endpoints, no API key needed
        self.spot_exchange = ccxt.mexc({'enableRateLimit': True})
        self.futures_exchange = ccxt.mexc({
            'enableRateLimit': True,
            'options': {'defaultType': 'swap'},
        })

        # Load markets once at init
        self.spot_exchange.load_markets()
        self.futures_exchange.load_markets()
        log.info("MLStrategy: MEXC markets loaded (spot + futures)")

    # ------------------------------------------------------------------
    # ATR14 helper
    # ------------------------------------------------------------------
    @staticmethod
    def _atr14(df: pd.DataFrame) -> pd.Series:
        """True Range based ATR with period 14."""
        h = df['high']
        l = df['low']
        c_prev = df['close'].shift(1)
        tr = pd.concat([
            h - l,
            (h - c_prev).abs(),
            (l - c_prev).abs(),
        ], axis=1).max(axis=1)
        return tr.rolling(14).mean()

    # ------------------------------------------------------------------
    # Data fetching
    # ------------------------------------------------------------------
    async def _fetch_5m_spot(self) -> pd.DataFrame | None:
        """Fetch ~50 recent 5m BTC/USDT spot candles from MEXC."""
        try:
            candles = await asyncio.to_thread(
                self.spot_exchange.fetch_ohlcv, 'BTC/USDT', '5m', None, 50
            )
            if not candles or len(candles) < 20:
                log.error("MLStrategy: insufficient 5m spot candles (%d)", len(candles) if candles else 0)
                return None
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            return df
        except Exception:
            log.exception("MLStrategy: failed to fetch 5m spot candles")
            return None

    async def _fetch_15m_futures(self) -> pd.DataFrame | None:
        """Fetch ~30 recent 15m BTC/USDT futures candles from MEXC."""
        try:
            candles = await asyncio.to_thread(
                self.futures_exchange.fetch_ohlcv, 'BTC/USDT:USDT', '15m', None, 30
            )
            if not candles or len(candles) < 20:
                log.error("MLStrategy: insufficient 15m futures candles (%d)", len(candles) if candles else 0)
                return None
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            return df
        except Exception:
            log.exception("MLStrategy: failed to fetch 15m futures candles")
            return None

    async def _fetch_1h_futures(self) -> pd.DataFrame | None:
        """Fetch ~25 recent 1h BTC/USDT futures candles from MEXC."""
        try:
            candles = await asyncio.to_thread(
                self.futures_exchange.fetch_ohlcv, 'BTC/USDT:USDT', '1h', None, 25
            )
            if not candles or len(candles) < 15:
                log.error("MLStrategy: insufficient 1h futures candles (%d)", len(candles) if candles else 0)
                return None
            df = pd.DataFrame(candles, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
            return df
        except Exception:
            log.exception("MLStrategy: failed to fetch 1h futures candles")
            return None

    async def _fetch_funding(self) -> dict | None:
        """Fetch latest funding rate data from MEXC futures."""
        try:
            rates = await asyncio.to_thread(
                self.futures_exchange.fetch_funding_rate_history, 'BTC/USDT:USDT', None, 50
            )
            if not rates:
                log.error("MLStrategy: no funding rate data returned")
                return None
            funding_values = [r['fundingRate'] for r in rates if r.get('fundingRate') is not None]
            if not funding_values:
                log.error("MLStrategy: no valid funding rate values")
                return None
            current_rate = funding_values[-1]
            # Rolling z-score over available values
            if len(funding_values) >= 2:
                arr = np.array(funding_values)
                mean = arr.mean()
                std = arr.std()
                zscore = (current_rate - mean) / std if std > 0 else 0.0
            else:
                zscore = 0.0
            return {'funding_rate': current_rate, 'funding_zscore': zscore}
        except Exception:
            log.exception("MLStrategy: failed to fetch funding rate")
            return None

    async def _fetch_cvd(self) -> dict | None:
        """Fetch CVD data from MEXC futures REST API (NOT ccxt)."""
        try:
            now = int(datetime.now(timezone.utc).timestamp())
            start = now - 30 * 5 * 60  # ~30 candles of 5m = 2.5 hours
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    'https://contract.mexc.com/api/v1/contract/kline/BTC_USDT',
                    params={'interval': 'Min5', 'start': start, 'end': now},
                )
                resp.raise_for_status()
                data = resp.json()

            if not data.get('success') or not data.get('data'):
                log.error("MLStrategy: CVD API returned unsuccessful or empty response")
                return None

            kline = data['data']
            times = kline.get('time', [])
            opens = kline.get('open', [])
            highs = kline.get('high', [])
            lows = kline.get('low', [])
            closes = kline.get('close', [])
            vols = kline.get('vol', [])

            if len(times) < 22:  # need at least 21 closed + 1 live
                log.error("MLStrategy: insufficient CVD candles (%d)", len(times))
                return None

            # Compute buy/sell volume delta per candle
            deltas = []
            volumes = []
            for i in range(len(times)):
                h = float(highs[i])
                l = float(lows[i])
                c = float(closes[i])
                v = float(vols[i])
                if h == l:
                    buy_vol = v / 2
                    sell_vol = v / 2
                else:
                    buy_vol = v * (c - l) / (h - l)
                    sell_vol = v * (h - c) / (h - l)
                deltas.append(buy_vol - sell_vol)
                volumes.append(v)

            # Last candle is live (N), use N-1 as last closed
            # Index -1 is live, -2 is N-1 (last closed)
            n1_idx = len(deltas) - 2  # N-1

            delta_n1 = deltas[n1_idx]
            vol_n1 = volumes[n1_idx]
            delta_ratio = delta_n1 / vol_n1 if vol_n1 > 0 else 0.0

            # cvd_5: sum of last 5 closed candles' deltas (N-1 to N-5)
            cvd_5 = sum(deltas[max(0, n1_idx - 4):n1_idx + 1])

            # cvd_20: sum of last 20 closed candles' deltas (N-1 to N-20)
            cvd_20 = sum(deltas[max(0, n1_idx - 19):n1_idx + 1])

            cvd_trend = cvd_5 - cvd_20

            return {
                'delta_ratio': delta_ratio,
                'cvd_delta': delta_n1,
                'cvd_5': cvd_5,
                'cvd_20': cvd_20,
                'cvd_trend': cvd_trend,
            }
        except Exception:
            log.exception("MLStrategy: failed to fetch CVD data")
            return None

    # ------------------------------------------------------------------
    # Feature engineering
    # ------------------------------------------------------------------
    def _build_features(
        self,
        df_5m: pd.DataFrame,
        df_15m: pd.DataFrame,
        df_1h: pd.DataFrame,
        funding_data: dict,
        cvd_data: dict,
    ) -> list[float] | None:
        """Build the 25-feature vector in exact training order."""
        try:
            # --- A. 5-Minute Candle Structure (9 features) ---
            atr = self._atr14(df_5m)
            body = df_5m['close'] - df_5m['open']
            upper_wick = df_5m['high'] - df_5m[['open', 'close']].max(axis=1)
            lower_wick = df_5m[['open', 'close']].min(axis=1) - df_5m['low']
            vol_ma = df_5m['volume'].rolling(20).mean()

            body_ratio = body / atr
            upper_wick_ratio = upper_wick / atr
            lower_wick_ratio = lower_wick / atr
            volume_ratio = df_5m['volume'] / vol_ma

            body_ratio_n1 = body_ratio.iloc[-2]
            body_ratio_n2 = body_ratio.iloc[-3]
            body_ratio_n3 = body_ratio.iloc[-4]
            upper_wick_n1 = upper_wick_ratio.iloc[-2]
            upper_wick_n2 = upper_wick_ratio.iloc[-3]
            lower_wick_n1 = lower_wick_ratio.iloc[-2]
            lower_wick_n2 = lower_wick_ratio.iloc[-3]
            volume_ratio_n1 = volume_ratio.iloc[-2]
            volume_ratio_n2 = volume_ratio.iloc[-3]

            # --- B. 15-Minute Futures Features (3 features) ---
            atr_15m = self._atr14(df_15m)
            now_utc = datetime.now(timezone.utc)
            last_15m_ts = df_15m['timestamp'].iloc[-1]
            # If last candle started less than 15 min ago, it's still live
            if last_15m_ts + pd.Timedelta(minutes=15) > now_utc:
                idx_15m = -2
            else:
                idx_15m = -1

            body_ratio_15m = ((df_15m['close'] - df_15m['open']) / atr_15m).iloc[idx_15m]
            dir_15m = 1 if df_15m['close'].iloc[idx_15m] > df_15m['open'].iloc[idx_15m] else -1
            vol_ma_15m = df_15m['volume'].rolling(20).mean()
            volume_ratio_15m = (df_15m['volume'] / vol_ma_15m).iloc[idx_15m]

            # --- C. 1-Hour Futures Features (3 features) ---
            atr_1h = self._atr14(df_1h)
            last_1h_ts = df_1h['timestamp'].iloc[-1]
            if last_1h_ts + pd.Timedelta(hours=1) > now_utc:
                idx_1h = -2
            else:
                idx_1h = -1

            body_ratio_1h = ((df_1h['close'] - df_1h['open']) / atr_1h).iloc[idx_1h]
            dir_1h = 1 if df_1h['close'].iloc[idx_1h] > df_1h['open'].iloc[idx_1h] else -1
            ema9 = df_1h['close'].ewm(span=9, adjust=False).mean()
            ema9_slope_1h = ((ema9 - ema9.shift(1)) / atr_1h).iloc[idx_1h]

            # --- D. Funding Rate Features (2 features) ---
            funding_rate = funding_data['funding_rate']
            funding_zscore = funding_data['funding_zscore']

            # --- E. CVD Features (5 features) ---
            delta_ratio = cvd_data['delta_ratio']
            cvd_delta = cvd_data['cvd_delta']
            cvd_5 = cvd_data['cvd_5']
            cvd_20 = cvd_data['cvd_20']
            cvd_trend = cvd_data['cvd_trend']

            # --- F. Time Features (3 features) ---
            hour = now_utc.hour + now_utc.minute / 60.0
            hour_sin = math.sin(2 * math.pi * hour / 24)
            hour_cos = math.cos(2 * math.pi * hour / 24)
            day_of_week = now_utc.weekday()  # 0=Monday, 6=Sunday

            # Assemble in exact training order
            features = [
                body_ratio_n1, body_ratio_n2, body_ratio_n3,
                upper_wick_n1, upper_wick_n2,
                lower_wick_n1, lower_wick_n2,
                volume_ratio_n1, volume_ratio_n2,
                body_ratio_15m, dir_15m, volume_ratio_15m,
                body_ratio_1h, dir_1h, ema9_slope_1h,
                funding_rate, funding_zscore,
                delta_ratio, cvd_delta, cvd_5, cvd_20, cvd_trend,
                hour_sin, hour_cos, day_of_week,
            ]

            # Validate — no NaN or inf
            for i, val in enumerate(features):
                v = float(val)
                if math.isnan(v) or math.isinf(v):
                    log.warning(
                        "MLStrategy: feature '%s' (idx %d) is %s",
                        FEATURE_COLS[i], i, val,
                    )
                    return None
                features[i] = v

            log.debug("MLStrategy: features built — %s", dict(zip(FEATURE_COLS, features)))
            return features

        except Exception:
            log.exception("MLStrategy: feature computation failed")
            return None

    # ------------------------------------------------------------------
    # Main signal generation
    # ------------------------------------------------------------------
    async def check_signal(self) -> dict[str, Any] | None:
        """Generate ML-based signal for slot N+1. Called at T-85s."""
        try:
            # 1. Slot info
            slot = get_next_slot_info()
            if slot is None:
                log.error("MLStrategy: could not get next slot info")
                return None

            base = {
                'slot_n1_start_full': slot['slot_start_full'],
                'slot_n1_end_full':   slot['slot_end_full'],
                'slot_n1_start_str':  slot['slot_start_str'],
                'slot_n1_end_str':    slot['slot_end_str'],
                'slot_n1_ts':         slot['slot_start_ts'],
                'slot_n1_slug':       slot['slug'],
            }

            # 2. Fetch data (all 5 sources)
            df_5m, df_15m, df_1h, funding_data, cvd_data = await asyncio.gather(
                self._fetch_5m_spot(),
                self._fetch_15m_futures(),
                self._fetch_1h_futures(),
                self._fetch_funding(),
                self._fetch_cvd(),
            )

            if any(x is None for x in [df_5m, df_15m, df_1h, funding_data, cvd_data]):
                log.error("MLStrategy: data fetch failed for one or more sources")
                return None

            # 3. Build features
            features = self._build_features(df_5m, df_15m, df_1h, funding_data, cvd_data)
            if features is None:
                return {
                    **base,
                    'skipped': True,
                    'reason': 'Feature computation failed (NaN/inf detected)',
                    'pattern': None,
                }

            # 4. Predict
            prob = self.model.predict(np.array([features]))[0]
            log.info("MLStrategy: P(UP) = %.4f (threshold=%.3f)", prob, self.threshold)

            # 5. Threshold
            if prob >= self.threshold:
                side = 'Up'
                pattern = 'ML_UP'
            elif prob <= (1 - self.threshold):
                side = 'Down'
                pattern = 'ML_DOWN'
            else:
                return {
                    **base,
                    'skipped': True,
                    'reason': f'ML confidence below threshold ({prob:.3f})',
                    'pattern': None,
                }

            # 6. Polymarket prices
            prices = await get_slot_prices(slot['slug'])
            if prices is None:
                log.error("MLStrategy: could not get Polymarket prices for slug %s", slot['slug'])
                return None

            if side == 'Up':
                entry_price = prices['up_price']
                opposite_price = prices['down_price']
                token_id = prices['up_token_id']
            else:
                entry_price = prices['down_price']
                opposite_price = prices['up_price']
                token_id = prices['down_token_id']

            log.info(
                "MLStrategy: SIGNAL %s (prob=%.4f) | slot %s-%s UTC | entry=$%.4f",
                pattern, prob, slot['slot_start_str'], slot['slot_end_str'], entry_price,
            )

            # 7. Return signal
            return {
                **base,
                'skipped': False,
                'side': side,
                'entry_price': entry_price,
                'opposite_price': opposite_price,
                'token_id': token_id,
                'pattern': pattern,
            }

        except Exception:
            log.exception("MLStrategy.check_signal() unexpected error")
            return None
