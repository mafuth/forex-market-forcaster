# !pip install numpy pandas torch scikit-learn ta plotly
import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import RobustScaler
import ta
import plotly.graph_objects as go
import joblib
import math

# ═══════════════════════════════════════════════════════════════════════
# 1. DATA LOADING
# ═══════════════════════════════════════════════════════════════════════
def load_forex_data(csv_file):
    """
    Loads the CSV file into a Pandas DataFrame.
    Ensures it is sorted by date ascending.
    """
    df = pd.read_csv(csv_file)
    df['Gmt time'] = pd.to_datetime(df['Gmt time'], format='mixed')
    df.sort_values(by='Gmt time', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════
# 2. FEATURE ENGINEERING — percentage returns + cyclic time encoding
# ═══════════════════════════════════════════════════════════════════════
def add_features(df):
    """
    Builds features using percentage returns (stationary), technical indicators,
    and cyclic time-of-day / day-of-week encoding for M15 intraday data.
    """
    # --- Decomposed candle components (orthogonal targets) ---
    prev_close = df['Close'].shift(1)
    # These are used as INPUT features (correlated but informative)
    df['open_ret']  = (df['Open']  / prev_close) - 1
    df['high_ret']  = (df['High']  / prev_close) - 1
    df['low_ret']   = (df['Low']   / prev_close) - 1
    df['close_ret'] = (df['Close'] / prev_close) - 1

    # --- TARGETS: independent candle components ---
    # 1. Close return: main price direction
    df['tgt_close_ret'] = (df['Close'] / prev_close) - 1
    # 2. Open gap: how far open deviates from previous close
    df['tgt_open_gap']  = (df['Open'] / prev_close) - 1
    # 3. Upper wick: distance from candle body top to high (always >= 0)
    body_top = df[['Open', 'Close']].max(axis=1)
    df['tgt_high_ext']  = (df['High'] - body_top) / prev_close
    # 4. Lower wick: distance from candle body bottom to low (always >= 0)
    body_bot = df[['Open', 'Close']].min(axis=1)
    df['tgt_low_ext']   = (body_bot - df['Low']) / prev_close

    # --- Candle shape features (scale-free) ---
    candle_range = df['High'] - df['Low']
    candle_range = candle_range.replace(0, np.nan).ffill()
    df['body_ratio']       = (df['Close'] - df['Open']) / candle_range
    df['upper_wick_ratio'] = (df['High'] - df[['Open', 'Close']].max(axis=1)) / candle_range
    df['lower_wick_ratio'] = (df[['Open', 'Close']].min(axis=1) - df['Low']) / candle_range

    # --- Multi-timeframe returns ---
    for period in [4, 8, 16, 32]:  # 1h, 2h, 4h, 8h lookbacks on M15
        df[f'ret_{period}'] = df['Close'].pct_change(period)

    # --- Technical indicators (already percentage/bounded) ---
    df['rsi'] = ta.momentum.rsi(df['Close'], window=14) / 100.0  # normalize to 0-1
    df['rsi_7'] = ta.momentum.rsi(df['Close'], window=7) / 100.0

    stoch = ta.momentum.StochasticOscillator(df['High'], df['Low'], df['Close'], window=14, smooth_window=3)
    df['stoch_k'] = stoch.stoch() / 100.0
    df['stoch_d'] = stoch.stoch_signal() / 100.0

    # Bollinger Band position (where is price within the bands: 0=low, 1=high)
    bollinger = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
    bb_high = bollinger.bollinger_hband()
    bb_low = bollinger.bollinger_lband()
    bb_range = (bb_high - bb_low).replace(0, np.nan).ffill()
    df['bb_position'] = (df['Close'] - bb_low) / bb_range

    # MACD histogram normalized by price
    macd_ind = ta.trend.MACD(close=df['Close'])
    df['macd_hist'] = macd_ind.macd_diff() / df['Close']

    # ATR as percentage of price
    df['atr_pct'] = ta.volatility.average_true_range(df['High'], df['Low'], df['Close'], window=14) / df['Close']

    # --- Volume features (comprehensive) ---
    vol = df['Volume']
    df['vol_ma_ratio'] = vol / vol.rolling(20).mean()
    df['vol_change'] = vol.pct_change()
    df['vol_acceleration'] = df['vol_change'].diff()  # change of change

    # Volume moving averages at different timeframes
    df['vol_ma5_ratio'] = vol / vol.rolling(5).mean()
    df['vol_ma50_ratio'] = vol / vol.rolling(50).mean()

    # On Balance Volume (OBV) — normalized as pct change
    obv = ta.volume.on_balance_volume(df['Close'], vol)
    df['obv_pct'] = obv.pct_change(4)  # OBV momentum over 1 hour

    # Volume-Price Divergence: price going up but volume going down = weak
    df['vol_price_corr'] = df['close_ret'].rolling(8).corr(df['vol_change'])

    # Volume RSI — is volume overbought/oversold?
    df['vol_rsi'] = ta.momentum.rsi(vol, window=14) / 100.0

    # High volume spike detector (z-score)
    vol_mean = vol.rolling(20).mean()
    vol_std = vol.rolling(20).std().replace(0, np.nan).ffill()
    df['vol_zscore'] = (vol - vol_mean) / vol_std

    # VWAP-like feature: cumulative volume-weighted price position
    typical_price = (df['High'] + df['Low'] + df['Close']) / 3
    cum_tp_vol = (typical_price * vol).rolling(20).sum()
    cum_vol = vol.rolling(20).sum()
    vwap = cum_tp_vol / cum_vol.replace(0, np.nan).ffill()
    df['price_vs_vwap'] = (df['Close'] / vwap) - 1

    # --- Cyclic time-of-day encoding (crucial for M15 intraday) ---
    hour_frac = df['Gmt time'].dt.hour + df['Gmt time'].dt.minute / 60.0
    df['hour_sin'] = np.sin(2 * np.pi * hour_frac / 24.0)
    df['hour_cos'] = np.cos(2 * np.pi * hour_frac / 24.0)

    # Day of week encoding
    dow = df['Gmt time'].dt.dayofweek
    df['dow_sin'] = np.sin(2 * np.pi * dow / 5.0)  # 5 trading days
    df['dow_cos'] = np.cos(2 * np.pi * dow / 5.0)

    # --- Price position relative to moving averages ---
    ma20 = df['Close'].rolling(20).mean()
    ma50 = df['Close'].rolling(50).mean()
    df['price_vs_ma20'] = (df['Close'] / ma20) - 1
    df['price_vs_ma50'] = (df['Close'] / ma50) - 1
    df['ma20_vs_ma50'] = (ma20 / ma50) - 1

    # MA slopes as percentage
    df['ma20_slope_pct'] = ma20.pct_change(4)  # slope over 1 hour
    df['ma50_slope_pct'] = ma50.pct_change(4)

    # --- MOMENTUM HACK: Lagged features ---
    # Give the model direct sight of what happened in the previous 3 candles
    for lag in [1, 2, 3]:
        df[f'close_ret_lag_{lag}'] = df['tgt_close_ret'].shift(lag)
        df[f'vol_change_lag_{lag}'] = df['vol_change'].shift(lag)
        df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)

    # Clean up
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.ffill(inplace=True)
    df.bfill(inplace=True)
    return df


# ═══════════════════════════════════════════════════════════════════════
# 3. DATASET — predicts 4 OHLC returns for next N candles
# ═══════════════════════════════════════════════════════════════════════
class ForexDataset(Dataset):
    def __init__(self, features, targets, seq_length=60):
        """
        features: np.array [total_rows, num_features]
        targets:  np.array [total_rows, 4]  (open_ret, high_ret, low_ret, close_ret)
        """
        self.features = features
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        x = self.features[idx : idx + self.seq_length]
        # Target: the OHLC returns of the candle immediately after the sequence
        y = self.targets[idx + self.seq_length]
        return torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


# ═══════════════════════════════════════════════════════════════════════
# 4. ADVANCED MODEL — CNN-GRU + Feature Attention
# ═══════════════════════════════════════════════════════════════════════
class SELayer(nn.Module):
    """Squeeze-and-Excitation block to weigh important features."""
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, s = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class OHLCPredictor(nn.Module):
    def __init__(self, feature_size, hidden_size=256, num_gru_layers=3, nhead=8, dropout=0.15):
        super().__init__()
        
        # 1. CNN Front-end to capture local "shapes"
        self.cnn = nn.Sequential(
            nn.Conv1d(feature_size, hidden_size, kernel_size=3, padding=1),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            SELayer(hidden_size), # Feature Attention
            nn.Conv1d(hidden_size, hidden_size, kernel_size=5, padding=2),
            nn.BatchNorm1d(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # 2. GRU for long-term dependencies
        self.gru = nn.GRU(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_gru_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_gru_layers > 1 else 0,
        )
        self.gru_proj = nn.Linear(hidden_size * 2, hidden_size)

        # 3. Temporal Attention
        self.attn_norm = nn.LayerNorm(hidden_size)
        self.self_attn = nn.MultiheadAttention(hidden_size, nhead, dropout=dropout, batch_first=True)

        # 4. Specialized Residual Prediction Heads
        self.head_close = self._make_residual_head(hidden_size, dropout)
        self.head_gap   = self._make_residual_head(hidden_size, dropout)
        self.head_high  = self._make_residual_head(hidden_size, dropout)
        self.head_low   = self._make_residual_head(hidden_size, dropout)

    def _make_residual_head(self, hidden_size, dropout):
        return nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        # x: [batch, seq_len, features]
        
        # 1. CNN Front-end (Local patterns)
        # Transpose to [batch, features, seq_len] for Conv1d
        x_cnn = x.transpose(1, 2)
        x_cnn = self.cnn(x_cnn)
        x_cnn = x_cnn.transpose(1, 2) # Back to [batch, seq_len, hidden]

        # 2. GRU (Temporal dependencies)
        gru_out, _ = self.gru(x_cnn)
        gru_out = self.gru_proj(gru_out)

        # 3. Temporal Attention (Global patterns)
        normed = self.attn_norm(gru_out)
        attn_out, _ = self.self_attn(normed, normed, normed)
        
        # 4. Residual Connection (Global + Local context)
        combined = attn_out[:, -1, :] + gru_out[:, -1, :]

        # 5. Specialized Heads
        return torch.cat([
            self.head_close(combined),
            self.head_gap(combined),
            self.head_high(combined),
            self.head_low(combined)
        ], dim=-1)


# ═══════════════════════════════════════════════════════════════════════
# 5. CUSTOM LOSS — penalizes wrong direction + variance collapse
# ═══════════════════════════════════════════════════════════════════════
class OHLCLoss(nn.Module):
    """
    Combines:
    1. MSE loss on scaled targets (magnitude)
    2. Directional penalty (wrong trend direction)
    3. Variance penalty (prevents mean-collapse: all predictions identical)
    """
    def __init__(self, direction_weight=0.8, variance_weight=0.5):
        super().__init__()
        self.mse = nn.MSELoss()
        self.direction_weight = direction_weight
        self.variance_weight = variance_weight

    def forward(self, pred, target):
        # 1. Magnitude loss (MSE on scaled values — targets are now properly scaled)
        mag_loss = self.mse(pred, target)

        # 2. Directional penalty: penalize when sign of close_ret (idx 0) is wrong
        pred_sign = torch.sign(pred[:, 0])
        target_sign = torch.sign(target[:, 0])
        wrong = (pred_sign != target_sign).float()
        mag_loss = mag_loss + self.direction_weight * wrong.mean()

        # 3. Variance penalty: if batch prediction std is too low, penalize
        #    This prevents the model from outputting the same value for all samples
        pred_std = pred.std(dim=0).mean()  # average std across the 4 outputs
        target_std = target.std(dim=0).mean()
        # Penalize when pred variance is much lower than target variance
        variance_ratio = pred_std / (target_std + 1e-8)
        variance_penalty = torch.clamp(1.0 - variance_ratio, min=0.0)  # only penalize if pred_std < target_std

        return mag_loss + self.variance_weight * variance_penalty


# ═══════════════════════════════════════════════════════════════════════
# 6. TRAINING — with LR scheduling, gradient clipping, early stopping
# ═══════════════════════════════════════════════════════════════════════
def train_model(model, train_loader, val_loader, lr=1e-3, epochs=100, device='cpu'):
    criterion = OHLCLoss(direction_weight=0.3, variance_weight=0.5)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader),
        pct_start=0.2, anneal_strategy='cos'
    )
    model.to(device)

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0
    patience = 20

    for epoch in range(epochs):
        # --- Train ---
        model.train()
        train_losses = []
        for x_batch, y_batch in train_loader:
            x_batch, y_batch = x_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            output = model(x_batch)
            loss = criterion(output, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()
            scheduler.step()
            train_losses.append(loss.item())

        # --- Validate ---
        model.eval()
        val_losses = []
        correct_dirs = 0
        total_dirs = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                x_val, y_val = x_val.to(device), y_val.to(device)
                out_val = model(x_val)
                val_losses.append(criterion(out_val, y_val).item())
                # Track direction accuracy on close return
                correct_dirs += ((torch.sign(out_val[:, 3]) == torch.sign(y_val[:, 3])).sum().item())
                total_dirs += y_val.size(0)

        mean_train = np.mean(train_losses)
        mean_val = np.mean(val_losses)
        dir_acc = correct_dirs / max(total_dirs, 1) * 100
        cur_lr = optimizer.param_groups[0]['lr']

        # if (epoch + 1) % 5 == 0 or epoch == 0:
        print(f"Epoch [{epoch+1:3d}/{epochs}] Train: {mean_train:.6f} | Val: {mean_val:.6f} | DirAcc: {dir_acc:.1f}% | LR: {cur_lr:.2e}")

        # Early stopping
        if mean_val < best_val_loss:
            best_val_loss = mean_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)
        model.to(device)
        print(f"  Restored best model (val loss: {best_val_loss:.6f})")

    return model


# ═══════════════════════════════════════════════════════════════════════
# 7. EVALUATION — overlaid candlestick chart
# ═══════════════════════════════════════════════════════════════════════
def evaluate_and_plot(model, test_loader, target_scaler, full_df, test_start_idx, seq_length, device='cpu'):
    """
    Runs inference, inverse-transforms scaled predictions back to returns,
    reconstructs absolute OHLC prices, and plots an overlaid candlestick chart.
    """
    model.eval()
    all_preds = []
    all_reals = []

    with torch.no_grad():
        for x_batch, y_batch in test_loader:
            x_batch = x_batch.to(device)
            preds = model(x_batch).cpu().numpy()
            all_preds.append(preds)
            all_reals.append(y_batch.numpy())

    all_preds_scaled = np.concatenate(all_preds, axis=0)  # [N, 4] — still in scaled space
    all_reals_scaled = np.concatenate(all_reals, axis=0)

    # Inverse-transform predictions and actuals back to real return space
    all_preds = target_scaler.inverse_transform(all_preds_scaled)
    all_reals = target_scaler.inverse_transform(all_reals_scaled)

    # Reconstruct absolute prices from decomposed candle components
    # Targets: [close_ret, open_gap, high_ext, low_ext]
    pred_opens, pred_highs, pred_lows, pred_closes = [], [], [], []
    actual_opens, actual_highs, actual_lows, actual_closes = [], [], [], []
    timestamps = []

    for i in range(len(all_preds)):
        data_idx = test_start_idx + seq_length + i
        if data_idx >= len(full_df):
            break

        prev_close = full_df['Close'].iloc[data_idx - 1]

        # Decompose predictions: [close_ret, open_gap, high_ext, low_ext]
        p_close_ret = all_preds[i, 0]
        p_open_gap  = all_preds[i, 1]
        p_high_ext  = max(all_preds[i, 2], 0)  # wicks must be >= 0
        p_low_ext   = max(all_preds[i, 3], 0)

        # Reconstruct absolute prices
        p_close = prev_close * (1 + p_close_ret)
        p_open  = prev_close * (1 + p_open_gap)
        body_top = max(p_open, p_close)
        body_bot = min(p_open, p_close)
        p_high = body_top + prev_close * p_high_ext
        p_low  = body_bot - prev_close * p_low_ext

        pred_opens.append(p_open)
        pred_highs.append(p_high)
        pred_lows.append(p_low)
        pred_closes.append(p_close)

        actual_opens.append(full_df['Open'].iloc[data_idx])
        actual_highs.append(full_df['High'].iloc[data_idx])
        actual_lows.append(full_df['Low'].iloc[data_idx])
        actual_closes.append(full_df['Close'].iloc[data_idx])
        timestamps.append(full_df['Gmt time'].iloc[data_idx])

    pred_opens  = np.array(pred_opens)
    pred_highs  = np.array(pred_highs)
    pred_lows   = np.array(pred_lows)
    pred_closes = np.array(pred_closes)
    actual_opens  = np.array(actual_opens)
    actual_highs  = np.array(actual_highs)
    actual_lows   = np.array(actual_lows)
    actual_closes = np.array(actual_closes)

    # ─── Metrics ──────────────────────────────────────────────────────
    print(f"\n{'═'*60}")
    print(f"  MODEL EVALUATION — Reconstructed OHLC Candles")
    print(f"{'═'*60}")
    for name, actual, pred in [
        ('Open',  actual_opens,  pred_opens),
        ('High',  actual_highs,  pred_highs),
        ('Low',   actual_lows,   pred_lows),
        ('Close', actual_closes, pred_closes),
    ]:
        mse = np.mean((actual - pred) ** 2)
        mae = np.mean(np.abs(actual - pred))
        mape = np.mean(np.abs((actual - pred) / actual)) * 100
        print(f"  {name:5s} — MAE: {mae:.2f}  |  RMSE: {np.sqrt(mse):.2f}  |  MAPE: {mape:.3f}%")

    # Direction accuracy
    actual_dir = (actual_closes > actual_opens).astype(int)
    pred_dir   = (pred_closes > pred_opens).astype(int)
    dir_acc = np.mean(actual_dir == pred_dir) * 100
    print(f"\n  Candle Direction Accuracy: {dir_acc:.1f}%")

    # Close direction (up/down from previous close)
    actual_close_dir = (actual_closes > np.roll(actual_closes, 1))[1:]
    pred_close_dir   = (pred_closes > np.roll(actual_closes, 1))[1:]  # anchored on real prev close
    close_dir_acc = np.mean(actual_close_dir == pred_close_dir) * 100
    print(f"  Close Up/Down Accuracy:   {close_dir_acc:.1f}%")
    print(f"{'═'*60}\n")

    # ─── Single Overlaid Candlestick Chart ────────────────────────────
    display_count = min(300, len(timestamps))
    sl = slice(-display_count, None)

    fig = go.Figure()

    # Actual market candles
    fig.add_trace(go.Candlestick(
        x=list(range(display_count)),
        open=actual_opens[sl],
        high=actual_highs[sl],
        low=actual_lows[sl],
        close=actual_closes[sl],
        increasing_line_color='#26a69a',
        decreasing_line_color='#ef5350',
        increasing_fillcolor='#26a69a',
        decreasing_fillcolor='#ef5350',
        name='Actual (Market)',
        opacity=0.55,
    ))

    # Predicted candles overlaid
    fig.add_trace(go.Candlestick(
        x=list(range(display_count)),
        open=pred_opens[sl],
        high=pred_highs[sl],
        low=pred_lows[sl],
        close=pred_closes[sl],
        increasing_line_color='#ab47bc',
        decreasing_line_color='#29b6f6',
        increasing_fillcolor='#ab47bc',
        decreasing_fillcolor='#29b6f6',
        name='Predicted (Model)',
        opacity=0.9,
    ))

    fig.update_layout(
        template='plotly_dark',
        paper_bgcolor='#131722',
        plot_bgcolor='#131722',
        font=dict(color='#d1d4dc', size=12),
        title=dict(
            text='XAUUSD — Actual vs Predicted Candles (Overlay)',
            font=dict(size=20, color='#e0e3eb'),
            x=0.5,
        ),
        height=700,
        margin=dict(l=60, r=60, t=80, b=60),
        xaxis_rangeslider_visible=False,
        legend=dict(
            orientation='h', yanchor='bottom', y=1.01,
            xanchor='center', x=0.5,
            font=dict(size=13), bgcolor='rgba(0,0,0,0)',
        ),
        xaxis=dict(gridcolor='#1e222d', zeroline=False, title='Candle Index'),
        yaxis=dict(gridcolor='#1e222d', zeroline=False, title='Price'),
    )

    fig.show()
    print("Overlaid candlestick chart displayed.")


# ═══════════════════════════════════════════════════════════════════════
# 8. EXECUTION PIPELINE
# ═══════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    csv_file = r"https://github.com/mafuth/forex-market-forcaster/raw/refs/heads/main/XAUUSD_M15.csv"
    print("Loading data...")
    df = load_forex_data(csv_file)
    print(f"  Loaded {len(df)} rows")
    df = add_features(df)

    # ─── Define features and targets ──────────────────────────────────
    target_cols = ['tgt_close_ret', 'tgt_open_gap', 'tgt_high_ext', 'tgt_low_ext']

    feature_cols = [
        # Returns & candle shape
        'open_ret', 'high_ret', 'low_ret', 'close_ret',
        'body_ratio', 'upper_wick_ratio', 'lower_wick_ratio',
        # Multi-timeframe returns
        'ret_4', 'ret_8', 'ret_16', 'ret_32',
        # Momentum oscillators (already 0-1)
        'rsi', 'rsi_7', 'stoch_k', 'stoch_d',
        # Volatility & band position
        'bb_position', 'atr_pct', 'macd_hist',
        # Volume features (comprehensive)
        'vol_ma_ratio', 'vol_change', 'vol_acceleration',
        'vol_ma5_ratio', 'vol_ma50_ratio',
        'obv_pct', 'vol_price_corr', 'vol_rsi',
        'vol_zscore', 'price_vs_vwap',
        # Time encoding (cyclic)
        'hour_sin', 'hour_cos', 'dow_sin', 'dow_cos',
        # Price relative to MAs
        'price_vs_ma20', 'price_vs_ma50', 'ma20_vs_ma50',
        'ma20_slope_pct', 'ma50_slope_pct',
        # Lagged momentum hacks
        'close_ret_lag_1', 'close_ret_lag_2', 'close_ret_lag_3',
        'vol_change_lag_1', 'vol_change_lag_2', 'vol_change_lag_3',
        'rsi_lag_1', 'rsi_lag_2', 'rsi_lag_3',
    ]

    seq_length = 100  # 100 x 15min = 25 hours of lookback (full market day)

    # ─── Scale features AND targets ─────────────────────────────────────
    feature_data = df[feature_cols].values
    target_data  = df[target_cols].values

    # Use RobustScaler — less sensitive to outliers in financial data
    feature_scaler = RobustScaler()
    target_scaler  = RobustScaler()   # <-- KEY FIX: scale targets too!

    # Split points
    total_samples = len(df) - seq_length
    train_end = int(total_samples * 0.70)
    val_end   = int(total_samples * 0.85)

    # Fit scalers on training portion only
    feature_scaler.fit(feature_data[:train_end + seq_length])
    target_scaler.fit(target_data[:train_end + seq_length])

    features_scaled = feature_scaler.transform(feature_data)
    targets_scaled  = target_scaler.transform(target_data)

    # Clip extreme values after scaling (reduces outlier impact)
    features_scaled = np.clip(features_scaled, -5, 5)
    targets_scaled  = np.clip(targets_scaled, -5, 5)

    print(f"  Features: {len(feature_cols)}, Sequence length: {seq_length}")
    print(f"  Train: {train_end}, Val: {val_end - train_end}, Test: {total_samples - val_end}")
    print(f"  Target scale (median): {target_scaler.center_}")
    print(f"  Target scale (IQR):    {target_scaler.scale_}")

    # ─── Create datasets (using SCALED targets) ──────────────────────
    train_dataset = ForexDataset(features_scaled[:train_end + seq_length],
                                 targets_scaled[:train_end + seq_length],
                                 seq_length)
    val_dataset   = ForexDataset(features_scaled[train_end:val_end + seq_length],
                                 targets_scaled[train_end:val_end + seq_length],
                                 seq_length)
    test_dataset  = ForexDataset(features_scaled[val_end:],
                                 targets_scaled[val_end:],
                                 seq_length)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True,  drop_last=True)
    val_loader   = DataLoader(val_dataset,   batch_size=32, shuffle=False)
    test_loader  = DataLoader(test_dataset,  batch_size=32, shuffle=False)

    # ─── Build & train model ──────────────────────────────────────────
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"  Device: {device}\n")

    model = OHLCPredictor(
        feature_size=len(feature_cols),
        hidden_size=256,
        num_gru_layers=3,
        nhead=8,
        dropout=0.15,
    )
    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Model parameters: {total_params:,}\n")

    trained_model = train_model(model, train_loader, val_loader, lr=1e-3, epochs=150, device=device)

    # Save
    torch.save(trained_model.state_dict(), "ohlc_model.pth")
    joblib.dump(feature_scaler, "feature_scaler.pkl")
    joblib.dump(target_scaler, "target_scaler.pkl")
    print("\nModel saved to ohlc_model.pth")
    print("Scalers saved to feature_scaler.pkl / target_scaler.pkl")

    # ─── Evaluate & plot ──────────────────────────────────────────────
    evaluate_and_plot(trained_model, test_loader, target_scaler=target_scaler, full_df=df,
                      test_start_idx=val_end, seq_length=seq_length, device=device)