# Cryptocurrency Price Direction Prediction Model: Using logistic regression with technical indicators to predict next-day price movements

# Installing required libraries
!pip install ta yfinance scikit-learn matplotlib seaborn -q

import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import ta
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
import warnings
warnings.filterwarnings('ignore')

# asset-specific feature configurations - some assets performing better with fewer, cleaner features to reduce noise
asset_features = {
  'DOT-USD': [
    'rsi_14', 'macd', 'sma_7', 'sma_14', 'momentum',
    'bb_position', 'ret_1', 'sma_crossover'
  ],
  'AVAX-USD': [
    'rsi_14', 'rsi_7', 'macd', 'sma_7', 'momentum',
    'bb_position', 'ret_1', 'vol_7'
  ],
  'default': [
    'rsi_14', 'rsi_7', 'macd', 'sma_7', 'ema_7',
    'sma_14', 'ema_14', 'vol_7', 'momentum', 'bb_width', 'bb_position',
    'ret_1', 'ret_3', 'range_3',
    'rsi_14_lag1', 'macd_lag1', 'sma_7_lag1', 'bb_width_lag1',
    'rsi_macd_interaction', 'momentum_volatility_ratio', 'sma_crossover'
  ]
}

# training window and threshold configurations per asset
#(problematic assets uses larger training windows and adjusted thresholds)
asset_config = {
  'DOT-USD': {
    'min_train': 18,
    'threshold_big_move': 0.008,
    'model_type': 'lr',
    'skip_if_zeros': True
  },
  'AVAX-USD': {
    'min_train': 16,
    'threshold_big_move': 0.003,
    'model_type': 'lr',
    'skip_if_zeros': False
  },
  'default': {
    'min_train': 12,
    'threshold_big_move': 0.005,
    'model_type': 'lr',
    'skip_if_zeros': False
  }
}

# defining cryptocurrency tickers and data parameters
tickers = [
  'BTC-USD', 'ETH-USD', 'ADA-USD', 'SOL-USD', 'XRP-USD',
  'DOGE-USD', 'DOT-USD', 'LTC-USD', 'LINK-USD', 'AVAX-USD'
]
days_to_fetch = 60  # fetching extra data for indicator calculation
model_days = 30     # using most recent 30 days for modeling

def fetch_crypto_data(tickers, days):
  # fetching OHLCV data for cryptocurrency tickers from Yahoo Finance.
  # handling various column naming conventions and standardizing output.
  today = pd.Timestamp('today').normalize()
  start = today - pd.Timedelta(days=days+2)
  data = {}

  for ticker in tickers:
    try:
      df = yf.download(ticker, start=start, end=today, interval='1d', progress=False)
      if df.empty:
        print(f"No data for {ticker}")
        continue
      df = df.reset_index()

      # standardizing column names to lowercase
      standardized_cols = []
      for col in df.columns:
        if isinstance(col, tuple):
          cleaned = '_'.join(filter(None, [str(s).strip() for s in col]))
        else:
          cleaned = str(col)
        standardized_cols.append(cleaned.lower().replace(' ', '_').replace('-', '_').replace('.', ''))
      df.columns = standardized_cols

      # ensuring date column exists
      if 'date' not in df.columns and 'index' in df.columns:
        df = df.rename(columns={'index': 'date'})
      if 'date' not in df.columns:
        continue
      df['date'] = pd.to_datetime(df['date'])

      # ensuring close price column exists
      close_col = None
      if 'close' in df.columns:
        close_col = 'close'
      elif 'adj_close' in df.columns:
        close_col = 'adj_close'
      else:
        for col in df.columns:
          if col.startswith('close_') or col.startswith('adj_close_'):
            close_col = col
            break

      if close_col and close_col != 'close':
        df = df.rename(columns={close_col: 'close'})
      elif not close_col:
        continue

      # ensuring all OHLCV columns exist
      for col_name in ['open', 'high', 'low', 'volume']:
        if col_name not in df.columns:
          if col_name == 'volume':
            df['volume'] = 0
          else:
            df[col_name] = df['close']

      df['asset'] = ticker
      df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'asset']]
      data[ticker] = df.copy()
      print(f"Loaded {ticker}: {len(df)} rows")

    except Exception as e:
      print(f"Error loading {ticker}: {e}")

  return data

# fetching data for all tickers
all_data = fetch_crypto_data(tickers, days_to_fetch)

def calculate_technical_indicators(df):
  """
  Calculating technical indicators including RSI, MACD, moving averages,
  Bollinger Bands, momentum, and custom interaction features.
  """
  df = df.copy()

  # momentum indicators
  df['rsi_14'] = ta.momentum.rsi(df['close'], window=14)
  df['rsi_7'] = ta.momentum.rsi(df['close'], window=7)
  df['macd'] = ta.trend.macd_diff(df['close'])
  df['momentum'] = ta.momentum.roc(df['close'], window=10)

  # moving averages
  df['sma_7'] = ta.trend.sma_indicator(df['close'], window=7)
  df['ema_7'] = ta.trend.ema_indicator(df['close'], window=7)
  df['sma_14'] = ta.trend.sma_indicator(df['close'], window=14)
  df['ema_14'] = ta.trend.ema_indicator(df['close'], window=14)

  # volatility indicators
  df['vol_7'] = df['close'].rolling(7).std()
  df['bb_width'] = (ta.volatility.bollinger_hband(df['close'], window=20) -
                    ta.volatility.bollinger_lband(df['close'], window=20)) / df['close']
  df['bb_position'] = ((df['close'] - ta.volatility.bollinger_lband(df['close'], window=20)) /
                       (ta.volatility.bollinger_hband(df['close'], window=20) -
                        ta.volatility.bollinger_lband(df['close'], window=20)))

  # price change features
  df['ret_1'] = df['close'].pct_change()
  df['ret_3'] = df['close'].pct_change(3)
  df['high_3'] = df['high'].rolling(3).max()
  df['low_3'] = df['low'].rolling(3).min()
  df['range_3'] = (df['high_3'] - df['low_3']) / df['close']

  # interaction features capturing combined signals
  df['rsi_macd_interaction'] = df['rsi_14'] * df['macd']
  df['momentum_volatility_ratio'] = df['momentum'] / (df['vol_7'] + 1e-5)
  df['sma_crossover'] = (df['sma_7'] - df['sma_14']) / df['close']

  # lagged features for temporal patterns
  for col in ['rsi_14', 'macd', 'sma_7', 'bb_width']:
    df[f'{col}_lag1'] = df[col].shift(1)

  return df

# calculating indicators for all assets
for ticker in all_data:
  all_data[ticker] = calculate_technical_indicators(all_data[ticker])

def create_target_labels(df):
  """
  Creating binary target labels for next-day price direction.
  Target = 1 if next day close > today close, else 0.
  """
  df = df.copy()
  df['next_close'] = df['close'].shift(-1)
  df['target'] = (df['next_close'] > df['close']).astype(int)
  return df

def create_significant_move_target(df, threshold=0.005):
  """
  Creating binary labels for significant price moves.
  Target = 1 if return > threshold, 0 if return < -threshold, NaN otherwise.
  """
  df = df.copy()
  df['next_close'] = df['close'].shift(-1)
  df['next_return'] = (df['next_close'] - df['close']) / df['close']
  df['target_big_move'] = np.where(df['next_return'] > threshold, 1,
                                    np.where(df['next_return'] < -threshold, 0, np.nan))
  return df

# creating target labels with asset-specific thresholds
for ticker in all_data:
  all_data[ticker] = create_target_labels(all_data[ticker])
  config = asset_config.get(ticker, asset_config['default'])
  all_data[ticker] = create_significant_move_target(all_data[ticker],
                                                    threshold=config['threshold_big_move'])

def expanding_window_split(df, min_train=10, test_size=1, step=1):
  """
  Generating train-test splits using expanding window approach.
  Training set growing while test set moving forward one day at a time.
  """
  n = len(df)
  i = min_train
  while i + test_size <= n:
    train_idx = range(0, i)
    test_idx = range(i, i + test_size)
    yield df.iloc[train_idx], df.iloc[test_idx]
    i += step

# hyperparameter grid for logistic regression
lr_grid = {'penalty': ['l2'], 'C': [0.01, 0.1, 1, 10]}

def train_and_evaluate(df, features, target_col, min_train=10, test_size=1, model_type='lr'):
  """
  Training logistic regression model using expanding window validation.
  Returning predictions and actual labels across all test windows.
  """
  predictions = []
  actuals = []
  scaler = StandardScaler()

  for train, test in expanding_window_split(df, min_train, test_size):
    X_train = train[features].dropna()
    y_train = train.loc[X_train.index, target_col]
    X_test = test[features].dropna()
    y_test = test.loc[X_test.index, target_col]

    # skipping if insufficient data or single class
    if len(X_train) == 0 or len(X_train) < min_train or y_train.nunique() < 2 or len(X_test) == 0:
      continue

    # scaling features
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # training logistic regression with grid search
    if model_type in ('all', 'lr'):
      model = GridSearchCV(
        LogisticRegression(max_iter=500, class_weight='balanced'),
        lr_grid,
        scoring='f1',
        cv=TimeSeriesSplit(n_splits=3)
      )
      model.fit(X_train_scaled, y_train)
      y_pred = model.predict(X_test_scaled)
      predictions.extend(y_pred.tolist())
      actuals.extend(y_test.tolist())

  return predictions, actuals

# evaluating models for all assets
print("\nCryptocurrency Price Direction Prediction Results\n")

results = {}
for ticker, df in all_data.items():
  # getting asset-specific settings
  config = asset_config.get(ticker, asset_config['default'])
  features = asset_features.get(ticker, asset_features['default'])

  # preparing data using most recent days
  df_recent = df.dropna(subset=['target']).iloc[-model_days:]
  df_clean = df_recent.dropna(subset=features + ['target'])

  if len(df_clean) < 15:
    print(f"\n{ticker}: Insufficient data, skipping")
    continue

  # training and evaluating for normal direction prediction
  print(f"\n{ticker} - Next-Day Direction")
  print(f"  Training window: {config['min_train']} days, Features: {len(features)}")

  preds, acts = train_and_evaluate(df_clean, features, 'target',
                                   min_train=config['min_train'],
                                   test_size=1,
                                   model_type=config['model_type'])

  # checking if both classes present in actuals
  if len(np.unique(acts)) < 2:
    print(f"  Only one class present, skipping")
    continue

  # calculating metrics
  precision = precision_score(acts, preds)
  recall = recall_score(acts, preds)
  f1 = f1_score(acts, preds)
  accuracy = np.mean(np.array(acts) == np.array(preds))

  # skipping reporting if configured and F1 is zero
  if config['skip_if_zeros'] and f1 == 0:
    print(f"  Skipped - insufficient signal for prediction")
    continue

  print(f"  Precision: {precision:.3f}, Recall: {recall:.3f}, F1: {f1:.3f}, Accuracy: {accuracy:.3f}")

  results[ticker] = {
    'precision': precision,
    'recall': recall,
    'f1': f1,
    'accuracy': accuracy
  }

  # evaluating for significant move prediction
  df_big_move = df.dropna(subset=['target_big_move']).iloc[-model_days:]
  df_big_clean = df_big_move.dropna(subset=features + ['target_big_move'])

  if df_big_clean['target_big_move'].notnull().sum() > 12:
    print(f"\n{ticker} - Significant Moves (±{config['threshold_big_move']*100:.1f}%)")

    preds_big, acts_big = train_and_evaluate(df_big_clean, features, 'target_big_move',
                                              min_train=config['min_train'],
                                              test_size=1)

    # filtering out NaN values
    valid_idx = [i for i, val in enumerate(acts_big) if val in (0, 1)]
    preds_filtered = [preds_big[i] for i in valid_idx]
    acts_filtered = [acts_big[i] for i in valid_idx]

    if len(acts_filtered) > 0 and len(np.unique(acts_filtered)) > 1:
      prec_big = precision_score(acts_filtered, preds_filtered)
      rec_big = recall_score(acts_filtered, preds_filtered)
      f1_big = f1_score(acts_filtered, preds_filtered)
      acc_big = np.mean(np.array(acts_filtered) == np.array(preds_filtered))
      print(f"  Precision: {prec_big:.3f}, Recall: {rec_big:.3f}, F1: {f1_big:.3f}, Accuracy: {acc_big:.3f}")

# printing summary statistics
if results:
  print("\n\nSummary Statistics")

  avg_precision = np.mean([v['precision'] for v in results.values()])
  avg_recall = np.mean([v['recall'] for v in results.values()])
  avg_f1 = np.mean([v['f1'] for v in results.values()])
  avg_accuracy = np.mean([v['accuracy'] for v in results.values()])

  print(f"\nAverage Performance:")
  print(f"  Precision: {avg_precision:.3f}")
  print(f"  Recall: {avg_recall:.3f}")
  print(f"  F1-Score: {avg_f1:.3f}")
  print(f"  Accuracy: {avg_accuracy:.3f}")
  print(f"\nAssets Evaluated: {len(results)} out of 10")

  best = max(results.items(), key=lambda x: x[1]['f1'])
  worst = min(results.items(), key=lambda x: x[1]['f1'])
  print(f"\nBest Performer: {best[0]} (F1: {best[1]['f1']:.3f})")
  print(f"Weakest Performer: {worst[0]} (F1: {worst[1]['f1']:.3f})")

  print("\nModeling complete")
