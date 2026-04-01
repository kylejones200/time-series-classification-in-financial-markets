# Description: Short example for Time Series Classification in Financial Markets.



from data_io import read_csv
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import TimeSeriesSplit
from tensorflow import keras
import logging
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import tensorflow as tf

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)



def load_financial_data(symbol, start_date, end_date):
    """Load and prepare financial time series data."""
    df = read_csv(f"{symbol}_data.csv", parse_dates=['timestamp'])
    df = df[(df['timestamp'] >= start_date) & (df['timestamp'] <= end_date)]
    
    # Basic financial data preprocessing
    df['returns'] = df['close'].pct_change()
    df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
    df['volatility'] = df['returns'].rolling(window=20).std()
    
    return df

def align_market_hours(df):
    """Align data to market hours and handle off-market periods."""
    df = df[df['timestamp'].dt.dayofweek < 5]  # Remove weekends
    df = df[(df['timestamp'].dt.hour >= 9) & (df['timestamp'].dt.hour < 16)]  # Market hours
    return df

def normalize_financial_data(df):
    """Normalize financial data while preserving temporal relationships."""
    df['norm_price'] = (df['close'] - df['close'].rolling(20).mean()) / \
                       df['close'].rolling(20).std()
    return df

def calculate_rsi(prices, window=14):
    """Calculate Relative Strength Index."""
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(window=window).mean()
    loss = -delta.where(delta < 0, 0).rolling(window=window).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))


def calculate_macd(prices, fast=12, slow=26, signal=9):
    """Calculate MACD indicator."""
    ema_fast = prices.ewm(span=fast).mean()
    ema_slow = prices.ewm(span=slow).mean()
    macd = ema_fast - ema_slow
    signal_line = macd.ewm(span=signal).mean()
    return macd, signal_line


def calculate_bollinger_position(prices, window=20, num_std=2):
    """Calculate position within Bollinger Bands."""
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    upper_band = rolling_mean + (rolling_std * num_std)
    lower_band = rolling_mean - (rolling_std * num_std)
    return (prices - lower_band) / (upper_band - lower_band)


def extract_financial_features(df):
    """Extract features relevant for financial time series."""
    features = {}
    
    # Technical Indicators
    features['rsi'] = calculate_rsi(df['close'], window=14)
    features['macd'], features['macd_signal'] = calculate_macd(df['close'])
    features['bb_position'] = calculate_bollinger_position(df['close'])
    
    # Statistical Features
    features['volatility'] = df['returns'].rolling(20).std()
    features['skewness'] = df['returns'].rolling(20).skew()
    features['kurtosis'] = df['returns'].rolling(20).kurt()
    
    # Market Microstructure
    features['volume_vwap'] = df['volume'] * df['close'].rolling(20).mean()
    features['price_momentum'] = df['close'].pct_change(5)
    
    return pd.DataFrame(features)

def financial_dtw_distance(s1, s2, max_warping=0.1):
    """
    Compute DTW distance between two financial sequences
    with constraints specific to financial data.
    """
    n, m = len(s1), len(s2)
    max_warp = int(max_warping * min(n, m))  # Sakoe-Chiba band
    
    dtw_matrix = np.full((n+1, m+1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n+1):
        window_start = max(1, i - max_warp)
        window_end = min(m + 1, i + max_warp)
        
        for j in range(window_start, window_end):
            # Financial-specific cost function
            cost = abs(s1[i-1] - s2[j-1]) * (1 + abs(i/n - j/m))
            
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],    # insertion
                dtw_matrix[i, j-1],    # deletion
                dtw_matrix[i-1, j-1]   # match
            )
    
    return dtw_matrix[n, m]


def create_financial_cnn(input_shape, num_classes):
    """Create CNN model optimized for financial patterns."""
    model = keras.Sequential([
        # Capture local price patterns
        keras.layers.Conv1D(64, kernel_size=3, activation='relu', 
                           input_shape=input_shape),
        keras.layers.BatchNormalization(),
        
        # Capture medium-term patterns
        keras.layers.Conv1D(128, kernel_size=5, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(pool_size=2),
        
        # Capture long-term patterns
        keras.layers.Conv1D(256, kernel_size=7, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalAveragePooling1D(),
        
        keras.layers.Dense(num_classes, activation='softmax')
    ])
    return model

class FinancialClassificationBacktest:
    def __init__(self, data, model, transaction_costs=0.001):
        self.data = data
        self.model = model
        self.transaction_costs = transaction_costs
        self.positions = []
        self.returns = []
    
    def run_backtest(self):
        """Execute backtest with transaction costs and slippage."""
        window_size = 60  # Trading window
        
        for i in range(window_size, len(self.data)):
            # Get features for current window
            window_data = self.data[i-window_size:i]
            features = extract_financial_features(window_data)
            
            # Make prediction
            prediction = self.model.predict(features.values[-1].reshape(1, -1))
            position = self.determine_position(prediction)
            
            # Calculate returns with costs
            returns = self.calculate_returns(position, i)
            self.positions.append(position)
            self.returns.append(returns)
        
        return self.analyze_performance()
    
    def calculate_returns(self, position, index):
        """Calculate returns including transaction costs."""
        price_change = self.data['returns'].iloc[index]
        if len(self.positions) > 0 and position != self.positions[-1]:
            return position * price_change - self.transaction_costs
        return position * price_change
    
    def determine_position(self, prediction):
        """Convert prediction to trading position."""
        # Simple strategy: long if prediction > 0.6, short if < 0.4
        if prediction[0][1] > 0.6:
            return 1
        elif prediction[0][0] > 0.6:
            return -1
        return 0

def calculate_sharpe_ratio(returns, risk_free_rate=0.02):
    """Calculate annualized Sharpe ratio."""
    excess_returns = returns - risk_free_rate / 252
    return np.sqrt(252) * np.mean(excess_returns) / np.std(returns)


def calculate_max_drawdown(returns):
    """Calculate maximum drawdown."""
    cumulative = (1 + returns).cumprod()
    running_max = cumulative.expanding().max()
    drawdown = (cumulative - running_max) / running_max
    return drawdown.min()


def calculate_turnover(positions):
    """Calculate position turnover."""
    return np.sum(np.abs(np.diff(positions))) / len(positions)


def evaluate_trading_performance(returns, positions):
    """Calculate trading-specific performance metrics."""
    metrics = {
        'sharpe_ratio': calculate_sharpe_ratio(returns),
        'max_drawdown': calculate_max_drawdown(returns),
        'win_rate': sum(returns > 0) / len(returns),
        'profit_factor': abs(sum(returns[returns > 0]) / 
                           sum(returns[returns < 0])) if sum(returns < 0) != 0 else np.inf,
        'turnover': calculate_turnover(positions)
    }
    
    return pd.Series(metrics)

class RiskManager:
    def __init__(self, max_position=1.0, max_drawdown=0.02):
        """
        Initialize risk parameters.
        
        Parameters:
        -----------
        max_position : float
            Maximum allowed position size as fraction of equity
        max_drawdown : float
            Maximum allowed drawdown before trading stops
        """
        self.max_position = max_position
        self.max_drawdown = max_drawdown
        self.current_drawdown = 0
 
    def validate_trade(self, prediction, current_position, current_equity):
        """
        Validate and size trades based on risk parameters.
        
        Returns:
        --------
        float
            Position size as fraction of equity
        """
        if self.current_drawdown > self.max_drawdown:
            return 0  # Stop trading when drawdown limit reached
   
        position_size = self.calculate_position_size(prediction, current_equity)
        return min(position_size, self.max_position)
    
    def calculate_position_size(self, prediction, current_equity):
        """Calculate position size based on model confidence."""
        # Simple implementation: scale by prediction confidence
        confidence = abs(prediction[0][1] - 0.5) * 2  # Normalize to [0, 1]
        return confidence * self.max_position

"""
Complete Financial Time Series Classification Example
"""


def generate_financial_data(n_days=500, seed=42):
    """Generate synthetic financial market data."""
    np.random.seed(seed)
    dates = pd.date_range(start='2020-01-01', periods=n_days, freq='D')
    
    # Generate price series with trend and volatility
    returns = np.random.normal(0.0005, 0.02, n_days)  # Daily returns
    prices = 100 * np.exp(np.cumsum(returns))
    
    # Add volume
    volume = np.random.lognormal(10, 0.5, n_days)
    
    df = pd.DataFrame({
        'timestamp': dates,
        'close': prices,
        'volume': volume,
        'returns': returns
    })
    
    return df


def create_market_state_labels(df, window=20):
    """
    Create market state labels based on price movements.
    0 = Bearish, 1 = Bullish
    """
    future_returns = df['returns'].shift(-window).rolling(window).sum()
    labels = (future_returns > 0).astype(int)
    return labels


def main():
    """Run complete example."""
    logger.info("=" * 60)
    logger.info("Financial Time Series Classification")
    logger.info("=" * 60)
    
    # Generate data
    logger.info("\n1. Generating synthetic financial data...")
    df = generate_financial_data(n_days=500)
    logger.info(f"   Generated {len(df)} days of data")
    
    # Create labels
    logger.info("\n2. Creating market state labels...")
    labels = create_market_state_labels(df, window=20)
    df['label'] = labels
    df = df.dropna()
    logger.info(f"   Class distribution: {df['label'].value_counts().to_dict()}")
    
    # Extract features
    logger.info("\n3. Extracting financial features...")
    features_df = extract_financial_features(df)
    features_df = features_df.dropna()
    
    # Align with labels
    common_idx = features_df.index.intersection(df.index)
    X = features_df.loc[common_idx].values
    y = df.loc[common_idx]['label'].values
    
    logger.info(f"   Feature matrix shape: {X.shape}")
    
    # Time series split
    logger.info("\n4. Training classification model...")
    tscv = TimeSeriesSplit(n_splits=5)
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    scores = []
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model.fit(X_train, y_train)
        score = model.score(X_test, y_test)
        scores.append(score)
        logger.info(f"   Fold {fold} accuracy: {score:.4f}")
    
    logger.info(f"\n   Average accuracy: {np.mean(scores):.4f} ± {np.std(scores):.4f}")
    
    # Final evaluation
    train_idx, test_idx = list(tscv.split(X))[-1]
    y_pred = model.predict(X[test_idx])
    
    logger.info("\n5. Classification Report:")
    logger.info(classification_report(y[test_idx], y_pred, 
                              target_names=['Bearish', 'Bullish']))
    
    # Visualizations
    logger.info("\n6. Creating visualizations...")
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Price series with labels
    axes[0, 0].plot(df['timestamp'], df['close'], alpha=0.7, linewidth=1)
    bullish_periods = df[df['label'] == 1]
    axes[0, 0].scatter(bullish_periods['timestamp'], bullish_periods['close'],
                      c='green', alpha=0.3, s=10, label='Bullish')
    bearish_periods = df[df['label'] == 0]
    axes[0, 0].scatter(bearish_periods['timestamp'], bearish_periods['close'],
                      c='red', alpha=0.3, s=10, label='Bearish')
    axes[0, 0].set_title('Price Series with Market States', fontsize=12)
    axes[0, 0].set_xlabel('Date', fontsize=11)
    axes[0, 0].set_ylabel('Price', fontsize=11)
    axes[0, 0].legend()
        # Confusion matrix
    cm = confusion_matrix(y[test_idx], y_pred)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[0, 1],
                xticklabels=['Bearish', 'Bullish'],
                yticklabels=['Bearish', 'Bullish'])
    axes[0, 1].set_title('Confusion Matrix', fontsize=12)
    axes[0, 1].set_ylabel('True Label', fontsize=11)
    axes[0, 1].set_xlabel('Predicted Label', fontsize=11)
    
    # Feature importance
    feature_names = features_df.columns
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1][:10]
    axes[1, 0].barh(range(len(indices)), importances[indices])
    axes[1, 0].set_yticks(range(len(indices)))
    axes[1, 0].set_yticklabels([feature_names[i] for i in indices])
    axes[1, 0].set_title('Top 10 Feature Importances', fontsize=12)
    axes[1, 0].set_xlabel('Importance', fontsize=11)
        # Cross-validation scores
    axes[1, 1].plot(range(1, len(scores) + 1), scores, 'o-', linewidth=2)
    axes[1, 1].axhline(y=np.mean(scores), color='r', linestyle='--',
                      label=f'Mean: {np.mean(scores):.4f}')
    axes[1, 1].set_title('Cross-Validation Scores', fontsize=12)
    axes[1, 1].set_xlabel('Fold', fontsize=11)
    axes[1, 1].set_ylabel('Accuracy', fontsize=11)
    axes[1, 1].legend()
    plt.tight_layout()
    plt.savefig('financial_timeseries_classification.png', dpi=300)
    plt.close()
    logger.info("   Saved visualization to 'financial_timeseries_classification.png'")
    
    logger.info("\n" + "=" * 60)
    logger.info("Example completed successfully!")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
