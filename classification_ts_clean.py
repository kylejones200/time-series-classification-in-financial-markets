import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns

# Configuration
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
plt.rcParams['figure.facecolor'] = 'white'
np.random.seed(42)

# Generate synthetic time series (random walk with drift)
n_samples = 500
data = pd.Series(np.cumsum(np.random.randn(n_samples) + 0.1))

# Create features (no current value to avoid leakage)
df = pd.DataFrame({
    'value': data,
    'lag_1': data.shift(1),
    'lag_2': data.shift(2),
    'lag_3': data.shift(3),
    'rate_of_change': data.diff(),
    'acceleration': data.diff().diff(),
    'rolling_mean_5': data.shift(1).rolling(window=5).mean(),
    'rolling_std_5': data.shift(1).rolling(window=5).std()
})

# Target: 1 if next value increases, 0 if decreases
df['target'] = (data.shift(-1) > data).astype(int)
df = df.dropna()

# Temporal train/test split (80/20)
split_idx = int(len(df) * 0.8)
train_df = df.iloc[:split_idx]
test_df = df.iloc[split_idx:]

features = ['lag_1', 'lag_2', 'lag_3', 'rate_of_change', 
            'acceleration', 'rolling_mean_5', 'rolling_std_5']

X_train = train_df[features]
y_train = train_df['target']
X_test = test_df[features]
y_test = test_df['target']

# Train classifier
clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10)
clf.fit(X_train, y_train)

# Predictions
y_pred = clf.predict(X_test)
y_pred_proba = clf.predict_proba(X_test)[:, 1]

# Metrics
acc = accuracy_score(y_test, y_pred)
print(f"Accuracy: {acc:.3f}")
print(f"\nTrain samples: {len(X_train)} | Test samples: {len(X_test)}")
print(f"Class distribution - Train: {y_train.value_counts().to_dict()}")
print(f"Class distribution - Test:  {y_test.value_counts().to_dict()}\n")
print("Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Down', 'Up']))

# Visualization
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Original time series with train/test split
axes[0, 0].plot(df.index[:split_idx], df['value'][:split_idx], 
                color='#2c3e50', linewidth=1, label='Train')
axes[0, 0].plot(df.index[split_idx:], df['value'][split_idx:], 
                color='#e74c3c', linewidth=1, label='Test')
axes[0, 0].axvline(split_idx, color='red', linestyle='--', alpha=0.5)
axes[0, 0].set_title('Time Series with Train/Test Split', fontsize=12)
axes[0, 0].legend()
axes[0, 0].set_ylabel('Value')

# Predictions vs actual on test set
test_indices = test_df.index
correct = y_test == y_pred
incorrect = ~correct

axes[0, 1].scatter(test_indices[correct], test_df['value'][correct], 
                   c='#27ae60', s=30, alpha=0.6, label='Correct')
axes[0, 1].scatter(test_indices[incorrect], test_df['value'][incorrect], 
                   c='#e74c3c', s=30, alpha=0.6, label='Incorrect')
axes[0, 1].set_title(f'Classification Performance (Accuracy: {acc:.3f})', fontsize=12)
axes[0, 1].legend()
axes[0, 1].set_ylabel('Value')
axes[0, 1].set_xlabel('Index')

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[1, 0], 
            xticklabels=['Down', 'Up'], yticklabels=['Down', 'Up'],
            cbar_kws={'label': 'Count'})
axes[1, 0].set_title('Confusion Matrix', fontsize=12)
axes[1, 0].set_ylabel('Actual')
axes[1, 0].set_xlabel('Predicted')

# Feature importance
importances = clf.feature_importances_
feature_imp_df = pd.DataFrame({
    'feature': features,
    'importance': importances
}).sort_values('importance', ascending=True)

axes[1, 1].barh(feature_imp_df['feature'], feature_imp_df['importance'], 
                color='#3498db', alpha=0.7)
axes[1, 1].set_title('Feature Importance', fontsize=12)
axes[1, 1].set_xlabel('Importance')

plt.tight_layout()
plt.savefig('classification_ts_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

# Prediction probabilities plot
fig, ax = plt.subplots(figsize=(14, 4))
ax.plot(test_indices, y_pred_proba, color='#3498db', linewidth=1.5, alpha=0.7)
ax.axhline(0.5, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax.fill_between(test_indices, 0.5, y_pred_proba, where=(y_pred_proba >= 0.5), 
                alpha=0.3, color='#27ae60', label='Predicted Up')
ax.fill_between(test_indices, y_pred_proba, 0.5, where=(y_pred_proba < 0.5), 
                alpha=0.3, color='#e74c3c', label='Predicted Down')
ax.set_title('Prediction Probabilities Over Time', fontsize=12)
ax.set_ylabel('P(Up)')
ax.set_xlabel('Index')
ax.legend()
plt.tight_layout()
plt.savefig('classification_probabilities.png', dpi=300, bbox_inches='tight')
plt.show()

print(f"\nSaved: classification_ts_analysis.png")
print(f"Saved: classification_probabilities.png")

