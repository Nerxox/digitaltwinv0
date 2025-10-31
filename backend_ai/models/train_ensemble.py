"""
Enhanced energy prediction model with ensemble methods and error analysis.
"""
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Optional
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import TimeSeriesSplit, train_test_split
from scipy import stats
import matplotlib.dates as mdates
import xgboost as xgb
import lightgbm as lgb
import joblib
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Dropout, BatchNormalization, Input, concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class EnergyPredictor:
    def __init__(self, model_dir: str = "artifacts/models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        self.scaler = StandardScaler()
        self.models = {}
        self.feature_importances = {}
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def load_data(self, train_path: str, test_path: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Load and validate training data."""
        try:
            logger.info("Loading and validating data...")
            train_df = pd.read_csv(train_path, parse_dates=['ts'], index_col='ts')
            test_df = pd.read_csv(test_path, parse_dates=['ts'], index_col='ts')
            
            # Basic validation
            for df, name in [(train_df, "train"), (test_df, "test")]:
                if 'power_kw' not in df.columns:
                    raise ValueError(f"Missing 'power_kw' column in {name} data")
                if df.isnull().any().any():
                    logger.warning(f"Found missing values in {name} data")
                    
            return train_df, test_df
            
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise

    def create_features(self, df: pd.DataFrame, lags: List[int] = None) -> pd.DataFrame:
        """Create comprehensive time-series and domain-specific features."""
        if lags is None:
            lags = [1, 5, 15, 30, 60, 1440]  # Default lags in minutes
        
        df = df.copy()
        
        # 1. Basic lagged features
        for lag in lags:
            df[f'lag_{lag}'] = df['power_kw'].shift(lag)
        
        # 2. Rolling statistics
        for window in [15, 30, 60, 240, 1440]:
            df[f'roll_{window}_mean'] = df['power_kw'].rolling(window).mean()
            df[f'roll_{window}_std'] = df['power_kw'].rolling(window).std()
            df[f'roll_{window}_min'] = df['power_kw'].rolling(window).min()
            df[f'roll_{window}_max'] = df['power_kw'].rolling(window).max()
            
            if window > 1:
                df[f'roll_{window}_range'] = df[f'roll_{window}_max'] - df[f'roll_{window}_min']
                df[f'roll_{window}_cv'] = df[f'roll_{window}_std'] / (df[f'roll_{window}_mean'] + 1e-6)
        
        # 3. Time-based features
        df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
        df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
        df['dow_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
        df['dow_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
        df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
        df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)
        
        # 4. Time of day categories
        df['is_night'] = ((df.index.hour >= 0) & (df.index.hour <= 5)).astype(int)
        df['is_morning'] = ((df.index.hour >= 6) & (df.index.hour <= 11)).astype(int)
        df['is_afternoon'] = ((df.index.hour >= 12) & (df.index.hour <= 17)).astype(int)
        df['is_evening'] = ((df.index.hour >= 18) & (df.index.hour <= 23)).astype(int)
        
        # 5. Calendar features
        df['is_weekend'] = df.index.dayofweek.isin([5, 6]).astype(int)
        df['is_weekday'] = ~df.index.dayofweek.isin([5, 6]).astype(int)
        
        # 6. Domain-specific features
        df['is_peak_hour'] = ((df.index.hour >= 8) & (df.index.hour <= 20)).astype(int)
        df['prev_day_same_hour'] = df['power_kw'].shift(24 * 60)
        df['prev_week_same_hour'] = df['power_kw'].shift(7 * 24 * 60)
        
        # 7. Change features
        df['hourly_change'] = df['power_kw'].diff(60)
        df['daily_change'] = df['power_kw'].diff(24 * 60)
        
        # 8. Handle missing values
        df = df.ffill().bfill().fillna(0)
        
        return df

    def prepare_data(self, train_df: pd.DataFrame, test_df: pd.DataFrame, horizon: int) -> tuple:
        """Prepare data for training and testing."""
        logger.info("Preparing data...")
        train_features = self.create_features(train_df)
        test_features = self.create_features(test_df)
        
        # Align features
        common_cols = list(set(train_features.columns) & set(test_features.columns))
        X_train = train_features[common_cols]
        X_test = test_features[common_cols]
        
        # Create targets
        y_train = train_df['power_kw'].shift(-horizon).dropna()
        y_test = test_df['power_kw'].shift(-horizon).dropna()
        
        # Align features with targets
        X_train = X_train.loc[y_train.index]
        X_test = X_test.loc[y_test.index]
        
        return X_train, X_test, y_train, y_test

    def train_lstm(self, X_train: np.ndarray, y_train: np.ndarray) -> Model:
        """Train LSTM model."""
        logger.info("Training LSTM model...")
        
        # Reshape for LSTM [samples, timesteps, features]
        X_reshaped = X_train.reshape((X_train.shape[0], 1, X_train.shape[1]))
        
        # Build model
        inputs = Input(shape=(1, X_train.shape[1]))
        x = LSTM(128, return_sequences=True)(inputs)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = LSTM(64)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.3)(x)
        x = Dense(32, activation='relu')(x)
        outputs = Dense(1)(x)
        
        model = Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Callbacks
        callbacks = [
            EarlyStopping(patience=15, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5, min_lr=1e-6)
        ]
        
        # Train
        history = model.fit(
            X_reshaped, y_train,
            validation_split=0.2,
            epochs=100,
            batch_size=64,
            callbacks=callbacks,
            verbose=1
        )
        
        return model, history

    def train_xgboost(self, X_train: np.ndarray, y_train: np.ndarray) -> xgb.XGBRegressor:
        """Train XGBoost model with early stopping."""
        logger.info("Training XGBoost model...")
        
        # Split data for validation
        X_train, X_val, y_train, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=42, shuffle=False
        )
        
        model = xgb.XGBRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1,
            early_stopping_rounds=50,
            eval_metric='rmse'
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=100
        )
        
        return model

    def train_lightgbm(self, X_train: np.ndarray, y_train: np.ndarray) -> lgb.LGBMRegressor:
        """Train LightGBM model."""
        logger.info("Training LightGBM model...")
        
        model = lgb.LGBMRegressor(
            n_estimators=1000,
            learning_rate=0.01,
            max_depth=-1,
            num_leaves=31,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            callbacks=[
                lgb.early_stopping(stopping_rounds=50, verbose=True),
                lgb.log_evaluation(100)
            ]
        )
        
        return model

    def train_ensemble(self, X_train: np.ndarray, y_train: np.ndarray):
        """Train ensemble of models."""
        # Scale features
        X_scaled = self.scaler.fit_transform(X_train)
        
        # Train models
        lstm_model, lstm_history = self.train_lstm(X_scaled, y_train)
        xgb_model = self.train_xgboost(X_scaled, y_train)
        lgb_model = self.train_lightgbm(X_scaled, y_train)
        
        self.models = {
            'lstm': lstm_model,
            'xgb': xgb_model,
            'lgb': lgb_model
        }
        
        # Save models
        self._save_models()
        
        return lstm_history

    def predict_ensemble(self, X: np.ndarray) -> dict:
        """Make predictions using all models."""
        X_scaled = self.scaler.transform(X)
        
        # LSTM requires 3D input
        X_lstm = X_scaled.reshape((X_scaled.shape[0], 1, X_scaled.shape[1]))
        
        predictions = {
            'lstm': self.models['lstm'].predict(X_lstm).flatten(),
            'xgb': self.models['xgb'].predict(X_scaled),
            'lgb': self.models['lgb'].predict(X_scaled)
        }
        
        # Ensemble prediction (simple average)
        predictions['ensemble'] = np.mean([
            predictions['lstm'],
            predictions['xgb'],
            predictions['lgb']
        ], axis=0)
        
        return predictions

    def evaluate_models(self, X_test: np.ndarray, y_test: pd.Series) -> dict:
        """Evaluate all models and return metrics with confidence intervals."""
        predictions = self.predict_ensemble(X_test)
        
        metrics = {}
        for model_name, y_pred in predictions.items():
            # Calculate point estimates
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            r2 = r2_score(y_test, y_pred)
            mape = mean_absolute_percentage_error(y_test, y_pred) * 100  # as percentage
            
            # Calculate confidence intervals using bootstrap
            n_bootstraps = 1000
            boot_mae = []
            boot_rmse = []
            boot_r2 = []
            
            rng = np.random.RandomState(42)
            for _ in range(n_bootstraps):
                indices = rng.choice(len(y_test), len(y_test), replace=True)
                if len(np.unique(y_test.iloc[indices])) < 2:
                    continue
                
                boot_mae.append(mean_absolute_error(y_test.iloc[indices], y_pred[indices]))
                boot_rmse.append(np.sqrt(mean_squared_error(y_test.iloc[indices], y_pred[indices])))
                boot_r2.append(r2_score(y_test.iloc[indices], y_pred[indices]))
            
            # Calculate confidence intervals
            ci_mae = np.percentile(boot_mae, [2.5, 97.5]) if boot_mae else [np.nan, np.nan]
            ci_rmse = np.percentile(boot_rmse, [2.5, 97.5]) if boot_rmse else [np.nan, np.nan]
            ci_r2 = np.percentile(boot_r2, [2.5, 97.5]) if boot_r2 else [np.nan, np.nan]
            
            metrics[model_name] = {
                'mae': mae,
                'mae_ci': ci_mae,
                'rmse': rmse,
                'rmse_ci': ci_rmse,
                'r2': r2,
                'r2_ci': ci_r2,
                'mape': mape
            }
            
            logger.info(f"\n{model_name.upper()} Metrics (95% CI):")
            logger.info(f"MAE: {mae:.4f} ({ci_mae[0]:.4f} - {ci_mae[1]:.4f})")
            logger.info(f"RMSE: {rmse:.4f} ({ci_rmse[0]:.4f} - {ci_rmse[1]:.4f})")
            logger.info(f"R²: {r2:.4f} ({ci_r2[0]:.4f} - {ci_r2[1]:.4f})")
            logger.info(f"MAPE: {mape:.2f}%")
        
        return metrics, predictions

    def plot_predictions(self, y_true: pd.Series, predictions: dict, model_name: str, n_samples: int = 1000):
        """Plot predictions vs actual values with confidence intervals and residual analysis."""
        y_pred = predictions[model_name]
        errors = y_true - y_pred
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(15, 12), gridspec_kw={'height_ratios': [2, 1]})
        
        # Plot 1: Time series of actual vs predicted
        # Sample points for better visibility
        sample_idx = np.random.choice(len(y_true), min(n_samples, len(y_true)), replace=False)
        sample_idx = np.sort(sample_idx)
        
        # Calculate rolling statistics for confidence bands
        rolling_std = pd.Series(errors).rolling(window=24*7, min_periods=1).std()
        
        # Plot actual and predicted values
        ax1.plot(y_true.iloc[sample_idx].index, y_true.iloc[sample_idx], 'b-', label='Actual', alpha=0.7, linewidth=1.5)
        ax1.plot(y_true.iloc[sample_idx].index, y_pred[sample_idx], 'r--', label='Predicted', alpha=0.9, linewidth=1.2)
        
        # Add confidence bands (2 standard deviations)
        ax1.fill_between(
            y_true.index[sample_idx],
            y_pred[sample_idx] - 2*rolling_std.iloc[sample_idx],
            y_pred[sample_idx] + 2*rolling_std.iloc[sample_idx],
            color='gray', alpha=0.2, label='95% Confidence Interval'
        )
        
        # Formatting
        ax1.set_title(f'{model_name.upper()} - Actual vs Predicted Values\n(Showing {min(n_samples, len(y_true))} samples for clarity)', fontsize=12)
        ax1.set_ylabel('Power (kW)')
        ax1.legend(loc='upper right')
        ax1.grid(True, linestyle='--', alpha=0.6)
        
        # Plot 2: Residuals over time
        ax2.scatter(y_true.index[sample_idx], errors.iloc[sample_idx], 
                   alpha=0.5, s=10, color='green')
        ax2.axhline(0, color='r', linestyle='--', alpha=0.7)
        ax2.set_title('Residuals Over Time', fontsize=12)
        ax2.set_xlabel('Time')
        ax2.set_ylabel('Residuals (Actual - Predicted)')
        ax2.grid(True, linestyle='--', alpha=0.6)
        
        # Format x-axis with better date formatting
        for ax in [ax1, ax2]:
            ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d %H:%M'))
            plt.setp(ax.xaxis.get_majorticklabels(), rotation=45, ha='right')
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.model_dir / f'predictions_residuals_{model_name}_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved prediction and residual plot to {plot_path}")
        
        # Additional plot: Residuals vs Predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_pred, errors, alpha=0.5, s=10, color='purple')
        plt.axhline(0, color='r', linestyle='--', alpha=0.7)
        plt.title(f'{model_name.upper()} - Residuals vs Predicted Values', fontsize=12)
        plt.xlabel('Predicted Power (kW)')
        plt.ylabel('Residuals (Actual - Predicted)')
        plt.grid(True, linestyle='--', alpha=0.6)
        
        # Add a lowess smooth to show trend
        try:
            import statsmodels.api as sm
            lowess = sm.nonparametric.lowess(errors, y_pred, frac=0.2)
            plt.plot(lowess[:, 0], lowess[:, 1], 'b-', linewidth=2, label='Trend')
            plt.legend()
        except ImportError:
            logger.warning("Statsmodels not available for lowess smoothing")
        
        plt.tight_layout()
        
        # Save residual analysis plot
        residual_plot_path = self.model_dir / f'residual_analysis_{model_name}_{self.timestamp}.png'
        plt.savefig(residual_plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved residual analysis plot to {residual_plot_path}")

    def plot_error_distribution(self, y_true: pd.Series, predictions: dict, model_name: str):
        """Plot error distribution with detailed statistics."""
        errors = y_true - predictions[model_name]
        
        # Calculate statistics
        mean_err = np.mean(errors)
        median_err = np.median(errors)
        std_err = np.std(errors)
        skew = stats.skew(errors)
        kurt = stats.kurtosis(errors)
        
        # Create figure with 2 subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 6))
        
        # Histogram with KDE
        sns.histplot(errors, kde=True, bins=50, ax=ax1)
        ax1.axvline(mean_err, color='r', linestyle='--', label=f'Mean: {mean_err:.2f}')
        ax1.axvline(median_err, color='g', linestyle='-', label=f'Median: {median_err:.2f}')
        ax1.set_title(f'{model_name.upper()} Error Distribution')
        ax1.set_xlabel('Prediction Error (Actual - Predicted)')
        ax1.set_ylabel('Frequency')
        ax1.legend()
        
        # Q-Q plot
        stats.probplot(errors, dist="norm", plot=ax2)
        ax2.set_title('Q-Q Plot of Prediction Errors')
        
        # Add statistics as text
        stats_text = (
            f'Statistics:\n'
            f'Mean: {mean_err:.4f}\n'
            f'Std Dev: {std_err:.4f}\n'
            f'Skewness: {skew:.4f}\n'
            f'Kurtosis: {kurt:.4f}'
        )
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, 
                fontsize=10, verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        plt.tight_layout()
        
        # Save plot
        plot_path = self.model_dir / f'error_analysis_{model_name}_{self.timestamp}.png'
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved error analysis plot to {plot_path}")
        
        # Save error statistics to file
        stats_path = self.model_dir / f'error_stats_{model_name}_{self.timestamp}.txt'
        with open(stats_path, 'w') as f:
            f.write(f'Error Statistics for {model_name.upper()}\n')
            f.write('='*50 + '\n')
            f.write(f'Mean Error: {mean_err:.4f}\n')
            f.write(f'Median Error: {median_err:.4f}\n')
            f.write(f'Std Dev of Errors: {std_err:.4f}\n')
            f.write(f'Skewness: {skew:.4f}\n')
            f.write(f'Kurtosis: {kurt:.4f}\n')
            f.write(f'5th Percentile: {np.percentile(errors, 5):.4f}\n')
            f.write(f'95th Percentile: {np.percentile(errors, 95):.4f}\n')
            f.write(f'MAE: {np.mean(np.abs(errors)):.4f}\n')
            f.write(f'RMSE: {np.sqrt(np.mean(errors**2)):.4f}')
        
        logger.info(f"Saved error statistics to {stats_path}")

    def plot_feature_importance(self, X: pd.DataFrame):
        """Plot feature importance for tree-based models with SHAP values if available."""
        try:
            import shap
            shap_available = True
        except ImportError:
            shap_available = False
            logger.info("SHAP not available. Install with: pip install shap")
        
        for model_name in ['xgb', 'lgb']:
            if model_name in self.models:
                model = self.models[model_name]
                
                # Standard feature importance
                if hasattr(model, 'feature_importances_'):
                    importances = model.feature_importances_
                elif hasattr(model, 'feature_importance'):
                    importances = model.feature_importance()
                else:
                    continue
                
                # Create DataFrame for visualization
                feature_importance = pd.DataFrame({
                    'feature': X.columns,
                    'importance': importances
                }).sort_values('importance', ascending=False).head(20)
                
                # Plot standard importance
                plt.figure(figsize=(12, 10))
                sns.barplot(x='importance', y='feature', data=feature_importance, palette='viridis')
                plt.title(f'{model_name.upper()} - Top 20 Feature Importance', fontsize=14)
                plt.xlabel('Importance Score', fontsize=12)
                plt.ylabel('Features', fontsize=12)
                plt.tight_layout()
                
                # Save plot
                plot_path = self.model_dir / f'feature_importance_{model_name}_{self.timestamp}.png'
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                logger.info(f"Saved feature importance plot to {plot_path}")
                
                # SHAP values analysis if available
                if shap_available and hasattr(model, 'predict'):
                    try:
                        # Create SHAP explainer
                        if model_name == 'xgb':
                            explainer = shap.TreeExplainer(model)
                        else:  # LightGBM
                            explainer = shap.TreeExplainer(model, feature_perturbation="tree_path_dependent")
                        
                        # Calculate SHAP values (use a sample for large datasets)
                        sample_size = min(1000, X.shape[0])
                        X_sample = shap.sample(X, sample_size, random_state=42)
                        shap_values = explainer.shap_values(X_sample)
                        
                        # SHAP summary plot
                        plt.figure(figsize=(12, 10))
                        shap.summary_plot(shap_values, X_sample, plot_type="bar", max_display=20, show=False)
                        plt.title(f'{model_name.upper()} - SHAP Feature Importance', fontsize=14)
                        plt.tight_layout()
                        
                        # Save SHAP summary plot
                        shap_summary_path = self.model_dir / f'shap_summary_{model_name}_{self.timestamp}.png'
                        plt.savefig(shap_summary_path, dpi=300, bbox_inches='tight')
                        plt.close()
                        logger.info(f"Saved SHAP summary plot to {shap_summary_path}")
                        
                        # SHAP dependence plots for top features
                        top_features = feature_importance['feature'].head(3).tolist()
                        for feature in top_features:
                            plt.figure(figsize=(10, 6))
                            shap.dependence_plot(
                                feature, shap_values, X_sample, 
                                display_features=X_sample,
                                show=False, alpha=0.5
                            )
                            plt.title(f'SHAP Dependence Plot for {feature}', fontsize=12)
                            plt.tight_layout()
                            
                            # Save dependence plot
                            dep_plot_path = self.model_dir / f'shap_dependence_{model_name}_{feature[:20]}_{self.timestamp}.png'
                            plt.savefig(dep_plot_path, dpi=300, bbox_inches='tight')
                            plt.close()
                            
                    except Exception as e:
                        logger.error(f"Error generating SHAP plots: {str(e)}")
                        if 'plt' in locals() and plt.fignum_exists(plt.gcf().number):
                            plt.close()

    def _save_models(self):
        """Save all models and scaler."""
        # Save models
        for name, model in self.models.items():
            if name == 'lstm':
                model_path = self.model_dir / f'lstm_model_{self.timestamp}.h5'
                model.save(model_path)
            else:
                model_path = self.model_dir / f'{name}_model_{self.timestamp}.pkl'
                joblib.dump(model, model_path)
            logger.info(f"Saved {name.upper()} model to {model_path}")
        
        # Save scaler
        scaler_path = self.model_dir / f'scaler_{self.timestamp}.pkl'
        joblib.dump(self.scaler, scaler_path)
        logger.info(f"Saved scaler to {scaler_path}")

def main():
    try:
        # Initialize predictor
        predictor = EnergyPredictor()
        
        # Load data
        train_df, test_df = predictor.load_data(
            train_path="datasets/train.csv",
            test_path="datasets/test.csv"
        )
        
        # Prepare data
        X_train, X_test, y_train, y_test = predictor.prepare_data(
            train_df, test_df, horizon=15
        )
        
        # Train ensemble
        history = predictor.train_ensemble(X_train.values, y_train.values)
        
        # Evaluate models
        metrics, predictions = predictor.evaluate_models(X_test.values, y_test)
        
        # Generate visualizations
        for model_name in predictions.keys():
            predictor.plot_predictions(y_test, predictions, model_name)
            predictor.plot_error_distribution(y_test, predictions, model_name)
        
        # Plot feature importance for tree-based models
        predictor.plot_feature_importance(X_test)
        
        logger.info("\nTraining and evaluation completed successfully!")
        
    except Exception as e:
        logger.error(f"Error in training pipeline: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    main()
