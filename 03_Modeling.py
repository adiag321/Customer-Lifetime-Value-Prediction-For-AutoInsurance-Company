import pandas as pd
import numpy as np
import os
import sys
import logging
import argparse
from pathlib import Path
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for CI/CD
import scipy.stats as stats
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV, cross_validate
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso, ElasticNet, HuberRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor, ExtraTreesRegressor
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, mean_absolute_percentage_error
import warnings
warnings.filterwarnings("ignore")

# Configure logging for CI/CD environments
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Get project root directory (relative to this script)
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'

# Create results directory if it doesn't exist
RESULTS_DIR.mkdir(exist_ok=True)

logger.info(f"Project Root: {PROJECT_ROOT}")
logger.info(f"Data Directory: {DATA_DIR}")
logger.info(f"Results Directory: {RESULTS_DIR}")

########################################################################
#                         DATA LOADING AND PREPARATION
########################################################################
def load_and_prepare_data(data_path=None, test_size=0.30, random_state=42):
    """
    Args:
        data_path: Path to processed data CSV file
        test_size: Test set proportion
        random_state: Random seed for reproducibility
        
    Returns:
        tuple: (X_train_scaled, X_test_scaled, y_train, y_test, scaler, X)
    """
    if data_path is None:
        data_path = DATA_DIR / 'Processed_AutoInsurance.csv'
    
    if not Path(data_path).exists():
        raise FileNotFoundError(f"Data file not found: {data_path}")
    
    logger.info(f"Loading data from: {data_path}")
    data = pd.read_csv(data_path)
    logger.info(f"Data loaded: {data.shape}")
    
    # Feature and target preparation
    columns_to_drop = ['CLV', 'Policy Type_Personal Auto', 'Policy Type_Special Auto', 'Policy_Personal L1', 'Policy_Personal L2', 'Policy_Personal L3',
                       'Policy_Special L1', 'Policy_Special L2', 'Policy_Special L3']
    X = data.drop(columns_to_drop, axis=1)
    y = data['CLV']
    y = np.log(y)
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)
    
    # Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    logger.info(f"Data shapes:")
    logger.info(f"X_train: {X_train_scaled.shape}, y_train: {y_train.shape}")
    logger.info(f"X_test: {X_test_scaled.shape}, y_test: {y_test.shape}")
    logger.info(f"Number of features: {X.shape[1]}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X

########################################################################
#                         EVALUATION METRICS FUNCTION
########################################################################
def evaluate_model(y_true, y_pred, model_name, cv_scores=None):
    """
    Args:
        y_true: True target values
        y_pred: Predicted target values
        model_name: Name of the model for logging
        cv_scores: Cross-validation scores dictionary
        
    Returns:
        dict: Dictionary with RMSE, MAE, MAPE, and R2 scores
    """
    try:
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)
        mape = mean_absolute_percentage_error(y_true, y_pred)
        
        logger.info(f"{model_name}:")
        logger.info(f"  RMSE:  {rmse:.6f}")
        logger.info(f"  MAE:   {mae:.6f}")
        logger.info(f"  MAPE:  {mape:.6f}")
        logger.info(f"  R²:    {r2:.6f}")
        
        if cv_scores is not None:
            cv_mean = cv_scores['test_r2'].mean()
            cv_std = cv_scores['test_r2'].std()
            logger.info(f"  CV R² Mean: {cv_mean:.6f} (+/- {cv_std:.6f})")
        
        return {'RMSE': rmse, 'MAE': mae, 'MAPE': mape, 'R2': r2}
    except Exception as e:
        logger.error(f"Error evaluating model {model_name}: {str(e)}")
        raise e

########################################################################
#                         MODEL FUNCTIONS
########################################################################
def apply_linear_regression(X_train_scaled, X_test_scaled, y_train, y_test):
    """Apply Linear Regression with cross-validation"""
    logger.info("="*60)
    logger.info("1. LINEAR REGRESSION (With Cross-Validation)")
    logger.info("="*60)
    
    lr = LinearRegression()
    cv_scores_lr = cross_validate(lr, X_train_scaled, y_train, cv=5, scoring=['r2', 'neg_mean_squared_error', 'neg_root_mean_squared_error'], return_train_score=True)
    lr.fit(X_train_scaled, y_train)
    y_pred_lr = lr.predict(X_test_scaled)
    
    return evaluate_model(y_test, y_pred_lr, 'Linear Regression', cv_scores_lr), y_pred_lr, lr

def apply_ridge_regression(X_train_scaled, X_test_scaled, y_train, y_test):
    """Apply Ridge Regression with hyperparameter tuning"""
    logger.info("="*60)
    logger.info("2. RIDGE REGRESSION (with Hyperparameter Tuning)")
    logger.info("="*60)
    
    ridge_params = {'alpha': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0, 1000.0]}
    ridge_grid = GridSearchCV(Ridge(), ridge_params, cv=5, scoring='r2', n_jobs=-1)
    ridge_grid.fit(X_train_scaled, y_train)
    logger.info(f"Best Ridge alpha: {ridge_grid.best_params_['alpha']}")
    logger.info(f"Best CV R² Score: {ridge_grid.best_score_:.6f}")
    
    ridge = ridge_grid.best_estimator_
    y_pred_ridge = ridge.predict(X_test_scaled)
    
    return evaluate_model(y_test, y_pred_ridge, 'Ridge Regression'), y_pred_ridge, ridge

def apply_lasso_regression(X_train_scaled, X_test_scaled, y_train, y_test):
    """Apply Lasso Regression with hyperparameter tuning"""
    logger.info("="*60)
    logger.info("3. LASSO REGRESSION (with Hyperparameter Tuning)")
    logger.info("="*60)
    
    lasso_params = {'alpha': [0.0001, 0.001, 0.01, 0.1, 1.0, 10.0]}
    lasso_grid = GridSearchCV(Lasso(random_state=42, max_iter=5000), lasso_params, cv=5, scoring='r2', n_jobs=-1)
    lasso_grid.fit(X_train_scaled, y_train)
    logger.info(f"Best Lasso alpha: {lasso_grid.best_params_['alpha']}")
    logger.info(f"Best CV R² Score: {lasso_grid.best_score_:.6f}")
    
    lasso = lasso_grid.best_estimator_
    y_pred_lasso = lasso.predict(X_test_scaled)
    
    return evaluate_model(y_test, y_pred_lasso, 'Lasso Regression'), y_pred_lasso, lasso

def apply_elasticnet_regression(X_train_scaled, X_test_scaled, y_train, y_test):
    """Apply Elastic Net Regression with hyperparameter tuning"""
    logger.info("="*60)
    logger.info("4. ELASTIC NET REGRESSION (with Hyperparameter Tuning)")
    logger.info("="*60)
    
    elasticnet_params = {
        'alpha': [0.001, 0.01, 0.1, 1.0],
        'l1_ratio': [0.1, 0.5, 0.7, 0.9]
    }
    en_grid = GridSearchCV(ElasticNet(random_state=42, max_iter=5000), elasticnet_params, cv=5, scoring='r2', n_jobs=-1)
    en_grid.fit(X_train_scaled, y_train)
    logger.info(f"Best ElasticNet params: {en_grid.best_params_}")
    logger.info(f"Best CV R² Score: {en_grid.best_score_:.6f}")
    
    en = en_grid.best_estimator_
    y_pred_en = en.predict(X_test_scaled)
    
    return evaluate_model(y_test, y_pred_en, 'ElasticNet'), y_pred_en, en

def apply_decision_tree(X_train_scaled, X_test_scaled, y_train, y_test):
    """Apply Decision Tree Regressor with hyperparameter tuning"""
    logger.info("="*60)
    logger.info("5. DECISION TREE REGRESSOR (with Hyperparameter Tuning)")
    logger.info("="*60)
    
    dt_params = {
        'max_depth': [5, 10, 15, 20, None],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'criterion': ['squared_error', 'absolute_error']
    }
    dt_grid = GridSearchCV(DecisionTreeRegressor(random_state=42), dt_params, cv=5, scoring='r2', n_jobs=-1)
    dt_grid.fit(X_train_scaled, y_train)
    logger.info(f"Best Decision Tree params: {dt_grid.best_params_}")
    logger.info(f"Best CV R² Score: {dt_grid.best_score_:.6f}")
    
    dt = dt_grid.best_estimator_
    y_pred_dt = dt.predict(X_test_scaled)
    
    return evaluate_model(y_test, y_pred_dt, 'Decision Tree'), y_pred_dt, dt

def apply_random_forest(X_train_scaled, X_test_scaled, y_train, y_test):
    """Apply Random Forest Regressor with hyperparameter tuning"""
    logger.info("="*60)
    logger.info("6. RANDOM FOREST REGRESSOR (with Hyperparameter Tuning)")
    logger.info("="*60)
    
    rf_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'max_features': ['sqrt', 'log2']
    }
    rf_grid = GridSearchCV(RandomForestRegressor(random_state=42, n_jobs=-1), rf_params, cv=5, scoring='r2', n_jobs=-1)
    rf_grid.fit(X_train_scaled, y_train)
    logger.info(f"Best Random Forest params: {rf_grid.best_params_}")
    logger.info(f"Best CV R² Score: {rf_grid.best_score_:.6f}")
    
    rf = rf_grid.best_estimator_
    y_pred_rf = rf.predict(X_test_scaled)
    
    return evaluate_model(y_test, y_pred_rf, 'Random Forest'), y_pred_rf, rf

def apply_gradient_boosting(X_train_scaled, X_test_scaled, y_train, y_test):
    """Apply Gradient Boosting Regressor with hyperparameter tuning"""
    logger.info("="*60)
    logger.info("7. GRADIENT BOOSTING REGRESSOR (with Hyperparameter Tuning)")
    logger.info("="*60)
    
    gb_params = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 7],
        'min_samples_split': [2, 5],
        'subsample': [0.8, 0.9, 1.0]
    }
    gb_grid = GridSearchCV(GradientBoostingRegressor(random_state=42), gb_params, cv=5, scoring='r2', n_jobs=-1)
    gb_grid.fit(X_train_scaled, y_train)
    logger.info(f"Best Gradient Boosting params: {gb_grid.best_params_}")
    logger.info(f"Best CV R² Score: {gb_grid.best_score_:.6f}")
    
    gb = gb_grid.best_estimator_
    y_pred_gb = gb.predict(X_test_scaled)
    
    return evaluate_model(y_test, y_pred_gb, 'Gradient Boosting'), y_pred_gb, gb

def apply_extra_trees(X_train_scaled, X_test_scaled, y_train, y_test):
    """Apply Extra Trees Regressor with hyperparameter tuning"""
    logger.info("="*60)
    logger.info("8. EXTRA TREES REGRESSOR (with Hyperparameter Tuning)")
    logger.info("="*60)
    
    et_params = {
        'n_estimators': [50, 100, 200],
        'max_depth': [10, 20, 30],
        'min_samples_split': [2, 5],
        'max_features': ['sqrt', 'log2']
    }
    et_grid = GridSearchCV(ExtraTreesRegressor(random_state=42, n_jobs=-1), et_params, cv=5, scoring='r2', n_jobs=-1)
    et_grid.fit(X_train_scaled, y_train)
    logger.info(f"Best Extra Trees params: {et_grid.best_params_}")
    logger.info(f"Best CV R² Score: {et_grid.best_score_:.6f}")
    
    et = et_grid.best_estimator_
    y_pred_et = et.predict(X_test_scaled)
    
    return evaluate_model(y_test, y_pred_et, 'Extra Trees'), y_pred_et, et

def apply_adaboost(X_train_scaled, X_test_scaled, y_train, y_test):
    """Apply AdaBoost Regressor with hyperparameter tuning"""
    logger.info("="*60)
    logger.info("9. ADABOOST REGRESSOR (with Hyperparameter Tuning)")
    logger.info("="*60)
    
    ada_params = {
        'n_estimators': [50, 100, 150],
        'learning_rate': [0.01, 0.05, 0.1, 0.2],
        'loss': ['linear', 'square', 'exponential']
    }
    ada_grid = GridSearchCV(AdaBoostRegressor(random_state=42), ada_params, cv=5, scoring='r2', n_jobs=-1)
    ada_grid.fit(X_train_scaled, y_train)
    logger.info(f"Best AdaBoost params: {ada_grid.best_params_}")
    logger.info(f"Best CV R² Score: {ada_grid.best_score_:.6f}")
    
    ada = ada_grid.best_estimator_
    y_pred_ada = ada.predict(X_test_scaled)
    
    return evaluate_model(y_test, y_pred_ada, 'AdaBoost'), y_pred_ada, ada

def apply_svr(X_train_scaled, X_test_scaled, y_train, y_test):
    """Apply Support Vector Regressor with hyperparameter tuning"""
    logger.info("="*60)
    logger.info("10. SUPPORT VECTOR REGRESSOR (with Hyperparameter Tuning)")
    logger.info("="*60)
    
    svr_params = {
        'C': [0.1, 1, 10, 100],
        'kernel': ['linear', 'rbf', 'poly'],
        'gamma': ['scale', 'auto']
    }
    svr_grid = GridSearchCV(SVR(), svr_params, cv=5, scoring='r2', n_jobs=-1)
    svr_grid.fit(X_train_scaled, y_train)
    logger.info(f"Best SVR params: {svr_grid.best_params_}")
    logger.info(f"Best CV R² Score: {svr_grid.best_score_:.6f}")
    
    svr = svr_grid.best_estimator_
    y_pred_svr = svr.predict(X_test_scaled)
    
    return evaluate_model(y_test, y_pred_svr, 'Support Vector Regressor'), y_pred_svr, svr

def apply_huber(X_train_scaled, X_test_scaled, y_train, y_test):
    """Apply Huber Regressor with hyperparameter tuning"""
    logger.info("="*60)
    logger.info("11. HUBER REGRESSOR (with Hyperparameter Tuning)")
    logger.info("="*60)
    
    huber_params = {
        'epsilon': [1.05, 1.1, 1.2, 1.5],
        'alpha': [0.0001, 0.001, 0.01],
        'max_iter': [100, 500, 1000]
    }
    huber_grid = GridSearchCV(HuberRegressor(), huber_params, cv=5, scoring='r2', n_jobs=-1)
    huber_grid.fit(X_train_scaled, y_train)
    logger.info(f"Best Huber params: {huber_grid.best_params_}")
    logger.info(f"Best CV R² Score: {huber_grid.best_score_:.6f}")
    
    huber = huber_grid.best_estimator_
    y_pred_huber = huber.predict(X_test_scaled)
    
    return evaluate_model(y_test, y_pred_huber, 'Huber Regressor'), y_pred_huber, huber

########################################################################
#                     MODEL TRAINING AND EVALUATION
########################################################################
def train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test):
    """
    Returns:
        tuple: (results_df, models, y_pred_rf)
    """
    results = {}
    models = {}
    y_pred_rf = None
    
    try:
        # Apply all models using individual functions
        logger.info("Starting model training...")
        results['Linear Regression'], _, models['Linear Regression'] = apply_linear_regression(X_train_scaled, X_test_scaled, y_train, y_test)
        results['Ridge Regression'], _, models['Ridge Regression'] = apply_ridge_regression(X_train_scaled, X_test_scaled, y_train, y_test)
        results['Lasso Regression'], _, models['Lasso Regression'] = apply_lasso_regression(X_train_scaled, X_test_scaled, y_train, y_test)
        results['ElasticNet'], _, models['ElasticNet'] = apply_elasticnet_regression(X_train_scaled, X_test_scaled, y_train, y_test)
        results['Decision Tree'], _, models['Decision Tree'] = apply_decision_tree(X_train_scaled, X_test_scaled, y_train, y_test)
        results['Random Forest'], y_pred_rf, models['Random Forest'] = apply_random_forest(X_train_scaled, X_test_scaled, y_train, y_test)
        results['Gradient Boosting'], _, models['Gradient Boosting'] = apply_gradient_boosting(X_train_scaled, X_test_scaled, y_train, y_test)
        results['Extra Trees'], _, models['Extra Trees'] = apply_extra_trees(X_train_scaled, X_test_scaled, y_train, y_test)
        #results['AdaBoost'], _, models['AdaBoost'] = apply_adaboost(X_train_scaled, X_test_scaled, y_train, y_test)
        #results['SVR'], _, models['SVR'] = apply_svr(X_train_scaled, X_test_scaled, y_train, y_test)
        results['Huber'], _, models['Huber'] = apply_huber(X_train_scaled, X_test_scaled, y_train, y_test)
        logger.info("All models trained successfully.")
    
    except Exception as e:
        logger.error(f"Error during model training: {str(e)}")
        raise e
    
    # Results summary
    logger.info("="*60)
    logger.info("MODEL COMPARISON SUMMARY")
    logger.info("="*60)
    
    results_df = pd.DataFrame(results).T.sort_values('R2', ascending=False)
    logger.info("" + results_df.to_string())
    
    # Best model
    best_model = results_df.index[0]
    logger.info(f"BEST MODEL: {best_model}")
    logger.info(f"  R² Score: {results_df.loc[best_model, 'R2']:.6f}")
    logger.info(f"  RMSE: {results_df.loc[best_model, 'RMSE']:.6f}")
    logger.info(f"  MAE: {results_df.loc[best_model, 'MAE']:.6f}")
    
    return results_df, models, y_pred_rf

########################################################################
#                  FEATURE IMPORTANCE & VISUALIZATION
########################################################################
def analyze_and_visualize(X, models, results_df, y_test, y_pred_rf, best_model):
    """
   Args:
        X: Feature dataframe
        models: Dictionary of trained models
        results_df: Results dataframe
        y_test: Test target values
        y_pred_rf: Random Forest predictions
        best_model: Name of best model
    """
    try:
        logger.info("="*60)
        logger.info("FEATURE IMPORTANCE ANALYSIS")
        logger.info("="*60)
        
        # Random Forest feature importance
        rf_model = models['Random Forest']
        rf_importance = pd.DataFrame(rf_model.feature_importances_, index = X.columns, columns = ['Importance']).sort_values('Importance', ascending=False)
        
        logger.info("Top 15 Features (Random Forest):")
        logger.info(rf_importance.head(15).to_string())
        
        # Visualization
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        
        # Plot 1: Feature Importance (Bar)
        ax1 = axes[0, 0]
        rf_importance.head(15).plot(kind='barh', ax=ax1, color='steelblue')
        ax1.set_title('Top 15 Feature Importance (Random Forest)', fontsize=12, fontweight='bold')
        ax1.set_xlabel('Importance Score')
        
        # Plot 2: Model Comparison (R² Scores)
        ax2 = axes[0, 1]
        results_df['R2'].sort_values().plot(kind='barh', ax=ax2, color='darkgreen')
        ax2.set_title('Model Comparison - R² Scores', fontsize=12, fontweight='bold')
        ax2.set_xlabel('R² Score')
        
        # Plot 3: Model Comparison (RMSE)
        ax3 = axes[1, 0]
        results_df['RMSE'].sort_values().plot(kind='barh', ax=ax3, color='coral')
        ax3.set_title('Model Comparison - RMSE', fontsize=12, fontweight='bold')
        ax3.set_xlabel('RMSE')
        
        # Plot 4: Actual vs Predicted (Best Model - Random Forest)
        ax4 = axes[1, 1]
        ax4.scatter(y_test, y_pred_rf, alpha=0.5, color='purple')
        ax4.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax4.set_xlabel('Actual Values')
        ax4.set_ylabel('Predicted Values')
        ax4.set_title(f'Actual vs Predicted - {best_model}', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        # Save visualization
        viz_path = RESULTS_DIR / 'model_comparison_visualization.png'
        plt.savefig(viz_path, dpi=300, bbox_inches='tight')
        logger.info(f"Visualization saved to {viz_path}")
        plt.close()
        
        # Save results
        results_path = RESULTS_DIR / 'model_results_summary.csv'
        results_df.to_csv(results_path)
        logger.info(f"Results saved to {results_path}")
        
        logger.info("ANALYSIS & VISUALIZATION COMPLETE")
        
        return rf_importance
    except Exception as e:
        logger.error(f"Error during analysis and visualization: {str(e)}")
        raise e

########################################################################
#                       MAIN FUNCTION
########################################################################
def main(data_path=None, test_size=0.30, verbose=True):
    """
    Main function to run the complete modeling pipeline.
    
    Args:
        data_path: Path to processed data CSV file
        test_size: Test set proportion
        verbose: Whether to print detailed logs
        
    Returns:
        dict: Dictionary containing results and models
    """
    try:
        logger.info("Starting Customer Lifetime Value Modeling Pipeline")
        logger.info("="*60)
        
        # Data loading and preparation
        X_train_scaled, X_test_scaled, y_train, y_test, scaler, X = load_and_prepare_data(data_path=data_path, test_size=test_size)
        
        # Model training and evaluation
        results_df, models, y_pred_rf = train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test)
        
        # Analysis and visualization
        best_model = results_df.index[0]
        rf_importance = analyze_and_visualize(X, models, results_df, y_test, y_pred_rf, best_model)
        
        logger.info("="*60)
        logger.info("MODELING COMPLETE - PIPELINE SUCCESSFUL")
        logger.info("="*60)
        
        return {
            'results_df': results_df,
            'models': models,
            'feature_importance': rf_importance,
            'best_model': best_model,
            'scaler': scaler,
            'X_columns': X.columns.tolist()
        }
    except Exception as e:
        logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
        sys.exit(1)


########################################################################
#                       ENTRY POINT
########################################################################
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLV Modeling Pipeline')
    parser.add_argument('--data-path', type=str, default=None, help='Path to processed data CSV')
    parser.add_argument('--test-size', type=float, default=0.30, help='Test set proportion')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    result = main(data_path = args.data_path, test_size = args.test_size, verbose = args.verbose)

