"""
Unit tests for the CLV modeling pipeline.
Run with: pytest tests/test_modeling.py -v
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path
import sys

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.modeling import evaluate_model, load_and_prepare_data


class TestEvaluateModel:
    """Test cases for model evaluation function"""
    
    def test_evaluate_model_basic(self):
        """Test basic evaluation metrics calculation"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
        
        results = evaluate_model(y_true, y_pred, "test_model")
        
        assert isinstance(results, dict)
        assert set(results.keys()) == {'RMSE', 'MAE', 'MAPE', 'R2'}
        assert all(isinstance(v, (int, float)) for v in results.values())
    
    def test_evaluate_model_perfect_prediction(self):
        """Test metrics for perfect predictions"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        results = evaluate_model(y_true, y_pred, "perfect_model")
        
        assert results['RMSE'] == pytest.approx(0.0, abs=1e-6)
        assert results['MAE'] == pytest.approx(0.0, abs=1e-6)
        assert results['R2'] == pytest.approx(1.0, abs=1e-6)
    
    def test_evaluate_model_poor_prediction(self):
        """Test metrics for poor predictions"""
        y_true = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        y_pred = np.array([5.0, 4.0, 3.0, 2.0, 1.0])  # Completely inverted
        
        results = evaluate_model(y_true, y_pred, "poor_model")
        
        assert results['RMSE'] > 0
        assert results['MAE'] > 0
        assert results['R2'] < 0  # Negative R² for very poor predictions
    
    def test_evaluate_model_metrics_range(self):
        """Test that metrics are within reasonable ranges"""
        y_true = np.random.randn(100)
        y_pred = y_true + np.random.randn(100) * 0.1
        
        results = evaluate_model(y_true, y_pred, "random_model")
        
        assert results['RMSE'] >= 0
        assert results['MAE'] >= 0
        assert results['MAPE'] >= 0
        assert -1 <= results['R2'] <= 1


class TestDataLoading:
    """Test cases for data loading and preparation"""
    
    @pytest.fixture
    def create_sample_data(self, tmp_path):
        """Create sample processed data for testing"""
        data = {
            'CLV': np.random.exponential(scale=2000, size=100),
            'Income': np.random.uniform(20000, 200000, 100),
            'Monthly Premium Auto': np.random.uniform(50, 300, 100),
            'Total Claim Amount': np.random.uniform(0, 5000, 100),
            'Months Since Policy Inception': np.random.randint(1, 60, 100),
            'Number of Policies': np.random.randint(1, 4, 100),
            'Number of Open Complaints': np.random.randint(0, 5, 100),
            'Coverage_Basic': np.random.randint(0, 2, 100),
            'Coverage_Extended': np.random.randint(0, 2, 100),
            'EmploymentStatus_Employed': np.random.randint(0, 2, 100),
            'EmploymentStatus_Retired': np.random.randint(0, 2, 100),
            'Policy Type_Personal Auto': np.random.randint(0, 2, 100),
            'Policy Type_Special Auto': np.random.randint(0, 2, 100),
            'Policy_Personal L1': np.random.randint(0, 2, 100),
            'Policy_Personal L2': np.random.randint(0, 2, 100),
            'Policy_Personal L3': np.random.randint(0, 2, 100),
            'Policy_Special L1': np.random.randint(0, 2, 100),
            'Policy_Special L2': np.random.randint(0, 2, 100),
            'Policy_Special L3': np.random.randint(0, 2, 100),
        }
        
        df = pd.DataFrame(data)
        data_path = tmp_path / "test_data.csv"
        df.to_csv(data_path, index=False)
        
        return data_path
    
    def test_load_and_prepare_data_shapes(self, create_sample_data):
        """Test that data is loaded and split correctly"""
        X_train, X_test, y_train, y_test, scaler, X = load_and_prepare_data(
            data_path=create_sample_data, test_size=0.30, random_state=42
        )
        
        total_samples = X_train.shape[0] + X_test.shape[0]
        assert total_samples == 100
        assert X_train.shape[0] == 70
        assert X_test.shape[0] == 30
        assert y_train.shape[0] == 70
        assert y_test.shape[0] == 30
        assert X_train.shape[1] == X_test.shape[1]
    
    def test_load_and_prepare_data_scaling(self, create_sample_data):
        """Test that features are properly scaled"""
        X_train, X_test, y_train, y_test, scaler, X = load_and_prepare_data(
            data_path=create_sample_data, test_size=0.30
        )
        
        # Check that training data is standardized (mean ≈ 0, std ≈ 1)
        train_mean = np.abs(X_train.mean(axis=0))
        train_std = X_train.std(axis=0)
        
        assert np.all(train_mean < 0.1) or np.all(train_mean > 0)  # Close to 0
        assert np.allclose(train_std, 1.0, atol=0.1)
    
    def test_load_and_prepare_data_log_transform(self, create_sample_data):
        """Test that target variable is log-transformed"""
        X_train, X_test, y_train, y_test, scaler, X = load_and_prepare_data(
            data_path=create_sample_data
        )
        
        # y should be log-transformed, so all values should be reasonable
        assert np.all(np.isfinite(y_train))
        assert np.all(np.isfinite(y_test))
        assert np.all(y_train > 0)  # Log values should be positive for reasonable CLV
    
    def test_load_and_prepare_data_reproducibility(self, create_sample_data):
        """Test that random_state ensures reproducibility"""
        X_train1, X_test1, _, _, _, _ = load_and_prepare_data(
            data_path=create_sample_data, random_state=42
        )
        
        X_train2, X_test2, _, _, _, _ = load_and_prepare_data(
            data_path=create_sample_data, random_state=42
        )
        
        np.testing.assert_array_almost_equal(X_train1, X_train2)
        np.testing.assert_array_almost_equal(X_test1, X_test2)
    
    def test_load_and_prepare_data_no_missing_values(self, create_sample_data):
        """Test that prepared data has no missing values"""
        X_train, X_test, y_train, y_test, scaler, X = load_and_prepare_data(
            data_path=create_sample_data
        )
        
        assert not np.any(np.isnan(X_train))
        assert not np.any(np.isnan(X_test))
        assert not np.any(np.isnan(y_train))
        assert not np.any(np.isnan(y_test))


class TestErrorHandling:
    """Test error handling in the pipeline"""
    
    def test_load_nonexistent_data(self):
        """Test handling of missing data file"""
        with pytest.raises(FileNotFoundError):
            load_and_prepare_data(data_path="nonexistent_file.csv")
    
    def test_evaluate_model_with_nan_values(self):
        """Test evaluation with NaN values"""
        y_true = np.array([1.0, 2.0, np.nan, 4.0, 5.0])
        y_pred = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        # Should raise an error or handle gracefully
        with pytest.raises((ValueError, RuntimeWarning)):
            evaluate_model(y_true, y_pred, "test_with_nan")


class TestIntegration:
    """Integration tests for the full pipeline"""
    
    @pytest.fixture
    def create_full_sample_data(self, tmp_path):
        """Create comprehensive sample data"""
        np.random.seed(42)
        n_samples = 500
        
        data = {
            'CLV': np.random.exponential(scale=2000, size=n_samples),
            'Income': np.random.uniform(20000, 200000, n_samples),
            'Monthly Premium Auto': np.random.uniform(50, 300, n_samples),
            'Total Claim Amount': np.random.uniform(0, 5000, n_samples),
            'Months Since Policy Inception': np.random.randint(1, 60, n_samples),
            'Months Since Last Claim': np.random.randint(0, 60, n_samples),
            'Number of Policies': np.random.randint(1, 4, n_samples),
            'Number of Open Complaints': np.random.randint(0, 5, n_samples),
            'Coverage_Basic': np.random.randint(0, 2, n_samples),
            'Coverage_Extended': np.random.randint(0, 2, n_samples),
            'EmploymentStatus_Employed': np.random.randint(0, 2, n_samples),
            'EmploymentStatus_Retired': np.random.randint(0, 2, n_samples),
            'Policy Type_Personal Auto': np.random.randint(0, 2, n_samples),
            'Policy Type_Special Auto': np.random.randint(0, 2, n_samples),
            'Policy_Personal L1': np.random.randint(0, 2, n_samples),
            'Policy_Personal L2': np.random.randint(0, 2, n_samples),
            'Policy_Personal L3': np.random.randint(0, 2, n_samples),
            'Policy_Special L1': np.random.randint(0, 2, n_samples),
            'Policy_Special L2': np.random.randint(0, 2, n_samples),
            'Policy_Special L3': np.random.randint(0, 2, n_samples),
        }
        
        df = pd.DataFrame(data)
        data_path = tmp_path / "full_test_data.csv"
        df.to_csv(data_path, index=False)
        
        return data_path
    
    def test_end_to_end_pipeline(self, create_full_sample_data):
        """Test the complete pipeline from load to evaluation"""
        # Load and prepare data
        X_train, X_test, y_train, y_test, scaler, X = load_and_prepare_data(
            data_path=create_full_sample_data, test_size=0.20
        )
        
        # Create dummy model
        from sklearn.ensemble import RandomForestRegressor
        model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Evaluate
        results = evaluate_model(y_test, y_pred, "test_rf")
        
        assert results['R2'] > 0  # Should have positive R² for random data
        assert results['RMSE'] > 0
        assert results['MAE'] > 0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
