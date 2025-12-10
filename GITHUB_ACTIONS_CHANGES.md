# GitHub Actions Refactoring - Modeling File Changes

## Overview
The `03_Modeling.py` file has been refactored to be fully compatible with GitHub Actions CI/CD pipelines and automated testing workflows. Below are the key changes and improvements.

---

## Key Changes Made

### 1. **Cross-Platform Path Handling**
**Before:**
```python
os.chdir(r'D:/OneDrive - Northeastern University/Jupyter Notebook/Data Science Projects/Customer-Lifetime-Value-Prediction')
data = pd.read_csv("./data/Processed_AutoInsurance.csv")
plt.savefig('./results/model_comparison_visualization.png', dpi=300, bbox_inches='tight')
```

**After:**
```python
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / 'data'
RESULTS_DIR = PROJECT_ROOT / 'results'

RESULTS_DIR.mkdir(exist_ok=True)

data_path = DATA_DIR / 'Processed_AutoInsurance.csv'
viz_path = RESULTS_DIR / 'model_comparison_visualization.png'
```

**Why:** 
- `pathlib.Path` automatically handles Windows/Linux path differences
- Relative paths work regardless of working directory
- Absolute hardcoded paths break in CI/CD environments
- No need for `os.chdir()` which can cause issues in automation

---

### 2. **Logging Instead of Print Statements**
**Before:**
```python
print(f"\nData loaded: {data.shape}")
print(f"\n{model_name}:")
print(f"  RMSE:  {rmse:.6f}")
```

**After:**
```python
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

logger.info(f"Data loaded: {data.shape}")
logger.info(f"\n{model_name}:")
logger.info(f"  RMSE:  {rmse:.6f}")
```

**Why:**
- Logging provides structured output with timestamps
- GitHub Actions captures and archives logs automatically
- Print statements may not appear in CI/CD logs
- Easier to filter by log level (INFO, WARNING, ERROR)
- Professional standard for production code

---

### 3. **Non-Interactive Matplotlib Backend**
**New Addition:**
```python
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for CI/CD
```

**Why:**
- GUI backends (default) fail in headless CI/CD environments
- `Agg` backend saves plots to files without displaying windows
- Prevents `DISPLAY` not set errors on Linux CI runners
- Enables plot generation on servers without X11

---

### 4. **Error Handling with Try-Catch Blocks**
**Before:**
```python
rf_importance = pd.DataFrame(
    rf_model.feature_importances_,
    index=X.columns,
    columns=['Importance']
).sort_values('Importance', ascending=False)
```

**After:**
```python
def analyze_and_visualize(X, models, results_df, y_test, y_pred_rf, best_model):
    try:
        # ... code ...
        return rf_importance
    except Exception as e:
        logger.error(f"Error during analysis and visualization: {str(e)}")
        raise
```

**Why:**
- Graceful error handling with informative messages
- Stack traces logged for debugging
- CI/CD fails fast with meaningful error information
- Prevents silent failures

---

### 5. **Modular Functions for Testability**
**Before:**
```python
# All code executed at module load time
data = pd.read_csv("./data/Processed_AutoInsurance.csv")
# ... more code ...
results = {}
# ... training code ...
print("MODELING COMPLETE")
```

**After:**
```python
def load_and_prepare_data(data_path=None, test_size=0.30, random_state=42):
    """Load and prepare data for modeling."""
    # ... implementation ...
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, X

def train_and_evaluate_models(X_train_scaled, X_test_scaled, y_train, y_test):
    """Train and evaluate all models."""
    # ... implementation ...
    return results_df, models, y_pred_rf

def analyze_and_visualize(X, models, results_df, y_test, y_pred_rf, best_model):
    """Analyze feature importance and create visualizations."""
    # ... implementation ...
    return rf_importance

def main(data_path=None, test_size=0.30, verbose=True):
    """Main pipeline function."""
    # ... implementation ...
    return results_dict
```

**Why:**
- Functions can be imported and tested independently
- Enables unit testing for each pipeline stage
- Testable with pytest for CI/CD workflows
- Reusable in other scripts or notebooks

---

### 6. **Command-Line Arguments for CI/CD Flexibility**
**New Addition:**
```python
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLV Modeling Pipeline')
    parser.add_argument('--data-path', type=str, default=None, help='Path to processed data CSV')
    parser.add_argument('--test-size', type=float, default=0.30, help='Test set proportion')
    parser.add_argument('--verbose', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    result = main(
        data_path=args.data_path,
        test_size=args.test_size,
        verbose=args.verbose
    )
```

**Why:**
- GitHub Actions workflows can pass custom parameters
- Enables different runs for testing vs. production
- Example: `python 03_Modeling.py --data-path /path/to/data.csv --test-size 0.25`
- Flexible configuration without code changes

---

### 7. **Proper Exit Codes**
**New Addition:**
```python
except Exception as e:
    logger.error(f"Pipeline failed with error: {str(e)}", exc_info=True)
    sys.exit(1)  # Return error code to CI/CD
```

**Why:**
- CI/CD systems check exit codes (0 = success, non-zero = failure)
- Workflow can trigger notifications or additional jobs on failure
- Prevents masking of errors

---

## GitHub Actions Workflow Example

Here's how to use this refactored script in a GitHub Actions workflow:

```yaml
name: Model Training Pipeline

on:
  push:
    branches: [ main, develop ]
  schedule:
    - cron: '0 2 * * 0'  # Weekly on Sunday at 2 AM

jobs:
  train-models:
    runs-on: ubuntu-latest
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
    
    - name: Run data processing
      run: python 01_Data_processing.py
    
    - name: Train models
      run: python 03_Modeling.py --test-size 0.30 --verbose
    
    - name: Upload results
      uses: actions/upload-artifact@v3
      if: always()
      with:
        name: modeling-results
        path: results/
    
    - name: Upload logs
      uses: actions/upload-artifact@v3
      if: failure()
      with:
        name: error-logs
        path: logs/
```

---

## How to Use Locally

### Standard Usage
```bash
python 03_Modeling.py
```

### With Custom Data Path
```bash
python 03_Modeling.py --data-path /path/to/custom_data.csv
```

### With Custom Test Size
```bash
python 03_Modeling.py --test-size 0.25
```

### With Verbose Logging
```bash
python 03_Modeling.py --verbose
```

### Combined Options
```bash
python 03_Modeling.py --data-path ./data/data.csv --test-size 0.2 --verbose
```

---

## CI/CD Benefits

| Feature | Benefit |
|---------|---------|
| **Modular Functions** | Unit testing, faster CI runs |
| **Logging** | Debugging in CI/CD environments |
| **Error Handling** | Clear failure messages |
| **CLI Arguments** | Flexible test configurations |
| **Cross-Platform Paths** | Works on any CI/CD runner |
| **Non-Interactive Backend** | Plots work on headless servers |
| **Exit Codes** | Proper CI/CD integration |

---

## Next Steps for GitHub Actions Setup

1. **Create `.github/workflows/train.yml`** - Define the CI/CD pipeline
2. **Create `tests/test_modeling.py`** - Add unit tests for functions
3. **Create `requirements.txt`** - Specify exact dependency versions
4. **Create `.gitignore`** - Exclude data and model files as needed
5. **Add pre-commit hooks** - Lint and format code before commits

---

## Testing the Refactored Code

### Create `tests/test_modeling.py`

```python
import pytest
from pathlib import Path
from src.modeling import load_and_prepare_data, evaluate_model

def test_load_and_prepare_data():
    """Test data loading and preparation."""
    X_train, X_test, y_train, y_test, scaler, X = load_and_prepare_data()
    
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert y_train.shape[0] == X_train.shape[0]
    assert y_test.shape[0] == X_test.shape[0]

def test_evaluate_model():
    """Test model evaluation metrics."""
    import numpy as np
    y_true = np.array([1, 2, 3, 4, 5])
    y_pred = np.array([1.1, 2.1, 2.9, 4.2, 4.8])
    
    results = evaluate_model(y_true, y_pred, "test_model")
    
    assert 'RMSE' in results
    assert 'R2' in results
    assert 'MAE' in results
    assert 'MAPE' in results
```

---

## Recommendations

1. ✅ **Use this refactored version** - All changes are backward compatible
2. ✅ **Add unit tests** - Test individual functions in CI/CD
3. ✅ **Add code linting** - Use `pylint`, `black`, `flake8`
4. ✅ **Add type hints** - Improves code clarity for CI/CD tools
5. ✅ **Document environment** - Create `.python-version`, `pyproject.toml`
6. ✅ **Use secrets** - Store API keys/paths in GitHub Secrets if needed

---

## Summary of Improvements

| Aspect | Before | After |
|--------|--------|-------|
| **File Paths** | Hardcoded absolute paths | Dynamic relative paths with `pathlib` |
| **Output** | Print statements | Structured logging |
| **Plotting** | Interactive backend | Headless Agg backend |
| **Error Handling** | None | Try-catch with logging |
| **Code Structure** | Procedural | Modular functions |
| **CLI Support** | None | argparse with multiple options |
| **Exit Codes** | Implicit 0 | Explicit error codes |
| **Testability** | Poor | Excellent |
| **CI/CD Ready** | No | Yes |

The refactored code is production-ready for GitHub Actions and other CI/CD platforms!
