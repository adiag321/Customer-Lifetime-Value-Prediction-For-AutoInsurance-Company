# GitHub Actions Setup Guide

This guide walks you through setting up GitHub Actions for the CLV Modeling project.

#### Step 1: Create the workflow directory structure

The workflow file has already been created at `.github/workflows/model_training.yml`

#### Step 2: Verify file structure

Your project should now have:
```bash
├── .github/
│   └── workflows/
│       └── model_training.yml          # GitHub Actions workflow
├── tests/
│   └── test_modeling.py               # Unit tests
├── 01_Data_processing.py
├── 02_Data_Analysis.ipynb
├── 03_Modeling.py                     # Refactored for GitHub Actions
├── GITHUB_ACTIONS_CHANGES.md          # Documentation of changes
├── requirements.txt
└── README.md
```

#### Step 3: Push to GitHub

```bash
git add .
git commit -m "Add GitHub Actions CI/CD pipeline and refactored modeling code"
git push origin main
```

#### Step 4: Watch the workflow run

1. Go to your GitHub repository
2. Click on the **Actions** tab
3. You should see the "CLV Modeling Pipeline - GitHub Actions" workflow running

---

#### What the Workflow Does

The `.github/workflows/model_training.yml` file sets up an automated pipeline with **5 jobs**:

| Job | Purpose | When |
|-----|---------|------|
| **Data Processing** | Cleans and prepares data | Always runs first |
| **Model Training** | Trains ML models and generates results | After data processing |
| **Quality Checks** | Runs linting and code formatting | Parallel with training |
| **Notification** | Reports workflow status | After all jobs |
| **Performance Tracking** | Tracks model metrics over time | Only on success |

#### Triggers

The workflow runs automatically when:
- Code is pushed to `main` or `develop` branches
- Files in `03_Modeling.py`, `01_Data_processing.py`, `data/`, or `requirements.txt` change
- Pull requests are created
- **Weekly schedule**: Every Sunday at 2 AM UTC
- **Manual trigger**: Can be run manually via GitHub UI

#### Artifacts

After each run, GitHub stores:
- `model-results/` - CSV results and visualizations
- `performance-metrics/` - JSON metrics for tracking
- Logs from failed runs (if applicable)

Artifacts are kept for **30 days** and can be downloaded from the Actions tab.

---
#### GitHub Actions Workflow Example

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
### Installation & Setup

#### Prerequisites
- [Git](https://git-scm.com/) installed
- [Python 3.9+](https://www.python.org/)
- GitHub repository with admin access

#### Local Development Setup
1. **Clone the repository**
   ```bash
   git clone https://github.com/adiag321/Customer-Lifetime-Value-Prediction.git
   cd Customer-Lifetime-Value-Prediction
   ```
2. **Create a virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```
4. **Run the pipeline locally**
   ```bash
   python 03_Modeling.py
   ```
5. **Run tests locally**
   ```bash
   pip install pytest
   pytest tests/test_modeling.py -v
   ```
---

### Configuration

#### Environment Variables (Optional)
To pass custom settings, add secrets to GitHub:

1. Go to **Settings** → **Secrets and variables** → **Actions**
2. Click **New repository secret**
3. Add secrets like:
   - `DATA_PATH`: Custom data file path
   - `TEST_SIZE`: Custom test size (default 0.30)

Then use in workflow:
```yaml
- name: Train models
  env:
    DATA_PATH: ${{ secrets.DATA_PATH }}
  run: python 03_Modeling.py --data-path $DATA_PATH
```

#### Customizing Schedules

Edit `.github/workflows/model_training.yml` to change schedule:

```yaml
schedule:
  - cron: '0 2 * * 0'  # Change this cron expression
```

**Common schedule examples:**
- `'0 2 * * 0'` - Weekly Sunday 2 AM UTC
- `'0 0 * * *'` - Daily at midnight UTC
- `'0 */6 * * *'` - Every 6 hours
- `'0 0 1 * *'` - Monthly on 1st at midnight

#### Performance Thresholds

Adjust the R² threshold in `model_training.yml`:

```yaml
- name: Verify results
  run: |
    if best_r2 >= 0.85:  # Change this threshold
        print('✓ Performance meets quality threshold')
```

---

### Troubleshooting

#### Issue: Workflow fails with "FileNotFoundError"

**Cause**: Data file not found
**Solution**: 
- Ensure `data/Processed_AutoInsurance.csv` exists
- Run `01_Data_processing.py` first
- Check file paths are relative

#### Issue: "DISPLAY not set" error

**Cause**: matplotlib trying to use interactive backend
**Solution**: Already fixed in refactored code with `matplotlib.use('Agg')`

#### Issue: Memory errors on large datasets

**Cause**: GitHub Actions runs on limited hardware
**Solution**: Reduce hyperparameter search space or use smaller datasets in `--test-size`

#### Issue: Workflow not triggering

**Cause**: Wrong branch or paths
**Solution**: 
- Verify pushing to `main` or `develop`
- Check file path patterns in `on.push.paths`
- Try manual trigger via **Actions** tab

#### Issue: Tests failing

**Cause**: Missing test data or dependencies
**Solution**:
- Run `pip install pytest`
- Ensure test data is created in fixtures
- Check `requirements.txt` has all dependencies

---

### Monitoring & Analytics

#### View Workflow Results

1. **In GitHub UI**:
   - Actions tab → Select workflow run → View logs
   
2. **In Command Line**:
   ```bash
   gh run list --repo adiag321/Customer-Lifetime-Value-Prediction
   gh run view <RUN_ID> --log
   ```

3. **Download Artifacts**:
   ```bash
   gh run download <RUN_ID> -D ./artifacts
   ```

#### Performance Tracking

The workflow tracks model performance metrics:
- **Best R² Score**: Monitored across runs
- **RMSE**: Root mean squared error tracking
- **Training time**: Execution duration

View historical metrics in the **Performance Metrics** artifact.

#### Slack/Email Notifications

Add notifications for workflow completion:

```yaml
- name: Slack Notification
  if: failure()
  uses: slackapi/slack-github-action@v1
  with:
    webhook-url: ${{ secrets.SLACK_WEBHOOK }}
    payload: |
      {
        "text": "CLV Pipeline Failed",
        "blocks": [
          {
            "type": "section",
            "text": {
              "type": "mrkdwn",
              "text": "Training failed on ${{ github.ref }}"
            }
          }
        ]
      }
```
---

### Best Practices

#### Do's

- Keep `requirements.txt` up-to-date
- Run tests locally before pushing
- Use descriptive commit messages
- Review workflow logs after each run
- Version your data and models
- Set appropriate cron schedules (not too frequent)
- Use branch protection rules with required CI/CD checks

#### Don'ts

- Don't commit large data files (use Git LFS)
- Don't store secrets in code or workflow files
- Don't use hardcoded absolute paths
- Don't skip quality checks
- Don't run too many frequent schedules (GitHub quotas)

---

### Advanced Configuration

#### Adding Code Coverage

```yaml
- name: Install coverage
  run: pip install coverage pytest-cov

- name: Run tests with coverage
  run: pytest --cov=src tests/
```

#### Adding Code Quality Checks

```yaml
- name: SonarQube Scan
  uses: SonarSource/sonarcloud-github-action@master
```

#### Automatic Model Versioning

```yaml
- name: Tag model release
  if: github.event_name == 'push' && github.ref == 'refs/heads/main'
  run: |
    git tag -a model-$(date +%Y%m%d) -m "Model from GitHub Actions"
    git push origin --tags
```

---
### Useful Resources

- [GitHub Actions Documentation](https://docs.github.com/en/actions)
- [Workflow Syntax Reference](https://docs.github.com/en/actions/using-workflows/workflow-syntax-for-github-actions)
- [Python Workflow Example](https://github.com/actions/starter-workflows/blob/main/ci/python-app.yml)
- [Schedule Syntax (Cron)](https://crontab.guru/)

---

### Support

If you encounter issues:

1. **Check workflow logs** in GitHub Actions tab
2. **Review** `GITHUB_ACTIONS_CHANGES.md` for technical details
3. **Run locally** to test: `python 03_Modeling.py`
4. **Check Python version** matches what workflow expects
5. **Verify dependencies** in `requirements.txt`

---

### Summary

Your project is now set up with:
- Automated data processing
- Model training pipeline
- Automated testing
- Code quality checks
- Result artifacts
- Performance tracking

The refactored `03_Modeling.py` is production-ready for CI/CD environments!
