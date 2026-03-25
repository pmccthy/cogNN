# Running A2C Reversal ABC Gridsearch and Analysis

This guide explains how to run the gridsearch script and generate summary plots.

## Prerequisites

1. Ensure your conda environment is set up:
   ```bash
   conda env create -f environment.yml  # If not already created
   conda activate cog_nn
   ```

2. Ensure the task data file exists:
   - The script expects `/Users/pmccarthy/Documents/cogNN/task_data/reversal_abc.pkl`

## Step 1: Run the Gridsearch

The gridsearch script trains models for all parameter combinations defined in the script.

```bash
cd /Users/pmccarthy/Documents/cogNN
conda activate cog_nn  # If not already activated
python experiment_scripts/a2c_reversal_abc_gridsearch.py
```

**What it does:**
- Trains models for all combinations of hyperparameters (learning_rate, gamma, model_size, lick_no_reward, policy_clip)
- Runs 5 training runs per parameter combination (configurable via `num_runs_per_combination`)
- Saves results to: `/Users/pmccarthy/Documents/modelling_results/december_2025_a2c/ff_a2c_reversal_abc_gridsearch/`
- Each run saves:
  - `metrics.pkl` - Training metrics (lick probabilities, values, rewards)
  - `params.json` - Parameter configuration
  - `actor.pth` and `critic.pth` - Trained model weights
- Creates `gridsearch_summary.json` with a summary of all runs

**Note:** This can take a while depending on the number of parameter combinations. The script prints progress as it runs.

## Step 2: Run the Analysis Script

After the gridsearch completes, run the analysis script to generate summary plots:

```bash
cd /Users/pmccarthy/Documents/cogNN
conda activate cog_nn  # If not already activated
python experiment_scripts/analyse_gridsearch_results.py
```

**What it does:**
- Reads all results from the gridsearch output directory
- Groups runs by parameter combination
- Creates summary plots for each parameter combination showing:
  - Lick probabilities over training (for stimuli A, B, C)
  - Value estimates over training
  - Rewards over time (smoothed)
  - Lick probabilities at end of pre-reversal and post-reversal phases
- Saves plots to: `/Users/pmccarthy/Documents/modelling_results/december_2025_a2c/ff_a2c_reversal_abc_gridsearch/vis/`

## Output Files

### Gridsearch Output
- **Location:** `/Users/pmccarthy/Documents/modelling_results/december_2025_a2c/ff_a2c_reversal_abc_gridsearch/`
- **Files:**
  - `gridsearch_config.json` - Configuration used for the gridsearch
  - `gridsearch_summary.json` - Summary of all runs with paths to results
  - `combo_XXXX_run_XX_*/` - Directories for each run containing metrics, params, and model weights

### Analysis Output
- **Location:** `/Users/pmccarthy/Documents/modelling_results/december_2025_a2c/ff_a2c_reversal_abc_gridsearch/vis/`
- **Files:**
  - `combo_XXXX_*.png` - Summary plots for each parameter combination

## Customising Parameters

To modify the gridsearch parameters, edit `a2c_reversal_abc_gridsearch.py`:

```python
all_params = {
    "learning_rate": [0.0005, 0.001],  # Add/remove values to search
    "gamma": [0, 0.5],
    "model_size": [4, 8, 16],
    # ... etc
}

num_runs_per_combination = 5  # Change number of runs per combination
```

## Troubleshooting

1. **Python not found:** Make sure conda environment is activated
2. **Module not found:** Ensure you're in the project root directory
3. **Results directory doesn't exist:** The script will create it automatically
4. **Task data not found:** Check that `task_data/reversal_abc.pkl` exists

