# DoPred

 Python port of the `nl_did_vpyield_fixed.do` workflow for estimating Diebold-Li style yield curves with company and month fixed effects. 除了命令行脚本外，还提供了等价的 Jupyter notebook（`nl_did_vpyield_fixed.ipynb`）方便交互式调试。

## Quick start

1. Install dependencies (Python ≥3.10):

   ```bash
   pip install pandas numpy scipy matplotlib seaborn pyarrow
   ```

2. Run the estimator (replace the CSV path with your own export):

   ```bash
   python nl_did_vpyield_fixed.py \
     --input /path/to/bond_return_with_kcbz.csv \
     --output-dir outputs \
     --export-dta \
     --verbose
   ```

Key arguments mirror the original Stata script:

- `--key-date` and `--window-days` define the policy event window (default ±90 days around 2025-06-18).
- `--max-fe-iters` / `--fe-tol` control how the company/month fixed effects are iteratively partialled out inside each nonlinear step.
- `--max-nl-steps` controls the SciPy `least_squares` iterations (default 10,000 evaluations).
- `--initial-L` sets the starting value for the Diebold-Li decay parameter (default 1.0).

Outputs go to `--output-dir` (CSV of coefficients, margins table, parquet snapshot, optional `.dta`, scatter plot, and JSON summary).

### Notebook usage

1. 启动 Jupyter：`jupyter notebook` 或 `jupyter lab`。
2. 打开 `nl_did_vpyield_fixed.ipynb`，在“参数设置”单元中填入真实文件路径/事件日期。
3. 自上而下运行全部单元，输出会写入 `outputs_notebook/`（可在参数单元自定义）。*** End Patch