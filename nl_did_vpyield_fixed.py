#!/usr/bin/env python3
"""
Python translation of the Stata do-file `nl_did_vpyield_fixed.do`.
中文概要：
- 本脚本重现 Stata do-file 的完整处理流程，用 Python 实现数据清洗、DiD 变量生成、非线性 Diebold-Li 模型估计以及结果输出。

Workflow summary 工作流摘要：
1. Read the CSV exported from the bond return data pull.
   读取债券收益率导出的 CSV。
2. Coerce date and numeric columns to the correct dtypes.
   将日期/数值字段转换成适当的数据类型。
3. Generate treatment/Post/DiD indicators inside a ±90 day window around 2025-06-18.
   在政策窗口内生成 Treated、Post、DiD 指示变量。
4. Keep BdType ∈ {13, 14, 19}.
   仅保留指定债券类型。
5. Estimate a Diebold-Li style nonlinear model with company & month fixed effects,
   solved via alternating projections within each nonlinear least squares iteration.
   在每次非线性最小二乘迭代中交替剥离公司与月份固定效应，拟合 Diebold-Li 模型。
6. Output parameter estimates, predicted values, a scatter plot, marginal effects,
   and a Stata-compatible `.dta` snapshot for downstream checks.
   输出系数表、预测值、散点图、边际效应以及 .dta 备份。
"""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numpy.typing import NDArray
from scipy.optimize import least_squares
from scipy.stats import t as student_t


# 参数名称顺序列表，方便后续在数组中按名称定位（与 Stata `parameters()` 顺序一致）
PARAM_ORDER: List[str] = [
    "b1Cb",
    "b2Cb",
    "b3Cb",
    "b1Ca",
    "b2Ca",
    "b3Ca",
    "b1Tb",
    "b2Tb",
    "b3Tb",
    "b1Ta",
    "b2Ta",
    "b3Ta",
    "L",
]


def safe_to_parquet(df: pd.DataFrame, path: Path) -> Path:
    """
    优先写入 Parquet；若环境缺少 pyarrow/fastparquet，则降级为 CSV 并记录警告。
    返回实际写入的文件路径，便于日志提示。
    """
    try:
        df.to_parquet(path, index=False)
        return path
    except (ImportError, ModuleNotFoundError) as exc:
        fallback_path = path.with_suffix(".csv")
        logging.warning(
            "Parquet engine unavailable (%s); falling back to CSV: %s",
            exc,
            fallback_path,
        )
        df.to_csv(fallback_path, index=False)
        return fallback_path


def positive_float(value: str) -> float:
    """确保 CLI 输入的浮点数为正数，避免无效的 L 初值或窗口设置。"""
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive float.")
    return parsed


def parse_args() -> argparse.Namespace:
    """集中定义所有命令行参数，保持与 Stata do-file 参数一致并新增调试开关。"""
    parser = argparse.ArgumentParser(
        description="Nonlinear Diebold-Li FE estimator (Python port of the Stata do-file)."
    )
    parser.add_argument(
        "-i",
        "--input",
        required=True,
        type=Path,
        help="Path to `bond_return_with_kcbz.csv` (or equivalent).",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="Directory for results, figures, and exported data.",
    )
    parser.add_argument(
        "--key-date",
        type=pd.Timestamp,
        default=pd.Timestamp("2025-06-18"),
        help="Policy event date used for Post/DiD (default: 2025-06-18).",
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=90,
        help="Half-width of the symmetric time window around `key-date`.",
    )
    parser.add_argument(
        "--max-fe-iters",
        type=int,
        default=50,
        help="Maximum alternating-projection iterations for the FE sweeps.",
    )
    parser.add_argument(
        "--fe-tol",
        type=float,
        default=1e-10,
        help="Tolerance for FE convergence (max absolute change).",
    )
    parser.add_argument(
        "--max-nl-steps",
        type=int,
        default=10_000,
        help="Maximum nonlinear least-squares iterations (SciPy evaluations).",
    )
    parser.add_argument(
        "--initial-L",
        type=positive_float,
        default=1.0,
        help="Starting value for the Diebold-Li decay parameter L.",
    )
    parser.add_argument(
        "--scatter-filename",
        type=str,
        default="scatter_by_bdtype.png",
        help="Name of the scatter plot file written to `output-dir`.",
    )
    parser.add_argument(
        "--export-dta",
        action="store_true",
        help="Also export the processed sample as `did_analysis_data2.dta`.",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug logging and verbose optimizer output.",
    )
    return parser.parse_args()


def setup_logging(verbose: bool) -> None:
    """根据 verbose 选择日志级别，保证批量运行时也能记录关键步骤。"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def read_csv(path: Path) -> pd.DataFrame:
    """读取原始 CSV 并输出形状，用于快速确认输入数据量。"""
    logging.info("Reading CSV: %s", path)
    df = pd.read_csv(path)
    logging.debug("Raw shape: %s", df.shape)
    return df


def coerce_date_column(df: pd.DataFrame, column: str) -> None:
    """把指定列转换为日期；缺失列时仅提示，不终止流程。"""
    if column not in df.columns:
        logging.info("Date column `%s` not found; skipping.", column)
        return
    df[column] = pd.to_datetime(df[column], errors="coerce")
    logging.debug(
        "Converted `%s` to datetime. Missing ratio: %.3f",
        column,
        df[column].isna().mean(),
    )


def coerce_numeric_column(df: pd.DataFrame, column: str) -> None:
    """将字符串数字安全转换为浮点，便于后续矩阵计算。"""
    if column not in df.columns:
        logging.warning("Numeric column `%s` not found.", column)
        return
    df[column] = pd.to_numeric(df[column], errors="coerce")
    logging.debug(
        "Converted `%s` to numeric. Missing ratio: %.3f",
        column,
        df[column].isna().mean(),
    )


def normalize_kcbz(df: pd.DataFrame) -> None:
    """将 kcbz 字段统一转换为 0/1 dummy，并创建 Treated 变量。"""
    col = "kcbz"
    if col not in df.columns:
        logging.error("`kcbz` not found; defaulting Treated to 0.")
        df[col] = 0

    if pd.api.types.is_numeric_dtype(df[col]):
        numeric = df[col]
    else:
        lowered = df[col].astype(str).str.strip().str.lower()
        mapping = {
            "1": 1,
            "y": 1,
            "yes": 1,
            "true": 1,
            "t": 1,
            "0": 0,
            "n": 0,
            "no": 0,
            "false": 0,
            "f": 0,
        }
        numeric = lowered.map(mapping)
        numeric = numeric.fillna(pd.to_numeric(df[col], errors="coerce"))
        numeric = numeric.fillna(0)

    df[col] = (numeric.fillna(0) != 0).astype(np.int8)
    df["Treated"] = df[col].astype(np.int8)
    logging.debug("Created Treated dummy from `kcbz`.")


def ensure_end_date(df: pd.DataFrame) -> None:
    """确保存在 EndDt 列，否则回退到 ValueDt 或 IssDt。"""
    if "EndDt" in df.columns:
        return
    for candidate in ("ValueDt", "IssDt"):
        if candidate in df.columns:
            df["EndDt"] = pd.to_datetime(df[candidate], errors="coerce")
            logging.info("Using `%s` as EndDt.", candidate)
            return
    raise ValueError("No EndDt/ValueDt/IssDt found. Cannot proceed.")


def filter_bdtype(df: pd.DataFrame) -> pd.DataFrame:
    """筛选 BdType ∈ {13,14,19}，与 Stata keep 逻辑一致。"""
    if "BdType" not in df.columns:
        raise ValueError("BdType not found. Required for filtering.")
    numeric = pd.to_numeric(df["BdType"], errors="coerce")
    mask = numeric.isin({13, 14, 19})
    kept = df.loc[mask].copy()
    logging.info(
        "Filtered BdType. Kept %d / %d rows (%.1f%%).",
        kept.shape[0],
        df.shape[0],
        100 * kept.shape[0] / max(df.shape[0], 1),
    )
    kept["BdType"] = numeric.loc[mask].astype("Int64")
    return kept


def assign_time_window(
    df: pd.DataFrame, key_date: pd.Timestamp, window_days: int
) -> pd.DataFrame:
    """生成时间窗口样本、Post、DiD 变量，完全对齐 Stata 中的逻辑。"""
    if "EndDt" not in df.columns:
        raise ValueError("EndDt is required before time window filtering.")

    df = df.copy()
    df["time_window"] = (
        df["EndDt"].sub(key_date).dt.days.abs() <= window_days
    )
    before = df.shape[0]
    df = df.loc[df["time_window"]].copy()
    logging.info(
        "Applied ±%d day window around %s. Kept %d / %d rows.",
        window_days,
        key_date.date(),
        df.shape[0],
        before,
    )
    df["Post"] = (df["EndDt"] >= key_date).astype(np.int8)
    df["DiD"] = (df["Treated"] * df["Post"]).astype(np.int8)
    return df


def ensure_company_code(df: pd.DataFrame) -> None:
    """确认 CompanyCode 存在并转为字符串，避免 merge/panel 歧义。"""
    if "CompanyCode" not in df.columns:
        raise ValueError("CompanyCode not found; needed for firm FE.")
    df["CompanyCode"] = df["CompanyCode"].astype(str)


def prepare_dataframe(
    csv_path: Path, key_date: pd.Timestamp, window_days: int
) -> pd.DataFrame:
    """按 Stata 步骤顺序完成所有清洗、筛选、指示变量生成。"""
    df = read_csv(csv_path)

    # 1) 日期列：尝试转换 EndDt/ValueDt/MatDt
    for dcol in ("EndDt", "ValueDt", "MatDt"):
        if dcol in df.columns:
            coerce_date_column(df, dcol)
    ensure_end_date(df)
    df["EndDt_month"] = df["EndDt"].dt.year * 100 + df["EndDt"].dt.month

    # 2) 数值列：收益率、TrueYield、久期
    for vcol in ("VPYield", "Maturity", "TrueYield"):
        coerce_numeric_column(df, vcol)

    # 3) 处理组 dummy、公司识别
    normalize_kcbz(df)
    ensure_company_code(df)
    # 4) 债券类型筛选 + 时间窗口 + 有效观测
    df = filter_bdtype(df)
    df = assign_time_window(df, key_date, window_days)
    df = df[df["Maturity"].notna() & df["Maturity"].gt(0)]
    df = df[df["VPYield"].notna()]
    df = df[df["VPYield"] <= 20]
    df = df[df["EndDt_month"].notna()]

    # Drop rows with missing inputs used by the evaluator.
    required = [
        "VPYield",
        "Maturity",
        "Post",
        "Treated",
        "DiD",
        "CompanyCode",
        "EndDt_month",
    ]
    before = df.shape[0]
    df = df.dropna(subset=required)
    logging.info("Dropped %d rows due to missing required fields.", before - df.shape[0])
    df.reset_index(drop=True, inplace=True)
    return df


def describe_data(df: pd.DataFrame) -> None:
    """输出关键变量的描述统计，便于和 Stata summarize 对照。"""
    cols = ["VPYield", "Maturity", "Post", "Treated", "DiD"]
    available = [c for c in cols if c in df.columns]
    if not available:
        return
    summary = df[available].describe().transpose()
    logging.info("Descriptive statistics:\n%s", summary.to_string(float_format="%.4f"))


def _group_mean(values: NDArray[np.float64], groups: NDArray[np.int64], n_groups: int) -> NDArray[np.float64]:
    """使用 np.bincount 计算分组均值，复刻 bysort egen mean 的效果。"""
    sums = np.bincount(groups, weights=values, minlength=n_groups)
    counts = np.bincount(groups, minlength=n_groups)
    means = np.divide(
        sums,
        counts,
        out=np.zeros_like(sums, dtype=np.float64),
        where=counts > 0,
    )
    return means


def alternating_fe(
    resid: NDArray[np.float64],
    company_idx: NDArray[np.int64],
    month_idx: NDArray[np.int64],
    n_companies: int,
    n_months: int,
    *,
    maxiter: int,
    tol: float,
) -> Tuple[NDArray[np.float64], NDArray[np.float64], int]:
    """迭代在公司/月份之间交替平滑固定效应，直到收敛或达到上限。"""
    alpha = np.zeros(n_companies, dtype=np.float64)
    gamma = np.zeros(n_months, dtype=np.float64)

    for iteration in range(1, maxiter + 1):
        alpha_old = alpha.copy()
        gamma_old = gamma.copy()

        # Step A: 固定公司效应，利用月份均值更新月份效应 gamma
        tmp = resid - alpha[company_idx]
        gamma = _group_mean(tmp, month_idx, n_months)

        # Step B: 固定最新月份效应，按公司均值更新 alpha
        tmp2 = resid - gamma[month_idx]
        alpha = _group_mean(tmp2, company_idx, n_companies)

        center = (alpha.mean() + gamma.mean())
        alpha -= center / 2
        gamma -= center / 2

        combined_new = alpha[company_idx] + gamma[month_idx]
        combined_old = alpha_old[company_idx] + gamma_old[month_idx]
        maxchg = np.max(np.abs(combined_new - combined_old))
        if maxchg < tol:
            logging.debug("FE converged in %d iterations.", iteration)
            return alpha, gamma, iteration

    logging.warning(
        "FE solver hit max iterations (%d) without convergence (tol %.1e).",
        maxiter,
        tol,
    )
    return alpha, gamma, maxiter


class DieboldLiFEModel:
    """封装 Diebold-Li 结构项与公司/月度固定效应求解，便于 SciPy 调用。"""
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        max_fe_iters: int,
        fe_tol: float,
    ) -> None:
        """准备模型所需的 numpy 缓存和索引，加速后续迭代。"""
        self.df = df.copy()
        self.y = df["VPYield"].to_numpy(dtype=np.float64)
        self.maturity = df["Maturity"].to_numpy(dtype=np.float64)
        self.post = df["Post"].to_numpy(dtype=np.float64)
        self.treated = df["Treated"].to_numpy(dtype=np.float64)
        self.did = df["DiD"].to_numpy(dtype=np.float64)

        companies, company_idx = np.unique(df["CompanyCode"].astype(str), return_inverse=True)
        months, month_idx = np.unique(df["EndDt_month"].astype(int), return_inverse=True)

        self.company_codes = companies
        self.month_codes = months
        self.company_idx = company_idx.astype(np.int64)
        self.month_idx = month_idx.astype(np.int64)
        self.n_companies = companies.shape[0]
        self.n_months = months.shape[0]

        self.max_fe_iters = max_fe_iters
        self.fe_tol = fe_tol
        self.latest_alpha = np.zeros(self.n_companies)
        self.latest_gamma = np.zeros(self.n_months)
        self.latest_terms: Dict[str, NDArray[np.float64]] = {}
        self.last_fe_iterations = 0

    @staticmethod
    def _terms(maturity: NDArray[np.float64], L: float) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
        """计算 Diebold-Li 中的两个基函数 term1/term2，并对 L→0 做数值保护。"""
        safe_L = np.clip(L, 1e-10, None)
        denom = safe_L * maturity
        term1 = np.ones_like(maturity)
        mask = (maturity > 0) & (safe_L > 0)
        term1[mask] = (1 - np.exp(-safe_L * maturity[mask])) / denom[mask]
        term2 = term1 - np.exp(-safe_L * maturity)
        return term1, term2

    def structural_component(
        self, params: NDArray[np.float64]
    ) -> Tuple[NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]:
        """根据参数向量生成结构部分拟合值，并缓存 term1/term2。"""
        (
            b1Cb,
            b2Cb,
            b3Cb,
            b1Ca,
            b2Ca,
            b3Ca,
            b1Tb,
            b2Tb,
            b3Tb,
            b1Ta,
            b2Ta,
            b3Ta,
            L,
        ) = params

        term1, term2 = self._terms(self.maturity, L)

        # 结构项：依次包含基准载荷（Cb）、Post 载荷（Ca）、Treat 载荷（Tb）、DiD 载荷（Ta）
        fitted = (
            b1Cb
            + b2Cb * term1
            + b3Cb * term2
            + b1Ca * self.post
            + b2Ca * self.post * term1
            + b3Ca * self.post * term2
            + b1Tb * self.treated
            + b2Tb * self.treated * term1
            + b3Tb * self.treated * term2
            + b1Ta * self.did
            + b2Ta * self.did * term1
            + b3Ta * self.did * term2
        )

        self.latest_terms = {"term1": term1, "term2": term2}
        return fitted, term1, term2

    def residuals(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """供 least_squares 调用：先算结构项，再交替剥离固定效应，返回残差。"""
        fitted, _, _ = self.structural_component(params)
        resid = self.y - fitted
        alpha, gamma, iters = alternating_fe(
            resid,
            self.company_idx,
            self.month_idx,
            self.n_companies,
            self.n_months,
            maxiter=self.max_fe_iters,
            tol=self.fe_tol,
        )
        self.latest_alpha = alpha
        self.latest_gamma = gamma
        self.last_fe_iterations = iters
        fe_adjustment = alpha[self.company_idx] + gamma[self.month_idx]
        final_resid = self.y - (fitted + fe_adjustment)
        return final_resid

    def predict(self, params: NDArray[np.float64]) -> NDArray[np.float64]:
        """在估计完成后返回含 FE 的总体预测值，用于作图/导出。"""
        fitted, _, _ = self.structural_component(params)
        alpha_obs = self.latest_alpha[self.company_idx]
        gamma_obs = self.latest_gamma[self.month_idx]
        return fitted + alpha_obs + gamma_obs

    def margins_did(
        self,
        params: NDArray[np.float64],
        maturities: Iterable[float],
        cov: NDArray[np.float64],
        dof: int,
    ) -> pd.DataFrame:
        """计算不同期限下 DiD 的边际效应，附带 t 检验与 p 值，并在控制台打印。"""
        maturities = np.asarray(list(maturities), dtype=np.float64)
        term1, term2 = self._terms(maturities, params[-1])

        idx = [PARAM_ORDER.index(name) for name in ("b1Ta", "b2Ta", "b3Ta")]
        beta = params[idx]
        cov_sub = cov[np.ix_(idx, idx)]

        weights = np.column_stack((np.ones_like(maturities), term1, term2))
        effects = weights @ beta
        variances = np.einsum("ij,jk,ik->i", weights, cov_sub, weights)
        std_err = np.sqrt(np.maximum(variances, 0))
        with np.errstate(divide="ignore", invalid="ignore"):
            t_values = np.divide(effects, std_err, out=np.zeros_like(effects), where=std_err > 0)
        p_values = 2 * student_t.sf(np.abs(t_values), dof)

        crit = student_t.ppf(0.975, dof)
        ci_low = effects - crit * std_err
        ci_high = effects + crit * std_err

        margins_df = pd.DataFrame(
            {
                "Maturity": maturities,
                "term1": term1,
                "term2": term2,
                "dydx_DiD": effects,
                "std_err": std_err,
                "t_value": t_values,
                "p_value": p_values,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }
        )

        print("\nDiD marginal effects with t-tests")
        print("-" * 86)
        print(f"{'Maturity':>10} | {'Effect':>10}  {'Std. Err.':>10}  {'t':>7}  {'P>|t|':>7}     [95% conf. interval]")
        print("-" * 86)
        for row in margins_df.itertuples(index=False):
            print(
                f"{row.Maturity:10.2f} | {row.dydx_DiD:10.6f}  {row.std_err:10.6f}  "
                f"{row.t_value:7.2f}  {row.p_value:7.3f}     {row.ci_low:10.6f}  {row.ci_high:10.6f}"
            )
        print("-" * 86)

        return margins_df


def run_nl_regression(
    model: DieboldLiFEModel,
    *,
    initial_L: float,
    max_nl_steps: int,
    verbose: bool,
) -> Tuple[np.ndarray, least_squares, List[float]]:
    """调用 SciPy 最小二乘以 Stata 的初值为起点估计全部参数。"""
    # 与 Stata initial() 一致的初值（除 L 由 CLI 控制）
    initial_guess = np.array(
        [
            0.01,  # b1Cb
            0.01,  # b2Cb
            0.01,  # b3Cb
            0.0,  # b1Ca
            0.0,  # b2Ca
            0.0,  # b3Ca
            0.0,  # b1Tb
            0.0,  # b2Tb
            0.0,  # b3Tb
            0.0,  # b1Ta
            0.0,  # b2Ta
            0.0,  # b3Ta
            initial_L,  # L
        ],
        dtype=np.float64,
    )

    # L 需要保持正值，其他参数允许正负
    lower_bounds = np.array([-np.inf] * (len(PARAM_ORDER) - 1) + [1e-10])
    upper_bounds = np.array([np.inf] * len(PARAM_ORDER))

    # 记录每次迭代的残差平方和 (SSE)
    iteration_history: List[float] = []
    initial_resid = model.residuals(initial_guess)
    iteration_history.append(float(np.sum(initial_resid**2)))

    def _callback(*args) -> None:
        """兼容不同 SciPy 版本的回调接口，记录 SSE 轨迹。"""
        if len(args) == 3:
            _, cost, _ = args
            sse = float(2 * cost)
        else:
            params_current = args[0]
            resid = model.residuals(params_current)
            sse = float(np.sum(resid**2))
        iteration_history.append(sse)

    lsq_kwargs = dict(
        fun=model.residuals,
        x0=initial_guess,
        bounds=(lower_bounds, upper_bounds),
        max_nfev=max_nl_steps,
        verbose=2 if verbose else 0,
    )

    try:
        lsq = least_squares(**lsq_kwargs, callback=_callback)
    except TypeError as exc:
        if "callback" not in str(exc):
            raise
        logging.info("SciPy least_squares callback unsupported; falling back without intermediate tracking.")
        lsq = least_squares(**lsq_kwargs)
        iteration_history.append(float(2 * lsq.cost))
    if not lsq.success:
        logging.warning("Least squares did not converge: %s", lsq.message)

    params = lsq.x
    return params, lsq, iteration_history


def summarize_results(
    model: DieboldLiFEModel, params: NDArray[np.float64], lsq: least_squares
) -> Tuple[pd.DataFrame, Dict[str, float], NDArray[np.float64]]:
    """计算 SSE/R²/标准误并生成结果表，方便与 Stata estimates table 对照。"""
    residuals = lsq.fun
    n_obs = residuals.size
    dof = max(n_obs - len(params), 1)
    sse = np.sum(residuals**2)
    mse = sse / dof
    y = model.y
    sst = np.sum((y - np.mean(y)) ** 2)
    r2 = 1 - sse / sst if sst > 0 else np.nan

    jac = lsq.jac
    try:
        cov = mse * np.linalg.inv(jac.T @ jac)
    except np.linalg.LinAlgError:
        cov = mse * np.linalg.pinv(jac.T @ jac)

    stderr = np.sqrt(np.diag(cov))
    t_values = params / stderr
    p_values = 2 * student_t.sf(np.abs(t_values), dof)
    crit = student_t.ppf(0.975, dof)
    ci_low = params - crit * stderr
    ci_high = params + crit * stderr
    results = pd.DataFrame(
        {
            "param": PARAM_ORDER,
            "estimate": params,
            "std_err": stderr,
            "t_value": t_values,
            "p_value": p_values,
            "ci_low": ci_low,
            "ci_high": ci_high,
        }
    )
    adj_r2 = 1 - (1 - r2) * (n_obs - 1) / max(n_obs - len(params), 1)
    stats = {
        "r2": r2,
        "adj_r2": adj_r2,
        "root_mse": np.sqrt(mse),
        "n_obs": n_obs,
        "sse": sse,
        "mse": mse,
        "dof": dof,
    }
    return results, stats, cov


def save_outputs(
    df: pd.DataFrame,
    params_df: pd.DataFrame,
    margins_df: pd.DataFrame,
    *,
    output_dir: Path,
    scatter_filename: str,
    export_dta: bool,
) -> None:
    """将估计输出写入 CSV/Parquet/图像/可选 Stata dta，复用输出目录。"""
    output_dir.mkdir(parents=True, exist_ok=True)

    params_path = output_dir / "nl_regression_results.csv"
    params_df.to_csv(params_path, index=False)
    logging.info("Saved parameter table to %s", params_path)

    margins_path = output_dir / "margins_did.csv"
    margins_df.to_csv(margins_path, index=False)
    logging.info("Saved margins table to %s", margins_path)

    data_path = output_dir / "did_analysis_data2.parquet"
    actual_data_path = safe_to_parquet(df, data_path)
    logging.info("Saved processed dataset to %s", actual_data_path)

    if export_dta:
        dta_path = output_dir / "did_analysis_data2.dta"
        df.to_stata(dta_path, write_index=False, version=118)
        logging.info("Saved processed dataset to %s", dta_path)

    if "y_hat" in df.columns:
        plt.figure(figsize=(8, 6))
        sns.scatterplot(data=df, x="y_hat", y="VPYield", hue="BdType", palette="deep")
        plt.title("Actual vs Predicted VPYield (colored by BdType)")
        plt.xlabel("Predicted VPYield")
        plt.ylabel("Actual VPYield")
        plt.tight_layout()
        scatter_path = output_dir / scatter_filename
        plt.savefig(scatter_path, dpi=300)
        plt.close()
        logging.info("Saved scatter plot to %s", scatter_path)


def print_iteration_history(history: List[float]) -> None:
    """按照指定格式打印每次迭代的残差平方和。"""
    print()
    for idx, value in enumerate(history):
        label = f"Iteration {idx}:"
        print(f"{label:<12} Residual SS = {value:11.5f}")
    print()


def print_regression_summary(stats: Dict[str, float], results: pd.DataFrame) -> None:
    """以 Stata 风格输出回归概览与系数表。"""
    print(
        f"Nonlinear regression".ljust(52)
        + f"Number of obs = {stats['n_obs']:>10,.0f}"
    )
    print("".ljust(52) + f"R-squared     = {stats['r2']:>10.4f}")
    print("".ljust(52) + f"Adj R-squared = {stats['adj_r2']:>10.4f}")
    print("".ljust(52) + f"Root MSE      = {stats['root_mse']:>10.6f}")

    print("\n" + "-" * 78)
    print(
        f"{'Param':>12} | {'Coefficient':>11}  {'Std. Err.':>10}  {'t':>7}  {'P>|t|':>7}     [95% conf. interval]"
    )
    print("-" * 78)
    for row in results.itertuples(index=False):
        name = f"/{row.param}"
        print(
            f"{name:>12} | {row.estimate:11.6f}  {row.std_err:10.6f}  "
            f"{row.t_value:7.2f}  {row.p_value:7.3f}     {row.ci_low:10.6f}  {row.ci_high:10.6f}"
        )
    print("-" * 78 + "\n")


def main() -> None:
    """脚本入口：解析参数、准备数据、估计、保存并写 summary JSON。"""
    args = parse_args()
    setup_logging(args.verbose)

    df = prepare_dataframe(args.input, args.key_date, args.window_days)
    describe_data(df)

    model = DieboldLiFEModel(
        df,
        max_fe_iters=args.max_fe_iters,
        fe_tol=args.fe_tol,
    )

    params, lsq, iteration_history = run_nl_regression(
        model,
        initial_L=args.initial_L,
        max_nl_steps=args.max_nl_steps,
        verbose=args.verbose,
    )
    print_iteration_history(iteration_history)

    y_hat = model.predict(params)
    df = df.copy()
    df["y_hat"] = y_hat

    maturities = [0.25, 0.5, 1, 2, 3, 5, 10, 15]
    params_df, stats, cov = summarize_results(model, params, lsq)
    print_regression_summary(stats, params_df)
    margins = model.margins_did(params, maturities, cov, int(stats["dof"]))
    save_outputs(
        df,
        params_df,
        margins,
        output_dir=args.output_dir,
        scatter_filename=args.scatter_filename,
        export_dta=args.export_dta,
    )

    summary_path = args.output_dir / "nl_results_summary.json"
    summary_payload = {
        "param_order": PARAM_ORDER,
        "params": params.tolist(),
        "optimizer_success": bool(lsq.success),
        "optimizer_message": lsq.message,
        "fe_iterations": model.last_fe_iterations,
        "stats": stats,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    logging.info("Wrote summary JSON to %s", summary_path)


if __name__ == "__main__":
    main()
