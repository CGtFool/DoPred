#!/usr/bin/env python3
"""
Python translation of the Stata do-file `nl_did_vpyield_fixed.do`.

Workflow summary:
1. Read the CSV exported from the bond return data pull.
2. Coerce date and numeric columns to the correct dtypes.
3. Generate treatment/Post/DiD indicators inside a ±90 day window around 2025-06-18.
4. Keep BdType ∈ {13, 14, 19}.
5. Estimate a Diebold-Li style nonlinear model with company & month fixed effects,
   solved via alternating projections within each nonlinear least squares iteration.
6. Output parameter estimates, predicted values, a scatter plot, marginal effects,
   and a Stata-compatible `.dta` snapshot for downstream checks.
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


def positive_float(value: str) -> float:
    parsed = float(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("Expected a positive float.")
    return parsed


def parse_args() -> argparse.Namespace:
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
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def read_csv(path: Path) -> pd.DataFrame:
    logging.info("Reading CSV: %s", path)
    df = pd.read_csv(path)
    logging.debug("Raw shape: %s", df.shape)
    return df


def coerce_date_column(df: pd.DataFrame, column: str) -> None:
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
    if "EndDt" in df.columns:
        return
    for candidate in ("ValueDt", "IssDt"):
        if candidate in df.columns:
            df["EndDt"] = pd.to_datetime(df[candidate], errors="coerce")
            logging.info("Using `%s` as EndDt.", candidate)
            return
    raise ValueError("No EndDt/ValueDt/IssDt found. Cannot proceed.")


def filter_bdtype(df: pd.DataFrame) -> pd.DataFrame:
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
    if "CompanyCode" not in df.columns:
        raise ValueError("CompanyCode not found; needed for firm FE.")
    df["CompanyCode"] = df["CompanyCode"].astype(str)


def prepare_dataframe(
    csv_path: Path, key_date: pd.Timestamp, window_days: int
) -> pd.DataFrame:
    df = read_csv(csv_path)

    for dcol in ("EndDt", "ValueDt", "MatDt"):
        if dcol in df.columns:
            coerce_date_column(df, dcol)
    ensure_end_date(df)
    df["EndDt_month"] = df["EndDt"].dt.year * 100 + df["EndDt"].dt.month

    for vcol in ("VPYield", "Maturity", "TrueYield"):
        coerce_numeric_column(df, vcol)

    normalize_kcbz(df)
    ensure_company_code(df)
    df = filter_bdtype(df)
    df = assign_time_window(df, key_date, window_days)
    df = df[df["Maturity"].notna() & df["Maturity"].gt(0)]
    df = df[df["VPYield"].notna()]
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
    cols = ["VPYield", "Maturity", "Post", "Treated", "DiD"]
    available = [c for c in cols if c in df.columns]
    if not available:
        return
    summary = df[available].describe().transpose()
    logging.info("Descriptive statistics:\n%s", summary.to_string(float_format="%.4f"))


def _group_mean(values: NDArray[np.float64], groups: NDArray[np.int64], n_groups: int) -> NDArray[np.float64]:
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
    alpha = np.zeros(n_companies, dtype=np.float64)
    gamma = np.zeros(n_months, dtype=np.float64)

    for iteration in range(1, maxiter + 1):
        alpha_old = alpha.copy()
        gamma_old = gamma.copy()

        tmp = resid - alpha[company_idx]
        gamma = _group_mean(tmp, month_idx, n_months)

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
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        max_fe_iters: int,
        fe_tol: float,
    ) -> None:
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
        fitted, _, _ = self.structural_component(params)
        alpha_obs = self.latest_alpha[self.company_idx]
        gamma_obs = self.latest_gamma[self.month_idx]
        return fitted + alpha_obs + gamma_obs

    def margins_did(self, params: NDArray[np.float64], maturities: Iterable[float]) -> pd.DataFrame:
        maturities = np.asarray(list(maturities), dtype=np.float64)
        L = params[-1]
        term1, term2 = self._terms(maturities, L)

        b1Ta = params[PARAM_ORDER.index("b1Ta")]
        b2Ta = params[PARAM_ORDER.index("b2Ta")]
        b3Ta = params[PARAM_ORDER.index("b3Ta")]

        effect = b1Ta + b2Ta * term1 + b3Ta * term2
        return pd.DataFrame(
            {
                "Maturity": maturities,
                "term1": term1,
                "term2": term2,
                "dydx_DiD": effect,
            }
        )


def run_nl_regression(
    model: DieboldLiFEModel,
    *,
    initial_L: float,
    max_nl_steps: int,
    verbose: bool,
) -> Tuple[np.ndarray, least_squares]:
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

    lower_bounds = np.array([-np.inf] * (len(PARAM_ORDER) - 1) + [1e-10])
    upper_bounds = np.array([np.inf] * len(PARAM_ORDER))

    lsq = least_squares(
        model.residuals,
        initial_guess,
        bounds=(lower_bounds, upper_bounds),
        max_nfev=max_nl_steps,
        verbose=2 if verbose else 0,
    )
    if not lsq.success:
        logging.warning("Least squares did not converge: %s", lsq.message)

    params = lsq.x
    return params, lsq


def summarize_results(model: DieboldLiFEModel, params: NDArray[np.float64], lsq: least_squares) -> pd.DataFrame:
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
    results = pd.DataFrame(
        {
            "param": PARAM_ORDER,
            "estimate": params,
            "std_err": stderr,
            "t_value": t_values,
        }
    )
    logging.info("R-squared: %.4f, SSE: %.4f, observations: %d", r2, sse, n_obs)
    logging.info("Last FE iterations: %d", model.last_fe_iterations)
    logging.info("Parameter estimates:\n%s", results.to_string(index=False, float_format="%.6f"))
    return results.assign(r2=r2, sse=sse, n_obs=n_obs)


def save_outputs(
    df: pd.DataFrame,
    params_df: pd.DataFrame,
    margins_df: pd.DataFrame,
    *,
    output_dir: Path,
    scatter_filename: str,
    export_dta: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    params_path = output_dir / "nl_regression_results.csv"
    params_df.to_csv(params_path, index=False)
    logging.info("Saved parameter table to %s", params_path)

    margins_path = output_dir / "margins_did.csv"
    margins_df.to_csv(margins_path, index=False)
    logging.info("Saved margins table to %s", margins_path)

    data_path = output_dir / "did_analysis_data2.parquet"
    df.to_parquet(data_path, index=False)
    logging.info("Saved processed dataset (parquet) to %s", data_path)

    if export_dta:
        dta_path = output_dir / "did_analysis_data2.dta"
        df.to_stata(dta_path, write_index=False)
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


def main() -> None:
    args = parse_args()
    setup_logging(args.verbose)

    df = prepare_dataframe(args.input, args.key_date, args.window_days)
    describe_data(df)

    model = DieboldLiFEModel(
        df,
        max_fe_iters=args.max_fe_iters,
        fe_tol=args.fe_tol,
    )

    params, lsq = run_nl_regression(
        model,
        initial_L=args.initial_L,
        max_nl_steps=args.max_nl_steps,
        verbose=args.verbose,
    )

    y_hat = model.predict(params)
    df = df.copy()
    df["y_hat"] = y_hat

    maturities = [0.25, 0.5, 1, 2, 3, 5, 10, 15]
    margins = model.margins_did(params, maturities)

    params_df = summarize_results(model, params, lsq)
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
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")
    logging.info("Wrote summary JSON to %s", summary_path)


if __name__ == "__main__":
    main()
