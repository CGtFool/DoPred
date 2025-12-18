
import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from typing import Tuple, Dict, List
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import t as student_t

# Assuming the previous classes (DieboldLiFEModel) and functions exist in the context
# I will define the new classes/functions here.

class DieboldLiEventStudyModel(DieboldLiFEModel):
    """
    扩展模型以支持事件研究（Event Study），用于平行趋势检验。
    将 DiD 交互项替换为 Treated * RelativeMonth 的一系列交互。
    """
    def __init__(
        self, 
        df: pd.DataFrame, 
        key_date: pd.Timestamp,
        window_months: int = 3,
        **kwargs
    ):
        super().__init__(df, **kwargs)
        self.key_date = key_date
        self.window_months = window_months
        
        # 计算相对月份
        # 简单处理：(Year - KeyYear) * 12 + (Month - KeyMonth)
        key_y, key_m = key_date.year, key_date.month
        dates = pd.to_datetime(df['EndDt'], errors='coerce')
        # 既然 df['EndDt_month'] 已经是 yyyymm 格式
        y = (df['EndDt_month'] // 100).astype(int)
        m = (df['EndDt_month'] % 100).astype(int)
        
        self.rel_month = (y - key_y) * 12 + (m - key_m)
        
        # 生成相对时间 dummy 列表 [-window, ..., +window]
        # 注意：通常省略一期作为基准（例如 t=-1 或 t=0）
        # 这里我们选择 -1 作为基准
        self.rel_periods = [t for t in range(-window_months, window_months + 1) if t != -1]
        
        # 预计算每个相对期的 dummy * Treated
        # 形状: (n_periods, n_obs)
        self.event_dummies = {}
        for t in self.rel_periods:
            # 只有在对应相对月份且是 Treated 组的才为 1
            mask = (self.rel_month == t) & (self.treated == 1)
            self.event_dummies[t] = mask.astype(np.float64)
            
        # 参数数量：
        # 原有: Cb(3) + Ca(3) + Tb(3) = 9 (DiD项被取代)
        # 新增: len(rel_periods) * 3 (Level, Slope, Curvature per period)
        # L: 1
        self.n_event_params = len(self.rel_periods) * 3
        self.base_params_count = 9 # Cb, Ca, Tb
        
    def structural_component(
        self, params: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        # 解包基础参数
        # Cb (3), Ca (3), Tb (3)
        # 顺序: b1Cb, b2Cb, b3Cb, b1Ca, b2Ca, b3Ca, b1Tb, b2Tb, b3Tb
        base_params = params[:9]
        event_params = params[9:-1]
        L = params[-1]
        
        b1Cb, b2Cb, b3Cb = base_params[0:3]
        b1Ca, b2Ca, b3Ca = base_params[3:6]
        b1Tb, b2Tb, b3Tb = base_params[6:9]
        
        term1, term2 = self._terms(self.maturity, L)
        
        # 基础部分
        fitted = (
            b1Cb + b2Cb * term1 + b3Cb * term2 +
            b1Ca * self.post + b2Ca * self.post * term1 + b3Ca * self.post * term2 +
            b1Tb * self.treated + b2Tb * self.treated * term1 + b3Tb * self.treated * term2
        )
        
        # 事件研究部分
        # event_params 排列: [b1_t1, b2_t1, b3_t1, b1_t2, b2_t2, b3_t2, ...]
        for i, t in enumerate(self.rel_periods):
            p_idx = i * 3
            b1_t = event_params[p_idx]
            b2_t = event_params[p_idx+1]
            b3_t = event_params[p_idx+2]
            
            dummy = self.event_dummies[t]
            fitted += (
                b1_t * dummy +
                b2_t * dummy * term1 +
                b3_t * dummy * term2
            )
            
        return fitted, term1, term2

def run_parallel_trend_test(
    df: pd.DataFrame,
    key_date: pd.Timestamp,
    fixed_L: float,
    window_months: int = 3,
    **kwargs
):
    """
    执行平行趋势检验（事件研究法）。
    固定 L 为主模型估计值以减少计算负担。
    """
    logging.info("Starting Parallel Trend Test (Event Study)...")
    model = DieboldLiEventStudyModel(
        df, key_date, window_months=window_months, **kwargs
    )
    
    # 构造初始参数
    # Base (9) + Event (3 * n_periods) + L (1)
    n_params = 9 + len(model.rel_periods) * 3 + 1
    initial_guess = np.zeros(n_params)
    initial_guess[-1] = fixed_L # 设为传入的固定 L
    
    # 稍微给点初值避免完全为0
    initial_guess[0] = 2.0 # Constant level approx
    
    # 约束 L 不变 (通过 bounds)
    # 或者简单点，允许微调，或者设上下界极窄
    lower = [-np.inf] * (n_params - 1) + [fixed_L - 1e-6]
    upper = [np.inf] * (n_params - 1) + [fixed_L + 1e-6]
    
    res = least_squares(
        model.residuals,
        initial_guess,
        bounds=(lower, upper),
        max_nfev=2000, # 通常不需要太多，因为几乎是线性的
        verbose=1
    )
    
    # 提取结果用于绘图
    # 我们主要关注 Level (b1) 的系数，或者展示三者
    # 这里为了简便，提取 b1 (Level Factor) 的系数轨迹
    coefs = []
    ses = [] # 标准误
    periods = []
    
    # 计算协方差矩阵
    mse = (res.fun ** 2).sum() / (len(res.fun) - n_params)
    try:
        jac = res.jac
        cov = mse * np.linalg.inv(jac.T @ jac)
        stderr = np.sqrt(np.diag(cov))
    except:
        stderr = np.zeros(n_params)
    
    print("\nEvent Study Coefficients (Level Factor b1):")
    print(f"{'Period':>6} | {'Coef':>10} | {'SE':>10} | {'t':>6}")
    print("-" * 40)
    
    event_params = res.x[9:-1]
    event_se = stderr[9:-1]
    
    results_list = []
    
    for i, t in enumerate(model.rel_periods):
        # b1 is at index i*3
        idx = i * 3
        coef = event_params[idx]
        se = event_se[idx]
        t_val = coef / se if se > 0 else 0
        
        coefs.append(coef)
        ses.append(se)
        periods.append(t)
        
        results_list.append({
            'period': t,
            'coef': coef,
            'se': se,
            'ci_lower': coef - 1.96 * se,
            'ci_upper': coef + 1.96 * se
        })
        
        print(f"{t:6d} | {coef:10.4f} | {se:10.4f} | {t_val:6.2f}")

    # 添加基准期 (t=-1, coef=0)
    results_list.append({
        'period': -1, 'coef': 0.0, 'se': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0
    })
    results_df = pd.DataFrame(results_list).sort_values('period')
    
    # 绘图
    plt.figure(figsize=(10, 6))
    plt.errorbar(
        results_df['period'], 
        results_df['coef'], 
        yerr=1.96 * results_df['se'], 
        fmt='o-', 
        capsize=5,
        label='Coefficient (95% CI)'
    )
    plt.axhline(0, color='red', linestyle='--')
    plt.axvline(-0.5, color='gray', linestyle=':')
    plt.title('Parallel Trend Test: Event Study Coefficients (Level Factor)')
    plt.xlabel('Months Relative to Event')
    plt.ylabel('Effect on Yield Level')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    return results_df, plt

def run_placebo_test(
    original_df: pd.DataFrame,
    model_class: type,
    n_permutations: int = 50, # 演示用 50，实际建议 500+
    fixed_L: float = 1.0,
    **kwargs
):
    """
    执行安慰剂检验。
    随机打乱 Treated 分配（保持公司级别一致性），重复估计。
    """
    logging.info(f"Starting Placebo Test with {n_permutations} permutations...")
    
    # 获取唯一的公司列表
    companies = original_df['CompanyCode'].unique()
    n_companies = len(companies)
    # 计算原始 Treated 比例
    # 注意：原始逻辑中 Treated 是由 kcbz 决定的，是公司固有的属性
    # 我们需要知道有多少个 Treated 公司
    treated_companies = original_df[original_df['Treated'] == 1]['CompanyCode'].unique()
    n_treated = len(treated_companies)
    
    did_coefs = []
    
    # 原始数据副本
    df_perm = original_df.copy()
    
    for i in range(n_permutations):
        if i % 10 == 0:
            logging.info(f"Permutation {i}/{n_permutations}")
            
        # 1. 随机抽样 Treated 公司
        np.random.shuffle(companies)
        fake_treated = set(companies[:n_treated])
        
        # 2. 更新 DataFrame 的 Treated 和 DiD
        # 向量化更新
        df_perm['Treated'] = df_perm['CompanyCode'].isin(fake_treated).astype(np.int8)
        df_perm['DiD'] = (df_perm['Treated'] * df_perm['Post']).astype(np.int8)
        
        # 3. 运行模型
        # 为加速，固定 L，且减少 FE 迭代上限
        model = model_class(
            df_perm, 
            max_fe_iters=20, # 略微降低精度以加速
            fe_tol=1e-6
        )
        
        # 构造参数结构，注意 PARAM_ORDER 顺序
        # b1Ta, b2Ta, b3Ta 是我们需要关注的 DiD 系数
        # 它们的索引在 PARAM_ORDER 中通常是 9, 10, 11
        
        # 初始化 guess
        initial_guess = np.zeros(13)
        initial_guess[-1] = fixed_L
        
        # 锁定 L
        lower = [-np.inf] * 12 + [fixed_L - 1e-8]
        upper = [np.inf] * 12 + [fixed_L + 1e-8]
        
        try:
            res = least_squares(
                model.residuals,
                initial_guess,
                bounds=(lower, upper),
                max_nfev=100  # 限制步数
            )
            # 提取 b1Ta (DiD on Level) 系数，索引 9
            # PARAM_ORDER: ..., b1Ta, b2Ta, b3Ta, L
            did_coefs.append(res.x[9]) 
        except Exception as e:
            logging.warning(f"Permutation {i} failed: {e}")
            
    # 绘图
    did_coefs = np.array(did_coefs)
    plt.figure(figsize=(8, 6))
    sns.histplot(did_coefs, kde=True, color='blue', label='Placebo Estimates')
    # 假设真实系数 known_true_coef 传入或之后添加
    plt.title('Placebo Test: Distribution of DiD Coefficients')
    plt.xlabel('Estimated DiD Coefficient (Level)')
    plt.ylabel('Frequency')
    
    return did_coefs, plt

