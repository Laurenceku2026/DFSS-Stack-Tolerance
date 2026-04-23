# ==================== 子应用认证与语言设置 ====================
import streamlit as st
from urllib.parse import parse_qs

# 获取 URL 参数
query_params = st.query_params

# 检查用户登录状态
if "user_id" in query_params:
    st.session_state.user_id = query_params["user_id"]
    st.session_state.user_email = query_params.get("email", [""])[0]
    
    # 从邮箱中提取用户名（@前面的完整部分）
    if st.session_state.user_email:
        # 修复：取 @ 前面的全部字符，不只是第一个
        username = st.session_state.user_email.split('@')[0]
        st.session_state.username = username
    else:
        st.session_state.username = "User"
else:
    st.warning("请从 TechLife Portal 登录后访问")
    st.stop()

# 设置语言
if "lang" in query_params:
    lang_param = query_params["lang"]
    if lang_param == "zh":
        st.session_state.lang = "zh"
    elif lang_param == "en":
        st.session_state.lang = "en"
else:
    st.session_state.lang = "zh"

# 根据语言显示用户标签
user_label = "用户" if st.session_state.lang == "zh" else "User"
lang_label = "语言" if st.session_state.lang == "zh" else "Language"
lang_value = "中文" if st.session_state.lang == "zh" else "English"

# 显示用户信息（侧边栏）
with st.sidebar:
    st.success(f"👤 {user_label}: {st.session_state.username}")
    # 可选：显示语言（如果需要的话）
    # st.info(f"🌐 {lang_label}: {lang_value}")

# ==========原有的工具代码 ==========
# ... 原有逻辑

# app.py - 免费 SaaS 版（无授权码限制）
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math
import re
import base64
from io import BytesIO
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime, timedelta
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml.ns import qn
import json
import os

st.set_page_config(page_title="Para_Variation - 蒙特卡洛模拟", layout="wide")

# ==================== 多语言文本字典 ====================
TEXTS = {
    "zh": {
        "title": "📊 Para_Variation - 基于蒙特卡洛模拟分析",
        "subtitle": "根据输入参数的分布进行随机抽样，计算用户定义的公式结果，分析输出分布及各参数贡献度。",
        "sim_settings": "⚙️ 模拟设置",
        "trail_number": "模拟次数 (Trail number)",
        "spec_limits": "规格限（可留空）",
        "usl": "规格上限 (USL)",
        "lsl": "规格下限 (LSL)",
        "random_seed": "随机种子",
        "about_system": "关于分析系统",
        "about_desc1": "**设计变量**：根据输入参数的公式计算得出。",
        "about_desc2": "**参数抽样**：每个参数独立根据其概率密度函数（PDF）进行随机抽样。",
        "output_title": "**输出结果：**",
        "output1": "- 预测设计变量在量产阶段的分布形态、均值及失效率。",
        "output2": "- 量化各设计参数对输出变量的影响百分比。",
        "output3": "- 通过调节规格上下限，快速确定合理的失效率目标，从而定义合理的工程规格。",
        "analyst_info": "分析人信息",
        "analyst_name": "分析人姓名",
        "analyst_title": "分析人头衔（可选）",
        "contact": "联系：",
        "email": "电邮: Techlife2027@gmail.com",
        "param_input": "📝 参数输入",
        "letter": "字母",
        "param_name": "参数名称",
        "mean": "均值(Typ)",
        "std": "标准差(Std)",
        "distribution": "分布",
        "delete": "删除",
        "add_row": "➕ 添加参数行",
        "new_param_default": "新参数",
        "configure": "⚙️ 配置 {} 参数",
        "dist_full": "正态分布（完整）",
        "dist_pos": "正态分布（正值）",
        "dist_neg": "正态分布（负值）",
        "dist_uniform": "均匀分布",
        "dist_lognorm": "对数正态分布",
        "dist_weibull": "威布尔分布",
        "dist_tri": "三角分布",
        "uniform_low": "下限",
        "uniform_high": "上限",
        "lognorm_meanlog": "对数均值 (μ_log)",
        "lognorm_sigmalog": "对数标准差 (σ_log)",
        "weibull_shape": "形状参数 (k)",
        "weibull_scale": "尺度参数 (λ)",
        "tri_left": "最小值",
        "tri_mode": "最可能值",
        "tri_right": "最大值",
        "error_low_high": "下限必须小于上限",
        "error_sigma": "对数标准差必须大于0",
        "error_weibull": "形状和尺度参数必须 > 0",
        "error_tri": "必须满足：最小值 ≤ 最可能值 ≤ 最大值",
        "formula_def": "📐 公式定义（设计值）",
        "design_var_name": "📌 设计变量名称",
        "formula_label": "📝 计算公式",
        "formula_hint": "💡 可直接在公式中使用字母（A, B, C...）代表对应参数，系统将自动识别。例如：A*E*7/1000*60/(B+C+D)",
        "formula_supported": "支持的运算: + - * / **, 括号, 函数: sqrt, exp, log, sin, cos, tan, pi, e 等。公式中的空格会被自动优化。",
        "design_value": "📌 当前设计值（基于均值）:",
        "formula_invalid": "公式无效或参数不匹配，无法计算设计值。请检查公式中的字母是否与上方对应关系一致，并确保运算正确。",
        "start_sim": "开始\n蒙特卡洛模拟",
        "sim_result": "📈 模拟结果: {}",
        "mean_val": "{} 均值",
        "std_val": "{} 标准差",
        "max_val": "最大值",
        "min_val": "最小值",
        "cpk_val": "Cpk",
        "failure_ppm": "失效率 - ppm level",
        "ppm_hint": "💡 可调节上下限以实时观察PPM水平的变化（留空表示无此限）",
        "no_limits": "未提供任何规格限，无法计算CPK和PPM。",
        "histogram": "{} 分布直方图",
        "hist_caption": "横轴：{}   |   纵轴：频次",
        "effect_chart": "{} 设计参数影响百分比",
        "effect_caption": "横轴：影响百分比   |   纵轴：设计参数",
        "view_contrib": "查看贡献百分比数据表",
        "view_data": "查看全部模拟数据",
        "download_csv": "📥 下载模拟数据 (CSV)",
        "download_report": "📄 下载专业报告 (Word)",
        "success": "模拟完成！",
        "report_title": "{} - DFSS模拟分析报告",
        "analyst_info_report": "分析人信息",
        "analyst_name_report": "分析人姓名：",
        "analyst_title_report": "头衔：",
        "not_filled": "未填写",
        "sim_settings_report": "1. 模拟设置",
        "output_var": "输出变量名称：",
        "formula_report": "公式：",
        "sim_times": "模拟次数：",
        "random_seed_report": "随机种子：",
        "usl_report": "规格上限 (USL)：",
        "lsl_report": "规格下限 (LSL)：",
        "none": "无",
        "param_table": "2. 输入参数表",
        "result_stats": "3. {}模拟结果统计",
        "statistic": "统计量",
        "value": "数值",
        "mean_stat": "均值",
        "std_stat": "标准差",
        "max_stat": "最大值",
        "min_stat": "最小值",
        "cpk_stat": "Cpk",
        "fail_all": "Failure All (ppm)",
        "fail_up": "Failure Up (ppm)",
        "fail_dn": "Failure Dn (ppm)",
        "histogram_report": "4. 分布直方图",
        "effect_report": "5. 设计参数对 {} 影响百分比",
        "detail_table": "详细数据表",
        "param": "参数",
        "contribution": "贡献百分比",
        "contact_report": "联系电邮：Techlife2027@gmail.com",
        "report_date": "报告生成时间：{}",
        "confirm_button": "确定",
        "param_missing_for_letter": "公式中的字母 '{}' 对应的参数（行号 {}）缺少有效的参数名称、均值或标准差，请填写完整。",
        "letter_not_found": "公式中使用了未定义的字母 '{}'，请检查参数表格或修改公式。",
        "no_valid_params": "没有有效的参数（公式中的字母未在参数表中找到）。",
    },
    "en": {
        "title": "📊 Para_Variation - Monte Carlo Simulation",
        "subtitle": "Randomly sample input parameters based on their distributions, compute user-defined formula, analyze output distribution and parameter contribution.",
        "sim_settings": "⚙️ Simulation Settings",
        "trail_number": "Trail number",
        "spec_limits": "Specification limits (leave blank if none)",
        "usl": "Upper Spec Limit (USL)",
        "lsl": "Lower Spec Limit (LSL)",
        "random_seed": "Random seed",
        "about_system": "About Analysis System",
        "about_desc1": "**Design Variable**: Calculated from input parameters using the formula.",
        "about_desc2": "**Parameter Sampling**: Each parameter is independently sampled based on its Probability Density Function (PDF).",
        "output_title": "**Outputs:**",
        "output1": "- Predict the distribution shape, mean, and failure rate of the design variable in mass production.",
        "output2": "- Quantify the percentage impact of each design parameter on the output variable.",
        "output3": "- Adjust spec limits to quickly determine reasonable failure rate targets and define engineering specifications.",
        "analyst_info": "Analyst Information",
        "analyst_name": "Analyst Name",
        "analyst_title": "Analyst Title (optional)",
        "contact": "Contact:",
        "email": "Email: Techlife2027@gmail.com",
        "param_input": "📝 Parameter Input",
        "letter": "Letter",
        "param_name": "Parameter Name",
        "mean": "Mean (Typ)",
        "std": "Std Dev (Std)",
        "distribution": "Distribution",
        "delete": "Delete",
        "add_row": "➕ Add Parameter Row",
        "new_param_default": "New Parameter",
        "configure": "⚙️ Configure {} Parameters",
        "dist_full": "Normal (Full)",
        "dist_pos": "Normal (Positive only)",
        "dist_neg": "Normal (Negative only)",
        "dist_uniform": "Uniform",
        "dist_lognorm": "Log-normal",
        "dist_weibull": "Weibull",
        "dist_tri": "Triangular",
        "uniform_low": "Lower bound",
        "uniform_high": "Upper bound",
        "lognorm_meanlog": "Log mean (μ_log)",
        "lognorm_sigmalog": "Log std dev (σ_log)",
        "weibull_shape": "Shape (k)",
        "weibull_scale": "Scale (λ)",
        "tri_left": "Minimum",
        "tri_mode": "Most likely",
        "tri_right": "Maximum",
        "error_low_high": "Lower bound must be less than upper bound",
        "error_sigma": "Log standard deviation must be > 0",
        "error_weibull": "Shape and scale must be > 0",
        "error_tri": "Must satisfy: min ≤ mode ≤ max",
        "formula_def": "📐 Formula Definition (Design Value)",
        "design_var_name": "📌 Design Variable Name",
        "formula_label": "📝 Formula",
        "formula_hint": "💡 Use letters (A, B, C...) in the formula to represent parameters. System will automatically recognize. Example: A*E*7/1000*60/(B+C+D)",
        "formula_supported": "Supported operators: + - * / **, parentheses, functions: sqrt, exp, log, sin, cos, tan, pi, e, etc. Spaces are automatically optimized.",
        "design_value": "📌 Current Design Value (based on means):",
        "formula_invalid": "Invalid formula or parameter mismatch. Cannot compute design value. Please check if letters match the table above.",
        "start_sim": "Start\nMonte Carlo",
        "sim_result": "📈 Simulation Results: {}",
        "mean_val": "{} Mean",
        "std_val": "{} Std Dev",
        "max_val": "Max",
        "min_val": "Min",
        "cpk_val": "Cpk",
        "failure_ppm": "Failure ppm level",
        "ppm_hint": "💡 Adjust limits to see PPM levels in real time (leave blank if no limit).",
        "no_limits": "No specification limits provided. Cannot compute CPK and PPM.",
        "histogram": "{} Distribution Histogram",
        "hist_caption": "X-axis: {}   |   Y-axis: Frequency",
        "effect_chart": "Parameter Contribution to {}",
        "effect_caption": "X-axis: Contribution %   |   Y-axis: Parameters",
        "view_contrib": "View Contribution Data Table",
        "view_data": "View All Simulation Data",
        "download_csv": "📥 Download Simulation Data (CSV)",
        "download_report": "📄 Download Professional Report (Word)",
        "success": "Simulation completed!",
        "report_title": "{} - DFSS Simulation Report",
        "analyst_info_report": "Analyst Information",
        "analyst_name_report": "Analyst Name:",
        "analyst_title_report": "Title:",
        "not_filled": "Not filled",
        "sim_settings_report": "1. Simulation Settings",
        "output_var": "Output Variable Name:",
        "formula_report": "Formula:",
        "sim_times": "Number of simulations:",
        "random_seed_report": "Random seed:",
        "usl_report": "Upper Spec Limit (USL):",
        "lsl_report": "Lower Spec Limit (LSL):",
        "none": "None",
        "param_table": "2. Input Parameter Table",
        "result_stats": "3. {} Simulation Result Statistics",
        "statistic": "Statistic",
        "value": "Value",
        "mean_stat": "Mean",
        "std_stat": "Std Dev",
        "max_stat": "Max",
        "min_stat": "Min",
        "cpk_stat": "Cpk",
        "fail_all": "Failure All (ppm)",
        "fail_up": "Failure Up (ppm)",
        "fail_dn": "Failure Dn (ppm)",
        "histogram_report": "4. Distribution Histogram",
        "effect_report": "5. Parameter Contribution to {}",
        "detail_table": "Detail Data Table",
        "param": "Parameter",
        "contribution": "Contribution %",
        "contact_report": "Contact Email: Techlife2027@gmail.com",
        "report_date": "Report Date: {}",
        "confirm_button": "OK",
        "param_missing_for_letter": "Parameter for letter '{}' (row {}) is missing valid name, mean, or std. Please complete it.",
        "letter_not_found": "Undefined letter '{}' used in formula. Please check parameter table or modify formula.",
        "no_valid_params": "No valid parameters (letters in formula not found in parameter table).",
    }
}

# 分布中英文映射
DIST_TRANSLATION = {
    "正态分布（完整）": "Normal (Full)",
    "正态分布（正值）": "Normal (Positive only)",
    "正态分布（负值）": "Normal (Negative only)",
    "均匀分布": "Uniform",
    "对数正态分布": "Log-normal",
    "威布尔分布": "Weibull",
    "三角分布": "Triangular",
}
DIST_TRANSLATION_REVERSE = {v: k for k, v in DIST_TRANSLATION.items()}

# ==================== 初始化 Session State ====================
if "lang" not in st.session_state:
    st.session_state.lang = "zh"
if "analyst_name" not in st.session_state:
    st.session_state.analyst_name = ""
if "analyst_title" not in st.session_state:
    st.session_state.analyst_title = ""
if "sim_results_raw" not in st.session_state:
    st.session_state.sim_results_raw = None
if "formula" not in st.session_state:
    st.session_state.formula = "A * E * 7 / 1000 * 60 / (B + C + D)"
if "output_name" not in st.session_state:
    st.session_state.output_name = "Runtime"
if "usl_str" not in st.session_state:
    st.session_state.usl_str = "40.0"
if "lsl_str" not in st.session_state:
    st.session_state.lsl_str = "30.0"
if "params" not in st.session_state:
    st.session_state.params = pd.DataFrame({
        "参数名称": ["Cell Cap", "Suction P", "Brush P", "Other(Pump+display)", "V"],
        "均值(Typ)": [2450.0, 70.0, 30.0, 15.0, 3.6],
        "标准差(Std)": [20.74, 0.77, 0.90, 0.45, 0.0036],
        "分布": ["正态分布（完整）" for _ in range(5)],
        "分布参数": [{} for _ in range(5)]
    })

# ==================== 辅助函数 ====================
def t(key):
    return TEXTS[st.session_state.lang].get(key, key)

def update_param_letters():
    letters = [chr(ord('A') + i) for i in range(len(st.session_state.params))]
    st.session_state.param_letters = {
        row["参数名称"]: letters[i] for i, row in st.session_state.params.iterrows()
    }
update_param_letters()

def update_default_param_names_for_lang():
    """当语言切换时，将参数名称中的默认值（'新参数' 或 'New Parameter'）统一更新为当前语言的默认名称"""
    target_name = "新参数" if st.session_state.lang == "zh" else "New Parameter"
    for idx, row in st.session_state.params.iterrows():
        current_name = row["参数名称"]
        if current_name == "新参数" or current_name == "New Parameter":
            st.session_state.params.at[idx, "参数名称"] = target_name
            key = f"param_name_{idx}"
            if key in st.session_state:
                st.session_state[key] = target_name
    update_param_letters()

def update_dist_display_for_lang():
    """当语言切换时，将分布下拉框的 session_state 值更新为当前语言对应的显示值"""
    for idx, row in st.session_state.params.iterrows():
        stored_dist = row["分布"]
        if st.session_state.lang == "zh":
            display_dist = stored_dist
        else:
            display_dist = DIST_TRANSLATION.get(stored_dist, stored_dist)
        key = f"param_dist_{idx}"
        if key in st.session_state:
            st.session_state[key] = display_dist

# ==================== 蒙特卡洛模拟核心函数 ====================
def parse_limit(s: str) -> Optional[float]:
    if s is None or s.strip() == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None

def sync_usl_from_main(): st.session_state.usl_str = st.session_state.main_usl
def sync_lsl_from_main(): st.session_state.lsl_str = st.session_state.main_lsl
def sync_usl_from_sidebar(): st.session_state.usl_str = st.session_state.usl_sidebar
def sync_lsl_from_sidebar(): st.session_state.lsl_str = st.session_state.lsl_sidebar

def clean_formula(formula: str) -> str:
    formula = formula.strip()
    formula = re.sub(r'\s+', ' ', formula)
    formula = re.sub(r'(?<=[0-9a-zA-Z)])\s*([+\-*/])\s*(?=[0-9a-zA-Z(])', r' \1 ', formula)
    formula = re.sub(r'\(\s+', '(', formula)
    formula = re.sub(r'\s+\)', ')', formula)
    return formula

def replace_letters_with_names(expr: str, param_letters: Dict[str, str]) -> str:
    reverse_map = {v: k for k, v in param_letters.items()}
    for letter, name in sorted(reverse_map.items(), key=lambda x: len(x[0]), reverse=True):
        pattern = r'(?<![a-zA-Z0-9_])' + re.escape(letter) + r'(?![a-zA-Z0-9_])'
        expr = re.sub(pattern, name, expr)
    return expr

def safe_eval_with_mapping(expr: str, param_names: List[str], context_values: List[float], param_letters: Dict[str, str]) -> float:
    expr = clean_formula(expr)
    expr_with_names = replace_letters_with_names(expr, param_letters)
    temp_names = [f"__p{i}__" for i in range(len(param_names))]
    sorted_params = sorted(zip(param_names, temp_names), key=lambda x: len(x[0]), reverse=True)
    expr_temp = expr_with_names
    for orig, temp in sorted_params:
        pattern = r'(?<![a-zA-Z0-9_])' + re.escape(orig) + r'(?![a-zA-Z0-9_])'
        expr_temp = re.sub(pattern, temp, expr_temp)
    context = {temp: val for temp, val in zip(temp_names, context_values)}
    allowed_names = {
        "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan, "pi": math.pi, "e": math.e,
        "abs": abs, "pow": pow
    }
    allowed_names.update(context)
    try:
        result = eval(expr_temp, {"__builtins__": {}}, allowed_names)
        return float(result)
    except Exception:
        return np.nan

def compute_design_value(params_df: pd.DataFrame, formula: str, param_letters: Dict[str, str]) -> Optional[float]:
    param_names = params_df["参数名称"].astype(str).tolist()
    means = params_df["均值(Typ)"].values.astype(float)
    val = safe_eval_with_mapping(formula, param_names, means, param_letters)
    return val if not np.isnan(val) else None

def get_distributions():
    if st.session_state.lang == "zh":
        return [
            t("dist_full"),
            t("dist_pos"),
            t("dist_neg"),
            t("dist_uniform"),
            t("dist_lognorm"),
            t("dist_weibull"),
            t("dist_tri")
        ]
    else:
        return [
            "Normal (Full)",
            "Normal (Positive only)",
            "Normal (Negative only)",
            "Uniform",
            "Log-normal",
            "Weibull",
            "Triangular"
        ]

def generate_sample(dist: str, mean: float, std: float, dist_params: Dict, size: int = 1) -> np.ndarray:
    if "正态分布（完整）" in dist or "Normal (Full)" in dist:
        return np.random.normal(mean, std, size)
    elif "正态分布（正值）" in dist or "Normal (Positive only)" in dist:
        a, b = (0 - mean) / std if std > 0 else -np.inf, np.inf
        if std == 0:
            return np.full(size, max(mean, 0))
        return stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
    elif "正态分布（负值）" in dist or "Normal (Negative only)" in dist:
        a, b = -np.inf, (0 - mean) / std if std > 0 else np.inf
        if std == 0:
            return np.full(size, min(mean, 0))
        return stats.truncnorm.rvs(a, b, loc=mean, scale=std, size=size)
    elif "均匀分布" in dist or "Uniform" in dist:
        low = dist_params.get("low", mean - 3*std)
        high = dist_params.get("high", mean + 3*std)
        return np.random.uniform(low, high, size)
    elif "对数正态分布" in dist or "Log-normal" in dist:
        mean_log = dist_params.get("mean_log", 0.0)
        sigma_log = dist_params.get("sigma_log", 1.0)
        return np.random.lognormal(mean_log, sigma_log, size)
    elif "威布尔分布" in dist or "Weibull" in dist:
        shape = dist_params.get("shape", 1.0)
        scale = dist_params.get("scale", 1.0)
        return np.random.weibull(shape, size) * scale
    elif "三角分布" in dist or "Triangular" in dist:
        left = dist_params.get("left", mean - 3*std)
        mode = dist_params.get("mode", mean)
        right = dist_params.get("right", mean + 3*std)
        return np.random.triangular(left, mode, right, size)
    else:
        return np.random.normal(mean, std, size)

def run_monte_carlo(params_df: pd.DataFrame, formula: str, n_sim: int, param_letters: Dict[str, str], seed: int = 38) -> Dict[str, Any]:
    np.random.seed(seed)
    n_params = len(params_df)
    param_names = params_df["参数名称"].astype(str).tolist()
    means = params_df["均值(Typ)"].values.astype(float)
    stds = params_df["标准差(Std)"].values.astype(float)
    dists = params_df["分布"].tolist()
    dist_params_list = params_df["分布参数"].tolist()

    samples = np.zeros((n_sim, n_params))
    for i in range(n_params):
        samples[:, i] = generate_sample(dists[i], means[i], stds[i], dist_params_list[i], n_sim)

    results = []
    for i in range(n_sim):
        val = safe_eval_with_mapping(formula, param_names, samples[i, :], param_letters)
        if not np.isnan(val):
            results.append(val)

    results = np.array(results)
    if len(results) == 0:
        st.error("所有公式计算均失败，请检查公式！")
        return None

    mean_out = np.mean(results)
    std_out = np.std(results, ddof=1)
    max_out = np.max(results)
    min_out = np.min(results)

    hist_counts, bin_edges = np.histogram(results, bins=25, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    x_pdf = np.linspace(min_out, max_out, 200)
    pdf_theory = stats.norm.pdf(x_pdf, mean_out, std_out)

    return {
        "results": results,
        "samples": samples,
        "mean": mean_out,
        "std": std_out,
        "max": max_out,
        "min": min_out,
        "hist_counts": hist_counts,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "x_pdf": x_pdf,
        "pdf_theory": pdf_theory,
        "param_names": param_names,
    }

def sensitivity_analysis(params_df: pd.DataFrame, formula: str, n_sim: int, param_letters: Dict[str, str], seed: int = 38) -> Tuple[pd.DataFrame, List[float], List[str]]:
    np.random.seed(seed)
    n_params = len(params_df)
    param_names = params_df["参数名称"].astype(str).tolist()
    means = params_df["均值(Typ)"].values.astype(float)
    stds = params_df["标准差(Std)"].values.astype(float)
    dists = params_df["分布"].tolist()
    dist_params_list = params_df["分布参数"].tolist()

    variances = []
    for i in range(n_params):
        samples_i = generate_sample(dists[i], means[i], stds[i], dist_params_list[i], n_sim)
        results_i = []
        for val in samples_i:
            context_vals = means.copy()
            context_vals[i] = val
            res = safe_eval_with_mapping(formula, param_names, context_vals, param_letters)
            if not np.isnan(res):
                results_i.append(res)
        var_i = np.var(results_i, ddof=1) if len(results_i) > 1 else 0.0
        variances.append(var_i)

    total_var = sum(variances)
    contributions = [v / total_var if total_var > 0 else 0.0 for v in variances]

    df_contrib = pd.DataFrame({
        "参数": param_names,
        "方差贡献": variances,
        "贡献百分比": contributions
    })
    df_contrib = df_contrib.sort_values("贡献百分比", ascending=False).reset_index(drop=True)
    df_contrib["贡献百分比_显示"] = df_contrib["贡献百分比"].apply(lambda x: f"{x:.1%}")
    return df_contrib, contributions, param_names

def plot_pdf(dist: str, mean: float, std: float, dist_params: Dict, ax):
    if "正态分布（完整）" in dist or "Normal (Full)" in dist:
        x = np.linspace(mean - 4*std, mean + 4*std, 200)
        y = stats.norm.pdf(x, mean, std)
        ax.plot(x, y, 'b-')
        ax.fill_between(x, y, alpha=0.3)
        ax.set_title(f"N(μ={mean:.1f}, σ={std:.2f})", fontsize=8)
    elif "正态分布（正值）" in dist or "Normal (Positive only)" in dist:
        a, b = (0 - mean) / std if std > 0 else -np.inf, np.inf
        if std == 0:
            x = [max(mean, 0)]
            y = [1]
        else:
            x = np.linspace(0, mean + 4*std, 200)
            y = stats.truncnorm.pdf(x, a, b, loc=mean, scale=std)
        ax.plot(x, y, 'g-')
        ax.fill_between(x, y, alpha=0.3)
        ax.set_title(f"TruncNorm(≥0)", fontsize=8)
    elif "正态分布（负值）" in dist or "Normal (Negative only)" in dist:
        a, b = -np.inf, (0 - mean) / std if std > 0 else np.inf
        if std == 0:
            x = [min(mean, 0)]
            y = [1]
        else:
            x = np.linspace(mean - 4*std, 0, 200)
            y = stats.truncnorm.pdf(x, a, b, loc=mean, scale=std)
        ax.plot(x, y, 'r-')
        ax.fill_between(x, y, alpha=0.3)
        ax.set_title(f"TruncNorm(≤0)", fontsize=8)
    elif "均匀分布" in dist or "Uniform" in dist:
        low = dist_params.get("low", mean - 3*std)
        high = dist_params.get("high", mean + 3*std)
        x = np.linspace(low, high, 200)
        y = stats.uniform.pdf(x, low, high-low)
        ax.plot(x, y, 'purple')
        ax.fill_between(x, y, alpha=0.3)
        ax.set_title(f"U({low:.1f}, {high:.1f})", fontsize=8)
    elif "对数正态分布" in dist or "Log-normal" in dist:
        mean_log = dist_params.get("mean_log", 0.0)
        sigma_log = dist_params.get("sigma_log", 1.0)
        x = np.linspace(0, np.exp(mean_log + 3*sigma_log), 200)
        y = stats.lognorm.pdf(x, sigma_log, scale=np.exp(mean_log))
        ax.plot(x, y, 'orange')
        ax.fill_between(x, y, alpha=0.3)
        ax.set_title(f"LogN(μlog={mean_log:.1f}, σlog={sigma_log:.2f})", fontsize=8)
    elif "威布尔分布" in dist or "Weibull" in dist:
        shape = dist_params.get("shape", 1.0)
        scale = dist_params.get("scale", 1.0)
        x = np.linspace(0, scale * 3, 200)
        y = stats.weibull_min.pdf(x, shape, scale=scale)
        ax.plot(x, y, 'brown')
        ax.fill_between(x, y, alpha=0.3)
        ax.set_title(f"Weibull(k={shape:.1f}, λ={scale:.1f})", fontsize=8)
    elif "三角分布" in dist or "Triangular" in dist:
        left = dist_params.get("left", mean - 3*std)
        mode = dist_params.get("mode", mean)
        right = dist_params.get("right", mean + 3*std)
        x = np.linspace(left, right, 200)
        y = stats.triang.pdf(x, (mode-left)/(right-left), loc=left, scale=right-left)
        ax.plot(x, y, 'olive')
        ax.fill_between(x, y, alpha=0.3)
        ax.set_title(f"Tri({left:.1f}, {mode:.1f}, {right:.1f})", fontsize=8)
    ax.set_xlabel("Value", fontsize=6)
    ax.set_ylabel("Density", fontsize=6)
    ax.tick_params(axis='both', labelsize=6)

def plot_histogram(results, bin_centers, hist_counts, x_pdf, pdf_theory, usl, lsl, output_name, n_sim):
    fig, ax = plt.subplots(figsize=(11, 6), dpi=100)
    ax.bar(bin_centers, hist_counts, width=(bin_centers[1]-bin_centers[0])*0.9, alpha=0.6, label="Histogram", color="#6c757d")
    area = np.sum(hist_counts) * (bin_centers[1]-bin_centers[0])
    ax.plot(x_pdf, pdf_theory * area, 'r-', linewidth=2, label="Gaussian Fitting")
    if usl is not None:
        ax.axvline(usl, color='green', linestyle='--', linewidth=1.5, label=f"USL = {usl:.2f}")
    if lsl is not None:
        ax.axvline(lsl, color='orange', linestyle='--', linewidth=1.5, label=f"LSL = {lsl:.2f}")
    stats_text = f"NO.={n_sim}\nAVE={np.mean(results):.2f}\nSTD={np.std(results, ddof=1):.4f}\nMAX={np.max(results):.2f}\nMIN={np.min(results):.2f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8, edgecolor='gray'))
    ax.legend(loc='upper right', bbox_to_anchor=(0.95, 0.72), fontsize=9)
    ax.set_xlabel(output_name, fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    ax.set_title(f"{output_name} Distribution", fontsize=13, fontweight='bold')
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    return fig

def plot_contribution_horizontal(contributions: List[float], param_names: List[str], output_name: str):
    non_zero = [(p, c) for p, c in zip(param_names, contributions) if c > 0]
    if not non_zero:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No significant contribution", ha='center', va='center')
        ax.set_title(f"Design Factor Effect % on {output_name}", fontsize=13, fontweight='bold')
        return fig
    names, vals = zip(*non_zero)
    sorted_indices = np.argsort(vals)
    names = [names[i] for i in sorted_indices]
    vals = [vals[i] for i in sorted_indices]
    fig, ax = plt.subplots(figsize=(9, max(4, len(names)*0.4)))
    bars = ax.barh(names, vals, color='#6c757d')
    for bar, val in zip(bars, vals):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.1%}', va='center', fontsize=9)
    ax.set_xlabel("Effect %", fontsize=11)
    ax.set_title(f"Design Factor Effect % on {output_name}", fontsize=13, fontweight='bold')
    ax.set_xlim(0, max(vals) * 1.15)
    ax.grid(axis='x', linestyle='--', alpha=0.5)
    ax.legend().remove()
    return fig

def compute_cpk_ppm(results: np.ndarray, usl: Optional[float], lsl: Optional[float]):
    mean_out = np.mean(results)
    std_out = np.std(results, ddof=1)
    if std_out == 0:
        cpk = None
    else:
        if usl is not None and lsl is not None:
            cpk = min((usl - mean_out) / (3 * std_out), (mean_out - lsl) / (3 * std_out))
        elif usl is not None:
            cpk = (usl - mean_out) / (3 * std_out)
        elif lsl is not None:
            cpk = (mean_out - lsl) / (3 * std_out)
        else:
            cpk = None
    failures_up = np.sum(results > usl) / len(results) * 1e6 if usl is not None else None
    failures_dn = np.sum(results < lsl) / len(results) * 1e6 if lsl is not None else None
    failures_all = None
    if failures_up is not None and failures_dn is not None:
        failures_all = failures_up + failures_dn
    elif failures_up is not None:
        failures_all = failures_up
    elif failures_dn is not None:
        failures_all = failures_dn
    return cpk, failures_all, failures_up, failures_dn

def generate_word_report(raw, usl, lsl, n_sim, seed, formula, params_df, param_letters, analyst_name, analyst_title, output_name):
    results = raw["results"]
    cpk, failures_all, failures_up, failures_dn = compute_cpk_ppm(results, usl, lsl)

    doc = Document()
    title = doc.add_heading(t("report_title").format(output_name), 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    doc.add_heading(t("analyst_info_report"), level=1)
    p = doc.add_paragraph()
    p.add_run(f"{t('analyst_name_report')} {analyst_name if analyst_name else t('not_filled')}").bold = True
    p.add_run(f"\n{t('analyst_title_report')} {analyst_title if analyst_title else t('not_filled')}")
    doc.add_heading(t("sim_settings_report"), level=1)
    doc.add_paragraph(f"{t('output_var')} {output_name}")
    doc.add_paragraph(f"{t('formula_report')} {formula}")
    doc.add_paragraph(f"{t('sim_times')} {n_sim}")
    doc.add_paragraph(f"{t('random_seed_report')} {seed}")
    doc.add_paragraph(f"{t('usl_report')} {usl if usl is not None else t('none')}")
    doc.add_paragraph(f"{t('lsl_report')} {lsl if lsl is not None else t('none')}")
    doc.add_heading(t("param_table"), level=1)
    table = doc.add_table(rows=len(params_df)+1, cols=len(params_df.columns))
    table.style = 'Table Grid'
    for j, col in enumerate(params_df.columns):
        table.cell(0, j).text = col
    for i, row in params_df.iterrows():
        for j, col in enumerate(params_df.columns):
            if col == "分布参数":
                table.cell(i+1, j).text = str(row[col]) if row[col] else "{}"
            else:
                table.cell(i+1, j).text = str(row[col])
    doc.add_heading(t("result_stats").format(output_name), level=1)
    stats_table = doc.add_table(rows=5, cols=2)
    stats_table.style = 'Table Grid'
    stats_table.cell(0, 0).text = t("statistic")
    stats_table.cell(0, 1).text = t("value")
    stats_table.cell(1, 0).text = t("mean_stat")
    stats_table.cell(1, 1).text = f"{raw['mean']:.2f}"
    stats_table.cell(2, 0).text = t("std_stat")
    stats_table.cell(2, 1).text = f"{raw['std']:.4f}"
    stats_table.cell(3, 0).text = t("max_stat")
    stats_table.cell(3, 1).text = f"{raw['max']:.2f}"
    stats_table.cell(4, 0).text = t("min_stat")
    stats_table.cell(4, 1).text = f"{raw['min']:.2f}"
    if cpk is not None:
        stats_table.add_row().cells[0].text = t("cpk_stat")
        stats_table.rows[-1].cells[1].text = f"{cpk:.2f}"
        stats_table.add_row().cells[0].text = t("fail_all")
        stats_table.rows[-1].cells[1].text = f"{failures_all:.2f}" if failures_all is not None else "-"
        stats_table.add_row().cells[0].text = t("fail_up")
        stats_table.rows[-1].cells[1].text = f"{failures_up:.2f}" if failures_up is not None else "-"
        stats_table.add_row().cells[0].text = t("fail_dn")
        stats_table.rows[-1].cells[1].text = f"{failures_dn:.2f}" if failures_dn is not None else "-"
    doc.add_heading(t("histogram_report"), level=1)
    fig_hist = plot_histogram(results, raw["bin_centers"], raw["hist_counts"], raw["x_pdf"], raw["pdf_theory"], usl, lsl, output_name, n_sim)
    buf_hist = BytesIO()
    fig_hist.savefig(buf_hist, format='png', dpi=150, bbox_inches='tight')
    buf_hist.seek(0)
    doc.add_picture(buf_hist, width=Inches(6))
    plt.close(fig_hist)
    doc.add_heading(t("effect_report").format(output_name), level=1)
    fig_barh = plot_contribution_horizontal(raw["contributions"], raw["param_names"], output_name)
    buf_barh = BytesIO()
    fig_barh.savefig(buf_barh, format='png', dpi=150, bbox_inches='tight')
    buf_barh.seek(0)
    doc.add_picture(buf_barh, width=Inches(6))
    plt.close(fig_barh)
    doc.add_heading(t("detail_table"), level=2)
    df_contrib = raw["df_contrib"].copy()
    df_contrib["贡献百分比"] = df_contrib["贡献百分比_显示"]
    contrib_table = doc.add_table(rows=len(df_contrib)+1, cols=2)
    contrib_table.style = 'Table Grid'
    contrib_table.cell(0, 0).text = t("param")
    contrib_table.cell(0, 1).text = t("contribution")
    for i, row in df_contrib.iterrows():
        contrib_table.cell(i+1, 0).text = row["参数"]
        contrib_table.cell(i+1, 1).text = row["贡献百分比"]
    doc.add_paragraph()
    doc.add_paragraph()
    doc.add_paragraph()
    footer = doc.add_paragraph()
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run(t("contact_report")).italic = True
    footer.add_run(f"\n{t('report_date').format(datetime.now().strftime('%Y-%m-%d'))}")
    doc_bytes = BytesIO()
    doc.save(doc_bytes)
    doc_bytes.seek(0)
    return doc_bytes

# ==================== 辅助函数：过滤参数表，只保留公式中使用的字母对应的行 ====================
def filter_params_by_formula(params_df: pd.DataFrame, formula: str, param_letters: Dict[str, str]) -> Tuple[pd.DataFrame, Dict[str, str]]:
    letters_in_formula = set(re.findall(r'\b([A-Za-z])\b', formula))
    letters_in_formula = {l.upper() for l in letters_in_formula}
    
    letter_to_param = {letter: param_name for param_name, letter in param_letters.items()}
    keep_param_names = []
    for letter in letters_in_formula:
        if letter in letter_to_param:
            keep_param_names.append(letter_to_param[letter])
    
    filtered_df = params_df[params_df["参数名称"].isin(keep_param_names)].copy()
    new_param_letters = {}
    for idx, row in filtered_df.iterrows():
        param_name = row["参数名称"]
        for letter, pname in letter_to_param.items():
            if pname == param_name:
                new_param_letters[param_name] = letter
                break
    return filtered_df, new_param_letters

# ==================== 主函数 ====================
def main():
    st.markdown("""
    <style>
        html, body, .stApp, .stMarkdown, .stText, .stNumberInput, .stSelectbox, .stTextArea, .stDataFrame, .stMetric {
            color: #000000 !important;
        }
        .main-title { font-size: 2.5rem; font-weight: 600; color: #000000; margin-bottom: 1rem; }
        .section-header { font-size: 1.5rem; font-weight: 500; color: #000000; border-left: 5px solid #cccccc; padding-left: 15px; margin: 20px 0 15px 0; }
        .metric-card { background-color: #f8f9fa; border-radius: 10px; padding: 15px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }
        .metric-label { font-size: 1rem; color: #000000; margin-bottom: 5px; }
        .metric-value { font-size: 1.8rem; font-weight: 600; color: #000000; }
        .ppm-table { border-collapse: collapse; width: 100%; margin: 0 auto; }
        .ppm-table th, .ppm-table td { border: 2px solid #000000; padding: 10px 16px; text-align: center; font-size: 1rem; }
        .ppm-table th { background-color: #e9ecef; font-weight: 600; }
        button[data-testid="baseButton-primary"] {
            background-color: #dc3545 !important;
            color: white !important;
            font-weight: 500;
            border-radius: 5px;
            font-size: 1.2rem;
            margin-top: 20px;
            white-space: pre-line !important;
        }
        button[data-testid="baseButton-primary"]:hover { background-color: #c82333 !important; }
        button[data-testid="baseButton-primary"] * { white-space: pre-line !important; }
        .stButton > button:not([data-testid="baseButton-primary"]) {
            background-color: #3498db !important;
            color: white !important;
            font-weight: 500;
            border-radius: 5px;
        }
        .stButton > button:not([data-testid="baseButton-primary"]):hover { background-color: #2980b9 !important; }
        .lang-btn-wrap .stButton button {
            background-color: #dc3545 !important;
            color: white !important;
            font-weight: 500;
            border-radius: 5px;
        }
        .lang-btn-wrap .stButton button:hover { background-color: #c82333 !important; }
        .design-value-card { background-color: #e8f4fd; border-radius: 10px; padding: 15px; margin-top: 15px; text-align: center; border-left: 5px solid #cccccc; }
        .design-value-card strong { font-size: 1.1rem; color: #000000; }
        .design-value-number { font-size: 1.6rem; font-weight: 600; color: #000000; margin-top: 5px; }
        .big-label { font-size: 1.3rem; font-weight: 500; margin-bottom: 5px; color: #000000; }
        .param-letter { font-weight: bold; font-size: 1rem; text-align: center; background-color: #e9ecef; border-radius: 4px; padding: 6px 0; width: 40px; color: #000000; }
        .formula-hint { font-size: 0.9rem; color: #000000; margin-bottom: 5px; }
    </style>
    """, unsafe_allow_html=True)

    # 右上角语言切换
    col_left, col_spacer, col_zh, col_en = st.columns([0.5, 0.2, 0.15, 0.15])
    with col_zh:
        st.markdown('<div class="lang-btn-wrap">', unsafe_allow_html=True)
        if st.button("中文", key="lang_zh", use_container_width=True):
            st.session_state.lang = "zh"
            update_default_param_names_for_lang()
            update_dist_display_for_lang()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)
    with col_en:
        st.markdown('<div class="lang-btn-wrap">', unsafe_allow_html=True)
        if st.button("English", key="lang_en", use_container_width=True):
            st.session_state.lang = "en"
            update_default_param_names_for_lang()
            update_dist_display_for_lang()
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

    st.markdown(f'<div class="main-title">{t("title")}</div>', unsafe_allow_html=True)
    st.markdown(t("subtitle"))

    # 侧边栏
    with st.sidebar:
        st.markdown(f"## {t('sim_settings')}")
        n_sim = st.number_input(t("trail_number"), min_value=100, max_value=100000, value=1000, step=100)
        st.markdown(f"#### {t('spec_limits')}")
        usl_sidebar = st.text_input(t("usl"), value=st.session_state.usl_str, key="usl_sidebar", on_change=sync_usl_from_sidebar)
        lsl_sidebar = st.text_input(t("lsl"), value=st.session_state.lsl_str, key="lsl_sidebar", on_change=sync_lsl_from_sidebar)
        st.session_state.usl_str = usl_sidebar
        st.session_state.lsl_str = lsl_sidebar
        seed = st.number_input(t("random_seed"), value=38, step=1)

        st.markdown("---")
        st.markdown(f"### {t('about_system')}")
        st.markdown(t("about_desc1"))
        st.markdown(t("about_desc2"))
        st.markdown(t("output_title"))
        st.markdown(t("output1"))
        st.markdown(t("output2"))
        st.markdown(t("output3"))

        st.markdown("---")
        st.markdown(f"### {t('analyst_info')}")
        analyst_name = st.text_input(t("analyst_name"), value=st.session_state.analyst_name, key="analyst_name_input")
        analyst_title = st.text_input(t("analyst_title"), value=st.session_state.analyst_title, key="analyst_title_input")
        st.session_state.analyst_name = analyst_name
        st.session_state.analyst_title = analyst_title

        st.markdown("---")
        st.markdown(f"**{t('contact')}**")
        st.markdown(t("email"))

    # 参数输入表格
    st.markdown(f'<div class="section-header">{t("param_input")}</div>', unsafe_allow_html=True)
    header_cols = st.columns([0.3, 1.5, 1, 1, 1.2, 0.3])
    header_cols[0].markdown(f"**{t('letter')}**")
    header_cols[1].markdown(f"**{t('param_name')}**")
    header_cols[2].markdown(f"**{t('mean')}**")
    header_cols[3].markdown(f"**{t('std')}**")
    header_cols[4].markdown(f"**{t('distribution')}**")
    header_cols[5].markdown(f"**{t('delete')}**")

    rows_data = []
    distributions_list = get_distributions()
    for idx, row in st.session_state.params.iterrows():
        letter = chr(ord('A') + idx)
        cols = st.columns([0.3, 1.5, 1, 1, 1.2, 0.3])
        with cols[0]:
            st.markdown(f'<div class="param-letter">{letter}</div>', unsafe_allow_html=True)
        with cols[1]:
            name = st.text_input("", value=row["参数名称"], key=f"param_name_{idx}", label_visibility="collapsed")
        with cols[2]:
            mean_val = st.number_input("", value=float(row["均值(Typ)"]), step=1.0, key=f"param_mean_{idx}", label_visibility="collapsed")
        with cols[3]:
            std_val = st.number_input("", value=float(row["标准差(Std)"]), step=0.01, format="%.4f", key=f"param_std_{idx}", label_visibility="collapsed")
        with cols[4]:
            stored_dist = row["分布"]
            if st.session_state.lang == "zh":
                display_dist = stored_dist
            else:
                display_dist = DIST_TRANSLATION.get(stored_dist, stored_dist)
            try:
                dist_index = distributions_list.index(display_dist)
            except ValueError:
                dist_index = 0
            dist_val = st.selectbox("", distributions_list, index=dist_index, key=f"param_dist_{idx}", label_visibility="collapsed")
            if st.session_state.lang == "zh":
                stored_dist_val = dist_val
            else:
                stored_dist_val = DIST_TRANSLATION_REVERSE.get(dist_val, dist_val)
        with cols[5]:
            if st.button("🗑️", key=f"del_{idx}"):
                for key in [f"param_name_{idx}", f"param_mean_{idx}", f"param_std_{idx}", f"param_dist_{idx}", f"expander_{idx}"]:
                    if key in st.session_state:
                        del st.session_state[key]
                st.session_state.params.drop(index=idx, inplace=True)
                st.session_state.params.reset_index(drop=True, inplace=True)
                update_param_letters()
                st.rerun()

        current_dist_params = row.get("分布参数", {}) if isinstance(row.get("分布参数"), dict) else {}
        if stored_dist_val != stored_dist:
            if stored_dist_val == "均匀分布":
                current_dist_params = {"low": mean_val - 3 * std_val, "high": mean_val + 3 * std_val}
            elif stored_dist_val == "对数正态分布":
                current_dist_params = {"mean_log": 0.0, "sigma_log": 1.0}
            elif stored_dist_val == "威布尔分布":
                current_dist_params = {"shape": 1.0, "scale": 1.0}
            elif stored_dist_val == "三角分布":
                current_dist_params = {"left": mean_val - 3 * std_val, "mode": mean_val, "right": mean_val + 3 * std_val}
            else:
                current_dist_params = {}

        need_expand = stored_dist_val in ["均匀分布", "对数正态分布", "威布尔分布", "三角分布"]
        # 管理 expander 展开状态，默认展开
        expander_key = f"expander_{idx}"
        if expander_key not in st.session_state:
            st.session_state[expander_key] = True

        if need_expand:
            with st.expander(t("configure").format(dist_val), expanded=st.session_state[expander_key]):
                if stored_dist_val == "均匀分布":
                    low = st.number_input(t("uniform_low"), value=float(current_dist_params.get("low", mean_val - 3*std_val)), key=f"uniform_low_{idx}", step=0.1)
                    high = st.number_input(t("uniform_high"), value=float(current_dist_params.get("high", mean_val + 3*std_val)), key=f"uniform_high_{idx}", step=0.1)
                    if low >= high:
                        st.error(t("error_low_high"))
                    else:
                        current_dist_params["low"] = low
                        current_dist_params["high"] = high
                elif stored_dist_val == "对数正态分布":
                    mean_log = st.number_input(t("lognorm_meanlog"), value=float(current_dist_params.get("mean_log", 0.0)), key=f"lognorm_meanlog_{idx}", step=0.1)
                    sigma_log = st.number_input(t("lognorm_sigmalog"), value=float(current_dist_params.get("sigma_log", 1.0)), key=f"lognorm_sigmalog_{idx}", step=0.05, format="%.3f")
                    if sigma_log <= 0:
                        st.error(t("error_sigma"))
                    else:
                        current_dist_params["mean_log"] = mean_log
                        current_dist_params["sigma_log"] = sigma_log
                elif stored_dist_val == "威布尔分布":
                    shape = st.number_input(t("weibull_shape"), value=float(current_dist_params.get("shape", 1.0)), key=f"weibull_shape_{idx}", step=0.1, min_value=0.1)
                    scale = st.number_input(t("weibull_scale"), value=float(current_dist_params.get("scale", 1.0)), key=f"weibull_scale_{idx}", step=0.1, min_value=0.1)
                    if shape <= 0 or scale <= 0:
                        st.error(t("error_weibull"))
                    else:
                        current_dist_params["shape"] = shape
                        current_dist_params["scale"] = scale
                elif stored_dist_val == "三角分布":
                    left = st.number_input(t("tri_left"), value=float(current_dist_params.get("left", mean_val - 3*std_val)), key=f"tri_left_{idx}", step=0.1)
                    mode = st.number_input(t("tri_mode"), value=float(current_dist_params.get("mode", mean_val)), key=f"tri_mode_{idx}", step=0.1)
                    right = st.number_input(t("tri_right"), value=float(current_dist_params.get("right", mean_val + 3*std_val)), key=f"tri_right_{idx}", step=0.1)
                    if not (left <= mode <= right):
                        st.error(t("error_tri"))
                    else:
                        current_dist_params["left"] = left
                        current_dist_params["mode"] = mode
                        current_dist_params["right"] = right

                fig, ax = plt.subplots(figsize=(4, 2))
                plot_pdf(stored_dist_val, mean_val, std_val, current_dist_params, ax)
                st.pyplot(fig)
                plt.close(fig)

                # 确定按钮，点击后关闭 expander
                if st.button(t("confirm_button"), key=f"confirm_{idx}"):
                    st.session_state[expander_key] = False
                    st.rerun()

        rows_data.append((name, mean_val, std_val, stored_dist_val, current_dist_params, letter))

    new_params = []
    for (name, mean_val, std_val, dist_val, dist_params, letter) in rows_data:
        new_params.append({
            "参数名称": name,
            "均值(Typ)": mean_val,
            "标准差(Std)": std_val,
            "分布": dist_val,
            "分布参数": dist_params
        })
    st.session_state.params = pd.DataFrame(new_params)
    update_param_letters()

    # 添加行按钮
    if st.button(t("add_row"), use_container_width=True):
        default_name = "新参数" if st.session_state.lang == "zh" else "New Parameter"
        new_idx = len(st.session_state.params)
        new_row = pd.DataFrame({
            "参数名称": [default_name],
            "均值(Typ)": [0.0],
            "标准差(Std)": [0.0],
            "分布": ["正态分布（完整）"],
            "分布参数": [{}]
        })
        st.session_state.params = pd.concat([st.session_state.params, new_row], ignore_index=True)
        st.session_state[f"param_name_{new_idx}"] = default_name
        display_dist = "正态分布（完整）" if st.session_state.lang == "zh" else "Normal (Full)"
        st.session_state[f"param_dist_{new_idx}"] = display_dist
        update_param_letters()
        st.rerun()

    # 公式定义区域
    st.markdown(f'<div class="section-header">{t("formula_def")}</div>', unsafe_allow_html=True)
    st.markdown(f'<span class="big-label">{t("design_var_name")}</span>', unsafe_allow_html=True)
    output_name = st.text_input("", value=st.session_state.output_name, key="output_name_input", label_visibility="collapsed")
    st.session_state.output_name = output_name if output_name.strip() else "Output"
    st.markdown(f'<span class="big-label">{t("formula_label")}</span>', unsafe_allow_html=True)
    st.markdown(f'<div class="formula-hint">{t("formula_hint")}</div>', unsafe_allow_html=True)
    formula = st.text_area("", value=st.session_state.formula, height=100, key="formula_input", label_visibility="collapsed")
    st.session_state.formula = formula
    st.caption(t("formula_supported"))

    filtered_params, filtered_letters = filter_params_by_formula(st.session_state.params, formula, st.session_state.param_letters)
    if len(filtered_params) > 0:
        design_val = compute_design_value(filtered_params, formula, filtered_letters)
        if design_val is not None and not np.isnan(design_val):
            st.markdown(f"""
            <div class="design-value-card">
                <strong>{t("design_value")}</strong><br>
                <span class="design-value-number">{output_name} = {design_val:.2f}</span>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.warning(t("formula_invalid"))
    else:
        st.warning(t("formula_invalid"))

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        if st.button(t("start_sim"), type="primary", use_container_width=True):
            letters_in_formula = set(re.findall(r'\b([A-Za-z])\b', formula))
            letters_in_formula = {l.upper() for l in letters_in_formula}
            
            param_letters = st.session_state.param_letters
            letter_to_param = {letter: param_name for param_name, letter in param_letters.items()}
            
            missing_letters = []
            for letter in letters_in_formula:
                if letter not in letter_to_param:
                    missing_letters.append(letter)
            if missing_letters:
                st.error(t("letter_not_found").format(", ".join(missing_letters)))
                st.stop()
            
            invalid_params = []
            for letter in letters_in_formula:
                param_name = letter_to_param[letter]
                row = st.session_state.params[st.session_state.params["参数名称"] == param_name].iloc[0]
                param_name_val = str(row["参数名称"]).strip()
                mean_val = row["均值(Typ)"]
                std_val = row["标准差(Std)"]
                if param_name_val == "" or pd.isna(mean_val) or pd.isna(std_val):
                    idx = st.session_state.params[st.session_state.params["参数名称"] == param_name].index[0]
                    invalid_params.append((letter, idx+1))
            if invalid_params:
                for letter, row_num in invalid_params:
                    st.error(t("param_missing_for_letter").format(letter, row_num))
                st.stop()
            
            if not formula.strip():
                st.error(t("formula_invalid"))
                st.stop()

            filtered_params_for_sim, filtered_letters_for_sim = filter_params_by_formula(
                st.session_state.params, formula, st.session_state.param_letters
            )
            if len(filtered_params_for_sim) == 0:
                st.error(t("no_valid_params"))
                st.stop()

            with st.spinner(t("start_sim")):
                sim_res = run_monte_carlo(filtered_params_for_sim, formula, n_sim, filtered_letters_for_sim, seed)
            if sim_res is None:
                st.stop()

            with st.spinner(t("start_sim")):
                df_contrib, contributions, param_names = sensitivity_analysis(filtered_params_for_sim, formula, n_sim, filtered_letters_for_sim, seed)

            st.session_state.sim_results_raw = {
                "results": sim_res["results"],
                "samples": sim_res["samples"],
                "mean": sim_res["mean"],
                "std": sim_res["std"],
                "max": sim_res["max"],
                "min": sim_res["min"],
                "hist_counts": sim_res["hist_counts"],
                "bin_edges": sim_res["bin_edges"],
                "bin_centers": sim_res["bin_centers"],
                "x_pdf": sim_res["x_pdf"],
                "pdf_theory": sim_res["pdf_theory"],
                "param_names": sim_res["param_names"],
                "df_contrib": df_contrib,
                "contributions": contributions,
                "params_df": filtered_params_for_sim,
                "output_name": output_name,
                "formula": formula,
            }

    if st.session_state.sim_results_raw is not None:
        raw = st.session_state.sim_results_raw
        results = raw["results"]
        output_name = raw["output_name"]
        usl = parse_limit(st.session_state.usl_str)
        lsl = parse_limit(st.session_state.lsl_str)
        cpk, failures_all, failures_up, failures_dn = compute_cpk_ppm(results, usl, lsl)

        st.markdown(f'<div class="section-header">{t("sim_result").format(output_name)}</div>', unsafe_allow_html=True)
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t("mean_val").format(output_name)}</div><div class="metric-value">{raw["mean"]:.2f}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t("std_val").format(output_name)}</div><div class="metric-value">{raw["std"]:.4f}</div></div>', unsafe_allow_html=True)
        with col2:
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t("max_val")}</div><div class="metric-value">{raw["max"]:.2f}</div></div>', unsafe_allow_html=True)
            st.markdown(f'<div class="metric-card"><div class="metric-label">{t("min_val")}</div><div class="metric-value">{raw["min"]:.2f}</div></div>', unsafe_allow_html=True)
        with col3:
            if cpk is not None:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{t("cpk_val")}</div><div class="metric-value">{cpk:.2f}</div></div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="metric-card"><div class="metric-label">{t("cpk_val")}</div><div class="metric-value">-</div></div>', unsafe_allow_html=True)

        st.markdown(f'<div class="section-header">{t("failure_ppm")}</div>', unsafe_allow_html=True)
        st.caption(t("ppm_hint"))
        col_left, col_right = st.columns([1, 2])
        with col_left:
            main_usl = st.text_input(t("usl"), value=st.session_state.usl_str, key="main_usl", on_change=sync_usl_from_main)
            main_lsl = st.text_input(t("lsl"), value=st.session_state.lsl_str, key="main_lsl", on_change=sync_lsl_from_main)
            st.session_state.usl_str = main_usl
            st.session_state.lsl_str = main_lsl
            usl = parse_limit(main_usl)
            lsl = parse_limit(main_lsl)
            cpk, failures_all, failures_up, failures_dn = compute_cpk_ppm(results, usl, lsl)
        with col_right:
            if cpk is not None:
                def fmt(v): return f"{v:.2f}" if v is not None else "-"
                st.markdown(f"""
                <table class="ppm-table">
                    <tr><th>CPK</th><th>Failure All</th><th>Failure Up</th><th>Failure Dn</th></tr>
                    <tr><td style="text-align:center">{fmt(cpk)}</td><td style="text-align:center">{fmt(failures_all)}</td><td style="text-align:center">{fmt(failures_up)}</td><td style="text-align:center">{fmt(failures_dn)}</td></tr>
                </table>
                """, unsafe_allow_html=True)
            else:
                st.info(t("no_limits"))

        st.markdown(f"### {t('histogram').format(output_name)}")
        fig_hist = plot_histogram(results, raw["bin_centers"], raw["hist_counts"], raw["x_pdf"], raw["pdf_theory"], usl, lsl, output_name, n_sim)
        st.pyplot(fig_hist)
        st.caption(t("hist_caption").format(output_name))

        st.markdown(f"### {t('effect_chart').format(output_name)}")
        fig_barh = plot_contribution_horizontal(raw["contributions"], raw["param_names"], output_name)
        st.pyplot(fig_barh)
        st.caption(t("effect_caption"))

        with st.expander(t("view_contrib")):
            st.dataframe(raw["df_contrib"][["参数", "贡献百分比_显示"]].rename(columns={"贡献百分比_显示": "贡献百分比"}), use_container_width=True)

        with st.expander(t("view_data")):
            samples_df = pd.DataFrame(raw["samples"], columns=raw["param_names"])
            samples_df[output_name] = results
            st.dataframe(samples_df.round(2), use_container_width=True, height=400)
            csv = samples_df.to_csv(index=False, float_format="%.6f")
            st.download_button(t("download_csv"), data=csv, file_name=f"monte_carlo_data_{output_name}.csv", mime="text/csv")

        if st.button(t("download_report")):
            doc_bytes = generate_word_report(raw, usl, lsl, n_sim, seed, formula, raw["params_df"], st.session_state.param_letters, st.session_state.analyst_name, st.session_state.analyst_title, output_name)
            date_str = datetime.now().strftime("%Y%m%d")
            st.download_button(t("download_report"), data=doc_bytes, file_name=f"DFSS_Report_{output_name}_{date_str}.docx", mime="application/vnd.openxmlformats-officedocument.wordprocessingml.document")
        st.success(t("success"))

if __name__ == "__main__":
    main()
