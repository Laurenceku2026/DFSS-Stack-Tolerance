# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math
import re
from typing import List, Dict, Any, Tuple

# 页面配置
st.set_page_config(page_title="蒙特卡洛模拟器", layout="wide")
st.title("📊 蒙特卡洛模拟器 (基于正态分布)")
st.markdown("根据输入参数的分布进行随机抽样，计算用户定义的公式结果，分析输出分布及各参数贡献度。")

# 初始化session state
if "params" not in st.session_state:
    st.session_state.params = pd.DataFrame({
        "参数名称": ["Cell Cap", "Suction P", "Brush P", "Other(Pump+display)", "V"],
        "均值(Typ)": [2450.0, 70.0, 30.0, 15.0, 3.6],
        "标准差(Std)": [20.74, 0.77, 0.90, 0.45, 0.0036],
        "分布": ["正态分布", "正态分布", "正态分布", "正态分布", "正态分布"]
    })

if "sim_results" not in st.session_state:
    st.session_state.sim_results = None

# 辅助函数：安全计算公式
def safe_eval(expr: str, context: Dict[str, float]) -> float:
    """
    在给定的变量上下文中安全计算公式表达式
    """
    # 允许的数学函数
    allowed_names = {
        "sqrt": math.sqrt, "exp": math.exp, "log": math.log, "log10": math.log10,
        "sin": math.sin, "cos": math.cos, "tan": math.tan, "pi": math.pi, "e": math.e,
        "abs": abs, "pow": pow
    }
    # 合并用户变量
    allowed_names.update(context)
    try:
        # 使用eval并限制命名空间
        result = eval(expr, {"__builtins__": {}}, allowed_names)
        return float(result)
    except Exception as e:
        st.error(f"公式计算错误: {e}\n请检查变量名是否与参数名称一致，表达式是否合法。")
        return np.nan

# 蒙特卡洛主模拟
def run_monte_carlo(params_df: pd.DataFrame,
                    formula: str,
                    n_sim: int,
                    usl: float,
                    lsl: float,
                    seed: int = 42) -> Dict[str, Any]:
    """
    执行蒙特卡洛模拟
    返回: 输出数组, 统计量, CPK, ppm, 分布图数据等
    """
    np.random.seed(seed)
    n_params = len(params_df)
    param_names = params_df["参数名称"].tolist()
    means = params_df["均值(Typ)"].values.astype(float)
    stds = params_df["标准差(Std)"].values.astype(float)

    # 生成所有参数的随机样本矩阵 (n_sim x n_params)
    samples = np.random.normal(loc=means, scale=stds, size=(n_sim, n_params))

    # 计算结果
    results = []
    for i in range(n_sim):
        context = {name: samples[i, j] for j, name in enumerate(param_names)}
        val = safe_eval(formula, context)
        if not np.isnan(val):
            results.append(val)

    results = np.array(results)
    if len(results) == 0:
        st.error("所有公式计算均失败，请检查公式！")
        return None

    # 统计量
    mean_out = np.mean(results)
    std_out = np.std(results, ddof=1)
    max_out = np.max(results)
    min_out = np.min(results)

    # CPK 和 PPM
    cpk = min((usl - mean_out) / (3 * std_out), (mean_out - lsl) / (3 * std_out)) if std_out > 0 else 0
    failures_up = np.sum(results > usl) / len(results) * 1e6
    failures_dn = np.sum(results < lsl) / len(results) * 1e6
    failures_all = failures_up + failures_dn

    # 直方图数据 (25 bins)
    hist_counts, bin_edges = np.histogram(results, bins=25, density=False)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2
    # 理论正态分布PDF (用于叠加)
    x_pdf = np.linspace(min_out, max_out, 200)
    pdf_theory = stats.norm.pdf(x_pdf, mean_out, std_out)

    return {
        "results": results,
        "mean": mean_out,
        "std": std_out,
        "max": max_out,
        "min": min_out,
        "cpk": cpk,
        "failures_all": failures_all,
        "failures_up": failures_up,
        "failures_dn": failures_dn,
        "hist_counts": hist_counts,
        "bin_edges": bin_edges,
        "bin_centers": bin_centers,
        "x_pdf": x_pdf,
        "pdf_theory": pdf_theory,
    }

# 贡献度分析（固定其他参数，单参数变化）
def sensitivity_analysis(params_df: pd.DataFrame,
                         formula: str,
                         n_sim: int,
                         seed: int = 42) -> pd.DataFrame:
    """
    计算每个参数对输出方差的贡献百分比
    方法：对每个参数单独进行随机抽样（其他参数固定为均值），计算输出方差；总方差为各参数方差之和。
    """
    np.random.seed(seed)
    n_params = len(params_df)
    param_names = params_df["参数名称"].tolist()
    means = params_df["均值(Typ)"].values.astype(float)
    stds = params_df["标准差(Std)"].values.astype(float)

    variances = []
    for i in range(n_params):
        # 固定其他参数为均值，只对第i个参数抽样
        samples_i = np.random.normal(loc=means[i], scale=stds[i], size=n_sim)
        results_i = []
        for val in samples_i:
            context = {name: means[j] for j, name in enumerate(param_names)}
            context[param_names[i]] = val
            res = safe_eval(formula, context)
            if not np.isnan(res):
                results_i.append(res)
        var_i = np.var(results_i, ddof=1) if len(results_i) > 1 else 0.0
        variances.append(var_i)

    total_var = sum(variances)
    if total_var == 0:
        contributions = [0.0] * n_params
    else:
        contributions = [v / total_var for v in variances]

    # 排序并返回DataFrame
    df_contrib = pd.DataFrame({
        "参数": param_names,
        "方差贡献": variances,
        "贡献百分比": contributions
    })
    df_contrib = df_contrib.sort_values("贡献百分比", ascending=False).reset_index(drop=True)
    df_contrib["贡献百分比"] = df_contrib["贡献百分比"].apply(lambda x: f"{x:.2%}")
    return df_contrib, contributions, param_names

# 绘制直方图
def plot_histogram(results: np.ndarray,
                   bin_centers: np.ndarray,
                   hist_counts: np.ndarray,
                   x_pdf: np.ndarray,
                   pdf_theory: np.ndarray,
                   usl: float, lsl: float):
    fig, ax = plt.subplots(figsize=(10, 6))
    # 直方图
    ax.bar(bin_centers, hist_counts, width=(bin_centers[1]-bin_centers[0])*0.9,
           alpha=0.6, label="模拟频率", color="steelblue")
    # 理论密度曲线（缩放至与直方图面积匹配）
    bin_width = bin_centers[1] - bin_centers[0]
    area = np.sum(hist_counts) * bin_width
    ax.plot(x_pdf, pdf_theory * area, 'r-', linewidth=2, label="理论正态分布")
    # 规格线
    ax.axvline(usl, color='green', linestyle='--', label=f"USL = {usl}")
    ax.axvline(lsl, color='orange', linestyle='--', label=f"LSL = {lsl}")
    ax.set_xlabel("输出值")
    ax.set_ylabel("频次")
    ax.set_title("输出分布直方图")
    ax.legend()
    return fig

# 绘制贡献百分比饼图
def plot_contribution(contributions: List[float], param_names: List[str]):
    fig, ax = plt.subplots(figsize=(8, 6))
    # 过滤掉贡献为0的项
    non_zero = [(p, c) for p, c in zip(param_names, contributions) if c > 0]
    if not non_zero:
        ax.text(0.5, 0.5, "无显著贡献", ha='center', va='center')
        return fig
    names, vals = zip(*non_zero)
    ax.pie(vals, labels=names, autopct='%1.1f%%', startangle=90)
    ax.axis('equal')
    ax.set_title("各参数对输出方差的贡献百分比")
    return fig

# 主界面
def main():
    st.sidebar.header("⚙️ 模拟设置")
    n_sim = st.sidebar.number_input("模拟次数 (Trail number)", min_value=100, max_value=100000, value=1000, step=100)
    usl = st.sidebar.number_input("规格上限 (Upper L)", value=40.0, step=1.0)
    lsl = st.sidebar.number_input("规格下限 (Lower L)", value=30.0, step=1.0)
    seed = st.sidebar.number_input("随机种子 (保证可重复)", value=42, step=1)

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📐 公式定义")
    default_formula = "Cell Cap + Suction P + Brush P + Other(Pump+display) + V"
    formula = st.sidebar.text_area("输出公式 (使用参数名称作为变量)", value=default_formula, height=100)
    st.sidebar.caption("示例: (Cell Cap - 2450)/20.74 + Suction P * 0.5   \n支持的运算: + - * / **, 函数: sqrt, exp, log, sin 等")

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📝 参数编辑")
    st.sidebar.info("下方表格可直接编辑参数名称、均值、标准差和分布类型。")

    # 参数编辑表格
    edited_df = st.data_editor(st.session_state.params, num_rows="dynamic", use_container_width=True)

    if st.button("🚀 开始蒙特卡洛模拟", type="primary"):
        if edited_df.isnull().values.any():
            st.error("参数表中存在空值，请检查！")
            return

        # 检查参数名称唯一且合法（变量名不能包含空格和特殊字符，仅字母数字下划线）
        param_names = edited_df["参数名称"].astype(str).tolist()
        if len(set(param_names)) != len(param_names):
            st.error("参数名称必须唯一！")
            return
        # 简单检查变量名合法性
        valid_name_pattern = re.compile(r'^[a-zA-Z_][a-zA-Z0-9_]*$')
        for name in param_names:
            if not valid_name_pattern.match(name):
                st.error(f"参数名称 '{name}' 不合法，只能包含字母、数字、下划线且不能以数字开头。")
                return

        # 执行主模拟
        with st.spinner("正在进行蒙特卡洛模拟..."):
            sim_res = run_monte_carlo(edited_df, formula, n_sim, usl, lsl, seed)
        if sim_res is None:
            return

        # 执行贡献度分析
        with st.spinner("正在计算各参数贡献度..."):
            df_contrib, contributions, param_names = sensitivity_analysis(edited_df, formula, n_sim, seed)

        # 存储结果到session_state
        st.session_state.sim_results = {
            "sim_res": sim_res,
            "df_contrib": df_contrib,
            "contributions": contributions,
            "param_names": param_names,
            "formula": formula,
            "params_df": edited_df
        }

    # 显示结果
    if st.session_state.sim_results:
        res = st.session_state.sim_results["sim_res"]
        df_contrib = st.session_state.sim_results["df_contrib"]
        contributions = st.session_state.sim_results["contributions"]
        param_names = st.session_state.sim_results["param_names"]
        params_df = st.session_state.sim_results["params_df"]

        st.header("📈 模拟结果")
        col1, col2, col3 = st.columns(3)
        col1.metric("输出均值", f"{res['mean']:.4f}")
        col1.metric("输出标准差", f"{res['std']:.4f}")
        col2.metric("最大值", f"{res['max']:.4f}")
        col2.metric("最小值", f"{res['min']:.4f}")
        col3.metric("Cpk", f"{res['cpk']:.4f}")
        col3.metric("总失效 ppm", f"{res['failures_all']:.2f}")

        st.subheader("分布直方图")
        fig_hist = plot_histogram(res['results'], res['bin_centers'], res['hist_counts'],
                                  res['x_pdf'], res['pdf_theory'], usl, lsl)
        st.pyplot(fig_hist)

        st.subheader("📊 各参数对输出方差的贡献百分比")
        st.dataframe(df_contrib, use_container_width=True)

        fig_pie = plot_contribution(contributions, param_names)
        st.pyplot(fig_pie)

        # 可选：显示原始模拟数据预览
        with st.expander("查看模拟数据预览"):
            # 生成样本矩阵用于展示
            np.random.seed(seed)
            means = params_df["均值(Typ)"].values.astype(float)
            stds = params_df["标准差(Std)"].values.astype(float)
            samples = np.random.normal(loc=means, scale=stds, size=(min(100, n_sim), len(param_names)))
            df_samples = pd.DataFrame(samples, columns=param_names)
            df_samples["计算结果"] = res['results'][:100] if len(res['results']) >= 100 else np.pad(res['results'], (0,100-len(res['results'])), constant_values=np.nan)
            st.dataframe(df_samples, use_container_width=True)

        st.success("模拟完成！")

if __name__ == "__main__":
    main()
