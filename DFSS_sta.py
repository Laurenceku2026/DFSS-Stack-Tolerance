# app.py
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import math
import re
import base64
from io import BytesIO
from typing import List, Dict, Any, Tuple

st.set_page_config(page_title="Para_Variation - 蒙特卡洛模拟", layout="wide")
st.title("📊 Para_Variation - 基于蒙特卡洛模拟分析")
st.markdown("根据输入参数的分布进行随机抽样，计算用户定义的公式结果，分析输出分布及各参数贡献度。")

# 初始化 session state
if "params" not in st.session_state:
    st.session_state.params = pd.DataFrame({
        "参数名称": ["Cell Cap", "Suction P", "Brush P", "Other(Pump+display)", "V"],
        "均值(Typ)": [2450.0, 70.0, 30.0, 15.0, 3.6],
        "标准差(Std)": [20.74, 0.77, 0.90, 0.45, 0.0036],
        "分布": ["正态分布", "正态分布", "正态分布", "正态分布", "正态分布"]
    })

if "sim_results_raw" not in st.session_state:
    st.session_state.sim_results_raw = None

if "formula" not in st.session_state:
    st.session_state.formula = "Cell Cap * V * 7 / 1000 * 60 / (Suction P + Brush P + Other(Pump+display))"

if "output_name" not in st.session_state:
    st.session_state.output_name = "Runtime"

if "usl" not in st.session_state:
    st.session_state.usl = 40.0
if "lsl" not in st.session_state:
    st.session_state.lsl = 30.0
if "use_spec" not in st.session_state:
    st.session_state.use_spec = True  # 是否使用规格限

# ---------- 上下限双向同步回调函数 ----------
def sync_usl_from_main():
    st.session_state.usl = st.session_state.main_usl
def sync_lsl_from_main():
    st.session_state.lsl = st.session_state.main_lsl
def sync_usl_from_sidebar():
    st.session_state.usl = st.session_state.usl_sidebar
def sync_lsl_from_sidebar():
    st.session_state.lsl = st.session_state.lsl_sidebar

# 回调函数：追加参数名到公式
def append_param(param_name: str):
    current = st.session_state.formula
    if current and not current.endswith((' ', '+')):
        st.session_state.formula = current + " " + param_name
    else:
        st.session_state.formula = current + param_name

# 安全计算公式（支持任意参数名）
def safe_eval_with_mapping(expr: str, param_names: List[str], context_values: List[float]) -> float:
    temp_names = [f"__p{i}__" for i in range(len(param_names))]
    sorted_params = sorted(zip(param_names, temp_names), key=lambda x: len(x[0]), reverse=True)
    expr_temp = expr
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
    except Exception as e:
        st.error(f"公式计算错误: {e}\n请检查参数名是否与表格中的名称完全一致。")
        return np.nan

# 蒙特卡洛主模拟
def run_monte_carlo(params_df: pd.DataFrame,
                    formula: str,
                    n_sim: int,
                    seed: int = 42) -> Dict[str, Any]:
    np.random.seed(seed)
    n_params = len(params_df)
    param_names = params_df["参数名称"].astype(str).tolist()
    means = params_df["均值(Typ)"].values.astype(float)
    stds = params_df["标准差(Std)"].values.astype(float)

    samples = np.random.normal(loc=means, scale=stds, size=(n_sim, n_params))
    results = []
    for i in range(n_sim):
        val = safe_eval_with_mapping(formula, param_names, samples[i, :])
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

# 贡献度分析
def sensitivity_analysis(params_df: pd.DataFrame,
                         formula: str,
                         n_sim: int,
                         seed: int = 42) -> Tuple[pd.DataFrame, List[float], List[str]]:
    np.random.seed(seed)
    n_params = len(params_df)
    param_names = params_df["参数名称"].astype(str).tolist()
    means = params_df["均值(Typ)"].values.astype(float)
    stds = params_df["标准差(Std)"].values.astype(float)

    variances = []
    for i in range(n_params):
        samples_i = np.random.normal(loc=means[i], scale=stds[i], size=n_sim)
        results_i = []
        for val in samples_i:
            context_vals = means.copy()
            context_vals[i] = val
            res = safe_eval_with_mapping(formula, param_names, context_vals)
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
    df_contrib["贡献百分比_显示"] = df_contrib["贡献百分比"].apply(lambda x: f"{x:.6%}")
    return df_contrib, contributions, param_names

# 绘图函数：直方图（右上角添加统计信息框）
def plot_histogram(results, bin_centers, hist_counts, x_pdf, pdf_theory, usl, lsl, output_name, n_sim, use_spec):
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.bar(bin_centers, hist_counts, width=(bin_centers[1]-bin_centers[0])*0.9,
           alpha=0.6, label="Histogram", color="steelblue")
    bin_width = bin_centers[1] - bin_centers[0]
    area = np.sum(hist_counts) * bin_width
    ax.plot(x_pdf, pdf_theory * area, 'r-', linewidth=2, label="Gaussian Fitting")
    
    if use_spec:
        ax.axvline(usl, color='green', linestyle='--', label=f"USL = {usl:.2f}")
        ax.axvline(lsl, color='orange', linestyle='--', label=f"LSL = {lsl:.2f}")
    
    ax.set_xlabel(output_name)
    ax.set_ylabel("Frequency")
    ax.set_title(f"{output_name} Distribution")
    ax.legend(loc='upper right')
    
    # 在右上角添加统计信息框（仿附图）
    mean_val = np.mean(results)
    std_val = np.std(results, ddof=1)
    max_val = np.max(results)
    min_val = np.min(results)
    stats_text = f"NO.={len(results)}\nAVE={mean_val:.4f}\nSTD={std_val:.4f}\nMAX={max_val:.4f}\nMIN={min_val:.4f}"
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            fontsize=9, family='monospace')
    return fig

# 绘图函数：水平条形图（完全按照附图红色样式）
def plot_contribution_horizontal(contributions: List[float], param_names: List[str], output_name: str):
    non_zero = [(p, c) for p, c in zip(param_names, contributions) if c > 0]
    if not non_zero:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "无显著贡献", ha='center', va='center')
        ax.set_title(f"{output_name} 设计参数影响百分比")
        return fig
    names, vals = zip(*non_zero)
    # 按贡献值升序排列（水平条形图从下往上）
    sorted_indices = np.argsort(vals)
    names = [names[i] for i in sorted_indices]
    vals = [vals[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(8, max(4, len(names)*0.4)))
    bars = ax.barh(names, vals, color='steelblue')
    for bar, val in zip(bars, vals):
        ax.text(val + 0.01, bar.get_y() + bar.get_height()/2, f'{val:.6%}',
                va='center', fontsize=9)
    ax.set_xlabel("影响百分比")
    ax.set_title(f"{output_name} 设计参数影响百分比")
    ax.set_xlim(0, max(vals) * 1.15)
    ax.grid(axis='x', linestyle='--', alpha=0.7)
    return fig

# 计算 CPK 和 PPM（基于已保存的结果数组）
def compute_cpk_ppm(results: np.ndarray, usl: float, lsl: float):
    mean_out = np.mean(results)
    std_out = np.std(results, ddof=1)
    if std_out == 0:
        cpk = 0.0
    else:
        cpk = min((usl - mean_out) / (3 * std_out), (mean_out - lsl) / (3 * std_out))
    failures_up = np.sum(results > usl) / len(results) * 1e6
    failures_dn = np.sum(results < lsl) / len(results) * 1e6
    failures_all = failures_up + failures_dn
    return cpk, failures_all, failures_up, failures_dn

# 生成 HTML 报告
def generate_report(raw, usl, lsl, n_sim, seed, formula, params_df, use_spec):
    results = raw["results"]
    output_name = raw["output_name"]
    if use_spec:
        cpk, failures_all, failures_up, failures_dn = compute_cpk_ppm(results, usl, lsl)
    else:
        cpk, failures_all, failures_up, failures_dn = 0.0, 0.0, 0.0, 0.0

    fig_hist = plot_histogram(results, raw["bin_centers"], raw["hist_counts"],
                              raw["x_pdf"], raw["pdf_theory"], usl, lsl, output_name, n_sim, use_spec)
    buf_hist = BytesIO()
    fig_hist.savefig(buf_hist, format="png", dpi=150, bbox_inches="tight")
    hist_b64 = base64.b64encode(buf_hist.getvalue()).decode()
    plt.close(fig_hist)

    contributions = raw["contributions"]
    param_names = raw["param_names"]
    fig_barh = plot_contribution_horizontal(contributions, param_names, output_name)
    buf_barh = BytesIO()
    fig_barh.savefig(buf_barh, format="png", dpi=150, bbox_inches="tight")
    barh_b64 = base64.b64encode(buf_barh.getvalue()).decode()
    plt.close(fig_barh)

    df_contrib = raw["df_contrib"].copy()
    df_contrib["贡献百分比_显示"] = df_contrib["贡献百分比"].apply(lambda x: f"{x:.6%}")

    samples_df = pd.DataFrame(raw["samples"], columns=param_names)
    samples_df[output_name] = results
    preview_df = samples_df.head(100)

    params_html = params_df.to_html(index=False, classes="dataframe", border=1, justify="center", float_format="%.2f")
    contrib_html = df_contrib[["参数", "贡献百分比_显示"]].rename(columns={"贡献百分比_显示": "贡献百分比"}).to_html(index=False, classes="dataframe", border=1, justify="center")
    preview_html = preview_df.to_html(index=False, classes="dataframe", border=1, justify="center", float_format="%.2f")

    css = """
    <style>
        body { font-family: 'Segoe UI', Arial, sans-serif; margin: 20px; }
        h1 { color: #2c3e50; border-bottom: 2px solid #3498db; padding-bottom: 5px; }
        h2 { color: #34495e; margin-top: 25px; }
        h3 { color: #555; }
        .report-section { margin-bottom: 30px; }
        table.dataframe {
            border-collapse: collapse;
            width: 100%;
            margin: 10px 0;
            font-size: 12pt;
        }
        table.dataframe th, table.dataframe td {
            border: 1px solid black;
            padding: 8px;
            text-align: center;
            vertical-align: middle;
        }
        table.dataframe th {
            background-color: #f2f2f2;
            font-weight: bold;
        }
        .stats-table td, .stats-table th { text-align: center; }
        .footer { margin-top: 40px; font-size: 10pt; color: #7f8c8d; text-align: center; }
    </style>
    """

    if use_spec:
        stats_html = f"""
        <table class="dataframe stats-table">
            <tr><th>统计量</th><th>数值</th></tr>
            <tr><td>均值</td><td>{raw['mean']:.2f}</td></tr>
            <tr><td>标准差</td><td>{raw['std']:.2f}</td></tr>
            <tr><td>最大值</td><td>{raw['max']:.2f}</td></tr>
            <tr><td>最小值</td><td>{raw['min']:.2f}</td></tr>
            <tr><td>Cpk</td><td>{cpk:.2f}</td></tr>
            <tr><td>Failure All (ppm)</td><td>{failures_all:.2f}</td></tr>
            <tr><td>Failure Up (ppm)</td><td>{failures_up:.2f}</td></tr>
            <tr><td>Failure Dn (ppm)</td><td>{failures_dn:.2f}</td></tr>
        </table>
        """
    else:
        stats_html = f"""
        <table class="dataframe stats-table">
            <tr><th>统计量</th><th>数值</th></tr>
            <tr><td>均值</td><td>{raw['mean']:.2f}</td></tr>
            <tr><td>标准差</td><td>{raw['std']:.2f}</td></tr>
            <tr><td>最大值</td><td>{raw['max']:.2f}</td></tr>
            <tr><td>最小值</td><td>{raw['min']:.2f}</td></tr>
        </table>
        """

    report_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>蒙特卡洛模拟报告 - {output_name}</title>
        {css}
    </head>
    <body>
        <h1>蒙特卡洛模拟分析报告</h1>
        <div class="report-section">
            <h2>1. 模拟设置</h2>
            <ul>
                <li><strong>输出变量名称：</strong> {output_name}</li>
                <li><strong>公式：</strong> {formula}</li>
                <li><strong>模拟次数：</strong> {n_sim}</li>
                <li><strong>使用规格限：</strong> {"是" if use_spec else "否"}</li>
                {f"<li><strong>规格上限 (USL)：</strong> {usl:.2f}</li>" if use_spec else ""}
                {f"<li><strong>规格下限 (LSL)：</strong> {lsl:.2f}</li>" if use_spec else ""}
                <li><strong>随机种子：</strong> {seed}</li>
            </ul>
        </div>
        <div class="report-section">
            <h2>2. 输入参数表</h2>
            {params_html}
        </div>
        <div class="report-section">
            <h2>3. 模拟结果统计</h2>
            {stats_html}
        </div>
        <div class="report-section">
            <h2>4. 分布直方图</h2>
            <img src="data:image/png;base64,{hist_b64}" alt="Distribution Histogram" style="max-width:100%;">
        </div>
        <div class="report-section">
            <h2>5. 设计参数对 {output_name} 影响百分比</h2>
            <img src="data:image/png;base64,{barh_b64}" alt="Contribution Bar Chart" style="max-width:100%;">
            <h3>详细数据表</h3>
            {contrib_html}
        </div>
        <div class="report-section">
            <h2>6. 模拟数据预览（前100行）</h2>
            {preview_html}
        </div>
        <div class="footer">
            报告生成时间：{pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S")}
        </div>
    </body>
    </html>
    """
    return report_html

# 主程序
def main():
    st.sidebar.header("⚙️ 模拟设置")
    n_sim = st.sidebar.number_input("模拟次数 (Trail number)", min_value=100, max_value=100000, value=1000, step=100)
    use_spec = st.sidebar.checkbox("使用规格限", value=st.session_state.use_spec, key="use_spec_sidebar")
    st.session_state.use_spec = use_spec
    if use_spec:
        st.sidebar.number_input("规格上限 (Upper L)", value=st.session_state.usl, step=0.1, format="%.4f", key="usl_sidebar", on_change=sync_usl_from_sidebar)
        st.sidebar.number_input("规格下限 (Lower L)", value=st.session_state.lsl, step=0.1, format="%.4f", key="lsl_sidebar", on_change=sync_lsl_from_sidebar)
    seed = st.sidebar.number_input("随机种子", value=42, step=1)

    st.markdown("---")
    st.subheader("📝 参数输入")
    edited_df = st.data_editor(st.session_state.params, num_rows="dynamic", use_container_width=True)

    st.markdown("---")
    st.subheader("📐 公式定义")
    output_name = st.text_input("输出变量名称", value=st.session_state.output_name, key="output_name_input")
    st.session_state.output_name = output_name if output_name.strip() else "Output"

    st.caption("使用与表格中**完全一致**的参数名称，点击下方按钮可插入参数。")
    param_names = edited_df["参数名称"].astype(str).tolist()
    cols_per_row = 5
    for i in range(0, len(param_names), cols_per_row):
        cols = st.columns(min(cols_per_row, len(param_names) - i))
        for idx, name in enumerate(param_names[i:i+cols_per_row]):
            with cols[idx]:
                st.button(f"➕ {name}", key=f"btn_{name}", on_click=append_param, args=(name,))
    formula = st.text_area("计算公式", value=st.session_state.formula, height=100, key="formula_input")
    st.session_state.formula = formula
    st.caption("支持的运算: + - * / **, 括号, 函数: sqrt, exp, log, sin, cos, tan, pi, e 等")

    if st.button("🚀 开始蒙特卡洛模拟", type="primary"):
        if edited_df.isnull().values.any():
            st.error("参数表中存在空值，请检查！")
            return
        if len(set(param_names)) != len(param_names):
            st.error("参数名称必须唯一！")
            return
        if not formula.strip():
            st.error("公式不能为空！")
            return

        with st.spinner("正在进行蒙特卡洛模拟..."):
            sim_res = run_monte_carlo(edited_df, formula, n_sim, seed)
        if sim_res is None:
            return

        with st.spinner("正在计算各参数贡献度..."):
            df_contrib, contributions, param_names = sensitivity_analysis(edited_df, formula, n_sim, seed)

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
            "params_df": edited_df,
            "output_name": output_name,
            "formula": formula,
        }

    # 显示结果（如果已有模拟结果）
    if st.session_state.sim_results_raw is not None:
        raw = st.session_state.sim_results_raw
        results = raw["results"]
        output_name = raw["output_name"]
        use_spec = st.session_state.use_spec
        if use_spec:
            usl = st.session_state.usl
            lsl = st.session_state.lsl
            cpk, failures_all, failures_up, failures_dn = compute_cpk_ppm(results, usl, lsl)
        else:
            usl = lsl = 0.0
            cpk = failures_all = failures_up = failures_dn = 0.0

        st.header(f"📈 模拟结果: {output_name}")
        col1, col2, col3 = st.columns(3)
        col1.metric(f"{output_name} 均值", f"{raw['mean']:.2f}")
        col1.metric(f"{output_name} 标准差", f"{raw['std']:.2f}")
        col2.metric("最大值", f"{raw['max']:.2f}")
        col2.metric("最小值", f"{raw['min']:.2f}")

        # Failure ppm level 区域（仅在启用规格限时显示）
        if use_spec:
            st.subheader("Failure ppm level")
            st.caption("💡 可调节上下限以实时观察PPM水平的变化")
            col_left, col_right = st.columns([1, 2])
            with col_left:
                st.number_input("规格上限 (USL)", value=st.session_state.usl, step=0.1, format="%.4f", key="main_usl", on_change=sync_usl_from_main)
                st.number_input("规格下限 (LSL)", value=st.session_state.lsl, step=0.1, format="%.4f", key="main_lsl", on_change=sync_lsl_from_main)
            with col_right:
                ppm_html = f"""
                <style>
                .ppm-table {{
                    border-collapse: collapse;
                    width: auto;
                    margin: 0 auto;
                }}
                .ppm-table th, .ppm-table td {{
                    border: 2px solid black;
                    padding: 8px 16px;
                    text-align: center;
                    font-weight: normal;
                }}
                .ppm-table th {{
                    background-color: #f0f0f0;
                }}
                </style>
                <table class="ppm-table">
                    <tr><th>CPK</th><th>Failure All</th><th>Failure Up</th><th>Failure Dn</th></tr>
                    <tr>
                        <td>{cpk:.2f}</td>
                        <td>{failures_all:.2f}</td>
                        <td>{failures_up:.2f}</td>
                        <td>{failures_dn:.2f}</td>
                    </tr>
                </table>
                """
                st.markdown(ppm_html, unsafe_allow_html=True)

        # 分布直方图（右上角统计信息框）
        st.subheader("分布直方图")
        fig_hist = plot_histogram(results, raw["bin_centers"], raw["hist_counts"],
                                  raw["x_pdf"], raw["pdf_theory"], usl, lsl, output_name, n_sim, use_spec)
        st.pyplot(fig_hist)

        # 设计参数影响百分比（完全按照附图）
        st.subheader(f"设计参数对 {output_name} 影响百分比")
        contributions = raw["contributions"]
        param_names = raw["param_names"]
        fig_barh = plot_contribution_horizontal(contributions, param_names, output_name)
        st.pyplot(fig_barh)

        with st.expander("查看贡献百分比数据表"):
            st.dataframe(raw["df_contrib"][["参数", "贡献百分比_显示"]].rename(columns={"贡献百分比_显示": "贡献百分比"}), use_container_width=True)

        # 模拟数据预览
        with st.expander("查看全部模拟数据"):
            samples_df = pd.DataFrame(raw["samples"], columns=param_names)
            samples_df[output_name] = results
            display_df = samples_df.round(2)
            st.dataframe(display_df, use_container_width=True, height=400)
            csv = samples_df.to_csv(index=False, float_format="%.6f")
            st.download_button(
                label="📥 下载模拟数据 (CSV)",
                data=csv,
                file_name=f"monte_carlo_data_{output_name}.csv",
                mime="text/csv"
            )

        # 报告下载
        report_html = generate_report(raw, usl, lsl, n_sim, seed, formula, edited_df, use_spec)
        st.download_button(
            label="📄 下载专业报告 (HTML)",
            data=report_html,
            file_name=f"MonteCarlo_Report_{output_name}.html",
            mime="text/html"
        )

        st.success("模拟完成！")

if __name__ == "__main__":
    main()
