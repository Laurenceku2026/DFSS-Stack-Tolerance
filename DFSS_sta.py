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
from typing import List, Dict, Any, Tuple, Optional
from datetime import datetime
from docx import Document
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

st.set_page_config(page_title="Para_Variation - 蒙特卡洛模拟", layout="wide")

# ==================== 多语言文本字典 ====================
TEXTS = {
    "zh": {
        # 主界面
        "title": "📊 Para_Variation - 基于蒙特卡洛模拟分析",
        "subtitle": "根据输入参数的分布进行随机抽样，计算用户定义的公式结果，分析输出分布及各参数贡献度。",
        # 侧边栏
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
        # 参数输入表格
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
        # 分布类型
        "dist_full": "正态分布（完整）",
        "dist_pos": "正态分布（正值）",
        "dist_neg": "正态分布（负值）",
        "dist_uniform": "均匀分布",
        "dist_lognorm": "对数正态分布",
        "dist_weibull": "威布尔分布",
        "dist_tri": "三角分布",
        # 均匀分布参数
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
        # 公式定义
        "formula_def": "📐 公式定义（设计值）",
        "design_var_name": "📌 设计变量名称",
        "formula_label": "📝 计算公式",
        "formula_hint": "💡 可直接在公式中使用字母（A, B, C...）代表对应参数，系统将自动识别。例如：A*E*7/1000*60/(B+C+D)",
        "formula_supported": "支持的运算: + - * / **, 括号, 函数: sqrt, exp, log, sin, cos, tan, pi, e 等。公式中的空格会被自动优化。",
        "design_value": "📌 当前设计值（基于均值）:",
        "formula_invalid": "公式无效或参数不匹配，无法计算设计值。请检查公式中的字母是否与上方对应关系一致，并确保运算正确。",
        # 模拟按钮
        "start_sim": "开始\n蒙特卡洛模拟",
        # 模拟结果
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
        # 报告内容
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
    },
    "en": {
        # ... 英文部分保持不变，仅修改 failure_ppm 为 "Failure ppm level"（原文已是）...
    }
}
