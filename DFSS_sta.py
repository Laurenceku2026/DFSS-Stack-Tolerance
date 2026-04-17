def generate_word_report(raw, usl, lsl, n_sim, seed, formula, params_df, param_letters, analyst_name, analyst_title, output_name):
    results = raw["results"]
    cpk, failures_all, failures_up, failures_dn = compute_cpk_ppm(results, usl, lsl)

    doc = Document()
    
    # 标题
    title = doc.add_heading(f"{output_name} - DFSS模拟分析报告", 0)
    title.alignment = WD_ALIGN_PARAGRAPH.CENTER
    
    # 分析人信息
    doc.add_heading("分析人信息", level=1)
    p = doc.add_paragraph()
    p.add_run(f"分析人姓名：{analyst_name if analyst_name else '未填写'}").bold = True
    p.add_run(f"\n头衔：{analyst_title if analyst_title else '未填写'}")
    
    # 模拟设置
    doc.add_heading("1. 模拟设置", level=1)
    doc.add_paragraph(f"输出变量名称：{output_name}")
    doc.add_paragraph(f"公式：{formula}")
    doc.add_paragraph(f"模拟次数：{n_sim}")
    doc.add_paragraph(f"随机种子：{seed}")
    doc.add_paragraph(f"规格上限 (USL)：{usl if usl is not None else '无'}")
    doc.add_paragraph(f"规格下限 (LSL)：{lsl if lsl is not None else '无'}")
    
    # 输入参数表
    doc.add_heading("2. 输入参数表", level=1)
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
    
    # 模拟结果统计（动态标题，横向表格）
    doc.add_heading(f"3. {output_name}模拟结果统计", level=1)
    stats_table = doc.add_table(rows=5, cols=2)
    stats_table.style = 'Table Grid'
    # 设置列宽可选
    stats_table.cell(0, 0).text = "统计量"
    stats_table.cell(0, 1).text = "数值"
    stats_table.cell(1, 0).text = "均值"
    stats_table.cell(1, 1).text = f"{raw['mean']:.2f}"
    stats_table.cell(2, 0).text = "标准差"
    stats_table.cell(2, 1).text = f"{raw['std']:.4f}"
    stats_table.cell(3, 0).text = "最大值"
    stats_table.cell(3, 1).text = f"{raw['max']:.2f}"
    stats_table.cell(4, 0).text = "最小值"
    stats_table.cell(4, 1).text = f"{raw['min']:.2f}"
    if cpk is not None:
        # 添加 Cpk 和 PPM 行（增加行数）
        row_idx = len(stats_table.rows)
        stats_table.add_row().cells[0].text = "Cpk"
        stats_table.rows[-1].cells[1].text = f"{cpk:.2f}"
        stats_table.add_row().cells[0].text = "Failure All (ppm)"
        stats_table.rows[-1].cells[1].text = f"{failures_all:.2f}" if failures_all is not None else "-"
        stats_table.add_row().cells[0].text = "Failure Up (ppm)"
        stats_table.rows[-1].cells[1].text = f"{failures_up:.2f}" if failures_up is not None else "-"
        stats_table.add_row().cells[0].text = "Failure Dn (ppm)"
        stats_table.rows[-1].cells[1].text = f"{failures_dn:.2f}" if failures_dn is not None else "-"
    
    # 分布直方图
    doc.add_heading("4. 分布直方图", level=1)
    fig_hist = plot_histogram(results, raw["bin_centers"], raw["hist_counts"], raw["x_pdf"], raw["pdf_theory"], usl, lsl, output_name, n_sim)
    buf_hist = BytesIO()
    fig_hist.savefig(buf_hist, format='png', dpi=150, bbox_inches='tight')
    buf_hist.seek(0)
    doc.add_picture(buf_hist, width=Inches(6))
    plt.close(fig_hist)
    
    # 设计参数影响百分比
    doc.add_heading("5. 设计参数对 " + output_name + " 影响百分比", level=1)
    fig_barh = plot_contribution_horizontal(raw["contributions"], raw["param_names"], output_name)
    buf_barh = BytesIO()
    fig_barh.savefig(buf_barh, format='png', dpi=150, bbox_inches='tight')
    buf_barh.seek(0)
    doc.add_picture(buf_barh, width=Inches(6))
    plt.close(fig_barh)
    
    # 贡献百分比数据表
    doc.add_heading("详细数据表", level=2)
    df_contrib = raw["df_contrib"].copy()
    df_contrib["贡献百分比"] = df_contrib["贡献百分比_显示"]
    contrib_table = doc.add_table(rows=len(df_contrib)+1, cols=2)
    contrib_table.style = 'Table Grid'
    contrib_table.cell(0, 0).text = "参数"
    contrib_table.cell(0, 1).text = "贡献百分比"
    for i, row in df_contrib.iterrows():
        contrib_table.cell(i+1, 0).text = row["参数"]
        contrib_table.cell(i+1, 1).text = row["贡献百分比"]
    
    # 联系信息
    doc.add_page_break()
    footer = doc.add_paragraph()
    footer.alignment = WD_ALIGN_PARAGRAPH.CENTER
    footer.add_run("联系：电邮 Techlife2027@gmail.com").italic = True
    footer.add_run(f"\n报告生成时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    doc_bytes = BytesIO()
    doc.save(doc_bytes)
    doc_bytes.seek(0)
    return doc_bytes
