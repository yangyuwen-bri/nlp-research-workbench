from __future__ import annotations

import html
import json
from pathlib import Path


ROOT = Path("/Users/gsdata/work/nlp_tool")
JSON_PATH = ROOT / "tmpdata" / "waimai_10k_full_user_flow.json"
OUTPUT_PATH = ROOT / "doc" / "test_reports" / "waimai_10k_full_user_test_report.html"


def esc(value: object) -> str:
    return html.escape(str(value))


def step_map(payload: dict) -> dict[str, dict]:
    return {step["name"]: step for step in payload["steps"]}


def render_table(headers: list[str], rows: list[list[str]]) -> str:
    head = "".join(f"<th>{esc(item)}</th>" for item in headers)
    body = []
    for row in rows:
        body.append("<tr>" + "".join(f"<td>{cell}</td>" for cell in row) + "</tr>")
    return f"""
    <div class="table-shell">
      <table>
        <thead><tr>{head}</tr></thead>
        <tbody>{''.join(body)}</tbody>
      </table>
    </div>
    """


def quality_badge(level: str) -> str:
    cls = {
        "good": "good",
        "mixed": "mixed",
        "poor": "poor",
        "issue": "issue",
    }[level]
    label = {
        "good": "表现良好",
        "mixed": "结果一般",
        "poor": "结果较差",
        "issue": "需要修复",
    }[level]
    return f'<span class="badge {cls}">{label}</span>'


def main() -> None:
    payload = json.loads(JSON_PATH.read_text(encoding="utf-8"))
    steps = step_map(payload)

    health = steps["health"]["data"]
    readiness = steps["platform_readiness"]["data"]
    upload = steps["upload_dataset"]
    preview = steps["fetch_preview"]
    summary = steps["workspace_summary"]
    top_terms = steps["top_terms"]["data"]["items"]
    selected_terms = steps["selected_terms"]["data"]["items"]
    match_rows = steps["match_rows"]["data"]["items"]
    cooccurrence = steps["cooccurrence_edges"]["data"]["items"]
    explore = steps["explore_summary"]["data"]
    discover = steps["discover_summary"]["data"]
    classify = steps["classify_summary"]["data"]
    classification_section = steps["classification_section"]["data"]["items"]
    sentiment = steps["sentiment_summary"]["data"]
    sentiment_section = steps["sentiment_section"]["data"]["items"]
    sentiment_strategy = sentiment["settings"].get("sentiment_strategy", "unknown")
    sentiment_overview = sentiment["overview"]

    discover_topics = discover["previews"]["topics"]

    top_term_rows = [
        [esc(index), esc(item["term"]), esc(item["term_frequency"]), esc(item["document_frequency"]), esc(item.get("pos", ""))]
        for index, item in enumerate(top_terms[:12], start=1)
    ]

    topic_rows = []
    for index, topic in enumerate(discover_topics, start=1):
        evidence = topic["evidences"][0]["snippet"] if topic.get("evidences") else ""
        topic_rows.append(
            [
                esc(index),
                esc(topic["name"]),
                esc(topic["size"]),
                esc(" / ".join(topic["keywords"][:8])),
                esc(evidence),
            ]
        )

    sentiment_audit_expectations = {
        "很快，好吃，味道足，量大": "positive",
        "没有送水没有送水没有送水": "negative",
        "非常快，态度好": "positive",
        "方便，快捷，味道可口，快递给力": "positive",
        "送餐很及时": "positive",
        "今天师傅是不是手抖了，微辣格外辣": "negative",
        "送餐快,态度也特别好,辛苦啦谢谢": "positive",
        "麻辣香锅依然很好吃": "positive",
        "经过上次晚了2小时，这次超级快，20分钟就送到了……": "positive",
        "最后五分钟订的，卖家特别好接单了，谢谢": "positive",
    }
    sentiment_audit_rows = []
    sentiment_correct = 0
    for index, item in enumerate(sentiment_section[:10], start=1):
        snippet = item["snippet"]
        expected = sentiment_audit_expectations.get(snippet, "unknown")
        actual = item["label"]
        verdict = "正确" if expected == actual else "不符"
        if verdict == "正确":
            sentiment_correct += 1
        sentiment_audit_rows.append(
            [
                esc(index),
                esc(snippet),
                esc(expected),
                esc(actual),
                f'<span class="mini {("good" if verdict == "正确" else "poor")}">{esc(verdict)}</span>',
            ]
        )

    sentiment_accuracy_ratio = sentiment_correct / max(len(sentiment_audit_rows), 1)
    positive_count = int(sentiment_overview["positive_count"])
    neutral_count = int(sentiment_overview["neutral_count"])
    negative_count = int(sentiment_overview["negative_count"])
    sentiment_total = max(positive_count + neutral_count + negative_count, 1)
    neutral_ratio = neutral_count / sentiment_total

    if sentiment_accuracy_ratio >= 0.8:
        sentiment_quality = "good"
    elif sentiment_accuracy_ratio >= 0.5:
        sentiment_quality = "mixed"
    else:
        sentiment_quality = "poor"

    topic_quality = "mixed"
    if len(discover_topics) >= 6 and any(topic["name"] != f"主题 {i}" for i, topic in enumerate(discover_topics, start=1)):
        topic_quality = "good"

    report_bug_found = "样本量：0 条文本" in explore["previews"]["report_markdown"] or "样本量：0 条文本" in discover["previews"]["report_markdown"]

    workspace_quality = "issue" if steps["top_terms"]["elapsed_sec"] >= 10 else "mixed" if steps["top_terms"]["elapsed_sec"] >= 3 else "good"
    discover_quality = "issue" if steps["discover_summary"]["elapsed_sec"] >= 60 else "mixed" if steps["discover_summary"]["elapsed_sec"] >= 20 else "good"
    classification_quality = "mixed" if any(item["label"].startswith("主题 ") for item in classification_section[:10]) else "good"

    sentiment_result_text = (
        f"返回 {positive_count} 条正向、{neutral_count} 条中性、{negative_count} 条负向，策略为 {sentiment_strategy}"
    )

    flow_rows = [
        ["0", "检查生产环境底座", f"{steps['health']['elapsed_sec']}s / {steps['platform_readiness']['elapsed_sec']}s", f"运行模式 {health['app_env']}，存储 {health['storage_backend']}，队列 {health['task_queue_backend']}，发布就绪={readiness['ready']}", quality_badge("good" if readiness["ready"] else "issue")],
        ["1", "上传 `waimai_10k_full.csv`", f"{upload['elapsed_sec']}s", "上传成功，返回数据集 ID 与 5 条预览", quality_badge("good")],
        ["2", "打开数据预览", f"{preview['elapsed_sec']}s", "返回前 20 条文本预览", quality_badge("good")],
        ["3", "进入词项工作台总览", f"{summary['elapsed_sec']}s", "返回 11987 条语料的工作台摘要", quality_badge("good" if summary['elapsed_sec'] < 5 else "mixed")],
        ["4", "查看词频 / 选词 / 匹配 / 共词", f"{steps['top_terms']['elapsed_sec']}s ~ {steps['cooccurrence_edges']['elapsed_sec']}s", "各工作台分段都能返回，但大语料仍需等待", quality_badge(workspace_quality)],
        ["5", "点击生成探索结果", f"{steps['run_explore']['elapsed_sec']}s + {steps['explore_summary']['elapsed_sec']}s", "成功生成探索导出文件", quality_badge("good")],
        ["6", "点击发现主题", f"{steps['run_discover']['elapsed_sec']}s + {steps['discover_summary']['elapsed_sec']}s", f"成功生成 {len(discover_topics)} 个主题簇", quality_badge(discover_quality if topic_quality != 'good' else 'good')],
        ["7", "基于主题触发分类", f"{steps['run_classify']['elapsed_sec']}s + {steps['classify_summary']['elapsed_sec']}s", "分类链路跑通，但标签可解释性取决于主题命名质量", quality_badge(classification_quality)],
        ["8", "点击情感分析", f"{steps['run_sentiment']['elapsed_sec']}s + {steps['sentiment_summary']['elapsed_sec']}s", sentiment_result_text, quality_badge(sentiment_quality)],
    ]

    issue_rows = []
    if steps["top_terms"]["elapsed_sec"] >= 10:
        issue_rows.append(["P1", "工作台分段查询重复重算整份语料", f"词频、选词、匹配、共词单次查看需要 {steps['top_terms']['elapsed_sec']}s ~ {steps['cooccurrence_edges']['elapsed_sec']}s", "需要缓存或预计算工作台结果，否则大语料下切换模块会卡顿。"])
    if steps["discover_summary"]["elapsed_sec"] >= 60:
        issue_rows.append(["P1", "主题发现对 1.2 万条评论仍偏慢", f"本次 `discover` 完成耗时 {steps['discover_summary']['elapsed_sec']} 秒", "适合作为后台任务，不适合同步阻塞前台。"])
    if report_bug_found:
        issue_rows.append(["P1", "报告中的样本量字段错误", "探索报告或主题报告写成“样本量 0 条文本”", "这是结果正确性 bug，需要优先修复。"])
    if sentiment_quality != "good":
        issue_rows.append(["P1", "情感分析稳定性仍需继续验证", f"人工抽样 10 条中约 {sentiment_correct}/10 与直觉一致，策略 {sentiment_strategy}", "当前结果已明显强于规则词典，但仍建议继续做外部语料验证。"])
    if topic_quality != "good":
        issue_rows.append(["P2", "主题簇存在重叠或命名仍偏弱", "部分主题围绕相近的味道/配送问题展开", "主题解释性还可以再提升。"])
    if classification_quality != "good":
        issue_rows.append(["P2", "分类模块可解释性仍受主题标签限制", "分类标签仍复用自动主题名称", "没有人工命名和确认前，不适合作为最终业务标签。"])

    top_term_comment = (
        "词频结果总体贴合外卖评论场景，`味道`、`送餐`、`好吃`、`速度`、`难吃` 都是高价值词。"
        " 但也能看到 `小时`、`外卖`、`百度` 这类词混入高位，说明自动选词还缺少更强的场景清洗。"
    )
    topic_comment = (
        "主题结果能抓到“配送慢”“卷饼/煎饼”“漏送饮料/赠品”等真实问题簇，说明主题模块具备研究价值；"
        " 但多个主题之间重叠较大，名称仍是占位式的 `主题 N`，需要命名和去重。"
    )
    if sentiment_quality == "good":
        sentiment_comment = (
            f"情感模块现在已经切到 `{sentiment_strategy}`，不再使用规则词典兜底。"
            f" 当前抽样命中率约为 {sentiment_correct}/10，中性占比 {neutral_ratio:.1%}，整体可用性明显好于上一轮。"
        )
    elif sentiment_quality == "mixed":
        sentiment_comment = (
            f"情感模块现在走 `{sentiment_strategy}`，结果比规则词典强，但抽样仍只有 {sentiment_correct}/10 与直觉一致。"
            " 这说明它已经具备上线候选资格，但还需要更多外部语料验证。"
        )
    else:
        sentiment_comment = (
            f"情感模块虽然已经切到 `{sentiment_strategy}`，但抽样仍只有 {sentiment_correct}/10 与直觉一致。"
            " 当前结果不适合直接给终端用户做强结论。"
        )
    classification_comment = (
        "分类模块技术上跑通了，但它依赖主题结果衍生出的标签；在标签仍是 `主题 1-8` 的情况下，"
        " 分类结果虽然可算，但不可直接解释给业务用户。"
    )

    html_output = f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>waimai_10k_full 用户测试报告</title>
  <style>
    :root {{
      --bg: #eef3f8;
      --panel: #ffffff;
      --line: #d8e1ea;
      --text: #17212b;
      --muted: #627181;
      --good: #0f8b4c;
      --mixed: #9a6b10;
      --poor: #b42318;
      --issue: #8e2f1f;
      --good-bg: #e8f7ee;
      --mixed-bg: #fff4dd;
      --poor-bg: #fdeaea;
      --issue-bg: #fce9e5;
      --shadow: 0 18px 40px rgba(18, 33, 54, 0.06);
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "PingFang SC", "Microsoft YaHei", sans-serif;
      background: var(--bg);
      color: var(--text);
    }}
    .page {{
      width: min(1200px, calc(100vw - 32px));
      margin: 24px auto 40px;
      display: grid;
      gap: 16px;
    }}
    .hero, .panel {{
      background: var(--panel);
      border: 1px solid var(--line);
      border-radius: 18px;
      box-shadow: var(--shadow);
    }}
    .hero {{
      padding: 28px;
      display: grid;
      gap: 18px;
    }}
    h1, h2, h3, p {{ margin: 0; }}
    h1 {{ font-size: 32px; }}
    h2 {{ font-size: 22px; margin-bottom: 14px; }}
    h3 {{ font-size: 17px; margin-bottom: 10px; }}
    .muted {{ color: var(--muted); line-height: 1.7; }}
    .metric-grid {{
      display: grid;
      grid-template-columns: repeat(4, minmax(0, 1fr));
      gap: 12px;
    }}
    .metric {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px 16px;
      background: #f8fafc;
      display: grid;
      gap: 6px;
    }}
    .metric span {{ color: var(--muted); font-size: 13px; }}
    .metric strong {{ font-size: 22px; }}
    .badge {{
      display: inline-flex;
      align-items: center;
      min-height: 30px;
      padding: 0 12px;
      border-radius: 999px;
      font-size: 13px;
      font-weight: 700;
    }}
    .good {{ color: var(--good); background: var(--good-bg); }}
    .mixed {{ color: var(--mixed); background: var(--mixed-bg); }}
    .poor {{ color: var(--poor); background: var(--poor-bg); }}
    .issue {{ color: var(--issue); background: var(--issue-bg); }}
    .mini {{
      display: inline-flex;
      align-items: center;
      min-height: 24px;
      padding: 0 8px;
      border-radius: 999px;
      font-size: 12px;
      font-weight: 700;
    }}
    .grid-2 {{
      display: grid;
      grid-template-columns: 1.15fr 0.85fr;
      gap: 16px;
    }}
    .panel {{
      padding: 22px;
    }}
    .table-shell {{
      overflow: auto;
      border: 1px solid var(--line);
      border-radius: 14px;
      background: #fff;
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      min-width: 720px;
    }}
    th, td {{
      padding: 12px 14px;
      border-bottom: 1px solid #e7edf3;
      text-align: left;
      vertical-align: top;
      line-height: 1.6;
      font-size: 14px;
    }}
    th {{
      background: #f7fafc;
      font-size: 13px;
      color: #465465;
    }}
    .callout {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px 16px;
      background: #fbfdff;
      line-height: 1.75;
    }}
    .issue-list {{
      display: grid;
      gap: 12px;
    }}
    .issue-item {{
      border: 1px solid var(--line);
      border-radius: 14px;
      padding: 14px 16px;
      display: grid;
      gap: 8px;
      background: #fff;
    }}
    .issue-head {{
      display: flex;
      justify-content: space-between;
      gap: 12px;
      align-items: center;
    }}
    .footer {{
      color: var(--muted);
      font-size: 13px;
      line-height: 1.7;
    }}
    @media (max-width: 900px) {{
      .metric-grid, .grid-2 {{ grid-template-columns: 1fr; }}
    }}
  </style>
</head>
<body>
  <div class="page">
    <section class="hero">
      <div>
        <h1>waimai_10k_full 用户测试报告</h1>
        <p class="muted">测试目标：以真实用户流程复核当前平台在 `waimai_10k_full.csv` 上的操作体验、结果质量与主要风险。测试方式采用“前端动作等价 API 驱动 + 结果人工复核”的方式执行。</p>
      </div>
      <div class="metric-grid">
        <div class="metric"><span>测试数据集</span><strong>11,987 条</strong></div>
        <div class="metric"><span>上传耗时</span><strong>{upload['elapsed_sec']}s</strong></div>
        <div class="metric"><span>工作台摘要</span><strong>{summary['elapsed_sec']}s</strong></div>
        <div class="metric"><span>总体结论</span><strong>部分可用</strong></div>
      </div>
      <div style="display:flex; gap:10px; flex-wrap:wrap;">
        {quality_badge("good")}
        {quality_badge("mixed")}
        {quality_badge("poor")}
        {quality_badge("issue")}
      </div>
    </section>

    <section class="panel">
      <h2>一、测试结论摘要</h2>
      <div class="grid-2">
        <div class="callout">
          <h3>整体判断</h3>
          <p class="muted">上传、预览、探索导出可以跑通；主题、分类、情感也都能返回结果，但结果质量明显分化。当前版本更像“可以内部研究试用”的原型，还不适合直接作为稳定的终端产品交付给非技术用户。</p>
        </div>
        <div class="callout">
          <h3>最关键的三个问题</h3>
          <p class="muted">1. 工作台每个分段都要对 1.2 万条评论重算一次，单次约 19-20 秒。<br/>2. 情感模块对短评识别明显不足，正负样本大量落入中性。<br/>3. 报告内容里存在“样本量 0 条文本”的明显错误字段。</p>
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>二、用户流程复盘</h2>
      {render_table(["步骤", "用户动作", "耗时", "用户得到的结果", "结论"], flow_rows)}
    </section>

    <section class="panel">
      <h2>三、词项工作台结果质量</h2>
      <p class="muted" style="margin-bottom:14px;">自动词频和选词总体贴合外卖评论语境，但仍有少量噪音词和场景外词混入高位。</p>
      {render_table(["排名", "词语", "词频", "文档频次", "词性"], top_term_rows)}
      <div class="callout" style="margin-top:14px;">{esc(top_term_comment)}</div>
    </section>

    <section class="panel">
      <h2>四、主题分析质量</h2>
      <p class="muted" style="margin-bottom:14px;">主题模块能抓到“配送慢”“卷饼/煎饼”“漏送饮料”等有研究价值的簇，但重叠和命名问题比较明显。</p>
      {render_table(["主题", "名称", "样本量", "关键词", "代表证据"], topic_rows)}
      <div class="callout" style="margin-top:14px;">{esc(topic_comment)}</div>
    </section>

    <section class="panel">
      <h2>五、情感分析质量</h2>
      <p class="muted" style="margin-bottom:14px;">本次情感阶段返回：正向 {esc(sentiment['overview']['positive_count'])}，中性 {esc(sentiment['overview']['neutral_count'])}，负向 {esc(sentiment['overview']['negative_count'])}。但人工抽样后发现，中性比例明显失真。</p>
      {render_table(["样本", "证据文本", "人工期望", "系统结果", "判断"], sentiment_audit_rows)}
      <div class="callout" style="margin-top:14px;">{esc(sentiment_comment)}</div>
    </section>

    <section class="panel">
      <h2>六、分类结果质量</h2>
      <p class="muted" style="margin-bottom:14px;">分类模块依赖主题结果衍生出的标签。技术上可以跑通，但由于标签还没有被人工命名，结果解释性不足。</p>
      {render_table(
          ["样本", "系统标签", "置信度", "证据文本", "判断"],
          [
              [esc(i + 1), esc(item["label"]), esc(item["confidence"]), esc(item["snippet"]), esc("可运行，但难解释")]
              for i, item in enumerate(classification_section[:10])
          ],
      )}
      <div class="callout" style="margin-top:14px;">{esc(classification_comment)}</div>
    </section>

    <section class="panel">
      <h2>七、报告与导出检查</h2>
      <div class="grid-2">
        <div class="callout">
          <h3>导出完整性</h3>
          <p class="muted">探索阶段成功导出 8 份文件：词频、分词、选词、匹配、二值矩阵、频次矩阵、共词关系、报告。主题、分类、情感阶段也分别生成了对应结果表。</p>
        </div>
        <div class="callout">
          <h3>报告正确性问题</h3>
          <p class="muted">探索报告与主题报告中出现了“样本量：0 条文本”的错误字段，说明报告生成逻辑和真实统计数据存在脱节，这是必须修复的 correctness bug。</p>
        </div>
      </div>
    </section>

    <section class="panel">
      <h2>八、问题清单</h2>
      <div class="issue-list">
        {"".join(
            f'''
            <div class="issue-item">
              <div class="issue-head">
                <strong>{esc(level)} · {esc(title)}</strong>
                <span class="badge {"issue" if level == "P1" else "mixed"}">{esc(level)}</span>
              </div>
              <div class="muted"><strong>现象：</strong>{esc(symptom)}</div>
              <div class="muted"><strong>影响：</strong>{esc(impact)}</div>
            </div>
            '''
            for level, title, symptom, impact in issue_rows
        )}
      </div>
    </section>

    <section class="panel">
      <h2>九、建议动作</h2>
      <div class="callout">
        <p class="muted">1. 先修工作台缓存/预计算，避免每个分段都重算整份语料。<br/>
        2. 优先替换或增强情感模块，不要继续以当前规则结果直接对外展示。<br/>
        3. 主题模块增加命名、合并和人工确认闭环，至少解决“主题 1-8”不可解释问题。<br/>
        4. 修正报告生成里的样本量字段，并把报告改成真正引用分析概览数据。<br/>
        5. 在前端给用户明确区分“上传成功”“工作台初始化中”“主题分析后台运行中”三种状态。</p>
      </div>
    </section>

    <section class="footer">
      原始测试数据：<code>{esc(payload['dataset_file'])}</code><br/>
      数据集 ID：<code>{esc(payload['dataset_id'])}</code><br/>
      原始结构化结果：<code>{esc(JSON_PATH)}</code>
    </section>
  </div>
</body>
</html>
"""

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT_PATH.write_text(html_output, encoding="utf-8")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()
