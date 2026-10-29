# Tieba Hotspot Collector

一个独立的百度贴吧公开讨论采集器，面向 Hot News Radar 后续舆情深挖。

## 目标

把 Hot News Radar 中的贴吧热点继续展开为：

```text
Hot News Radar snapshot
        ↓
Tieba hottopic topic_id
        ↓
actual /p/<tid> threads
        ↓
thread floors
        ↓
subposts / 楼中楼
        ↓
SQLite + JSONL
```

它只负责采集与结构化保存，不负责 NLP 分析。分析工作交给单独的 `nlp-research-workbench` 或其他工具。

## 已实测通过

在 GitHub-hosted Ubuntu runner 上已验证：

- 单个贴吧帖子可抓取；
- 帖子楼层可抓取；
- 楼中楼可抓取；
- `hottopic` 聚合页可展开实际帖子；
- Hot News Radar HTML 快照可自动发现贴吧 hottopic，再展开真实帖子；
- SQLite / JSONL 可正常生成。

## 文件

```text
tieba_collector.py       核心采集器
run_daily.py             每日 Hot News Radar 自动入口
.github/workflows/
  collect.yml            GitHub Actions 定时任务
Dockerfile               可选容器运行
.gitignore
```

## 单帖

```bash
python tieba_collector.py \
  --thread 'https://tieba.baidu.com/p/11063972585' \
  --max-pages 3
```

## 单个贴吧热点页

```bash
python tieba_collector.py \
  --hot-topic 'https://tieba.baidu.com/hottopic/browse/hottopic?topic_id=28368227' \
  --max-threads-per-topic 5 \
  --max-pages 2
```

## Hot News Radar 快照

```bash
python tieba_collector.py \
  --snapshot 'https://raw.githubusercontent.com/LeilaoMi/hot-news-radar/reports/reports/2026-10-28/21-33.html' \
  --max-hot-topics 12 \
  --max-threads-per-topic 3 \
  --max-pages 2
```

## 每日模式

```bash
python run_daily.py
```

`run_daily.py` 会读取 Hot News Radar `reports` 分支的 `archive.html`，选择北京时间今天之前最近日期的最后一份快照，并调用采集器。

默认每日规模：

- 最多 12 个贴吧热点；
- 每个热点最多 3 个实际帖子；
- 每帖最多 2 页；
- 包含楼中楼；
- 请求间隔默认约 1.2 秒。

## 数据

数据库：

```text
data/tieba.sqlite3
```

每次运行的 JSONL：

```text
data/runs/YYYY-MM-DD/HH-MM/threads.jsonl
data/runs/YYYY-MM-DD/HH-MM/posts.jsonl
data/runs/YYYY-MM-DD/HH-MM/subposts.jsonl
data/runs/YYYY-MM-DD/HH-MM/discoveries.jsonl
data/runs/YYYY-MM-DD/HH-MM/failures.jsonl
data/runs/YYYY-MM-DD/HH-MM/run-meta.json
```

### threads

`tid / title / forum / author / create_time / reply_num / agree_num / collect_num / url / source_url`

### posts

`pid / tid / floor / author / create_time / content / agree_num / subpost_num / images_json`

### subposts

`spid / tid / pid / author / create_time / content / images_json`

## GitHub Actions

建议北京时间 09:30 执行，放在 Hot News Radar 09:00 日报之后。

工作流会：

1. 恢复最近一次 SQLite cache；
2. 运行 `python run_daily.py`；
3. 保存新的 SQLite cache；
4. 上传本次 JSONL + SQLite artifact，保留 30 天。

Actions cache 用于增量工作状态，不应视为永久归档。需要长期研究数据时，后续应同步到 Google Drive / S3 / NAS。

## 合规与稳定性

- 仅采集公开可访问页面和公开接口返回的数据；
- 不绕过验证码；
- 不抓私信；
- 不进行账号操作；
- 默认低速串行请求；
- 被删除、审核、权限限制的帖子可能无法完整获取；
- 百度接口字段可能变化，需要持续维护。

## 与文本分析工作台的关系

两个项目保持松耦合：

```text
Tieba Hotspot Collector
       ↓ JSONL / CSV
nlp-research-workbench
       ↓
主题 / 情感 / 分类 / 报告
```

分析工作台不承担爬虫调度和原始采集职责。
