# Tieba Hotspot Collector（贴吧热点采集器）

把 Hot News Radar 中的贴吧热点继续展开为实际帖子、楼层和楼中楼，形成适合后续文本分析、主题聚类和舆情研究的结构化语料。

## 数据流

```text
Hot News Radar snapshot
        ↓
Tieba hottopic topic_id
        ↓
actual /p/<tid> threads
        ↓
thread floors + subposts
        ↓
SQLite + JSONL
        ↓
nlp-research-workbench
```

## 采集字段

### threads
- tid
- title
- forum
- author
- create_time
- reply_num
- agree_num
- collect_num
- source_url

### posts
- pid
- tid
- floor
- author
- create_time
- content
- agree_num
- subpost_num
- image URLs

### subposts
- spid
- tid / pid
- author
- create_time
- content
- image URLs

另外保存 discovery 和 failure 记录，便于审计和排查。

## 使用方式

### 单帖

```bash
python collectors/tieba/tieba_collector.py \
  --thread 'https://tieba.baidu.com/p/11063972585' \
  --max-pages 3
```

### 单个贴吧热点页

```bash
python collectors/tieba/tieba_collector.py \
  --hot-topic 'https://tieba.baidu.com/hottopic/browse/hottopic?topic_id=28366306' \
  --max-threads-per-topic 20 \
  --max-pages 3
```

### 从 Hot News Radar 快照发现贴吧热点

```bash
python collectors/tieba/tieba_collector.py \
  --snapshot 'https://raw.githubusercontent.com/LeilaoMi/hot-news-radar/reports/reports/2026-10-28/21-33.html' \
  --max-hot-topics 8 \
  --max-threads-per-topic 12 \
  --max-pages 3
```

### 每日自动模式

`run_daily.py` 会读取 Hot News Radar archive，选择北京时间今天之前最近日期的最后一份快照，然后采集其中前 8 个贴吧热点，每个热点最多发现 12 个帖子，每个帖子默认抓前 3 页并包含楼中楼。

```bash
python collectors/tieba/run_daily.py
```

这些限制是为了先保证长期运行稳定，不是能力上限。后续可以按社会议题优先级筛选，再扩大重点话题的采集深度。

## 输出

```text
collectors/tieba/data/tieba.sqlite3
collectors/tieba/data/export/threads.jsonl
collectors/tieba/data/export/posts.jsonl
collectors/tieba/data/export/subposts.jsonl
collectors/tieba/data/export/failures.jsonl
```

GitHub Actions 每次运行会把 data 目录打包成 Artifact，默认保留 30 天。

## GitHub Actions

工作流：`.github/workflows/tieba-hotspot-collector.yml`

- 每天北京时间 09:20 自动运行
- 支持 `workflow_dispatch` 手动运行
- PR / main 分支代码变更时执行语法检查和贴吧网络 smoke test
- smoke test 即使贴吧网络受限也不会阻止代码检查；正式 daily job 若采集失败会失败并保留错误 Artifact

## Docker

```bash
cd collectors/tieba
docker compose build
docker compose run --rm collector --thread 11063972585 --max-pages 1 --no-subposts
```

## 数据与合规边界

- 仅采集公开可访问页面/接口返回的数据。
- 不抓私信，不操作用户账号，不绕过验证码。
- 默认限速并重试，避免高并发冲击。
- 已删除、权限受限或审核中的帖子可能无法获得完整内容。
- 贴吧接口和字段可能变化，需要维护。

## 下一步建议

第一阶段先验证 GitHub Actions 网络是否能稳定访问贴吧；验证通过后再做：

1. 对 Hot News Radar 贴吧热点做社会议题优先级评分；
2. 只深挖高价值热点，减少娱乐/电竞类噪声；
3. 把 JSONL 直接导入本仓库现有 NLP 分析流水线；
4. 增加关键词、地域、就业/住房/消费等主题标签；
5. 需要长期连续增量时，把 SQLite/JSONL 同步到持久化对象存储或 Google Drive，而不是依赖 Actions 临时磁盘。
