# 文本分析平台

一个中文优先的文本分析平台，面向语料上传、分词词频、共词分析、情感分析、主题聚类、标签分类、洞察报告和结果导出等产品化场景。

## 发布仓库内容

```text
backend/   FastAPI API、分析流水线、存储与任务队列
frontend/  Next.js 文本分析工作台
infra/     PostgreSQL 与 Redis 基础设施模板
scripts/   Worker 与发布自检脚本
```

内部 benchmark、实验、设计 demo、临时数据、运行态数据和本地依赖目录不属于发布仓库内容。

## 核心能力

- 语料导入：支持 `CSV / XLSX / JSONL`
- 文本探索：中文分词、词频、选词、匹配表、0/1 矩阵、共词边
- 智能分析：情感分析、主题聚类、用户确认标签分类、主题命名
- 报告导出：`CSV / XLSX / Markdown`
- 产品工作台：上传语料、运行任务、查看表格/主题/情感/报告、下载结果
- 发布底座：开发环境支持 JSON + 进程内任务，发布环境支持 PostgreSQL + Redis worker

## 本地开发

启动后端：

```bash
cd /Users/gsdata/work/nlp_tool/backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

后端默认地址：`http://127.0.0.1:8000`

默认配置：

- `STORAGE_BACKEND=json`
- `TASK_QUEUE_BACKEND=inprocess`
- `CORS_ALLOW_ORIGINS=http://127.0.0.1:3000,http://localhost:3000`

如果本机系统代理影响依赖安装，可以使用：

```bash
cd /Users/gsdata/work/nlp_tool
./scripts/install_backend_ai_deps.sh
```

启动前端：

```bash
cd /Users/gsdata/work/nlp_tool/frontend
npm install --include=optional
npm run dev
```

前端默认地址：`http://127.0.0.1:3000`

如需自定义 API 地址：

```bash
NEXT_PUBLIC_API_BASE_URL=http://127.0.0.1:8000 npm run dev
```

## 发布部署

启动发布依赖：

```bash
cd /Users/gsdata/work/nlp_tool
docker compose -f infra/docker-compose.platform.yml up -d
```

发布环境变量模板在 [backend/.env.release.example](/Users/gsdata/work/nlp_tool/backend/.env.release.example)。

建议发布环境配置：

```bash
APP_ENV=production
STRICT_STARTUP_CHECKS=true
STORAGE_BACKEND=postgres
TASK_QUEUE_BACKEND=redis
TASK_QUEUE_NAME=analysis_jobs
DATABASE_URL=postgresql://nlp:nlp@127.0.0.1:5432/nlp_workbench
REDIS_URL=redis://127.0.0.1:6379/0
CORS_ALLOW_ORIGINS=https://你的前端域名
```

启动 worker：

```bash
cd /Users/gsdata/work/nlp_tool
STORAGE_BACKEND=postgres \
TASK_QUEUE_BACKEND=redis \
TASK_QUEUE_NAME=analysis_jobs \
DATABASE_URL=postgresql://nlp:nlp@127.0.0.1:5432/nlp_workbench \
REDIS_URL=redis://127.0.0.1:6379/0 \
APP_ENV=production \
STRICT_STARTUP_CHECKS=true \
PYTHONPATH=/Users/gsdata/work/nlp_tool/backend \
backend/.venv/bin/python scripts/run_analysis_worker.py
```

发布自检：

```bash
cd /Users/gsdata/work/nlp_tool
STORAGE_BACKEND=postgres \
TASK_QUEUE_BACKEND=redis \
TASK_QUEUE_NAME=analysis_jobs \
DATABASE_URL=postgresql://nlp:nlp@127.0.0.1:5432/nlp_workbench \
REDIS_URL=redis://127.0.0.1:6379/0 \
APP_ENV=production \
STRICT_STARTUP_CHECKS=true \
PYTHONPATH=/Users/gsdata/work/nlp_tool/backend \
backend/.venv/bin/python scripts/check_platform_readiness.py
```

生产环境启动时会执行强校验；如果仍然使用 `json` 存储、`inprocess` 队列，或 CORS/连接配置不达标，服务会直接阻止启动。也可以通过 [http://127.0.0.1:8000/api/platform/readiness](http://127.0.0.1:8000/api/platform/readiness) 查看当前运行环境的发布就绪状态。

## 配置阿里云百炼

后端按阿里云百炼 OpenAI 兼容接口接入。复制 [backend/.env.example](/Users/gsdata/work/nlp_tool/backend/.env.example) 为 `backend/.env`，填入：

```bash
DASHSCOPE_API_KEY=你的百炼API Key
DASHSCOPE_MODEL=qwen3.6-plus-2026-04-02
```

默认兼容端点：

```text
https://dashscope.aliyuncs.com/compatible-mode/v1
```

默认模型：

- 聊天模型：`qwen3.6-plus-2026-04-02`
- 向量模型：`text-embedding-v4`

## 验证

后端测试：

```bash
cd /Users/gsdata/work/nlp_tool
PYTHONPATH=/Users/gsdata/work/nlp_tool/backend backend/.venv/bin/python -m pytest backend/tests
```

前端生产构建：

```bash
cd /Users/gsdata/work/nlp_tool/frontend
npm run build
```
