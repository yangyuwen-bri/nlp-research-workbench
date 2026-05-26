# 后端算法调研与常见方案对照

## 1. 文档目的

这份文档用于帮助开发同学和 coding agent 理解：

- 当前后端每个功能模块实际在用什么算法
- 学术研究与常见应用软件一般会采用哪些算法路线
- 当前实现相对常见方案的优点、短板和适用边界
- 下一阶段如果要继续优化，优先级应如何安排

本文基于当前代码实现与公开工具文档整理，不涉及代码改动。

---

## 2. 当前后端能力总览

当前后端不是单一模型，而是一套“规则 + 传统机器学习 + embedding 语义 + LLM”混合流水线。

主线能力包括：

- 语料导入与预处理
- 中文分词与词频/选词
- 匹配表与 0/1 矩阵
- 共词关系
- 情感分析
- 文本分类
- 主题聚类
- 报告生成与导出

其中算法主要分布在：

- `backend/app/utils/text.py`
- `backend/app/services/local_models.py`
- `backend/app/services/semantic.py`
- `backend/app/services/topic_models.py`
- `backend/app/services/embeddings.py`
- `backend/app/services/llm.py`
- `backend/app/services/analyze.py`

---

## 3. 当前项目各模块实际使用的算法

### 3.1 语料导入与清洗

当前实现：

- `pandas.read_csv / read_excel`
- `json.loads` 逐行读取 JSONL
- `drop_duplicates()` 去重
- 启发式字段识别正文列
- 文件大小、行数、列数、单条文本长度校验

这部分不是核心 NLP 算法，更偏数据接入。

评价：

- 足够支撑 MVP 和轻量研究场景
- 还没有前台可控的数据清洗策略，例如停用词表管理、字段映射规则、子语料切分等

---

### 3.2 中文分词

当前实现：

- 使用 `jieba.posseg` 做中文分词与词性标注
- 过滤长度小于 2 的词
- 过滤停用词
- 过滤纯符号词

代码位置：

- `backend/app/utils/text.py`

当前路线本质：

- 词典+统计式中文分词
- 同时拿到词和词性

常见研究/软件路线：

- `jieba`：轻量、常见、部署成本低
- `HanLP`：面向生产、支持分词、词性、NER、依存、关键词短语提取等多任务  
  参考：HanLP 官方仓库说明其支持分词、词性标注、命名实体识别、依存句法分析、关键词短语提取、自动摘要、文本分类等能力。  
  来源：[HanLP GitHub](https://github.com/hankcs/HanLP)

对比判断：

- 当前 `jieba` 路线适合起步、快、稳定
- 如果继续面向中文研究或准生产环境，`HanLP` 比 `jieba` 更完整，也更符合“一个平台覆盖更多中文基础 NLP 任务”的方向

建议：

- 近期可保留 `jieba`
- 中期建议引入可切换分词后端，优先评估 `HanLP`

---

### 3.3 词频统计

当前实现：

- 用 `Counter` 统计总词频
- 再统计文档频次 `document_frequency`
- 同时保留词性信息

当前输出：

- `term_frequency`
- `document_frequency`
- `pos`

常见研究/软件路线：

- 几乎所有文本分析工具都支持词频表
- Voyant 的 `Cirrus`、`Corpus Terms`、`Word List` 都是围绕高频词展开  
  来源：[Voyant Tools Tutorial](https://docs.voyant-tools.org/docs/tutorial-tools_.html)

对比判断：

- 当前实现是标准且合理的基础能力
- 但还缺少“相对显著词/关键词”区分，当前更接近高频词，而不是严格意义上的关键词提取

---

### 3.4 选词

当前实现：

- 先取词频表
- 仅保留词性以 `n / v / a` 开头的词
- 取 Top-K 作为选词结果

这属于启发式选词，不是标准统计关键词算法。

常见研究/软件路线：

- 词频筛选
- TF-IDF 关键词
- Keyness/keyword list（和参考语料比较）
- PMI/搭配强度
- KeyBERT / embedding 关键词

对比判断：

- 当前方法优点是简单、可解释
- 缺点是容易保留泛词，难以识别“相对重要但不一定高频”的术语

建议：

- 近期保留现有方案
- 中期增加 `TF-IDF / class-based TF-IDF / KeyBERT-inspired` 作为可选关键词路线

---

### 3.5 匹配表

当前实现：

- 对每条文本，检查命中了哪些选中词
- 输出 `matched_terms`

这是规则匹配，不是统计学习算法。

价值：

- 对研究用户很重要，因为它能帮助他们从选词回看文本命中情况

常见软件对照：

- 文科工具里常通过词表、索引、上下文、过滤条件来做类似回看
- 这块当前实现是合理的，但还缺“回原文上下文联动”

---

### 3.6 0/1 矩阵

当前实现：

- 每篇文本对每个选中词做出现/未出现二值化

这是标准文档-词项二值特征矩阵。

常见研究/软件路线：

- 文档-词矩阵
- TF/TF-IDF 矩阵
- 二值矩阵

评价：

- 适合作为轻量分析和导出资产
- 若后续支持更多统计建模，建议同时支持 TF 或 TF-IDF 版本

---

### 3.7 共词关系

当前实现：

- 在 0/1 矩阵上，统计同一文本里共同出现的词对
- 输出 `source / target / weight`

本质算法：

- 共现计数

常见研究/软件路线：

- 共现网络
- Jaccard / PMI / Dice / Log-Likelihood 等加权方式
- KH Coder 公开介绍中明确支持 `Co-occurrence networks`  
  来源：[KH Coder 官网](https://khcoder.net/en/index.html)
- Voyant 也提供 `Collocates Graph`，其定义是“在接近位置上共同出现的关键词和词项的网络图”  
  来源：[Voyant Collocates Graph](https://docs.voyant-tools.org/docs/tutorial-collocatesgraph.html)

对比判断：

- 当前实现是“最基础的共现边计数”
- 对探索式使用已经有价值
- 但相较主流内容分析软件，缺少：
  - 窗口型共现
  - 显著性加权
  - 网络指标
  - 更强的可视化表达

建议：

- 保留边计数作为默认结果
- 后续增加 `window co-occurrence + PMI/Jaccard`

---

### 3.8 情感分析

当前实现是多级回退链。

#### 当前一级：本地 transformer 情感模型

- Hugging Face `pipeline("text-classification")`
- 模型默认值：`IDEA-CCNL/Erlangshen-Roberta-110M-Sentiment`
- 根据正负标签概率和阈值生成 `positive / neutral / negative`

#### 当前二级：参考集监督模型

- 训练数据：外卖、网购、酒店公开情感语料
- 特征：`TfidfVectorizer(analyzer="char_wb", ngram_range=(2,4))`
- 分类器：`ComplementNB`
- 额外支持 exact match 记忆

#### 当前三级：embedding 原型情感

- 文档向量 vs 正/中/负原型文本向量
- 相似度后做 softmax

#### 当前四级：规则词典情感

- 正负词典
- 否定词翻转
- 程度副词加权
- 句级累加得分

常见研究/软件路线：

- Lexicon-based sentiment：Orange 的情感组件明确支持 `Liu & Hu`、`VADER`、多语种词典、手工字典等 lexicon-based 路线  
  来源：[Orange Sentiment Analysis](https://orange3-text.readthedocs.io/en/stable/widgets/sentimentanalysis.html)
- 规则+词典：在数字人文和文本教学工具中很常见
- 监督分类：传统 `SVM / NB / Logistic Regression`
- 深度学习/预训练模型：BERT 类情感模型

对比判断：

- 当前路线的优点是容错好，失败时还有回退
- 相比常见软件，算法上不弱，甚至更“工程化”
- 但短板在于：
  - aspect sentiment 仍较浅
  - 缺少可配置词典和人工校正
  - 缺少“为什么判断成这样”的解释界面

建议：

- 保留当前四级链路
- 中期加入可配置情感词典与领域词典
- 如果继续强化中文场景，可评估方面级情感工具如 `PyABSA`

---

### 3.9 文本分类

当前实现同样是多级路线。

#### 当前一级：本地 zero-shot 分类

- Hugging Face `zero-shot-classification` pipeline
- NLI 式零样本分类
- 当前默认模型：`MoritzLaurer/mDeBERTa-v3-base-mnli-xnli`

Hugging Face 官方说明：

- zero-shot classification 可以直接对用户给定的 `candidate_labels` 做推断  
  来源：[Hugging Face Zero-Shot Classification](https://huggingface.co/tasks/zero-shot-classification)

#### 当前二级：公开 benchmark 参考分类器

- `TNEWS`：`char_wb TF-IDF + ComplementNB`
- `IFLYTEK`：`char+word TF-IDF + LinearSVC` 与 `char_wb TF-IDF + ComplementNB` 加权集成
- 如果文本完全命中参考集，支持 exact match 记忆
- 还叠加标签语义融合：文本与标签描述做相似度比较再混合

相关基础算法说明：

- `LinearSVC` 是线性 SVM 分类器，适合大样本线性文本分类  
  来源：[scikit-learn LinearSVC](https://scikit-learn.org/1.6/modules/generated/sklearn.svm.LinearSVC.html)
- `ComplementNB` 设计出来就是为了修正标准 Multinomial NB 的假设，特别适合不平衡数据集  
  来源：[scikit-learn ComplementNB](https://scikit-learn.org/stable/modules/generated/sklearn.naive_bayes.ComplementNB.html)
- `TfidfVectorizer` 用于将原始文档转换成 TF-IDF 特征矩阵  
  来源：[scikit-learn TfidfVectorizer](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html)

#### 当前三级：embedding 原型分类

- 文档向量与标签原型文本向量比较
- softmax 产出标签概率

#### 当前四级：规则关键词分类

- 预置标签关键词表
- 命中计数最高的标签胜出

常见研究/软件路线：

- 规则词表/字典匹配
- SVM、朴素贝叶斯、逻辑回归
- Zero-shot 分类
- 监督微调的预训练分类模型

对比判断：

- 当前分类路线相对成熟，尤其适合“用户先给一个标签名单，再让系统分类”
- 优点是兼顾：
  - 轻量传统模型
  - zero-shot
  - embedding 原型
  - exact memory
- 短板是：
  - 通用性仍依赖标签设计质量
  - 对开放标签空间的支持还不够自然
  - 缺少人工修订与 label profiling 前台交互

---

### 3.10 主题分析

当前实现：

#### 当前一级：BERTopic 优先

- 如果可用 embedding，则优先语义向量
- BERTopic 的标准流程本身是：embedding → 降维 → 聚类 → c-TF-IDF → topic representation  
  来源：[BERTopic 文档](https://bertopic.readthedocs.io/en/latest/)

但当前项目做了定制：

- 不用默认 HDBSCAN，改成 `KMeans`
- 向量器使用 CountVectorizer
- 主题关键词从 `get_topic()` 中拿，或自己做对比关键词提取

#### 当前二级：KMeans 回退

- `embedding + KMeans`
- 或 `TF-IDF + KMeans`

#### 当前三级：关键词提取

- 对簇内文本做 TF-IDF
- 或对比型关键词提取

常见研究/软件路线：

- `LDA / LSI / HDP`：Orange 的 Topic Modelling 小组件直接支持这三类，并说明 Topic Modelling wraps gensim’s LSI/LDA/HDP  
  来源：[Orange Topic Modelling](https://orange3-text.readthedocs.io/en/stable/widgets/topicmodelling-widget.html)
- `BERTopic`：embedding + c-TF-IDF 路线
- `Top2Vec`
- NMF / PLSA / STM 等
- Voyant 也支持 topic modeling，但更偏教学与探索  
  来源：[Voyant Topics](https://docs.voyant-tools.org/docs/tutorial-topics.html)

对比判断：

- 当前项目选 BERTopic 是偏现代的路线，语义表达上优于单纯 LDA
- 但当前实现并不是标准 BERTopic，而是“BERTopic 壳 + KMeans 聚类”的定制版
- 优点：
  - 稳定、好控主题数
  - 更容易和现有前端结果整合
- 风险：
  - 失去 HDBSCAN 自动发现噪声和簇结构的优势
  - 对细粒度主题和离群点处理较弱

建议：

- 如果目标是稳定交付，当前路线可接受
- 如果目标是学术/研究质量再提升，建议评估：
  - 标准 BERTopic（UMAP + HDBSCAN）
  - LDA/NMF 作为对照基线
  - 子语料对比主题

---

### 3.11 向量语义层

当前实现：

- 调 DashScope embedding 接口
- 批量并发请求
- 内存缓存 + SQLite 落盘缓存

使用位置：

- embedding 原型情感
- embedding 原型分类
- 主题聚类

评价：

- 这是当前系统里最有“现代语义检索/分类”味道的一层
- 工程实现很稳，且已经有缓存设计

常见软件路线：

- 传统工具更多用 BoW/TF-IDF
- 新一代工具越来越多引入 embedding、semantic viewer、vector similarity  
  Orange Text Mining 已经把 `Document Embedding`、`Semantic Viewer` 作为标准组件之一  
  来源：[Orange Text Mining Docs](https://orange3-text.readthedocs.io/en/stable/widgets/topicmodelling-widget.html)

---

### 3.12 LLM 增强层

当前实现：

- 文档级情感/分类分析
- 主题命名
- 洞察卡片补充
- Markdown 报告生成

本质：

- 不直接替代全部分析
- 更偏“解释层”和“表达层”

常见研究/软件路线：

- 传统 DH / 内容分析软件大多不内置 LLM
- 新一代应用开始用 LLM 做：
  - 自动命名主题
  - 自动摘要
  - 自动标签建议

对比判断：

- 当前方向是合理的
- 关键要控制 hallucination，所以现在代码里已经限制“不得编造比例、条数、因果关系”，这点非常好

---

## 4. 常见研究工具/软件的算法取向

下面是几个有代表性的参考对象。

### 4.1 Voyant Tools

典型功能：

- 高频词
- 词频分布
- 关键词上下文 `Contexts / KWIC`
- Collocates / Collocates Graph
- Phrases
- Topics

特点：

- 以探索式阅读为中心
- 更强调“词、上下文、共现、趋势”
- 强交互，不强调重模型

来源：

- [Voyant Tools Getting Started](https://docs.voyant-tools.org/docs/tutorial-start.html)
- [Voyant Tools / Tools](https://docs.voyant-tools.org/docs/tutorial-tools_.html)
- [Voyant Contexts](https://docs.voyant-tools.org/docs/tutorial-contexts.html)
- [Voyant Collocates Graph](https://docs.voyant-tools.org/docs/tutorial-collocatesgraph.html)

对本项目启示：

- 上下文联读非常重要
- 共现/趋势/短语不应只停留在表格层

### 4.2 KH Coder

典型功能：

- 定量内容分析
- 文本挖掘
- 共现网络
- Jaccard
- 分类与统计

特点：

- 面向内容分析研究场景
- 很强调词语关系结构和统计分析

来源：

- [KH Coder 官网](https://khcoder.net/en/index.html)

对本项目启示：

- 当前共词功能是对的，但还偏基础
- 可以向“内容分析研究软件”更靠拢，而不只是导出结果表

### 4.3 Orange3 Text Mining

典型功能：

- Topic Modeling：LSI / LDA / HDP
- Sentiment Analysis：lexicon-based 路线
- Word Cloud
- Concordance
- Collocations
- Corpus to Network
- Document Embedding

来源：

- [Orange Topic Modelling](https://orange3-text.readthedocs.io/en/stable/widgets/topicmodelling-widget.html)
- [Orange Sentiment Analysis](https://orange3-text.readthedocs.io/en/stable/widgets/sentimentanalysis.html)

对本项目启示：

- 当前项目比 Orange 更偏“混合算法工程化”
- 但 Orange 在研究工作流可视化和组件化上更成熟

### 4.4 HanLP

典型功能：

- 分词
- 词性
- 命名实体识别
- 依存分析
- 关键词/短语提取
- 摘要
- 文本分类

来源：

- [HanLP GitHub](https://github.com/hankcs/HanLP)

对本项目启示：

- 如果要强化中文基础 NLP，HanLP 是比 jieba 更完整的一站式选项

---

## 5. 当前项目与常见方案的总体对照

### 当前项目相对强的地方

- 有规则、传统 ML、embedding、LLM 的分层回退
- 中文短文本场景考虑较充分
- 分类链路设计相对成熟
- 导出资产完整
- 缓存和工程容错做得不错

### 当前项目相对弱的地方

- 基础 NLP 仍偏轻，分词后缺少更深语法/实体/短语层
- 关键词、短语、KWIC、趋势这些“研究高频动作”还缺
- 共词目前是基础边计数，不够研究化
- 主题算法路线现代，但可解释控制与标准对照还不够完整
- 缺少人工修订闭环

---

## 6. 适合当前产品阶段的算法策略建议

### 第一优先级：先补“研究动作”，不急着堆更复杂模型

优先补：

1. KWIC / 上下文回看
2. 短语/搭配
3. 子语料比较
4. 结果回原文联动

理由：

- 这些是研究用户高频动作
- 能明显提升可用性
- 不需要立刻大换后端

### 第二优先级：补强中文基础 NLP

建议评估：

- HanLP 分词/词性/NER/关键词短语

理由：

- 当前 `jieba` 足够起步，但上限有限
- 如果后续要做实体、关系、术语短语，会更自然

### 第三优先级：做主题分析双路线

建议：

- 保留当前 BERTopic/KMeans 路线
- 增加 LDA/NMF 作为对照模式

理由：

- 研究场景里用户常常需要对照而不是只信一种主题模型
- 也便于做更透明的解释

### 第四优先级：加强人工修订

建议：

- 允许人工改主题名
- 允许人工维护标签定义
- 允许人工修正分类/情感样本

理由：

- 应用最终价值不在“算法自动跑完”，而在“研究者能把结果变成可用证据”

---

## 7. 结论

当前后端算法并不弱，尤其在“分类、情感、主题、导出”这几条线上已经超过很多教学型文本分析工具。  
真正的问题不在于“缺算法”，而在于：

- 缺少一些研究用户高频动作
- 缺少更强的中文基础 NLP 能力
- 缺少把算法结果转化为研究证据的交互层

所以产品下一阶段最合理的方向，不是盲目引入更多模型，而是：

1. 先补研究动作
2. 再补中文基础 NLP
3. 最后再扩展更强主题与标签闭环
