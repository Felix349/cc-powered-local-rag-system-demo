# 本地 RAG 系统

完全本地运行的检索增强生成系统，支持文本和结构化数据的混合查询。

## 项目结构

```
local_rag/
  utils/document.py              # 核心数据结构 Document
  loaders/document_loader.py     # 文档加载器（txt/md/pdf/docx/xlsx/json）
  chunkers/chunker.py            # 文本分割器（滑动窗口 + 按行分组）
  embeddings/embedder.py         # 向量化（TF-IDF / sentence-transformers / Ollama）
  vectorstore/vector_store.py    # 向量数据库（SQLite + NumPy）
  router/router.py               # 问题路由分类器
  retriever/retriever.py         # 三路检索引擎
  llm/answer_generator.py        # Prompt 构建 + LLM 答案生成
  main.py                        # 系统总入口
```

## 快速开始

### 安装依赖

```bash
pip install python-docx pandas openpyxl pdfminer.six numpy scikit-learn

# 生产推荐（语义 embedding）
pip install sentence-transformers

# 本地 LLM
# 安装 Ollama: https://ollama.com
ollama pull qwen2.5:7b
```

### 构建知识库并提问

```bash
cd local_rag_system
# 首次构建
python main.py --docs ./test_docs_rag --rebuild

# 再次运行（使用已有知识库）
python main.py

# 生产模式（真实语义 embedding + LLM）
python main.py --docs ./test_docs_rag --embedding ollama --rebuild

# 生产模式（真实语义 embedding + LLM）
python main.py --docs ./test_docs_rag --embedding sentence_transformer --rebuild
```

### 代码中使用

```python
from local_rag.main import RAGSystem

rag = RAGSystem(
    embedding_backend="sentence_transformer",  # 生产推荐
    llm_model="qwen2.5:7b",
)
rag.build("./my_docs", rebuild=True)

answer = rag.ask("销售额最高的季度是哪个？")
print(answer.display())
```

## 三条查询路径

| 路径        | 适合问题           | 示例                                          |
| ----------- | ------------------ | --------------------------------------------- |
| semantic    | 概念解释、定性描述 | "这个方法有什么优势？"                        |
| numeric     | 数值计算、聚合统计 | "平均销售额是多少？"                          |
| chain_table | 多步骤复杂推理     | "销售额超过10000的月份里，哪个产品占比最高？" |

## 生产部署注意事项

1. **Embedding 模型**：将 `embedding_backend` 从 `"tfidf"` 改为 `"bge-m3"`
2. **中文模型**：推荐 `paraphrase-multilingual-MiniLM-L12-v2`（384 维，支持中英文）
3. **LLM**：安装 Ollama 并拉取 `qwen2.5:7b`（中文效果最好）
4. **向量库**：数据量 > 50 万条时，将 `vector_store` backend 改为 `chromadb`



## 测试问题

| 文件格式             | 测试路径                     | 示例问题                                                     |
| -------------------- | ---------------------------- | ------------------------------------------------------------ |
| 01_公司简介.txt      | TXTSemantic                  | "星桥科技是哪年成立的？" / "公司的核心业务有哪些？"          |
| 02_产品技术文档.md   | MDSemantic                   | "BridgeCS 的意图识别准确率是多少？" / "系统的最低部署配置是什么？" |
| 03_年度研究报告.pdf  | PDFSemantic                  | "企业AI落地的三大障碍是什么？" / "金融行业的AI成熟度评分是多少？" |
| 04_人力资源报告.docx | DOCXSemantic + Numeric       | "各部门人员占比" / "算法工程师平均薪资是多少？"              |
| 05_财务销售数据.xlsx | XLSXNumeric + Chain-of-Table | "哪个季度毛利最高？" / "2024年营收总和是多少？" / "毛利率超过62%的季度里，哪个客户数最多？" |
| 06_竞品分析数据.json | JSONNumeric + Chain-of-Table | "哪家竞品的NPS最高？" / "响应延迟低于200ms的竞品中，市场份额最大的是谁？" |

## 部署参数

**Embedding 模型选择：**

| 模型                  | 典型分数范围 | 说明                            |
| --------------------- | ------------ | ------------------------------- |
| nomic-embed-text      | 0.5 ~ 0.9    | 分布较宽，高质量命中容易超 0.72 |
| bge-m3                | 0.3 ~ 0.75   | 整体偏低，0.65 已是优质命中     |
| sentence-transformers | 0.4 ~ 0.95   | 视具体模型而定                  |

**四个预设配置说明：**

| Profile 名             | embedding 模型          | 命中阈值 | 适用场景                   |
| ---------------------- | ----------------------- | -------- | -------------------------- |
| `development`          | TF-IDF                  | 0.30     | 开发调试，无需任何外部服务 |
| `nomic`                | nomic-embed-text        | 0.72     | 英文为主，速度优先         |
| `bge_m3`               | bge-m3                  | 0.65     | 中文为主，当前推荐 ★       |
| `sentence_transformer` | paraphrase-multilingual | 0.75     | 本地运行，无需 Ollama      |

每个 Profile 里还可以独立设置 `chunk_size`、`max_attempts`、`llm_model` 等，bge-m3 的 `chunk_size` 已经调大到 800（因为它支持 8192 token，可以放更多内容进每个 chunk）。

**换模型完整流程：**

python

```python
# 1. 打开 config.py，改顶部一行
ACTIVE_PROFILE = "nomic"   # 换成 nomic

# 2. 重新构建知识库（换了 embedding 模型必须 rebuild）
python -m local_rag.main --docs ./test_docs_rag --rebuild

# 3. 正常使用，阈值自动跟着 Profile 走
python -m local_rag.main
```

**对话历史记录和回答评价：**

正常提问后，每条回答下面会多一行提示：

```
[已记录 a3f8c21d04b1]  输入 /good 或 /bad 评价本条回答
```

立即评价：

```
你的问题：/good 回答完整准确
你的问题：/bad 来源文件错了，应该是产品文档
```

其他命令：

```
/log       → 显示统计摘要（好评率、平均分、低分问答列表）
/export    → 导出 logs/review_export.md，完整可读版本，适合人工审查
```

------

**生成的文件：**

```
logs/
  conversations.jsonl    ← 机器可读，每行一条 JSON
  conversations.md       ← 快速浏览用的表格
  review_export.md       ← /export 后生成，完整问答+来源+反馈
```

`conversations.jsonl` 是后续改进的核心资产——每条记录包含问题、回答、命中分数、来源文件、用户反馈，可以直接用来：分析哪类问题失败率高、筛选差评记录补充知识库、积累标注数据为后续微调做准备。
