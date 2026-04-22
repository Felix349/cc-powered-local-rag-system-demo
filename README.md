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

1. **Embedding 模型**：将 `embedding_backend` 从 `"tfidf"` 改为 `"sentence_transformer"`
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

