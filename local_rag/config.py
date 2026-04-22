"""
RAG 系统统一配置文件
=====================================
所有模型、阈值、路径参数集中在这里管理。
换模型时只需修改这一个文件，其余代码不用动。

使用步骤：
  1. 在下方 ACTIVE_PROFILE 填入你想用的配置名称
  2. 在对应的 Profile 类里调整参数
  3. 重新运行系统（如果换了 embedding 模型，需要加 --rebuild）
"""

# ═══════════════════════════════════════════════════════════
#  ★  在这里选择当前使用的配置  ★
# ═══════════════════════════════════════════════════════════
ACTIVE_PROFILE = "bge_m3"   # 可选: "development", "nomic", "bge_m3", "sentence_transformer"
# ═══════════════════════════════════════════════════════════


# ───────────────────────────────────────────────────────────
# Profile 1：开发测试（无需 GPU，无需 Ollama）
# ───────────────────────────────────────────────────────────
class DevelopmentProfile:
    # ── 标识
    name = "development"
    description = "TF-IDF embedding，无需任何外部服务，用于快速开发和调试"

    # ── Embedding
    embedding_backend = "tfidf"    # 不需要 Ollama
    embedding_model   = None       # tfidf 不需要模型名
    # TF-IDF 的相似度分布较低，阈值设低一些
    min_score         = 0.30

    # ── LLM（答案生成）
    llm_model  = "qwen2.5:7b"
    ollama_url = "http://localhost:11434"

    # ── 检索参数
    top_k          = 5
    max_attempts   = 1             # 开发时不重试，节省时间
    router_backend = "rule"

    # ── Chunking
    chunk_size    = 500
    chunk_overlap = 50

    # ── 其他
    stream      = True
    kb_name     = "dev_kb"
    storage_dir = "./vector_store"


# ───────────────────────────────────────────────────────────
# Profile 2：nomic-embed-text（轻量生产，274MB）
# ───────────────────────────────────────────────────────────
class NomicProfile:
    name = "nomic"
    description = "nomic-embed-text，768维，速度快，适合英文为主的场景"

    # ── Embedding
    embedding_backend = "ollama"
    embedding_model   = "nomic-embed-text"
    # nomic 的相似度分布：高质量命中通常在 0.70~0.85
    min_score         = 0.72

    # ── LLM
    llm_model  = "qwen2.5:7b"
    ollama_url = "http://localhost:11434"

    # ── 检索参数
    top_k          = 5
    max_attempts   = 3
    router_backend = "rule"

    # ── Chunking
    # nomic 上下文约 512 token，中文约 600 字符安全
    chunk_size    = 500
    chunk_overlap = 50

    # ── 其他
    stream      = True
    kb_name     = "nomic_kb"
    storage_dir = "./vector_store"


# ───────────────────────────────────────────────────────────
# Profile 3：bge-m3（中文最优，1.2GB）★ 当前推荐
# ───────────────────────────────────────────────────────────
class BgeM3Profile:
    name = "bge_m3"
    description = "bge-m3，1024维，中英文最优，上下文8192token，当前推荐配置"

    # ── Embedding
    embedding_backend = "ollama"
    embedding_model   = "bge-m3"
    # bge-m3 的相似度分布整体偏低：高质量命中通常在 0.60~0.75
    min_score         = 0.65

    # ── LLM
    llm_model  = "qwen2.5:7b"
    ollama_url = "http://localhost:11434"

    # ── 检索参数
    top_k          = 5
    max_attempts   = 3
    router_backend = "rule"

    # ── Chunking
    # bge-m3 上下文 8192 token，chunk 可以更大
    chunk_size    = 800
    chunk_overlap = 80

    # ── 其他
    stream      = True
    kb_name     = "bge_kb"
    storage_dir = "./vector_store"


# ───────────────────────────────────────────────────────────
# Profile 4：sentence-transformers（本地，无需 Ollama）
# ───────────────────────────────────────────────────────────
class SentenceTransformerProfile:
    name = "sentence_transformer"
    description = "paraphrase-multilingual，本地运行，无需 Ollama，需安装 sentence-transformers"

    # ── Embedding
    embedding_backend = "sentence_transformer"
    embedding_model   = "paraphrase-multilingual-MiniLM-L12-v2"
    # sentence-transformers 的分布：高质量命中通常在 0.75~0.95
    min_score         = 0.75

    # ── LLM
    llm_model  = "qwen2.5:7b"
    ollama_url = "http://localhost:11434"

    # ── 检索参数
    top_k          = 5
    max_attempts   = 3
    router_backend = "rule"

    # ── Chunking
    chunk_size    = 500
    chunk_overlap = 50

    # ── 其他
    stream      = True
    kb_name     = "st_kb"
    storage_dir = "./vector_store"


# ═══════════════════════════════════════════════════════════
# 配置加载器（不需要修改）
# ═══════════════════════════════════════════════════════════

_PROFILES = {
    "development":        DevelopmentProfile,
    "nomic":              NomicProfile,
    "bge_m3":             BgeM3Profile,
    "sentence_transformer": SentenceTransformerProfile,
}


def get_config():
    """
    返回当前激活的配置 Profile 实例

    使用示例：
        from local_rag.config import get_config
        cfg = get_config()
        print(cfg.embedding_model)   # bge-m3
        print(cfg.min_score)         # 0.65
    """
    if ACTIVE_PROFILE not in _PROFILES:
        available = list(_PROFILES.keys())
        raise ValueError(
            f"未知的 ACTIVE_PROFILE: {ACTIVE_PROFILE!r}\n"
            f"可选值: {available}"
        )
    return _PROFILES[ACTIVE_PROFILE]()


def list_profiles():
    """打印所有可用配置"""
    print(f"\n{'─'*55}")
    print(f"  当前激活: {ACTIVE_PROFILE}")
    print(f"{'─'*55}")
    for name, cls in _PROFILES.items():
        marker = "★" if name == ACTIVE_PROFILE else " "
        print(f"  {marker} {name:<25} {cls.description}")
    print(f"{'─'*55}\n")
