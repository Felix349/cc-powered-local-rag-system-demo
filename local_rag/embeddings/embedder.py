"""
向量化模块（Embeddings）

职责：把文本字符串转成固定维度的浮点向量，供向量数据库存储和相似度计算。

架构设计：面向接口编程
  BaseEmbedder          ← 抽象基类，定义统一接口
    ├── SentenceTransformerEmbedder  ← 生产用，sentence-transformers 本地模型
    ├── OllamaEmbedder               ← 生产用，Ollama HTTP API
    └── TFIDFEmbedder                ← 开发/测试用，无需 GPU，纯 sklearn

切换方式：只改 EmbeddingConfig 里的 backend 字段，其余代码不动。
"""

import hashlib
import json
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional
import urllib.request

import numpy as np


# ─────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────

class EmbeddingConfig:
    """
    向量化配置，集中管理所有参数

    backend 选项：
      "sentence_transformer" → 本地模型，推荐生产使用
      "ollama"               → 通过 Ollama HTTP API 调用本地模型
      "tfidf"                → 轻量级，仅用于开发测试
    """

    # ── 通用参数
    backend: str = "sentence_transformer"
    batch_size: int = 32          # 批量处理，避免内存溢出
    show_progress: bool = True

    # ── sentence-transformers 参数
    # 中英文混合场景推荐：paraphrase-multilingual-MiniLM-L12-v2（384 维）
    # 纯英文场景推荐：all-MiniLM-L6-v2（384 维，更快）
    st_model_name: str = "paraphrase-multilingual-MiniLM-L12-v2"
    st_device: str = "cpu"        # 没有 GPU 时用 cpu，有 GPU 改成 "cuda"

    # ── Ollama 参数
    ollama_base_url: str = "http://localhost:11434"
    ollama_model: str = "nomic-embed-text"   # 专用 embedding 模型，768 维

    # ── TF-IDF 参数（测试用）
    tfidf_max_features: int = 512


# ─────────────────────────────────────────────
# 基类
# ─────────────────────────────────────────────

class BaseEmbedder(ABC):
    """
    所有 Embedder 的抽象基类

    子类只需实现两个方法：
      embed_one(text)   → 单条文本向量化
      embed_batch(texts) → 批量向量化（可以覆盖以利用批处理加速）
    """

    @abstractmethod
    def embed_one(self, text: str) -> np.ndarray:
        """把一段文本转成向量，返回 1D numpy 数组"""
        pass

    def embed_batch(self, texts: List[str], show_progress: bool = False) -> List[np.ndarray]:
        """
        批量向量化，默认逐条调用 embed_one
        子类可以覆盖此方法以使用原生批处理加速
        """
        vectors = []
        total = len(texts)
        for i, text in enumerate(texts):
            vectors.append(self.embed_one(text))
            if show_progress and (i + 1) % 10 == 0:
                print(f"  向量化进度: {i + 1}/{total}")
        return vectors

    @property
    @abstractmethod
    def dimension(self) -> int:
        """返回向量维度，向量数据库初始化时需要知道这个值"""
        pass

    @property
    @abstractmethod
    def model_name(self) -> str:
        """返回模型名称，存入 metadata 用于溯源"""
        pass


# ─────────────────────────────────────────────
# 生产用 Embedder 1：sentence-transformers
# ─────────────────────────────────────────────

class SentenceTransformerEmbedder(BaseEmbedder):
    """
    使用 sentence-transformers 库在本地运行 embedding 模型
    模型文件在首次使用时自动下载到 ~/.cache/huggingface/

    安装：pip install sentence-transformers
    推荐模型：
      paraphrase-multilingual-MiniLM-L12-v2  中英文，384 维，~120MB
      all-MiniLM-L6-v2                        纯英文，384 维，~80MB
    """

    def __init__(self, model_name: str = "paraphrase-multilingual-MiniLM-L12-v2",
                 device: str = "cpu"):
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError:
            raise ImportError(
                "请先安装 sentence-transformers:\n"
                "  pip install sentence-transformers"
            )

        print(f"加载 embedding 模型: {model_name} (device={device})")
        self._model = SentenceTransformer(model_name, device=device)
        self._model_name = model_name
        self._dim = self._model.get_sentence_embedding_dimension()
        print(f"模型加载完成，向量维度: {self._dim}")

    def embed_one(self, text: str) -> np.ndarray:
        return self._model.encode(text, convert_to_numpy=True)

    def embed_batch(self, texts: List[str], show_progress: bool = False) -> List[np.ndarray]:
        """使用 sentence-transformers 原生批处理，比逐条快很多"""
        vectors = self._model.encode(
            texts,
            convert_to_numpy=True,
            show_progress_bar=show_progress,
            batch_size=32,
        )
        return list(vectors)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return self._model_name


# ─────────────────────────────────────────────
# 生产用 Embedder 2：Ollama
# ─────────────────────────────────────────────

class OllamaEmbedder(BaseEmbedder):
    """
    通过 Ollama 的 HTTP API 调用本地 embedding 模型

    前置条件：
      1. 安装 Ollama: https://ollama.com
      2. 拉取模型:    ollama pull nomic-embed-text
      3. 启动服务:    ollama serve（通常安装后自动启动）

    nomic-embed-text 是专门为 embedding 设计的模型，768 维，
    比用 LLaMA 做 embedding 更快更准。
    """

    def __init__(
        self,
        model: str = "bge-m3",
        base_url: str = "http://localhost:11434",
    ):
        self._model = model
        self._base_url = base_url.rstrip("/")
        self._dim = None  # 第一次调用时自动探测

        # 检查 Ollama 是否在运行
        self._check_connection()

    def _check_connection(self):
        import urllib.request
        try:
            req = urllib.request.urlopen(f"{self._base_url}/api/tags", timeout=3)
            req.read()
        except Exception:
            raise ConnectionError(
                f"无法连接到 Ollama ({self._base_url})\n"
                f"请确认 Ollama 已启动：ollama serve"
            )

    def _call_api(self, text: str) -> list:
        """
        调用 /api/embed 接口，每次只传单条文本
        兼容所有 Ollama 版本，避免批量格式不兼容问题
        """
        payload = json.dumps({
            "model": self._model,
            "input": text,          # 始终传字符串，不传列表
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self._base_url}/api/embed",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=60) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            error_body = e.read().decode("utf-8")
            raise RuntimeError(
                f"Ollama 返回 {e.code} 错误\n"
                f"请求地址: {self._base_url}/api/embed\n"
                f"请求模型: {self._model}\n"
                f"错误详情: {error_body}"
            ) from e

        # ── 调试：打印实际结构
        emb = data["embeddings"][0]
        # print(f"[DEBUG] type={type(emb)}, len={len(emb)}")
        # print(f"[DEBUG] 第一个元素 type={type(emb[0])}, 值预览={str(emb[0])[:60]}")
        # ──

        return emb 

    def embed_one(self, text: str) -> np.ndarray:
        vector_raw = self._call_api(text)
        vector = np.array(vector_raw, dtype=np.float32)
        if self._dim is None:
            self._dim = len(vector)
        return vector

    def embed_batch(self, texts: List[str], show_progress: bool = False) -> List[np.ndarray]:
        """逐条调用，兼容所有 Ollama 版本"""
        results = []
        total = len(texts)
        for i, text in enumerate(texts):
            results.append(self.embed_one(text))
            if show_progress and (i + 1) % 10 == 0:
                print(f"  向量化进度: {i + 1}/{total}")
        return results

    @property
    def dimension(self) -> int:
        if self._dim is None:
            self.embed_one("test")
        return self._dim

    @property
    def model_name(self) -> str:
        return f"ollama/{self._model}"


# ─────────────────────────────────────────────
# 测试/开发用 Embedder：TF-IDF
# ─────────────────────────────────────────────

class TFIDFEmbedder(BaseEmbedder):
    """
    基于 TF-IDF 的轻量级向量化方案

    不需要 GPU，不需要下载模型，纯 sklearn，适合：
      - 开发阶段快速验证流程
      - 没有 GPU 的低配环境
      - 单元测试

    局限性：
      - 不理解语义（"苹果"和"水果"不相似）
      - 需要先 fit 一批文本（冷启动问题）
      - 向量维度由词汇表大小决定，不固定

    生产环境请换用 SentenceTransformerEmbedder 或 OllamaEmbedder。
    """

    def __init__(self, max_features: int = 512):
        from sklearn.feature_extraction.text import TfidfVectorizer
        self._vectorizer = TfidfVectorizer(
            max_features=max_features,
            analyzer="char_wb",    # 字符级 n-gram，对中文友好
            ngram_range=(2, 4),    # 2-4 字符的 n-gram
        )
        self._max_features = max_features
        self._is_fitted = False
        self._dim = max_features

    def fit(self, texts: List[str]):
        """
        用一批文本训练 TF-IDF 词汇表
        必须在 embed_one / embed_batch 之前调用
        """
        print(f"TF-IDF fit: {len(texts)} 条文本，max_features={self._max_features}")
        self._vectorizer.fit(texts)
        self._is_fitted = True
        # 实际词汇表可能小于 max_features
        self._dim = len(self._vectorizer.vocabulary_)
        print(f"词汇表大小: {self._dim}")

    def embed_one(self, text: str) -> np.ndarray:
        if not self._is_fitted:
            # 冷启动：用当前文本自己 fit（质量差，仅用于测试）
            self.fit([text])
        sparse = self._vectorizer.transform([text])
        return sparse.toarray()[0].astype(np.float32)

    def embed_batch(self, texts: List[str], show_progress: bool = False) -> List[np.ndarray]:
        if not self._is_fitted:
            self.fit(texts)
        sparse = self._vectorizer.transform(texts)
        dense = sparse.toarray().astype(np.float32)
        return list(dense)

    @property
    def dimension(self) -> int:
        return self._dim

    @property
    def model_name(self) -> str:
        return f"tfidf-char-ngram-{self._max_features}"


# ─────────────────────────────────────────────
# 向量工具函数
# ─────────────────────────────────────────────

def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """
    计算两个向量的余弦相似度，范围 [-1, 1]
    1 = 完全相同方向，0 = 正交（无关），-1 = 完全相反

    向量检索时用这个度量"语义距离"。
    """
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


def normalize(vector: np.ndarray) -> np.ndarray:
    """
    L2 归一化：把向量缩放到单位长度
    归一化后，余弦相似度等价于点积，计算更快
    ChromaDB 等向量库通常在存入时自动做归一化

    零向量（OOV 文本）：返回均匀分布的单位向量，而非全零。
    全零向量与任何向量的余弦相似度都是 0，
    均匀分布让它在检索时不会"消失"，而是得到一个低但非零的相似度。
    """
    norm = np.linalg.norm(vector)
    if norm == 0:
        # OOV 情况：返回均匀分布的单位向量
        uniform = np.ones_like(vector, dtype=np.float32)
        return uniform / np.linalg.norm(uniform)
    return vector / norm


# ─────────────────────────────────────────────
# 统一工厂函数
# ─────────────────────────────────────────────

def create_embedder(backend: str = "tfidf", **kwargs) -> BaseEmbedder:
    """
    根据 backend 字符串创建对应的 Embedder

    使用示例：
        # 开发测试
        embedder = create_embedder("tfidf")

        # 生产：sentence-transformers
        embedder = create_embedder("sentence_transformer",
                                   model_name="paraphrase-multilingual-MiniLM-L12-v2")

        # 生产：Ollama
        embedder = create_embedder("ollama", model="nomic-embed-text")
    """
    registry = {
        "sentence_transformer": SentenceTransformerEmbedder,
        "ollama": OllamaEmbedder,
        "tfidf": TFIDFEmbedder,
    }

    if backend not in registry:
        raise ValueError(
            f"未知 backend: {backend!r}，"
            f"可选: {list(registry.keys())}"
        )

    return registry[backend](**kwargs)


# ─────────────────────────────────────────────
# 向量化流水线：把 chunk 列表转成 (chunk, vector) 对
# ─────────────────────────────────────────────

class EmbeddingPipeline:
    """
    把 List[Document]（chunk）批量向量化

    返回格式：List[tuple[Document, np.ndarray]]
    每个 tuple = (chunk, 对应的向量)
    这个格式直接传给向量数据库的 insert 方法。

    使用示例：
        embedder = create_embedder("tfidf")
        pipeline = EmbeddingPipeline(embedder)
        results = pipeline.run(chunks)
    """

    def __init__(self, embedder: BaseEmbedder):
        self.embedder = embedder

    def run(
        self,
        chunks: List,                    # List[Document]
        batch_size: int = 32,
    ) -> List[tuple]:                    # List[tuple[Document, np.ndarray]]
        """
        分批向量化所有 chunk，返回 (chunk, vector) 列表
        分批处理的目的：避免一次性把所有文本塞进内存
        """
        if not chunks:
            return []

        print(f"\n向量化开始: {len(chunks)} 个 chunk，"
              f"模型={self.embedder.model_name}，"
              f"维度={self.embedder.dimension}")

        texts = [chunk.content for chunk in chunks]
        results = []
        total_batches = (len(texts) + batch_size - 1) // batch_size

        t0 = time.time()
        for batch_idx in range(0, len(texts), batch_size):
            batch_texts = texts[batch_idx: batch_idx + batch_size]
            batch_chunks = chunks[batch_idx: batch_idx + batch_size]
            batch_vectors = self.embedder.embed_batch(batch_texts)

            for chunk, vector in zip(batch_chunks, batch_vectors):
                # 写入 metadata：记录用了哪个模型、向量维度
                chunk.metadata["embedding_model"] = self.embedder.model_name
                chunk.metadata["embedding_dim"] = len(vector)
                results.append((chunk, normalize(vector)))

            current_batch = batch_idx // batch_size + 1
            print(f"  批次 {current_batch}/{total_batches} 完成 "
                  f"({min(batch_idx + batch_size, len(texts))}/{len(texts)})")

        elapsed = time.time() - t0
        print(f"向量化完成，耗时 {elapsed:.2f}s，"
              f"平均 {elapsed / len(chunks) * 1000:.1f}ms/chunk")

        return results
