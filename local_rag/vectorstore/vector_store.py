"""
向量数据库模块（VectorStore）

职责：
  1. 持久化存储 chunk 的文本内容、元数据、向量
  2. 给定查询向量，找出最相似的 Top-K 个 chunk

存储设计：双文件结构
  {name}.db    → SQLite，存 chunk 文本 + 元数据（结构化，便于过滤）
  {name}.npy   → NumPy，存向量矩阵（连续内存，适合批量点积运算）

为什么不直接把向量存进 SQLite？
  SQLite 是行存储，每次取向量都要反序列化。
  NumPy 的 .npy 是列存储，整个矩阵一次性 mmap 进内存，
  批量矩阵乘法（所有向量 vs 查询向量）在 numpy 里是一条指令。

架构：
  BaseVectorStore              ← 抽象基类
    └── NumpySQLiteVectorStore ← 本地实现（sqlite + numpy）
    └── ChromaDBVectorStore    ← 生产用占位（chromadb）
"""

import json
import pickle
import sqlite3
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, List, Optional, Tuple

import numpy as np

from local_rag.utils.document import Document, DocType


# ─────────────────────────────────────────────
# 检索结果数据结构
# ─────────────────────────────────────────────

class SearchResult:
    """
    单条检索结果

    Attributes:
        document    : 命中的 chunk Document
        score       : 相似度分数，范围 [0, 1]，越高越相关
        rank        : 排名，从 1 开始
    """

    def __init__(self, document: Document, score: float, rank: int):
        self.document = document
        self.score = score
        self.rank = rank

    def __repr__(self):
        src = self.document.metadata.get("filename", "unknown")
        preview = self.document.content[:60].replace("\n", " ")
        return (
            f"SearchResult(rank={self.rank}, score={self.score:.4f}, "
            f"source={src!r}, preview={preview!r}...)"
        )


# ─────────────────────────────────────────────
# 基类
# ─────────────────────────────────────────────

class BaseVectorStore(ABC):

    @abstractmethod
    def insert(self, chunks_and_vectors: List[Tuple[Document, np.ndarray]]) -> int:
        """
        批量插入 (chunk, vector) 对
        返回成功插入的条数
        """
        pass

    @abstractmethod
    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> List[SearchResult]:
        """
        向量相似度检索
        query_vector    : 查询向量，与存储向量维度相同
        top_k           : 返回最相似的前 K 条
        filter_metadata : 可选的元数据过滤条件，如 {"doc_type": "structured"}
        """
        pass

    @abstractmethod
    def count(self) -> int:
        """返回当前存储的 chunk 总数"""
        pass

    @abstractmethod
    def clear(self):
        """清空所有数据（谨慎使用）"""
        pass

    @abstractmethod
    def persist(self):
        """将内存中的数据持久化到磁盘"""
        pass


# ─────────────────────────────────────────────
# 核心实现：NumpySQLiteVectorStore
# ─────────────────────────────────────────────

class NumpySQLiteVectorStore(BaseVectorStore):
    """
    基于 SQLite + NumPy 的本地向量数据库

    文件结构：
      storage_dir/
        {name}.db      ← SQLite，存 id / content / metadata / doc_type
        {name}.npy     ← NumPy 矩阵，shape = (N, dim)，每行是一个向量
        {name}.meta    ← pickle，存维度 / 条目数 / 模型名等全局信息

    查询流程（暴力检索）：
      1. 取出整个向量矩阵 M，shape = (N, dim)
      2. 计算 M @ query_vector，得到 N 个点积（因向量已归一化，等于余弦相似度）
      3. argsort 取 Top-K 的下标
      4. 用下标去 SQLite 取对应的 chunk 数据

    时间复杂度：O(N × dim)，N=10万 dim=512 约 50ms（numpy 优化后）
    适合个人知识库（N < 100万）；更大规模请换 FAISS / Qdrant。
    """

    def __init__(
        self,
        name: str = "default",
        storage_dir: str = "./vector_store",
        embedding_dim: Optional[int] = None,
    ):
        """
        Args:
            name         : 知识库名称，对应磁盘上的文件前缀
            storage_dir  : 存储目录
            embedding_dim: 向量维度，首次插入时自动探测，之后必须一致
        """
        self.name = name
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        self._db_path = self.storage_dir / f"{name}.db"
        self._npy_path = self.storage_dir / f"{name}.npy"
        self._meta_path = self.storage_dir / f"{name}.meta"

        self._dim = embedding_dim         # 向量维度
        self._vectors: Optional[np.ndarray] = None  # 内存中的向量矩阵
        self._id_map: List[int] = []   # 向量下标 → SQLite id 的映射
        self._dirty = False               # 是否有未持久化的修改

        # 初始化 SQLite 和加载已有数据
        self._init_db()
        self._load_from_disk()

    # ── 初始化

    def _init_db(self):
        """建表，如果表已存在则跳过"""
        with self._get_conn() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS chunks (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    content     TEXT    NOT NULL,
                    doc_type    TEXT    NOT NULL,
                    metadata    TEXT    NOT NULL,   -- JSON 序列化
                    inserted_at REAL    NOT NULL    -- unix timestamp
                )
            """)
            # 常用查询字段加索引，加速 filter_metadata
            conn.execute("""
                CREATE INDEX IF NOT EXISTS idx_doc_type ON chunks(doc_type)
            """)

    def _get_conn(self) -> sqlite3.Connection:
        """获取 SQLite 连接，使用上下文管理器自动 commit / rollback"""
        conn = sqlite3.connect(str(self._db_path))
        conn.row_factory = sqlite3.Row   # 让查询结果支持列名访问
        return conn

    def _load_from_disk(self):
        if self._npy_path.exists() and self._meta_path.exists():
            self._vectors = np.load(str(self._npy_path))
            with open(self._meta_path, "rb") as f:
                meta = pickle.load(f)
            self._dim = meta.get("dim", self._dim)
            # 从数据库按插入顺序读取所有 id，重建映射
            with self._get_conn() as conn:
                rows = conn.execute(
                    "SELECT id FROM chunks ORDER BY id"
                ).fetchall()
            self._id_map = [row["id"] for row in rows]
            print(f"向量库 [{self.name}] 从磁盘加载: {len(self._id_map)} 条记录，向量维度={self._dim}")
        else:
            self._vectors = None
            self._id_map = []

    # ── 插入

    def insert(self, chunks_and_vectors: List[Tuple[Document, np.ndarray]]) -> int:
        """
        批量插入 chunk + 向量

        SQLite 和 NumPy 分开写入：
          - SQLite：批量 INSERT，用事务包裹提升速度
          - NumPy：追加到向量矩阵，延迟写磁盘（调用 persist() 时才写）
        """
        if not chunks_and_vectors:
            return 0

        # 验证维度一致性
        first_vec = chunks_and_vectors[0][1]
        if self._dim is None:
            self._dim = len(first_vec)
        elif len(first_vec) != self._dim:
            raise ValueError(
                f"向量维度不一致：期望 {self._dim}，实际 {len(first_vec)}。"
                f"请确认使用了相同的 embedding 模型。"
            )

        # 收集新向量
        new_vectors = []
        rows = []
        now = time.time()

        for chunk, vector in chunks_and_vectors:
            # metadata 里不存 raw_data（DataFrame 太大），只存可序列化的字段
            safe_metadata = {
                k: v for k, v in chunk.metadata.items()
                if isinstance(v, (str, int, float, bool, list, type(None)))
            }
            rows.append((
                chunk.content,
                chunk.doc_type.value,
                json.dumps(safe_metadata, ensure_ascii=False),
                now,
            ))
            new_vectors.append(vector.astype(np.float32))

        # 写入 SQLite
        with self._get_conn() as conn:
            cursor = conn.cursor()
            for row in rows:
                cursor.execute(
                    "INSERT INTO chunks (content, doc_type, metadata, inserted_at) "
                    "VALUES (?, ?, ?, ?)",
                    row,
                )
                self._id_map.append(cursor.lastrowid)   # ← 记录实际分配的 id

        # 更新内存向量矩阵
        new_matrix = np.stack(new_vectors, axis=0)   # shape: (batch, dim)
        if self._vectors is None:
            self._vectors = new_matrix
        else:
            self._vectors = np.vstack([self._vectors, new_matrix])

        self._dirty = True
        return len(rows)

    # ── 检索

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> List[SearchResult]:
        """
        Top-K 相似度检索

        核心计算：
          scores = M @ q
          M : (N, dim) 存储矩阵，每行已归一化
          q : (dim,)   查询向量，已归一化
          scores[i] = cosine_similarity(M[i], q)

        filter_metadata 用法：
          {"doc_type": "structured"}       → 只在结构化文档里搜
          {"filename": "report.pdf"}       → 只在某个文件里搜
        """
        if self._vectors is None or self._vectors.shape[0] == 0:
            return []

        q = query_vector.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # 矩阵点积，一次算出所有余弦相似度
        scores = self._vectors @ q           # shape: (N,)
        # [DEBUG] 打印分数统计信息，检查是否合理
        print(f"[DEBUG] scores 最大={scores.max():.4f} 最小={scores.min():.4f} 均值={scores.mean():.4f}")

        # 如果有元数据过滤，先获取符合条件的 id 集合
        valid_ids = self._get_filtered_ids(filter_metadata)
        # [DEBUG] 打印过滤结果，检查过滤条件是否生效
        print(f"[DEBUG] filter_metadata={filter_metadata}, valid_ids={valid_ids}")

        # 转成 (db_row_id, score) 并过滤
        # SQLite 的 id 从 1 开始，向量矩阵下标从 0 开始
        n = len(scores)
        candidates = []
        for idx in range(n):
            if idx >= len(self._id_map):
                continue
            db_id = self._id_map[idx]   # ← 用映射表取真实 id
            if valid_ids is not None and db_id not in valid_ids:
                continue
            candidates.append((db_id, float(scores[idx])))

        # 按相似度降序，取 Top-K
        candidates.sort(key=lambda x: x[1], reverse=True)
        # [DEBUG] 打印候选结果，检查过滤是否生效
        print(f"[DEBUG] top5候选: {[(id, round(s,4)) for id,s in candidates[:5]]}")
        top_candidates = candidates[:top_k]

        if not top_candidates:
            return []

        # 批量从 SQLite 取 chunk 数据
        ids = [c[0] for c in top_candidates]
        print(f"[DEBUG] 查询ids={ids}")

        placeholders = ",".join("?" * len(ids))
        with self._get_conn() as conn:
            rows = conn.execute(
                f"SELECT id, content, doc_type, metadata FROM chunks "
                f"WHERE id IN ({placeholders})",
                ids,
            ).fetchall()

        print(f"[DEBUG] SQLite返回行数={len(rows)}")

        # 同时查一下数据库里实际有什么
        with self._get_conn() as conn:
            all_ids = conn.execute("SELECT id FROM chunks LIMIT 10").fetchall()
        print(f"[DEBUG] 数据库前10个id={[r['id'] for r in all_ids]}")

        # 还原 Document 对象，按 score 排序
        row_map = {row["id"]: row for row in rows}
        results = []
        for rank, (db_id, score) in enumerate(top_candidates, start=1):
            if db_id not in row_map:
                continue
            row = row_map[db_id]
            metadata = json.loads(row["metadata"])
            doc = Document(
                content=row["content"],
                metadata=metadata,
                doc_type=DocType(row["doc_type"]),
            )
            results.append(SearchResult(document=doc, score=score, rank=rank))

        return results

    def _get_filtered_ids(self, filter_metadata: Optional[dict]) -> Optional[set]:
        """
        根据 filter_metadata 从 SQLite 查出符合条件的 id 集合
        返回 None 表示不过滤（全量搜索）
        """
        if not filter_metadata:
            return None

        conditions = []
        params = []

        # doc_type 是独立列，直接用 SQL 过滤（高效）
        if "doc_type" in filter_metadata:
            conditions.append("doc_type = ?")
            params.append(filter_metadata["doc_type"])

        # 其他字段在 JSON metadata 列里，用 json_extract 过滤
        for key, value in filter_metadata.items():
            if key == "doc_type":
                continue
            conditions.append(f"json_extract(metadata, '$.{key}') = ?")
            params.append(value)

        where = " AND ".join(conditions)
        with self._get_conn() as conn:
            rows = conn.execute(
                f"SELECT id FROM chunks WHERE {where}", params
            ).fetchall()

        return {row["id"] for row in rows}

    # ── 统计与管理

    def count(self) -> int:
        with self._get_conn() as conn:
            row = conn.execute("SELECT COUNT(*) as n FROM chunks").fetchone()
            return row["n"]

    def clear(self):
        with self._get_conn() as conn:
            conn.execute("DELETE FROM chunks")
            conn.execute("DELETE FROM sqlite_sequence WHERE name='chunks'")  # ← 重置自增
        self._vectors = None
        self._id_map = []   # ← 清空映射
        self._dirty = False
        for path in [self._npy_path, self._meta_path]:
            if path.exists():
                path.unlink()
        print(f"向量库 [{self.name}] 已清空")  

    def persist(self):
        """将内存中的向量矩阵写入磁盘"""
        if not self._dirty:
            return
        if self._vectors is not None:
            np.save(str(self._npy_path), self._vectors)
            with open(self._meta_path, "wb") as f:
                pickle.dump({"dim": self._dim, "count": self.count()}, f)
        self._dirty = False
        print(f"向量库 [{self.name}] 持久化完成: {self.count()} 条，"
              f"文件={self._npy_path.name}")

    def stats(self) -> dict:
        """返回数据库统计信息"""
        with self._get_conn() as conn:
            type_rows = conn.execute(
                "SELECT doc_type, COUNT(*) as n FROM chunks GROUP BY doc_type"
            ).fetchall()

        type_counts = {row["doc_type"]: row["n"] for row in type_rows}
        return {
            "name": self.name,
            "total_chunks": self.count(),
            "embedding_dim": self._dim,
            "type_breakdown": type_counts,
            "vector_matrix_shape": (
                list(self._vectors.shape) if self._vectors is not None else None
            ),
            "db_path": str(self._db_path),
            "npy_path": str(self._npy_path),
        }


# ─────────────────────────────────────────────
# 占位：ChromaDBVectorStore（生产用）
# ─────────────────────────────────────────────

class ChromaDBVectorStore(BaseVectorStore):
    """
    基于 ChromaDB 的向量数据库（生产推荐）

    相比 NumpySQLiteVectorStore 的优势：
      - 支持 HNSW 近似最近邻索引，百万级数据检索 < 10ms
      - 内置持久化，无需手动调用 persist()
      - 支持更丰富的元数据过滤语法

    安装：pip install chromadb
    """

    def __init__(
        self,
        name: str = "default",
        storage_dir: str = "./vector_store",
    ):
        try:
            import chromadb
        except ImportError:
            raise ImportError("请安装 chromadb：pip install chromadb")

        self._client = chromadb.PersistentClient(path=storage_dir)
        self._collection = self._client.get_or_create_collection(
            name=name,
            metadata={"hnsw:space": "cosine"},
        )
        print(f"ChromaDB [{name}] 初始化，当前 {self._collection.count()} 条")

    def insert(self, chunks_and_vectors: List[Tuple[Document, np.ndarray]]) -> int:
        import uuid
        ids, embeddings, documents, metadatas = [], [], [], []
        for chunk, vector in chunks_and_vectors:
            ids.append(str(uuid.uuid4()))
            embeddings.append(vector.tolist())
            documents.append(chunk.content)
            safe_meta = {
                k: v for k, v in chunk.metadata.items()
                if isinstance(v, (str, int, float, bool))
            }
            safe_meta["doc_type"] = chunk.doc_type.value
            metadatas.append(safe_meta)

        self._collection.add(
            ids=ids,
            embeddings=embeddings,
            documents=documents,
            metadatas=metadatas,
        )
        return len(ids)

    def search(
        self,
        query_vector: np.ndarray,
        top_k: int = 5,
        filter_metadata: Optional[dict] = None,
    ) -> List[SearchResult]:
        where = filter_metadata or None
        result = self._collection.query(
            query_embeddings=[query_vector.tolist()],
            n_results=top_k,
            where=where,
        )
        results = []
        for rank, (doc_text, meta, distance) in enumerate(zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
        ), start=1):
            doc_type_str = meta.pop("doc_type", "text")
            doc = Document(
                content=doc_text,
                metadata=meta,
                doc_type=DocType(doc_type_str),
            )
            # ChromaDB 返回的是距离（越小越好），转成相似度
            score = 1.0 - distance
            results.append(SearchResult(document=doc, score=score, rank=rank))
        return results

    def count(self) -> int:
        return self._collection.count()

    def clear(self):
        self._collection.delete(where={})

    def persist(self):
        pass  # ChromaDB 自动持久化


# ─────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────

def create_vector_store(
    backend: str = "numpy_sqlite",
    name: str = "default",
    storage_dir: str = "./vector_store",
    **kwargs,
) -> BaseVectorStore:
    """
    根据 backend 创建对应的向量数据库

    使用示例：
        # 开发 / 本地使用
        store = create_vector_store("numpy_sqlite", name="my_kb")

        # 生产使用（安装 chromadb 后）
        store = create_vector_store("chromadb", name="my_kb")
    """
    registry = {
        "numpy_sqlite": NumpySQLiteVectorStore,
        "chromadb": ChromaDBVectorStore,
    }
    if backend not in registry:
        raise ValueError(f"未知 backend: {backend!r}，可选: {list(registry.keys())}")

    return registry[backend](name=name, storage_dir=storage_dir, **kwargs)
