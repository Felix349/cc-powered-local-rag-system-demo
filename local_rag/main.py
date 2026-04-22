"""
本地 RAG 系统主入口

两种使用方式：
  1. 命令行交互模式（直接运行）：
       python main.py
       python main.py --docs ./my_docs --rebuild

  2. 作为库引入（Python 代码中）：
       from main import RAGSystem
       rag = RAGSystem()
       rag.build("./my_docs")
       answer = rag.ask("这份报告的核心结论是什么？")
       print(answer.display())

系统流程：
  文档加载 → Chunking → Embedding → 向量存储
              ↓
  用户提问 → 路由判断 → 检索（语义/数值/Chain-of-Table）→ LLM 生成答案
"""

import argparse
import sys
from pathlib import Path


# ── 所有模块导入
from local_rag.loaders.document_loader import DocumentLoader
from local_rag.chunkers.chunker import ChunkingPipeline
from local_rag.embeddings.embedder import (
    TFIDFEmbedder, EmbeddingPipeline, create_embedder
)
from local_rag.vectorstore.vector_store import (
    NumpySQLiteVectorStore, create_vector_store
)
from local_rag.router.router import create_router
from local_rag.retriever.retriever import Retriever
from local_rag.llm.answer_generator import AnswerGenerator, Answer


# ─────────────────────────────────────────────
# RAGSystem：完整系统的封装
# ─────────────────────────────────────────────

class RAGSystem:
    """
    本地 RAG 系统的完整封装

    参数说明：
        storage_dir     : 向量库持久化目录
        kb_name         : 知识库名称（对应向量库文件前缀）
        embedding_backend: "tfidf"（开发/测试）或
                           "sentence_transformer"（生产，需安装库）或
                           "ollama"（生产，需 Ollama 运行）
        llm_model       : Ollama 模型名称，如 "qwen2.5:7b"
        ollama_url      : Ollama 服务地址
        chunk_size      : 每个 chunk 的最大字符数
        chunk_overlap   : 相邻 chunk 的重叠字符数
        top_k           : 语义检索返回的 chunk 数量
        stream          : 是否流式输出 LLM 答案
        router_backend  : "rule"、"llm" 或 "hybrid"
    """

    def __init__(
        self,
        storage_dir: str = "./vector_store",
        kb_name: str = "default",
        embedding_backend: str = "tfidf",
        llm_model: str = "qwen2.5:7b",
        ollama_url: str = "http://localhost:11434",
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        top_k: int = 5,
        stream: bool = True,
        router_backend: str = "rule",
    ):
        self.storage_dir = storage_dir
        self.kb_name = kb_name
        self.ollama_url = ollama_url
        self.llm_model = llm_model

        # ── 初始化各模块
        self._loader  = DocumentLoader()
        self._chunker = ChunkingPipeline(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )

        self._embedder = create_embedder(embedding_backend)

        self._store = create_vector_store(
            backend="numpy_sqlite",
            name=kb_name,
            storage_dir=storage_dir,
        )

        router_kwargs = {}
        if router_backend in ("llm", "hybrid"):
            router_kwargs = {"ollama_base_url": ollama_url, "llm_model": llm_model}
        self._router = create_router(backend=router_backend, **router_kwargs)

        self._retriever = Retriever(
            embedder=self._embedder,
            vector_store=self._store,
            router=self._router,
            ollama_base_url=ollama_url,
            llm_model=llm_model,
            top_k=top_k,
        )

        self._generator = AnswerGenerator(
            ollama_base_url=ollama_url,
            llm_model=llm_model,
            stream=stream,
        )

        print(f"\n{'='*60}")
        print(f"本地 RAG 系统初始化完成")
        print(f"  知识库: {kb_name}  ({self._store.count()} 条 chunk)")
        print(f"  Embedding: {embedding_backend}")
        print(f"  Router: {router_backend}")
        print(f"  LLM: {llm_model} @ {ollama_url}")
        print(f"{'='*60}\n")

    # ── 知识库构建

    def build(
        self,
        docs_dir: str,
        rebuild: bool = False,
        recursive: bool = True,
    ) -> int:
        """
        从目录加载文档，构建向量知识库

        Args:
            docs_dir  : 文档目录路径
            rebuild   : True = 清空已有数据重建；False = 增量追加
            recursive : 是否递归扫描子目录

        Returns:
            成功索引的 chunk 数量
        """
        if rebuild:
            print(f"清空知识库 [{self.kb_name}]，重新构建...")
            self._store.clear()

        print(f"\n── 步骤 1/4：加载文档（目录：{docs_dir}）")
        docs = self._loader.load_directory(docs_dir, recursive=recursive)
        if not docs:
            print("未找到任何支持的文档！")
            return 0

        print(f"\n── 步骤 2/4：文本分割")
        chunks = self._chunker.run(docs)

        print(f"\n── 步骤 3/4：向量化")
        # TF-IDF 需要先 fit
        if isinstance(self._embedder, TFIDFEmbedder):
            self._embedder.fit([c.content for c in chunks])
        emb_results = EmbeddingPipeline(self._embedder).run(chunks)

        print(f"\n── 步骤 4/4：存入向量库")
        n = self._store.insert(emb_results)
        self._store.persist()

        stats = self._store.stats()
        print(f"\n知识库构建完成！")
        print(f"  总 chunk 数: {stats['total_chunks']}")
        print(f"  向量维度:   {stats['embedding_dim']}")
        print(f"  类型分布:   {stats['type_breakdown']}")
        return n

    def add_file(self, file_path: str) -> int:
        """
        向知识库增量添加单个文件
        """
        print(f"\n增量添加文件: {file_path}")
        docs = self._loader.load(file_path)
        chunks = self._chunker.run(docs)
        if isinstance(self._embedder, TFIDFEmbedder) and not self._embedder._is_fitted:
            self._embedder.fit([c.content for c in chunks])
        emb_results = EmbeddingPipeline(self._embedder).run(chunks)
        n = self._store.insert(emb_results)
        self._store.persist()
        print(f"已添加 {n} 个 chunk")
        return n

    # ── 问答

    def ask(self, question: str) -> Answer:
        """
        提问，返回 Answer 对象
        """
        retrieval_result = self._retriever.retrieve(question)
        answer = self._generator.generate(retrieval_result)
        return answer

    def stats(self) -> dict:
        """返回知识库统计信息"""
        return self._store.stats()


# ─────────────────────────────────────────────
# 命令行入口
# ─────────────────────────────────────────────

def parse_args():
    parser = argparse.ArgumentParser(
        description="本地 RAG 系统",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            示例：
              # 首次构建知识库并进入交互模式
              python main.py --docs ./my_docs --rebuild

              # 直接进入交互模式（使用已有知识库）
              python main.py

              # 指定 embedding 模型（生产推荐）
              python main.py --docs ./my_docs --embedding sentence_transformer

              # 单次提问（非交互）
              python main.py --question "这份报告的核心结论是什么？"
        """),
    )
    parser.add_argument("--docs",      type=str, default=None,
                        help="文档目录路径，指定后自动构建知识库")
    parser.add_argument("--rebuild",   action="store_true",
                        help="清空知识库重新构建（与 --docs 配合使用）")
    parser.add_argument("--question",  type=str, default=None,
                        help="直接提问（非交互模式），回答后退出")
    parser.add_argument("--storage",   type=str, default="./vector_store",
                        help="向量库存储目录（默认 ./vector_store）")
    parser.add_argument("--kb",        type=str, default="default",
                        help="知识库名称（默认 default）")
    parser.add_argument("--embedding", type=str, default="tfidf",
                        choices=["tfidf", "sentence_transformer", "ollama"],
                        help="Embedding 后端（默认 tfidf）")
    parser.add_argument("--model",     type=str, default="qwen2.5:7b",
                        help="Ollama LLM 模型名（默认 qwen2.5:7b）")
    parser.add_argument("--ollama",    type=str, default="http://localhost:11434",
                        help="Ollama 服务地址")
    parser.add_argument("--router",    type=str, default="rule",
                        choices=["rule", "llm", "hybrid"],
                        help="路由器类型（默认 rule）")
    parser.add_argument("--no-stream", action="store_true",
                        help="关闭流式输出")
    return parser.parse_args()


def run_interactive(rag: RAGSystem):
    """交互式问答循环"""
    print("\n进入交互模式。输入问题后回车，输入 'q' 或 'exit' 退出。")
    print("特殊命令：")
    print("  /stats   → 显示知识库统计")
    print("  /add <路径> → 增量添加文件")
    print()

    while True:
        try:
            question = input("你的问题：").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n退出。")
            break

        if not question:
            continue

        if question.lower() in ("q", "exit", "quit", "退出"):
            print("退出。")
            break

        if question.startswith("/stats"):
            stats = rag.stats()
            print(f"\n知识库统计：")
            for k, v in stats.items():
                print(f"  {k}: {v}")
            print()
            continue

        if question.startswith("/add "):
            file_path = question[5:].strip()
            rag.add_file(file_path)
            continue

        answer = rag.ask(question)
        print(answer.display())


def main():
    import textwrap
    args = parse_args()

    rag = RAGSystem(
        storage_dir=args.storage,
        kb_name=args.kb,
        embedding_backend=args.embedding,
        llm_model=args.model,
        ollama_url=args.ollama,
        stream=not args.no_stream,
        router_backend=args.router,
    )

    # 如果指定了文档目录，先构建知识库
    if args.docs:
        rag.build(args.docs, rebuild=args.rebuild)

    # 单次提问模式
    if args.question:
        answer = rag.ask(args.question)
        print(answer.display())
        return

    # 检查知识库是否有内容
    if rag.stats()["total_chunks"] == 0:
        print("知识库为空！请先用 --docs 参数指定文档目录构建知识库。")
        print("示例：python main.py --docs ./my_docs --rebuild")
        return

    # 进入交互模式
    run_interactive(rag)


if __name__ == "__main__":
    import textwrap
    main()
