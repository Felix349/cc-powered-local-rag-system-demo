"""
文本分割器模块（Chunker）

核心问题：向量模型有 token 上限（通常 512），长文档必须切成小块。
切得太短 → 每块缺少上下文，语义不完整。
切得太长 → 超出模型上限，或检索时引入太多噪音。

不同文档类型采用不同策略：
  - TEXT / DOCUMENT → 按字符滑动窗口切割，保留重叠区域
  - STRUCTURED      → 按行分组切割，保留表头，不破坏行的完整性
"""

from abc import ABC, abstractmethod
from typing import List

from local_rag.utils.document import Document, DocType


# ─────────────────────────────────────────────
# 基类
# ─────────────────────────────────────────────

class BaseChunker(ABC):

    @abstractmethod
    def split(self, document: Document) -> List[Document]:
        """把一个 Document 切成多个 chunk，每个 chunk 仍是 Document"""
        pass

    def _make_chunk(
        self,
        content: str,
        parent: Document,
        chunk_index: int,
        extra_metadata: dict = None,
    ) -> Document:
        """
        工厂方法：创建一个 chunk Document
        继承父文档的所有 metadata，追加 chunk 专属字段
        """
        metadata = {
            **parent.metadata,               # 继承来源信息
            "chunk_index": chunk_index,      # 第几块
            "chunk_content_preview": content[:50].replace("\n", " "),
        }
        if extra_metadata:
            metadata.update(extra_metadata)

        # 加文件名前缀，增强语义锚定，让检索更准确
        filename = parent.metadata.get("filename", "")
        prefix = f"【来源：{filename}】\n" if filename else ""
        enriched_content = prefix + content

        return Document(
            content=content,
            metadata=metadata,
            doc_type=parent.doc_type,
            raw_data=parent.raw_data,        # 结构化文件：每个 chunk 共享同一份 DataFrame
        )


# ─────────────────────────────────────────────
# 文本 Chunker：滑动窗口切割
# ─────────────────────────────────────────────

class TextChunker(BaseChunker):
    """
    滑动窗口切割，适用于 TEXT 和 DOCUMENT 类型

    原理：
      chunk_size    = 每块的最大字符数
      chunk_overlap = 相邻两块重叠的字符数

    重叠的作用：防止一个语义完整的句子恰好被切断在边界处。
    例如 "这项研究证明了..." 如果在句子中间切断，
    两边的 chunk 都拿不到完整语义，检索时都会失效。
    重叠区域让两边的 chunk 都包含这句话，至少有一个能被检索到。

    示意图（chunk_size=10, overlap=3）：
      原文:  [A B C D E F G H I J K L M]
      chunk0: [A B C D E F G H I J]
      chunk1:             [H I J K L M]  ← HI J 是重叠区
    """

    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 50):
        """
        Args:
            chunk_size    : 每块最大字符数。
                            中文场景推荐 300-600，英文推荐 500-1000。
            chunk_overlap : 相邻块重叠字符数，建议为 chunk_size 的 10%。
        """
        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap 必须小于 chunk_size")
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def split(self, document: Document) -> List[Document]:
        content = document.content.strip()
        if not content:
            return []

        # 先把 Markdown 表格整体提取，替换为占位符，防止被切断
        content, tables = self._extract_tables(content)

        if len(content) <= self.chunk_size:
            chunks = [self._make_chunk(
                self._restore_tables(content, tables), document, 0
            )]
            return chunks

        chunks = []
        start = 0
        chunk_index = 0
        step = self.chunk_size - self.chunk_overlap

        while start < len(content):
            end = start + self.chunk_size
            chunk_text = content[start:end]

            if end < len(content):
                chunk_text = self._snap_to_sentence_boundary(chunk_text)

            chunk_text = chunk_text.strip()
            if chunk_text:
                # 还原表格占位符
                restored = self._restore_tables(chunk_text, tables)
                chunks.append(self._make_chunk(
                    content=restored,
                    parent=document,
                    chunk_index=chunk_index,
                    extra_metadata={
                        "char_start": start,
                        "char_end": start + len(chunk_text),
                    }
                ))
                chunk_index += 1

            start += step

        return chunks

    def _extract_tables(self, content: str) -> tuple[str, dict]:
        """
        把 Markdown 表格替换成 __TABLE_0__ 这样的占位符
        返回 (替换后的文本, {占位符: 原始表格文本} 的字典)
        
        Markdown 表格特征：
        - 第一行是表头：| CPU | 最低配置 | 推荐配置 |
        - 第二行是分隔：| --- | ------- | ------- |
        - 后续行是数据：| 8 核 | 16 核 |
        """
        import re
        tables = {}
        
        # 匹配完整表格：连续的以 | 开头的行
        table_pattern = re.compile(
            r'(\|.+\|\n\|[-| :]+\|\n(?:\|.+\|\n?)*)',
            re.MULTILINE
        )
        
        counter = [0]
        def replace_table(m):
            key = f"__TABLE_{counter[0]}__"
            tables[key] = m.group(0)
            counter[0] += 1
            return key + "\n"
        
        result = table_pattern.sub(replace_table, content)
        return result, tables

    def _restore_tables(self, text: str, tables: dict) -> str:
        """把占位符还原成原始表格"""
        for key, table in tables.items():
            text = text.replace(key, table)
        return text

    def _snap_to_sentence_boundary(self, text: str) -> str:
        """
        在文本末尾向前找最近的合适断点，优先级：
        1. 表格结束位置（最后一个以 | 开头的行末尾）
        2. 段落边界（连续两个换行）
        3. 句子边界（。！？.!?）
        4. 找不到则保留原文
        """
        # 优先级 1：在最后一个表格行末尾截断
        lines = text.split("\n")
        last_table_line = -1
        for i in range(len(lines) - 1, -1, -1):
            if lines[i].strip().startswith("|"):
                last_table_line = i
                break
        if last_table_line > 0:
            # 确保截断点后面还有非表格内容（说明表格确实被截断了）
            remaining = "\n".join(lines[last_table_line + 1:]).strip()
            if remaining:
                return "\n".join(lines[:last_table_line + 1])

        # 优先级 2：段落边界（空行）
        max_lookback = max(1, len(text) // 4)
        para_pos = text.rfind("\n\n", len(text) - max_lookback)
        if para_pos > len(text) // 2:
            return text[:para_pos + 2]

        # 优先级 3：句子边界
        sentence_endings = "。！？.!?\n"
        for i in range(len(text) - 1, len(text) - max_lookback, -1):
            if text[i] in sentence_endings:
                return text[:i + 1]

        return text


# ─────────────────────────────────────────────
# 结构化 Chunker：按行分组切割
# ─────────────────────────────────────────────

class StructuredChunker(BaseChunker):
    """
    按行分组切割，适用于 STRUCTURED 类型（Excel、JSON 列表）

    与文本 Chunker 的关键区别：
    - 不能在行的中间切割（会破坏一条记录的完整性）
    - 每个 chunk 必须包含列头，否则向量模型不知道每列是什么含义
    - raw_data（完整 DataFrame）始终附在每个 chunk 上，供计算路径使用

    示意图（rows_per_chunk=3）：
      原表：[表头] [行1] [行2] [行3] [行4] [行5]
      chunk0: [表头] [行1] [行2] [行3]
      chunk1: [表头] [行4] [行5]
    """

    def __init__(self, rows_per_chunk: int = 30, max_chars: int = 600):
        """
        Args:
            rows_per_chunk : 每块包含的最大数据行数（不含表头）。
                             30 行 × 平均每行 50 字符 ≈ 1500 字符，
                             在大多数 embedding 模型的 512 token 范围内。
        """
        self.rows_per_chunk = rows_per_chunk
        self.max_chars = max_chars 

    def split(self, document: Document) -> List[Document]:
        content = document.content.strip()
        if not content:
            return []

        lines = content.split("\n")

        # 找到表头行（第一行通常是 "[Sheet: xxx]" 或 "[JSON...]"，第二行是列名）
        header_lines, data_lines = self._extract_header(lines)

        # 数据行少于阈值 且 总字符数在限制内，不切割
        if len(data_lines) <= self.rows_per_chunk and len(content) <= self.max_chars:
            return [self._make_chunk(content, document, 0)]

        chunks = []
        chunk_index = 0
        header_text = "\n".join(header_lines)

        # 按行分组，同时检查字符数上限
        current_batch = []
        for line in data_lines:
            current_batch.append(line)
            trial = header_text + "\n" + "\n".join(current_batch)
            # 超过行数上限 或 超过字符上限，切一刀
            if len(current_batch) >= self.rows_per_chunk or len(trial) >= self.max_chars:
                chunk_content = header_text + "\n" + "\n".join(current_batch)
                row_start = data_lines.index(current_batch[0]) + 1
                row_end   = data_lines.index(current_batch[-1]) + 1
                chunks.append(self._make_chunk(
                    content=chunk_content,
                    parent=document,
                    chunk_index=chunk_index,
                    extra_metadata={"row_start": row_start, "row_end": row_end},
                ))
                chunk_index += 1
                current_batch = []

        # 处理剩余行
        if current_batch:
            chunk_content = header_text + "\n" + "\n".join(current_batch)
            chunks.append(self._make_chunk(
                content=chunk_content,
                parent=document,
                chunk_index=chunk_index,
                extra_metadata={
                    "row_start": data_lines.index(current_batch[0]) + 1,
                    "row_end": len(data_lines),
                },
            ))

        return chunks

    def _extract_header(self, lines: List[str]):
        """
        将内容行分成"表头部分"和"数据行部分"

        约定的格式（来自 ExcelLoader / JSONLoader 的序列化输出）：
          [Sheet: 产品销售]      ← 第 0 行：描述行
          列: 产品, Q1销售额     ← 第 1 行：列名行
          行1: 产品=iPhone, ...  ← 第 2 行开始：数据行

        前两行归入 header，其余为数据行。
        如果格式不符合约定，第一行归 header，其余为数据行。
        """
        if len(lines) >= 2 and (
            lines[0].startswith("[") or lines[1].startswith("列:")
        ):
            return lines[:2], lines[2:]
        else:
            return lines[:1], lines[1:]


# ─────────────────────────────────────────────
# 统一入口：ChunkingPipeline
# ─────────────────────────────────────────────

class ChunkingPipeline:
    """
    根据 doc_type 自动选择合适的 Chunker

    使用示例：
        pipeline = ChunkingPipeline()
        chunks = pipeline.run(documents)
    """

    def __init__(
        self,
        chunk_size: int = 500,
        chunk_overlap: int = 50,
        rows_per_chunk: int = 30,
    ):
        self._text_chunker = TextChunker(chunk_size, chunk_overlap)
        self._structured_chunker = StructuredChunker(rows_per_chunk)

        # doc_type → chunker 的路由表
        self._routing = {
            DocType.TEXT: self._text_chunker,
            DocType.DOCUMENT: self._text_chunker,       # PDF/DOCX 同样用滑动窗口
            DocType.STRUCTURED: self._structured_chunker,
        }

    def run(self, documents: List[Document]) -> List[Document]:
        """处理一批 Document，返回所有 chunk 的列表"""
        all_chunks = []
        for doc in documents:
            chunker = self._routing[doc.doc_type]
            chunks = chunker.split(doc)
            all_chunks.extend(chunks)

        self._print_summary(documents, all_chunks)
        return all_chunks

    def _print_summary(self, docs: List[Document], chunks: List[Document]):
        print(f"Chunking 完成: {len(docs)} 个文档 → {len(chunks)} 个 chunk")
        # 按文件汇总
        from collections import defaultdict
        file_counts = defaultdict(int)
        for chunk in chunks:
            file_counts[chunk.metadata.get("filename", "unknown")] += 1
        for filename, count in file_counts.items():
            print(f"  {filename}: {count} 个 chunk")
