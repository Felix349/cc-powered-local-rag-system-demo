"""
文档加载器模块
支持 .txt, .md, .pdf, .docx, .xlsx, .json 六种格式
所有加载器统一输出 List[Document]
"""

import json
import os
from abc import ABC, abstractmethod
from pathlib import Path
from typing import List, Optional

import pandas as pd

from local_rag.utils.document import Document, DocType


# ─────────────────────────────────────────────
# 基类：所有加载器继承此类
# ─────────────────────────────────────────────

class BaseLoader(ABC):
    """加载器基类，定义统一接口"""

    @abstractmethod
    def load(self, file_path: str) -> List[Document]:
        """
        加载文件，返回 Document 列表
        大多数格式返回单个 Document；
        多 sheet 的 Excel 可能返回多个。
        """
        pass

    def _base_metadata(self, file_path: str) -> dict:
        """生成通用元数据"""
        p = Path(file_path)
        return {
            "source": str(p.resolve()),
            "filename": p.name,
            "extension": p.suffix.lower(),
            "file_size_bytes": p.stat().st_size if p.exists() else 0,
        }


# ─────────────────────────────────────────────
# 纯文本加载器：.txt, .md
# ─────────────────────────────────────────────

class TextLoader(BaseLoader):
    """
    加载 .txt 和 .md 文件
    自动尝试 utf-8，失败后回退到 gbk（兼容中文 Windows 文件）
    """

    SUPPORTED = {".txt", ".md"}

    def load(self, file_path: str) -> List[Document]:
        path = Path(file_path)
        metadata = self._base_metadata(file_path)

        # 尝试多种编码
        for encoding in ("utf-8", "gbk", "latin-1"):
            try:
                content = path.read_text(encoding=encoding)
                metadata["encoding"] = encoding
                metadata["char_count"] = len(content)
                return [Document(
                    content=content,
                    metadata=metadata,
                    doc_type=DocType.TEXT,
                )]
            except UnicodeDecodeError:
                continue

        raise ValueError(f"无法解码文件 {file_path}，请检查文件编码")


# ─────────────────────────────────────────────
# PDF 加载器：.pdf
# ─────────────────────────────────────────────

class PDFLoader(BaseLoader):
    """
    加载 PDF 文件，使用 pdfminer.six 提取文本
    保留页码信息用于溯源引用
    """

    SUPPORTED = {".pdf"}

    def load(self, file_path: str) -> List[Document]:
        try:
            from pdfminer.high_level import extract_text_to_fp, extract_pages
            from pdfminer.layout import LAParams, LTTextContainer
            from io import StringIO
        except ImportError:
            raise ImportError("请安装 pdfminer.six: pip install pdfminer.six")

        metadata = self._base_metadata(file_path)
        pages_text = []

        # 逐页提取，记录页码
        for page_num, page_layout in enumerate(extract_pages(file_path, laparams=LAParams()), start=1):
            page_content = []
            for element in page_layout:
                if isinstance(element, LTTextContainer):
                    page_content.append(element.get_text())
            pages_text.append((page_num, "".join(page_content)))

        full_text = "\n".join(text for _, text in pages_text)
        metadata["page_count"] = len(pages_text)
        metadata["char_count"] = len(full_text)

        return [Document(
            content=full_text,
            metadata=metadata,
            doc_type=DocType.DOCUMENT,
        )]


# ─────────────────────────────────────────────
# Word 文档加载器：.docx
# ─────────────────────────────────────────────

class DocxLoader(BaseLoader):
    """
    加载 .docx 文件，使用 python-docx
    提取正文段落 + 表格内容
    """

    SUPPORTED = {".docx"}

    def load(self, file_path: str) -> List[Document]:
        try:
            from docx import Document as DocxDocument
        except ImportError:
            raise ImportError("请安装 python-docx: pip install python-docx")

        metadata = self._base_metadata(file_path)
        doc = DocxDocument(file_path)
        parts = []

        # 提取段落
        for para in doc.paragraphs:
            text = para.text.strip()
            if text:
                parts.append(text)

        # 提取表格（转为 Markdown 格式，便于阅读和向量化）
        for table_idx, table in enumerate(doc.tables):
            rows = []
            for row in table.rows:
                cells = [cell.text.strip() for cell in row.cells]
                rows.append(" | ".join(cells))
            if rows:
                parts.append(f"\n[表格 {table_idx + 1}]\n" + "\n".join(rows))

        content = "\n".join(parts)
        metadata["paragraph_count"] = len(doc.paragraphs)
        metadata["table_count"] = len(doc.tables)
        metadata["char_count"] = len(content)

        return [Document(
            content=content,
            metadata=metadata,
            doc_type=DocType.DOCUMENT,
        )]


# ─────────────────────────────────────────────
# Excel 加载器：.xlsx
# ─────────────────────────────────────────────

class ExcelLoader(BaseLoader):
    """
    加载 .xlsx 文件，每个 sheet 生成一个 Document
    content    : 序列化后的文本（用于向量化和语义检索）
    raw_data   : 原始 DataFrame（用于 Text-to-Code / Chain-of-Table 精确计算）

    序列化策略：将每行转成 "列名: 值, 列名: 值" 格式的自然语言描述
    """

    SUPPORTED = {".xlsx", ".xls"}

    def load(self, file_path: str) -> List[Document]:
        metadata = self._base_metadata(file_path)

        try:
            xl = pd.ExcelFile(file_path)
        except Exception as e:
            raise ValueError(f"无法读取 Excel 文件 {file_path}: {e}")

        documents = []
        for sheet_name in xl.sheet_names:
            df = xl.parse(sheet_name)

            # 清理：去掉全空行和全空列
            df = df.dropna(how="all").dropna(axis=1, how="all")
            df = df.reset_index(drop=True)

            # 序列化为自然语言（用于向量检索）
            content = self._serialize_dataframe(df, sheet_name)

            sheet_metadata = {
                **metadata,
                "sheet_name": sheet_name,
                "row_count": len(df),
                "col_count": len(df.columns),
                "columns": list(df.columns),
                "char_count": len(content),
            }

            documents.append(Document(
                content=content,
                metadata=sheet_metadata,
                doc_type=DocType.STRUCTURED,
                raw_data=df,  # 保留原始 DataFrame
            ))

        return documents

    def _serialize_dataframe(self, df: pd.DataFrame, sheet_name: str) -> str:
        """
        将 DataFrame 序列化为适合向量化的文本格式

        格式示例：
          [Sheet: 销售数据]
          列: 月份, 销售额, 增长率
          行1: 月份=1月, 销售额=12000, 增长率=5.2%
          行2: 月份=2月, 销售额=15000, 增长率=25.0%
        """
        lines = [f"[Sheet: {sheet_name}]"]
        lines.append(f"列: {', '.join(str(c) for c in df.columns)}")

        for idx, row in df.iterrows():
            row_parts = [f"{col}={val}" for col, val in row.items() if pd.notna(val)]
            lines.append(f"行{idx + 1}: {', '.join(row_parts)}")

        return "\n".join(lines)


# ─────────────────────────────────────────────
# JSON 加载器：.json
# ─────────────────────────────────────────────

class JSONLoader(BaseLoader):
    """
    加载 .json 文件
    支持两种常见结构：
    1. 列表格式 [{...}, {...}]  → 转为 DataFrame，走结构化路径
    2. 嵌套对象 {key: value}   → 序列化为文本，走文本路径
    """

    SUPPORTED = {".json"}

    def load(self, file_path: str) -> List[Document]:
        metadata = self._base_metadata(file_path)

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        # 情况 1：列表格式，可转为 DataFrame
        if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
            df = pd.DataFrame(data)
            content = self._serialize_list(data)
            metadata.update({
                "json_structure": "list_of_records",
                "record_count": len(data),
                "keys": list(data[0].keys()),
            })
            return [Document(
                content=content,
                metadata=metadata,
                doc_type=DocType.STRUCTURED,
                raw_data=df,
            )]

        # 情况 2：嵌套对象，序列化为文本
        else:
            content = self._serialize_nested(data)
            metadata["json_structure"] = "nested_object"
            return [Document(
                content=content,
                metadata=metadata,
                doc_type=DocType.TEXT,
                raw_data=data,
            )]

    def _serialize_list(self, data: list) -> str:
        lines = [f"[JSON 列表，共 {len(data)} 条记录]"]
        for i, record in enumerate(data):
            parts = [f"{k}={v}" for k, v in record.items()]
            lines.append(f"记录{i + 1}: {', '.join(parts)}")
        return "\n".join(lines)

    def _serialize_nested(self, data: dict, prefix: str = "", depth: int = 0) -> str:
        lines = []
        indent = "  " * depth
        if isinstance(data, dict):
            for key, value in data.items():
                full_key = f"{prefix}.{key}" if prefix else key
                if isinstance(value, (dict, list)):
                    lines.append(f"{indent}{key}:")
                    lines.append(self._serialize_nested(value, full_key, depth + 1))
                else:
                    lines.append(f"{indent}{key}: {value}")
        elif isinstance(data, list):
            for i, item in enumerate(data):
                lines.append(self._serialize_nested(item, f"{prefix}[{i}]", depth))
        else:
            lines.append(f"{indent}{data}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# 统一入口：DocumentLoader
# ─────────────────────────────────────────────

class DocumentLoader:
    """
    统一的文档加载入口
    根据文件扩展名自动选择对应的加载器

    使用示例：
        loader = DocumentLoader()
        docs = loader.load("report.pdf")
        docs = loader.load_directory("./my_docs")
    """

    def __init__(self):
        self._loaders: dict[str, BaseLoader] = {}
        # 注册所有加载器
        for loader_cls in [TextLoader, PDFLoader, DocxLoader, ExcelLoader, JSONLoader]:
            loader = loader_cls()
            for ext in loader.SUPPORTED:
                self._loaders[ext] = loader

    @property
    def supported_extensions(self) -> set:
        return set(self._loaders.keys())

    def load(self, file_path: str) -> List[Document]:
        """加载单个文件"""
        ext = Path(file_path).suffix.lower()
        if ext not in self._loaders:
            raise ValueError(
                f"不支持的文件格式 '{ext}'，"
                f"当前支持: {sorted(self.supported_extensions)}"
            )
        return self._loaders[ext].load(file_path)

    def load_directory(
        self,
        dir_path: str,
        recursive: bool = True,
        skip_errors: bool = True,
    ) -> List[Document]:
        """
        批量加载目录下所有支持的文件

        Args:
            dir_path    : 目录路径
            recursive   : 是否递归扫描子目录
            skip_errors : 遇到错误是否跳过（True=跳过并打印警告，False=抛出异常）
        """
        documents = []
        dir_path = Path(dir_path)
        pattern = "**/*" if recursive else "*"

        files = [
            f for f in dir_path.glob(pattern)
            if f.is_file() and f.suffix.lower() in self.supported_extensions
        ]

        print(f"发现 {len(files)} 个支持的文件")

        for file in files:
            try:
                docs = self.load(str(file))
                documents.extend(docs)
                print(f"  ✓ {file.name}  ({len(docs)} 个文档块)")
            except Exception as e:
                if skip_errors:
                    print(f"  ✗ {file.name}  错误: {e}")
                else:
                    raise

        print(f"加载完成，共 {len(documents)} 个 Document")
        return documents
