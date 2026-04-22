"""
统一的文档数据结构
所有加载器的输出、所有模块的输入都使用这个格式
"""

from dataclasses import dataclass, field
from typing import Any, Optional
from enum import Enum


class DocType(Enum):
    TEXT = "text"          # .txt, .md 等纯文本
    DOCUMENT = "document"  # .pdf, .docx 等富文本
    STRUCTURED = "structured"  # .xlsx, .json 等结构化数据


@dataclass
class Document:
    """
    系统中流通的基本数据单元

    Attributes:
        content     : 主要内容
                      - 文本类文件：原始字符串
                      - 结构化文件：序列化后的字符串（用于向量化）
        metadata    : 来源信息，检索后用于引用和溯源
        doc_type    : 文档类型，决定后续的 chunking 和路由策略
        raw_data    : 结构化文件专用，保存原始 DataFrame 或 dict
                      用于 Chain-of-Table / Text-to-Code 路径的精确计算
    """
    content: str
    metadata: dict = field(default_factory=dict)
    doc_type: DocType = DocType.TEXT
    raw_data: Optional[Any] = None  # DataFrame 或 dict，结构化文件专用

    def __repr__(self):
        preview = self.content[:80].replace("\n", " ")
        return (
            f"Document(type={self.doc_type.value}, "
            f"source={self.metadata.get('source', 'unknown')!r}, "
            f"preview={preview!r}...)"
        )
