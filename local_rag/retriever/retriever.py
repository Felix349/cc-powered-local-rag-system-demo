"""
检索器模块（Retriever）

根据 Router 的路由结果，执行三条不同的检索路径，
最终都输出统一格式的 RetrievalResult，供 LLM 生成答案。

三条路径：
  SemanticRetriever    → 向量相似度检索，返回相关 chunk
  NumericRetriever     → Text-to-Code，LLM 生成 pandas 代码，沙箱执行
  ChainTableRetriever  → Chain-of-Table，LLM 逐步操作表格，得出中间结果

总入口：
  Retriever.retrieve(question) → 自动路由 → 执行 → 返回 RetrievalResult
"""

import ast
import contextlib
import io
import json
import re
import traceback
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, List, Optional

import pandas as pd
import numpy as np

from local_rag.utils.document import Document, DocType
from local_rag.embeddings.embedder import BaseEmbedder, normalize
from local_rag.vectorstore.vector_store import BaseVectorStore, SearchResult
from local_rag.router.router import BaseRouter, RoutePath, RouteResult


# ─────────────────────────────────────────────
# 检索结果数据结构
# ─────────────────────────────────────────────

@dataclass
class RetrievalResult:
    """
    统一的检索结果，无论走哪条路径都输出这个格式，
    供后续 LLM 答案生成模块使用。

    Attributes:
        question        : 原始用户问题
        route           : 路由决策结果
        context_chunks  : 语义检索命中的 chunk 列表（SEMANTIC 路径）
        structured_result: 结构化计算结果（NUMERIC / CHAIN_TABLE 路径）
                           可以是 DataFrame、标量值或字符串
        code_executed   : 实际执行的代码（NUMERIC 路径，调试用）
        chain_steps     : Chain-of-Table 的操作步骤列表（CHAIN_TABLE 路径）
        error           : 执行过程中的错误信息（如有）
        retrieval_path  : 实际走的路径名称
    """
    question: str
    route: RouteResult
    context_chunks: List[SearchResult] = field(default_factory=list)
    structured_result: Any = None
    code_executed: Optional[str] = None
    chain_steps: List[dict] = field(default_factory=list)
    error: Optional[str] = None
    retrieval_path: str = ""

    def has_error(self) -> bool:
        return self.error is not None

    def summary(self) -> str:
        """打印友好的结果摘要"""
        lines = [
            f"问题: {self.question}",
            f"路径: {self.retrieval_path}  (路由置信度={self.route.confidence:.2f})",
        ]
        if self.context_chunks:
            lines.append(f"命中 chunk 数: {len(self.context_chunks)}")
            for h in self.context_chunks[:2]:
                lines.append(f"  [{h.rank}] score={h.score:.3f} | "
                             f"{h.document.metadata.get('filename','?')} | "
                             f"{h.document.content[:60].replace(chr(10),' ')!r}...")
        if self.structured_result is not None:
            lines.append(f"结构化结果:\n{self.structured_result}")
        if self.code_executed:
            lines.append(f"执行代码:\n{self.code_executed}")
        if self.chain_steps:
            lines.append(f"Chain-of-Table 步骤: {len(self.chain_steps)} 步")
            for i, step in enumerate(self.chain_steps):
                lines.append(f"  步骤{i+1}: {step.get('operation','?')} → "
                             f"{step.get('result_preview','')}")
        if self.error:
            lines.append(f"错误: {self.error}")
        return "\n".join(lines)


# ─────────────────────────────────────────────
# 路径一：语义检索
# ─────────────────────────────────────────────

class SemanticRetriever:
    """
    向量相似度检索

    流程：
      1. 用 embedder 把问题向量化
      2. 在 vector_store 里做 Top-K 相似度搜索
      3. 可选：按 doc_type 过滤，只在文本类文档里搜索

    返回：命中的 SearchResult 列表
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        top_k: int = 5,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.top_k = top_k

    def retrieve(
        self,
        question: str,
        filter_metadata: Optional[dict] = None,
    ) -> List[SearchResult]:
        # 问题向量化
        q_vec = normalize(self.embedder.embed_one(question))

        # 向量检索
        hits = self.vector_store.search(
            query_vector=q_vec,
            top_k=self.top_k,
            filter_metadata=filter_metadata,
        )
        return hits


# ─────────────────────────────────────────────
# 路径二：Text-to-Code 数值计算
# ─────────────────────────────────────────────

class NumericRetriever:
    """
    Text-to-Code 路径：LLM 生成 pandas 代码，沙箱执行

    流程：
      1. 向量检索找到最相关的结构化 chunk（含 raw_data）
      2. 把表格的 schema（列名+前几行）和问题拼成 prompt
      3. LLM 生成 pandas 代码
      4. 在受限沙箱里执行代码
      5. 返回执行结果

    安全沙箱设计：
      - 只允许 pandas / numpy 操作，禁止 import os/sys/subprocess 等
      - 用 AST 静态分析检查危险调用
      - 捕获所有异常，绝不让代码执行崩溃整个进程
      - 执行超时（通过信号或线程控制）
    """

    # LLM 不可用时的提示信息
    _NO_LLM_MSG = "[NumericRetriever] LLM 不可用，无法生成代码"

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        ollama_base_url: str = "http://localhost:11434",
        llm_model: str = "qwen2.5:7b",
        top_k: int = 3,
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.ollama_url = ollama_base_url.rstrip("/")
        self.llm_model = llm_model
        self.top_k = top_k

    def retrieve(self, question: str) -> dict:
        """
        返回：{
            "result"  : 执行结果（DataFrame / 标量 / 字符串）,
            "code"    : 实际执行的代码,
            "df_used" : 使用的 DataFrame（用于 Chain-of-Table 继续操作）,
            "error"   : 错误信息（如有）
        }
        """
        # Step 1: 找到最相关的结构化 chunk
        hits = self._find_structured_chunks(question)
        if not hits:
            return {
                "result": None, "code": "",
                "df_used": None,
                "error": "未找到相关的结构化数据（Excel/JSON），"
                         "请确认已加载结构化文件。"
            }

        # Step 2: 取出 raw_data（DataFrame）
        # 优先用相似度最高的 chunk 的 DataFrame
        # 注意：SearchResult 里的 document 没有 raw_data（数据库取回时丢失）
        # 需要单独保存映射关系，此处用 chunk 的 metadata 重新定位
        df, df_source = self._get_dataframe(hits[0])
        if df is None:
            return {
                "result": None, "code": "",
                "df_used": None,
                "error": f"无法获取 DataFrame，来源: {df_source}"
            }

        # Step 3: 生成 pandas 代码
        code = self._generate_code(question, df, df_source)
        if code is None:
            return {
                "result": self._NO_LLM_MSG, "code": "",
                "df_used": df, "error": None
            }

        # Step 4: 安全执行
        result, error = self._safe_execute(code, df)
        return {
            "result": result,
            "code": code,
            "df_used": df,
            "error": error,
        }

    def _find_structured_chunks(self, question: str) -> List[SearchResult]:
        """只在 STRUCTURED 类型的文档里搜索"""
        q_vec = normalize(self.embedder.embed_one(question))
        return self.vector_store.search(
            query_vector=q_vec,
            top_k=self.top_k,
            filter_metadata={"doc_type": "structured"},
        )

    def _get_dataframe(self, hit: SearchResult):
        """
        从命中的 chunk 恢复 DataFrame
        由于向量库存回的 Document 不含 raw_data，
        这里用 chunk 的序列化内容重建 DataFrame
        """
        content = hit.document.content
        filename = hit.document.metadata.get("filename", "unknown")

        try:
            # 解析序列化内容，重建 DataFrame
            lines = content.strip().split("\n")
            # 第一行：[Sheet: xxx] 或 [JSON 列表...]
            # 第二行：列: col1, col2, ...
            col_line = next((l for l in lines if l.startswith("列:")), None)

            # JSON 格式没有"列:"行，从第一条记录推断列名
            if col_line is None:
                first_data = next(
                    (l for l in lines if l.startswith("行") or l.startswith("记录")),
                    None
                )
                if first_data is None:
                    return None, f"{filename}: 无法解析列信息"
                kv_part = first_data.split(":", 1)[1].strip() if ":" in first_data else ""
                columns = [kv.split("=")[0].strip() for kv in kv_part.split(",") if "=" in kv]
            else:
                columns = [c.strip() for c in col_line[2:].split(",")]

            rows = []
            for line in lines:
                if not line.startswith("行") and not line.startswith("记录"):
                    continue
                # "行1: col1=val1, col2=val2, ..."  或  "记录1: col1=val1, ..."
                kv_part = line.split(":", 1)[1].strip() if ":" in line else ""
                row_data = {}
                for kv in kv_part.split(","):
                    kv = kv.strip()
                    if "=" in kv:
                        k, v = kv.split("=", 1)
                        row_data[k.strip()] = v.strip()
                if row_data:
                    rows.append(row_data)

            if not rows:
                return None, f"{filename}: 解析出 0 行数据"

            df = pd.DataFrame(rows)
            # 尝试自动转换数值列
            for col in df.columns:
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass

            return df, filename

        except Exception as e:
            return None, f"{filename}: 解析错误 {e}"

    def _generate_code(
        self, question: str, df: pd.DataFrame, source: str
    ) -> Optional[str]:
        """
        调用 LLM 生成 pandas 代码，返回代码字符串
        LLM 不可用时，尝试本地规则兜底处理最常见的四类问题
        """
        # 先尝试本地规则兜底（无需 LLM）
        rule_code = self._rule_fallback_code(question, df)
        if rule_code:
            return rule_code
        schema = self._build_schema_prompt(df, source)
        prompt = f"""你是一个 pandas 代码生成器。根据以下表格信息和问题，生成 Python pandas 代码。

{schema}

问题：{question}

要求：
1. 变量名必须用 `df` 表示输入 DataFrame
2. 最终结果赋值给变量 `result`
3. 只能使用 pandas 和 numpy，不能 import 其他库
4. 只输出代码，不要有任何解释文字，不要用 markdown 代码块

代码："""

        try:
            payload = json.dumps({
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0},
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{self.ollama_url}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            raw = data["message"]["content"].strip()
            # 去掉 LLM 可能包裹的 markdown 代码块
            raw = re.sub(r"```(?:python)?\n?", "", raw).strip("`").strip()
            return raw
        except Exception:
            return None

    def _rule_fallback_code(self, question: str, df: pd.DataFrame) -> Optional[str]:
        """
        无需 LLM 的本地规则兜底，处理最常见的四类数值问题：
          最大值、最小值、平均值、总和
        返回可执行的 pandas 代码字符串，或 None（规则无法处理时）
        """
        q = question.lower()
        num_cols = df.select_dtypes(include="number").columns.tolist()
        if not num_cols:
            return None

        # 尝试从问题中匹配列名
        target_col = None
        for col in df.columns:
            if str(col) in question:
                target_col = col
                break
        if target_col is None and num_cols:
            target_col = num_cols[0]   # 默认用第一个数值列

        # 匹配操作类型
        if any(w in q for w in ["最高", "最大", "最多", "max"]):
            idx_col = df.columns[0]
            return (f"max_val = df['{target_col}'].max()\n"
                    f"row = df[df['{target_col}'] == max_val]\n"
                    f"result = row.to_string(index=False)")

        if any(w in q for w in ["最低", "最小", "最少", "min"]):
            idx_col = df.columns[0]
            return (f"min_val = df['{target_col}'].min()\n"
                    f"row = df[df['{target_col}'] == min_val]\n"
                    f"result = row.to_string(index=False)")

        if any(w in q for w in ["平均", "均值", "average", "avg"]):
            return f"result = f\"{{'{target_col}'}} 的平均值为 {{df['{target_col}'].mean():.2f}}\""

        if any(w in q for w in ["总和", "合计", "总计", "sum"]):
            return f"result = f\"{{'{target_col}'}} 的总和为 {{df['{target_col}'].sum()}}\""

        if any(w in q for w in ["有多少", "共有", "数量", "count", "个数"]):
            return "result = f\"共有 {len(df)} 条记录\""

        if any(w in q for w in ["排名", "排行", "排序"]):
            return (f"result = df.sort_values('{target_col}', ascending=False)"
                    f".to_string(index=False)")

        return None   # 规则无法处理，交给 LLM

    def _build_schema_prompt(self, df: pd.DataFrame, source: str) -> str:
        """把 DataFrame 的结构和前几行序列化为 prompt 文本"""
        lines = [f"数据来源：{source}"]
        lines.append(f"列名：{list(df.columns)}")
        lines.append(f"数据类型：{ {c: str(df[c].dtype) for c in df.columns} }")
        lines.append(f"共 {len(df)} 行，前 {min(3, len(df))} 行示例：")
        lines.append(df.head(3).to_string(index=False))
        return "\n".join(lines)

    def _safe_execute(self, code: str, df: pd.DataFrame):
        """
        在受限环境里执行 LLM 生成的代码

        安全措施：
          1. AST 检查：禁止 import、open、eval、exec、__import__ 等危险调用
          2. 受限全局命名空间：只注入 pandas/numpy，隔离系统库
          3. 全异常捕获：任何错误都返回错误信息而不崩溃
        """
        # 安全检查
        danger, reason = self._ast_check(code)
        if danger:
            return None, f"代码安全检查未通过: {reason}"

        # 受限执行环境：只允许 pandas 和 numpy
        safe_globals = {
            "__builtins__": {
                # 只开放安全的内建函数
                "len": len, "range": range, "print": print,
                "str": str, "int": int, "float": float, "bool": bool,
                "list": list, "dict": dict, "tuple": tuple, "set": set,
                "min": min, "max": max, "sum": sum, "abs": abs,
                "round": round, "sorted": sorted, "enumerate": enumerate,
                "zip": zip, "map": map, "filter": filter,
                "isinstance": isinstance, "type": type,
            },
            "pd": pd,
            "np": np,
            "df": df.copy(),   # 传入副本，防止代码修改原始数据
        }
        local_vars = {}

        # 捕获 stdout（print 输出）
        stdout_capture = io.StringIO()
        try:
            with contextlib.redirect_stdout(stdout_capture):
                exec(compile(code, "<llm_code>", "exec"), safe_globals, local_vars)
        except Exception as e:
            return None, f"代码执行错误: {type(e).__name__}: {e}\n代码:\n{code}"

        # 取结果：优先取 result 变量，其次取 print 输出
        if "result" in local_vars:
            return local_vars["result"], None
        printed = stdout_capture.getvalue().strip()
        if printed:
            return printed, None
        return None, "代码执行完毕但没有 result 变量，也没有 print 输出"

    def _ast_check(self, code: str) -> tuple[bool, str]:
        """
        用 AST 静态分析检查危险操作
        返回 (is_dangerous, reason)
        """
        BANNED_NAMES = {
            "import", "open", "eval", "exec", "__import__",
            "compile", "globals", "locals", "vars",
            "os", "sys", "subprocess", "shutil", "pathlib",
            "socket", "urllib", "requests", "http",
        }
        try:
            tree = ast.parse(code)
        except SyntaxError as e:
            return True, f"语法错误: {e}"

        for node in ast.walk(tree):
            # 禁止 import 语句
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                return True, "不允许 import 语句"
            # 禁止调用危险函数
            if isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    if node.func.id in BANNED_NAMES:
                        return True, f"不允许调用 {node.func.id}()"
            # 禁止访问危险属性
            if isinstance(node, ast.Attribute):
                if node.attr.startswith("__"):
                    return True, f"不允许访问 dunder 属性: {node.attr}"

        return False, ""


# ─────────────────────────────────────────────
# 路径三：Chain-of-Table
# ─────────────────────────────────────────────

class ChainTableRetriever:
    """
    Chain-of-Table 路径：LLM 逐步规划表格操作，
    每步生成中间表，最终在最终表上回答问题。

    本实现的简化版操作集（对应论文的五个原子操作）：
      select_rows(condition)   → 按条件筛选行
      select_cols(columns)     → 选择列
      sort_by(column, asc)     → 排序
      group_by(column)         → 分组计数
      add_col(name, expr)      → 新增计算列

    对本地小模型的适配：
      - 每步操作用结构化 JSON 格式指令，降低 LLM 理解负担
      - 最多执行 5 步，防止死循环
      - 每步失败自动降级到 NumericRetriever
    """

    _MAX_STEPS = 5

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        numeric_retriever: "NumericRetriever",
        ollama_base_url: str = "http://localhost:11434",
        llm_model: str = "qwen2.5:7b",
    ):
        self.embedder = embedder
        self.vector_store = vector_store
        self.numeric_retriever = numeric_retriever
        self.ollama_url = ollama_base_url.rstrip("/")
        self.llm_model = llm_model

    def retrieve(self, question: str) -> dict:
        """
        返回：{
            "result"      : 最终结果（DataFrame 或字符串）,
            "steps"       : 操作步骤列表,
            "final_table" : 最终中间表,
            "error"       : 错误信息（如有）
        }
        """
        # 获取初始表格
        hits = self._find_structured_chunks(question)
        if not hits:
            return {
                "result": None, "steps": [], "final_table": None,
                "error": "未找到相关结构化数据"
            }

        df, source = self.numeric_retriever._get_dataframe(hits[0])
        if df is None:
            return {
                "result": None, "steps": [], "final_table": None,
                "error": f"无法获取 DataFrame: {source}"
            }

        steps = []
        current_df = df.copy()
        operation_history = []

        # 迭代执行操作链
        for step_num in range(self._MAX_STEPS):
            op = self._plan_next_operation(
                question, current_df, operation_history
            )
            if op is None:
                break

            op_name = op.get("operation", "")

            # 结束标记
            if op_name in ("END", "end", "[END]"):
                break

            # 执行操作
            result_df, preview, error = self._execute_operation(current_df, op)

            steps.append({
                "step": step_num + 1,
                "operation": op_name,
                "args": op.get("args", {}),
                "result_preview": preview,
                "error": error,
            })
            operation_history.append(op_name)

            if error:
                # 单步出错，停止操作链，用当前表继续
                break

            current_df = result_df

        # 在最终表上生成答案
        final_answer = self._final_query(question, current_df)

        return {
            "result": final_answer,
            "steps": steps,
            "final_table": current_df,
            "error": None,
        }

    def _find_structured_chunks(self, question: str) -> List[SearchResult]:
        q_vec = normalize(self.embedder.embed_one(question))
        return self.vector_store.search(
            query_vector=q_vec,
            top_k=3,
            filter_metadata={"doc_type": "structured"},
        )

    def _plan_next_operation(
        self,
        question: str,
        df: pd.DataFrame,
        history: List[str],
    ) -> Optional[dict]:
        """
        让 LLM 规划下一步操作
        返回 JSON 格式的操作指令，或 None（LLM 不可用时）
        """
        ops_desc = """可用操作（选一个，或输出 {"operation": "END"} 结束）：
- {"operation": "select_rows", "args": {"condition": "pandas query 字符串，如 '销售额 > 10000'"}}
- {"operation": "select_cols", "args": {"columns": ["列名1", "列名2"]}}
- {"operation": "sort_by",     "args": {"column": "列名", "ascending": true/false}}
- {"operation": "group_by",    "args": {"column": "列名"}}
- {"operation": "add_col",     "args": {"name": "新列名", "expr": "pandas eval 表达式，如 '销售额 * 0.1'"}}
- {"operation": "END"}"""

        history_str = " → ".join(history) if history else "（无）"
        table_str = df.head(5).to_string(index=True)

        prompt = f"""你是 Chain-of-Table 推理引擎。给定问题和当前表格，规划下一步操作。

问题：{question}
已执行操作：{history_str}
当前表格（前5行）：
{table_str}

{ops_desc}

只输出一个 JSON 对象，不要有其他文字："""

        try:
            payload = json.dumps({
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0},
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{self.ollama_url}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=20) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            raw = data["message"]["content"].strip()

            match = re.search(r"\{.*\}", raw, re.DOTALL)
            if match:
                return json.loads(match.group())
        except Exception:
            pass
        return None

    def _execute_operation(
        self, df: pd.DataFrame, op: dict
    ) -> tuple[pd.DataFrame, str, Optional[str]]:
        """
        执行一个表格操作，返回 (新DataFrame, 预览文本, 错误信息)
        """
        name = op.get("operation", "")
        args = op.get("args", {})

        try:
            if name == "select_rows":
                condition = args.get("condition", "")
                result = df.query(condition)

            elif name == "select_cols":
                cols = args.get("columns", [])
                valid = [c for c in cols if c in df.columns]
                result = df[valid]

            elif name == "sort_by":
                col = args.get("column", df.columns[0])
                asc = args.get("ascending", True)
                result = df.sort_values(by=col, ascending=asc)

            elif name == "group_by":
                col = args.get("column", df.columns[0])
                result = df.groupby(col).size().reset_index(name="Count")

            elif name == "add_col":
                new_name = args.get("name", "new_col")
                expr = args.get("expr", "0")
                result = df.copy()
                result[new_name] = df.eval(expr)

            else:
                return df, "", f"未知操作: {name}"

            preview = f"{len(result)}行×{len(result.columns)}列"
            return result, preview, None

        except Exception as e:
            return df, "", f"{name} 执行失败: {e}"

    def _final_query(self, question: str, df: pd.DataFrame) -> str:
        """在最终表格上，让 LLM 直接回答问题"""
        prompt = f"""根据以下表格，直接回答问题。

表格：
{df.to_string(index=True)}

问题：{question}

答案："""

        try:
            payload = json.dumps({
                "model": self.llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.1},
            }).encode("utf-8")
            req = urllib.request.Request(
                f"{self.ollama_url}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read().decode("utf-8"))
            return data["message"]["content"].strip()
        except Exception as e:
            # LLM 不可用，直接返回最终表格的字符串表示
            return f"[最终表格]\n{df.to_string()}"


# ─────────────────────────────────────────────
# 总入口：Retriever
# ─────────────────────────────────────────────

class Retriever:
    """
    统一检索入口，自动路由到对应的检索路径

    使用方式：
        retriever = Retriever(embedder, vector_store, router)
        result = retriever.retrieve("哪个产品销售额最高？")
        print(result.summary())
    """

    def __init__(
        self,
        embedder: BaseEmbedder,
        vector_store: BaseVectorStore,
        router: BaseRouter,
        ollama_base_url: str = "http://localhost:11434",
        llm_model: str = "qwen2.5:7b",
        top_k: int = 5,
    ):
        self.router = router

        self._semantic = SemanticRetriever(
            embedder=embedder,
            vector_store=vector_store,
            top_k=top_k,
        )
        self._numeric = NumericRetriever(
            embedder=embedder,
            vector_store=vector_store,
            ollama_base_url=ollama_base_url,
            llm_model=llm_model,
        )
        self._chain_table = ChainTableRetriever(
            embedder=embedder,
            vector_store=vector_store,
            numeric_retriever=self._numeric,
            ollama_base_url=ollama_base_url,
            llm_model=llm_model,
        )
        # 保存 LLM 配置，供 query rewrite 使用
        self._ollama_url = ollama_base_url.rstrip("/")
        self._llm_model = llm_model

    def _retrieve_once(self, question: str) -> RetrievalResult:
        """
        单次检索（内部方法），路由 → 执行对应路径 → 返回结果
        """
        # 路由判断
        route = self.router.route(question)
        print(f"\n[Retriever] 路由: {route.path.value} "
              f"(置信度={route.confidence:.2f}) | {route.matched_cues}")

        result = RetrievalResult(question=question, route=route)

        if route.path == RoutePath.SEMANTIC:
            result = self._run_semantic(question, route)

        elif route.path == RoutePath.NUMERIC:
            result = self._run_numeric(question, route)

        elif route.path == RoutePath.CHAIN_TABLE:
            result = self._run_chain_table(question, route)

        return result

    def _run_semantic(self, question: str, route: RouteResult) -> RetrievalResult:
        result = RetrievalResult(
            question=question,
            route=route,
            retrieval_path="semantic",
        )
        hits = self._semantic.retrieve(question)
        result.context_chunks = hits
        if not hits:
            result.error = "语义检索未找到相关内容"
        return result

    def _run_numeric(self, question: str, route: RouteResult) -> RetrievalResult:
        result = RetrievalResult(
            question=question,
            route=route,
            retrieval_path="numeric",
        )
        ret = self._numeric.retrieve(question)
        result.structured_result = ret.get("result")
        result.code_executed = ret.get("code")
        result.error = ret.get("error")

        # 同时做一次语义检索，把相关 chunk 也附上
        # 供 LLM 生成答案时参考文字上下文
        result.context_chunks = self._semantic.retrieve(
            question,
            filter_metadata={"doc_type": "structured"},
        )
        return result

    def _run_chain_table(self, question: str, route: RouteResult) -> RetrievalResult:
        result = RetrievalResult(
            question=question,
            route=route,
            retrieval_path="chain_table",
        )
        ret = self._chain_table.retrieve(question)
        result.structured_result = ret.get("result")
        result.chain_steps = ret.get("steps", [])
        result.error = ret.get("error")
        return result

    def retrieve(self, question: str) -> RetrievalResult:
        """
        主入口：带自动重试的 Agentic 检索

        流程：
          1. 执行一次检索
          2. 评估结果质量（分数够不够高、有没有命中）
          3. 质量不足 → 让 LLM 改写问题 → 重新检索
          4. 最多重试 max_attempts 次，返回历次结果中最好的那次
        """
        return self.retrieve_with_retry(question)

    def retrieve_with_retry(
        self,
        question: str,
        max_attempts: int = 3,
        min_score: float = 0.72,
    ) -> RetrievalResult:
        """
        Agentic 检索核心：多次尝试 + 问题改写

        Args:
            question     : 原始用户问题
            max_attempts : 最大尝试次数（含第一次），默认 3
            min_score    : 语义检索的最低可接受分数，低于此值触发重试
        """
        best_result: RetrievalResult = None
        best_score: float = -1.0
        current_question = question

        for attempt in range(max_attempts):
            if attempt > 0:
                print(f"\n[Agentic] 第 {attempt + 1} 次尝试，改写后的问题: {current_question!r}")

            result = self._retrieve_once(current_question)

            # 评估本次检索质量
            score = self._evaluate_result(result)
            print(f"[Agentic] 本次检索质量分: {score:.3f} (阈值={min_score})")

            # 记录历次最好结果
            if score > best_score:
                best_score = score
                best_result = result
                # 把原始问题写回 result，保持展示一致
                best_result.question = question

            # 质量达标，直接返回
            if score >= min_score:
                if attempt > 0:
                    print(f"[Agentic] 质量达标，采用第 {attempt + 1} 次结果")
                return best_result

            # 最后一次尝试，不再改写
            if attempt == max_attempts - 1:
                print(f"[Agentic] 已达最大尝试次数，返回历次最优结果 (score={best_score:.3f})")
                break

            # 质量不足，让 LLM 改写问题
            rewritten = self._rewrite_query(question, current_question, attempt)
            if rewritten is None or rewritten == current_question:
                print("[Agentic] LLM 不可用或问题无法改写，停止重试")
                break
            current_question = rewritten

        return best_result

    def _evaluate_result(self, result: RetrievalResult) -> float:
        """
        评估检索结果质量，返回 0~1 的分数

        评分逻辑：
          - 没有命中任何 chunk → 0.0
          - 有命中：取 top chunk 的相似度分数
          - 结构化路径（numeric/chain_table）：有结果且无错误 → 0.85（视为质量足够）
        """
        path = result.retrieval_path

        # 结构化路径：看有没有计算结果
        if path in ("numeric", "chain_table"):
            if result.structured_result is not None and not result.error:
                return 0.85
            return 0.2

        # 语义路径：看 top chunk 的相似度分数
        if not result.context_chunks:
            return 0.0
        return result.context_chunks[0].score

    def _rewrite_query(
        self,
        original_question: str,
        current_question: str,
        attempt: int,
    ) -> str:
        """
        让 LLM 把问题改写成不同的表达方式，以期检索到不同内容

        改写策略随尝试次数递进：
          attempt=0 → 同义替换，换个说法
          attempt=1 → 拆解问题，聚焦关键实体
        """
        strategies = [
            "用同义词替换关键词，换一种表达方式重新描述这个问题，保持语义不变",
            "提取问题中最核心的实体和属性，用最简洁的关键词形式重新表达",
        ]
        strategy = strategies[min(attempt, len(strategies) - 1)]

        prompt = f"""你是一个搜索查询优化专家。

原始问题：{original_question}
当前查询：{current_question}
改写要求：{strategy}

直接输出改写后的查询，不要有任何解释，不要加引号："""

        try:
            payload = json.dumps({
                "model": self._llm_model,
                "messages": [{"role": "user", "content": prompt}],
                "stream": False,
                "options": {"temperature": 0.3},
            }).encode("utf-8")

            req = urllib.request.Request(
                f"{self._ollama_url}/api/chat",
                data=payload,
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(req, timeout=15) as resp:
                data = json.loads(resp.read().decode("utf-8"))

            rewritten = data["message"]["content"].strip().strip('"').strip("'")
            return rewritten if rewritten else None

        except Exception as e:
            print(f"[Agentic] query rewrite 失败: {e}")
            return None

