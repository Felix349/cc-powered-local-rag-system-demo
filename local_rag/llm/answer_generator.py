"""
LLM 接口模块

职责：
  1. 把 RetrievalResult 组装成结构化 prompt
  2. 调用本地 LLM（Ollama）生成最终答案
  3. LLM 不可用时，直接从检索结果提取答案返回

三条路径的 prompt 策略不同：
  SEMANTIC    → 把命中的 chunk 作为上下文，让 LLM 基于上下文回答
  NUMERIC     → 把执行结果和生成的代码告诉 LLM，让它翻译成自然语言
  CHAIN_TABLE → 把操作步骤链和最终表格告诉 LLM，让它总结出答案
"""

import json
import textwrap
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

from local_rag.retriever.retriever import RetrievalResult
from local_rag.router.router import RoutePath


# ─────────────────────────────────────────────
# 最终答案数据结构
# ─────────────────────────────────────────────

@dataclass
class Answer:
    """
    系统的最终输出

    Attributes:
        question    : 原始问题
        answer      : 生成的答案文本
        sources     : 引用的来源列表（文件名 + chunk 预览）
        path        : 走的路径（semantic / numeric / chain_table）
        generated_by: "llm" 或 "rule"（规则兜底时）
    """
    question: str
    answer: str
    sources: List[dict]
    path: str
    generated_by: str = "llm"

    def display(self) -> str:
        """格式化展示，供命令行打印"""
        lines = [
            "─" * 60,
            f"问题：{self.question}",
            "─" * 60,
            f"回答：",
            "",
            self.answer,
            "",
        ]
        if self.sources:
            lines.append("来源：")
            for src in self.sources:
                lines.append(
                    f"  • {src.get('filename', '?')}  "
                    f"[相似度 {src.get('score', 0):.3f}]"
                )
                if src.get("preview"):
                    lines.append(f"    {src['preview']}")
        lines.append(f"\n[路径: {self.path} | 生成方式: {self.generated_by}]")
        lines.append("─" * 60)
        return "\n".join(lines)


# ─────────────────────────────────────────────
# Prompt 构建器
# ─────────────────────────────────────────────

class PromptBuilder:
    """
    根据检索结果和路径类型构建 prompt

    每条路径的 prompt 结构不同：
      SEMANTIC    : [系统指令] + [上下文 chunks] + [问题]
      NUMERIC     : [系统指令] + [执行结果/代码] + [问题]
      CHAIN_TABLE : [系统指令] + [操作步骤链] + [最终表格] + [问题]
    """

    # 所有路径共用的系统角色指令
    _SYSTEM_BASE = (
        "你是一个知识库问答助手，根据提供的参考内容准确回答用户问题。"
        "回答要简洁、准确、有条理。如果参考内容不足以回答问题，请如实说明。"
    )

    def build(self, result: RetrievalResult) -> tuple[str, str]:
        """
        返回 (system_prompt, user_prompt)
        """
        if result.route.path == RoutePath.SEMANTIC:
            return self._build_semantic(result)
        elif result.route.path == RoutePath.NUMERIC:
            return self._build_numeric(result)
        elif result.route.path == RoutePath.CHAIN_TABLE:
            return self._build_chain_table(result)
        else:
            return self._build_semantic(result)

    # ── 语义路径 prompt

    def _build_semantic(self, result: RetrievalResult) -> tuple[str, str]:
        system = self._SYSTEM_BASE

        if not result.context_chunks:
            user = f"问题：{result.question}\n\n（未找到相关参考内容，请根据通用知识回答。）"
            return system, user

        # 把命中的 chunk 拼成上下文块
        context_parts = []
        for i, hit in enumerate(result.context_chunks, start=1):
            src = hit.document.metadata.get("filename", "未知来源")
            context_parts.append(
                f"[参考 {i}]（来源：{src}，相似度：{hit.score:.3f}）\n"
                f"{hit.document.content.strip()}"
            )

        context_text = "\n\n".join(context_parts)
        user = textwrap.dedent(f"""\
            以下是从知识库检索到的相关内容：

            {context_text}

            ---
            请根据上述参考内容，回答以下问题：
            {result.question}
        """)
        return system, user

    # ── 数值路径 prompt

    def _build_numeric(self, result: RetrievalResult) -> tuple[str, str]:
        system = (
            self._SYSTEM_BASE
            + "已经通过代码计算出了数值结果，请将计算结果用自然语言表达出来，"
            + "不要重复代码内容，直接给出结论。"
        )

        parts = [f"问题：{result.question}", ""]

        if result.error:
            parts.append(f"计算过程遇到错误：{result.error}")
        else:
            if result.structured_result is not None:
                parts.append("计算结果：")
                parts.append(str(result.structured_result))
                parts.append("")
            if result.code_executed:
                parts.append("执行的代码：")
                parts.append(result.code_executed)
                parts.append("")

        parts.append("请根据上述计算结果，用一句话回答问题。")
        return system, "\n".join(parts)

    # ── Chain-of-Table 路径 prompt

    def _build_chain_table(self, result: RetrievalResult) -> tuple[str, str]:
        system = (
            self._SYSTEM_BASE
            + "已经通过逐步表格操作完成了推理，请根据推理过程和最终结果回答问题。"
        )

        parts = [f"问题：{result.question}", ""]

        if result.chain_steps:
            parts.append("推理步骤：")
            for step in result.chain_steps:
                err = f"（错误：{step['error']}）" if step.get("error") else ""
                parts.append(
                    f"  步骤{step['step']}: {step['operation']} "
                    f"→ {step.get('result_preview','')}{err}"
                )
            parts.append("")

        if result.structured_result is not None:
            parts.append("推理结论：")
            parts.append(str(result.structured_result))
            parts.append("")

        parts.append("请根据以上推理过程，简洁地回答问题。")
        return system, "\n".join(parts)


# ─────────────────────────────────────────────
# LLM 调用器
# ─────────────────────────────────────────────

class LLMCaller:
    """
    调用 Ollama 本地 LLM 生成答案

    支持流式和非流式两种模式：
      stream=False : 等待完整回复，适合程序化处理
      stream=True  : 逐 token 打印，适合命令行交互体验
    """

    def __init__(
        self,
        base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        temperature: float = 0.1,
        max_tokens: int = 1024,
    ):
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        stream: bool = False,
    ) -> str:
        """
        调用 LLM，返回生成的文本
        stream=True 时边生成边打印，同时返回完整文本
        """
        payload = json.dumps({
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",   "content": user_prompt},
            ],
            "stream": stream,
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self.base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )

        if stream:
            return self._call_stream(req)
        else:
            return self._call_blocking(req)

    def _call_blocking(self, req) -> str:
        with urllib.request.urlopen(req, timeout=60) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["message"]["content"].strip()

    def _call_stream(self, req) -> str:
        """流式输出：边打印边收集，最终返回完整文本"""
        full_text = []
        print()   # 换行，与 prompt 分开
        with urllib.request.urlopen(req, timeout=60) as resp:
            for line in resp:
                line = line.strip()
                if not line:
                    continue
                try:
                    chunk = json.loads(line.decode("utf-8"))
                    token = chunk.get("message", {}).get("content", "")
                    if token:
                        print(token, end="", flush=True)
                        full_text.append(token)
                    if chunk.get("done"):
                        break
                except json.JSONDecodeError:
                    continue
        print()   # 流式结束后换行
        return "".join(full_text)

    def is_available(self) -> bool:
        """检查 Ollama 是否在运行"""
        try:
            req = urllib.request.urlopen(
                f"{self.base_url}/api/tags", timeout=2
            )
            req.read()
            return True
        except Exception:
            return False


# ─────────────────────────────────────────────
# 规则兜底：无 LLM 时直接提取答案
# ─────────────────────────────────────────────

class RuleFallbackAnswerer:
    """
    LLM 不可用时，直接从检索结果组装答案文本
    质量不如 LLM，但保证系统始终能返回有意义的结果
    """

    def answer(self, result: RetrievalResult) -> str:
        path = result.route.path

        if path == RoutePath.NUMERIC:
            return self._answer_numeric(result)
        elif path == RoutePath.CHAIN_TABLE:
            return self._answer_chain_table(result)
        else:
            return self._answer_semantic(result)

    def _answer_semantic(self, result: RetrievalResult) -> str:
        if not result.context_chunks:
            return "未在知识库中找到与该问题相关的内容。"

        top = result.context_chunks[0]
        src = top.document.metadata.get("filename", "知识库")
        preview = top.document.content[:300].strip().replace("\n", " ")
        return (
            f"根据知识库中《{src}》的内容：\n\n"
            f"{preview}..."
            + (f"\n\n（共找到 {len(result.context_chunks)} 条相关内容，"
               f"显示最相关一条，相似度 {top.score:.3f}）"
               if len(result.context_chunks) > 1 else "")
        )

    def _answer_numeric(self, result: RetrievalResult) -> str:
        if result.error:
            return f"计算时遇到问题：{result.error}"
        if result.structured_result is not None:
            return str(result.structured_result)
        return "未能完成数值计算，请检查数据文件是否已加载。"

    def _answer_chain_table(self, result: RetrievalResult) -> str:
        if result.structured_result:
            return str(result.structured_result)
        if result.chain_steps:
            steps_str = "，".join(
                s["operation"] for s in result.chain_steps
            )
            return f"已完成操作链：{steps_str}，但未生成最终答案。"
        return "Chain-of-Table 推理未完成。"


# ─────────────────────────────────────────────
# 总入口：AnswerGenerator
# ─────────────────────────────────────────────

class AnswerGenerator:
    """
    把 RetrievalResult 转换成最终 Answer

    使用示例：
        generator = AnswerGenerator()
        answer = generator.generate(retrieval_result)
        print(answer.display())
    """

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        llm_model: str = "qwen2.5:7b",
        temperature: float = 0.1,
        stream: bool = True,
    ):
        self._prompt_builder = PromptBuilder()
        self._llm = LLMCaller(
            base_url=ollama_base_url,
            model=llm_model,
            temperature=temperature,
        )
        self._fallback = RuleFallbackAnswerer()
        self._stream = stream
        self._llm_available: Optional[bool] = None  # 懒加载检查

    def generate(self, result: RetrievalResult) -> Answer:
        """
        主入口：构建 prompt → 调用 LLM（或规则兜底）→ 返回 Answer
        """
        # 检查 LLM 可用性（只在第一次调用时检查）
        if self._llm_available is None:
            self._llm_available = self._llm.is_available()
            if not self._llm_available:
                print("[AnswerGenerator] Ollama 不可用，使用规则兜底模式")

        system_prompt, user_prompt = self._prompt_builder.build(result)

        if self._llm_available:
            try:
                answer_text = self._llm.call(
                    system_prompt, user_prompt, stream=self._stream
                )
                generated_by = "llm"
            except Exception as e:
                print(f"[AnswerGenerator] LLM 调用失败: {e}，切换规则兜底")
                answer_text = self._fallback.answer(result)
                generated_by = "rule"
        else:
            answer_text = self._fallback.answer(result)
            generated_by = "rule"

        # 整理来源信息
        sources = []
        for hit in result.context_chunks:
            sources.append({
                "filename": hit.document.metadata.get("filename", "?"),
                "score": hit.score,
                "preview": hit.document.content[:80].replace("\n", " "),
            })

        return Answer(
            question=result.question,
            answer=answer_text,
            sources=sources,
            path=result.retrieval_path,
            generated_by=generated_by,
        )
