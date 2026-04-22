"""
查询路由模块（Router）

职责：判断用户问题应该走哪条处理路径

三条路径：
  SEMANTIC      → 语义检索路径
                  适合：概念解释、定性描述、"是什么/为什么/怎么做"类问题
                  示例："这个产品有什么特点？"

  NUMERIC       → Text-to-Code 路径（pandas / SQL 执行）
                  适合：数值计算、聚合统计、精确比较
                  示例："哪个季度销售额最高？"、"平均绩效分是多少？"

  CHAIN_TABLE   → Chain-of-Table 路径（多步表格推理）
                  适合：需要多步操作才能得出结论的复杂表格问题
                  示例："增长率超过 10% 的月份里，哪个产品占比最高？"

两套实现：
  RuleBasedRouter  → 关键词匹配，无需 LLM，速度快，覆盖常见模式
  LLMRouter        → 用本地 LLM 做意图分类，准确率更高，需要 Ollama
  HybridRouter     → 先用规则，置信度低时交给 LLM 兜底（推荐）
"""

import json
import re
import urllib.request
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional


# ─────────────────────────────────────────────
# 路由结果数据结构
# ─────────────────────────────────────────────

class RoutePath(Enum):
    SEMANTIC     = "semantic"      # 语义检索
    NUMERIC      = "numeric"       # Text-to-Code 数值计算
    CHAIN_TABLE  = "chain_table"   # Chain-of-Table 多步推理


@dataclass
class RouteResult:
    """
    路由判断结果

    Attributes:
        path        : 选择的处理路径
        confidence  : 置信度，0.0 ~ 1.0
        reasoning   : 判断依据（调试用）
        matched_cues: 触发判断的关键线索列表
    """
    path: RoutePath
    confidence: float
    reasoning: str
    matched_cues: List[str] = field(default_factory=list)

    def __repr__(self):
        return (
            f"RouteResult(path={self.path.value}, "
            f"confidence={self.confidence:.2f}, "
            f"cues={self.matched_cues})"
        )


# ─────────────────────────────────────────────
# 基类
# ─────────────────────────────────────────────

class BaseRouter(ABC):

    @abstractmethod
    def route(self, question: str) -> RouteResult:
        """分析问题，返回路由结果"""
        pass

    def _normalize(self, text: str) -> str:
        """统一转小写、去多余空格，便于关键词匹配"""
        return re.sub(r"\s+", " ", text.lower().strip())


# ─────────────────────────────────────────────
# 实现一：规则路由器
# ─────────────────────────────────────────────

class RuleBasedRouter(BaseRouter):
    """
    基于关键词和语言模式的规则路由器

    设计思路：
    为三条路径各定义一组"触发词"，对问题打分，
    分数最高的路径胜出。

    打分规则：
      每命中一个触发词 +1 分
      多步推理的复杂连接词（"并且"/"同时满足"等）给 CHAIN_TABLE 额外加分
      问题里没有任何数字相关词 → SEMANTIC 获得基础分

    置信度 = 胜出路径的分数 / (胜出分数 + 第二名分数 + 1)
    """

    # ── NUMERIC 触发词：明确指向数值计算
    _NUMERIC_CUES = [
        # 聚合操作
        "最高", "最低", "最大", "最小", "最多", "最少",
        "总计", "合计", "总和", "求和", "总量", "累计",
        "平均", "均值", "中位数", "方差", "标准差",
        "数量", "个数", "计数", "有多少", "共有多少",
        # 比较 / 排名
        "排名", "排行", "排序", "第几", "前几", "前三", "前五",
        "比较", "对比", "差值", "差距", "增长了多少", "下降了多少",
        "高于", "低于", "超过", "不足", "大于", "小于",
        # 统计词
        "占比", "比例", "百分比", "份额",
        "同比", "环比", "增长率", "变化率",
        # 英文（混合文档场景）
        "sum", "count", "average", "avg", "max", "min",
        "total", "percent", "ratio", "rank",
    ]

    # 强数值信号词：命中一个算两分
    _STRONG_NUMERIC_CUES = {
        "排名", "排行", "排序", "rank",
        "总和", "求和", "sum",
        "平均", "均值", "avg", "average",
    }

    # ── CHAIN_TABLE 触发词：多步骤、多条件
    _CHAIN_TABLE_CUES = [
        # 多条件连接
        "并且", "同时", "而且", "以及", "还要",
        r"满足.{0,6}条件", r"符合.{0,6}要求", r"既.{0,10}又",
        # 先后步骤（中间允许有其他词）
        r"先.{0,10}再", r"首先.{0,15}然后",
        r"筛选.{0,10}之后", r"过滤.{0,8}再",
        r"在.{0,8}基础上",
        # "X部门/类别里 Y最高" → 先缩小范围再取极值
        r".{1,8}[里中].{1,8}最[高低大小多少]",
        r".{1,8}[里中].{1,8}[排名排行]",
        # 超过某条件后再聚合
        r"超过.{0,15}[总和平均最]",
        r"满足.{0,15}[总和平均最]",
        # 复杂聚合
        r"每个.{0,8}最", r"按.{0,8}分组.{0,8}最",
        r"哪些.{0,10}最",
        # 跨维度
        r"跨.{0,8}比较", r"不同.{0,8}之间",
    ]

    # ── SEMANTIC 触发词：解释、描述、概念类
    _SEMANTIC_CUES = [
        "是什么", "是哪", "什么是", "介绍", "描述",
        "解释", "说明", "概述", "总结", "简介",
        "为什么", "原因", "怎么", "如何", "方法",
        "特点", "特征", "优势", "劣势", "区别",
        "背景", "历史", "来源", "意义", "作用",
        "what is", "how to", "why", "describe", "explain",
    ]

    def route(self, question: str) -> RouteResult:
        q = self._normalize(question)

        numeric_hits     = self._scan(q, self._NUMERIC_CUES)
        chain_table_hits = self._scan_patterns(q, self._CHAIN_TABLE_CUES)
        semantic_hits    = self._scan(q, self._SEMANTIC_CUES)

        # 打分：强数值词命中算 2 分，普通词算 1 分
        score_numeric = sum(
            2 if hit in self._STRONG_NUMERIC_CUES else 1
            for hit in numeric_hits
        )
        score_chain_table = len(chain_table_hits) * 1.5   # 多步推理额外权重
        score_semantic    = len(semantic_hits) + 0.5       # 语义有基础分兜底

        # CHAIN_TABLE 需要同时具备"数值特征 + 多步骤特征"才生效
        # 如果只有多步骤词但没有任何数值词，降级到 SEMANTIC
        if score_chain_table > 0 and score_numeric == 0:
            score_chain_table *= 0.3

        # 反向加分：同时触发多步骤词 AND 数值词，说明真的是多步骤计算
        # 例："超过X的月份里，总和是多少" → chain_table 获得数值词加成
        if score_chain_table > 0 and score_numeric > 0:
            score_chain_table += score_numeric * 0.8

        scores = {
            RoutePath.NUMERIC:     score_numeric,
            RoutePath.CHAIN_TABLE: score_chain_table,
            RoutePath.SEMANTIC:    score_semantic,
        }

        best_path = max(scores, key=lambda p: scores[p])
        best_score = scores[best_path]
        second_score = sorted(scores.values(), reverse=True)[1]

        # 置信度：领先越多越自信
        confidence = best_score / (best_score + second_score + 1e-6)
        confidence = min(0.95, max(0.4, confidence))

        all_cues = numeric_hits + chain_table_hits + semantic_hits
        reasoning = self._build_reasoning(
            best_path, scores, numeric_hits, chain_table_hits, semantic_hits
        )

        return RouteResult(
            path=best_path,
            confidence=round(confidence, 3),
            reasoning=reasoning,
            matched_cues=all_cues,
        )

    def _scan(self, text: str, cues: List[str]) -> List[str]:
        """在文本中查找命中的关键词列表"""
        return [cue for cue in cues if cue in text]

    def _scan_patterns(self, text: str, patterns: List[str]) -> List[str]:
        """支持正则表达式的关键词扫描"""
        hits = []
        for pattern in patterns:
            if re.search(pattern, text):
                hits.append(pattern)
        return hits

    def _build_reasoning(
        self, path, scores, numeric_hits, chain_hits, semantic_hits
    ) -> str:
        lines = [f"选择路径: {path.value}"]
        lines.append(f"各路径分数: numeric={scores[RoutePath.NUMERIC]:.1f}, "
                     f"chain_table={scores[RoutePath.CHAIN_TABLE]:.1f}, "
                     f"semantic={scores[RoutePath.SEMANTIC]:.1f}")
        if numeric_hits:
            lines.append(f"数值关键词: {numeric_hits}")
        if chain_hits:
            lines.append(f"多步骤关键词: {chain_hits}")
        if semantic_hits:
            lines.append(f"语义关键词: {semantic_hits}")
        return " | ".join(lines)


# ─────────────────────────────────────────────
# 实现二：LLM 路由器
# ─────────────────────────────────────────────

class LLMRouter(BaseRouter):
    """
    用本地 LLM（通过 Ollama）做意图分类

    相比规则路由器的优势：
      - 能理解复杂语境（"给我看看表现最突出的那批数据"→ NUMERIC）
      - 不依赖关键词覆盖率
      - 可以处理口语化、模糊的提问方式

    需要 Ollama 运行且有可用模型（qwen2.5:7b 或更轻量的模型均可）
    """

    _SYSTEM_PROMPT = """你是一个查询路由分类器。
分析用户问题，判断它属于哪种查询类型，只能回答 JSON，格式如下：
{"path": "<类型>", "confidence": <0.0到1.0的浮点数>, "reasoning": "<一句话理由>"}

三种类型：
- "semantic"    : 语义/概念类问题，需要文字描述回答（如"介绍一下X"，"X是什么"，"为什么Y"）
- "numeric"     : 数值计算类问题，需要对结构化数据做聚合/统计/比较（如"最高的X是多少"，"平均Y是多少"）
- "chain_table" : 复杂多步推理，需要先筛选再聚合或多条件组合（如"满足A且B的记录中，C最大的是哪个"）

只输出 JSON，不要有任何其他文字。"""

    def __init__(
        self,
        ollama_base_url: str = "http://localhost:11434",
        model: str = "qwen2.5:7b",
        timeout: int = 10,
    ):
        self._base_url = ollama_base_url.rstrip("/")
        self._model = model
        self._timeout = timeout

    def route(self, question: str) -> RouteResult:
        try:
            response = self._call_ollama(question)
            return self._parse_response(response, question)
        except Exception as e:
            # LLM 调用失败时，回退到规则路由
            fallback = RuleBasedRouter().route(question)
            fallback.reasoning = f"[LLM失败:{e}，回退到规则] " + fallback.reasoning
            fallback.confidence = min(fallback.confidence, 0.6)
            return fallback

    def _call_ollama(self, question: str) -> str:
        payload = json.dumps({
            "model": self._model,
            "messages": [
                {"role": "system", "content": self._SYSTEM_PROMPT},
                {"role": "user",   "content": question},
            ],
            "stream": False,
            "options": {"temperature": 0},   # 分类任务用 temperature=0，结果稳定
        }).encode("utf-8")

        req = urllib.request.Request(
            f"{self._base_url}/api/chat",
            data=payload,
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=self._timeout) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        return data["message"]["content"].strip()

    def _parse_response(self, raw: str, question: str) -> RouteResult:
        # 提取 JSON（LLM 有时会在 JSON 前后加文字）
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if not match:
            raise ValueError(f"LLM 返回格式错误: {raw!r}")

        data = json.loads(match.group())
        path_str = data.get("path", "semantic")

        path_map = {
            "semantic":    RoutePath.SEMANTIC,
            "numeric":     RoutePath.NUMERIC,
            "chain_table": RoutePath.CHAIN_TABLE,
        }
        path = path_map.get(path_str, RoutePath.SEMANTIC)

        return RouteResult(
            path=path,
            confidence=float(data.get("confidence", 0.7)),
            reasoning=f"[LLM] {data.get('reasoning', '')}",
            matched_cues=[],
        )


# ─────────────────────────────────────────────
# 实现三：混合路由器（推荐）
# ─────────────────────────────────────────────

class HybridRouter(BaseRouter):
    """
    先用规则路由，置信度不足时交给 LLM 兜底

    流程：
      1. 规则路由器先判断
      2. 置信度 >= confidence_threshold → 直接采用规则结果
      3. 置信度 <  confidence_threshold → 交给 LLM 重新判断
      4. LLM 也不可用               → 采用规则结果（降级）

    这样做的好处：
      - 90% 的常见问题规则就能搞定，速度快
      - 边界模糊的问题交给 LLM，准确率高
      - LLM 挂了也不影响整体运行
    """

    def __init__(
        self,
        confidence_threshold: float = 0.65,
        ollama_base_url: str = "http://localhost:11434",
        llm_model: str = "qwen2.5:7b",
    ):
        self._rule_router = RuleBasedRouter()
        self._llm_router = LLMRouter(
            ollama_base_url=ollama_base_url,
            model=llm_model,
        )
        self._threshold = confidence_threshold

    def route(self, question: str) -> RouteResult:
        # 第一步：规则路由
        rule_result = self._rule_router.route(question)

        # 置信度足够 → 直接返回
        if rule_result.confidence >= self._threshold:
            rule_result.reasoning = "[规则] " + rule_result.reasoning
            return rule_result

        # 置信度不足 → 尝试 LLM
        llm_result = self._llm_router.route(question)

        # LLM 是回退结果（含"回退"标记）则用规则
        if "[LLM失败" in llm_result.reasoning:
            rule_result.reasoning = "[规则-低置信度兜底] " + rule_result.reasoning
            return rule_result

        # LLM 成功，保留 LLM 结果，附上规则判断作参考
        llm_result.reasoning += (
            f" | 规则备选: {rule_result.path.value}"
            f"(conf={rule_result.confidence:.2f})"
        )
        return llm_result


# ─────────────────────────────────────────────
# 工厂函数
# ─────────────────────────────────────────────

def create_router(
    backend: str = "rule",
    **kwargs,
) -> BaseRouter:
    """
    创建路由器

    backend 选项：
      "rule"    → RuleBasedRouter，无依赖，推荐开发阶段使用
      "llm"     → LLMRouter，需要 Ollama
      "hybrid"  → HybridRouter，推荐生产使用
    """
    registry = {
        "rule":   RuleBasedRouter,
        "llm":    LLMRouter,
        "hybrid": HybridRouter,
    }
    if backend not in registry:
        raise ValueError(f"未知 backend: {backend!r}，可选: {list(registry.keys())}")
    return registry[backend](**kwargs)
