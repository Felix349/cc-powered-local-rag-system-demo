"""
对话记录模块（ConversationLogger）

职责：把每次问答的完整过程持久化到本地，用于：
  - 分析系统回答质量
  - 发现高频失败问题
  - 积累标注数据，为后续微调做准备

存储设计：双格式并存
  logs/conversations.jsonl  → 每行一条记录（JSON Lines），便于程序解析
  logs/conversations.md     → 可读的 Markdown 格式，便于人工审查

每条记录包含：
  - 时间戳、会话 ID、问题、回答
  - 检索路径、命中分数、命中来源
  - 配置信息（embedding 模型、LLM 模型）
  - 用户反馈（可选，事后标注）
"""

import json
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

from local_rag.llm.answer_generator import Answer


class ConversationLogger:
    """
    对话记录器

    使用示例：
        logger = ConversationLogger(log_dir="./logs")
        session_id = logger.new_session()

        # 每次问答后记录
        logger.log(answer, session_id=session_id)

        # 事后标注反馈
        logger.add_feedback(record_id, rating=1, comment="回答正确")

        # 分析统计
        logger.print_stats()
    """

    def __init__(self, log_dir: str = "./logs"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self._jsonl_path = self.log_dir / "conversations.jsonl"
        self._md_path    = self.log_dir / "conversations.md"

        # 初始化 Markdown 文件头（仅首次创建时写入）
        if not self._md_path.exists():
            self._md_path.write_text(
                "# RAG 系统对话记录\n\n"
                "| 时间 | 问题 | 路径 | 分数 | 反馈 |\n"
                "|------|------|------|------|------|\n",
                encoding="utf-8",
            )

    def new_session(self) -> str:
        """生成一个新的会话 ID，用于关联同一次交互中的多条问答"""
        session_id = datetime.now().strftime("%Y%m%d_%H%M%S") + "_" + uuid.uuid4().hex[:6]
        print(f"[Logger] 新会话开始: {session_id}")
        return session_id

    def log(
        self,
        answer: Answer,
        session_id: str = "default",
        embedding_model: str = "",
        extra: dict = None,
    ) -> str:
        """
        记录一条问答

        Args:
            answer         : RAGSystem.ask() 返回的 Answer 对象
            session_id     : 会话 ID，同一次启动的多次问答用同一个 ID
            embedding_model: 当前使用的 embedding 模型名
            extra          : 额外字段，会原样存入记录

        Returns:
            record_id: 本条记录的唯一 ID，用于后续 add_feedback()
        """
        record_id = uuid.uuid4().hex[:12]
        timestamp = datetime.now().isoformat(timespec="seconds")

        # 提取命中来源信息
        sources = []
        top_score = 0.0
        for src in answer.sources:
            sources.append({
                "filename": src.get("filename", ""),
                "score":    round(src.get("score", 0.0), 4),
                "preview":  src.get("preview", "")[:80],
            })
            if src.get("score", 0.0) > top_score:
                top_score = src.get("score", 0.0)

        record = {
            "record_id":      record_id,
            "session_id":     session_id,
            "timestamp":      timestamp,
            "question":       answer.question,
            "answer":         answer.answer,
            "path":           answer.path,
            "generated_by":   answer.generated_by,
            "top_score":      round(top_score, 4),
            "source_count":   len(sources),
            "sources":        sources,
            "embedding_model": embedding_model,
            "feedback":       None,   # 事后通过 add_feedback() 填入
            "feedback_comment": None,
        }
        if extra:
            record.update(extra)

        # 写入 JSONL（追加一行）
        with open(self._jsonl_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # 写入 Markdown 摘要行
        q_short = answer.question[:30].replace("|", "｜")
        a_short = answer.answer[:40].replace("\n", " ").replace("|", "｜")
        with open(self._md_path, "a", encoding="utf-8") as f:
            f.write(
                f"| {timestamp} "
                f"| {q_short} "
                f"| {answer.path} "
                f"| {top_score:.3f} "
                f"| ⬜ 待评 |\n"
            )

        return record_id

    def add_feedback(
        self,
        record_id: str,
        rating: int,
        comment: str = "",
    ):
        """
        为已记录的问答添加反馈标注

        Args:
            record_id : log() 返回的记录 ID
            rating    : 1=好评 / 0=差评
            comment   : 可选的文字说明，如"回答不完整"、"来源错误"
        """
        if not self._jsonl_path.exists():
            print("[Logger] 暂无记录")
            return

        lines = self._jsonl_path.read_text(encoding="utf-8").splitlines()
        updated = False
        new_lines = []
        for line in lines:
            if not line.strip():
                continue
            record = json.loads(line)
            if record.get("record_id") == record_id:
                record["feedback"]         = rating
                record["feedback_comment"] = comment
                updated = True
            new_lines.append(json.dumps(record, ensure_ascii=False))

        if updated:
            self._jsonl_path.write_text(
                "\n".join(new_lines) + "\n", encoding="utf-8"
            )
            label = "👍 好评" if rating == 1 else "👎 差评"
            print(f"[Logger] 反馈已记录: {record_id} → {label} {comment}")
        else:
            print(f"[Logger] 未找到记录: {record_id}")

    def print_stats(self):
        """打印对话记录的统计摘要"""
        if not self._jsonl_path.exists():
            print("[Logger] 暂无对话记录")
            return

        records = self._load_all()
        if not records:
            print("[Logger] 暂无对话记录")
            return

        total       = len(records)
        rated       = [r for r in records if r.get("feedback") is not None]
        good        = [r for r in rated if r.get("feedback") == 1]
        paths       = {}
        scores      = [r["top_score"] for r in records if r.get("top_score", 0) > 0]
        low_score   = [r for r in records if r.get("top_score", 1) < 0.60]

        for r in records:
            p = r.get("path", "unknown")
            paths[p] = paths.get(p, 0) + 1

        print(f"\n{'═'*50}")
        print(f"  对话记录统计")
        print(f"{'═'*50}")
        print(f"  总问答数:      {total}")
        print(f"  已评价:        {len(rated)} / {total}")
        if rated:
            print(f"  好评率:        {len(good)}/{len(rated)} = {len(good)/len(rated)*100:.1f}%")
        if scores:
            print(f"  平均命中分数:  {sum(scores)/len(scores):.3f}")
            print(f"  低分问答(<0.6):{len(low_score)} 条  ← 重点改进方向")
        print(f"\n  路径分布:")
        for path, count in sorted(paths.items(), key=lambda x: -x[1]):
            print(f"    {path:<15} {count} 条")

        if low_score:
            print(f"\n  低分问答（需改进）:")
            for r in low_score[:5]:
                print(f"    [{r['top_score']:.3f}] {r['question'][:50]}")
        print(f"{'═'*50}\n")
        print(f"  完整记录: {self._jsonl_path}")
        print(f"  可读版本: {self._md_path}\n")

    def export_for_review(self, output_path: str = None, min_records: int = 1):
        """
        导出完整对话记录为易读的 Markdown 文件，用于人工审查

        每条记录包含完整问题、完整回答、来源列表和反馈状态
        """
        records = self._load_all()
        if not records:
            print("[Logger] 暂无记录可导出")
            return

        output = Path(output_path or self.log_dir / "review_export.md")
        lines = [
            "# RAG 对话记录 — 人工审查版\n",
            f"导出时间：{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  ",
            f"总记录数：{len(records)}\n",
            "---\n",
        ]

        for i, r in enumerate(records, 1):
            feedback_str = {1: "👍 好评", 0: "👎 差评"}.get(
                r.get("feedback"), "⬜ 未评价"
            )
            comment = r.get("feedback_comment") or ""

            lines += [
                f"## [{i}] {r['timestamp']}",
                f"**路径**: {r['path']}  |  "
                f"**命中分数**: {r['top_score']:.3f}  |  "
                f"**反馈**: {feedback_str}",
                f"{'  ' + comment if comment else ''}",
                f"\n**问题**：{r['question']}\n",
                f"**回答**：\n{r['answer']}\n",
            ]

            if r.get("sources"):
                lines.append("**来源**：")
                for src in r["sources"]:
                    lines.append(
                        f"- {src['filename']}  "
                        f"[相似度 {src['score']:.3f}]  "
                        f"`{src['preview']}`"
                    )

            lines.append("\n---\n")

        output.write_text("\n".join(lines), encoding="utf-8")
        print(f"[Logger] 审查文件已导出: {output}  ({len(records)} 条记录)")
        return str(output)

    def _load_all(self) -> list:
        """加载所有记录"""
        if not self._jsonl_path.exists():
            return []
        records = []
        for line in self._jsonl_path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        return records
