"""SQLite 存储层：唯一真相源 + 原始层寻址 + 冷存分帧。

职责边界（对应 SPEC-v1.1 §1）：
- ``memory.db`` 是唯一真相源，JSONL 只是导出格式；
- ``raw/<day>.jsonl`` 只追加、不可变，每行一条消息；
- ``source`` 表记录每条消息的字节区间与 sha256，这是「精确回取原文」的落点；
- 压缩逻辑（compress.py）不在这里，本层只负责「存得住、取得回、可幂等」。
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

from .chunks import (
    DEFAULT_FRAME_LINES,
    ChunkError,
    ChunkIndex,
    container_path,
    read_index,
    read_line,
    write_container,
)
from .clock import Clock, SystemClock, to_iso, to_utc
from .ids import claim_id, message_id
from .schema import validate_card

DDL_PATH = Path(__file__).resolve().parents[2] / "ddl.sql"

JSON_COLUMNS = (
    "supersedes",
    "source_ids",
    "entities",
    "keywords",
    "facts",
    "decisions",
    "todos",
    "constraints",
)

CARD_COLUMNS = (
    "id",
    "schema_version",
    "session_id",
    "tier",
    "importance",
    "network",
    "status",
    "observed_at",
    "valid_from",
    "valid_to",
    "half_life_days",
    "last_access",
    "access_count",
    "supersedes",
    "superseded_by",
    "source_ids",
    "entities",
    "keywords",
    "summary",
    "facts",
    "decisions",
    "todos",
    "constraints",
    "raw_quote",
    "confidence",
    "dedup_key",
    "created_at",
)


class RawIntegrityError(RuntimeError):
    """回取到的原文与写入时的 sha256 不一致。"""


@dataclass(frozen=True)
class CardWriteResult:
    """写入结果。``bool(result)`` 等价于 inserted，便于 ``assertTrue(store.put_card(c))``。"""

    inserted: bool
    superseded: tuple[str, ...] = ()

    def __bool__(self) -> bool:
        return self.inserted


class Store:
    """本地记忆库。``root`` 下会生成 memory.db / raw/ / chunks/。"""

    def __init__(
        self,
        root: str | Path,
        *,
        clock: Clock | None = None,
        codec: str = "zlib",
        frame_lines: int = DEFAULT_FRAME_LINES,
        ddl_path: str | Path | None = None,
        apply_optional: bool = True,
    ) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.raw_dir = self.root / "raw"
        self.chunk_dir = self.root / "chunks"
        self.raw_dir.mkdir(exist_ok=True)
        self.chunk_dir.mkdir(exist_ok=True)
        self.db_path = self.root / "memory.db"
        self.clock: Clock = clock or SystemClock()
        self.codec = codec
        self.frame_lines = frame_lines

        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.warnings: list[str] = []
        self.capabilities = self._apply_ddl(ddl_path or DDL_PATH, apply_optional=apply_optional)

    # ------------------------------------------------------------------ DDL

    @staticmethod
    def _split_ddl(sql: str) -> tuple[str, dict[str, str]]:
        core: list[str] = []
        optional: dict[str, list[str]] = {}
        current: str | None = None
        for line in sql.splitlines():
            stripped = line.strip()
            if stripped.startswith("-- @@optional:"):
                current = stripped.split(":", 1)[1].strip()
                optional[current] = []
                continue
            if stripped == "-- @@end":
                current = None
                continue
            (optional[current] if current else core).append(line)
        return "\n".join(core), {k: "\n".join(v) for k, v in optional.items()}

    def _apply_ddl(self, ddl_path: str | Path, *, apply_optional: bool) -> set[str]:
        """执行建表语句；可选块（fts5 / vec0）失败时降级而不是中断。"""
        sql = Path(ddl_path).read_text(encoding="utf-8")
        core, optional = self._split_ddl(sql)
        self.conn.executescript(core)
        applied: set[str] = set()
        if not apply_optional:
            return applied
        for name, block in optional.items():
            try:
                self.conn.executescript(block)
                applied.add(name)
            except sqlite3.OperationalError as exc:
                self.warnings.append(f"可选能力 {name} 不可用，已降级：{exc}")
        return applied

    # ------------------------------------------------------------- 原始层

    def append_message(
        self,
        session_id: str,
        text: str,
        *,
        ts: Any | None = None,
        meta: Mapping[str, Any] | None = None,
    ) -> str:
        """追加一条原始消息，返回 source_id。写入后字节内容不可再变。"""
        if not isinstance(text, str):
            raise TypeError("消息内容必须是 str")
        moment = to_utc(self.clock.now()) if ts is None else to_utc(ts)
        day = moment.date().isoformat()
        raw_file = self.raw_dir / f"{day}.jsonl"

        with raw_file.open("ab") as handle:
            byte_off = handle.tell()
            line_no = self._next_line_no(day)
            record: dict[str, Any] = {
                "id": message_id(day, session_id, line_no),
                "session_id": session_id,
                "ts": to_iso(moment),
                "text": text,
            }
            if meta:
                record["meta"] = dict(meta)
            blob = json.dumps(record, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
            handle.write(blob + b"\n")

        with self.conn:
            self.conn.execute(
                """
                INSERT INTO source
                    (id, session_id, day, line_no, byte_off, byte_len, sha256, created_at, is_archived)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    record["id"],
                    session_id,
                    day,
                    line_no,
                    byte_off,
                    len(blob),
                    hashlib.sha256(blob).hexdigest(),
                    to_iso(moment),
                ),
            )
        return str(record["id"])

    def _next_line_no(self, day: str) -> int:
        row = self.conn.execute(
            "SELECT COALESCE(MAX(line_no), -1) + 1 AS next FROM source WHERE day = ?", (day,)
        ).fetchone()
        return int(row["next"])

    def get_message(self, source_id: str) -> dict[str, Any]:
        """按 source_id 取回原始记录（含全文），并校验 sha256。"""
        row = self.conn.execute("SELECT * FROM source WHERE id = ?", (source_id,)).fetchone()
        if row is None:
            raise KeyError(f"未知 source_id：{source_id}")

        if row["is_archived"]:
            container = self._container_for(str(row["day"]))
            if container is None:
                raise ChunkError(f"第 {row['day']} 天标记为已归档，但冷存容器缺失")
            blob = read_line(container, int(row["line_no"]))
        else:
            raw_file = self.raw_dir / f"{row['day']}.jsonl"
            with raw_file.open("rb") as handle:
                handle.seek(int(row["byte_off"]))
                blob = handle.read(int(row["byte_len"]))

        digest = hashlib.sha256(blob).hexdigest()
        if digest != row["sha256"]:
            raise RawIntegrityError(
                f"{source_id} 原文校验失败：期望 {row['sha256'][:12]}…，实际 {digest[:12]}…"
            )
        return json.loads(blob.decode("utf-8"))

    def get_raw(self, source_id: str) -> str:
        """按 source_id 取回精确原文。"""
        return str(self.get_message(source_id)["text"])

    def get_raw_many(self, source_ids: Iterable[str]) -> list[str]:
        return [self.get_raw(sid) for sid in source_ids]

    def _container_for(self, day: str) -> Path | None:
        preferred = container_path(self.chunk_dir, day, self.codec)
        if preferred.exists() and Path(str(preferred) + ".idx.json").exists():
            return preferred
        for candidate in sorted(self.chunk_dir.glob(f"{day}.*")):
            if candidate.name.endswith(".idx.json"):
                continue
            if Path(str(candidate) + ".idx.json").exists():
                return candidate
        return None

    def archive_day(
        self,
        day: str,
        *,
        codec: str | None = None,
        frame_lines: int | None = None,
    ) -> ChunkIndex:
        """把某天的 raw JSONL 按帧冷存，并把 source 行标记为已归档。

        原文 JSONL 本身不删（可回溯、可重算），冷存是寻址副本：
        归档后 ``get_raw`` 会走分帧解压，只解 1 帧。
        """
        raw_file = self.raw_dir / f"{day}.jsonl"
        if not raw_file.exists():
            raise FileNotFoundError(f"没有该日期的原始文件：{raw_file}")
        blob = raw_file.read_bytes()
        lines = blob.split(b"\n")
        if lines and lines[-1] == b"":
            lines.pop()

        used_codec = codec or self.codec
        container = container_path(self.chunk_dir, day, used_codec)
        index = write_container(
            container,
            lines,
            day=day,
            codec=used_codec,
            frame_lines=frame_lines or self.frame_lines,
        )
        with self.conn:
            self.conn.execute("UPDATE source SET is_archived = 1 WHERE day = ?", (day,))
        return index

    def chunk_index(self, day: str) -> ChunkIndex:
        container = self._container_for(day)
        if container is None:
            raise ChunkError(f"第 {day} 天没有冷存容器")
        return read_index(container)

    # --------------------------------------------------------------- 记忆卡

    def put_card(self, card: Mapping[str, Any], *, validate: bool = True) -> CardWriteResult:
        """写入一张卡。

        行为约定：
        - 同一张卡重复写（同 dedup_key）是无副作用的**重放**，返回 inserted=False；
        - 同一段原文 + 同一 tier 但摘要变了（例如换了提示词）=> 新卡取代旧卡：
          旧卡 status='superseded' 并回填 superseded_by，历史不删除（冲突不覆盖）；
        - 同一段原文抽出不同 tier/不同摘要的卡（决策 / 待办 / 约束）可以并存。
        """
        payload = dict(card)
        if validate:
            validate_card(payload)
        created = to_iso(to_utc(self.clock.now()))
        canonical = sorted(set(payload.get("source_ids") or ()))
        payload["source_ids"] = canonical
        span_key = json.dumps(canonical, ensure_ascii=False)

        values: list[Any] = []
        for column in CARD_COLUMNS:
            if column == "created_at":
                values.append(created)
                continue
            value = payload.get(column)
            if column in JSON_COLUMNS:
                value = json.dumps(list(value or []), ensure_ascii=False)
            values.append(value)

        placeholders = ", ".join("?" for _ in CARD_COLUMNS)
        with self.conn:
            cursor = self.conn.execute(
                f"INSERT OR IGNORE INTO card ({', '.join(CARD_COLUMNS)}) VALUES ({placeholders})",
                values,
            )
            if cursor.rowcount == 0:
                return CardWriteResult(inserted=False)

            for source_id in canonical:
                self.conn.execute(
                    "INSERT OR IGNORE INTO card_source (card_id, source_id) VALUES (?, ?)",
                    (payload["id"], source_id),
                )

            for claim in payload.get("claims", ()) or ():
                day = str(payload["observed_at"])[:10]
                self.conn.execute(
                    """
                    INSERT OR IGNORE INTO claim
                        (id, card_id, kind, subject_entity, predicate, object, polarity,
                         due_at, done_at, valid_from, valid_to, superseded_by)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    (
                        claim_id(day, str(payload["id"]), claim["kind"], claim["predicate"], claim.get("object")),
                        payload["id"],
                        claim["kind"],
                        claim.get("subject_entity"),
                        claim["predicate"],
                        claim.get("object"),
                        int(claim.get("polarity", 1)),
                        claim.get("due_at"),
                        claim.get("done_at"),
                        claim.get("valid_from"),
                        claim.get("valid_to"),
                        claim.get("superseded_by"),
                    ),
                )

            replaced = [
                row["id"]
                for row in self.conn.execute(
                    """
                    SELECT id FROM card
                     WHERE session_id = ? AND tier = ? AND source_ids = ?
                       AND status = 'active' AND id <> ?
                    """,
                    (payload["session_id"], payload["tier"], span_key, payload["id"]),
                )
            ]
            if replaced:
                marks = ", ".join("?" for _ in replaced)
                self.conn.execute(
                    f"UPDATE card SET status = 'superseded', superseded_by = ? WHERE id IN ({marks})",
                    [payload["id"], *replaced],
                )
                inherited = sorted(set(payload.get("supersedes") or ()) | set(replaced))
                self.conn.execute(
                    "UPDATE card SET supersedes = ? WHERE id = ?",
                    (json.dumps(inherited, ensure_ascii=False), payload["id"]),
                )

        return CardWriteResult(inserted=True, superseded=tuple(sorted(replaced)))

    def sources_of_card(self, card_id: str) -> list[str]:
        """这张卡引用到的全部 source_id（回链的显式查询入口）。"""
        return [
            row["source_id"]
            for row in self.conn.execute(
                "SELECT source_id FROM card_source WHERE card_id = ? ORDER BY source_id", (card_id,)
            )
        ]

    def cards_citing(self, source_id: str) -> list[str]:
        """引用了这条原文的全部卡（重压缩与失效传播都要用）。"""
        return [
            row["card_id"]
            for row in self.conn.execute(
                "SELECT card_id FROM card_source WHERE source_id = ? ORDER BY card_id", (source_id,)
            )
        ]

    def get_card(self, card_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM card WHERE id = ?", (card_id,)).fetchone()
        if row is None:
            return None
        record: dict[str, Any] = {key: row[key] for key in row.keys()}
        for column in JSON_COLUMNS:
            record[column] = json.loads(record[column] or "[]")
        record["claims"] = [
            {key: claim[key] for key in claim.keys()}
            for claim in self.conn.execute(
                "SELECT * FROM claim WHERE card_id = ? ORDER BY id", (card_id,)
            )
        ]
        return record

    def cards(
        self,
        *,
        session_id: str | None = None,
        tier: str | None = None,
        status: str = "active",
    ) -> list[dict[str, Any]]:
        """按条件取卡（默认只要 active），供召回器使用。"""
        sql = "SELECT id FROM card WHERE status = ?"
        params: list[Any] = [status]
        if session_id is not None:
            sql += " AND session_id = ?"
            params.append(session_id)
        if tier is not None:
            sql += " AND tier = ?"
            params.append(tier)
        sql += " ORDER BY observed_at DESC, id"
        return [card for row in self.conn.execute(sql, params) if (card := self.get_card(row["id"]))]

    # --------------------------------------------------------------- 游标

    def get_cursor(self, session_id: str) -> dict[str, Any] | None:
        row = self.conn.execute("SELECT * FROM cursor WHERE session_id = ?", (session_id,)).fetchone()
        return None if row is None else {key: row[key] for key in row.keys()}

    def set_cursor(self, session_id: str, *, last_source_id: str, last_day: str, last_line: int) -> None:
        """记录压缩进度。崩溃后从这里续跑，保证不重复不遗漏。"""
        with self.conn:
            self.conn.execute(
                """
                INSERT INTO cursor (session_id, last_source_id, last_day, last_line, updated_at)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (session_id) DO UPDATE SET
                    last_source_id = excluded.last_source_id,
                    last_day       = excluded.last_day,
                    last_line      = excluded.last_line,
                    updated_at     = excluded.updated_at
                """,
                (session_id, last_source_id, last_day, last_line, to_iso(to_utc(self.clock.now()))),
            )

    def messages_for_session(
        self,
        session_id: str,
        *,
        after_day: str | None = None,
        after_line: int | None = None,
    ) -> list[dict[str, Any]]:
        """按时间顺序返回会话消息（不读取正文，仅元数据）。"""
        sql = "SELECT id, session_id, day, line_no, created_at FROM source WHERE session_id = ?"
        params: list[Any] = [session_id]
        if after_day is not None and after_line is not None:
            sql += " AND (day > ? OR (day = ? AND line_no > ?))"
            params.extend([after_day, after_day, after_line])
        sql += " ORDER BY day, line_no"
        return [
            {
                "id": row["id"],
                "session_id": row["session_id"],
                "day": row["day"],
                "line_no": row["line_no"],
                "created_at": row["created_at"],
            }
            for row in self.conn.execute(sql, params)
        ]

    def pending_messages(self, session_id: str) -> list[dict[str, Any]]:
        """尚未压缩的消息（依据 cursor）。这是触发机制幂等的关键。"""
        cursor = self.get_cursor(session_id)
        if cursor is None:
            return self.messages_for_session(session_id)
        return self.messages_for_session(
            session_id,
            after_day=cursor["last_day"],
            after_line=int(cursor["last_line"]),
        )

    # ---------------------------------------------------------------- 运行记录

    def record_run(
        self,
        *,
        session_id: str,
        from_source: str | None = None,
        to_source: str | None = None,
        model: str | None = None,
        prompt_hash: str | None = None,
        status: str = "ok",
    ) -> str:
        """记录一次抽取运行。

        同时记下模型名与提示词指纹：日后「结果变了」时，才能判断是换了模型还是改了提示词。
        ID 由 (会话, 区间, 模型, 提示词, 时间) 派生，同一秒内的同一次运行重放是幂等的。
        """
        created = to_iso(to_utc(self.clock.now()))
        run_id = "run_" + hashlib.sha256(
            "|".join([session_id, from_source or "", to_source or "", model or "", prompt_hash or "", created]).encode()
        ).hexdigest()[:16]
        with self.conn:
            self.conn.execute(
                """
                INSERT OR IGNORE INTO extraction_run
                    (id, session_id, from_source, to_source, model, prompt_hash, created_at, status)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (run_id, session_id, from_source, to_source, model, prompt_hash, created, status),
            )
        return run_id

    def runs(self, session_id: str | None = None, *, limit: int = 20) -> list[dict[str, Any]]:
        sql = "SELECT * FROM extraction_run"
        params: list[Any] = []
        if session_id is not None:
            sql += " WHERE session_id = ?"
            params.append(session_id)
        sql += " ORDER BY created_at DESC, id DESC LIMIT ?"
        params.append(limit)
        return [{key: row[key] for key in row.keys()} for row in self.conn.execute(sql, params)]

    # ------------------------------------------------------------ 访问计费

    def note_access(
        self,
        card_id: str,
        session_id: str,
        *,
        render: str = "full",
        ts: Any | None = None,
    ) -> int:
        """记录一次「真正进入上下文」的访问，返回累计次数。

        候选召回不计入——否则被检索到一次就升级，层级会被噪声推着走。
        """
        moment = to_utc(self.clock.now()) if ts is None else to_utc(ts)
        stamp = to_iso(moment)
        with self.conn:
            self.conn.execute(
                "INSERT INTO card_access (card_id, session_id, accessed_at, render) VALUES (?, ?, ?, ?)",
                (card_id, session_id, stamp, render),
            )
            self.conn.execute(
                """
                UPDATE card
                   SET access_count = access_count + 1,
                       last_access  = ?
                 WHERE id = ?
                """,
                (stamp, card_id),
            )
        row = self.conn.execute("SELECT access_count FROM card WHERE id = ?", (card_id,)).fetchone()
        return int(row["access_count"]) if row else 0

    def access_stats(self, card_id: str, *, since: Any) -> tuple[int, int]:
        """返回 (近 since 起访问次数, 不同会话数) —— next_tier 的两个入参。"""
        stamp = to_iso(to_utc(since) if not isinstance(since, str) else since)
        row = self.conn.execute(
            """
            SELECT COUNT(*) AS hits, COUNT(DISTINCT session_id) AS sessions
              FROM card_access
             WHERE card_id = ? AND accessed_at >= ?
            """,
            (card_id, stamp),
        ).fetchone()
        return int(row["hits"]), int(row["sessions"])

    # ---------------------------------------------------------------- 杂项

    def stats(self) -> dict[str, int]:
        rows = self.conn.execute("SELECT tier, COUNT(*) AS n FROM card GROUP BY tier").fetchall()
        counts = {tier: 0 for tier in ("T0", "T1", "T2", "T3", "T4", "T5", "T6", "T7", "T8", "T9")}
        for row in rows:
            counts[row["tier"]] = int(row["n"])
        return counts

    def close(self) -> None:
        self.conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def __repr__(self) -> str:  # pragma: no cover - 调试用
        return f"Store(root={str(self.root)!r}, capabilities={sorted(self.capabilities)})"
