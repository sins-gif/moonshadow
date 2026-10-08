-- Moonshadow v1.1 建表语句（Phase 0/1）
-- SQLite 是唯一真相源；raw/*.jsonl 为不可变追加日志；chunks/* 为冷存分帧。
-- 带 @@optional 标记的块在能力缺失时可跳过（见 store.py::_apply_ddl）。

PRAGMA journal_mode = WAL;
PRAGMA foreign_keys = ON;

-- ---------------------------------------------------------------------------
-- 原始层寻址表：source_id -> 精确字节区间。没有这张表，「精确回取原文」无法实现。
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS source (
  id          TEXT PRIMARY KEY,          -- msg_...
  session_id  TEXT NOT NULL,
  day         TEXT NOT NULL,             -- 'YYYY-MM-DD'
  line_no     INTEGER NOT NULL,          -- 0-based 行号，主寻址键
  byte_off    INTEGER NOT NULL,          -- 在 raw/<day>.jsonl 中的字节偏移
  byte_len    INTEGER NOT NULL,          -- 不含换行符
  sha256      TEXT NOT NULL,             -- 回取后校验，保证逐字节一致
  created_at  TEXT NOT NULL,
  is_archived INTEGER NOT NULL DEFAULT 0 CHECK (is_archived IN (0, 1)),
  UNIQUE (day, line_no)
);
CREATE INDEX IF NOT EXISTS idx_source_session ON source (session_id, day, line_no);

-- ---------------------------------------------------------------------------
-- 记忆卡：状态与元数据（数组字段用 JSON1 存储，claims 单独成表以便检索）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS card (
  id             TEXT PRIMARY KEY,
  schema_version INTEGER NOT NULL,
  session_id     TEXT NOT NULL,
  tier           TEXT NOT NULL CHECK (tier IN ('T0','T1','T2','T3','T4','T5','T6','T7','T8','T9')),
  importance     INTEGER NOT NULL CHECK (importance BETWEEN 1 AND 10),
  network        TEXT NOT NULL CHECK (network IN ('world','experience','opinion','observation')),
  status         TEXT NOT NULL CHECK (status IN ('active','superseded','source_expired','archived')),
  observed_at    TEXT NOT NULL,
  valid_from     TEXT,
  valid_to       TEXT,
  half_life_days REAL NOT NULL DEFAULT 0 CHECK (half_life_days >= 0),
  last_access    TEXT,
  access_count   INTEGER NOT NULL DEFAULT 0 CHECK (access_count >= 0),
  supersedes     TEXT NOT NULL DEFAULT '[]',   -- JSON array
  superseded_by  TEXT,
  source_ids     TEXT NOT NULL DEFAULT '[]',   -- JSON array（排序去重后的规范化形式）
  entities       TEXT NOT NULL DEFAULT '[]',   -- JSON array
  keywords       TEXT NOT NULL DEFAULT '[]',   -- JSON array
  summary        TEXT NOT NULL,
  facts          TEXT NOT NULL DEFAULT '[]',
  decisions      TEXT NOT NULL DEFAULT '[]',
  todos          TEXT NOT NULL DEFAULT '[]',
  constraints    TEXT NOT NULL DEFAULT '[]',
  raw_quote      TEXT,
  confidence     REAL CHECK (confidence IS NULL OR (confidence >= 0 AND confidence <= 1)),
  dedup_key      TEXT NOT NULL,
  created_at     TEXT NOT NULL
);
CREATE UNIQUE INDEX IF NOT EXISTS uq_card_dedup ON card (dedup_key);
CREATE INDEX IF NOT EXISTS idx_card_tier_status ON card (tier, status);
CREATE INDEX IF NOT EXISTS idx_card_observed ON card (observed_at);
CREATE INDEX IF NOT EXISTS idx_card_span ON card (session_id, tier, source_ids, status);

-- 卡 -> 原文 的显式链接：回链是核心承诺，不能只藏在 JSON 字段里。
-- 「这条消息被哪些卡引用」用于重压缩、失效传播与冲突定位。
CREATE TABLE IF NOT EXISTS card_source (
  card_id   TEXT NOT NULL REFERENCES card (id) ON DELETE CASCADE,
  source_id TEXT NOT NULL,
  PRIMARY KEY (card_id, source_id)
);
CREATE INDEX IF NOT EXISTS idx_card_source_source ON card_source (source_id);

-- ---------------------------------------------------------------------------
-- 断言表：冲突判定、未完成待办、观点演变都从这里查，不靠模型即兴发挥
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS claim (
  id             TEXT PRIMARY KEY,
  card_id        TEXT NOT NULL REFERENCES card (id) ON DELETE CASCADE,
  kind           TEXT NOT NULL CHECK (kind IN ('fact','decision','todo','constraint','preference')),
  subject_entity TEXT,
  predicate      TEXT NOT NULL,
  object         TEXT,
  polarity       INTEGER NOT NULL DEFAULT 1 CHECK (polarity IN (0, 1)),
  due_at         TEXT,
  done_at        TEXT,
  valid_from     TEXT,
  valid_to       TEXT,
  superseded_by  TEXT
);
CREATE INDEX IF NOT EXISTS idx_claim_sp ON claim (subject_entity, predicate);
CREATE INDEX IF NOT EXISTS idx_claim_open ON claim (kind, done_at) WHERE kind = 'todo';

-- ---------------------------------------------------------------------------
-- 实体消解：别名表 + 实体边表（图查询用递归 CTE）
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS entity (
  id             TEXT PRIMARY KEY,
  canonical_name TEXT NOT NULL UNIQUE,
  type           TEXT,
  created_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS entity_alias (
  alias     TEXT PRIMARY KEY,
  entity_id TEXT NOT NULL REFERENCES entity (id) ON DELETE CASCADE,
  source    TEXT NOT NULL DEFAULT 'manual' CHECK (source IN ('manual','auto','profile'))
);
CREATE TABLE IF NOT EXISTS entity_edge (
  id         TEXT PRIMARY KEY,
  src_id     TEXT NOT NULL REFERENCES entity (id) ON DELETE CASCADE,
  rel        TEXT NOT NULL,
  dst_id     TEXT NOT NULL REFERENCES entity (id) ON DELETE CASCADE,
  card_id    TEXT REFERENCES card (id) ON DELETE CASCADE,
  valid_from TEXT,
  valid_to   TEXT
);
CREATE INDEX IF NOT EXISTS idx_edge_src ON entity_edge (src_id, rel);
CREATE INDEX IF NOT EXISTS idx_edge_dst ON entity_edge (dst_id, rel);
CREATE TABLE IF NOT EXISTS card_entity (
  card_id   TEXT NOT NULL REFERENCES card (id) ON DELETE CASCADE,
  entity_id TEXT NOT NULL REFERENCES entity (id) ON DELETE CASCADE,
  PRIMARY KEY (card_id, entity_id)
);

-- ---------------------------------------------------------------------------
-- 压缩游标与运行记录：保证幂等、可续跑、可审计
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS cursor (
  session_id     TEXT PRIMARY KEY,
  last_source_id TEXT,
  last_day       TEXT,
  last_line      INTEGER,
  updated_at     TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS extraction_run (
  id           TEXT PRIMARY KEY,
  session_id   TEXT NOT NULL,
  from_source  TEXT,
  to_source    TEXT,
  model        TEXT,
  prompt_hash  TEXT,
  created_at   TEXT NOT NULL,
  status       TEXT NOT NULL CHECK (status IN ('running','ok','failed'))
);

-- ---------------------------------------------------------------------------
-- 召回访问计费：只有真正进入上下文的卡才计数（候选召回不算）
-- id 用自增主键：派生 ID 会让「同一秒内的重复访问」撞 UNIQUE 约束。
-- ---------------------------------------------------------------------------
CREATE TABLE IF NOT EXISTS card_access (
  id         INTEGER PRIMARY KEY,
  card_id    TEXT NOT NULL REFERENCES card (id) ON DELETE CASCADE,
  session_id TEXT NOT NULL,
  accessed_at TEXT NOT NULL,
  render     TEXT NOT NULL DEFAULT 'full' CHECK (render IN ('full','lite'))
);
CREATE INDEX IF NOT EXISTS idx_access_card_time ON card_access (card_id, accessed_at);

-- ---------------------------------------------------------------------------
-- 可选：全文索引。FTS5 缺失时整个块被跳过，系统退化为纯 SQL LIKE 检索。
-- ---------------------------------------------------------------------------
-- @@optional:fts5
CREATE VIRTUAL TABLE IF NOT EXISTS card_fts USING fts5(
  summary,
  facts,
  entities,
  keywords,
  content='',
  tokenize='unicode61'
);
-- @@end

-- ---------------------------------------------------------------------------
-- 可选：向量索引。需先加载 sqlite-vec 扩展；缺失时 recall 跳过语义一路。
-- ---------------------------------------------------------------------------
-- @@optional:vec0
CREATE VIRTUAL TABLE IF NOT EXISTS card_vec USING vec0(
  card_id TEXT PRIMARY KEY,
  embedding float[768]
);
-- @@end

PRAGMA user_version = 1;
