"""SQLite persistence — one WAL file holds the whole engine state.

Everything here is DERIVED data (the caller's source of truth is wherever the
memories actually live — markdown files, a DB, an app). Dropping the file and
re-indexing is always safe, which is what makes the engine adoptable next to
an existing store without migration risk.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

_SCHEMA = """
CREATE TABLE IF NOT EXISTS nodes(
  id TEXT PRIMARY KEY, kind TEXT DEFAULT 'note', title TEXT DEFAULT '',
  tags TEXT DEFAULT '[]', text_len INT DEFAULT 0, updated_at REAL,
  access_count INT DEFAULT 0, last_access REAL DEFAULT 0,
  pinned INT DEFAULT 0, importance REAL DEFAULT 1.0
);
CREATE TABLE IF NOT EXISTS postings(
  term TEXT, node_id TEXT, tf REAL, PRIMARY KEY(term, node_id)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_postings_node ON postings(node_id);
CREATE TABLE IF NOT EXISTS vectors(node_id TEXT PRIMARY KEY, dim INT, vec BLOB);
CREATE TABLE IF NOT EXISTS teacher_vecs(node_id TEXT PRIMARY KEY, model TEXT, dim INT, vec BLOB);
CREATE TABLE IF NOT EXISTS edges(
  src TEXT, dst TEXT, etype INT, w REAL, updated REAL,
  PRIMARY KEY(src, dst, etype)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_edges_src ON edges(src);
CREATE INDEX IF NOT EXISTS idx_edges_dst ON edges(dst);
CREATE TABLE IF NOT EXISTS params(key TEXT PRIMARY KEY, blob BLOB);
CREATE TABLE IF NOT EXISTS feedback(
  rowid INTEGER PRIMARY KEY AUTOINCREMENT,
  ts REAL, query_hash TEXT, node_id TEXT, features BLOB,
  shown INT DEFAULT 1, used INT DEFAULT 0, label_src TEXT DEFAULT ''
);
"""

EDGE_LINK, EDGE_TAG, EDGE_KNN, EDGE_COACCESS = 0, 1, 2, 3


class Store:
    """Thread-safe (single lock) wrapper over the SQLite state."""

    def __init__(self, path: str = ":memory:") -> None:
        self._conn = sqlite3.connect(path, check_same_thread=False)
        self._lock = threading.RLock()
        with self._lock:
            if path != ":memory:":
                self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    # ── nodes ────────────────────────────────────────────────────────
    def upsert_node(
        self, node_id: str, *, kind: str, title: str, tags: Sequence[str],
        text_len: int, updated_at: float, pinned: bool, importance: float,
    ) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO nodes(id,kind,title,tags,text_len,updated_at,pinned,importance)"
                " VALUES(?,?,?,?,?,?,?,?)"
                " ON CONFLICT(id) DO UPDATE SET kind=excluded.kind,title=excluded.title,"
                " tags=excluded.tags,text_len=excluded.text_len,updated_at=excluded.updated_at,"
                " pinned=excluded.pinned,importance=excluded.importance",
                (node_id, kind, title, json.dumps(list(tags), ensure_ascii=False),
                 text_len, updated_at, int(pinned), importance),
            )
            self._conn.commit()

    def get_node(self, node_id: str) -> Optional[Dict[str, Any]]:
        cur = self._conn.execute(
            "SELECT id,kind,title,tags,text_len,updated_at,access_count,last_access,"
            "pinned,importance FROM nodes WHERE id=?", (node_id,))
        row = cur.fetchone()
        return self._node_row(row) if row else None

    def nodes(self, ids: Optional[Iterable[str]] = None) -> List[Dict[str, Any]]:
        if ids is None:
            cur = self._conn.execute(
                "SELECT id,kind,title,tags,text_len,updated_at,access_count,last_access,"
                "pinned,importance FROM nodes")
            return [self._node_row(r) for r in cur.fetchall()]
        ids = list(ids)
        if not ids:
            return []
        q = ",".join("?" for _ in ids)
        cur = self._conn.execute(
            f"SELECT id,kind,title,tags,text_len,updated_at,access_count,last_access,"
            f"pinned,importance FROM nodes WHERE id IN ({q})", ids)
        return [self._node_row(r) for r in cur.fetchall()]

    @staticmethod
    def _node_row(r: Tuple) -> Dict[str, Any]:
        return {
            "id": r[0], "kind": r[1], "title": r[2], "tags": json.loads(r[3] or "[]"),
            "text_len": r[4], "updated_at": r[5], "access_count": r[6],
            "last_access": r[7], "pinned": bool(r[8]), "importance": r[9],
        }

    def remove_node(self, node_id: str) -> None:
        with self._lock:
            for sql in (
                "DELETE FROM nodes WHERE id=?", "DELETE FROM postings WHERE node_id=?",
                "DELETE FROM vectors WHERE node_id=?", "DELETE FROM teacher_vecs WHERE node_id=?",
                "DELETE FROM edges WHERE src=? ", "DELETE FROM feedback WHERE node_id=?",
            ):
                self._conn.execute(sql, (node_id,))
            self._conn.execute("DELETE FROM edges WHERE dst=?", (node_id,))
            self._conn.commit()

    def touch_access(self, ids: Iterable[str], ts: Optional[float] = None) -> None:
        ts = ts or time.time()
        with self._lock:
            self._conn.executemany(
                "UPDATE nodes SET access_count=access_count+1,last_access=? WHERE id=?",
                [(ts, i) for i in ids])
            self._conn.commit()

    def count_nodes(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM nodes").fetchone()[0])

    # ── postings (BM25) ──────────────────────────────────────────────
    def replace_postings(self, node_id: str, tf: Dict[str, float]) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM postings WHERE node_id=?", (node_id,))
            self._conn.executemany(
                "INSERT OR REPLACE INTO postings(term,node_id,tf) VALUES(?,?,?)",
                [(t, node_id, f) for t, f in tf.items()])
            self._conn.commit()

    def postings_for_terms(self, terms: Sequence[str]) -> Dict[str, List[Tuple[str, float]]]:
        if not terms:
            return {}
        q = ",".join("?" for _ in terms)
        cur = self._conn.execute(
            f"SELECT term,node_id,tf FROM postings WHERE term IN ({q})", list(terms))
        out: Dict[str, List[Tuple[str, float]]] = {}
        for term, node_id, tf in cur.fetchall():
            out.setdefault(term, []).append((node_id, tf))
        return out

    def doc_lens(self) -> Dict[str, int]:
        cur = self._conn.execute("SELECT id,text_len FROM nodes")
        return {r[0]: r[1] for r in cur.fetchall()}

    # ── vectors ──────────────────────────────────────────────────────
    def put_vector(self, node_id: str, vec: bytes, dim: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO vectors(node_id,dim,vec) VALUES(?,?,?)",
                (node_id, dim, vec))
            self._conn.commit()

    def all_vectors(self) -> List[Tuple[str, int, bytes]]:
        return list(self._conn.execute("SELECT node_id,dim,vec FROM vectors").fetchall())

    def put_teacher(self, node_id: str, model: str, vec: bytes, dim: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO teacher_vecs(node_id,model,dim,vec) VALUES(?,?,?,?)",
                (node_id, model, dim, vec))
            self._conn.commit()

    def teachers(self) -> List[Tuple[str, str, int, bytes]]:
        return list(self._conn.execute(
            "SELECT node_id,model,dim,vec FROM teacher_vecs").fetchall())

    # ── edges ────────────────────────────────────────────────────────
    def upsert_edges(self, rows: Iterable[Tuple[str, str, int, float]]) -> None:
        ts = time.time()
        with self._lock:
            self._conn.executemany(
                "INSERT INTO edges(src,dst,etype,w,updated) VALUES(?,?,?,?,?)"
                " ON CONFLICT(src,dst,etype) DO UPDATE SET w=excluded.w,updated=excluded.updated",
                [(s, d, t, w, ts) for s, d, t, w in rows])
            self._conn.commit()

    def replace_edges_from(self, node_id: str, etype: int,
                           rows: Iterable[Tuple[str, float]]) -> None:
        ts = time.time()
        with self._lock:
            self._conn.execute(
                "DELETE FROM edges WHERE src=? AND etype=?", (node_id, etype))
            self._conn.executemany(
                "INSERT OR REPLACE INTO edges(src,dst,etype,w,updated) VALUES(?,?,?,?,?)",
                [(node_id, d, etype, w, ts) for d, w in rows])
            self._conn.commit()

    def edges_by_type(self, etype: int) -> List[Tuple[str, str, float, float]]:
        return list(self._conn.execute(
            "SELECT src,dst,w,updated FROM edges WHERE etype=?", (etype,)).fetchall())

    def get_edge(self, src: str, dst: str, etype: int) -> Optional[Tuple[float, float]]:
        row = self._conn.execute(
            "SELECT w,updated FROM edges WHERE src=? AND dst=? AND etype=?",
            (src, dst, etype)).fetchone()
        return (row[0], row[1]) if row else None

    def set_edge(self, src: str, dst: str, etype: int, w: float) -> None:
        self.upsert_edges([(src, dst, etype, w)])

    def prune_edges(self, etype: int, floor: float) -> int:
        with self._lock:
            cur = self._conn.execute(
                "DELETE FROM edges WHERE etype=? AND w<?", (etype, floor))
            self._conn.commit()
            return cur.rowcount

    # ── params / feedback ────────────────────────────────────────────
    def put_param(self, key: str, blob: bytes) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO params(key,blob) VALUES(?,?)", (key, blob))
            self._conn.commit()

    def get_param(self, key: str) -> Optional[bytes]:
        row = self._conn.execute("SELECT blob FROM params WHERE key=?", (key,)).fetchone()
        return row[0] if row else None

    def add_feedback(self, query_hash: str, node_id: str, features: bytes,
                     used: bool, label_src: str, cap: int) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO feedback(ts,query_hash,node_id,features,shown,used,label_src)"
                " VALUES(?,?,?,?,1,?,?)",
                (time.time(), query_hash, node_id, features, int(used), label_src))
            # FIFO cap.
            self._conn.execute(
                "DELETE FROM feedback WHERE rowid <= "
                "(SELECT MAX(rowid) FROM feedback) - ?", (cap,))
            self._conn.commit()

    def feedback_rows(self, limit: int) -> List[Tuple[str, bytes, int]]:
        return list(self._conn.execute(
            "SELECT query_hash,features,used FROM feedback ORDER BY rowid DESC LIMIT ?",
            (limit,)).fetchall())

    def feedback_count(self) -> int:
        return int(self._conn.execute("SELECT COUNT(*) FROM feedback").fetchone()[0])

    def close(self) -> None:
        with self._lock:
            self._conn.commit()
            self._conn.close()
