"""
Seed Data Hub với test data để demo UI.
Chạy một lần: python scripts/seed_datahub.py
Không cần backend server.
"""
import sqlite3
import json
import uuid
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

_ROOT = Path(__file__).parent.parent
DB_PATH = str(_ROOT / "backend" / "storage" / "cv_flow.db")
STORAGE = _ROOT / "backend" / "storage"

def seed_relational():
    conn = sqlite3.connect(DB_PATH)
    now = datetime.now(timezone.utc)

    # 5 execution sessions
    sessions = [
        (str(uuid.uuid4()), "pipeline-001", (now - timedelta(hours=2)).isoformat(),
         (now - timedelta(hours=1, minutes=50)).isoformat(), "completed", 3600, None, "sequential"),
        (str(uuid.uuid4()), "pipeline-001", (now - timedelta(hours=1)).isoformat(),
         (now - timedelta(minutes=48)).isoformat(), "completed", 2880, None, "multiprocess"),
        (str(uuid.uuid4()), "pipeline-002", (now - timedelta(minutes=30)).isoformat(),
         (now - timedelta(minutes=25)).isoformat(), "error", 0, "ONNX model not found", "sequential"),
        (str(uuid.uuid4()), "pipeline-001", (now - timedelta(minutes=10)).isoformat(),
         None, "running", 0, None, "sequential"),
        (str(uuid.uuid4()), "pipeline-003", (now - timedelta(days=1)).isoformat(),
         (now - timedelta(hours=23)).isoformat(), "completed", 7200, None, "sequential"),
    ]

    conn.executemany("""
        INSERT OR IGNORE INTO execution_sessions
        (id, pipeline_id, started_at, ended_at, status, frame_count, error_msg, mode)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, sessions)

    # node_metrics cho 2 completed sessions
    for sid, _, _, _, status, fc, _, mode in sessions:
        if status != "completed": continue
        conn.executemany("""
            INSERT INTO node_metrics (session_id, node_id, avg_ms, p95_ms, fps, errors, recorded_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, [
            (sid, "face_detect", 45.2, 62.1, 22.0, 0, now.isoformat()),
            (sid, "face_embed",  12.8, 18.4, 22.0, 0, now.isoformat()),
        ])

    # 50 detection events
    event_types = ["face_matched", "face_unknown", "counter_update"]
    for i in range(50):
        event_sid = sessions[0][0]
        conn.execute("""
            INSERT INTO detection_events (session_id, pipeline_id, node_id, timestamp, event_type, payload_json)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            event_sid, "pipeline-001", "face_detect",
            (now - timedelta(seconds=50-i)).isoformat(),
            event_types[i % 3],
            json.dumps({"bbox": [100+i,100,200+i,200], "confidence": 0.9, "identity": f"person_{i%5}"})
        ))

    conn.commit()
    conn.close()
    print("✅ Relational DB seeded")

def seed_vector():
    vectordb_dir = STORAGE / "vectordb"
    vectordb_dir.mkdir(parents=True, exist_ok=True)

    # Collection: faces (empty)
    (vectordb_dir / "faces").mkdir(exist_ok=True)
    np.save(str(vectordb_dir / "faces" / "index.npy"), np.zeros((0, 512), dtype=np.float32))
    (vectordb_dir / "faces" / "meta.json").write_text("[]")

    # Collection: test_embeddings (3 random 512-dim vecs)
    (vectordb_dir / "test_embeddings").mkdir(exist_ok=True)
    vecs = np.random.randn(3, 512).astype(np.float32)
    vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
    np.save(str(vectordb_dir / "test_embeddings" / "index.npy"), vecs)
    meta = [
        {"id": "vec_001", "label": "Test Vector 1", "source": "seed"},
        {"id": "vec_002", "label": "Test Vector 2", "source": "seed"},
        {"id": "vec_003", "label": "Test Vector 3", "source": "seed"},
    ]
    (vectordb_dir / "test_embeddings" / "meta.json").write_text(json.dumps(meta, indent=2))
    print("✅ Vector DB seeded: faces (0 vecs), test_embeddings (3 vecs)")

if __name__ == "__main__":
    seed_relational()
    seed_vector()
    print("✅ Data Hub seed complete")
