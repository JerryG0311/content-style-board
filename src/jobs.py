import os
import json
from typing import Optional
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone

DATA_DIR = Path("data")
DB_FILE = DATA_DIR / "app.db"

def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()

@contextmanager
def get_db():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

# Job types
JOB_DISCOVER_ACCOUNTS_FOR_NICHE = "discover_accounts_for_niche"
JOB_CRAWL_INSTAGRAM_ACCOUNT = "crawl_instagram_account"
JOB_CLASSIFY_REEL_VIDEO = "classify_reel_video"


def create_crawl_job(job_type: str, target: str, status: str = "queued") -> dict:
    """
    Store an async job record in SQLite so the app can track what was queued.
    """
    job_type = (job_type or "").strip()
    target = (target or "").strip()
    status = (status or "queued").strip()
    now = utc_now_iso()

    if not job_type:
        raise ValueError("job_type is required")
    if not target:
        raise ValueError("target is required")

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO crawl_jobs (job_type, target, status, created_at)
            VALUES (?, ?, ?, ?)
            """,
            (job_type, target, status, now),
        )
        row = conn.execute(
            "SELECT * FROM crawl_jobs ORDER BY id DESC LIMIT 1"
        ).fetchone()

    return dict(row) if row else {}

def update_crawl_job_status(
        job_id: int,
        status: str,
        error_message: str = "",
        started_at: str = "",
        finished_at: str = "",
) -> dict:
    """
    Update a crawl_jobs row as work moves through the worker lifecycle.
    """
    if not job_id:
        raise ValueError("job_id is required")
    
    status = (status or "").strip()
    error_message = (error_message or "").strip()
    started_at = (started_at or "").strip()
    finished_at = (finished_at or "").strip()

    with get_db() as conn:
        conn.execute(
            """
            UPDATE crawl_jobs
            SET status = ?,
                error_message = ?,
                started_at = CASE WHEN ? <> '' THEN ? ELSE started_at END,
                finished_at = CASE WHEN ? <> '' THEN ? ELSE finished_at END
            WHERE id = ?
            """,
            (
                status,
                error_message,
                started_at,
                started_at,
                finished_at,
                finished_at,
                job_id,
            ),
        )
        row = conn.execute(
            "SELECT * FROM crawl_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    
    return dict(row) if row else {}


def publish_rabbitmq_job(job_type: str, target: str, payload: Optional[dict] = None) -> None:
    """
    Publish a durable job message to RabbitMQ.
    """
    payload = payload or {}
    rabbitmq_url = (os.getenv("RABBITMQ_URL") or "amqp://guest:guest@localhost:5672/%2F").strip()

    try:
        import pika
    except Exception as e:
        raise RuntimeError(
            "RabbitMQ publishing requires pika. Install it with: pip install pika"
        ) from e

    params = pika.URLParameters(rabbitmq_url)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue="content_jobs", durable=True)

    body = json.dumps(
        {
            "job_type": job_type,
            "target": target,
            "payload": payload,
            "created_at": utc_now_iso(),
        }
    )

    channel.basic_publish(
        exchange="",
        routing_key="content_jobs",
        body=body,
        properties=pika.BasicProperties(delivery_mode=2),
    )

    connection.close()