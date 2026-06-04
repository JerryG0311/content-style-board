import os
import json
from typing import Optional
import sqlite3
from pathlib import Path
from contextlib import contextmanager
from datetime import datetime, timezone, timedelta

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
JOB_EXPAND_NICHE = "expand_niche"
JOB_ENRICH_MISSING_POST_METRICS = "enrich_missing_post_metrics"

# Queue names for different job types
QUEUE_EXPAND_JOBS = "content_expand_jobs"
QUEUE_CRAWL_JOBS = "content_crawl_jobs"
QUEUE_CLASSIFY_JOBS = "content_classify_jobs"
QUEUE_ENRICH_JOBS = "content_enrich_jobs"


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
            INSERT INTO crawl_jobs (job_type, target, status, created_at, retry_count)
            VALUES (?, ?, ?, ?, 0)
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

def get_crawl_job(job_id: int) -> dict:
    """
    Fetch one crawl_jobs row by id.
    """
    if not job_id:
        raise ValueError("job_id is required")
    
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM crawl_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    return dict(row) if row else {}

def increment_crawl_job_retry(job_id: int) -> dict:
    """
    Increment retry_count for a job and return updated row.
    """
    if not job_id:
        raise ValueError("job_id is required")
    
    with get_db() as conn:
        conn.execute(
            """
            UPDATE crawl_jobs
            SET retry_count = retry_count + 1
            WHERE id = ?
            """,
            (job_id,),
        )
        row = conn.execute(
            "SELECT * FROM crawl_jobs WHERE id = ?",
            (job_id,),
        ).fetchone()
    return dict(row) if row else {}

def has_recent_or_active_expand_job(
        platform: str,
        style: str,
        niche: str,
        cooldown_minutes: int = 30,
) -> bool:
    platform = (platform or "instagram").strip().lower()
    style = (style or "carousel").strip().lower()
    niche = (niche or "").strip()

    if not niche:
        return False
    
    target = f"{platform}:{style}:{niche}"
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=cooldown_minutes)).isoformat()

    with get_db() as conn:
        row = conn.execute(
            """
            SELECT id
            FROM crawl_jobs
            WHERE job_type = ?
                AND target = ?
                AND (
                        status IN ('queued', 'processing', 'running')
                        OR created_at >= ?
                    )
            ORDER BY id DESC
            LIMIT 1
            """,
            (JOB_EXPAND_NICHE, target, cutoff),
        ).fetchone()
    
    return row is not None

def queue_expand_niche_job_if_needed(
        platform: str,
        style: str,
        niche: str,
        discovery_limit: int = 16,
        crawl_accounts: int = 10,
        posts_per_account: int = 24,
        cooldown_minutes: int = 30, 
) -> dict:
    platform = (platform or "instagram").strip().lower()
    style = (style or "carousel").strip().lower()
    niche = (niche or "").strip()

    if not niche:
        raise ValueError("niche is required")
    
    if has_recent_or_active_expand_job(
        platform=platform,
        style=style,
        niche=niche,
        cooldown_minutes=cooldown_minutes,
    ):
        return {
            "queued": False,
            "reason": "recent_or_active_expand_job_exists",
            "platform": platform,
            "style": style,
            "niche": niche,
            "cooldown_minutes": cooldown_minutes,
        }
    
    job = queue_expand_niche_job(
        platform=platform,
        style=style,
        niche=niche,
        discovery_limit=discovery_limit,
        crawl_accounts=crawl_accounts,
        posts_per_account=posts_per_account,
    )

    return {
        "queued": True,
        "reason": "expand_niche_job_queued",
        "job": job,
        "platform": platform,
        "style": style,
        "niche": niche,
        "cooldown_minutes": cooldown_minutes,
    }

def has_recent_or_active_enrichment_job(
        platform: str = "instagram",
        cooldown_minutes: int = 60,
) -> bool:
    platform = (platform or "instagram").strip().lower()
    target = f"{platform}:missing_post_metrics"
    cutoff = (
        datetime.now(timezone.utc)
        - timedelta(minutes=cooldown_minutes)
    ).isoformat()

    with get_db() as conn:
        row = conn.execute(
            """
            SELECT id
            FROM crawl_jobs
            WHERE job_type = ?
                AND target = ?
                AND (
                        status IN ('queued', 'processing', 'running')
                    )
            ORDER BY id DESC
            LIMIT 1
            """,
            (
                JOB_ENRICH_MISSING_POST_METRICS,
                target,
                cutoff,
            ),
        ).fetchone()
    
    return row is not None

def queue_enrich_missing_post_metrics_job_if_needed(
        platform: str = "instagram",
        limit: int = 25,
        cooldown_minutes: int = 60,
) -> dict:
    platform = (platform or "instagram").strip().lower()
    limit = int(limit or 25)

    if has_recent_or_active_enrichment_job(
        platform=platform,
        cooldown_minutes=cooldown_minutes,
    ):
        return {
            "queued": False, 
            "reason": "recent_or_active_enrichment_job_exists",
            "platform": platform,
            "limit": limit,
            "cooldown_minutes": cooldown_minutes,
        }
    
    job = queue_enrich_missing_post_metrics_job(
        platform=platform,
        limit=limit,
    )

    return {
        "queued": True,
        "reason": "enrich_missing_post_metrics_job_queued",
        "job": job,
        "platform": platform,
        "limit": limit,
        "cooldown_minutes": cooldown_minutes,
    }

def get_queue_name_for_job_type(job_type: str) -> str:
    job_type = (job_type or "").strip()

    if job_type == JOB_EXPAND_NICHE:
        return QUEUE_EXPAND_JOBS

    if job_type == JOB_CRAWL_INSTAGRAM_ACCOUNT:
        return QUEUE_CRAWL_JOBS

    if job_type == JOB_CLASSIFY_REEL_VIDEO:
        return QUEUE_CLASSIFY_JOBS

    if job_type == JOB_ENRICH_MISSING_POST_METRICS:
        return QUEUE_ENRICH_JOBS

    return "content_jobs"

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
    queue_name = get_queue_name_for_job_type(job_type)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.queue_declare(queue=queue_name, durable=True)

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
        routing_key=queue_name,
        body=body,
        properties=pika.BasicProperties(delivery_mode=2),
    )

    connection.close()


def queue_expand_niche_job(
    platform: str,
    style: str,
    niche: str,
    discovery_limit: int = 16,
    crawl_accounts: int = 10,
    posts_per_account: int = 24,
) -> dict:
    """
    Create and publish a background job that expands a niche content pool.
    Search can return immediately while workers discover/crawl/classify content.
    """
    platform = (platform or "instagram").strip().lower()
    style = (style or "carousel").strip().lower()
    niche = (niche or "").strip()

    if not niche:
        raise ValueError("niche is required")

    target = f"{platform}:{style}:{niche}"
    payload = {
        "platform": platform,
        "style": style,
        "niche": niche,
        "discovery_limit": int(discovery_limit),
        "crawl_accounts": int(crawl_accounts),
        "posts_per_account": int(posts_per_account),
    }

    job = create_crawl_job(
        job_type=JOB_EXPAND_NICHE,
        target=target,
        status="queued",
    )

    publish_rabbitmq_job(
        job_type=JOB_EXPAND_NICHE,
        target=target,
        payload={
            **payload,
            "job_id": job.get("id"),
        },
    )

    return job

def queue_enrich_missing_post_metrics_job(
    platform: str = "instagram",
    limit: int = 25,
) -> dict:
    """
    Create and publish a background job that backfills engagement metrics
    for existing posts missing likes/comments/views/engagement metadata.
    """
    platform = (platform or "instagram").strip().lower()
    limit = int(limit or 25)

    target = f"{platform}:missing_post_metrics"

    payload = {
        "platform": platform,
        "limit": limit,
    }

    job = create_crawl_job(
        job_type=JOB_ENRICH_MISSING_POST_METRICS,
        target=target,
        status="queued",
    )

    publish_rabbitmq_job(
        job_type=JOB_ENRICH_MISSING_POST_METRICS,
        target=target,
        payload={
            **payload,
            "job_id": job.get("id"),
        },
    )

    return job


def queue_classify_reel_video_job(
    post_url: str,
    fps: float = 1.0,
) -> dict:
    """
    Create and publish a background job that classifies one Instagram reel.
    """
    post_url = (post_url or "").strip()

    if not post_url:
        raise ValueError("post_url is required")

    job = create_crawl_job(
        job_type=JOB_CLASSIFY_REEL_VIDEO,
        target=post_url,
        status="queued",
    )

    publish_rabbitmq_job(
        job_type=JOB_CLASSIFY_REEL_VIDEO,
        target=post_url,
        payload={
            "job_id": job.get("id"),
            "post_url": post_url,
            "fps": float(fps or 1.0),
        },
    )

    return job


def queue_missing_reel_classification_jobs(
    limit: int = 25,
    fps: float = 1.0,
) -> dict:
    """
    Queue classification jobs for reels that have not been visually classified yet.
    """
    limit = int(limit or 25)
    limit = max(1, min(limit, 100))

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT p.post_url
            FROM posts p
            WHERE p.post_type = 'reel'
              AND p.post_url != ''
              AND COALESCE(p.classified_post_type, '') = ''
              AND NOT EXISTS (
                  SELECT 1
                  FROM crawl_jobs cj
                  WHERE cj.job_type = ?
                    AND cj.target = p.post_url
                    AND cj.status IN ('queued', 'processing', 'running')
              )
            ORDER BY p.collected_at DESC, p.id DESC
            LIMIT ?
            """,
            (JOB_CLASSIFY_REEL_VIDEO, limit),
        ).fetchall()

    queued = 0

    for row in rows:
        post_url = (row["post_url"] or "").strip()
        if not post_url:
            continue

        queue_classify_reel_video_job(
            post_url=post_url,
            fps=fps,
        )

        queued += 1
        print(f"[CLASSIFY QUEUE] Queued reel classification: {post_url}")

    return {
        "ok": True,
        "found": len(rows),
        "queued": queued,
        "limit": limit,
        "fps": fps,
    }