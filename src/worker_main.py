import os
import json
import random
import time
import pika

from .workers.classify_reel_video import handle_classify_reel_video_job
from .workers.crawl_instagram_account import handle_crawl_instagram_account_job
from .expansion_service import expand_niche_if_needed
from .jobs import (
    JOB_CLASSIFY_REEL_VIDEO,
    JOB_CRAWL_INSTAGRAM_ACCOUNT,
    JOB_EXPAND_NICHE,
    JOB_ENRICH_MISSING_POST_METRICS,
    QUEUE_EXPAND_JOBS,
    QUEUE_CRAWL_JOBS,
    QUEUE_CLASSIFY_JOBS,
    QUEUE_ENRICH_JOBS,
    update_crawl_job_status,
)

RABBITMQ_URL = os.getenv(
    "RABBITMQ_URL",
    "amqp://guest:guest@localhost:5672/%2F",
)

def get_worker_queue_name() -> str:
    worker_mode = (os.getenv("WORKER_MODE") or "all").strip().lower()

    if worker_mode == "expand":
        return QUEUE_EXPAND_JOBS
    
    if worker_mode == "crawl":
        return QUEUE_CRAWL_JOBS
    
    if worker_mode == "classify":
        return QUEUE_CLASSIFY_JOBS
    
    if worker_mode == "enrich":
        return QUEUE_ENRICH_JOBS
    
    return "content_jobs"

def handle_enrich_missing_post_metrics_job(payload: dict) -> dict:
    """
    Backfill engagement metrics for older posts that were saved before
    like/comment/view metrics existed.
    """
    from .app import get_db, get_seed_account_metrics, utc_now_iso
    from .playwright_helper import fetch_instagram_post_metadata_playwright

    platform = (payload.get("platform") or "instagram").strip().lower()
    limit = int(payload.get("limit") or 10)
    limit = max(1, min(limit, 25))

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT
                id,
                platform,
                account_handle,
                post_url,
                like_count,
                comment_count,
                view_count,
                engagement_score,
                normalized_engagement_score,
                metrics_collected_at
            FROM posts
            WHERE platform = ?
              AND post_url != ''
              AND (
                    metrics_collected_at = ''
                    OR (
                        like_count = 0
                        AND comment_count = 0
                        AND view_count = 0
                    )
                  )
            ORDER BY collected_at DESC, id DESC
            LIMIT ?
            """,
            (platform, limit),
        ).fetchall()

    print(f"[ENRICH] Found {len(rows)} posts missing metrics")

    updated_count = 0
    skipped_count = 0

    for row in rows:
        post_id = row["id"]
        post_url = row["post_url"]
        account_handle = row["account_handle"] or ""

        try:
            print(f"[ENRICH] Fetching metrics for post #{post_id}: {post_url}")
            metadata = fetch_instagram_post_metadata_playwright(post_url) or {}

            like_count = int(metadata.get("like_count") or 0)
            comment_count = int(metadata.get("comment_count") or 0)
            view_count = int(metadata.get("view_count") or 0)

            if not (like_count or comment_count or view_count):
                skipped_count += 1
                print(f"[ENRICH] No metrics found for post #{post_id}")
                time.sleep(random.randint(5, 12))
                continue

            raw_engagement_points = like_count + (comment_count * 2)
            engagement_score = 0.0

            if view_count > 0:
                engagement_score = round((raw_engagement_points / view_count) * 100, 4)
            elif raw_engagement_points:
                engagement_score = float(raw_engagement_points)

            creator_metrics = get_seed_account_metrics(platform, account_handle)
            follower_count = int(creator_metrics.get("follower_count") or 0)
            normalized_engagement_score = 0.0

            if follower_count > 0 and raw_engagement_points > 0:
                normalized_engagement_score = round((raw_engagement_points / follower_count) * 100, 4)

            with get_db() as conn:
                conn.execute(
                    """
                    UPDATE posts
                    SET
                        like_count = ?,
                        comment_count = ?,
                        view_count = ?,
                        engagement_score = ?,
                        normalized_engagement_score = ?,
                        metrics_collected_at = ?
                    WHERE id = ?
                    """,
                    (
                        like_count,
                        comment_count,
                        view_count,
                        engagement_score,
                        normalized_engagement_score,
                        utc_now_iso(),
                        post_id,
                    ),
                )

            updated_count += 1
            print(
                f"[ENRICH] Updated post #{post_id}: "
                f"likes={like_count}, comments={comment_count}, views={view_count}, "
                f"score={engagement_score}, normalized={normalized_engagement_score}"
            )

            time.sleep(random.randint(8, 18))

        except Exception as e:
            skipped_count += 1
            print(f"[ENRICH ERROR] post #{post_id}: {e}")
            time.sleep(random.randint(15, 30))

    return {
        "ok": True,
        "platform": platform,
        "checked_count": len(rows),
        "updated_count": updated_count,
        "skipped_count": skipped_count,
    }

def summarize_worker_result(result: dict) -> dict:
    if not isinstance(result, dict):
        return {"result": result}
    
    return {
        "ok": result.get("ok"),
        "expanded": result.get("expanded"),
        "reason": result.get("reason"),
        "queued_count": result.get("queued_count"),
        "skipped_count": result.get("skipped_count"),
        "discovery_limit": result.get("discovery_limit"),
        "niche_health": result.get("niche_health"),
        "discovered_accounts_count": len(result.get("discovered_accounts") or []),
        "existing_seed_accounts_count": len(result.get("existing_seed_accounts") or []),
        "selected_accounts_count": len(result.get("selected_accounts") or []),
        "crawl_jobs_count": len(result.get("crawl_jobs") or []),
    }


def main():
    params = pika.URLParameters(RABBITMQ_URL)
    params.heartbeat = 0
    params.blocked_connection_timeout = 300
    connection = pika.BlockingConnection(params)
    channel = connection.channel()
    channel.basic_qos(prefetch_count=1)

    queue_name = get_worker_queue_name()

    channel.queue_declare(
        queue=queue_name,
        durable=True,
    )

    print(
        f"worker started."
        f"Mode={os.getenv('WORKER_MODE') or 'all'} "
        f"Queue={queue_name}. "
        f"Waiting for jobs..."
    )

    def callback(ch, method, properties, body):
        try:
            job = json.loads(body)
            job_type = job.get("job_type")
            payload = job.get("payload", {})
            job_id = payload.get("job_id") or job.get("job_id") or job.get("id")

            print(f"\nReceived job: {job_type}")

            if job_id:
                update_crawl_job_status(
                    job_id=job_id,
                    status="processing",
                )

            if job_type == JOB_CLASSIFY_REEL_VIDEO:
                handle_classify_reel_video_job(payload)

            elif job_type == JOB_CRAWL_INSTAGRAM_ACCOUNT:
                handle_crawl_instagram_account_job(payload)

            elif job_type == JOB_EXPAND_NICHE:
                result = expand_niche_if_needed(
                    platform=payload.get("platform", "instagram"),
                    style=payload.get("style", "carousel"),
                    niche=payload.get("niche", ""),
                    limit=int(payload.get("crawl_accounts", 10)),
                )
                print(f"Expand niche summary: {summarize_worker_result(result)}")

            elif job_type == JOB_ENRICH_MISSING_POST_METRICS:
                result = handle_enrich_missing_post_metrics_job(payload)
                print(f"Enrich missing post metrics summary: {result}")

            else:
                print(f"Unknown job type: {job_type}")

            if job_id:
                update_crawl_job_status(
                    job_id=job_id,
                    status="completed",
                )

            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            print(f"Worker error: {e}")
            try:
                failed_job = json.loads(body)
                failed_payload = failed_job.get("payload", {})
                failed_job_id = failed_payload.get("job_id") or failed_job.get("job_id") or failed_job.get("id")
                if failed_job_id:
                    update_crawl_job_status(
                        job_id=failed_job_id,
                        status="failed",
                        error_message=str(e),
                    )
            except Exception as status_error:
                print(f"Failed to update job status after error: {status_error}")

            try:
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as ack_error:
                print(f"Ack failed after worker error: {ack_error}")

    channel.basic_consume(
        queue=queue_name,
        on_message_callback=callback,
    )
    channel.start_consuming()


if __name__ == "__main__":
    main()