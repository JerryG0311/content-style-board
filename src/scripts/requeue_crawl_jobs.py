from src.jobs import (
    JOB_CRAWL_INSTAGRAM_ACCOUNT,
    get_db,
    publish_rabbitmq_job,
)

DEFAULT_LIMIT = 200

def requeue_crawl_jobs(limit: int = DEFAULT_LIMIT) -> dict:
    limit = int(limit or DEFAULT_LIMIT)

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT id, job_type, target, status
            FROM crawl_jobs
            WHERE job_type = ?
                AND status = 'queued'
            ORDER BY id ASC
            LIMIT ?
            """,
            (
                JOB_CRAWL_INSTAGRAM_ACCOUNT,
                limit,
            ),
        ).fetchall()
    
    requeued = 0

    for row in rows:
        job_id = row["id"]
        target = row["target"] or ""

        handle = target.strip().lstrip("@")

        if not handle:
            print(
                f"[REQUEUE] Skipping job #{job_id}: missing target"
            )
            continue

        publish_rabbitmq_job(
            job_type=JOB_CRAWL_INSTAGRAM_ACCOUNT,
            target=target,
            payload={
                "job_id": job_id,
                "platform": "instagram",
                "handle": handle,
            },
        )

        requeued += 1

        print(
            f"[REQUEUE] Requeued crawl job #{job_id}: @{handle}"
        )
    
    return {
        "ok": True,
        "found": len(rows),
        "requeued": requeued,
        "limit": limit,
    }

if __name__ == "__main__":
    result = requeue_crawl_jobs()
    print(result)
