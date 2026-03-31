from ..app import collect_instagram_seed_account
from ..jobs import update_crawl_job_status, utc_now_iso

def handle_crawl_instagram_account_job(payload: dict):
    """
    Worker handler for crawling a single Instagram seed account.
    """
    job_id = payload.get("job_id") or 0
    platform = (payload.get("platform") or "").lower().strip()
    handle = (payload.get("handle") or "").strip().lstrip("@")
    niche = (payload.get("niche") or "").strip()
    if platform != "instagram":
        raise ValueError(f"Unsupported platform: {platform}")
    if not handle:
        raise ValueError("Mising handle in payload")
    if job_id:
        update_crawl_job_status(
            job_id=job_id,
            status="processing",
            started_at=utc_now_iso(),
        )
    
    try:
        collect_instagram_seed_account(handle=handle, niche=niche)
        if job_id:
            update_crawl_job_status(
                job_id=job_id,
                status="completed",
                finished_at=utc_now_iso(),
            )
        print(f"Done crawling account: {handle}")
    except Exception as e:
        if job_id:
            update_crawl_job_status(
                job_id=job_id,
                status="failed",
                error_message=str(e),
                finished_at=utc_now_iso(),
            )
        print(f"Worker failed for account crawl: {handle}")
        print(e)
        raise
    