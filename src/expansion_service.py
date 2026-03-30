from .discovery_service import discover_instagram_accounts_for_niche
from .jobs import (
    JOB_CRAWL_INSTAGRAM_ACCOUNT,
    create_crawl_job,
    get_db,
    publish_rabbitmq_job,
    utc_now_iso,
)
from .search_service import get_niche_health

def needs_niche_expansion(platform: str, style: str, niche: str = "") -> dict:
    """
    Decide whether the local DB is still too thin for this niche/style.

    Intentionally kept this logic separate from /api/search so expansion can be
    triggered from routes, workers, cron jobs, or future internal dashboards.
    """

    health = get_niche_health(platform=platform, style=style, niche=niche)
    return {
        "platform": platform,
        "style": style,
        "niche": niche,
        "healthy": bool(health.get("healthy")),
        "niche_health": health,
        "should_expand": not bool(health.get("healthy")),
    }

def create_seed_account_if_missing(platform: str, handle: str, niche: str = "") -> dict:
    """
    Insert a discovered account into seed_accounts only if it does not already exist.
    """

    platform = (platform or "").lower().strip()
    handle = (handle or "").strip().lstrip("@")
    niche = (niche or "").strip()
    if not platform:
        raise ValueError("platform is required")
    if not handle:
        raise ValueError("handle is required")
    
    with get_db() as conn:
        existing = conn.execute(
            """
            SELECT *
            FROM seed_accounts
            WHERE platform = ? AND LOWER(handle) = LOWER(?)
            LIMIT 1
            """,
            (platform, handle),
        ).fetchone()
        if existing:
            return {
                "created": False,
                "seed_account": dict(existing),
            }
        
        now = utc_now_iso()
        conn.execute(
            """
            INSERT INTO seed_accounts (platform, handle, niche, is_active, created_at, last_crawled_at)
            VALUES (?, ?, ?, 1, ?, NULL)
            """,
            (platform, handle, niche, now),
        )
        row = conn.execute(
            """
            SELECT *
            FROM seed_accounts
            WHERE platform = ? AND LOWER(handle) = LOWER(?)
            ORDER BY id DESC
            LIMIT 1
            """,
            (platform, handle),
        ).fetchone()
    return {
        "created": True,
        "seed_account": dict(row) if row else {},
    }

def enqueue_account_crawl(platform: str, handle: str, niche: str = "") -> dict:
    """
    Queue a crawl job for a discovered Instagram account, with dedupe protection.
    """

    platform = (platform or "").lower().strip()
    handle = (handle or "").strip().lstrip("@")
    niche = (niche or "").strip()
    if not platform:
        raise ValueError("platform is required")
    if not handle:
        raise ValueError("handle is required")
    
    target = f"{platform}:{handle}"

    with get_db() as conn:
        existing = conn.execute(
            """
            SELECT id
            FROM crawl_jobs
            WHERE target = ?
                AND job_type = ?
                AND status IN ('queued', 'processing')
            LIMIT 1
            """,
            (target, JOB_CRAWL_INSTAGRAM_ACCOUNT),
        ).fetchone()
    if existing:
        return {
            "queued": False,
            "reason": "already_active",
            "existing_job_id": existing[0],
        }
    
    job = create_crawl_job(
        job_type=JOB_CRAWL_INSTAGRAM_ACCOUNT,
        target=target,
        status="queued",
    )
    publish_rabbitmq_job(
        job_type=JOB_CRAWL_INSTAGRAM_ACCOUNT,
        target=target,
        payload={
            "job_id": job["id"],
            "platform": platform,
            "handle": handle,
            "niche": niche,
        },
    )

    return {
        "queued": True,
        "job": job, 
    }

def expand_niche_if_needed(platform: str, style: str, niche: str, limit: int = 10) -> dict:
    """
    Main niche-expansion entry point.

    Flow:
    1. Check whether the DB is already healthy for this niche/style.
    2. If not, discover candidate accounts for the niche.
    3. Insert new seed accounts.
    4. Enqueue crawl jobs for those accounts.
    """

    platform = (platform or "").lower().strip()
    style = (style or "").lower().strip()
    niche = (niche or "").strip()
    limit = int(limit or 10)
    
    decision = needs_niche_expansion(platform=platform, style=style, niche=niche)
    if not decision.get("should_expand"):
        return {
            "ok": True,
            "expanded": False, 
            "reason": "niche_already_healthy",
            "discovered_accounts": [],
            "seed_results": [],
            "crawl_jobs": [],
        }
    
    if platform != "instagram":
        return {
            "ok": False, 
            "expanded": False,
            "reason": f"unsupported_platform: {platform}",
            "niche_health": decision.get("niche_health", {}),
            "discovered_accounts": [],
            "seed_results": [],
            "crawl_jobs": [], 
        }
    
    discovered_accounts = discover_instagram_accounts_for_niche(niche=niche, limit=limit)
    seed_results = []
    crawl_jobs = []
    for account in discovered_accounts:
        handle = (account.get("handle") or "").strip().lstrip("@")
        if not handle:
            continue
        seed_result = create_seed_account_if_missing(
            platform="instagram",
            handle=handle,
            niche=niche, 
        )
        seed_results.append(seed_result)

        crawl_result = enqueue_account_crawl(
            platform="instagram",
            handle=handle,
            niche=niche,
        )
        crawl_jobs.append(crawl_result)
    
    return {
        "ok": True, 
        "expanded": True,
        "reason": "niche_expansion_triggered",
        "niche_health": decision.get("niche_health", {}),
        "discovered_accounts": discovered_accounts,
        "seed_results": seed_results,
        "crawl_jobs": crawl_jobs,
    }