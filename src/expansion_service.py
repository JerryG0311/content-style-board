from datetime import datetime, timezone, timedelta
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


def get_existing_seed_accounts_for_niche(
    platform: str,
    niche: str = "",
    limit: int = 10,
    min_hours_since_crawl: int = 24,
) -> list[dict]:
    """
    Fetch active seed accounts for a niche, but avoid immediately re-crawling
    accounts that were recently crawled and mostly return already-saved posts.
    """

    platform = (platform or "instagram").lower().strip()
    niche = (niche or "").strip().lower()
    limit = max(1, int(limit or 10))
    min_hours_since_crawl = max(1, int(min_hours_since_crawl or 24))

    cutoff = (datetime.now(timezone.utc) - timedelta(hours=min_hours_since_crawl)).isoformat()

    where = ["platform = ?", "is_active = 1"]
    params = [platform]

    if niche:
        where.append("LOWER(niche) LIKE ?")
        params.append(f"%{niche}%")

    where.append("(last_crawled_at IS NULL OR last_crawled_at < ?)")
    params.append(cutoff)

    sql = f"""
        SELECT *
        FROM seed_accounts
        WHERE {' AND '.join(where)}
        ORDER BY
            CASE WHEN last_crawled_at IS NULL THEN 0 ELSE 1 END ASC,
            last_crawled_at ASC,
            id DESC
        LIMIT ?
    """
    params.append(limit)

    with get_db() as conn:
        rows = conn.execute(sql, params).fetchall()

    return [dict(row) for row in rows]


def expand_niche_if_needed(platform: str, style: str, niche: str, limit: int = 10) -> dict:
    """
    Main niche-expansion entry point.

    This is now an orchestration/dispatcher function only.
    It does NOT scrape Instagram directly.

    Flow:
    1. Check whether the DB is already healthy for this niche/style.
    2. Discover candidate accounts for the niche.
    3. Mix discovered accounts with existing active seed accounts.
    4. Insert missing seed accounts.
    5. Enqueue smaller crawl_instagram_account jobs for workers.
    """

    platform = (platform or "instagram").lower().strip()
    style = (style or "carousel").lower().strip()
    niche = (niche or "").strip()
    limit = max(1, int(limit or 10))

    decision = needs_niche_expansion(platform=platform, style=style, niche=niche)
    if not decision.get("should_expand"):
        return {
            "ok": True,
            "expanded": False,
            "reason": "niche_already_healthy",
            "niche_health": decision.get("niche_health", {}),
            "discovered_accounts": [],
            "existing_seed_accounts": [],
            "selected_accounts": [],
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
            "existing_seed_accounts": [],
            "selected_accounts": [],
            "seed_results": [],
            "crawl_jobs": [],
        }

    discovery_limit = max(limit * 3, 30)

    try:
        discovered_accounts = discover_instagram_accounts_for_niche(niche=niche, limit=discovery_limit)
    except Exception as exc:
        print(f"[EXPANSION] Discovery failed for niche={niche}: {exc}")
        discovered_accounts = []

    existing_seed_accounts = get_existing_seed_accounts_for_niche(
        platform=platform,
        niche=niche,
        limit=limit,
        min_hours_since_crawl=24,
    )

    merged_accounts = []
    seen_handles = set()

    for account in list(discovered_accounts or []) + list(existing_seed_accounts or []):
        if not isinstance(account, dict):
            try:
                account = dict(account)
            except Exception:
                account = {"handle": str(account or "")}

        handle = (account.get("handle") or account.get("username") or "").strip().lstrip("@")
        if not handle:
            continue

        handle_key = handle.lower()
        if handle_key in seen_handles:
            continue

        seen_handles.add(handle_key)
        merged_accounts.append({
            **account,
            "handle": handle,
        })

    selected_accounts = merged_accounts[:limit]

    seed_results = []
    crawl_jobs = []

    for account in selected_accounts:
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

    queued_count = len([job for job in crawl_jobs if job.get("queued")])
    skipped_count = len([job for job in crawl_jobs if not job.get("queued")])

    if queued_count == 0:
        return {
            "ok": True,
            "expanded": False,
            "reason": "no_eligible_accounts_available",
            "niche_health": decision.get("niche_health", {}),
            "discovered_accounts": discovered_accounts,
            "existing_seed_accounts": existing_seed_accounts,
            "selected_accounts": selected_accounts,
            "seed_results": seed_results,
            "crawl_jobs": crawl_jobs,
            "queued_count": queued_count,
            "skipped_count": skipped_count,
            "seed_recrawl_cooldown_hours": 24,
            "discovery_limit": discovery_limit,
        }

    return {
        "ok": True,
        "expanded": True,
        "reason": "niche_expansion_queued_crawl_jobs",
        "niche_health": decision.get("niche_health", {}),
        "discovered_accounts": discovered_accounts,
        "existing_seed_accounts": existing_seed_accounts,
        "selected_accounts": selected_accounts,
        "seed_results": seed_results,
        "crawl_jobs": crawl_jobs,
        "queued_count": queued_count,
        "skipped_count": skipped_count,
        "seed_recrawl_cooldown_hours": 24,
        "discovery_limit": discovery_limit,
    }