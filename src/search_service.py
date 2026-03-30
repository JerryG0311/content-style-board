from .jobs import get_db


def get_niche_health(platform: str, style: str, niche: str = "") -> dict:
    """
    Measure how healthy our local content DB is for a niche/style.
    Used to decide whether to rely on local results or expand discovery.
    """
    platform = (platform or "").lower().strip()
    style = (style or "").lower().strip()
    niche = (niche or "").strip().lower()

    where = ["platform = ?"]
    params = [platform]

    if niche:
        where.append("LOWER(niche) LIKE ?")
        params.append(f"%{niche}%")

    sql_total = f"""
        SELECT COUNT(*) AS total_posts,
               COUNT(DISTINCT account_handle) AS distinct_accounts,
               MAX(collected_at) AS newest_collected_at
        FROM posts
        WHERE {' AND '.join(where)}
    """

    style_where = list(where)
    style_params = list(params)

    if style == "carousel":
        style_where.append("post_type = ?")
        style_params.append("carousel")
    elif style in ("single-clip", "multi-clip", "talking-head"):
        style_where.append("classified_post_type = ?")
        style_params.append(style)
    elif style == "reel":
        style_where.append("post_type = ?")
        style_params.append("reel")
    elif style:
        style_where.append("post_type = ?")
        style_params.append(style)

    sql_style = f"""
        SELECT COUNT(*) AS style_posts
        FROM posts
        WHERE {' AND '.join(style_where)}
    """

    with get_db() as conn:
        total_row = conn.execute(sql_total, params).fetchone()
        style_row = conn.execute(sql_style, style_params).fetchone()

    total_posts = int((total_row["total_posts"] if total_row else 0) or 0)
    distinct_accounts = int((total_row["distinct_accounts"] if total_row else 0) or 0)
    newest_collected_at = (total_row["newest_collected_at"] if total_row else "") or ""
    style_posts = int((style_row["style_posts"] if style_row else 0) or 0)

    healthy = (
        total_posts >= 30
        and distinct_accounts >= 8
        and style_posts >= 5
    )

    return {
        "platform": platform,
        "style": style,
        "niche": niche,
        "total_posts": total_posts,
        "distinct_accounts": distinct_accounts,
        "style_posts": style_posts,
        "newest_collected_at": newest_collected_at,
        "healthy": healthy,
    }