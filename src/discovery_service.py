import os
from urllib.parse import quote

import requests

from .jobs import get_db

def normalize_instagram_handle(raw_handle: str) -> str:
    """
    Normalize INstagram handles into a clean DB-sage form.
    Examples:
    - '@jonahhodges_' -> 'jonahhodges_'
    - ' https://www.instagram.com/jonahhodges_/ ' -> 'jonahhodges_'
    - 'jonahhodges_' -> 'jonahhodges_'
    """
    raw_handle = (raw_handle or "").strip()
    if not raw_handle:
        return ""

    raw_handle = raw_handle.replace("https://www.instagram.com/", "")
    raw_handle = raw_handle.replace("http://www.instagram.com/", "")
    raw_handle = raw_handle.replace("https://instagram.com/", "")
    raw_handle = raw_handle.replace("http://instagram.com/", "")
    raw_handle = raw_handle.strip("/").strip()
    raw_handle = raw_handle.lstrip("@").strip()

    if "/" in raw_handle:
        raw_handle = raw_handle.split("/", 1)[0].strip()
    return raw_handle.lower()


def dedupe_discovered_accounts(accounts: list[dict]) -> list[dict]:
    """
    Dedupe discovered account candidates by normalized handle while preserving order.
    """

    seen = set()
    out = []

    for account in accounts or []:
        handle = normalize_instagram_handle(account.get("handle") or "")
        if not handle:
            continue
        if handle in seen:
            continue
        seen.add(handle)
        normalized = dict(account)
        normalized["handle"] = handle
        out.append(normalized)
    
    return out 

def filter_existing_seed_accounts(platform: str, accounts: list[dict]) -> list[dict]:
    """
    Remove accounts that already exist in seed_accounts.
    """

    platform = (platform or "").lower().strip()
    deduped = dedupe_discovered_accounts(accounts)
    if not deduped:
        return []
    
    handles = [a["handle"] for a in deduped]
    placeholders = ", ".join("?" for _ in handles)

    with get_db() as conn:
        rows = conn.execute(
            f"""
            SELECT LOWER(handle) AS handle
            FROM seed_accounts
            WHERE platform = ?
                AND LOWER(handle) IN ({placeholders})
            """,
            [platform, *handles],
        ).fetchall()
    
    existing = {row["handle"] for row in rows}
    return [a for a in deduped if a["handle"] not in existing]

def build_instagram_discovery_headers(query: str = "") -> tuple[dict, dict]:
    """
    Build headers/cookies for Instagram-native discovery requests.

    Uses the same optional session env vars as the rest of the app, but stays
    self-contained so discovery_service does not import app.py and create
    circular imports.
    """
    referer = "https://www.instagram.com/"
    if query:
        referer = f"https://www.instagram.com/explore/search/keyword/?q={quote(query)}"
    
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Referer": referer,
        "X-IG-App-ID": os.getenv("INSTAGRAM_X_IG_APP_ID", "936619743392459"),
        "X-ASBD-ID": os.getenv("INSTAGRAM_X_ASBD_ID", "129477"),
        "X-Requested-With": "XMLHttpRequest",
    }

    csrftoken = (os.getenv("INSTAGRAM_CSRFTOKEN") or "").strip()
    if csrftoken:
        headers["X-CSRFToken"] = csrftoken
    
    cookies = {}
    cookie_env_map = {
        "sessionid": "INSTAGRAM_SESSIONID",
        "csrftoken": "INSTAGRAM_CSRFTOKEN",
        "ds_user_id": "INSTAGRAM_DS_USER_ID",
        "mid": "INSTAGRAM_MID",
        "ig_did": "INSTAGRAM_IG_DID",
        "rur": "INSTAGRAM_RUR",
    }
    for cookie_name, env_name in cookie_env_map.items():
        value = (os.getenv(env_name) or "").strip()
        if value:
            cookies[cookie_name] = value
    return headers, cookies

def score_account_for_niche(niche: str, username: str, full_name: str, category: str = "") -> float:
    """
    Simple production-safe relevance score for discovered Instagram accounts.
    Higher when niche words appear in username/full name/category.
    """
    niche = (niche or "").lower().strip()
    hay = " ".join([
        (username or "").lower(),
        (full_name or "").lower(),
        (category or "").lower(),
    ]).strip()
    if not niche or not hay:
        return 0.0
    
    niche_words = [w for w in niche.split() if w.strip()]
    if not niche_words:
        return 0.0
    
    score = 0.0
    if niche in hay:
        score += 0.6
    for word in niche_words:
        if word in hay:
            score += 0.2
    for word in niche_words:
        if hay.count(word) >= 2:
            score += 0.1
    
    return float(min(score, 1.0))

def fetch_instagram_topsearch_accounts(query: str, limit: int = 10) -> list[dict]:
    """
    Query Instagram's own topsearch endpoint for account candidates.
    Side note for later...it may work better with session cookies.
    """

    query = (query or "").strip()
    limit = int(limit or 10)
    if not query:
        return []
    
    headers, cookies = build_instagram_discovery_headers(query=query)
    endpoint =  f"https://www.instagram.com/web/search/topsearch/?context=blended&query={quote(query)}&count={max(limit, 10)}"
    try:
        r = requests.get(
            endpoint,
            headers=headers,
            cookies=cookies,
            timeout=20,
            allow_redirects=True,
        )
        r.raise_for_status()
        data = r.json()
    except Exception:
        return []
    
    users = data.get("users") or []
    out = []
    for item in users:
        user = item.get("user") if isinstance(item, dict) else None
        if not isinstance(user, dict):
            continue
        handle = normalize_instagram_handle(user.get("username") or "")
        if not handle:
            continue
        full_name = (user.get("full_name") or "").strip()
        category = (user.get("category") or user.get("category_name") or "").strip()
        profile_pic_url = (user.get("profile_pic_url") or "").strip()
        is_verified = bool(user.get("is_verified"))
        is_private = bool(user.get("is_private"))

        out.append(
            {
                "handle": handle,
                "full_name": full_name,
                "category": category,
                "profile_pic_url": profile_pic_url,
                "is_verified": is_verified,
                "is_private": is_private,
                "source": "instagram_topsearch",
            }
        )

    return dedupe_discovered_accounts(out)[:limit]
    

def discover_instagram_accounts_for_niche(niche: str, limit: int = 10) -> list[dict]:
    """
    Return candidate Instagram accounts for a niche using Instagram-native
    discovery, not Brave.

    Strategy:
    1. Query Instagram topsearch with the full niche phrase.
    2. Query again with smaller niche fragments derived from the niche itself.
    3. Score each candidate for niche relevance.
    4. Remove already-seeded accounts.
    5. Return the best candidates.
    """
    niche = (niche or "").strip()
    limit = int(limit or 10)

    if not niche:
        return []
    
    queries = [niche]
    words = [w for w in niche.split() if w.strip()]
    if len(words) >= 2:
        queries.append(" ".join(words[:2]))
    if len(words) >= 3:
        queries.append(" ".join(words[-2:]))

    # Add generic niche-driven fragments without hardcoding any one industry.
    niche_tokens = [w for w in words if len(w) >= 4]

    for token in niche_tokens:
        if token not in queries:
            queries.append(token)

    if len(niche_tokens) >= 2:
        for i in range(len(niche_tokens) - 1):
            pair = f"{niche_tokens[i]} {niche_tokens[i + 1]}".strip()
            if pair and pair not in queries:
                queries.append(pair)

    raw_accounts = []
    for query in queries:
        raw_accounts.extend(fetch_instagram_topsearch_accounts(query=query, limit=limit))

    deduped = dedupe_discovered_accounts(raw_accounts)

    scored = []
    for account in deduped:
        score = score_account_for_niche(
            niche=niche,
            username=account.get("handle") or "",
            full_name=account.get("full_name") or "",
            category=account.get("category") or "",
        )
        enriched = dict(account)
        enriched["niche_score"] = score
        scored.append(enriched)

    scored.sort(
        key=lambda a: (
            float(a.get("niche_score") or 0.0),
            bool(a.get("is_verified")),
            not bool(a.get("is_private")),
            a.get("handle") or "",
        ),
        reverse=True,
    )

    fresh = filter_existing_seed_accounts("instagram", scored)
    return fresh[:limit]