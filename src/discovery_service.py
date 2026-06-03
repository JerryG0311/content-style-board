import os
import time
import re
from urllib.parse import quote

import requests

from .jobs import get_db

INSTAGRAM_DISCOVERY_COOLDOWN_UNTIL = 0.0
INSTAGRAM_DISCOVERY_COOLDOWN_SECONDS = 60 * 20

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

def normalize_handle_for_similarity(handle: str) -> str:
    return re.sub(r"[^a-z0-9]", "", (handle or "").lower())

def is_near_duplicate_handle(candidate: str, existing_handles: set[str]) -> bool:
    normalized = normalize_handle_for_similarity(candidate)
    if not normalized:
        return True
    
    for existing in existing_handles:
        if not existing:
            continue
        if (
            normalized == existing
            or normalized.startswith(existing)
            or existing.startswith(normalized)
            or normalized in existing 
            or existing in normalized
        ):
            return True
    return False 

# Annotate discovered accounts with whether they already exist in seed_accounts.
# This is for crawl decisions, not for filtering them out.
def annotate_existing_seed_accounts(platform: str, accounts: list[dict]) -> list[dict]:
    """
    Annotate discovered accounts with whether they already exist in seed_accounts.
    This is for crawl decisions, not for filtering them out.
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
    out = []
    for account in deduped:
        enriched = dict(account)
        enriched["already_seeded"] = enriched["handle"] in existing
        out.append(enriched)
    return out

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


# Helper to log non-JSON Instagram responses
def log_instagram_non_json_response(response: requests.Response, query: str, endpoint: str) -> None:
    """
    Log enough context to diagnose when Instagram returns HTML, login pages,
    challenges, rate limits, or other non-JSON responses.
    """
    content_type = response.headers.get("content-type", "")
    location = response.headers.get("location", "")
    preview = (response.text or "")[:500].replace("\n", " ").strip()

    print("[DISCOVERY ERROR] Non-JSON response from Instagram")
    print(f"[DISCOVERY ERROR] Query: {query}")
    print(f"[DISCOVERY ERROR] Status: {response.status_code}")
    print(f"[DISCOVERY ERROR] Content-Type: {content_type}")
    if location:
        print(f"[DISCOVERY ERROR] Redirect Location: {location}")
    print(f"[DISCOVERY ERROR] Endpoint: {endpoint}")
    print(f"[DISCOVERY ERROR] Preview: {preview}")


# Cooldown helpers for Instagram discovery
def instagram_discovery_is_in_cooldown() -> bool:
    return time.time() < INSTAGRAM_DISCOVERY_COOLDOWN_UNTIL


def start_instagram_discovery_cooldown(reason: str = "non_json_response") -> None:
    global INSTAGRAM_DISCOVERY_COOLDOWN_UNTIL
    INSTAGRAM_DISCOVERY_COOLDOWN_UNTIL = time.time() + INSTAGRAM_DISCOVERY_COOLDOWN_SECONDS
    print(
        f"[DISCOVERY COOLDOWN] Pausing Instagram discovery for "
        f"{INSTAGRAM_DISCOVERY_COOLDOWN_SECONDS} seconds. Reason: {reason}"
    )

def build_niche_signal_terms(niche: str) -> list[str]:
    niche = (niche or "").lower().strip()
    if not niche:
        return []
    
    words = [w.strip() for w in re.split(r"\s+", niche) if len(w.strip()) >= 3]
    terms = []

    def add(value: str):
        value = (value or "").lower().strip()
        if value and value not in terms:
            terms.append(value)
    
    add(niche)
    add(niche.replace(" ", ""))
    add(niche.replace(" ", "_"))

    for word in words:
        add(word)
    
    niche_expansions = {
        "amazon": ["fba", "seller", "sellers", "selling", "wholesale", "ecommerce", "ecom", "arbitrage", "reseller", "reselling", "sourcing", "prep", "distribution", "brand direct", "online arbitrage"],
        "fba": ["amazon", "seller", "sellers", "selling", "wholesale", "ecommerce", "ecom", "arbitrage", "reseller", "reselling", "sourcing", "prep", "private label", "brand direct"],
        "wholesale": ["seller", "sellers", "amazon", "fba", "distribution", "distributor", "brand direct", "sourcing", "reseller", "reselling", "ecommerce", "ecom"],
        "ecommerce": ["ecom", "shopify", "amazon", "seller", "selling", "brand", "dtc", "store", "product", "commerce"],
        "ecom": ["ecommerce", "shopify", "amazon", "seller", "selling", "brand", "dtc", "store", "product", "commerce"],
        "fitness": ["training", "trainer", "workout", "strength", "conditioning", "nutrition", "coach", "gym", "bodybuilding", "fat loss"],
        "health": ["wellness", "nutrition", "functional", "gut", "hormones", "holistic", "practitioner", "coach", "healing", "root cause"],
        "functional": ["health", "medicine", "wellness", "gut", "hormones", "holistic", "practitioner", "root cause"],
        "medicine": ["health", "functional", "wellness", "practitioner", "holistic", "root cause"],
        "real estate": ["realtor", "realty", "property", "broker", "agent", "homes", "investor", "mortgage"],
        "music": ["producer", "production", "ableton", "logic", "mixing", "mastering", "dj", "artist", "songwriter", "studio"],
        "production": ["producer", "music", "ableton", "logic", "mixing", "mastering", "studio", "beats"],
        "marketing": ["content", "growth", "brand", "copywriting", "ads", "funnel", "social media", "creator", "lead generation"],
        "content": ["creator", "marketing", "social media", "brand", "growth", "reels", "short form", "copywriting"],
        "personal brand": ["creator", "content", "authority", "audience", "social media", "thought leader", "brand"],
    }

    for key, expansions in niche_expansions.items():
        if key in niche or any(key_part in words for key_part in key.split()):
            for term in expansions:
                add(term)
    
    return terms

def negative_account_quality_terms() -> list[str]:
    return [
        "meme", "memes", "humor", "parody", "fan page", "fanpage",
        "quotes", "daily quotes", "giveaway", "deals", "coupon",
        "freebie", "casino", "betting", "forex", "crypto", "nft",
        "gaming", "onlyfans",
    ]

def score_account_for_niche(niche: str, username: str, full_name: str, category: str = "") -> float:
    """
    Niche relevance score for disovered Instagram accounts.
    """
    niche = (niche or "").lower().strip()
    username = (username or "").lower().strip()
    full_name = (full_name or "").lower().strip()
    category = (category or "").lower().strip()
    hay = " ".join([username, full_name, category]).strip()
    if not niche or not hay:
        return 0.0

    niche_words = [w for w in niche.split() if w.strip()]
    if not niche_words:
        return 0.0
    
    score = 0.0
    if niche in hay:
        score += 0.6

    matched_words = 0    
    for word in niche_words:
        if word in hay:
            matched_words += 1
            score += 0.15
        if hay.count(word) >= 2:
            score += 0.05
    if matched_words >= 2:
        score += 0.15
    if matched_words == len(niche_words) and len(niche_words) >= 2:
        score += 0.15
    if full_name and matched_words > 0:
        score += 0.05
    if category and matched_words > 0:
        score += 0.05
    if matched_words <= 1 and len(niche_words) >= 2:
        score -= 0.15
    if len(username) <= 8 and matched_words <= 1:
        score -= 0.1

    positive_terms = build_niche_signal_terms(niche)
    positive_hits = 0

    for term in positive_terms:
        if term and term in hay:
            positive_hits += 1
    
    if positive_hits >= 1:
        score += 0.10
    if positive_hits >= 2:
        score += 0.15
    if positive_hits >= 3:
        score += 0.20
    
    negative_hits = 0
    for term in negative_account_quality_terms():
        if term and term in hay:
            negative_hits += 1
    
    if negative_hits >= 1:
        score -= 0.25
    if negative_hits >= 2:
        score -= 0.40
    
    username_parts = [p for p in re.split(r"[._-]+", username) if p]
    looks_like_personal_brand = (
        1 <= len(username_parts) <= 3
        and 5 <= len(username) <= 32
        and negative_hits == 0
    )

    if looks_like_personal_brand and matched_words == 0 and positive_hits == 0:
        score += 0.05
    
    if len(niche_words) >= 2 and niche not in hay and matched_words < 2 and positive_hits < 2:
        score = min(score, 0.24)
    
    return float(max(0.0, min(score, 1.0)))



# ---- Discovery V2 helpers ----
def build_niche_discovery_queries(niche: str, limit: int = 30) -> list[str]:
    """
    Build a broader set of Instagram topsearch queries for a niche.
    This prevents discovery from depending on only the exact niche phrase.
    """
    niche = (niche or "").strip().lower()
    limit = max(1, int(limit or 30))
    if not niche:
        return []

    words = [w.strip() for w in re.split(r"\s+", niche) if w.strip()]
    meaningful_words = [w for w in words if len(w) >= 3]

    candidates = []

    def add(value: str):
        value = (value or "").strip().lower()
        if value and value not in candidates:
            candidates.append(value)

    add(niche)
    add(niche.replace(" ", ""))
    add(niche.replace(" ", "_"))

    if len(words) >= 2:
        add(" ".join(words[:2]))
        add("_".join(words[:2]))
        add("".join(words[:2]))
        add(" ".join(words[-2:]))
        add("_".join(words[-2:]))
        add("".join(words[-2:]))

    for word in meaningful_words:
        add(word)

    # Generic modifier expansions. These are broad on purpose, but still built
    # from the user's niche so we are not hardcoding one industry forever.
    modifiers = [
        "coach",
        "creator",
        "expert",
        "mentor",
        "tips",
        "strategy",
        "community",
        "academy",
        "business",
        "consultant",
    ]

    for modifier in modifiers:
        add(f"{niche} {modifier}")
        add(f"{niche.replace(' ', '_')}_{modifier}")
        add(f"{niche.replace(' ', '')}{modifier}")

    if len(meaningful_words) >= 2:
        for i in range(len(meaningful_words) - 1):
            pair = f"{meaningful_words[i]} {meaningful_words[i + 1]}".strip()
            add(pair)
            add(pair.replace(" ", "_"))
            add(pair.replace(" ", ""))

    return candidates[:limit]


def merge_and_score_discovered_accounts(niche: str, accounts: list[dict]) -> list[dict]:
    """
    Dedupe, score, annotate, and sort discovered account candidates.
    """
    deduped = dedupe_discovered_accounts(accounts)
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
            not bool(a.get("already_seeded")),
            not bool(a.get("near_duplicate_handle")),
            len((a.get("full_name") or "").strip()),
            bool(a.get("is_verified")),
            not bool(a.get("is_private")),
            a.get("handle") or "",
        ),
        reverse=True,
    )

    return scored


def fetch_instagram_topsearch_accounts(query: str, limit: int = 10) -> list[dict]:
    """
    Query Instagram's own topsearch endpoint for account candidates.
    Side note for later...it may work better with session cookies.
    """

    query = (query or "").strip()
    limit = int(limit or 10)
    if not query:
        return []
    if instagram_discovery_is_in_cooldown():
        print(f"[DISCOVERY COOLDOWN] Skipping query during cooldown: {query}")
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
        if r.status_code in (401, 403, 429):
            log_instagram_non_json_response(r, query=query, endpoint=endpoint)
            start_instagram_discovery_cooldown(reason=f"status_{r.status_code}")
            return []

        r.raise_for_status()

        content_type = (r.headers.get("content-type") or "").lower()
        if "json" not in content_type:
            log_instagram_non_json_response(r, query=query, endpoint=endpoint)
            start_instagram_discovery_cooldown(reason="non_json_content_type")
            return []

        try:
            data = r.json()
        except Exception:
            log_instagram_non_json_response(r, query=query, endpoint=endpoint)
            start_instagram_discovery_cooldown(reason="json_parse_failed")
            return []

    except Exception as e:
        print("[DISCOVERY ERROR] Request failed:", str(e))
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

def fetch_instagram_accounts_with_fallback(query: str, limit: int = 10) -> list[dict]:
    """
    Try Instagram topsearch first. If it returns nothing because Instagram served
    HTML/non-JSON or discovery is cooling down, fallback to Playwright UI search.
    """
    query = (query or "").strip()
    limit = int(limit or 10)

    if not query:
        return []
    
    accounts = fetch_instagram_topsearch_accounts(
        query=query,
        limit=limit,
    )

    if accounts:
        return accounts
    
    try:
        from .playwright_helper import discover_instagram_accounts_playwright

        print(f"[DISCOVERY FALLBACK] Trying playwright discovery for query: {query}")

        fallback_accounts = discover_instagram_accounts_playwright(
            query=query,
            limit=limit,
        )

        print(
            f"[DISCOVERY FALLBACK] Playwright found "
            f"{len(fallback_accounts)} accounts for query: {query}"
        )

        return dedupe_discovered_accounts(fallback_accounts)[:limit]
    
    except Exception as e:
        print(f"[DISCOVERY FALLBACK ERROR] {query} {e}")
        return []
    

def discover_instagram_accounts_for_niche(niche: str, limit: int = 10) -> list[dict]:
    """
    Return candidate Instagram accounts for a niche using a broader Discovery V2 flow.

    Strategy:
    1. Build many niche query variants, not just the exact phrase.
    2. Query Instagram topsearch for each variant.
    3. If fresh discovery is thin, expand from existing seed-account handles.
    4. Score accounts by niche relevance.
    5. Annotate already-seeded accounts instead of blindly hiding everything.
    6. Return fresh accounts first, with useful fallback candidates after.
    """
    niche = (niche or "").strip()
    limit = max(1, int(limit or 10))

    if not niche:
        return []

    query_limit = max(limit * 2, 30)
    per_query_limit = max(8, min(limit, 20))
    queries = build_niche_discovery_queries(niche=niche, limit=query_limit)

    raw_accounts = []
    for query in queries:
        raw_accounts.extend(fetch_instagram_accounts_with_fallback(query=query, limit=per_query_limit))
        time.sleep(1.5)

    print(f"[DISCOVERY] Queries used: {len(queries)}")
    print(f"[DISCOVERY] Raw accounts count: {len(raw_accounts)}")

    deduped = dedupe_discovered_accounts(raw_accounts)
    print(f"[DISCOVERY] Deduped accounts count: {len(deduped)}")

    scored = merge_and_score_discovered_accounts(niche=niche, accounts=deduped)
    print(f"[DISCOVERY] Scored accounts count: {len(scored)}")

    annotated = annotate_existing_seed_accounts("instagram", scored)
    fresh_count = sum(1 for a in annotated if not bool(a.get("already_seeded")))
    print(f"[DISCOVERY] Fresh (not already seeded) count: {fresh_count}")

    # Prefer meaningful matches, but keep a softer fallback when the niche is hard.
    strong_matches = [a for a in annotated if float(a.get("niche_score") or 0.0) >= 0.35]
    soft_matches = [a for a in annotated if 0.20 <= float(a.get("niche_score") or 0.0) < 0.35]

    fresh_strong = [a for a in strong_matches if not a.get("already_seeded")]
    fresh_soft = [a for a in soft_matches if not a.get("already_seeded")]
    seeded_strong = [a for a in strong_matches if a.get("already_seeded")]

    expanded_from_seeds = []
    if len(fresh_strong) < limit:
        expanded_from_seeds = expand_accounts_from_seed(
            platform="instagram",
            niche=niche,
            limit=max(limit * 2, 20),
        )

    expanded_scored = merge_and_score_discovered_accounts(niche=niche, accounts=expanded_from_seeds)
    expanded_annotated = annotate_existing_seed_accounts("instagram", expanded_scored)
    expanded_fresh = [
        a for a in expanded_annotated
        if not a.get("already_seeded")
        and float(a.get("niche_score") or 0.0) >= 0.20
    ]

    combined = []
    seen = set()

    def add_many(accounts: list[dict]):
        for account in accounts:
            handle = normalize_instagram_handle(account.get("handle") or "")
            if not handle or handle in seen:
                continue
            seen.add(handle)
            combined.append(account)

    add_many(fresh_strong)
    add_many(expanded_fresh)
    add_many(fresh_soft)
    add_many(seeded_strong)

    filtered_count = len(fresh_strong) + len(expanded_fresh) + len(fresh_soft)
    print(f"[DISCOVERY] Fresh strong count: {len(fresh_strong)}")
    print(f"[DISCOVERY] Expanded fresh count: {len(expanded_fresh)}")
    print(f"[DISCOVERY] Fresh soft count: {len(fresh_soft)}")
    print(f"[DISCOVERY] Filtered usable count: {filtered_count}")

    return combined[:limit]

def expand_accounts_from_seed(platform: str, niche: str, limit: int = 20) -> list[dict]:
    """
    Expand discovery using already-seeded accounts.
    Uses their handles as queries to find adjacent creators.
    """

    platform = (platform or "").lower().strip()

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT handle
            FROM seed_accounts
            WHERE platform = ?
            ORDER BY
                CASE WHEN last_crawled_at IS NULL THEN 0 ELSE 1 END ASC,
                last_crawled_at ASC,
                RANDOM()
            LIMIT 25
            """,
            [platform],
        ).fetchall()

    seed_rows = [dict(row) if not isinstance(row, dict) else row for row in rows]
    seed_handles = [row["handle"] for row in seed_rows if row.get("handle")]

    existing_normalized_handles = set()
    for row in seed_rows:
        handle = (row.get("handle") or "").strip().lower()
        if handle:
            existing_normalized_handles.add(normalize_handle_for_similarity(handle))

    print(f"[EXPANSION] Using {len(seed_handles)} seed accounts")

    raw_accounts = []
    for handle in seed_handles:
        raw_accounts.extend(
            fetch_instagram_accounts_with_fallback(query=handle, limit=max(10, min(limit, 25)))
        )
        time.sleep(1.5)
    print(f"[EXPANSION] Raw expanded accounts: {len(raw_accounts)}")

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
    
    scored.sort(key=lambda a: float(a.get("niche_score") or 0.0), reverse=True)
    annotated = annotate_existing_seed_accounts(platform, scored)
    fresh = [a for a in annotated if not a.get("already_seeded")]

    filtered = []
    for account in fresh:
        handle = (account.get("handle") or "").strip().lower()
        if not handle:
            continue

        is_dup = is_near_duplicate_handle(handle, existing_normalized_handles)

        enriched_account = dict(account)
        enriched_account["near_duplicate_handle"] = is_dup

        filtered.append(enriched_account)

    print(f"[EXPANSION] New accounts found: {len(fresh)}")
    
    dup_count = sum(1 for a in filtered if a.get("near_duplicate_handle"))
    print(f"[EXPANSION] Marked {dup_count} near-duplicate accounts (not removed)")

    return filtered[:limit]
