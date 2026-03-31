from dotenv import load_dotenv
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from datetime import datetime, timezone, timedelta
from pydantic import BaseModel
from json import JSONDecodeError

import json
import base64
import requests
import hashlib
import time
import re
import sqlite3
import subprocess


from fastapi import Response
from fastapi.responses import StreamingResponse
from urllib.parse import unquote, quote
from .jobs import (
    JOB_CLASSIFY_REEL_VIDEO,
    create_crawl_job,
    get_crawl_job,
    publish_rabbitmq_job,
    update_crawl_job_status,
    get_db,
    utc_now_iso,
)
from .search_service import get_niche_health
from .expansion_service import expand_niche_if_needed

load_dotenv()


app = FastAPI(title="Content Style Board")


from urllib.parse import urlparse

BRAVE_ENDPOINT = "https://api.search.brave.com/res/v1/web/search"

def brave_web_search(query: str, count: int = 8, safesearch: str = "moderate"):
    """ 
    Safe Brave call — never prints the API key.
    """
    api_key = os.getenv("BRAVE_API_KEY")
    if not api_key:
        raise RuntimeError("BRAVE_API_KEY is not set")

    headers = {
        "Accept": "application/json",
        "X-Subscription-Token": api_key,
    }

    params = {
        "q": query,
        "count": count,
        "search_lang": "en",
        "country": "us",
        # Brave can be overly aggressive with IG links under safesearch.
        # We'll override this per-platform in /api/search when needed.
        "safesearch": safesearch,
    }

    r = requests.get(BRAVE_ENDPOINT, headers=headers, params=params, timeout=20)
    r.raise_for_status()
    return r.json()


def build_query(platform: str, style: str, niche: str = "", seed: str = "") -> str:
    platform = (platform or "").lower().strip()
    style = (style or "").lower().strip()

    niche = (niche or "").strip()
    seed = (seed or "").strip()

    # IMPORTANT:
    # We want content IN the niche.
    # We do NOT want meta-content about making carousels/reels.

    # Keep negatives light. Over-filtering often yields 0 results.
    negative = "-template -templates -canva -capcut -ugc -ads -advertising -marketing"

    seed_hint = seed

    def join_parts(*parts: str) -> str:
        return " ".join([p for p in parts if (p or "").strip()]).strip()

    if platform == "instagram":
        if style == "carousel":
            # Niche-first. No "swipe/carousel/slide 1" hints here
            # because those tend to pull meta “carousel ideas” content.
            return join_parts(
                "site:instagram.com/p/",
                niche,
                seed_hint,
                negative,
            )

        if style in ("multi-clip", "single-clip"):
            extra = "part 1" if style == "multi-clip" else ""
            return join_parts(
                "site:instagram.com/reel/",
                niche,
                extra,
                seed_hint,
                negative,
            )

    if platform == "tiktok":
        return join_parts(
            "site:tiktok.com",
            niche,
            seed_hint,
            negative,
        )

    if platform in ("x", "twitter"):
        return join_parts(
            "site:x.com/status",
            niche,
            seed_hint,
            negative,
        )

    return join_parts(platform, style, niche, seed_hint)



def normalize_url(u: str) -> str:
    return (u or "").strip().lower()


# Helper for Brave thumbnail extraction
def extract_thumbnail(item: dict):
    """Best-effort thumbnail extraction from Brave result items."""
    if not isinstance(item, dict):
        return None

    t = item.get("thumbnail")
    if isinstance(t, dict):
        return t.get("src") or t.get("url")
    if isinstance(t, str):
        return t

    # Some Brave payloads may use different keys
    for k in ("image", "thumbnail_url", "image_url"):
        v = item.get(k)
        if isinstance(v, str) and v:
            return v
        if isinstance(v, dict):
            return v.get("src") or v.get("url")

    return None

UNFURL_CACHE_FILE = Path("data") / "unfurl_cache.json"
UNFURL_TTL_SECONDS = 60 * 60 * 24 * 7 # 7 days
unfurl_cache = None

def load_unfurl_cache() -> dict:
    global unfurl_cache
    if unfurl_cache is not None:
        return unfurl_cache
    
    try:
        if UNFURL_CACHE_FILE.exists():
            raw = UNFURL_CACHE_FILE.read_text(encoding="utf-8").strip()
            unfurl_cache = json.loads(raw) if raw else {}
        else:
            unfurl_cache = {}
    except Exception:
        unfurl_cache = {}
    
    return unfurl_cache

def save_unfurl_cache() -> None:
    try:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        UNFURL_CACHE_FILE.write_text(json.dumps(unfurl_cache or {}, indent=2), encoding="utf-8")
    except Exception:
        pass

def cache_key(url: str) -> str:
    u = (url or "").strip()
    return hashlib.sha256(u.encode("utf-8")).hexdigest()

def extract_meta(html: str, key: str) -> str:
    """Best-effort extraction for og:/twitter: meta tags.

    Handles cases where attributes are single-quoted and/or appear in different orders.
    """
    if not html:
        return ""

    k = re.escape(key)

    # Match <meta ... (property|name)="key" ... content="..."> with any attribute order
    meta_tag_pattern = re.compile(
        rf"<meta\b[^>]*?(?:property|name)\s*=\s*['\"]{k}['\"][^>]*?>",
        re.IGNORECASE,
    )
    content_pattern = re.compile(r"content\s*=\s*['\"]([^'\"]+)['\"]", re.IGNORECASE)

    for m in meta_tag_pattern.finditer(html):
        tag = m.group(0)
        cm = content_pattern.search(tag)
        if cm:
            return (cm.group(1) or "").strip()

    # Fallback: some pages use itemprop
    meta_tag_pattern2 = re.compile(rf"<meta\b[^>]*?itemprop\s*=\s*['\"]{k}['\"][^>]*?>", re.IGNORECASE)
    for m in meta_tag_pattern2.finditer(html):
        tag = m.group(0)
        cm = content_pattern.search(tag)
        if cm:
            return (cm.group(1) or "").strip()

    return ""


# --- Begin new helpers for unfurl_url fallbacks ---
def extract_jsonld_image(html: str) -> str:
    """Try to extract an image/thumbnail from JSON-LD blocks."""
    if not html:
        return ""

    # Find all <script type="application/ld+json"> blocks
    scripts = re.findall(
        r"<script[^>]+type=([\"']?)application/ld\+json\1[^>]*>(.*?)</script>",
        html,
        flags=re.IGNORECASE | re.DOTALL,
    )

    def _walk(obj):
        if obj is None:
            return ""
        if isinstance(obj, str):
            return ""
        if isinstance(obj, list):
            for it in obj:
                got = _walk(it)
                if got:
                    return got
            return ""
        if isinstance(obj, dict):
            # common keys
            for k in ("thumbnailUrl", "thumbnailURL", "thumbnail", "image", "contentUrl", "url"):
                v = obj.get(k)
                if isinstance(v, str) and v.startswith("http"):
                    return v
                if isinstance(v, list):
                    for vv in v:
                        if isinstance(vv, str) and vv.startswith("http"):
                            return vv
                        if isinstance(vv, dict):
                            u = vv.get("url") or vv.get("contentUrl")
                            if isinstance(u, str) and u.startswith("http"):
                                return u
                if isinstance(v, dict):
                    u = v.get("url") or v.get("contentUrl")
                    if isinstance(u, str) and u.startswith("http"):
                        return u

            # recurse
            for vv in obj.values():
                got = _walk(vv)
                if got:
                    return got
        return ""

    for _, raw in scripts:
        raw = (raw or "").strip()
        if not raw:
            continue
        try:
            data = json.loads(raw)
        except Exception:
            continue
        got = _walk(data)
        if got:
            return got

    return ""


def extract_instagram_display_url(html: str) -> str:
    """Instagram often embeds a display_url in inline JSON; try to pull it out."""
    if not html:
        return ""

    # Common patterns: "display_url":"https:\/\/..."
    m = re.search(r'"display_url"\s*:\s*"(https?:\\/\\/[^\"]+)"', html)
    if m:
        try:
            return m.group(1).replace('\\/', '/')
        except Exception:
            return ""

    # Sometimes: "thumbnail_url":"https:\/\/..."
    m2 = re.search(r'"thumbnail_url"\s*:\s*"(https?:\\/\\/[^\"]+)"', html)
    if m2:
        try:
            return m2.group(1).replace('\\/', '/')
        except Exception:
            return ""

    return ""


# --- New helper: extract_instagram_json_image ---
def extract_instagram_json_image(url: str, headers: dict) -> str:
    """Best-effort: hit IG's JSON-ish endpoint to remember the image URL.

    This sometimes works for public posts and can return a stable display_url.
    If it fails (403/429/HTML), we just return "".
    """
    try:
        u = (url or "").strip()
        if not u.startswith("http"):
            return ""

        p = urlparse(u)
        host = (p.netloc or "").lower()
        path = p.path or ""
        if "instagram.com" not in host:
            return ""

        # Pull shortcode from /p/<code>/ or /reel/<code>/
        m = re.search(r"/(p|reel|reels)/([A-Za-z0-9_-]+)/?", path)
        if not m:
            return ""
        kind = m.group(1)
        shortcode = m.group(2)

        # __a=1 endpoint (may or may not be available depending on IG changes)
        json_url = f"https://www.instagram.com/{kind}/{shortcode}/?__a=1&__d=dis"

        h = dict(headers or {})
        h.update({
            "Accept": "application/json,text/plain,*/*",
            # This app id is widely used by IG web clients; harmless if ignored.
            "X-IG-App-ID": "936619743392459",
        })

        rr = requests.get(json_url, headers=h, timeout=20, allow_redirects=True)
        rr.raise_for_status()

        # IG sometimes returns HTML even with 200. Bail if it isn't JSON-like.
        ct = (rr.headers.get("Content-Type") or "").lower()
        txt = rr.text or ""
        if ("json" not in ct) and (not txt.strip().startswith("{") and not txt.strip().startswith("[")):
            return ""

        try:
            data = rr.json()
        except Exception:
            return ""

        # Common structures (best-effort)
        media = None
        if isinstance(data, dict):
            # newer-ish
            media = (((data.get("graphql") or {}).get("shortcode_media")) if isinstance(data.get("graphql"), dict) else None)
            # older-ish
            if media is None and "items" in data and isinstance(data.get("items"), list) and data["items"]:
                media = data["items"][0]

        if not isinstance(media, dict):
            return ""

        # 1) Single image/video thumb
        img = media.get("display_url") or media.get("thumbnail_src")
        if isinstance(img, str) and img.startswith("http"):
            return img

        # 2) Carousel
        sidecar = media.get("edge_sidecar_to_children")
        if isinstance(sidecar, dict):
            edges = sidecar.get("edges") or []
            if isinstance(edges, list) and edges:
                node = edges[0].get("node") if isinstance(edges[0], dict) else None
                if isinstance(node, dict):
                    img2 = node.get("display_url") or node.get("thumbnail_src")
                    if isinstance(img2, str) and img2.startswith("http"):
                        return img2

        return ""
    except Exception:
        return ""
# --- End new helpers for unfurl_url fallbacks ---

# Helper to unfurl og/twitter image for a URL
def best_effort_unfurl_image(url: str) -> str:
    """Try to fetch an og/twitter image for a URL using the unfurl cache."""
    try:
        u = unfurl_url(url)
        if u.get("ok"):
            return (u.get("image") or "").strip()
    except Exception:
        pass
    return ""

def unfurl_url(url: str) -> dict:
    url = (url or "").strip()
    if not url.startswith("http"):
        return {"ok": False, "error": "Invalid URL"}

    # Some platforms (especially Instagram) often hide og:image on the normal URL
    # but expose it on the /embed/ version. We'll try a small set of candidate URLs.
    def _candidate_unfurl_urls(u: str) -> list:
        try:
            p = urlparse(u)
            host = (p.netloc or "").lower()
            path = p.path or ""
        except Exception:
            host, path = "", ""

        # Prefer Instagram embed pages when possible.
        if "instagram.com" in host and ("/p/" in path or "/reel/" in path or "/reels/" in path):
            # Ensure trailing slash before adding embed/
            base_path = path
            if not base_path.endswith("/"):
                base_path += "/"
            embed_url = f"https://www.instagram.com{base_path}embed/"
            # Try embed first, then the original.
            return [embed_url, u]

        return [u]

    candidate_urls = _candidate_unfurl_urls(url)
    
    cache = load_unfurl_cache()
    key = cache_key(url)
    now = int(time.time())

    cached = cache.get(key)
    if isinstance(cached, dict):
        ts = int(cached.get("ts", 0) or 0)
        if ts and (now - ts) < UNFURL_TTL_SECONDS:
            return {"ok": True, "url": url, **cached.get("data", {})}
        
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
    }

    r = None
    html = ""
    last_err = None

    for u in candidate_urls:
        try:
            r = requests.get(u, headers=headers, timeout=20, allow_redirects=True)
            html = r.text or ""
            # If we got any HTML at all, proceed to parsing.
            if html.strip():
                break
        except Exception as e:
            last_err = e
            r = None
            html = ""

    if not html.strip():
        return {"ok": False, "error": str(last_err) if last_err else "Failed to fetch URL"}
    
    image = (
        extract_meta(html, "og:image")
        or extract_meta(html, "og:image:url")
        or extract_meta(html, "og:image:secure_url")
        or extract_meta(html, "twitter:image")
        or extract_meta(html, "twitter:image:src")
    )

    # Fallbacks when og/twitter tags are missing (common on Instagram)
    if not (image or "").strip():
        image = extract_jsonld_image(html) or ""

    if not (image or "").strip() and "instagram.com" in (str(getattr(r, 'url', url)) or url):
        # 1) Try inline JSON patterns (fast)
        image = extract_instagram_display_url(html) or ""

    if not (image or "").strip() and "instagram.com" in (str(getattr(r, 'url', url)) or url):
        # 2) Try IG's JSON-ish endpoint (best-effort; can fail depending on IG)
        image = extract_instagram_json_image(url, headers) or ""

    title = extract_meta(html, "og:title") or extract_meta(html, "twitter:title")
    if not title:
        m = re.search(r"<title[^>]*>(.*?)</title>", html, re.IGNORECASE | re.DOTALL)
        if m:
            title = re.sub(r"\s+", " ", (m.group(1) or "").strip())

    site = extract_meta(html, "og:site_name") or extract_meta(html, "twitter:site")

    data = {
        "image": image or "",
        "title": title or "",
        "site": site or "",
        "final_url": str(getattr(r, "url", url) or url),
    }

    cache[key] = {"ts": now, "data": data}
    save_unfurl_cache()

    return {"ok": True, "url": url, **data}


def url_matches_platform(url: str, platform: str, style: str = None) -> bool:
    """
    Platform + style URL filtering.
    """
    u = normalize_url(url)
    if not u:
        return False

    try:
        parsed = urlparse(u)
        host = parsed.netloc or ""
        path = parsed.path or ""
    except Exception:
        host, path = "", ""

    platform = (platform or "").lower().strip()
    style = (style or "").lower().strip()

    if platform == "instagram":
        if "instagram.com" not in host:
            return False
        # carousels/posts must be /p/
        if style in ("carousel", "post"):
            return "/p/" in path
        # reels must be /reel/ or /reels/
        if style in ("reel", "reels", "multi-clip", "single-clip"):
            return ("/reel/" in path) or ("/reels/" in path)
        return True

    if platform == "tiktok":
        if "tiktok.com" not in host:
            return False
        return ("/video/" in path) or ("tiktok.com/@" in u)

    if platform in ("x", "twitter"):
        if not ("x.com" in host or "twitter.com" in host):
            return False
        # posts/threads usually have /status/<id>
        if style in ("post", "short", "thread"):
            return ("/status/" in path) or ("/statuses/" in path)
        return True

    return True


def dedupe_and_truncate(results: list, limit: int = 8) -> list:
    seen = set()
    out = []
    for r in results:
        url = normalize_url(r.get("url", ""))
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(r)
        if len(out) >= limit:
            break
    return out

def looks_like_tutorial_or_ads(title: str) -> bool:
    t = (title or "").lower()

    # Always block obvious ads/marketing/meta-content
    ad_keywords = [
        "ads", "ad ", "advertising", "marketing", "ugc",
        "for your ads", "pin text", "variations"
    ]

    # Block META content about making content (not niche education)
    meta_keywords = [
        "make a carousel", "create a carousel",
        "carousel template", "carousel templates",
        "viral carousel", "carousel ideas", "carousel hooks", "hook ideas",
        "reel ideas", "reels ideas",
        "content strategy", "social media strategy", "instagram strategy", "tiktok strategy",
        "canva", "capcut",
        "video editing", "editing tutorial",
        "content ideas", "content calendar",
        "carousel trends",
    ]

    if any(k in t for k in ad_keywords):
        return True

    if any(k in t for k in meta_keywords):
        return True

    return False

@app.get("/api/brave_test")
def brave_test(q: str = "instagram carousel example"):
    data = brave_web_search(q, count=5)


    # Brave response structure: data["web"]["results"] is the useful list
    results = []
    for item in (data.get("web", {}).get("results", []) or []):
        results.append(
            {
                "title": item.get("title"),
                "url": item.get("url"),
                "description": item.get("description"),
            }
        )
    return {"ok": True, "q": q, "results": results}

# Project paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
FRONTEND_DIR = PROJECT_ROOT / "frontend"
INDEX_HTML = FRONTEND_DIR / "index.html"


DATA_DIR = Path("data")
BOARD_FILE = DATA_DIR / "board.json"

DB_FILE = DATA_DIR / "app.db"



def init_db() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    with get_db() as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS seed_accounts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT NOT NULL,
                handle TEXT NOT NULL,
                niche TEXT NOT NULL DEFAULT '',
                is_active INTEGER NOT NULL DEFAULT 1,
                created_at TEXT NOT NULL,
                last_crawled_at TEXT
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_seed_accounts_platform_handle
            ON seed_accounts(platform, handle);

            CREATE TABLE IF NOT EXISTS posts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                platform TEXT NOT NULL,
                account_handle TEXT NOT NULL DEFAULT '',
                post_url TEXT NOT NULL,
                shortcode TEXT NOT NULL DEFAULT '',
                post_type TEXT NOT NULL DEFAULT '',
                classified_post_type TEXT NOT NULL DEFAULT '',
                classifier_confidence REAL NOT NULL DEFAULT 0,
                classifier_version TEXT NOT NULL DEFAULT '',
                classified_at TEXT NOT NULL DEFAULT '',
                caption TEXT NOT NULL DEFAULT '',
                preview_mode TEXT NOT NULL DEFAULT '',
                preview_url TEXT NOT NULL DEFAULT '',
                embed_url TEXT NOT NULL DEFAULT '',
                niche TEXT NOT NULL DEFAULT '',
                collected_at TEXT NOT NULL,
                created_at TEXT NOT NULL
            );

            CREATE UNIQUE INDEX IF NOT EXISTS idx_posts_post_url
            ON posts(post_url);

            CREATE INDEX IF NOT EXISTS idx_posts_platform_post_type
            ON posts(platform, post_type);

            CREATE INDEX IF NOT EXISTS idx_posts_niche
            ON posts(niche);

            CREATE TABLE IF NOT EXISTS crawl_jobs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                job_type TEXT NOT NULL,
                target TEXT NOT NULL,
                status TEXT NOT NULL,
                error_message TEXT NOT NULL DEFAULT '',
                created_at TEXT NOT NULL,
                started_at TEXT,
                finished_at TEXT,
                retry_count INTEGER NOT NULL DEFAULT 0
            );

            CREATE TABLE IF NOT EXISTS post_tags (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                post_id INTEGER NOT NULL,
                tag_type TEXT NOT NULL,
                tag_value TEXT NOT NULL,
                score REAL NOT NULL DEFAULT 0,
                FOREIGN KEY(post_id) REFERENCES posts(id) ON DELETE CASCADE
            );
            """
        )

        existing_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(posts)").fetchall()
        }

        if "classified_post_type" not in existing_cols:
            conn.execute(
                "ALTER TABLE posts ADD COLUMN classified_post_type TEXT NOT NULL DEFAULT ''"
            )
        if "classifier_confidence" not in existing_cols:
            conn.execute(
                "ALTER TABLE posts ADD COLUMN classifier_confidence REAL NOT NULL DEFAULT 0"
            )
        if "classifier_version" not in existing_cols:
            conn.execute(
                "ALTER TABLE posts ADD COLUMN classifier_version TEXT NOT NULL DEFAULT ''"
            )
        if "classified_at" not in existing_cols:
            conn.execute(
                "ALTER TABLE posts ADD COLUMN classified_at TEXT NOT NULL DEFAULT ''"
            )
        
        existing_job_cols = {
            row[1]
            for row in conn.execute("PRAGMA table_info(crawl_jobs)").fetchall()
        }

        if "retry_count" not in existing_job_cols:
            conn.execute(
                "ALTER TABLE crawl_jobs ADD COLUMN retry_count INTEGER NOT NULL DEFAULT 0"
            )


def shortcode_from_url(url: str) -> str:
    u = (url or "").strip()
    try:
        p = urlparse(u)
        path = p.path or ""
    except Exception:
        path = ""

    m = re.search(r"/(p|reel|reels|status)/([A-Za-z0-9_-]+)/?", path)
    if m:
        return m.group(2)
    return ""


def build_embed_url(platform: str, post_url: str) -> str:
    platform = (platform or "").lower().strip()
    post_url = (post_url or "").strip()
    if not post_url:
        return ""

    if platform == "instagram":
        try:
            u = urlparse(post_url)
            path = u.path or ""
            if not path.endswith("/"):
                path += "/"
            if path.endswith("/embed/"):
                return post_url
            return f"https://www.instagram.com{path}embed/"
        except Exception:
            return ""

    return ""


def create_seed_account(platform: str, handle: str, niche: str = "") -> dict:
    platform = (platform or "").lower().strip()
    handle = (handle or "").strip().lstrip("@")
    niche = (niche or "").strip()
    now = utc_now_iso()

    if not platform:
        raise ValueError("platform is required")
    if not handle:
        raise ValueError("handle is required")

    with get_db() as conn:
        conn.execute(
            """
            INSERT OR IGNORE INTO seed_accounts (platform, handle, niche, is_active, created_at)
            VALUES (?, ?, ?, 1, ?)
            """,
            (platform, handle, niche, now),
        )
        row = conn.execute(
            "SELECT * FROM seed_accounts WHERE platform = ? AND handle = ?",
            (platform, handle),
        ).fetchone()

    return dict(row) if row else {}


def list_seed_accounts() -> list:
    with get_db() as conn:
        rows = conn.execute(
            "SELECT * FROM seed_accounts ORDER BY created_at DESC, id DESC"
        ).fetchall()
    return [dict(r) for r in rows]


def upsert_post(
    platform: str,
    post_url: str,
    title: str = "",
    description: str = "",
    tag: str = "",
    preview_url: str = "",
    account_handle: str = "",
    niche: str = "",
    classified_post_type: str = "",
    classifier_confidence: float = 0.0,
    classifier_version: str = "",
) -> dict:
    platform = (platform or "").lower().strip()
    post_url = (post_url or "").strip()
    title = (title or "").strip()
    description = (description or "").strip()
    tag = (tag or "").strip()
    preview_url = (preview_url or "").strip()
    account_handle = (account_handle or "").strip().lstrip("@")
    niche = (niche or "").strip()
    classified_post_type = (classified_post_type or "").strip()
    classifier_version = (classifier_version or "").strip()

    if not platform:
        raise ValueError("platform is required")
    if not post_url:
        raise ValueError("post_url is required")

    caption = title
    if description:
        caption = f"{title}\n{description}".strip()

    shortcode = shortcode_from_url(post_url)
    preview_mode = "embed" if platform == "instagram" else ("image" if preview_url else "")
    embed_url = build_embed_url(platform, post_url)
    now = utc_now_iso()
    classified_at = now if classified_post_type else ""

    with get_db() as conn:
        conn.execute(
            """
            INSERT INTO posts (
                platform, account_handle, post_url, shortcode, post_type,
                classified_post_type, classifier_confidence, classifier_version, classified_at,
                caption, preview_mode, preview_url, embed_url, niche,
                collected_at, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(post_url) DO UPDATE SET
                platform = excluded.platform,
                account_handle = excluded.account_handle,
                shortcode = excluded.shortcode,
                post_type = excluded.post_type,
                classified_post_type = CASE
                    WHEN excluded.classified_post_type <> '' THEN excluded.classified_post_type
                    ELSE posts.classified_post_type
                END,
                classifier_confidence = CASE
                    WHEN excluded.classified_post_type <> '' THEN excluded.classifier_confidence
                    ELSE posts.classifier_confidence
                END,
                classifier_version = CASE
                    WHEN excluded.classified_post_type <> '' THEN excluded.classifier_version
                    ELSE posts.classifier_version
                END,
                classified_at = CASE
                    WHEN excluded.classified_post_type <> '' THEN excluded.classified_at
                    ELSE posts.classified_at
                END,
                caption = excluded.caption,
                preview_mode = excluded.preview_mode,
                preview_url = excluded.preview_url,
                embed_url = excluded.embed_url,
                niche = excluded.niche,
                collected_at = excluded.collected_at
            """,
            (
                platform,
                account_handle,
                post_url,
                shortcode,
                tag,
                classified_post_type,
                float(classifier_confidence or 0),
                classifier_version,
                classified_at,
                caption,
                preview_mode,
                preview_url,
                embed_url,
                niche,
                now,
                now,
            ),
        )
        row = conn.execute("SELECT * FROM posts WHERE post_url = ?", (post_url,)).fetchone()

    return dict(row) if row else {}


def search_posts_index(platform: str, style: str, niche: str = "", limit: int = 24) -> list:
    platform = (platform or "").lower().strip()
    style = (style or "").lower().strip()
    niche = (niche or "").strip().lower()

    where = ["platform = ?"]
    params = [platform]

    if style:
        if style == "carousel":
            where.append("post_type = ?")
            params.append("carousel")
        elif style in ("single-clip", "multi-clip", "talking-head"):
            where.append("classified_post_type = ?")
            params.append(style)
        elif style == "reel":
            where.append("post_type = ?")
            params.append("reel")
        else:
            where.append("post_type = ?")
            params.append(style)

    if niche:
        where.append("(LOWER(niche) LIKE ? OR LOWER(caption) LIKE ? OR LOWER(account_handle) LIKE ?)")
        like = f"%{niche}%"
        params.extend([like, like, like])

    sql = f"""
        SELECT *
        FROM posts
        WHERE {' AND '.join(where)}
        ORDER BY collected_at DESC, id DESC
        LIMIT ?
    """
    params.append(limit)

    with get_db() as conn:
        rows = conn.execute(sql, params).fetchall()

    out = []
    for r in rows:
        row = dict(r)
        out.append(
            {
                "title": (row.get("caption") or row.get("post_url") or "").splitlines()[0][:140],
                "url": row.get("post_url") or "",
                "platform": row.get("platform") or platform,
                "tag": row.get("classified_post_type") or style,
                "raw_post_type": row.get("post_type") or "",
                "classified_post_type": row.get("classified_post_type") or "",
                "classifier_confidence": row.get("classifier_confidence") or 0,
                "description": row.get("caption") or "",
                "thumbnail": row.get("preview_url") or "",
                "preview_mode": row.get("preview_mode") or "",
                "embed_url": row.get("embed_url") or "",
                "account_handle": row.get("account_handle") or "",
                "niche": row.get("niche") or "",
                "source": "local_index",
            }
        )
    return out





# --- Instagram seed account collector ---

def build_instagram_session_headers(referer_url: str = "https://www.instagram.com/") -> tuple[dict, dict]:
    """
    Build headers/cookies for authenticated Instagram web requests.
    These env vars are optional. If absent, requests still run in public mode.

    Supported env vars:
      INSTAGRAM_SESSIONID
      INSTAGRAM_CSRFTOKEN
      INSTAGRAM_DS_USER_ID
      INSTAGRAM_MID
      INSTAGRAM_IG_DID
      INSTAGRAM_RUR
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Accept": "*/*",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Referer": referer_url,
        # This app id is commonly used by the Instagram web app.
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



def profile_api_payload_has_media(data: dict) -> bool:
    """
    Return True when the Instagram profile payload clearly contains media edges.
    """
    if not isinstance(data, dict):
        return False

    try:
        edges = (((data.get("data") or {}).get("user") or {}).get("edge_owner_to_timeline_media") or {}).get("edges") or []
        if isinstance(edges, list) and len(edges) > 0:
            return True
    except Exception:
        pass

    return False

def fetch_instagram_profile_api_payload(handle: str) -> dict:
    """
    First-class V1 collector source.
    Attempts Instagram's web profile info endpoint using authenticated session headers/cookies.
    Returns {} on any failure so callers can safely fall back.
    """
    handle = (handle or "").strip().lstrip("@")
    if not handle:
        return {}

    url = "https://www.instagram.com/api/v1/users/web_profile_info/"
    params = {"username": handle}

    headers, cookies = build_instagram_session_headers(
        referer_url=f"https://www.instagram.com/{handle}/"
    )

    cookie_parts = []
    for cookie_name in ("sessionid", "csrftoken", "ds_user_id", "mid", "ig_did", "rur"):
        cookie_value = (cookies.get(cookie_name) or "").strip()
        if cookie_value:
            cookie_parts.append(f"{cookie_name}={cookie_value}")

    if cookie_parts:
        headers["Cookie"] = "; ".join(cookie_parts)


    try:
        r = requests.get(
            url,
            params=params,
            headers=headers,
            timeout=20,
            allow_redirects=True,
        )
        r.raise_for_status()
        data = r.json()
        if profile_api_payload_has_media(data):
            return data
    except Exception as e:
        print(f"[fetch_instagram_profile_api_payload][requests] failed for @{handle}: {e}")

    # Fallback: use the exact curl-style request path that already proved to work
    # in the terminal for this endpoint.
    try:
        curl_cmd = [
            "curl",
            "-s",
            f"{url}?username={handle}",
            "-H",
            f"X-IG-App-ID: {headers.get('X-IG-App-ID', '')}",
            "-H",
            f"X-ASBD-ID: {headers.get('X-ASBD-ID', '')}",
            "-H",
            "X-Requested-With: XMLHttpRequest",
            "-H",
            f"X-CSRFToken: {(cookies.get('csrftoken') or '').strip()}",
            "-H",
            f"Referer: https://www.instagram.com/{handle}/",
        ]

        if cookie_parts:
            curl_cmd.extend(["-b", "; ".join(cookie_parts)])

        result = subprocess.run(
            curl_cmd,
            capture_output=True,
            text=True,
            timeout=20,
        )

        raw = (result.stdout or "").strip()

        try:
            DATA_DIR.mkdir(parents=True, exist_ok=True)
            (DATA_DIR / f"debug_{handle}_curl_raw.json").write_text(raw, encoding="utf-8")
            (DATA_DIR / f"debug_{handle}_curl_stderr.txt").write_text(result.stderr or "", encoding="utf-8")
        except Exception:
            pass

        if raw:
            data = json.loads(raw)
            return data if isinstance(data, dict) else {}
    except Exception as e:
        print(f"[fetch_instagram_profile_api_payload][curl fallback] failed for @{handle}: {e}")

    return {}

def extract_post_urls_from_profile_api_payload(data: dict, max_posts: int = 12) -> list[str]:
    """
    Parse post/reel URLs from Instagram's web profile payload.
    Supports a couple of payload shapes seen in the wild.
    """
    if not isinstance(data, dict):
        return []

    candidates = []

    def append_shortcode(shortcode: str, is_video: bool = False):
        shortcode = (shortcode or "").strip()
        if not shortcode:
            return
        kind = "reel" if is_video else "p"
        candidates.append(f"https://www.instagram.com/{kind}/{shortcode}/")

    user = data.get("data", {}).get("user") if isinstance(data.get("data"), dict) else None
    if isinstance(user, dict):
        edges = (((user.get("edge_owner_to_timeline_media") or {}).get("edges"))
                 if isinstance(user.get("edge_owner_to_timeline_media"), dict) else None)
        if isinstance(edges, list):
            for edge in edges:
                node = edge.get("node") if isinstance(edge, dict) else None
                if not isinstance(node, dict):
                    continue
                append_shortcode(node.get("shortcode") or "", bool(node.get("is_video")))
        for key in ("timeline_media", "edge_felix_video_timeline", "edge_saved_media"):
            bucket = user.get(key)
            if isinstance(bucket, dict):
                edges2 = bucket.get("edges")
                if isinstance(edges2, list):
                    for edge in edges2:
                        node = edge.get("node") if isinstance(edge, dict) else None
                        if not isinstance(node, dict):
                            continue
                        append_shortcode(node.get("shortcode") or "", bool(node.get("is_video")))

    items = data.get("items")
    if isinstance(items, list):
        for item in items:
            if not isinstance(item, dict):
                continue
            append_shortcode(item.get("code") or item.get("shortcode") or "", bool(item.get("media_type") == 2))
    
    def walk(obj):
        if isinstance(obj, dict):
            shortcode = obj.get("shortcode") or obj.get("code") or ""
            is_video = bool(obj.get("is_video")) or bool(obj.get("media_type") == 2)
            if shortcode:
                append_shortcode(shortcode, is_video)
            for value in obj.values():
                walk(value)
        elif isinstance(obj, list):
            for item in obj:
                walk(item)
    walk(data)

    seen = set()
    out = []
    for url in candidates:
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(url)
        if len(out) >= max_posts:
            break
    return out

def mark_seed_account_crawled(platform: str, handle: str) -> None:
    platform = (platform or "").lower().strip()
    handle = (handle or "").strip().lstrip("@")
    now = utc_now_iso()
    with get_db() as conn:
        conn.execute(
            """
            UPDATE seed_accounts
            SET last_crawled_at = ?
            WHERE platform = ? AND handle = ?
            """,
            (now, platform, handle),
        )



def fetch_instagram_profile_post_urls(handle: str, max_posts: int = 12) -> list[str]:
    """
    Best-effort V1 collector for public Instagram profiles.
    Reads the public profile page HTML and extracts /p/ and /reel/ links.
    Instagram changes its HTML often, so we scan several patterns.
    """
    handle = (handle or "").strip().lstrip("@")
    if not handle:
        return []

    profile_url = f"https://www.instagram.com/{handle}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Referer": "https://www.instagram.com/",
    }

    try:
        r = requests.get(profile_url, headers=headers, timeout=20, allow_redirects=True)
        r.raise_for_status()
        html = r.text or ""
    except Exception:
        return []

    candidates = []

    patterns = [
        # Normal hrefs in server-rendered HTML
        r'href=["\'](/(?:p|reel|reels)/[A-Za-z0-9_-]+/)["\']',
        # JSON escaped relative paths like \/p\/ABC123\/
        r'\\/(?:p|reel|reels)\\/[A-Za-z0-9_-]+\\/',
        # Absolute Instagram URLs inside JSON/script blobs
        r'https:\\/\\/www\.instagram\.com\\/(?:p|reel|reels)\\/[A-Za-z0-9_-]+\\/',
        # Non-escaped absolute URLs
        r'https://www\.instagram\.com/(?:p|reel|reels)/[A-Za-z0-9_-]+/',
        # permalink fields in embedded JSON
        r'"permalink"\s*:\s*"(https?:\\/\\/www\.instagram\.com\\/(?:p|reel|reels)\\/[A-Za-z0-9_-]+\\/)"',
        # Plain shortcode fields in JSON blobs
        r'"shortcode"\s*:\s*"([A-Za-z0-9_-]+)"',
        # mediaType hint next to shortcode (2=video/reel, 8=carousel, 1=image)
        r'"code"\s*:\s*"([A-Za-z0-9_-]+)"',
    ]

    for pattern in patterns:
        matches = re.findall(pattern, html, flags=re.IGNORECASE)
        if not matches:
            continue
        if isinstance(matches[0], tuple):
            for tup in matches:
                for item in tup:
                    if item:
                        candidates.append(item)
        else:
            candidates.extend(matches)

    # Secondary extraction path: build URLs from shortcode/mediaType JSON when
    # Instagram does not expose direct permalinks in the HTML.
    shortcode_entries = []

    # Example shapes seen in IG HTML blobs:
    #   "shortcode":"ABC123"
    #   "mediaType":8
    # or
    #   "code":"ABC123"
    #   "media_type":2
    pair_patterns = [
        re.compile(r'"shortcode"\s*:\s*"([A-Za-z0-9_-]+)".{0,200}?"mediaType"\s*:\s*(\d+)', re.IGNORECASE | re.DOTALL),
        re.compile(r'"mediaType"\s*:\s*(\d+).{0,200}?"shortcode"\s*:\s*"([A-Za-z0-9_-]+)"', re.IGNORECASE | re.DOTALL),
        re.compile(r'"code"\s*:\s*"([A-Za-z0-9_-]+)".{0,200}?"media_type"\s*:\s*(\d+)', re.IGNORECASE | re.DOTALL),
        re.compile(r'"media_type"\s*:\s*(\d+).{0,200}?"code"\s*:\s*"([A-Za-z0-9_-]+)"', re.IGNORECASE | re.DOTALL),
    ]

    for rx in pair_patterns:
        for m in rx.findall(html):
            if not m:
                continue
            if isinstance(m, tuple) and len(m) == 2:
                a, b = m
                # normalize whether regex returned (shortcode, mediaType) or reverse
                if str(a).isdigit():
                    media_type = int(a)
                    shortcode = str(b)
                else:
                    shortcode = str(a)
                    media_type = int(b) if str(b).isdigit() else 0
                shortcode_entries.append((shortcode, media_type))

    for shortcode, media_type in shortcode_entries:
        if not shortcode:
            continue
        kind = "reel" if media_type == 2 else "p"
        candidates.append(f"https://www.instagram.com/{kind}/{shortcode}/")

    # Final fallback: if we only saw plain shortcodes with no media type,
    # assume /p/ so we at least collect likely post URLs.
    plain_shortcodes = re.findall(r'"shortcode"\s*:\s*"([A-Za-z0-9_-]+)"', html, flags=re.IGNORECASE)
    for shortcode in plain_shortcodes:
        if shortcode:
            candidates.append(f"https://www.instagram.com/p/{shortcode}/")

    seen = set()
    out = []

    for raw in candidates:
        raw = (raw or "").strip()
        if not raw:
            continue

        # Unescape JSON-style URLs
        normalized = raw.replace('\\/', '/')

        # Convert relative paths to full URLs
        if normalized.startswith("/"):
            full_url = f"https://www.instagram.com{normalized}"
        else:
            full_url = normalized

        # Final sanity check
        if not re.search(r'https://www\.instagram\.com/(?:p|reel|reels)/[A-Za-z0-9_-]+/', full_url, flags=re.IGNORECASE):
            continue

        # Skip obvious non-post URLs and malformed placeholders.
        if any(x in full_url for x in ("/explore/", "/accounts/", "/stories/")):
            continue
        if full_url in seen:
            continue
        seen.add(full_url)
        out.append(full_url)

        if len(out) >= max_posts:
            break

    return out

def fetch_instagram_posts_via_ytdlp(handle: str, max_posts: int = 12) -> list[str]:
    handle = (handle or "").strip().lstrip("@")
    if not handle:
        return []
    
    url = f"https://www.instagram.com/{handle}/"
    try:
        result = subprocess.run(
            [
                "yt-dlp",
                "--flat-playlist",
                "--dump-json",
                url,
            ],
            capture_output=True,
            text=True,
            timeout=30
        )

        lines = result.stdout.strip().split("\n")
        urls = []
        for line in lines:
            if not line.strip():
                continue
            try:
                data = json.loads(line)
                post_url = data.get("url") or ""
                if post_url.startswith("http"):
                    urls.append(post_url)
            except:
                continue
        return urls[:max_posts]
    except Exception:
        return []

def debug_fetch_instagram_profile(handle: str) -> dict:
    handle = (handle or "").strip().lstrip("@")
    if not handle:
        return {"ok": False, "error": "handle is required"}

    profile_url = f"https://www.instagram.com/{handle}/"
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Pragma": "no-cache",
        "Referer": "https://www.instagram.com/",
    }

    try:
        r = requests.get(profile_url, headers=headers, timeout=20, allow_redirects=True)
        status_code = r.status_code
        final_url = str(r.url or profile_url)
        html = r.text or ""
        content_type = r.headers.get("Content-Type", "")
    except Exception as e:
        return {"ok": False, "error": str(e)}

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    debug_file = DATA_DIR / f"debug_{handle}.html"
    try:
        debug_file.write_text(html, encoding="utf-8")
    except Exception:
        pass

    api_payload = fetch_instagram_profile_api_payload(handle)
    api_file = DATA_DIR / f"debug_{handle}_api.json"
    try:
        if api_payload:
            api_file.write_text(json.dumps(api_payload, indent=2), encoding="utf-8")
    except Exception:
        pass

    extracted_html = fetch_instagram_profile_post_urls(handle, max_posts=20)
    extracted_api = extract_post_urls_from_profile_api_payload(api_payload, max_posts=20)

    return {
        "ok": True,
        "handle": handle,
        "profile_url": profile_url,
        "final_url": final_url,
        "status_code": status_code,
        "content_type": content_type,
        "html_length": len(html),
        "saved_html": str(debug_file),
        "saved_api_payload": str(api_file) if api_payload else "",
        "api_payload_present": bool(api_payload),
        "extracted_html_count": len(extracted_html),
        "extracted_html": extracted_html[:20],
        "extracted_api_count": len(extracted_api),
        "extracted_api": extracted_api[:20],
        "html_preview": html[:1000],
    }



def collect_instagram_seed_account(handle: str, niche: str = "", max_posts: int = 12) -> list:
    handle = (handle or "").strip().lstrip("@")
    niche = (niche or "").strip()

    urls = []

    # 1. Primary: extract post/reel URLs from the public profile HTML.
    urls = fetch_instagram_profile_post_urls(handle, max_posts=max_posts)

    # 2. Secondary: try the authenticated profile API if HTML yields nothing.
    if not urls:
        api_payload = fetch_instagram_profile_api_payload(handle)
        urls = extract_post_urls_from_profile_api_payload(api_payload, max_posts=max_posts)
    
    if not urls:
        print(f"[Playwright fallback] Fetching posts for @{handle}")
        from .playwright_helper import fetch_instagram_posts_playwright
        urls = fetch_instagram_posts_playwright(handle, max_posts=max_posts)

    if not urls:
        print(f"[yt-dlp fallback] Fetching posts for @{handle}")
        urls = fetch_instagram_posts_via_ytdlp(handle, max_posts=max_posts)

    created = []

    for post_url in urls:
        path = (urlparse(post_url).path or "").lower()
        tag = "carousel" if "/p/" in path else "reel"
        print("DEBUG COLLECT TAG:", post_url, "->", tag)

        unfurled = unfurl_url(post_url)
        title = ""
        preview_url = ""
        if unfurled.get("ok"):
            title = (unfurled.get("title") or "").strip()
            preview_url = (unfurled.get("image") or "").strip()

        # If the unfurled title is generic or empty, fall back to something deterministic.
        if not title or title.lower() in ("instagram", "instagram post"):
            shortcode = shortcode_from_url(post_url)
            title = f"{handle} {shortcode}".strip()

        row = upsert_post(
            platform="instagram",
            post_url=post_url,
            title=title,
            description="",
            tag=tag,
            preview_url=preview_url,
            account_handle=handle,
            niche=niche,
            classified_post_type="carousel" if tag == "carousel" else "",
            classifier_confidence=0.99 if tag == "carousel" else 0.0,
            classifier_version="raw_structural_v1" if tag == "carousel" else "",
        )
        created.append(row)

    mark_seed_account_crawled("instagram", handle)
    return created

class BoardPlayload(BaseModel):
    board: list


# --- SQLite persistence models ---
class SeedAccountPayload(BaseModel):
    platform: str
    handle: str
    niche: str = ""


class PostPayload(BaseModel):
    platform: str
    post_url: str
    title: str = ""
    description: str = ""
    tag: str = ""
    preview_url: str = ""
    account_handle: str = ""
    niche: str = ""
    classified_post_type: str = ""
    classifier_confidence: float = 0.0
    classifier_version: str = ""

@app.post("/api/board/save")
def save_board(payload: BoardPlayload):
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    BOARD_FILE.write_text(json.dumps(payload.board, indent=2), encoding="utf-8")
    return {"ok": True, "saved": len(payload.board)}

@app.get("/api/board/load")
def load_board():
    if not BOARD_FILE.exists():
        return {"ok": True, "board": []}
    
    raw = BOARD_FILE.read_text(encoding="utf-8").strip()
    if not raw:
        return {"ok": True, "board": []}
    
    try:
        data = json.loads(raw)
    except JSONDecodeError:
        # If the file somehow got corrupted, don't crash the app
        return {"ok": True, "board": []}
        
   
    return {"ok": True, "board": data}


# --- SQLite persistence API endpoints ---
@app.post("/api/seed_accounts")
def api_create_seed_account(payload: SeedAccountPayload):
    try:
        row = create_seed_account(payload.platform, payload.handle, payload.niche)
        return {"ok": True, "seed_account": row}
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)


@app.get("/api/seed_accounts")
def api_list_seed_accounts():
    return {"ok": True, "seed_accounts": list_seed_accounts()}


@app.post("/api/posts")
def api_upsert_post(payload: PostPayload):
    try:
        row = upsert_post(
            platform=payload.platform,
            post_url=payload.post_url,
            title=payload.title,
            description=payload.description,
            tag=payload.tag,
            preview_url=payload.preview_url,
            account_handle=payload.account_handle,
            niche=payload.niche,
            classified_post_type=payload.classified_post_type,
            classifier_confidence=payload.classifier_confidence,
            classifier_version=payload.classifier_version,
        )
        return {"ok": True, "post": row}
    except ValueError as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)
    

@app.get("/api/debug/instagram_profile")
def api_debug_instagram_profile(handle: str = ""):
    result = debug_fetch_instagram_profile(handle)
    if not result.get("ok"):
        return JSONResponse(result, status_code=400)
    return result

def classify_reel_row(row: sqlite3.Row) -> tuple[str, float, str]:
    """
    Temporary V1 classifier stub.
    Right now this still uses text heuristics.
    Later, this is the exact function we will replace
    with real rendered-embed visual classification.
    """
    post_url = row["post_url"] or ""
    caption = row["caption"] or ""
    raw_post_type = (row["post_type"] or "").strip().lower()

    if raw_post_type == "carousel":
        return ("carousel", 0.99, "raw_structural_v1")
    
    if raw_post_type != "reel":
        return ("", 0.0, "")
    
    analysis = analyze_item(
        url=post_url,
        title=caption.splitlines()[0] if caption else "",
        description=caption,
        platform="instagram",
        tag="",
        niche=row["niche"] or "",
        seed="",
    )

    style = (analysis.get("style") or "").strip().lower()
    confidence = float(analysis.get("confidence") or 0)

    if style in ("single-clip", "multi-clip", "talking-head"):
        return (style, confidence, "heuristic_text_v1")
    
    return ("", 0.0, "")

@app.post("/api/classify/reels")
def api_classify_reels(limit: int = 10, fps: float = 1.0):
    """
    Main production async classifier entry point.

    This route no longer performs expensive classification work inline.
    Instead it:
    1. finds pending Instagram reels
    2. creates tracking rows in SQLite
    3. publishes one durable RabbitMQ job per reel

    Actual processing is handled by a worker process.
    """
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM posts
            WHERE platform = 'instagram'
              AND post_type = 'reel'
              AND (classified_post_type = '' OR classified_post_type IS NULL)
            ORDER BY collected_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    queued = []

    for row in rows:
        post_url = (row["post_url"] or "").strip()
        if not post_url:
            continue

        # Deduplication: do not enqueue the same reel if a queued or
        # currently processing classification job already exists.
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
                (post_url, JOB_CLASSIFY_REEL_VIDEO),
            ).fetchone()

        if existing:
            continue

        job = create_crawl_job(
            job_type=JOB_CLASSIFY_REEL_VIDEO,
            target=post_url,
            status="queued",
        )

        publish_rabbitmq_job(
            job_type=JOB_CLASSIFY_REEL_VIDEO,
            target=post_url,
            payload={
                "job_id": job["id"],
                "post_url": post_url,
                "platform": row["platform"] or "instagram",
                "account_handle": row["account_handle"] or "",
                "niche": row["niche"] or "",
                "fps": float(fps or 1.0),
            },
        )

        queued.append(job)

    return {
        "ok": True,
        "queued": len(queued),
        "jobs": queued,
        "job_type": JOB_CLASSIFY_REEL_VIDEO,
        "mode": "async",
        "note": "Jobs were enqueued to RabbitMQ. A worker process must consume and execute them.",
    }


# Alias route for explicit video+AI classifier path
@app.post("/api/classify/reels/video_ai")
def api_classify_reels_video_ai(limit: int = 5, fps: float = 1.0):
    """
    Explicit alias for the main async RabbitMQ-backed reel classifier.
    This calls the same enqueue path as /api/classify/reels.
    """
    return api_classify_reels(limit=limit, fps=fps)

@app.get("/api/jobs/failed")
def api_list_failed_jobs(limit: int = 50):
    """
    List recently failed async jobs.
    """
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM crawl_jobs
            WHERE status = 'failed'
            ORDER BY COALESCE(finished_at, created_at) DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()

    jobs = [dict(r) for r in rows]
    return {
        "ok": True,
        "failed": len(jobs),
        "jobs": jobs,
    }

@app.get("/api/jobs")
def api_list_jobs(
    status: str = "",
    job_type: str = "",
    limit: int = 50,
):
    """
    List jobs with optional filtering by status and job_type.
    """
    where = []
    params = []
    if status:
        where.append("status = ?")
        params.append(status)
    if job_type:
        where.append("job_type = ?")
        params.append(job_type)
    
    where_sql = ""
    if where:
        where_sql = "WHERE " + " AND ".join(where)
    
    query = f"""
        SELECT *
        FROM crawl_jobs
        {where_sql}
        ORDER BY COALESCE(finished_at, created_at) DESC, id DESC
        LIMIT ?
    """
    params.append(limit)
    with get_db() as conn:
        rows = conn.execute(query, params).fetchall()
    jobs = [dict(r) for r in rows]
    
    return {
        "ok": True, 
        "count": len(jobs),
        "jobs": jobs,
        "filters": {
            "status": status,
            "job_type": job_type,
            "limit": limit,
        },
    }


# --- Stale job inspection and recovery endpoints ---

@app.get("/api/jobs/stale")
def api_list_stale_jobs(stale_after_minutes: int = 15, limit: int = 50):
    """
    List jobs stuck in processing longer than the stale threshold.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=stale_after_minutes)).isoformat()

    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM crawl_jobs
            WHERE status = 'processing'
              AND started_at <> ''
              AND started_at IS NOT NULL
              AND started_at < ?
            ORDER BY started_at ASC, id ASC
            LIMIT ?
            """,
            (cutoff, limit),
        ).fetchall()

    jobs = [dict(r) for r in rows]
    return {
        "ok": True,
        "stale": len(jobs),
        "jobs": jobs,
        "filters": {
            "stale_after_minutes": stale_after_minutes,
            "limit": limit,
        },
    }


@app.post("/api/jobs/recover_stale")
def api_recover_stale_jobs(stale_after_minutes: int = 15, limit: int = 25):
    """
    Recover stale processing jobs by marking them failed and, when safe,
    creating a fresh queued job and publishing it to RabbitMQ.
    """
    cutoff = (datetime.now(timezone.utc) - timedelta(minutes=stale_after_minutes)).isoformat()

    with get_db() as conn:
        stale_rows = conn.execute(
            """
            SELECT *
            FROM crawl_jobs
            WHERE status = 'processing'
              AND started_at <> ''
              AND started_at IS NOT NULL
              AND started_at < ?
            ORDER BY started_at ASC, id ASC
            LIMIT ?
            """,
            (cutoff, limit),
        ).fetchall()

    recovered = []

    for row in stale_rows:
        stale_job = dict(row)
        job_id = stale_job.get("id") or 0
        job_type = (stale_job.get("job_type") or "").strip()
        target = (stale_job.get("target") or "").strip()

        if not job_id or not target or job_type != JOB_CLASSIFY_REEL_VIDEO:
            continue

        update_crawl_job_status(
            job_id=job_id,
            status="failed",
            error_message="stale processing job recovered",
            finished_at=utc_now_iso(),
        )

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
                (target, job_type),
            ).fetchone()
            if existing:
                recovered.append(
                    {
                        "stale_job_id": job_id,
                        "queued": 0,
                        "skipped": True,
                        "reason": "already_active",
                        "existing_job_id": existing[0],
                    }
                )
                continue

            newer_completed = conn.execute(
                """
                SELECT id
                FROM crawl_jobs
                WHERE target = ?
                  AND job_type = ?
                  AND status = 'completed'
                  AND id > ?
                ORDER BY id DESC
                LIMIT 1
                """,
                (target, job_type, job_id),
            ).fetchone()
            if newer_completed:
                recovered.append(
                    {
                        "stale_job_id": job_id,
                        "queued": 0,
                        "skipped": True,
                        "reason": "newer_completed_job_exists",
                        "existing_job_id": newer_completed[0],
                    }
                )
                continue

            post_row = conn.execute(
                "SELECT * FROM posts WHERE post_url = ? LIMIT 1",
                (target,),
            ).fetchone()

        new_job = create_crawl_job(
            job_type=job_type,
            target=target,
            status="queued",
        )

        payload = {
            "job_id": new_job["id"],
            "post_url": target,
            "platform": "instagram",
            "account_handle": "",
            "niche": "",
            "fps": 1.0,
        }

        if post_row:
            payload.update(
                {
                    "platform": post_row["platform"] or "instagram",
                    "account_handle": post_row["account_handle"] or "",
                    "niche": post_row["niche"] or "",
                }
            )

        publish_rabbitmq_job(
            job_type=job_type,
            target=target,
            payload=payload,
        )

        recovered.append(
            {
                "stale_job_id": job_id,
                "queued": 1,
                "new_job": new_job,
            }
        )

    return {
        "ok": True,
        "recovered": len(recovered),
        "jobs": recovered,
        "filters": {
            "stale_after_minutes": stale_after_minutes,
            "limit": limit,
        },
    }


@app.post("/api/jobs/{job_id}/retry")
def api_retry_failed_job(job_id: int):
    """
    Retry failed job.
    """
    job = get_crawl_job(job_id)
    if not job:
        return {"ok": False, "error": "Job not found"}
    if (job.get("status") or "").strip() != "failed":
        return {"ok": False, "error": "Only failed jobs can be retried"}
    
    job_type = (job.get("job_type") or "").strip()
    target = (job.get("target") or "").strip()
    if job_type != JOB_CLASSIFY_REEL_VIDEO:
        return {"ok": False, "error": f"Unsupported job_type: {job_type}"}
    
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
            (target, job_type),
        ).fetchone()
        if existing:
            return {
                "ok": True,
                "queued": 0,
                "skipped": True,
                "reason": "already_active",
                "existing_job_id": existing[0],
            }

        newer_completed = conn.execute(
            """
            SELECT id
            FROM crawl_jobs
            WHERE target = ?
              AND job_type = ?
              AND status = 'completed'
              AND id > ?
            ORDER BY id DESC
            LIMIT 1
            """,
            (target, job_type, job_id),
        ).fetchone()
        if newer_completed:
            return {
                "ok": True,
                "queued": 0,
                "skipped": True,
                "reason": "newer_completed_job_exists",
                "existing_job_id": newer_completed[0],
            }

        post_row = conn.execute(
            "SELECT * FROM posts WHERE post_url = ? LIMIT 1",
            (target,),
        ).fetchone()

    new_job = create_crawl_job(
        job_type=job_type,
        target=target,
        status="queued",
    )

    payload = {
        "job_id": new_job["id"],
        "post_url": target,
        "platform": "instagram",
        "account_handle": "",
        "niche": "",
        "fps": 1.0,
    }

    if post_row:
        payload.update({
            "platform": post_row["platform"] or "instagram",
            "account_handle": post_row["account_handle"] or "",
            "niche": post_row["niche"] or "",
        })
    
    publish_rabbitmq_job(
        job_type=job_type,
        target=target,
        payload=payload,
    )

    return {
        "ok": True,
        "queued": 1,
        "original_job_id": job_id,
        "new_job": new_job,
    }

@app.get("/api/reels/pending_visual_classification")
def api_pending_visual_classification(limit: int = 24):
    """
    Return raw Instagram reels that still need visual classification.
    The frontend can use this to know which rendered embeds still need to be analyzed.
    """
    with get_db() as conn:
        rows = conn.execute(
            """
            SELECT *
            FROM posts
            WHERE platform = 'instagram'
                AND post_type = 'reel'
                AND (classified_post_type = '' OR classified_post_type IS NULL)
            ORDER BY collected_at DESC, id DESC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    
    out = []
    for row in rows:
        item = dict(row)
        out.append(
            {
                "post_url": item.get("post_url") or "",
                "embed_url": item.get("embed_url") or "",
                "caption": item.get("caption") or "",
                "account_handle": item.get("account_handle") or "",
                "niche": item.get("niche") or "",
                "raw_post_type": item.get("post_type") or "",
            }
        )
    
    return {
        "ok": True,
        "pending": len(out),
        "posts": out,
        "note": "These reels should be classified from the actual rendered embed content shown in the board.",
    }

class VisualClassificationPayload(BaseModel):
    post_url: str
    text_density: float = 0.0
    scene_change_score: float = 0.0
    face_ratio: float = 0.0
    has_large_face: bool = False
    sampled_frames: int = 0
    source: str = "rendered_embed_v1"


class VisualClassificationBatchPayload(BaseModel):
    items: list[VisualClassificationPayload]

@app.post("/api/classify/reels/visual")
def api_classify_reels_visual(payload: VisualClassificationBatchPayload):
    """
    Accept visual-signal measurements gathered from the actual rendered embeds
    in the board, then write the classification result back into SQLite.
    """
    updated = []

    for item in payload.items:
        classified_post_type, confidence, version = classify_reel_from_visual_signals(
            text_density=item.text_density,
            scene_change_score=item.scene_change_score,
            face_ratio=item.face_ratio,
            has_large_face=item.has_large_face,
            sampled_frames=item.sampled_frames,
        )

        if not classified_post_type:
            continue

        row = update_post_classification_by_url(
            post_url=item.post_url,
            classified_post_type=classified_post_type,
            classifier_confidence=confidence,
            classifier_version=f"{version}:{item.source}",
        )

        if row:
            updated.append(row)
    
    return {
        "ok": True,
        "classified": len(updated),
        "posts": updated,
        "classifier": "rendered_embed_visual_v1",
    }


@app.post("/api/crawl/seed_accounts")
def crawl_seed_accounts():
    """
    V1 real collector:
    - Reads seed accounts from DB
    - For Instagram seeds, fetches the public profile page
    - Extracts /p/ and /reel/ URLs
    - Upserts collected posts into the posts table
    - Updates last_crawled_at on each seed account
    """
    seeds = list_seed_accounts()
    if not seeds:
        return {"ok": True, "message": "No seed accounts found", "seed_accounts": 0, "posts_created": 0, "posts": []}

    created = []
    summary = []

    for seed in seeds:
        handle = seed.get("handle") or ""
        platform = (seed.get("platform") or "").lower().strip()
        niche = seed.get("niche") or ""

        if platform != "instagram":
            summary.append({
                "platform": platform,
                "handle": handle,
                "created": 0,
                "skipped": True,
                "reason": "platform_not_supported_yet",
            })
            continue

        rows = collect_instagram_seed_account(handle=handle, niche=niche, max_posts=12)
        created.extend(rows)
        summary.append({
            "platform": platform,
            "handle": handle,
            "created": len(rows),
            "skipped": False,
            "collector": "instagram_web_profile_info_then_html_fallback",
        })

    return {
        "ok": True,
        "seed_accounts": len(seeds),
        "posts_created": len(created),
        "summary": summary,
        "posts": created,
    }

# For local dev: allow the frontend to call the API easily

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later we can tighten this up
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- Startup hook for DB ---
@app.on_event("startup")
def startup_init():
    init_db()

@app.get("/api/health")
def health():
    return {"status": "ok", "db": str(DB_FILE), "db_exists": DB_FILE.exists()}

@app.get("/api/search")
def search(
    platform: str = "instagram",
    style: str = "carousel",
    niche: str = "",
    seed: str = "",
    best: int = 1,
    debug: int = 0,
):
    platform = (platform or "instagram").lower().strip()
    style = (style or "carousel").lower().strip()
    indexed_results = search_posts_index(platform=platform, style=style, niche=niche, limit=24)
    niche_health = get_niche_health(platform=platform, style=style, niche=niche)
    if indexed_results and niche_health.get("healthy"):
        resp = {
            "platform": platform,
            "style": style,
            "q": "local_index",
            "results": dedupe_and_truncate(indexed_results, limit=8),
        }
        if debug:
            resp["debug"] = {
                "source": "local_index",
                "db": str(DB_FILE),
                "count": len(indexed_results),
                "niche_health": niche_health,
            }
        return resp

    # If we reach here, we only use local DB results (no Brave fallback)
    results = dedupe_and_truncate(indexed_results, limit=8) if indexed_results else []

    resp = {
        "platform": platform,
        "style": style,
        "q": "local_index_only",
        "results": results,
    }

    if debug:
        resp["debug"] = {
            "source": "local_index_only",
            "db": str(DB_FILE),
            "local_index_count": len(indexed_results),
            "niche_health": niche_health,
            "used_local_only": True,
            "note": "Brave search fully removed. Results are DB-only.",
        }

    return resp
@app.post("/api/expand/niche")
def api_expand_niche(
    platform: str = "instagram",
    style: str = "",
    niche: str = "",
    limit: int = 10,
):
    """
    Expand local niche coverage by discovering accounts, adding missing seeds,
    and enqueueing crawl jobs for those accounts.
    """

    platform = (platform or "instagram").lower().strip()
    style = (style or "").lower().strip()
    niche = (niche or "").strip()
    limit = int(limit or 10)
    if not niche:
        return {
            "ok": False,
            "expanded": False,
            "reason": "niche_required",
            "discovered_accounts": [],
            "seed_results": [],
            "crawl_jobs": [],
        }
    
    return expand_niche_if_needed(
        platform=platform,
        style=style,
        niche=niche,
        limit=limit,
    )


from typing import Optional, Dict, Any, List
from urllib.parse import urlparse
import re

class AnalyzePayload(BaseModel):
    url: str
    title: Optional[str] = ""
    description: Optional[str] = ""
    platform: Optional[str] = ""
    tag: Optional[str] = ""
    niche: Optional[str] = ""
    seed: Optional[str] = ""

class ReelFramesPayload(BaseModel):
    post_url: str
    fps: float = 1.0


TUTORIAL_HINTS = [
    "make a carousel", "create a carousel",
    "carousel template", "carousel templates",
    "viral carousel", "carousel ideas", "carousel hooks", "hook ideas",
    "reel ideas", "reels ideas",
    "content strategy", "social media strategy", "instagram strategy", "tiktok strategy",
    "canva", "capcut",
    "video editing", "editing tutorial",
    "content ideas", "content calendar",
]


def classify_from_url(url: str) -> Dict[str, Any]:
    u = (url or "").strip()
    lu = u.lower()
    parsed = urlparse(lu)
    host = parsed.netloc or ""
    path = parsed.path or ""

    platform = "unknown"
    fmt = "unknown"

    if "instagram.com" in host:
        platform = "instagram"
        if "/p/" in path:
            fmt = "instagram_post"
        elif "/reel/" in path or "/reels/" in path:
            fmt = "instagram_reel"
        else:
            fmt = "instagram_other"

    elif "tiktok.com" in host:
        platform = "tiktok"
        if "/video/" in path:
            fmt = "tiktok_video"
        else:
            fmt = "tiktok_other"

    elif "x.com" in host or "twitter.com" in host:
        platform = "x"
        if "/status/" in path or "/statuses/" in path:
            fmt = "x_status"
        else:
            fmt = "x_other"

    return {"platform": platform, "format": fmt}

def looks_like_tutorial(text: str) -> bool:
    t = (text or "").lower()
    return any(h in t for h in TUTORIAL_HINTS)

def tokenize(text: str) -> set:
    # simple keyword tokenizer for niche matching
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9\s@#_-]+", " ", t)
    parts = [p.strip() for p in t.split() if p.strip()]
    # drop tiny tokens
    return set([p for p in parts if len(p) >= 3])

def score_niche_relevance(niche: str, text: str) -> float:
    """
    Returns 0.0–1.0 based on how strongly the content looks related to the niche.
    Lightweight heuristic: keyword overlap + a few boosts for exact phrase matches.
    """
    niche = (niche or "").strip().lower()
    if not niche:
        return 1.0 # if user didn't provide niche, don't block anything
    
    hay = (text or "").lower()

    # phrase boost if niche appears as a phrase
    phrase_boost = 0.25 if niche in hay else 0.0

    niche_tokens = tokenize(niche)
    text_tokens = tokenize(text)

    if not niche_tokens:
        return 0.0
    
    overlap = len(niche_tokens.intersection(text_tokens))
    ratio = overlap / max(len(niche_tokens), 1)

    # clamp + combine
    score = min(1.0, ratio + phrase_boost)

    return float(max(0.0, min(1.0, score)))


def infer_style(platform: str, tag: str, fmt: str, text: str = "") -> Dict[str, Any]:
    """
    Decide what style a piece of content most likely is.

    platform:
        which platform the content came from (instagram, tiktok, x)
    
    tag:
        The existing stored tag, if we already have one

    fmt:
        The structural format inferred from the URL
        Example:
            instagram_post
            instagram_reel
            tiktok_video
            x_status
    
    text:
        Combined searchable text signal from title + description + url
        We use this to improve reel subtype classification
    """

    platform = (platform or "").lower().strip()
    tag = (tag or "").lower().strip()
    fmt = (fmt or "").lower().strip()
    text = (text or "").lower().strip()

    if platform == "instagram":
        if fmt == "instagram_post":
            return {"style": "carousel", "confidence": 0.85}
        if fmt == "instagram_reel":
            if tag in ("multi-clip", "single-clip", "talking-head"):
                return {"style": tag, "confidence": 0.75}
            
            multi_clip_hints = [
                "part 1",
                "part 2",
                "pt 1",
                "pt.1",
                "series",
                "day 1",
                "episode",
            ]

            single_clip_hints = [
                "text overlay",
                "caption",
                "on screen text",
                "read this",
                "watch this",
            ]

            talking_head_hints = [
                "talking",
                "explaining",
                "face to camera",
                "direct to camera",
                "here's how",
                "let me explain",
                "i'm going to show you",
            ]

            if any(hint in text for hint in multi_clip_hints):
                return {"style": "multi-clip", "confidence": 0.7}
            
            if any(hint in text for hint in single_clip_hints):
                return {"style": "single-clip", "confidence": 0.7}
            
            if any(hint in text for hint in talking_head_hints):
                return {"style": "talking-head", "confidence": 0.7}

            return {"style": "reel", "confidence": 0.5}

    if platform == "tiktok":
        if fmt == "tiktok_video":
            return {"style": tag or "video", "confidence": 0.55}

    if platform == "x":
        if fmt == "x_status":
            return {"style": tag or "post", "confidence": 0.60}

    return {"style": tag or "unknown", "confidence": 0.30}


def analyze_item(url: str, title: str, description: str, platform: str, tag: str, niche: str, seed: str):
    base = classify_from_url(url)
    platform2 = (platform or base["platform"] or "unknown").lower().strip()
    fmt = base["format"]

    combined = f"{title or ''}\n{description or ''}\n{url or ''}".strip()

    niche_score = score_niche_relevance(niche or "", combined)
    niche_match = (niche_score >= 0.45) if (niche or "").strip() else True

    tutorial_flag = looks_like_tutorial(combined)
    style_guess = infer_style(platform2, tag or "", fmt, combined)

    return {
        "ok": True,
        "url": url,
        "platform": platform2,
        "format": fmt,
        "style": style_guess["style"],
        "confidence": style_guess["confidence"],
        "is_tutorial_like": tutorial_flag,
        "niche": (niche or "").strip(),
        "niche_score": niche_score,
        "niche_match": niche_match,
    }

def classify_reel_from_visual_signals(
        text_density: float,
        scene_change_score: float, 
        face_ratio: float,
        has_large_face: bool,
        sampled_frames: int,
) -> tuple[str, float, str]:
    """
    First real visual-classification decision layer.

    IMPORTANT:
    This function does NOT generate the visual signals itself.
    It expects the frontend/browser to inspect the actual rendered embed
    and POST the measured signals back to this backend.

    That keeps the classifier grounded in the same post content the user sees
    on the board, instead of relying on weak metadata or fallback preview guesses.
    """

    td = max(0.0, min(1.0, float(text_density or 0.0)))
    sc = max(0.0, min(1.0, float(scene_change_score or 0.0)))
    fr = max(0.0, min(1.0, float(face_ratio or 0.0)))
    sampled = max(0, int(sampled_frames or 0))
    has_face = bool(has_large_face)

    # Talking-head reels can still have a lot of on-screen text/captions.
    # So we should not reject them just because text_density is high.
    if has_face and fr >= 0.18 and sc <= 0.35:
        confidence = min(0.95, 0.70 + (fr * 0.35) - (sc * 0.10))
        return ("talking-head", float(max(0.0, confidence)), "rendered_embed_visual_v1")
    
    if sampled >= 3 and sc >= 0.45:
        confidence = min(0.95, 0.68 + (sc *0.25))
        return ("multi-clip", float(max(0.0, confidence)), "rendered_embed_visual_v1")
    
    if td >= 0.18 and sc <= 0.30 and (not has_face or fr < 0.18):
        confidence = min(0.92, 0.62 + (td * 0.35) - (sc * 0.10))
        return ("single-clip", float(max(0.0, confidence)), "rendered_embed_visual_v1")
    
    return ("", 0.0, "")

def update_post_classification_by_url(
        post_url: str, 
        classified_post_type: str,
        classifier_confidence: float, 
        classifier_version: str,
) -> dict:
    """
    Update one post's classified output fields using post_url as the key.
    """
    post_url = (post_url or "").strip()
    classified_post_type = (classified_post_type or "").strip()
    classifier_version = (classifier_version or "").strip()

    if not post_url:
        raise ValueError("post_url is required")
    if not classified_post_type:
        raise ValueError("classified_post_type is required")
    
    now = utc_now_iso()

    with get_db() as conn:
        conn.execute(
            """
            UPDATE posts
            SET classified_post_type = ?,
                classifier_confidence = ?,
                classifier_version = ?,
                classified_at = ?
            WHERE post_url = ?
            """,
            (
                classified_post_type,
                float(classifier_confidence or 0),
                classifier_version,
                now,
                post_url,
            ),
        )

        row = conn.execute(
            "SELECT * FROM posts WHERE post_url = ?",
            (post_url,),
        ).fetchone()

    return dict(row) if row else {}

def safe_media_stem_from_url(post_url: str) -> str:
    """
    Build a safe folder/file stem from the reel shortcode.
    Falls back to a generic name if shortcode extraction fails.
    """
    shortcode = shortcode_from_url(post_url)
    if shortcode:
        return f"ig_{shortcode}"
    return "ig_reel"

def download_instagram_reel_video(post_url: str) -> str:
    """
    Download the real Instagram reel video to disk using yt-dlp.

    Returns:
        Absolute path to the merged .mp4 file.

    Important:
        This is REAL media download, not thumbnails or preview images.
    """
    post_url = (post_url or "").strip()
    if not post_url:
        raise ValueError("post_url is required")
    
    media_dir = DATA_DIR / "media"
    media_dir.mkdir(parents=True, exist_ok=True)

    stem = safe_media_stem_from_url(post_url)

    output_template = str(media_dir / f"{stem}.%(ext)s")

    final_path = media_dir / f"{stem}.mp4"
    if final_path.exists():
        return str(final_path.resolve())

    cmd = [
        "yt-dlp",
        "--no-playlist",
        "--merge-output-format",
        "mp4",
        "-o",
        output_template,
        post_url,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"yt-dlp failed: {(result.stderr or result.stdout or '').strip()}"
        )

    if not final_path.exists():
        raise RuntimeError(f"Expected merged video not found: {final_path}")

    return str(final_path.resolve())

def extract_frames_from_video(video_path: str, fps: float = 1.0) -> list[str]:
    """
    Extract frames from a local video file using ffmpeg.

    Returns:
        A sorted list of absolute frame image paths.

    Important:
        This is the frame source we can later send into AI vision.
    """
    video_path = (video_path or "").strip()
    if not video_path:
        raise ValueError("video_path is required")
    
    video_file = Path(video_path)
    if not video_file.exists():
        raise FileNotFoundError(f"Video file not found: {video_file}")
    
    frames_dir = DATA_DIR / "frames" / video_file.stem
    frames_dir.mkdir(parents=True, exist_ok=True)

    existing_frames = sorted(str(p.resolve()) for p in frames_dir.glob("*.jpg"))
    if existing_frames:
        return existing_frames

    output_pattern = str(frames_dir / "frame_%03d.jpg")

    cmd = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_file),
        "-vf",
        f"fps={float(fps or 1.0)}",
        output_pattern,
    ]

    result = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed: {(result.stderr or result.stdout or '').strip()}"
        )

    frame_paths = sorted(str(p.resolve()) for p in frames_dir.glob("*.jpg"))
    if not frame_paths:
        raise RuntimeError("No frames were extracted")
    return frame_paths    

@app.post("/api/analyze")
def analyze(payload: AnalyzePayload):
    url = (payload.url or "").strip()
    if not url.startswith("http"):
        return JSONResponse({"ok": False, "error": "Invalid URL"}, status_code=400)

    base = classify_from_url(url)
    platform = (payload.platform or base["platform"] or "unknown").lower().strip()
    fmt = base["format"]

    title = payload.title or ""
    desc = payload.description or ""
    combined = f"{title}\n{desc}\n{url}".strip()
    niche_score = score_niche_relevance(payload.niche or "", combined)

    tutorial_flag = looks_like_tutorial(combined)
    style_guess = infer_style(platform, payload.tag or "", fmt, combined)

    notes = []
    if platform == "instagram" and fmt == "instagram_post":
        notes.append("Instagram /p/ link (post). Often used for carousels.")
    if platform == "instagram" and fmt == "instagram_reel":
        notes.append("Instagram /reel/ link (reel).")
        notes.append("Can’t auto-watch frames without IG access; using URL + text hints.")
    if tutorial_flag:
        notes.append("Looks tutorial-like (based on title/description keywords).")
    else:
        notes.append("Looks more like inspo/example (based on title/description).")

    return {
        "ok": True,
        "url": url,
        "platform": platform,
        "format": fmt,
        "style": style_guess["style"],
        "confidence": style_guess["confidence"],
        "is_tutorial_like": tutorial_flag,
        "notes": notes,
        "niche": (payload.niche or "").strip(),
        "niche_score": niche_score,
        "niche_match": niche_score >= 0.45,
    }

@app.post("/api/debug/reel_frames")
def api_debug_reel_frames(payload: ReelFramesPayload):
    """
    One-off backend test route:
    1. Download the real Instagram reel video
    2. Extract frames with ffmpeg
    3. Return the saved file paths

    This is the clean bridge into AI vision later.
    """
    try:
        video_path = download_instagram_reel_video(payload.post_url)
        frame_paths = extract_frames_from_video(video_path, fps=payload.fps)
        frame_analysis = analyze_frames_with_ai(frame_paths)
        classified_post_type, classifier_confidence, classifier_version = classify_reel_from_visual_signals(
            text_density=frame_analysis.get("text_density", 0.0),
            scene_change_score=frame_analysis.get("scene_change_score", 0.0),
            face_ratio=frame_analysis.get("face_ratio", 0.0),
            has_large_face=frame_analysis.get("has_large_face", False),
            sampled_frames=len(frame_paths),
        )

        updated_post = None
        if classified_post_type:
            updated_post = update_post_classification_by_url(
                post_url=payload.post_url,
                classified_post_type=classified_post_type,
                classifier_confidence=classifier_confidence,
                classifier_version=f"{classifier_version}:openai_frames_v1",
            )

        return {
            "ok": True,
            "post_url": payload.post_url,
            "video_path": video_path,
            "frames_extracted": len(frame_paths),
            "analysis": frame_analysis,
            "classified_post_type": classified_post_type,
            "classifier_confidence": classifier_confidence,
            "classifier_version": f"{classifier_version}:openai_frames_v1" if classified_post_type else "",
            "db_updated": bool(updated_post),
            "updated_post": updated_post,
        }
    except Exception as e:
        return JSONResponse(
            {"ok": False, "error": str(e)},
            status_code=400,
        )


def analyze_frames_with_ai(frame_paths: list[str]) -> dict:
    """
    Send sampled frames to OpenAI Vision for classification.
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")
    
    max_frames = 6
    if len(frame_paths) <= max_frames:
        sample = frame_paths
    else:
        step = (len(frame_paths) - 1) / float(max_frames - 1)
        sample_indexes = [round(i * step) for i in range(max_frames)]
        sample = [frame_paths[i] for i in sample_indexes]
    print("DEBUG AI FRAME SAMPLE:", sample)

    images = []
    for path in sample:
        with open(path, "rb") as f:
            b64 = base64.b64encode(f.read()).decode("utf-8")
            images.append({
                "type": "input_image",
                "image_url": f"data:image/jpeg;base64,{b64}",
            })
    
    prompt = """
You are classifying a short-form vertical video.

Return ONLY a JSON object with:
- text_density (0.0 to 1.0)
- scene_change_score (0.0 to 1.0)
- face_ratio (0.0 to 1.0)
- has_large_face (true/false)

Guidelines:
- text_density = how much text is on screen across frames
- scene_change_score = how often scenes change
- face_ratio = how dominant a face is across frames
- has_large_face = is a face taking up large portion of frame
"""
    response = requests.post(
        "https://api.openai.com/v1/responses",
        headers={
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": "gpt-4.1-mini",
            "temperature": 0,
            "input": [
                {
                    "role": "user",
                    "content": [{"type": "input_text", "text": prompt}] + images,
                }
            ],
        },
        timeout=60,
    )

    if not response.ok:
        raise RuntimeError(
            f"OpenAI Responses API error {response.status_code}: {(response.text or '').strip()}"
        )
    
    data = response.json()
    text = data.get("output_text") or ""
    if not text:
        try:
            output_items = data.get("output") or []
            for item in output_items:
                if item.get("type") != "message":
                    continue
                for content_item in item.get("content") or []:
                    if content_item.get("type") == "output_text":
                        text = (content_item.get("text") or "").strip()
                        if text:
                            break
                if text:
                    break
        except Exception:
            text = ""
    if not text:
        raise RuntimeError(f"Invalid AI response payload: {json.dumps(data)[:2000]}")
    text = text.strip()
    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()
    try:
        return json.loads(text)
    except Exception:
        raise RuntimeError(f"Invalid AI response text: {text}")


@app.get("/")
def serve_index():
    """
    Serve the frontend HTML from /frontend/index.html
    """
    if INDEX_HTML.exists():
        return FileResponse(INDEX_HTML)
    return JSONResponse({"error": "frontend/index.html not found"}, status_code=404)

@app.get("/api/unfurl")
def api_unfurl(url: str = ""):
    result = unfurl_url(url)
    if not result.get("ok"):
        return JSONResponse(result, status_code=400)
    return result


# Proxy endpoint for streaming images (thumbnails)
@app.get("/api/proxy")
def api_proxy(url: str = ""):
    """Proxy remote images so the browser can display thumbnails reliably.

    Many OG images (especially from IG/TikTok/X CDNs) can fail due to hotlinking rules
    or require specific headers. This endpoint fetches the image server-side and streams
    it back from our own origin.
    """
    src = (url or "").strip()
    if not src.startswith("http"):
        return JSONResponse({"ok": False, "error": "Invalid URL"}, status_code=400)

    # Simple allowlist: only proxy common image types / CDNs.
    # (You can loosen this later if needed.)
    try:
        parsed = urlparse(src)
        host = (parsed.netloc or "").lower()
        if not host:
            return JSONResponse({"ok": False, "error": "Invalid host"}, status_code=400)
    except Exception:
        return JSONResponse({"ok": False, "error": "Invalid URL"}, status_code=400)

    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0 Safari/537.36",
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        # Some CDNs are picky about Referer.
        "Referer": "https://www.instagram.com/",
        "Accept-Language": "en-US,en;q=0.9",
    }

    try:
        r = requests.get(src, headers=headers, stream=True, timeout=20, allow_redirects=True)
        r.raise_for_status()
    except Exception as e:
        return JSONResponse({"ok": False, "error": str(e)}, status_code=400)

    content_type = (r.headers.get("Content-Type") or "image/jpeg").split(";")[0].strip()

    # Ensure compressed responses are decoded properly.
    try:
        r.raw.decode_content = True
    except Exception:
        pass

    def iter_bytes():
        try:
            for chunk in r.iter_content(chunk_size=8192):
                if chunk:
                    yield chunk
        finally:
            try:
                r.close()
            except Exception:
                pass

    # Stream the bytes back. Add caching so the browser doesn't refetch constantly.
    return StreamingResponse(
        iter_bytes(),
        media_type=content_type,
        headers={
            "Cache-Control": "public, max-age=86400",
        },
    )
