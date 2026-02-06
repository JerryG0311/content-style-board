from dotenv import load_dotenv
import os
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pathlib import Path
from pydantic import BaseModel
from json import JSONDecodeError
import json
import requests
import hashlib
import time
import re

from fastapi import Response
from fastapi.responses import StreamingResponse
from urllib.parse import unquote

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

class BoardPlayload(BaseModel):
    board: list

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

# For local dev: allow the frontend to call the API easily
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # later we can tighten this up
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/api/health")
def health():
    return {"status": "ok"}

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

    # 1) Build a base query
    q = build_query(platform, style, niche=niche, seed=seed)

    # Brave can behave inconsistently for instagram.com queries depending on safesearch.
    # We'll try "moderate" first, then retry with "off" when needed.
    safe_primary = "moderate"
    safe_fallback = "off" if platform == "instagram" else None

    niche_hint = (niche or "").strip()
    seed_hint = (seed or "").strip()

    # Instagram is extra sensitive to query structure.
    # We'll try a small set of progressively broader candidate queries.
    candidate_queries = [q]

    if platform == "instagram":
        if style == "carousel":
            candidate_queries += [
                f"site:instagram.com/p/ {niche_hint} {seed_hint} -template -templates -canva -capcut -ugc -ads -advertising -marketing".strip(),
                f"site:instagram.com/p/ {niche_hint} -template -templates -canva -capcut -ugc -ads -advertising -marketing".strip(),
                f"instagram.com/p/ {niche_hint}".strip(),
            ]
        elif style in ("multi-clip", "single-clip"):
            extra = "part 1" if style == "multi-clip" else "text overlay"
            candidate_queries += [
                f"site:instagram.com/reel/ {niche_hint} {seed_hint}".strip(),
                f"site:instagram.com/reel/ {niche_hint} {extra}".strip(),
                f"instagram.com/reel/ {niche_hint} {extra}".strip(),
                "site:instagram.com/reel/ part 1",
                "instagram.com/reel/",
            ]

    # de-dupe while preserving order
    _seen_q = set()
    candidate_queries = [cq for cq in candidate_queries if cq and not (cq in _seen_q or _seen_q.add(cq))]

    dbg = {
        "app_file": __file__,
        "cwd": os.getcwd(),
        "q": q,
        "safesearch": safe_primary,
        "brave_total": 0,
        "attempts": [],
        "filtered_tutorial_or_ads": 0,
        "filtered_empty_url": 0,
        "filtered_platform_mismatch": 0,
        "kept": 0,
        "sample_titles": [],
        "rate_limited": False,
    }

    def _do_brave(query: str, ss: str):
        try:
            # Brave Web Search API: count max is 20. Using >20 causes 422.
            data = brave_web_search(query, count=20, safesearch=ss)
            web = data.get("web", {}) or {}
            results = web.get("results", []) or []
            dbg["attempts"].append({"q": query, "safesearch": ss, "count": len(results)})
            return results
        except requests.HTTPError as e:
            status = getattr(getattr(e, "response", None), "status_code", None)
            dbg["attempts"].append({
                "q": query,
                "safesearch": ss,
                "count": 0,
                "error": f"{status} {str(e)}".strip(),
            })
            if status == 429:
                dbg["rate_limited"] = True
                return None  # sentinel so caller can stop retrying
            return []
        except Exception as e:
            dbg["attempts"].append({
                "q": query,
                "safesearch": ss,
                "count": 0,
                "error": str(e),
            })
            return []

    brave_results = []
    used_q = q
    used_safe = safe_primary

    try:
        # Try candidate queries until we get results.
        for cq in candidate_queries:
            used_q = cq

            # 1) primary safesearch
            brave_results = _do_brave(cq, safe_primary)
            used_safe = safe_primary

            # If we were rate-limited, stop trying more queries.
            if brave_results is None:
                brave_results = []
                break

            # 2) instagram retry with safesearch=off
            if platform == "instagram" and len(brave_results) == 0 and safe_fallback:
                brave_results = _do_brave(cq, safe_fallback)
                used_safe = safe_fallback

                if brave_results is None:
                    brave_results = []
                    break

            if len(brave_results) > 0:
                break

    except Exception as exc:
        print("Brave search failed:", str(exc))
        brave_results = []

    # Update q + safesearch to what actually ran (useful for debugging)
    q = used_q
    dbg["q"] = q
    dbg["safesearch"] = used_safe
    dbg["brave_total"] = len(brave_results)

    # capture a few raw titles to sanity check what Brave returned
    for it in brave_results[:5]:
        dbg["sample_titles"].append(it.get("title") or it.get("url") or "")

    # 3) Normalize + filter
    normalized = []
    for item in brave_results:
        url = item.get("url") or ""
        title = item.get("title") or url
        description = item.get("description") or ""

        if looks_like_tutorial_or_ads(title):
            dbg["filtered_tutorial_or_ads"] += 1
            continue

        if not url:
            dbg["filtered_empty_url"] += 1
            continue

        if not url_matches_platform(url, platform, style):
            dbg["filtered_platform_mismatch"] += 1
            continue

        thumb = extract_thumbnail(item)
        if not thumb:
            thumb = best_effort_unfurl_image(url)

        normalized.append({
            "title": title,
            "url": url,
            "platform": platform,
            "tag": style,
            "description": description,
            "thumbnail": thumb,
        })

    # 4) Dedupe + truncate
    results = dedupe_and_truncate(normalized, limit=8)
    dbg["kept"] = len(results)

    # If requested, filter down to only the best matches (style match + niche match)
    if best:
        best_out = []
        for r in results:
            a = analyze_item(
                url=r.get("url", ""),
                title=r.get("title", ""),
                description=r.get("description", ""),
                platform=platform,
                tag=style,
                niche=niche,
                seed=seed,
            )
            style_match = (a.get("style") or "").lower().strip() == (style or "").lower().strip()
            niche_ok = (not (niche or "").strip()) or bool(a.get("niche_match"))
            if style_match and niche_ok:
                # carry analysis fields forward (optional, useful for UI/debug)
                r["niche_score"] = a.get("niche_score")
                r["niche_match"] = a.get("niche_match")
                r["confidence"] = a.get("confidence")
                r["is_tutorial_like"] = a.get("is_tutorial_like")
                best_out.append(r)

        results = dedupe_and_truncate(best_out, limit=8)
        dbg["kept"] = len(results)

    # 5) Fallback: if we got nothing, try a broader query (still platform-correct)

    if len(results) == 0:
        q2 = None

        if platform == "instagram" and style == "multi-clip":
            q2 = (
                'site:instagram.com (inurl:/reel/ OR inurl:/reels/) '
                '("part 1" OR "pt 1" OR "1/3" OR "2/3" OR "day 1" OR "day 2") '
                '-tutorial -how -guide -tips -template'
            )

        if platform == "instagram" and style == "carousel" and not niche_hint:
            # Only when niche is empty do we allow broad “carousel-ish” discovery.
            q2 = (
                'site:instagram.com/p/ '
                '(swipe OR "slide 1" OR carousel OR "1/10" OR "1/7" OR "1/8") '
                '-template -templates -canva -capcut -ugc -ads -advertising -marketing '
                '-"carousel template" -"viral carousel" -"carousel ideas" -"hook ideas" '
                '-"reel ideas" -"content strategy" -"social media strategy"'
            )

        if q2:
            try:
                # Brave Web Search API: count max is 20. Using >20 causes 422.
                data2 = brave_web_search(q2, count=20, safesearch=safe_primary)
                web2 = data2.get("web", {}) or {}
                brave_results2 = web2.get("results", []) or []
            except Exception as exc:
                print("Brave fallback failed:", str(exc))
                brave_results2 = []

            normalized2 = []
            for item in brave_results2:
                url = item.get("url") or ""
                title = item.get("title") or url
                description = item.get("description") or ""

                if not url:
                    continue
                if not url_matches_platform(url, platform, style):
                    continue
                if looks_like_tutorial_or_ads(title):
                    continue

                thumb = extract_thumbnail(item)
                if not thumb:
                    thumb = best_effort_unfurl_image(url)

                normalized2.append(
                    {
                        "title": title,
                        "url": url,
                        "platform": platform,
                        "tag": style,
                        "description": description,
                        "thumbnail": thumb,
                    }
                )

            # THIS WAS MISSING: actually use the fallback results
            results = dedupe_and_truncate(normalized2, limit=8)

            if best:
                best_out2 = []
                for r in results:
                    a = analyze_item(
                        url=r.get("url", ""),
                        title=r.get("title", ""),
                        description=r.get("description", ""),
                        platform=platform,
                        tag=style,
                        niche=niche,
                        seed=seed,
                    )
                    style_match = (a.get("style") or "").lower().strip() == (style or "").lower().strip()
                    niche_ok = (not (niche or "").strip()) or bool(a.get("niche_match"))
                    if style_match and niche_ok:
                        r["niche_score"] = a.get("niche_score")
                        r["niche_match"] = a.get("niche_match")
                        r["confidence"] = a.get("confidence")
                        r["is_tutorial_like"] = a.get("is_tutorial_like")
                        best_out2.append(r)

                results = dedupe_and_truncate(best_out2, limit=8)

            # optional: show which query we ended up using
            if len(results) > 0:
                q = q2

    resp = {"platform": platform, "style": style, "q": q, "results": results}
    if debug:
        resp["debug"] = dbg
    return resp

from typing import Optional, Dict, Any
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

def score_nich_relevance(niche: str, text: str) -> float:
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


def infer_style(platform: str, tag: str, fmt: str) -> Dict[str, Any]:
    platform = (platform or "").lower().strip()
    tag = (tag or "").lower().strip()

    if platform == "instagram":
        if fmt == "instagram_post":
            return {"style": "carousel", "confidence": 0.85}
        if fmt == "instagram_reel":
            if tag in ("multi-clip", "single-clip"):
                return {"style": tag, "confidence": 0.65}
            return {"style": "reel", "confidence": 0.60}

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

    combined = f"{title or ''}\n{description or ''}".strip()

    niche_score = score_nich_relevance(niche or "", combined)
    niche_match = (niche_score >= 0.45) if (niche or "").strip() else True

    tutorial_flag = looks_like_tutorial(combined)
    style_guess = infer_style(platform2, tag or "", fmt)

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
    combined = f"{title}\n{desc}".strip()
    niche_score = score_nich_relevance(payload.niche or "", combined)

    tutorial_flag = looks_like_tutorial(combined)
    style_guess = infer_style(platform, payload.tag or "", fmt)

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
