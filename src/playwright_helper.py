from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import os

import re


def parse_instagram_metric(value: str) -> int:
    """
    Convert Instagram-style metric strings into integers.
    Examples: "1,234 likes" -> 1234, "4.5K views" -> 4500, "2M" -> 2000000.
    """
    value = (value or "").strip().lower()
    if not value:
        return 0

    value = value.replace(",", "")
    match = re.search(r"(\d+(?:\.\d+)?)\s*([kmb])?", value)
    if not match:
        return 0

    number = float(match.group(1))
    suffix = match.group(2) or ""

    if suffix == "k":
        number *= 1_000
    elif suffix == "m":
        number *= 1_000_000
    elif suffix == "b":
        number *= 1_000_000_000

    return int(number)


def extract_metric_from_text(text: str, labels: list[str]) -> int:
    """
    Search visible page text for a metric near one of the provided labels.
    """
    text = (text or "").replace("\n", " ").strip()
    if not text:
        return 0

    for label in labels:
        pattern = rf"(\d[\d,]*(?:\.\d+)?\s*[KkMmBb]?)\s+{re.escape(label)}"
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            return parse_instagram_metric(match.group(1))

    return 0


# --- NEW HELPERS ---

def instagram_cookie_list() -> list[dict]:
    cookies = [
        {"name": "sessionid", "value": os.getenv("INSTAGRAM_SESSIONID", ""), "domain": ".instagram.com", "path": "/"},
        {"name": "csrftoken", "value": os.getenv("INSTAGRAM_CSRFTOKEN", ""), "domain": ".instagram.com", "path": "/"},
        {"name": "ds_user_id", "value": os.getenv("INSTAGRAM_DS_USER_ID", ""), "domain": ".instagram.com", "path": "/"},
        {"name": "mid", "value": os.getenv("INSTAGRAM_MID", ""), "domain": ".instagram.com", "path": "/"},
        {"name": "ig_did", "value": os.getenv("INSTAGRAM_IG_DID", ""), "domain": ".instagram.com", "path": "/"},
        {"name": "rur", "value": os.getenv("INSTAGRAM_RUR", ""), "domain": ".instagram.com", "path": "/"},
    ]
    return [c for c in cookies if c["value"]]


def extract_metrics_from_page_source(source: str) -> dict:
    """
    Instagram often stores engagement metrics in embedded JSON instead of clean DOM text.
    This extracts common JSON metric fields from the raw page source.
    """
    source = source or ""

    patterns = {
        "like_count": [
            r'"edge_media_preview_like"\s*:\s*\{\s*"count"\s*:\s*(\d+)',
            r'"edge_liked_by"\s*:\s*\{\s*"count"\s*:\s*(\d+)',
            r'"like_count"\s*:\s*(\d+)',
            r'"likeCount"\s*:\s*(\d+)',
        ],
        "comment_count": [
            r'"edge_media_to_parent_comment"\s*:\s*\{\s*"count"\s*:\s*(\d+)',
            r'"edge_media_to_comment"\s*:\s*\{\s*"count"\s*:\s*(\d+)',
            r'"comment_count"\s*:\s*(\d+)',
            r'"commentCount"\s*:\s*(\d+)',
        ],
        "view_count": [
            r'"video_view_count"\s*:\s*(\d+)',
            r'"video_play_count"\s*:\s*(\d+)',
            r'"play_count"\s*:\s*(\d+)',
            r'"view_count"\s*:\s*(\d+)',
            r'"viewCount"\s*:\s*(\d+)',
        ],
    }

    metrics = {
        "like_count": 0,
        "comment_count": 0,
        "view_count": 0,
    }

    for key, regexes in patterns.items():
        for pattern in regexes:
            match = re.search(pattern, source, flags=re.IGNORECASE)
            if match:
                try:
                    metrics[key] = int(match.group(1))
                    break
                except Exception:
                    continue

    return metrics


# --- NEW PROFILE METRICS HELPERS ---

def extract_profile_metrics_from_page_source(source: str) -> dict:
    """
    Extract profile-level metrics from Instagram embedded page source.
    """
    source = source or ""

    def first_int(patterns: list[str]) -> int:
        for pattern in patterns:
            match = re.search(pattern, source, flags=re.IGNORECASE)
            if match:
                try:
                    return int(match.group(1))
                except Exception:
                    continue
        return 0

    def first_text(patterns: list[str]) -> str:
        for pattern in patterns:
            match = re.search(pattern, source, flags=re.IGNORECASE | re.DOTALL)
            if match:
                value = match.group(1) or ""
                value = value.replace("\\u0026", "&")
                value = value.replace("\\/", "/")
                value = value.replace("\\n", " ")
                value = re.sub(r"\\s+", " ", value).strip()
                return value
        return ""

    return {
        "profile_name": first_text([
            r'"full_name"\s*:\s*"([^"]*)"',
            r'"fullName"\s*:\s*"([^"]*)"',
        ]),
        "bio": first_text([
            r'"biography"\s*:\s*"([^"]*)"',
            r'"bio"\s*:\s*"([^"]*)"',
        ]),
        "profile_pic_url": first_text([
            r'"profile_pic_url_hd"\s*:\s*"([^"]*)"',
            r'"profile_pic_url"\s*:\s*"([^"]*)"',
            r'"profilePicUrl"\s*:\s*"([^"]*)"',
        ]),
        "follower_count": first_int([
            r'"edge_followed_by"\s*:\s*\{\s*"count"\s*:\s*(\d+)',
            r'"follower_count"\s*:\s*(\d+)',
            r'"followerCount"\s*:\s*(\d+)',
        ]),
        "following_count": first_int([
            r'"edge_follow"\s*:\s*\{\s*"count"\s*:\s*(\d+)',
            r'"following_count"\s*:\s*(\d+)',
            r'"followingCount"\s*:\s*(\d+)',
        ]),
        "post_count": first_int([
            r'"edge_owner_to_timeline_media"\s*:\s*\{\s*"count"\s*:\s*(\d+)',
            r'"media_count"\s*:\s*(\d+)',
            r'"mediaCount"\s*:\s*(\d+)',
        ]),
    }


# --- PROFILE META DESCRIPTION PARSER ---

def parse_profile_meta_description(description: str, handle: str = "") -> dict:
    """
    Parse Instagram profile meta descriptions like:
    "12K Followers, 500 Following, 123 Posts - Name (@handle) on Instagram: Bio..."
    """
    description = (description or "").strip()
    handle = (handle or "").strip().lstrip("@").lower()

    result = {
        "profile_name": "",
        "bio": "",
        "follower_count": 0,
        "following_count": 0,
        "post_count": 0,
    }

    if not description:
        return result

    # Only trust the description if it clearly belongs to the requested handle.
    # Instagram can return metadata for the logged-in viewer instead of the target profile.
    if handle and f"@{handle}" not in description.lower():
        return result

    result["follower_count"] = extract_metric_from_text(description, ["followers", "follower"])
    result["following_count"] = extract_metric_from_text(description, ["following"])
    result["post_count"] = extract_metric_from_text(description, ["posts", "post"])

    name_match = re.search(
        r"(?:followers|following|posts)\s*-\s*(.*?)\s*\(@",
        description,
        flags=re.IGNORECASE,
    )
    if name_match:
        result["profile_name"] = re.sub(r"\s+", " ", name_match.group(1)).strip()

    bio_match = re.search(r"on Instagram:\s*(.*)$", description, flags=re.IGNORECASE | re.DOTALL)
    if bio_match:
        result["bio"] = re.sub(r"\s+", " ", bio_match.group(1)).strip()

    return result



def fetch_instagram_profile_metrics_playwright(handle: str) -> dict:
    handle = (handle or "").strip().lstrip("@")
    if not handle:
        return {
            "profile_name": "",
            "bio": "",
            "profile_pic_url": "",
            "follower_count": 0,
            "following_count": 0,
            "post_count": 0,
        }

    url = f"https://www.instagram.com/{handle}/"

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1280, "height": 1600})
            context.add_cookies(instagram_cookie_list())
            page = context.new_page()
            page.goto(url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(2500)

            try:
                page_source = page.content()
            except Exception:
                page_source = ""

            metrics = extract_profile_metrics_from_page_source(page_source)
            # Instagram page source can include the logged-in viewer's profile data.
            # Do not trust profile identity fields from raw page source unless we
            # can verify them from handle-matching meta description below.
            metrics["profile_name"] = ""
            metrics["bio"] = ""
            metrics["profile_pic_url"] = ""

            try:
                og_title = (
                    page.locator("meta[property='og:title']").first.get_attribute("content")
                    or ""
                ).strip()
            except Exception:
                og_title = ""

            try:
                description = (
                    page.locator("meta[name='description']").first.get_attribute("content")
                    or ""
                ).strip()
            except Exception:
                description = ""

            desc_metrics = parse_profile_meta_description(description, handle)

            if desc_metrics.get("follower_count") and not metrics.get("follower_count"):
                metrics["follower_count"] = desc_metrics["follower_count"]
            if desc_metrics.get("following_count") and not metrics.get("following_count"):
                metrics["following_count"] = desc_metrics["following_count"]
            if desc_metrics.get("post_count") and not metrics.get("post_count"):
                metrics["post_count"] = desc_metrics["post_count"]

            if not metrics.get("profile_name") and desc_metrics.get("profile_name"):
                metrics["profile_name"] = desc_metrics["profile_name"]

            if not metrics.get("bio") and desc_metrics.get("bio"):
                metrics["bio"] = desc_metrics["bio"]

            # Only trust og:title if it clearly belongs to the requested handle.
            # Otherwise Instagram may return the logged-in user's title.
            if not metrics.get("profile_name") and f"@{handle.lower()}" in og_title.lower():
                metrics["profile_name"] = og_title.split("(@")[0].strip()

            context.close()
            browser.close()

            profile_name = metrics.get("profile_name") or ""
            bio = metrics.get("bio") or ""
            profile_pic_url = metrics.get("profile_pic_url") or ""
            follower_count = int(metrics.get("follower_count") or 0)
            following_count = int(metrics.get("following_count") or 0)
            post_count = int(metrics.get("post_count") or 0)

            if profile_name == "Jerry J. Goldman":
                profile_name = ""

            return {
                "profile_name": profile_name,
                "bio": bio,
                "profile_pic_url": profile_pic_url,
                "follower_count": follower_count,
                "following_count": following_count,
                "post_count": post_count,
            }

    except Exception:
        return {
            "profile_name": "",
            "bio": "",
            "profile_pic_url": "",
            "follower_count": 0,
            "following_count": 0,
            "post_count": 0,
        }

def fetch_instagram_posts_playwright(handle: str, max_posts: int = 12):
    handle = (handle or "").strip().lstrip("@") 
    if not handle:
        return []
    
    url = f"https://www.instagram.com/{handle}/"
    posts = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        context.add_cookies(instagram_cookie_list())
        page = context.new_page()

        page.goto(url, wait_until="domcontentloaded", timeout=60000)
        page.wait_for_timeout(3000)

        try:
            page.wait_for_selector('a[href*="/p/"], a[href*="/reel/"]', state="attached", timeout=10000)
        except PlaywrightTimeoutError:
            print(f"Playwright: no post links appeared for @{handle} within timeout")

        page.mouse.wheel(0, 5000)
        page.wait_for_timeout(2000)

        links = page.query_selector_all('a[href*="/p/"], a[href*="/reel/"]')
        print("Total links found:", len(links))
        for link in links:
            href = (link.get_attribute("href") or "").strip()
            if not href:
                continue
            if href.startswith("http"):
                full_url = href
            else:
                full_url = f"https://www.instagram.com{href}"
            expected_prefix = f"https://www.instagram.com/{handle}/"
            if not full_url.startswith(expected_prefix):
                continue
            if not "/p/" in full_url and "/reel/" not in full_url:
                continue
            if full_url not in posts:
                posts.append(full_url)
            if len(posts) >= max_posts:
                break
        
        browser.close()
    
    return posts

def fetch_instagram_post_metadata_playwright(post_url: str) -> dict:
    post_url = (post_url or "").strip()
    if not post_url:
        return {
            "title": "",
            "preview_url": "",
            "like_count": 0,
            "comment_count": 0,
            "view_count": 0,
        }
    
    from playwright.sync_api import sync_playwright

    try:
        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            context = browser.new_context(viewport={"width": 1280, "height": 1600})
            context.add_cookies(instagram_cookie_list())
            page = context.new_page()
            page.goto(post_url, wait_until="domcontentloaded", timeout=30000)
            page.wait_for_timeout(2500)

            title = ""
            preview_url = ""
            like_count = 0
            comment_count = 0
            view_count = 0

            try:
                title = (
                    page.locator("meta[property='og:title']").first.get_attribute("content")
                    or ""
                ).strip()
            except Exception:
                title = ""
            try:
                preview_url = (
                    page.locator("meta[property='og:image']").first.get_attribute("content")
                    or ""
                ).strip()
            except Exception:
                preview_url = ""
            if not preview_url:
                img_selectors = [
                    "article img",
                    "img[decoding='auto']",
                    "img",
                ]
                for selector in img_selectors:
                    try:
                        img = page.locator(selector).first
                        if img.count() > 0:
                            src = (img.get_attribute("src") or "").strip()
                            if src.startswith("http"):
                                preview_url = src
                                break
                    except Exception:
                        continue

            try:
                visible_text = page.locator("body").inner_text(timeout=5000)
            except Exception:
                visible_text = ""

            try:
                og_description = (
                    page.locator("meta[property='og:description']").first.get_attribute("content")
                    or ""
                ).strip()
            except Exception:
                og_description = ""

            try:
                page_source = page.content()
            except Exception:
                page_source = ""

            combined_text = " ".join([title, og_description, visible_text])
            source_metrics = extract_metrics_from_page_source(page_source)

            like_count = source_metrics.get("like_count") or extract_metric_from_text(
                combined_text,
                labels=["likes", "like"],
            )
            comment_count = source_metrics.get("comment_count") or extract_metric_from_text(
                combined_text,
                labels=["comments", "comment"],
            )
            view_count = source_metrics.get("view_count") or extract_metric_from_text(
                combined_text,
                labels=["views", "view", "plays", "play"],
            )

            context.close()
            browser.close()
            return {
                "title": title,
                "preview_url": preview_url,
                "like_count": like_count,
                "comment_count": comment_count,
                "view_count": view_count,
            }
    except Exception:
        return {
            "title": "",
            "preview_url": "",
            "like_count": 0,
            "comment_count": 0,
            "view_count": 0,
        }