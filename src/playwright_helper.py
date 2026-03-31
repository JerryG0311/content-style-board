from playwright.sync_api import sync_playwright, TimeoutError as PlaywrightTimeoutError
import os

def fetch_instagram_posts_playwright(handle: str, max_posts: int = 12):
    handle = (handle or "").strip().lstrip("@")
    if not handle:
        return []
    
    url = f"https://www.instagram.com/{handle}/"
    posts = []

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        cookies = [
            {"name": "sessionid", "value": os.getenv("INSTAGRAM_SESSIONID", ""), "domain": ".instagram.com", "path": "/"},
            {"name": "csrftoken", "value": os.getenv("INSTAGRAM_CSRFTOKEN", ""), "domain": ".instagram.com", "path": "/"},
            {"name": "ds_user_id", "value": os.getenv("INSTAGRAM_DS_USER_ID", ""), "domain": ".instagram.com", "path": "/"},
            {"name": "mid", "value": os.getenv("INSTAGRAM_MID", ""), "domain": ".instagram.com", "path": "/"},
            {"name": "ig_did", "value": os.getenv("INSTAGRAM_IG_DID", ""), "domain": ".instagram.com", "path": "/"},
            {"name": "rur", "value": os.getenv("INSTAGRAM_RUR", ""), "domain": ".instagram.com", "path": "/"},
        ]
        cookies = [c for c in cookies if c["value"]]
        context.add_cookies(cookies)
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