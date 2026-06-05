import random
import time

from src.jobs import (
    get_db,
    queue_expand_niche_job_if_needed,
    queue_missing_reel_classification_jobs,
)

# Core niches to constantly expand and refresh
SCHEDULED_NICHES = [
    # Ecommerce / Business
    "amazon fba",
    "amazon wholesale",
    "shopify",
    "dropshipping",
    "ecommerce",
    "email marketing",
    "copywriting",
    "sales",
    "entrepreneurship",
    "business",
    "online business",
    "digital marketing",
    "personal branding",
    "social media marketing",
    "content creation",
    "ai business",
    "real estate",

    # Fitness / Health
    "fitness",
    "bodybuilding",
    "fat loss",
    "weight loss",
    "running",
    "crossfit",
    "mens health",
    "womens health",
    "functional medicine",
    "nutrition",
    "biohacking",
    "wellness",

    # Self Improvement
    "self improvement",
    "discipline",
    "motivation",
    "mindset",
    "productivity",
    "psychology",
    "mental health",
    "confidence",

    # Dating / Relationships
    "dating",
    "relationships",
    "marriage",
    "masculinity",
    "femininity",
    "attachment styles",

    # Finance
    "investing",
    "stock market",
    "crypto",
    "personal finance",
    "side hustles",

    # Creator / Education
    "youtube growth",
    "instagram growth",
    "podcasting",
    "online coaching",
    "course creators",

    # Tech
    "artificial intelligence",
    "software engineering",
    "coding",
    "programming",
    "web development",
    "saas",
    "ai",

    # Lifestyle
    "luxury lifestyle",
    "travel",
    "fashion",
    "mens fashion",
    "beauty",
    "skincare",

    # Misc Viral Niches
    "memes",
    "storytelling",
    "life advice",
    "career advice",
]

STYLES = [
    "single-clip",
    "multi-clip",
    "talking-head",
    "carousel",
]

REFRESH_INTERVAL_SECONDS = 60 * 30  # every 30 minutes
MAX_JOBS_PER_CYCLE = 12
MAX_PENDING_JOBS = 75
CLASSIFICATION_JOBS_PER_CYCLE = 0
MIN_DELAY_BETWEEN_JOBS = 8
MAX_DELAY_BETWEEN_JOBS = 20

def get_pending_job_count() -> int:
    with get_db() as conn:
        row = conn.execute(
            """
            SELECT COUNT(*) AS count
            FROM crawl_jobs
            WHERE status IN ('queued', 'processing', 'running')
            """
        ).fetchone()
    
    return int(row["count"] or 0)

def queue_classification_jobs_for_cycle() -> None:
    try:
        print(
            f"[SCHEDULER] Queueing up to "
            f"{CLASSIFICATION_JOBS_PER_CYCLE} reel classification jobs..."
        )

        classification_result = queue_missing_reel_classification_jobs(
            limit=CLASSIFICATION_JOBS_PER_CYCLE,
            fps=1.0,
        )

        print(
            f"[SCHEDULER] Classification queue result: "
            f"{classification_result}"
        )

    except Exception as e:
        print(f"[SCHEDULER CLASSIFICATION ERROR] {e}")

def run_scheduler():
    print("Scheduler started...")

    while True:
        pending_jobs = get_pending_job_count()

        print(f"[SCHEDULER] Pending jobs: {pending_jobs}")

        queue_classification_jobs_for_cycle()

        if pending_jobs >= MAX_PENDING_JOBS:
            print(
                f"[SCHEDULER] Backlog too high ({pending_jobs} pending). "
                f"Sleeping for {REFRESH_INTERVAL_SECONDS} seconds..."
            )

            time.sleep(REFRESH_INTERVAL_SECONDS)
            continue

        queued_count = 0

        shuffled_niches = list(SCHEDULED_NICHES)
        random.shuffle(shuffled_niches)

        shuffled_styles = list(STYLES)
        random.shuffle(shuffled_styles)

        for niche in shuffled_niches:
            for style in shuffled_styles:
                if queued_count >= MAX_JOBS_PER_CYCLE:
                    print(
                        f"[SCHEDULER] Reached cycle job limit ({MAX_JOBS_PER_CYCLE})"
                    )
                    break

                try:
                    print(f"[SCHEDULER] Queueing expansion: {niche} | {style}")

                    queue_expand_niche_job_if_needed(
                        platform="instagram",
                        niche=niche,
                        style=style,
                    )

                    queued_count += 1

                    sleep_seconds = random.randint(
                        MIN_DELAY_BETWEEN_JOBS,
                        MAX_DELAY_BETWEEN_JOBS,
                    )

                    print(
                        f"[SCHEDULER] Cooling down for {sleep_seconds} seconds..."
                    )

                    time.sleep(sleep_seconds)

                except Exception as e:
                    print(f"[SCHEDULER ERROR] {niche} | {style} -> {e}")
        
        print(
            f"[SCHEDULER] Sleeping for {REFRESH_INTERVAL_SECONDS} seconds..."
        )

        time.sleep(REFRESH_INTERVAL_SECONDS)

if __name__ == "__main__":
    run_scheduler()