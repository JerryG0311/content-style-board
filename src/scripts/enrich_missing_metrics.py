from src.jobs import queue_enrich_missing_post_metrics_job_if_needed

if __name__ == "__main__":
    result = queue_enrich_missing_post_metrics_job_if_needed(
        platform="instagram",
        limit=10,
        cooldown_minutes=60,
    )

    print(result)