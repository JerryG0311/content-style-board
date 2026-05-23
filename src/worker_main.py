import os
import json
import pika

from .workers.classify_reel_video import handle_classify_reel_video_job
from .workers.crawl_instagram_account import handle_crawl_instagram_account_job
from .expansion_service import expand_niche_if_needed
from .jobs import (
    JOB_CLASSIFY_REEL_VIDEO,
    JOB_CRAWL_INSTAGRAM_ACCOUNT,
    JOB_EXPAND_NICHE,
)

RABBITMQ_URL = os.getenv(
    "RABBITMQ_URL",
    "amqp://guest:guest@localhost:5672/%2F",
)


def main():
    params = pika.URLParameters(RABBITMQ_URL)
    connection = pika.BlockingConnection(params)
    channel = connection.channel()

    channel.queue_declare(queue="content_jobs", durable=True)

    print("Worker started. Waiting for jobs...")

    def callback(ch, method, properties, body):
        try:
            job = json.loads(body)
            job_type = job.get("job_type")
            payload = job.get("payload", {})

            print(f"\nReceived job: {job_type}")

            if job_type == JOB_CLASSIFY_REEL_VIDEO:
                handle_classify_reel_video_job(payload)

            elif job_type == JOB_CRAWL_INSTAGRAM_ACCOUNT:
                handle_crawl_instagram_account_job(payload)

            elif job_type == JOB_EXPAND_NICHE:
                result = expand_niche_if_needed(
                    platform=payload.get("platform", "instagram"),
                    style=payload.get("style", "carousel"),
                    niche=payload.get("niche", ""),
                    limit=int(payload.get("crawl_accounts", 10)),
                )
                print(f"Expand niche queued crawl work: {result}")

            else:
                print(f"Unknown job type: {job_type}")

            ch.basic_ack(delivery_tag=method.delivery_tag)

        except Exception as e:
            print(f"Worker error: {e}")
            try:
                ch.basic_ack(delivery_tag=method.delivery_tag)
            except Exception as ack_error:
                print(f"Ack failed after worker error: {ack_error}")

    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(
        queue="content_jobs",
        on_message_callback=callback,
    )
    channel.start_consuming()


if __name__ == "__main__":
    main()