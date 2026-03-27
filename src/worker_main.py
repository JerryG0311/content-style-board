import os
import json
import pika

from .workers.classify_reel_video import handle_classify_reel_video_job

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

            if job_type == "classify_reel_video":
                handle_classify_reel_video_job(payload)
            else:
                print(f"Unknown job type: {job_type}")
            
            ch.basic_ack(delivery_tag=method.delivery_tag)
        except Exception as e:
            print(f"Worker error: {e}")

            ch.basic_ack(delivery_tag=method.delivery_tag)
    
    channel.basic_qos(prefetch_count=1)
    channel.basic_consume(
        queue="content_jobs",
        on_message_callback=callback,
    )
    channel.start_consuming()

if __name__ == "__main__":
    main()