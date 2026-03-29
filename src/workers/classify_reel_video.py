import traceback

from ..app import (
    download_instagram_reel_video,
    extract_frames_from_video,
    analyze_frames_with_ai,
    classify_reel_from_visual_signals,
    update_post_classification_by_url,
)

from ..jobs import (
    JOB_CLASSIFY_REEL_VIDEO,
    get_crawl_job,
    increment_crawl_job_retry,
    publish_rabbitmq_job,
    update_crawl_job_status,
    utc_now_iso,
)

def handle_classify_reel_video_job(payload: dict):
    """
    Worker handler for classifying a single Instagram reel.
    """
    job_id = payload.get("job_id") or 0
    post_url = (payload.get("post_url") or "").strip()
    fps = float(payload.get("fps") or 1.0)
    max_retries = 2

    if not post_url:
        print("Missing post_url in payload")
        return
    
    print(f"Processing reel: {post_url}")

    if job_id:
        update_crawl_job_status(
            job_id=job_id,
            status="processing",
            started_at=utc_now_iso(),
        )

    try:
        video_path = download_instagram_reel_video(post_url)
        frame_paths = extract_frames_from_video(video_path, fps=fps)
        frame_analysis = analyze_frames_with_ai(frame_paths)
        classified_post_type, classifier_confidence, classifier_version = classify_reel_from_visual_signals(
            text_density=frame_analysis.get("text_density", 0.0),
            scene_change_score=frame_analysis.get("scene_change_score", 0.0),
            face_ratio=frame_analysis.get("face_ratio", 0.0),
            has_large_face=frame_analysis.get("has_large_face", False),
            sampled_frames=len(frame_paths),
        )

        if classified_post_type:
            update_post_classification_by_url(
                post_url=post_url,
                classified_post_type=classified_post_type,
                classifier_confidence=classifier_confidence,
                classifier_version=f"{classifier_version}:openai_frames_v1",
            )
        
        if job_id:
            update_crawl_job_status(
                job_id=job_id,
                status="completed",
                finished_at=utc_now_iso(),
            )
        
        print(f"Done: {post_url} -> {classified_post_type}")
    
    except Exception as e:
        retried = False

        if job_id:
            current_job = get_crawl_job(job_id)
            retry_count = int(current_job.get("retry_count") or 0)
            if retry_count < max_retries:
                updated_job = increment_crawl_job_retry(job_id)
                update_crawl_job_status(
                    job_id=job_id,
                    status="queued",
                    error_message=str(e),
                    finished_at=utc_now_iso(),
                )
                retry_payload = dict(payload)
                retry_payload["job_id"] = job_id
                publish_rabbitmq_job(
                    job_type=JOB_CLASSIFY_REEL_VIDEO,
                    target=post_url,
                    payload=retry_payload,
                )
                retried = True
                print(
                    f"Retrying job {job_id} for {post_url} "
                    f"({int(updated_job.get('retry_count') or 0)}/{max_retries})"
                )
            else:
                update_crawl_job_status(
                    job_id=job_id,
                    status="failed",
                    error_message=str(e),
                    finished_at=utc_now_iso(),
                )
        print(f"Worker failed for {post_url}")
        print(e)
        if retried:
            print("Job was requeued for retry.")
        traceback.print_exc()
