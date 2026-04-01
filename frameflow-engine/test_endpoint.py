"""
FrameFlow Engine — Endpoint Test Script

Usage:
  RUNPOD_API_KEY=rpa_xxx python test_endpoint.py

Or set the env var in your shell first.
"""

import os
import sys
import json
import time
import requests

# ── Config ────────────────────────────────────────────────────
ENDPOINT_ID = "rkrk0w6kgy960r"
BASE_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"
API_KEY = os.environ.get("RUNPOD_API_KEY", "")

if not API_KEY:
    print("ERROR: Set RUNPOD_API_KEY environment variable")
    print("  export RUNPOD_API_KEY=rpa_xxx")
    sys.exit(1)

HEADERS = {
    "Authorization": f"Bearer {API_KEY}",
    "Content-Type": "application/json",
}

# ── Test payload ──────────────────────────────────────────────
# Public domain test image (Unsplash sneaker photo)
TEST_PAYLOAD = {
    "input": {
        "first_frame_url": "https://images.unsplash.com/photo-1542291026-7eec264c27ff?w=1280&q=80",
        "prompt": "sneaker rotating slowly on white background",
        "quality": "fast",
    }
}


def submit_job():
    print(f"Submitting job to {BASE_URL}/run ...")
    print(f"Payload: {json.dumps(TEST_PAYLOAD, indent=2)}")
    print()

    r = requests.post(f"{BASE_URL}/run", headers=HEADERS, json=TEST_PAYLOAD, timeout=30)
    r.raise_for_status()
    data = r.json()

    job_id = data.get("id")
    status = data.get("status")
    print(f"Job submitted: id={job_id} status={status}")
    return job_id


def poll_status(job_id, timeout=600, interval=5):
    print(f"\nPolling status every {interval}s (timeout {timeout}s)...")
    start = time.time()

    while time.time() - start < timeout:
        elapsed = time.time() - start
        r = requests.get(f"{BASE_URL}/status/{job_id}", headers=HEADERS, timeout=15)
        data = r.json()
        status = data.get("status", "UNKNOWN")

        print(f"  [{elapsed:5.0f}s] status={status}")

        if status == "COMPLETED":
            return data
        elif status == "FAILED":
            print("\n═══ JOB FAILED ═══")
            print(json.dumps(data, indent=2, default=str))
            return data
        elif status in ("CANCELLED", "TIMED_OUT"):
            print(f"\nJob {status}")
            return data

        time.sleep(interval)

    print(f"\nLocal polling timed out after {timeout}s")
    return None


def print_result(data):
    if not data:
        print("No data received")
        return

    output = data.get("output", {})
    status = output.get("status", data.get("status"))

    print(f"\n{'═' * 60}")
    print(f"STATUS: {status}")
    print(f"{'═' * 60}")

    if isinstance(output, dict):
        # Print everything except large base64 fields
        for key, value in output.items():
            if "base64" in key:
                if value:
                    size_kb = len(value) * 3 / 4 / 1024
                    print(f"  {key}: [{size_kb:.0f} KB base64 data]")
                else:
                    print(f"  {key}: null")
            elif key == "params_used":
                print(f"  {key}:")
                for pk, pv in value.items():
                    print(f"    {pk}: {pv}")
            elif key == "overcapture":
                print(f"  {key}:")
                for ok, ov in value.items():
                    print(f"    {ok}: {ov}")
            elif key == "content_detected":
                print(f"  {key}: {json.dumps(value)}")
            else:
                print(f"  {key}: {value}")
    else:
        print(f"  output: {output}")

    # Execution time from RunPod
    exec_time = data.get("executionTime")
    if exec_time:
        print(f"\n  RunPod executionTime: {exec_time}ms ({exec_time/1000:.1f}s)")


def main():
    print("FrameFlow Engine — Endpoint Test")
    print(f"Endpoint: {BASE_URL}")
    print()

    job_id = submit_job()
    if not job_id:
        print("Failed to submit job")
        sys.exit(1)

    result = poll_status(job_id)
    print_result(result)


if __name__ == "__main__":
    main()
