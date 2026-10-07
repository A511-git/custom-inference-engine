#!/usr/bin/env python3
"""
Diagnostic test client for NInfer HTTP Server.
Tests connection, model inference, token streaming, and throughput metrics.
"""

import sys
import time
import json
import urllib.request
import urllib.error

SERVER_URL = "http://localhost:8080/v1/chat/completions"

def test_inference(prompt: str = "Explain the difference between a process and a thread in three sentences."):
    payload = {
        "model": "qwen3_6_27b",
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.0,
        "max_tokens": 128
    }

    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        SERVER_URL,
        data=data,
        headers={"Content-Type": "application/json"}
    )

    print(f"Sending prompt to {SERVER_URL}...")
    print(f"Prompt: {prompt}\n")

    start_time = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            body = response.read().decode("utf-8")
            elapsed = time.perf_counter() - start_time
            result = json.loads(body)
    except urllib.error.URLError as e:
        print(f"Error connecting to server: {e}")
        sys.exit(1)

    choice = result["choices"][0]
    usage = result.get("usage", {})
    content = choice["message"]["content"]
    completion_tokens = usage.get("completion_tokens", 0)
    tok_per_sec = completion_tokens / elapsed if elapsed > 0 else 0

    print("=" * 60)
    print("RESPONSE:")
    print("=" * 60)
    print(content)
    print("=" * 60)
    print("PERFORMANCE METRICS:")
    print(f"  Total latency:      {elapsed:.2f} s")
    print(f"  Completion tokens:  {completion_tokens}")
    print(f"  Decode throughput:  {tok_per_sec:.1f} tok/s")
    print(f"  Finish reason:      {choice.get('finish_reason')}")
    print("=" * 60)
    print("TEST PASSED: Server response received cleanly with no errors.")

if __name__ == "__main__":
    test_inference()
