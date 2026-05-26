from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import httpx


ROOT = Path("/Users/gsdata/work/nlp_tool")
DATASET_PATH = ROOT / "bench_data" / "waimai_10k_full.csv"
OUTPUT_PATH = ROOT / "tmpdata" / "waimai_10k_full_user_flow.json"
BASE_URL = "http://127.0.0.1:8000"


def timed_request(client: httpx.Client, name: str, method: str, url: str, **kwargs: Any) -> dict[str, Any]:
    started = time.perf_counter()
    response = client.request(method, url, timeout=None, **kwargs)
    elapsed = round(time.perf_counter() - started, 3)
    response.raise_for_status()
    payload = response.json()
    return {
        "name": name,
        "elapsed_sec": elapsed,
        "data": payload,
    }


def poll_run_summary(client: httpx.Client, run_id: str, *, name: str, poll_interval: float = 2.0) -> dict[str, Any]:
    started = time.perf_counter()
    while True:
        response = client.get(f"{BASE_URL}/api/analyses/{run_id}/summary", timeout=None)
        response.raise_for_status()
        payload = response.json()
        if payload["status"] in {"completed", "failed"}:
            return {
                "name": name,
                "elapsed_sec": round(time.perf_counter() - started, 3),
                "data": payload,
            }
        time.sleep(poll_interval)


def build_label_schema_from_topics(topics: list[dict[str, Any]]) -> dict[str, Any]:
    labels = []
    profiles = []
    for topic in topics:
        name = str(topic.get("name", "")).strip()
        if not name:
            continue
        labels.append(name)
        profiles.append(
            {
                "name": name,
                "description": topic.get("summary", ""),
                "keywords": topic.get("keywords", [])[:8],
                "positive_examples": [
                    evidence.get("snippet", "")
                    for evidence in topic.get("evidences", [])[:2]
                    if evidence.get("snippet")
                ],
                "negative_examples": [],
                "source_topic_ids": [topic.get("topic_id", "")],
            }
        )
    return {
        "id": "derived_topics",
        "name": "主题衍生分类",
        "description": "根据主题发现结果自动生成的分类标签",
        "labels": labels,
        "profiles": profiles,
    }


def main() -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with httpx.Client() as client:
        health = timed_request(client, "health", "GET", f"{BASE_URL}/api/health")
        readiness = timed_request(client, "platform_readiness", "GET", f"{BASE_URL}/api/platform/readiness")

        with DATASET_PATH.open("rb") as file_obj:
            upload = timed_request(
                client,
                "upload_dataset",
                "POST",
                f"{BASE_URL}/api/datasets/upload",
                files={"file": (DATASET_PATH.name, file_obj, "text/csv")},
            )
        dataset_id = upload["data"]["dataset"]["id"]

        steps: list[dict[str, Any]] = [health, readiness, upload]
        steps.append(
            timed_request(
                client,
                "fetch_preview",
                "GET",
                f"{BASE_URL}/api/datasets/{dataset_id}",
                params={"limit": 20},
            )
        )
        steps.append(
            timed_request(
                client,
                "workspace_summary",
                "GET",
                f"{BASE_URL}/api/datasets/{dataset_id}/workspace/summary",
            )
        )

        workspace_sections = [
            "top_terms",
            "selected_terms",
            "match_rows",
            "binary_matrix",
            "frequency_matrix",
            "cooccurrence_edges",
        ]
        for section in workspace_sections:
            steps.append(
                timed_request(
                    client,
                    section,
                    "GET",
                    f"{BASE_URL}/api/datasets/{dataset_id}/workspace/sections/{section}",
                    params={"page": 1, "page_size": 50},
                )
            )

        explore_run = timed_request(
            client,
            "run_explore",
            "POST",
            f"{BASE_URL}/api/analyses/run",
            json={
                "dataset_id": dataset_id,
                "analysis_stage": "explore",
                "top_k_terms": 25,
                "topic_count": 8,
                "use_llm": False,
                "smart_topic_names": False,
                "write_exports": True,
                "export_xlsx": False,
            },
        )
        steps.append(explore_run)
        steps.append(poll_run_summary(client, explore_run["data"]["run"]["id"], name="explore_summary"))

        discover_run = timed_request(
            client,
            "run_discover",
            "POST",
            f"{BASE_URL}/api/analyses/run",
            json={
                "dataset_id": dataset_id,
                "analysis_stage": "discover",
                "top_k_terms": 25,
                "topic_count": 8,
                "use_llm": False,
                "smart_topic_names": False,
                "write_exports": True,
                "export_xlsx": False,
            },
        )
        steps.append(discover_run)
        discover_summary = poll_run_summary(client, discover_run["data"]["run"]["id"], name="discover_summary")
        steps.append(discover_summary)

        topics = discover_summary["data"]["previews"]["topics"]
        label_schema = build_label_schema_from_topics(topics)

        classify_run = timed_request(
            client,
            "run_classify",
            "POST",
            f"{BASE_URL}/api/analyses/run",
            json={
                "dataset_id": dataset_id,
                "analysis_stage": "classify",
                "top_k_terms": 25,
                "topic_count": 8,
                "use_llm": False,
                "smart_topic_names": False,
                "write_exports": True,
                "export_xlsx": False,
                "label_schema": label_schema,
            },
        )
        steps.append(classify_run)
        classify_summary = poll_run_summary(client, classify_run["data"]["run"]["id"], name="classify_summary")
        steps.append(classify_summary)
        steps.append(
            timed_request(
                client,
                "classification_section",
                "GET",
                f"{BASE_URL}/api/analyses/{classify_run['data']['run']['id']}/sections/classification",
                params={"page": 1, "page_size": 50},
            )
        )

        sentiment_run = timed_request(
            client,
            "run_sentiment",
            "POST",
            f"{BASE_URL}/api/analyses/run",
            json={
                "dataset_id": dataset_id,
                "analysis_stage": "sentiment",
                "top_k_terms": 25,
                "topic_count": 8,
                "use_llm": False,
                "smart_topic_names": False,
                "write_exports": True,
                "export_xlsx": False,
            },
        )
        steps.append(sentiment_run)
        sentiment_summary = poll_run_summary(client, sentiment_run["data"]["run"]["id"], name="sentiment_summary")
        steps.append(sentiment_summary)
        steps.append(
            timed_request(
                client,
                "sentiment_section",
                "GET",
                f"{BASE_URL}/api/analyses/{sentiment_run['data']['run']['id']}/sections/sentiment",
                params={"page": 1, "page_size": 50},
            )
        )

        OUTPUT_PATH.write_text(
            json.dumps(
                {
                    "dataset_file": str(DATASET_PATH),
                    "dataset_id": dataset_id,
                    "steps": steps,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        print(str(OUTPUT_PATH))


if __name__ == "__main__":
    main()
