from __future__ import annotations

import os
import sys
import time
import json
from typing import Dict, Any, List

import boto3
from botocore.config import Config

REGION = os.getenv("AWS_REGION", os.getenv("AWS_DEFAULT_REGION", "us-west-2"))
USE_SSM = os.getenv("USE_SSM_PARAMS", "1") == "1"
SSM_PREFIX = os.getenv("SSM_PARAM_PREFIX")  # defaults to /{account}-{region}/kb


def resolve_ids() -> Dict[str, str]:
    sts = boto3.client("sts")
    account_id = sts.get_caller_identity()["Account"]
    region = boto3.Session().region_name or REGION

    if USE_SSM:
        ssm = boto3.client("ssm", region_name=region)
        prefix = SSM_PREFIX or f"/{account_id}-{region}/kb"
        kb_id = ssm.get_parameter(Name=f"{prefix}/knowledge-base-id")["Parameter"]["Value"]
        ds_id = ssm.get_parameter(Name=f"{prefix}/data-source-id")["Parameter"]["Value"]
        bucket = f"{account_id}-{region}-kb-data-bucket"
        return {"kb_id": kb_id, "ds_id": ds_id, "bucket": bucket, "region": region}

    kb_id = os.getenv("KB_ID")
    ds_id = os.getenv("DATA_SOURCE_ID")
    if not kb_id or not ds_id:
        raise RuntimeError("Missing KB_ID or DATA_SOURCE_ID and USE_SSM_PARAMS=0")
    # Bucket is optional in this path; provide if you want listing
    bucket = os.getenv("S3_BUCKET", f"{account_id}-{region}-kb-data-bucket")
    return {"kb_id": kb_id, "ds_id": ds_id, "bucket": bucket, "region": region}

def list_bucket_objects(bucket: str) -> List[str]:
    s3 = boto3.client("s3", region_name=REGION)
    try:
        resp = s3.list_objects_v2(Bucket=bucket)
        return [o["Key"] for o in resp.get("Contents", [])]
    except Exception as e:
        print(f"[warn] could not list bucket {bucket}: {e}")
        return []

def start_and_wait_ingestion(kb_id: str, ds_id: str, *, description: str = "Quick sync", poll_seconds: int = 10) -> Dict[str, Any]:
    agent = boto3.client("bedrock-agent", region_name=REGION, config=Config(retries={"max_attempts": 3, "mode": "adaptive"}))
    resp = agent.start_ingestion_job(
        knowledgeBaseId=kb_id,
        dataSourceId=ds_id,
        description=description,
    )
    job_id = resp["ingestionJob"]["ingestionJobId"]
    print(f"[info] started ingestion job {job_id}")

    while True:
        job = agent.get_ingestion_job(
            knowledgeBaseId=kb_id,
            dataSourceId=ds_id,
            ingestionJobId=job_id,
        )["ingestionJob"]
        status = job["status"]
        print(f"[info] job {job_id} status: {status}")
        if status in ("COMPLETE", "FAILED", "STOPPED"):
            return job
        time.sleep(poll_seconds)

def main(argv: List[str]) -> int:
    do_list = "--list-s3" in argv
    do_sync = "--sync" in argv

    ids = resolve_ids()
    kb_id, ds_id, bucket = ids["kb_id"], ids["ds_id"], ids["bucket"]

    if do_list:
        files = list_bucket_objects(bucket)
        if files:
            print(f"[info] {bucket} contains {len(files)} objects:")
            for k in files:
                print("  -", k)
        else:
            print(f"[info] no objects found or listing failed for {bucket}")

    if do_sync:
        job = start_and_wait_ingestion(kb_id, ds_id)
        stats = job.get("statistics", {})
        summary = {
            "status": job.get("status"),
            "ingestionJobId": job.get("ingestionJobId"),
            "documentsScanned": stats.get("numberOfDocumentsScanned", 0),
            "documentsAdded": stats.get("numberOfNewDocumentsIndexed", 0),
            "documentsUpdated": stats.get("numberOfModifiedDocumentsIndexed", 0),
            "documentsDeleted": stats.get("numberOfDocumentsDeleted", 0),
        }
        print(json.dumps(summary, indent=2))

    if not (do_list or do_sync):
        print("Usage: python kb_ingest_sync.py [--list-s3] [--sync]")

    return 0

if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
