import os
import sys
import argparse
import git
from dvc.repo import Repo
from urllib.parse import urlparse
import boto3
from botocore.exceptions import ClientError

# Ensure dummy credentials to avoid boto3 connection issues with real AWS
os.environ.setdefault("AWS_ACCESS_KEY_ID", "mock_key")
os.environ.setdefault("AWS_SECRET_ACCESS_KEY", "mock_secret")
os.environ.setdefault("AWS_DEFAULT_REGION", "us-east-1")

def get_s3_client():
    # Fetch configurable S3 endpoint (defaulting to local mock server)
    endpoint_url = os.getenv("AWS_S3_ENDPOINT_URL", "http://localhost:5000")
    print(f"🔌 Connecting to S3 Endpoint: {endpoint_url}")
    return boto3.client("s3", endpoint_url=endpoint_url, region_name="us-east-1")

def parse_s3_url(s3_url):
    parsed = urlparse(s3_url)
    if parsed.scheme != "s3":
        raise ValueError("Invalid URL scheme. Must be s3://bucket/key")
    bucket = parsed.netloc
    key = parsed.path.lstrip("/")
    return bucket, key

def upload(args):
    s3 = get_s3_client()
    bucket, key = parse_s3_url(args.s3_url)
    
    if not os.path.exists(args.local_path):
        print(f"❌ Local file not found: {args.local_path}")
        sys.exit(1)
        
    print(f"📂 Preparing to upload '{args.local_path}' to 's3://{bucket}/{key}'...")
    
    # Ensure bucket exists
    try:
        s3.create_bucket(Bucket=bucket)
        print(f"✅ Created S3 Bucket: {bucket}")
    except ClientError as e:
        # If bucket already exists, proceed
        error_code = e.response.get("Error", {}).get("Code")
        if error_code in ("BucketAlreadyExists", "BucketAlreadyOwnedByYou"):
            print(f"ℹ️ Bucket '{bucket}' already exists.")
        else:
            print(f"❌ Failed to create bucket: {e}")
            sys.exit(1)
            
    # Upload file
    try:
        s3.upload_file(args.local_path, bucket, key)
        print(f"🚀 Successfully uploaded '{args.local_path}' to 's3://{bucket}/{key}'")
    except Exception as e:
        print(f"❌ Failed to upload file: {e}")
        sys.exit(1)

def extract(args):
    s3 = get_s3_client()
    bucket, key = parse_s3_url(args.s3_url)
    
    # Ensure parent directories for local path exist
    local_dir = os.path.dirname(args.local_path)
    if local_dir:
        os.makedirs(local_dir, exist_ok=True)
        
    print(f"📥 Extracting 's3://{bucket}/{key}' to '{args.local_path}'...")
    
    try:
        s3.download_file(bucket, key, args.local_path)
        print(f"✅ Successfully extracted file to: {args.local_path}")
    except Exception as e:
        print(f"❌ Failed to download file from S3: {e}")
        sys.exit(1)

def track(args):
    if not os.path.exists(args.local_path):
        print(f"❌ Target path does not exist: {args.local_path}")
        sys.exit(1)
        
    print(f"📦 Tracking '{args.local_path}' with DVC...")
    try:
        # Prevent Git ownership check warnings when running inside a docker container
        try:
            git.Git().config("--global", "--add", "safe.directory", "*")
        except Exception as e:
            print(f"ℹ️ Git config warning: {e}")
        
        # Run DVC command natively
        repo = Repo(".")
        repo.add(args.local_path)
        print(f"✅ Successfully added '{args.local_path}' to DVC tracking.")
    except Exception as e:
        print(f"❌ DVC Add Failed: {e}")
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description="MLOps Data Importer and Tracker CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)
    
    # Upload subcommand
    parser_upload = subparsers.add_parser("upload", help="Upload local file to mock S3")
    parser_upload.add_argument("local_path", help="Path to local file")
    parser_upload.add_argument("s3_url", help="S3 URL (s3://bucket/key)")
    parser_upload.set_defaults(func=upload)
    
    # Extract subcommand
    parser_extract = subparsers.add_parser("extract", help="Download file from mock S3")
    parser_extract.add_argument("s3_url", help="S3 URL (s3://bucket/key)")
    parser_extract.add_argument("local_path", help="Path to save local file")
    parser_extract.set_defaults(func=extract)
    
    # Track subcommand
    parser_track = subparsers.add_parser("track", help="Track file with DVC")
    parser_track.add_argument("local_path", help="Path to file to track")
    parser_track.set_defaults(func=track)
    
    args = parser.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
