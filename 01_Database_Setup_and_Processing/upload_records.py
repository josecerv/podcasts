import boto3
import json
import os

# -----------------------------
# CONFIGURATION
# -----------------------------
# The name of your S3 bucket
BUCKET_NAME = 'podcast-episodes-bucket'
# The local filename of your JSONL file (adjust the path as necessary)
LOCAL_FILE = 'prod_db3.jsonl'
# The prefix in the bucket to store individual podcast files
TARGET_PREFIX = 'podcasts-v2/'

def split_and_upload():
    # Create an S3 client (credentials are automatically picked up from AWS CLI configuration)
    s3 = boto3.client('s3', region_name='us-east-1')

    # Check if the local file exists
    if not os.path.exists(LOCAL_FILE):
        print(f"Error: {LOCAL_FILE} not found in the current directory.")
        return

    # Open your local JSONL file for reading
    with open(LOCAL_FILE, 'r', encoding='utf-8') as infile:
        line_number = 0
        uploaded = 0
        for line in infile:
            line = line.strip()
            if not line:
                continue  # Skip any blank lines
            line_number += 1
            try:
                # Parse each line as JSON
                record = json.loads(line)
                podcast_id = record.get("podcast_id")
                if not podcast_id:
                    print(f"Skipping line {line_number}: 'podcast_id' key not found.")
                    continue

                # Define the S3 object key using the podcast_id (e.g., podcasts/<podcast_id>.json)
                s3_key = f"{TARGET_PREFIX}{podcast_id}.json"
                # Convert the record back into a compact JSON string
                content = json.dumps(record, separators=(',', ':'))
                
                # Upload the object to S3
                s3.put_object(Bucket=BUCKET_NAME, Key=s3_key, Body=content.encode('utf-8'))
                print(f"Uploaded line {line_number} as {s3_key}")
                uploaded += 1
            except json.JSONDecodeError as je:
                print(f"Error parsing JSON on line {line_number}: {je}")
            except Exception as e:
                print(f"Error on line {line_number}: {e}")

    print(f"Upload complete: {uploaded} records uploaded out of {line_number} lines.")

if __name__ == "__main__":
    split_and_upload()
