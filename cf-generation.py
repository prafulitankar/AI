import boto3
import botocore.config
import json
import re
import logging
from datetime import datetime

logger = logging.getLogger()
logger.setLevel(logging.INFO)

# === CONFIG ===
MODEL_ID = "mistral.mistral-7b-instruct-v0:2"
REGION = "us-east-1"
S3_BUCKET = "aws-bedrock-praful"  # <-- change this

# === Extract YAML block from response text ===
def extract_yaml(text: str) -> str:
    match = re.search(r"```yaml(.*?)```", text, re.DOTALL | re.IGNORECASE)
    if match:
        return match.group(1).strip()
    # fallback: try just ```
    match = re.search(r"```(.*?)```", text, re.DOTALL)
    if match:
        return match.group(1).strip()
    return text.strip()

# === Call Bedrock Mistral Model ===
def generate_cfn_with_bedrock(prompt_text: str) -> str:
    client = boto3.client("bedrock-runtime", region_name=REGION)

    body = {
        "prompt": f"<s>[INST] Generate a valid AWS CloudFormation YAML template for the following request. "
                  f"Do NOT include any explanation or markdown formatting:\n\n{prompt_text} [/INST]</s>",
        "max_tokens": 1024,
        "temperature": 0.3,
        "top_p": 0.9
    }

    try:
        response = client.invoke_model(
            modelId=MODEL_ID,
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )

        response_content = response['body'].read()
        logger.info(f"Raw model response: {response_content.decode('utf-8')}")
        response_data = json.loads(response_content)
        model_output = response_data.get("completion") or response_data.get("outputs", [{}])[0].get("text", "")

        yaml_only = extract_yaml(model_output)
        return yaml_only

    except Exception as e:
        logger.error(f"Bedrock invocation error: {e}", exc_info=True)
        raise

# === Save file to S3 ===
def save_to_s3(key: str, content: str):
    s3 = boto3.client("s3")
    s3.put_object(Bucket=S3_BUCKET, Key=key, Body=content)
    logger.info(f"Saved generated template to s3://{S3_BUCKET}/{key}")

# === Main Lambda Handler ===
def lambda_handler(event, context):
    try:
        logger.info(f"Incoming event: {json.dumps(event)}")
        payload = json.loads(event.get("body", "{}"))
        request_text = payload.get("cfn_request")

        if not request_text:
            return {
                "statusCode": 400,
                "body": json.dumps({"error": "Missing 'cfn_request' in request body"})
            }

        # Generate template
        yaml_template = generate_cfn_with_bedrock(request_text)
        if not yaml_template:
            return {
                "statusCode": 500,
                "body": json.dumps({"error": "Model returned empty template"})
            }

        # Save to S3
        timestamp = datetime.utcnow().strftime('%Y%m%d%H%M%S')
        s3_key = f"cloudformation/generated-template-{timestamp}.yaml"
        save_to_s3(s3_key, yaml_template)

        return {
            "statusCode": 200,
            "body": json.dumps({
                "s3_key": s3_key,
                "preview": yaml_template[:300]
            })
        }

    except Exception as e:
        logger.error(f"Unhandled exception: {e}", exc_info=True)
        return {
            "statusCode": 500,
            "body": json.dumps({"error": str(e)})
        }
