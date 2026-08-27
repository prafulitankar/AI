#This code Generate the CF template itself in Postman not storing anywhere when you hit API Gateway this will create a CF template
import os
import json
import re
import pydantic
import pydantic_core
from jiter import jiter
from portkey_ai import Portkey

def lambda_handler(event, context):
    portkey_api_key = os.environ.get("PORTKEY_API_KEY")
    if not portkey_api_key:
        return {
            "statusCode": 500,
            "body": "Missing PORTKEY_API_KEY in environment variables"
        }

    try:
        body = json.loads(event.get("body", "{}"))
    except Exception:
        return {
            "statusCode": 400,
            "body": "Invalid JSON in request body"
        }

    prompt_text = body.get("prompt")
    if not prompt_text or not prompt_text.strip():
        return {
            "statusCode": 400,
            "body": "Missing or empty 'prompt' in event payload. Please provide a meaningful prompt."
        }

    portkey = Portkey(
        base_url="https://portkeygateway.perficient.com/v1",
        api_key=portkey_api_key
    )

    try:
        response = portkey.chat.completions.create(
            model="@aws-bedrock-use2/us.anthropic.claude-opus-4-6-v1",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates AWS CloudFormation templates in YAML format. Do not wrap the output in markdown fences."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=2048,
            temperature=0.3
        )

        reply = response.choices[0].message.content

        # Strip markdown fences if present
        # Remove ```yaml ... ``` or ``` ... ```
        reply = re.sub(r"^```[a-zA-Z]*\n?", "", reply)   # remove opening fence
        reply = re.sub(r"\n?```$", "", reply)            # remove closing fence

        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "text/yaml"
            },
            "body": reply.strip()
        }

    except Exception as e:
        return {
            "statusCode": 500,
            "body": f"Error: {str(e)}"
        }
