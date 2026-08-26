#This code is working to connect from AWS Lambda to Perficient Portkey AI Models and able to Answer some basic qauetions not Generating the Exact CF Templates
import os
import json
import pydantic
import pydantic_core
from jiter import jiter
from portkey_ai import Portkey

def lambda_handler(event, context):
    # Load API key from environment variable
    portkey_api_key = os.environ.get("PORTKEY_API_KEY")
    if not portkey_api_key:
        return {
            "statusCode": 500,
            "body": "Missing PORTKEY_API_KEY in environment variables"
        }

    # Parse body from API Gateway event
    try:
        body = json.loads(event.get("body", "{}"))
    except Exception:
        return {
            "statusCode": 400,
            "body": "Invalid JSON in request body"
        }

    # Ensure a real prompt is provided
    prompt_text = body.get("prompt")
    if not prompt_text or not prompt_text.strip():
        return {
            "statusCode": 400,
            "body": "Missing or empty 'prompt' in event payload. Please provide a meaningful prompt."
        }

    # Initialize Portkey client
    portkey = Portkey(
        base_url="https://portkeygateway.perficient.com/v1",
        api_key=portkey_api_key
    )

    try:
        # Call Anthropic Claude Opus model via Portkey
        response = portkey.chat.completions.create(
            model="@aws-bedrock-use2/us.anthropic.claude-opus-4-6-v1",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt_text}
            ],
            max_tokens=512,
            temperature=0.7
        )

        # Debug: log raw response to CloudWatch
        print("Portkey raw response:", response)

        # Extract the assistant's reply
        reply = response.choices[0].message.content

        # Return dependency versions + AI reply
        return {
            "statusCode": 200,
            "body": json.dumps({
                "pydantic_version": pydantic.__version__,
                "pydantic_core_version": pydantic_core.__version__,
                "jiter_available": hasattr(jiter, "__call__") or hasattr(jiter, "__name__"),
                "ai_reply": reply
            })
        }

    except Exception as e:
        print("Error:", str(e))
        return {
            "statusCode": 500,
            "body": f"Error: {str(e)}"
        }
