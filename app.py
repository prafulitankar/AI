import boto3
import botocore.config
import json
from datetime import datetime

def blog_generate_using_bedrock(blogtopic: str) -> str:
    prompt = f"""<s>[INST] Write a 200-word blog on the topic: {blogtopic} 
    Assistant:[/INST]</s>"""

    body = {
        "prompt": prompt,
        "max_tokens": 512,
        "temperature": 0.5,
        "top_p": 0.9
    }

    try:
        bedrock = boto3.client(
            "bedrock-runtime",
            region_name="us-east-1",
            config=botocore.config.Config(read_timeout=300, retries={'max_attempts': 3})
        )

        response = bedrock.invoke_model(
            modelId="mistral.mistral-7b-instruct-v0:2",
            body=json.dumps(body),
            contentType="application/json",
            accept="application/json"
        )

        response_content = response['body'].read()
        response_data = json.loads(response_content)
        print("Bedrock response:", response_data)

        return response_data.get("outputs", [{}])[0].get("text", "")

    except Exception as e:
        print(f"Error generating the blog: {e}")
        return ""

def save_blog_details_s3(s3_key, s3_bucket, generate_blog):
    s3 = boto3.client('s3')

    try:
        s3.put_object(Bucket=s3_bucket, Key=s3_key, Body=generate_blog)
        print("Blog saved to S3.")
    except Exception as e:
        print(f"Error saving the blog to S3: {e}")

def lambda_handler(event, context):
    print("Event received:", json.dumps(event))

    if 'body' not in event:
        return {
            'statusCode': 400,
            'body': json.dumps({'error': "Missing 'body' in request"})
        }

    try:
        event_body = json.loads(event['body'])
        blogtopic = event_body.get('blog_topic')

        if not blogtopic:
            return {
                'statusCode': 400,
                'body': json.dumps({'error': "Missing 'blog_topic' in request body"})
            }

        generated_blog = blog_generate_using_bedrock(blogtopic=blogtopic)

        if generated_blog:
            current_time = datetime.now().strftime('%H%M%S')
            s3_key = f"blog-output/{current_time}.txt"
            s3_bucket = 'aws-bedrock-praful'
            save_blog_details_s3(s3_key, s3_bucket, generated_blog)
        else:
            print("No blog was generated")

        return {
            'statusCode': 200,
            'body': json.dumps('Blog generation is completed')
        }

    except Exception as e:
        print(f"Unhandled exception: {e}")
        return {
            'statusCode': 500,
            'body': json.dumps({'error': str(e)})
        }
