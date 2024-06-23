import json
import boto3
import logging
import uuid

s3_bucket_out_put_transcribe = ""

def generate_response(body, model_id):
    logging.log("Start generate response by invoking AWS Bedrock\n")
    bedrock_client = boto3.client(
        "bedrock-runtime",
        region_name="us-east-1",
    )
    accept = "application/json"
    content_type = "application/json"
    response = bedrock_client.invoke_model(
        body=body, modelId=model_id, accept=accept, contentType=content_type
    )
    response_body = json.loads(response.get("body").read())
    error_reason = response_body.get("error")

    if error_reason is not None:
        raise ValueError(error_reason)

    logging.info("Successfully generated response")

    return response_body


def transcribe_voice_to_text(s3_uri):
    transcribe_client = boto3.client(
        "transcribe",
    )
    logging.log("Start transcribe voice to text\n")
    job_name = str(uuid.uuid4())
    output_file_name = f"{str(uuid.uuid4())}.txt"
    response = transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        LanguageCode='en-US',  # Set the language code if different
        Media={
            'MediaFileUri': s3_uri
        },
        OutputBucketName=s3_bucket_out_put_transcribe,
        OutputKey=output_file_name
    )
    logging.log(f"Transcription job with job name: {job_name}")
    logging.log(f"Transcription response: {response}")

    return {
        output_file_name
    }

def handler(event, context):
    # TODO: Receive event from KVS and get S3 path to receive audio file

    s3_uri = ""

    transcribe_out_put_file_name = transcribe_voice_to_text(s3_uri)

    # TODO: Use transcribe_out_put_file_name to create a prompt and invoke Bedrock to generate response

    prompt = """
    write a summary of Manchester United
    """

    model_id = "amazon.titan-text-lite-v1"
    body = json.dumps(
        {
            "inputText": prompt,
            "textGenerationConfig": {
                "maxTokenCount": 3072,
                "stopSequences": [],
                "temperature": 0.7,
                "topP": 0.9,
            },
        }
    )
    response = generate_response(body, model_id)
    generation = response["results"][0]["outputText"]
    print(generation)

    return {
        "statusCode": 200,
        "message": response 
    }
