import json
import boto3
import time

s3_bucket_out_put_transcribe = ""

def generate_response(body, model_id):
    print("Start generate response by invoking AWS Bedrock\n")
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

    print("Successfully generated response")

    return response_body


def transcribe_voice_to_text(s3_uri):
    transcribe_client = boto3.client(
        "transcribe",
    )
    print("Start transcribe voice to text\n")
    unique_name = str(int(time.time()))
    job_name = unique_name
    output_file_name = f"{unique_name}.txt"
    response = transcribe_client.start_transcription_job(
        TranscriptionJobName=job_name,
        LanguageCode="en-US",  # Set the language code if different
        Media={"MediaFileUri": s3_uri},
        OutputBucketName=s3_bucket_out_put_transcribe,
        OutputKey=output_file_name,
    )
    print(f"Transcription job with job name: {job_name}")
    print(f"Transcription response: {response}")

    # Wait for the transcription job to complete
    while True:
        transcribe_response = transcribe_client.get_transcription_job(
            TranscriptionJobName=job_name
        )

        transcription_job_status = transcribe_response["TranscriptionJob"][
            "TranscriptionJobStatus"
        ]

        if transcription_job_status == "COMPLETED":
            break

        if transcription_job_status == "FAILED":
            raise ValueError(
                f'Transcription job failed with error: {transcribe_response["TranscriptionJob"]["FailureReason"]}'
            )
    
        time.sleep(1)

    return {output_file_name}


def get_transcribe_text(transcribe_out_put_file_name):
    s3 = boto3.client("s3")
    response = s3.get_object(
        Bucket=s3_bucket_out_put_transcribe, Key=transcribe_out_put_file_name
    )

    # Read the content of the object
    data = response["Body"].read().decode("utf-8")
    load_data = json.loads(data)

    transcribe_text = load_data["results"]["transcripts"][0]["transcript"]

    # Print the content
    print(f"transcribe_text: {transcribe_text}")
    return transcribe_text


def generate_prompt(transcribe_text):
    # TODO: handle prompt with transcribe text
    prompt = """
    write a summary of Manchester United
    """

    return prompt


def handler(event, context):
    # TODO: Receive event from KVS and get S3 path to receive audio file
    s3_uri = ""

    transcribe_out_put_file_name = transcribe_voice_to_text(s3_uri)

    # TODO: Use transcribe_out_put_file_name to create a prompt and invoke Bedrock to generate response

    transcribe_text = get_transcribe_text(transcribe_out_put_file_name)

    prompt = generate_prompt(transcribe_text)

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

    return {"statusCode": 200, "message": response}
