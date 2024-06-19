import json
import boto3

bedrock_client = boto3.client('bedrock')

def connect_bedrock(event, context):
  print("The first function has been invoked!!")
  return {
    'statusCode': 200,
    'body': "Hello, World!.\n This is the first function."
  }