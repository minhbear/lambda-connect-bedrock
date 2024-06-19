import json
import boto3

bedrock_client = boto3.client(
  'bedrock-runtime',
  region_name = 'us-east-1',
)

prompt = "write a summary of Manchester United"
# Lets create a text prompt
# Lets put other details to our Key word argument
kwargs={
    'modelId':'amazon.titan-text-lite-v1',
    'contentType':'application/json',
    'accept':'*/*',
    'body':json.dumps(
    {
        'inputText':prompt,
        "textGenerationConfig":{
            'maxTokenCount':500,
            'temperature':0.7,
            'topP':0.9
        }
    }
    )
}


def connect_bedrock(event, context):
  print("Start invoke Bedrock!!\n")

  response = bedrock_client.invoke_model(**kwargs)
  response_body = json.loads(response.get('body').read())

  generation = response_body['results'][0]['outputText']  
  print("Printing invoked Bedrock!!\n")
  print(generation)

  return {
    'statusCode': 200,
    'body': "Hello, World!.\n This is the first function."
  }