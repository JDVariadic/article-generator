### Note:
Credits to https://www.kaggle.com/datasets/fabiochiusano/medium-articles for the dataset

# Use-Case:
This article generator is to brainstorm introduction and ideas for blogging purposes. The model being used is the smallest GPT-2 variant with around 125M parameters. Due to hardware limitations, the API can only effectively return up to 300-400 tokens. The max number of tokens that can be generated by the API is capped at 400 with the parameter top_k capped at 50.

Link to API: https://jdvariadic-gpt2-article-generator.hf.space

POST endpoint used for API: generate-article

# Query Parameters:

**title** (str): Generates a short article given the title

**max_length** (int, optional): Specifies the max number of tokens generated by GPT-2

**top_k** (int, optional): Specifices the parameter top_k for the sampling method used.

## Usage:
Curl:
``` 
curl -X 'POST' \
  'https://jdvariadic-gpt2-article-generator.hf.space/generate-article' \
  -H 'accept: application/json' \
  -H 'Content-Type: application/json' \
  -d '{
  "title": "Insert article title here",
  "max_length": 400,
  "top_k": 50
}'
```

Or with Python:
```
import requests

url = 'https://jdvariadic-gpt2-article-generator.hf.space/generate-article'
myobj = {'title': 'Why I started reading books this year.', "max_length": 350}

x = requests.post(url, json = myobj)
```

# Setup Guide:

Go to https://huggingface.co/spaces

Click on Create New Space.

Specify name, license, and click on Blank Docker template

Select Space hardware (For this implementation, the CPU basic is used)

Build the endpoint by selecting Create Space.

# Model Fine-Tuning guide:

## Adding special [TITLE] and [/TITLE] tokens:

custom-gpt2-tokenizer is a modified version of the original gpt-2 tokenizer but with added title tags to add separation between the training data.

## Training Proper

Feel free to reference preprocessing.ipynb

the medium_articles.csv file is concatenated into a text-file. Fine-tuning was done using the Trainer class from the transformers library.
