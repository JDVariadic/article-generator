### Note:
Credits to https://www.kaggle.com/datasets/fabiochiusano/medium-articles for the dataset

# Use-Case:
This article generator is to brainstorm introduction and ideas for blogging purposes. Due to hardware limitations, the API can only effectively return up to 300-400 tokens.

Link to API: https://jdvariadic-gpt2-article-generator.hf.space
POST endpoint: generate-article

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

# Setup Guide:

Go to https://huggingface.co/spaces

Click on Create New Space.

Specify name, license, and click on Blank Docker template

Select Space hardware (For this implementation, the CPU basic is used)

Build the endpoint by selecting Create Space.

# Model Fine-Tuning guide:

Feel free to reference preprocessing.ipynb