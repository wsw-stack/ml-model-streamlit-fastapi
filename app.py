import os
import time
import uvicorn
import torch
from transformers import pipeline, AutoImageProcessor
from fastapi import FastAPI, Request
from scripts import s3
from scripts.data_model import NLPDataInput, NLPDataOutput, ImageDataInput, ImageDataOutput

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
app = FastAPI()
model_ckpt = "google/vit-base-patch16-224-in21k"
image_processor = AutoImageProcessor.from_pretrained(model_ckpt, use_fast=True)

# Download ml models if they do not exist
model_name = 'tinybert-sentiment-analysis'
local_path = 'ml-models/' + model_name
if not os.path.isdir(local_path):
    s3.download_dir(local_path, model_name)

sentiment_model = pipeline('text-classification', model=local_path, device=device)

model_name = 'tinybert-disaster-tweet'
local_path = 'ml-models/' + model_name
if not os.path.isdir(local_path):
    s3.download_dir(local_path, model_name)
disaster_model = pipeline('text-classification', model=local_path, device=device)

model_name = 'vit-human-pose-classification'
local_path = 'ml-models/' + model_name
if not os.path.isdir(local_path):
    s3.download_dir(local_path, model_name)
pose_model = pipeline('image-classification', model=local_path, device=device, image_processor=image_processor)

@app.get('/')
def hello():
    return 'Hello world!'

@app.post('/api/v1/sentiment_analysis')
def sentiment_analysis(data: NLPDataInput):
    start = time.time()
    output = sentiment_model(data.text)
    end = time.time()
    prediction_time = int(1000 * (end - start))
    labels = [x['label'] for x in output]
    scores = [x['score'] for x in output]

    output = NLPDataOutput(model_name='sentiment_analysis',
                          text=data.text,
                          labels=labels,
                          scores=scores,
                          prediction_time=prediction_time)
    return output

@app.post('/api/v1/disaster_classifier')
def disaster_classifier(data: NLPDataInput):
    start = time.time()
    output = disaster_model(data.text)
    end = time.time()
    prediction_time = int(1000 * (end - start))
    labels = [x['label'] for x in output]
    scores = [x['score'] for x in output]

    output = NLPDataOutput(model_name='tinybert_disaster_tweet',
                          text=data.text,
                          labels=labels,
                          scores=scores,
                          prediction_time=prediction_time)
    return output

@app.post('/api/v1/pose_classifier')
def pose_classifier(data: ImageDataInput):
    urls = [str(x) for x in data.url]
    start = time.time()
    output = pose_model(urls)
    end = time.time()
    prediction_time = int(1000 * (end - start))
    labels = [x[0]['label'] for x in output]
    scores = [x[0]['score'] for x in output]

    output = ImageDataOutput(model_name='vit-human-pose-classification',
                          url=data.url,
                          labels=labels,
                          scores=scores,
                          prediction_time=prediction_time)
    return output

if __name__ == '__main__':
    uvicorn.run(app="app:app", port=8000, reload=True)