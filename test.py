import os
import requests
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from transformers import pipeline
from flask import Flask, request, jsonify

from dotenv import load_dotenv


# loading the environment variable
load_dotenv()
#print('Hello world')
HF_KEY = os.getenv('HF_KEY')
#print(HF_KEY)

# loading the preprocessor
processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

def image_description(image_url):
  processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
  blip_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large")

  raw_image = Image.open(requests.get(image_url, stream=True).raw).convert('RGB')
  
  text = "a photography of"
  inputs = processor(raw_image, text, return_tensors="pt")

  out = blip_model.generate(**inputs)
  return processor.decode(out[0], skip_special_tokens=True)
#text_of_the_image = image_description(image_url)


## text to emotion classification
#classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")
def text_to_emotion(text):
    classifier = pipeline("sentiment-analysis", model="michellejieli/emotion_text_classifier")
    return classifier(text)[0]['label']
def process_image(image_url):
   description = image_description(image_url)
   emotion = text_to_emotion(description)
   return description, emotion



app = Flask(__name__)

@app.route('/analyze', methods=['POST'])
def analyze():
    data = request.get_json()
    image_url = data.get('https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg')
    description, emotion = process_image(image_url)
    return jsonify({'description': description, 'emotion': emotion})

if __name__ == '__main__':
    app.run(debug=True)