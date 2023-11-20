import torch
import requests
from PIL import Image
from transformers import AutoProcessor, Blip2ForConditionalGeneration
import sys
import gc
import os

MODEL_MAPPING = {
    "1": "Salesforce/blip2-opt-2.7b",
    "2": "Salesforce/blip2-flan-t5-xl",
    "3": "Salesforce/blip2-opt-6.7b",
    "4": "Salesforce/blip2-opt-6.7b-coco"
}

def generate_image_caption(path, num_model="1", clean_mem=False):

    # Clean torch cuda memory 
    if clean_mem:           
        torch.cuda.empty_cache()
        gc.collect()

    path = path if os.path.isabs(path) else requests.get(path, stream=True).raw
    image = Image.open(path).convert('RGB')  

    if num_model in MODEL_MAPPING:
        name_model = MODEL_MAPPING[num_model]
    else:
        name_model = "Salesforce/blip2-opt-2.7b"

    processor = AutoProcessor.from_pretrained(name_model)
    model = Blip2ForConditionalGeneration.from_pretrained(name_model, torch_dtype=torch.float16)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    inputs = processor(image, return_tensors="pt").to(device, torch.float16)

    generated_ids = model.generate(**inputs, max_new_tokens=20)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
    return generated_text

def generate_multiple_image_captions(paths, num_model="1"):

    # Clean torch cuda memory      
    torch.cuda.empty_cache()
    gc.collect()

    images = []
    for path in paths:        
        images.append(Image.open(path).convert('RGB'))  

    if num_model in MODEL_MAPPING:
        name_model = MODEL_MAPPING[num_model]
    else:
        name_model = "Salesforce/blip2-opt-2.7b"

    processor = AutoProcessor.from_pretrained(name_model)
    model = Blip2ForConditionalGeneration.from_pretrained(name_model, torch_dtype=torch.float16)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    generated_texts = []
    for i, image in enumerate(images):
        print('Generating description of image', str(i+1).zfill(3), 'of', str(len(images)).zfill(3) + '...')
        inputs = processor(image, return_tensors="pt").to(device, torch.float16)
        generated_ids = model.generate(**inputs, max_new_tokens=20)
        generated_texts.append(processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip())

    return generated_texts

if __name__ == "__main__" and len(sys.argv) > 1:
    num_model = str(sys.argv[2]) if len(sys.argv) > 2 else "1"
    clean_mem = True if (len(sys.argv) > 3) and (sys.argv[3] == 1) else False
    print(generate_image_caption(sys.argv[1], num_model, clean_mem))
