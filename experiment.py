import sys
import os
import pandas as pd
from image_caption import generate_multiple_image_captions, MODEL_MAPPING
from language_tool_python import LanguageTool

CHAKRABARTY_SYSTEM_FOLDER = 'SarcasmGeneration-ACL2020-modified'
sys.path.insert(0, CHAKRABARTY_SYSTEM_FOLDER)
from reverse import reverse_valence, getWordNetAntonyms, correct_sentence

# Número de imágenes del experimento
NUM_IMAGES = 100

# Número de modelo de generación de subtítulos de imágenes
num_model = str(sys.argv[1]) if len(sys.argv) > 1 else "1"
print('Model', MODEL_MAPPING[num_model], 'selected.')

# Carga de datos
print('Loading data...')
with open(os.getcwd()+'/data-of-multimodal-sarcasm-detection-master/text/test2.txt', 'r', encoding='utf-8') as file:
    lines = file.readlines()

data = []
for line in lines:
    list = eval(line)
    element = {
        "Image": list[0],
        "Text": list[1],
        "Hashtag label": int(list[2]),
        "Human label": int(list[3])
    }
    data.append(element)

# Seleccionamos muestras positivas
df = pd.DataFrame(data)
df = df[df['Text'].apply(lambda x: isinstance(x, str) and all(ord(c) < 128 for c in x))]
df = df[(df["Hashtag label"] == 1) & (df["Human label"] == 1)]
df = df.sample(n=NUM_IMAGES).reset_index(drop=True)

# Ruta de imágenes
images = []
for img in df['Image']:
    images.append(os.getcwd()+'\\dataset_image\\'+str(img)+'.jpg')

# Descripciones de imágenes
print('Generating image descriptions...')
descriptions = generate_multiple_image_captions(images, num_model)

# Descripciones irónicas de imágenes (inversión de valencias)
print('Applying reversal of valence...\n')
os.chdir(os.getcwd()+'\\'+CHAKRABARTY_SYSTEM_FOLDER)
antonyms = getWordNetAntonyms()
correction_tool = LanguageTool('en-US')
count_modified = 0
for i, desc in enumerate(descriptions):
    print('Image', str(i+1).zfill(3), "of", str(NUM_IMAGES).zfill(3), '-', images[i])
    print("Text        -", df['Text'][i])
    print("Description -", desc)
    irony = reverse_valence(desc, antonyms, correction_tool)
    print("Irony       -", irony, '\n')
    if correct_sentence(desc, correction_tool).lower().replace('.','') != irony.lower().replace('.',''):
        count_modified +=1
print("The percentage of ironic subtitles generated is", str('%.2f'%(count_modified/NUM_IMAGES*100)) + "%")
