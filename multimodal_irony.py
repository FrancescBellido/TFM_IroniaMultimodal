import sys
import os
from urllib.parse import urlparse
from image_caption import generate_image_caption

CHAKRABARTY_SYSTEM_FOLDER = 'SarcasmGeneration-ACL2020-modified'
sys.path.insert(0, CHAKRABARTY_SYSTEM_FOLDER)
from reverse import reverse_valence

if len(sys.argv) > 1:
    # Description of image
    url = urlparse(sys.argv[1])
    path_image = sys.argv[1] if os.path.isabs(sys.argv[1]) or (url.scheme and url.netloc) else os.getcwd() + '\\dataset_image\\' + str(sys.argv[1]) + '.jpg'
    num_model = str(sys.argv[2]) if len(sys.argv) > 2 else "1"
    clean_mem = True if (len(sys.argv) > 3) and (sys.argv[3] == 1) else False    
    description = generate_image_caption(path_image, num_model, clean_mem)
    # Reversal of valence
    os.chdir(os.getcwd()+'\\'+CHAKRABARTY_SYSTEM_FOLDER)
    irony = reverse_valence(description)
    print(irony)
else:
    print('ERROR: Enter an image path as an argument.')
