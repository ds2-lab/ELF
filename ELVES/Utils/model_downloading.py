import os
import torch
#from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer, AutoImageProcessor, ResNetForImageClassification
from transformers import AutoModelForSequenceClassification
from sentence_transformers import SentenceTransformer
from Utils.config import *
from Utils.utils import folder_making_fun

def model_downloading(model_name_list):
    cnt = 0
    for model_name in model_name_list:
        print("\n", cnt, "Downloading Model:", model_name, "...")
        model_downloading_fun(model_name)
        cnt += 1

def model_downloading_fun(model_name):
    # Use the replace method to change '/' to '_' for folder & path management.
    model_item_folder_name = model_name.replace('/', '_')
    model_foler = model_original_folder+model_item_folder_name+"/"
    folder_making_fun(model_foler)
    model_path = model_foler+"pytorch_model.bin"
    if not os.path.exists(model_path):
        if model_name == "microsoft_resnet-50":
            model = ResNetForImageClassification.from_pretrained("microsoft/resnet-50")
        elif model_name == "sentence-transformers_all-MiniLM-L6-v2":
            model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        else:
            model = AutoModelForSequenceClassification.from_pretrained(model_name)
        torch.save(model.state_dict(), model_path)

