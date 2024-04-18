import sys
import pickle
import os
import numpy as np
import torch
import shutil
from collections import OrderedDict
from tqdm import tqdm

def rounding(num):
    return round(num, 4)

def model_loading_fun(model_path):
    #print("model path:", model_path)
    try:
        model = torch.load(model_path, map_location='cpu')
    except Exception as e:
        print("Model load by torch failed, please check the model existance and model format!")
        sys.exit()
    return model

def get_shared_storage_tensor_dict(model):
    # finding the layers that shareing memory
    try:
        same_storage_dict = dict()
        for layer_name in model:
            if layer_name in same_storage_dict:
                continue
            layer = model[layer_name]
            for layer_name_check in model:
                if layer_name == layer_name_check:
                    continue
                layer_check = model[layer_name_check]
                if layer.storage().data_ptr() == layer_check.storage().data_ptr():
                    same_storage_dict[layer_name_check] = layer_name
                    if not torch.equal(layer, layer_check):
                        print("not torch.equal(layer, layer_check)")
                        sys.exit()
        return same_storage_dict
    except Exception as e:
        # Code that runs if an exception of ExceptionType occurs
        print(f"An error occurred: get_shared_storage_tensor_dict")
        sys.exit()


def get_folder_path_for_distance(model_weights_file):
    model_weights_flg = False
    for i in range(len(model_weights_file)-1, 0, -1):
        if model_weights_file[i] == '/':
            if model_weights_flg is False:
                model_weights_flg = True
                continue
            return model_weights_file[:i+1]

def delete_file(file_path):
    try:
        os.remove(file_path)
    except FileNotFoundError:
        print(f"File not found: {file_path}")

def delete_folder(folder_path):
    try:
        # Use shutil.rmtree() to delete the folder and its contents
        shutil.rmtree(folder_path)
        #print(f"Folder '{folder_path}' and all its contents have been deleted successfully.")
    except FileNotFoundError:
        print(f"Folder '{folder_path}' not found.")
    except Exception as e:
        print(f"Error while deleting folder: {e}")

def get_folder_size(folder_path):
    if not os.path.exists(folder_path):
        return 0
    if not os.path.isdir(folder_path):
        print("get_folder_size function argument should be a folder:", folder_path)
        sys.exit()
    folder_size_total = 0
    for item in os.listdir(folder_path):
        item_path = os.path.join(folder_path, item)
        if os.path.isdir(item_path):
            folder_size_total += get_folder_size(item_path)
        else:
            folder_size_total += os.path.getsize(item_path)
    return folder_size_total

def folder_making_fun(folder):
    if not os.path.exists(os.path.dirname(folder)):
        #print("Making dir:", folder)
        os.makedirs(os.path.dirname(folder))

def seal_pickle(file_name, data_name):
    file_open = open(file_name, "wb")
    pickle.dump(data_name, file_open)
    file_open.close()


def unseal_pickle(file_name):
    #print("unpickling", file_name)
    with open(file_name, 'rb') as file:
        data_name = pickle.load(file)
    return data_name

def list_print(list_data):
    for i in range(len(list_data)):
        print(i, list_data[i])

def set_print(data_set):
    cnt = 0
    for item in data_set:
        print(cnt, item)
        cnt += 1

def dict_print(dict_data):
    cnt = 0
    for key in dict_data:
        print(cnt, key, dict_data[key])
        cnt += 1


