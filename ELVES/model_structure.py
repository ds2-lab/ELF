import os
import pickle
import torch
import hashlib
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
from hashing_deduplication import get_shared_storage_tensor_dict, model_loading_fun
import sys
from Utils.utils import *
from Utils.config import *

def save_model_structure_and_flatten_weights(model_path_list):
    layer_hash_repeat_value_set_file = dup_layer_folder+"hash_layer_repeat_set.pkl"
    if not os.path.exists(layer_hash_repeat_value_set_file):
        print("file", layer_hash_repeat_value_set_file, "not exists. Hash layer deduplication needed first.")
        return

    model_cmp_structure_weights_folder = model_elves_compression
    #print("~"*4, "get_model_structure_and_flatten_weights", "~"*4)
    layer_hash_repeat_value_set = unseal_pickle(layer_hash_repeat_value_set_file)
    if not os.path.exists(os.path.dirname(model_cmp_structure_weights_folder)):
        #print("Making dir:", model_cmp_structure_weights_folder)
        os.makedirs(os.path.dirname(model_cmp_structure_weights_folder))

    cnt = 0
    model_size_total = 0
    total_layer_num = 0

    model_size_file_consis_ratio = dict()
    err_ratio_dict = dict()

    for model_path in model_path_list:
        model_name = model_path.split('/')[-2]
        #model_path = model_original_folder+model_name+"/pytorch_model.bin"
        model_size = os.path.getsize(model_path)
        print("\n", cnt, model_name, model_path, str(model_size/MB)+" MB")
        model_numpy_size_total = 0
        cnt += 1

        model_struct_weights_folder_individual = model_cmp_structure_weights_folder+model_name+"/"
        if not os.path.exists(os.path.dirname(model_struct_weights_folder_individual)):
            os.makedirs(os.path.dirname(model_struct_weights_folder_individual))
        else:
            if os.path.exists(model_struct_weights_folder_individual+"model_structure.pkl") and os.path.exists(model_struct_weights_folder_individual+"fl_weights/"):
                print("structure exists..")
                continue

        fl_weights_file_path = model_struct_weights_folder_individual+"fl_weights/"
        if not os.path.exists(os.path.dirname(fl_weights_file_path)):
            os.makedirs(os.path.dirname(fl_weights_file_path))

        model = model_loading_fun(model_path)
        # finding the layers that shareing memory
        same_storage_dict = get_shared_storage_tensor_dict(model)

        non_fl_layer_flg = False
        non_fl_layer_file_path = model_struct_weights_folder_individual+"non_fl_layers/"
        model_weights_flatten_f16 = list()
        model_weights_flatten_f32 = list()
        model_weights_flatten_f64 = list()
        model_structure = OrderedDict()
        pbar = tqdm(total=len(model))
        for layer_name in model:
            pbar.update(1)
            # this layer is a repeated layer within the model
            if layer_name in same_storage_dict:
                model_structure[layer_name] = [0, same_storage_dict[layer_name]]
                continue

            layer = model[layer_name]
            weights_numpy = layer.numpy()
            md5_hash = hashlib.md5(weights_numpy.tobytes()).hexdigest()
            # this layer is a repeated layer with other models by calculating the hash value
            if md5_hash in layer_hash_repeat_value_set:
                model_structure[layer_name] = [1, md5_hash, weights_numpy.shape]
                layer_hash_size = os.path.getsize(dup_layer_folder+md5_hash+".pkl")
                model_numpy_size_total += layer_hash_size
                continue

            layer_dtype = layer.dtype
            # if layer_dtype is float16, float32, float64, the weights would be added to lists seperately. We would perform the compression on it. And the shape would be recorded for recovery.
            if layer_dtype == torch.float16:
                model_structure[layer_name] = [16, weights_numpy.shape]
                adding_para_to_list(model_weights_flatten_f16, weights_numpy)
            elif layer_dtype == torch.float32:
                model_structure[layer_name] = [32, weights_numpy.shape]
                adding_para_to_list(model_weights_flatten_f32, weights_numpy)
            elif layer_dtype == torch.float64:
                model_structure[layer_name] = [64, weights_numpy.shape]
                adding_para_to_list(model_weights_flatten_f64, weights_numpy)
            else:
                # if there is any non float layer, then create this foler, otherwise no folder needed.
                if non_fl_layer_flg is False:
                    if not os.path.exists(os.path.dirname(non_fl_layer_file_path)):
                        os.makedirs(os.path.dirname(non_fl_layer_file_path))
                    non_fl_layer_flg = True
                non_fl_layer_file_item = non_fl_layer_file_path+layer_name+".pkl"
                seal_pickle(non_fl_layer_file_item, weights_numpy)
                # this layer is not a float based, so this would be stored as it is. And we can retrieve it by the layer_name.pkl
                model_structure[layer_name] = [2, weights_numpy.shape]
                model_numpy_size_total += os.path.getsize(non_fl_layer_file_item)
        pbar.close()
        if len(model_weights_flatten_f16) > 0:
            model_weights_flatten_f16_file = saving_weights_fl_flatten(model_weights_flatten_f16, fl_weights_file_path, "f16_")
            model_weights_size = os.path.getsize(model_weights_flatten_f16_file)
            model_numpy_size_total += model_weights_size
        if len(model_weights_flatten_f32) > 0:
            model_weights_flatten_f32_file = saving_weights_fl_flatten(model_weights_flatten_f32, fl_weights_file_path, "f32_")
            model_weights_size = os.path.getsize(model_weights_flatten_f32_file)
            model_numpy_size_total += model_weights_size
        if len(model_weights_flatten_f64) > 0:
            model_weights_flatten_f64_file = saving_weights_fl_flatten(model_weights_flatten_f64, fl_weights_file_path, "f64_")
            model_weights_size = os.path.getsize(model_weights_flatten_f64_file)
            model_numpy_size_total += model_weights_size

        model_structure_file = model_struct_weights_folder_individual+"model_structure.pkl"
        if not os.path.exists(model_structure_file):
            #print("model structure sealing:", model_structure_file)
            seal_pickle(model_structure_file, model_structure)


def adding_para_to_list(model_weights_flatten_list, weights_numpy):
    weights_numpy = weights_numpy.flatten()
    for para in weights_numpy:
        model_weights_flatten_list.append(para)

def saving_weights_fl_flatten(model_weights_flatten_list, fl_weights_file_path, dtype):
    model_weights_flatten = np.array(model_weights_flatten_list)
    para_len = len(model_weights_flatten)
    #model_weights_flatten_file = fl_weights_file_path + dtype + str(para_len)+".pkl"
    model_weights_flatten_file = fl_weights_file_path + dtype + str(para_len)+".bin"
    if not os.path.exists(model_weights_flatten_file):
        #print("model weights sealing:", model_weights_flatten_file)
        #seal_pickle(model_weights_flatten_file, model_weights_flatten)
        
        # so there using tofile to save list array to binary;
        # Write to binary file
        with open(model_weights_flatten_file, 'wb') as f:
            model_weights_flatten.tofile(f)

    #print("model_weights_flatten.shape:", model_weights_flatten.shape, para_len)
    return model_weights_flatten_file


