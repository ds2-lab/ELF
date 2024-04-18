import os
from tqdm import tqdm
import hashlib
import sys
import torch
import pickle
from Utils.utils import *
from Utils.config import *

def hash_deduplication(model_path_list):
    layer_hash_repeat_value_set_file = dup_layer_folder+"hash_layer_repeat_set.pkl"
    if os.path.exists(layer_hash_repeat_value_set_file):
        print("file", layer_hash_repeat_value_set_file, "exists, layer hash deduplication finished.")
        return

    layer_hash_value_set = set()
    layer_hash_repeat_value_set = set()

    cnt = 0
    model_size_total = 0
    model_hash_dedup_total = 0
    total_layer_num = 0
    hash_dedup_num = 0

    for model_path in model_path_list:
        model_name = model_path.split('/')[-2]
        #model_path = model_original_folder+model_name+"/pytorch_model.bin"
        model_size = os.path.getsize(model_path)
        print("\n", cnt, "Model:", model_name, str(round(model_size/MB, 2))+" MB")
        cnt += 1
        model = model_loading_fun(model_path)
        same_storage_dict = get_shared_storage_tensor_dict(model)
 
        try:
            model_size_total += model_size
            total_layer_num += len(model)
            pbar = tqdm(total=len(model))
            for layer_name in model:
                pbar.update(1)
                if layer_name in same_storage_dict:
                    continue

                layer = model[layer_name]
                weights_numpy = layer.numpy()
                md5_hash = hashlib.md5(weights_numpy.tobytes()).hexdigest()
                if md5_hash not in layer_hash_value_set:
                    layer_hash_value_set.add(md5_hash)
                    continue
                if md5_hash not in layer_hash_repeat_value_set:
                    layer_hash_repeat_value_set.add(md5_hash)
                    layer_hash_file = dup_layer_folder+md5_hash+".pkl"
                    seal_pickle(layer_hash_file, weights_numpy)
                model_hash_dedup_total += weights_numpy.nbytes
            pbar.close()
        except ExceptionType as e:
            # Code that runs if an exception of ExceptionType occurs
            print(f"An error occurred: {e}")
            sys.exit()
            #model_path_not_working_set.add(model_path)
    
    seal_pickle(layer_hash_repeat_value_set_file, layer_hash_repeat_value_set)
    hash_dedup_num = len(layer_hash_repeat_value_set)
    print("~~~~~ Hash Deduplication (HD) Summary ~~~~~")
    print("total_layer_num :", total_layer_num)
    print("hash_dedup_num  :", hash_dedup_num)
    print("model_total_size:", model_size_total/MB, "MB")
    print("hash_dedup_size :", model_hash_dedup_total/MB, "MB")
    print("HD Compression Ratio:", model_size_total/(model_size_total-model_hash_dedup_total))
    
