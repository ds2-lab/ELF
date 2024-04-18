import time
import pickle
import os
import sys
import math
import numpy as np
import torch
import shutil
import struct
from collections import OrderedDict
from tqdm import tqdm
import tarfile
from queue import PriorityQueue
import zstandard as zstd
import gzip
import hashlib
from Utils.config import *
from Utils.utils import folder_making_fun, get_folder_size, rounding, unseal_pickle, get_shared_storage_tensor_dict
from Utils.model_downloading import model_downloading 
from hashing_deduplication import hash_deduplication
from model_structure import save_model_structure_and_flatten_weights
from distance_encoding import distance_reference
from exponentless_floating import exponential_dedup

def main():
    folder_making()
    print("\n\n~~~~~~~~~~~~~~~~~~~~ Model Downloading ~~~~~~~~~~~~~~~~~~~~")
    model_downloading(model_name_list)
    model_path_list = get_model_path_list()

    print("\n\n~~~~~~~~~~~~~~~~~~~~ ELVES Starts ~~~~~~~~~~~~~~~~~~~~")
    print("\n~~~~ 1+1a. Hash Deduplicating (HD) ~~~~")
    hash_deduplication(model_path_list)
    print("\n~~~~ 1b+1c. Model Structures Saving ~~~~")
    save_model_structure_and_flatten_weights(model_path_list)
    model_weights_path_list = get_weigths_path_list(model_elves_compression)
    print("\n~~~~ 2a. Distance Encoding (DE) ~~~~")
    cmp_DE(model_weights_path_list)
    print("\n~~~~ 2b. Exponent-Less Floating (ELF) ~~~~")
    cmp_ELF(model_weights_path_list)
    print("\n~~~~ 3. zstd compression ~~~~")
    cmp_zstd(model_path_list, model_elves_compression)

    print("\n\n~~~~~~~~~~~~~~~~~~~~ ELVES Compression Overview ~~~~~~~~~~~~~~~~~~~~")
    cmp_overview(model_path_list)

    '''
    model_elf_cmp_folder_list = get_elf_cmp_folder_list(model_elves_compression)
    print("\n~~~~ Exponent-Less Floating (ELF) Decompression ~~~~~")
    decmp_ELF(model_elf_cmp_folder_list)
    '''


def folder_making():
    folder_making_fun(model_original_folder)
    folder_making_fun(model_compressed_folder)
    folder_making_fun(model_decompressed_folder)
    folder_making_fun(dup_layer_folder)

def get_model_path_list():
    model_path_list = list()
    for model_name in os.listdir(model_original_folder):
        model_folder = os.path.join(model_original_folder, model_name)
        for model_file in os.listdir(model_folder):
            model_path = os.path.join(model_folder, model_file)
            if os.path.isfile(model_path):
                model_path_list.append(model_path)
    return model_path_list

def get_weigths_path_list(model_cmp_structure_weights_folder):
    model_weights_file_list = list()
    for model_name in os.listdir(model_cmp_structure_weights_folder):
        model_path = os.path.join(model_cmp_structure_weights_folder, model_name)
        for model_file in os.listdir(model_path):
            if model_file == "fl_weights":
                model_weights_folder = os.path.join(model_path, model_file)
                for model_weights_file in os.listdir(model_weights_folder):
                    model_weights_file_path = os.path.join(model_weights_folder, model_weights_file)
                    model_weights_file_list.append(model_weights_file_path)
    return model_weights_file_list

def get_elf_cmp_folder_list(model_elves_compression):
    model_elf_cmp_folder_list = list()
    for model_name in os.listdir(model_elves_compression):
        model_path = os.path.join(model_elves_compression, model_name)
        for model_file in os.listdir(model_path):
            if model_file == "exponential_dedup":
                model_elf_folder = model_path+"/"+model_file+"/"
                model_elf_cmp_folder_list.append(model_elf_folder)
    return model_elf_cmp_folder_list

def cmp_overview(model_path_list):
    #print("\n~~~~~~~~~~ Compression Evaluation Summary ~~~~~~~~~~")
    model_hd_size_dict = get_repeated_hash_layer_size()
    #delete_folder(dup_layer_folder)
    cnt = 0
    org_total = 0
    elves_total = 0
    #gzip_total = 0
    #zstd_total = 0
    for model_name in model_name_list:
        model_path = model_original_folder+model_name+"/pytorch_model.bin"
        model_size_org = os.path.getsize(model_path)
        org_total += model_size_org

        model_elves_path = model_elves_compression+model_name+"/pytorch_model.tar.zst"
        model_size_elves = os.path.getsize(model_elves_path) + model_hd_size_dict[model_name] 
        elves_total += model_size_elves

        print("\n", cnt, model_name, "Original Size:", rounding(model_size_org/MB), "MB.  ELVES Size:", rounding(model_size_elves/MB), "MB.  Compression Ratio:", rounding(model_size_org/model_size_elves))
        cnt += 1

    print("\n\n~~~~~~~~~~ Overall Compression Ratio: ~~~~~~~~~~")
    print("ELVES:", rounding(org_total/elves_total))

def cmp_ELF(model_weights_path_list):
    cnt = 0
    for model_weights_file in model_weights_path_list:
        print("\n", cnt, model_weights_file)
        cnt += 1
        exponential_dedup([model_weights_file])
        #sys.exit()

def cmp_DE(model_weights_path_list):
    cnt = 0
    for model_weights_file in model_weights_path_list:
        print("\n", cnt, model_weights_file)
        #print(cnt, model_weights_file, os.path.getsize(model_weights_file)/MB, "MB.")
        cnt += 1 
        distance_reference([model_weights_file])
        #sys.exit()

def decmp_ELF(model_elf_cmp_folder_list):
    cnt = 0
    for model_elf_cmp_folder in model_elf_cmp_folder_list:
        print('\n', cnt, model_elf_cmp_folder)
        cnt += 1
        elf_decompression(model_elf_cmp_folder)
        #sys.exit()

def cmp_zstd(model_path_list, model_elves_compression):
    model_name_path_size_dict = dict()
    for model_path in model_path_list:
        model_name = model_path.split('/')[-2]
        model_name_path_size_dict[model_name] = os.path.getsize(model_path)

    cnt = 0
    for model_name in os.listdir(model_elves_compression):
        model_name_folder = model_elves_compression + model_name + '/'
        model_org_size = model_name_path_size_dict[model_name] 
        model_zstd_path = model_name_folder+"pytorch_model.tar.zst"
        
        if os.path.exists(model_zstd_path):
            print(cnt, model_name, "weights compressed to", model_zstd_path)
            cnt += 1
            continue

        zstd_new_source_folder = model_name_folder+"zstd_source/"
        folder_making_fun(zstd_new_source_folder)

        # .tar is the intermediate file
        model_tar_path  = model_name_folder+"pytorch_model.tar"
        model_structure_path = model_name_folder+"model_structure.pkl"

        non_float_layer_folder = model_name_folder+"non_fl_layers/"
        if os.path.exists(non_float_layer_folder):
            shutil.copytree(non_float_layer_folder, os.path.join(zstd_new_source_folder, non_float_layer_folder), dirs_exist_ok=True)
            #print(cnt, model_name)
            #break
        #continue

        weights_folder = model_name_folder+"fl_weights/"
        model_size_weights = get_folder_size(weights_folder)
        de_folder = model_name_folder+"distance_encoding/"
        model_size_de = get_folder_size(de_folder)
        elf_folder = model_name_folder+"exponential_dedup/"
        model_size_elf = get_folder_size(elf_folder)
        
        model_inter_cmp_size_min = min(model_org_size, model_size_weights, model_size_de, model_size_elf)
        # all hash deduped
        if model_size_weights == 0 or model_size_weights == model_inter_cmp_size_min or (not os.path.exists(de_folder) and not os.path.exists(elf_folder)):
            zstd_source_folder = weights_folder
            print("~"*40, "weights_folder", "~"*40)
        elif model_size_de == model_inter_cmp_size_min:
            zstd_source_folder = de_folder
            print("~"*40, "de_folder", "~"*40)
        elif model_size_elf == model_inter_cmp_size_min:
            zstd_source_folder = elf_folder
        else:
            print("model size calculating error.")
            sys.exit()
        
        model_zstd_structure_path = zstd_source_folder+"model_structure.pkl"
        shutil.copy(model_structure_path, model_zstd_structure_path)
        compress_folder(zstd_source_folder, model_tar_path)
        print(cnt, model_name, "weights compressed to", model_zstd_path)
        #model_zstd_cmp_size = os.path.getsize(model_zstd_path)
        #print(model_zstd_path, rounding(model_zstd_cmp_size/MB), ", CR:", rounding(model_org_size/model_zstd_cmp_size), '\n')
        cnt += 1
        #sys.exit()

        #all the intermediate folder & files should be deleted in the end.
        #delete_folder(de_folder)
        

def ELVES(model_elves_compression):
    print("\n~~~~~~~~~~ ELVES Decompressing ~~~~~~~~~~")
    cnt = 0
    for model_name in model_name_list:
        cnt += 1
        print("Model:", model_name, " Decompressing...")
        model_path = model_original_folder+model_name+"/pytorch_model.bin"
        model_decmp_folder = model_decompressed_folder + model_name + "/"
        folder_making_fun(model_decmp_folder)
        model_decmp_path = model_decmp_folder+"pytorch_model_decmp.bin"
        if os.path.exists(model_decmp_path):
            continue
        if model_decompression_dict[model_name] != model_elves_compression+model_name+"/exponential_dedup/":
            shutil.copy(model_path, model_decmp_path)
        else:
            exponent_decompression(model_name, model_decmp_folder, model_decmp_path)

        delete_folder(model_decompression_dict[model_name])
        delete_file(model_elves_compression+model_name+"/model_structure.pkl")
        non_fl_folder = model_elves_compression+model_name+"/non_fl_layers"
        if os.path.exists(non_fl_folder):
            delete_folder(non_fl_folder)
        
        '''
        source_folder = model_compressed_folder+"dup_layer_folder"
        output_file = model_compressed_folder+"dup_layer_folder.tar"
        compress_folder(source_folder, output_file)
        '''

# for compression 
def compress_folder(source_folder, output_file):
    # Create a tar archive of the folder
    with tarfile.open(output_file, 'w') as tar:
        tar.add(source_folder, arcname=os.path.basename(source_folder))

    # Compress the tar archive using zstd
    cctx = zstd.ZstdCompressor(level=3)  # Set the compression level as needed
    with open(output_file, 'rb') as tar_file:
        with open(f"{output_file}.zst", 'wb') as compressed_file:
            compressed_file.write(cctx.compress(tar_file.read()))

    # Remove the uncompressed tar archive
    os.remove(output_file)


def get_repeated_hash_layer_size():
    layer_hash_repeat_value_set_file = dup_layer_folder + "hash_layer_repeat_set.pkl"
    layer_hash_repeat_value_set = unseal_pickle(layer_hash_repeat_value_set_file)

    hash_dedup_size_total = 0
    layer_hash_repeat_value_amortized_size_dict = dict()
    for hash_value in layer_hash_repeat_value_set:
        layer_size = os.path.getsize(dup_layer_folder+hash_value+".pkl")
        layer_hash_repeat_value_amortized_size_dict[hash_value] = {"layer_size":layer_size, "cnt":0}
        hash_dedup_size_total += layer_size

    for model_name in model_name_list:
        model_path = model_original_folder+model_name+"/pytorch_model.bin"
        try:
            model = torch.load(model_path, map_location='cpu')
        except Exception as e:
            print(model_path, "model load unseccessful!")
            sys.exit()

        same_storage_dict = get_shared_storage_tensor_dict(model)
        for layer_name in model:
            if layer_name in same_storage_dict:
                continue
            layer = model[layer_name]
            weights_numpy = layer.numpy()
            md5_hash = hashlib.md5(weights_numpy.tobytes()).hexdigest()
            if md5_hash in layer_hash_repeat_value_amortized_size_dict:
                layer_hash_repeat_value_amortized_size_dict[md5_hash]["cnt"] += 1
    for layer_hash in layer_hash_repeat_value_amortized_size_dict:
        layer_hash_repeat_value_amortized_size_dict[layer_hash]["avg_size"] = layer_hash_repeat_value_amortized_size_dict[hash_value]["layer_size"] / layer_hash_repeat_value_amortized_size_dict[hash_value]["cnt"]
    model_hd_size_dict = dict()
    for model_name in model_name_list:
        model_hd_size_dict[model_name] = 0
        model_path = model_original_folder+model_name+"/pytorch_model.bin"
        try:
            model = torch.load(model_path, map_location='cpu')
        except Exception as e:
            print(model_path, "model load unseccessful!")
            sys.exit()

        same_storage_dict = get_shared_storage_tensor_dict(model)
        for layer_name in model:
            if layer_name in same_storage_dict:
                continue
            layer = model[layer_name]
            weights_numpy = layer.numpy()
            md5_hash = hashlib.md5(weights_numpy.tobytes()).hexdigest()
            if md5_hash in layer_hash_repeat_value_amortized_size_dict:
                model_hd_size_dict[model_name] += layer_hash_repeat_value_amortized_size_dict[md5_hash]["avg_size"]
    return model_hd_size_dict


if __name__=='__main__':
    total_start = time.time()
    main()
    total_end = time.time()
    print("\nTotal running time:", round((total_end - total_start)/60,2), "mins")
