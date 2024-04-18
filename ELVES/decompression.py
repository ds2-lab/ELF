import os
import math
import numpy as np
import torch
import shutil
import struct
import pickle
from collections import OrderedDict
from tqdm import tqdm
import tarfile
from queue import PriorityQueue
import sys
from Utils.utils import *
from Utils.config import *
import subprocess

# C++ version
def elf_decompression(model_elf_cmp_folder):
    #print("~~~~ decompressing ", model_elf_cmp_folder)
    model_name = model_elf_cmp_folder.split('/')[2]
    model_recovered_path = model_decompressed_folder+model_name+"/pytorch_model_decmp.bin"
    if os.path.exists(model_recovered_path):
        print(model_recovered_path, " exists.")
        return
    weights_path = os.path.dirname(os.path.dirname(model_elf_cmp_folder))+"/"+"fl_weights/"
    for para_type in os.listdir(model_elf_cmp_folder):
        elf_pthread = ["./elf_pthread", "-d"]
        model_elf_cmp_para_folder = model_elf_cmp_folder+para_type+"/"
        elf_pthread += ["-i", model_elf_cmp_para_folder]
        weights_dtype = para_type
        elf_pthread += ["-p", weights_dtype]
        output_folder = model_decompressed_folder+model_name+"/"
        folder_making_fun(output_folder)
        elf_pthread += ["-o", output_folder]
        if weights_dtype != "f32":
            for weights_file_item in os.listdir(weights_path):
                if weights_file_item.split('/')[-1].split('_')[0] == "f32":
                    continue
                org_weight_file_path = weights_path+weights_file_item
                new_weight_file_path = output_folder+"decmp_"+weights_dtype+".bin"
                print("from ", org_weight_file_path, "to", new_weight_file_path)
                shutil.copy(org_weight_file_path, new_weight_file_path)
            continue
        # this number should be on the elf cmped type_folder_num
        if not os.path.exists(weights_path):
            print(weights_path, "not exists.")
            sys.exit()
        for weights_file_item in os.listdir(weights_path):
            if weights_dtype == weights_file_item[:3]:
                weights_num = weights_file_item.split('_')[-1].split('.')[0]
        elf_pthread += ["-n", weights_num]
        print("elf_pthread:", elf_pthread)
        #sys.exit()
        try:
            result = subprocess.run(elf_pthread, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("ELF:", result.stdout, end="")
        except subprocess.CalledProcessError as e:
            print("Error occurred:", e.stderr)
    
    print("~~~~~~~~ weights error check ~~~~~~~~")
    model_weights_flatten_f16 = list()
    model_weights_flatten_f32 = list()
    model_weights_flatten_f64 = list()
    for weights_file_item in os.listdir(weights_path):
        weights_dtype = weights_file_item[:3]
        print("~~~~", weights_dtype, "~~~~")
        org_weight_file = weights_path+weights_file_item
        new_weight_file = model_decompressed_folder+model_name+"/"+"decmp_"+weights_dtype+".bin"
        if weights_dtype == "f16":
            model_weights_flatten_f16 = np.fromfile(new_weight_file, dtype=np.float16)
            org_model_weights_flatten_f16 = np.fromfile(org_weight_file, dtype=np.float16)
            #max_err_abs = weights_error_comparison(model_weights_flatten_f16, org_model_weights_flatten_f16)
            #print("max_err_abs:", max_err_abs)
            #if max_err_abs > (2 ** -11):
                #print("error too high.")
                #sys.exit()
        elif weights_dtype == "f32":
            model_weights_flatten_f32 = np.fromfile(new_weight_file, dtype=np.float32)
            org_model_weights_flatten_f32 = np.fromfile(org_weight_file, dtype=np.float32)
            #max_err_abs = weights_error_comparison(model_weights_flatten_f32, org_model_weights_flatten_f32)
            #print("max_err_abs:", max_err_abs)
            #if max_err_abs > (2 ** -24):
                #print("error too high.")
                #sys.exit()
        else:
            model_weights_flatten_f64 = np.fromfile(new_weight_file, dtype=np.float64)
            org_model_weights_flatten_f64 = np.fromfile(org_weight_file, dtype=np.float64)
            #max_err_abs = weights_error_comparison(model_weights_flatten_f64, org_model_weights_flatten_f64)
            #print("max_err_abs:", max_err_abs)
            #if max_err_abs > (2 ** -53):
                #print("error too high.")
                #sys.exit()
        os.remove(new_weight_file)
    model_structure_file = os.path.dirname(os.path.dirname(model_elf_cmp_folder))+"/"+"model_structure.pkl" 
    model_structure = unseal_pickle(model_structure_file)
    model_cmp_folder = os.path.dirname(os.path.dirname(model_elf_cmp_folder))
    try:
        print("fl weights info: len(model_weights_flatten_f16, 32, 64):", len(model_weights_flatten_f16), len(model_weights_flatten_f32), len(model_weights_flatten_f64))
        model_recovered = model_recovery_structure_weights(model_cmp_folder, model_structure, model_weights_flatten_f16, model_weights_flatten_f32, model_weights_flatten_f64)
    except Exception as e:
        print(f"An error occurred: {e}")
        shutil.rmtree(model_decompressed_folder+model_name)
        print("Decompression Failed.", "*" * 100)
        return
    model_recovered_path = model_decompressed_folder+model_name+"/pytorch_model_decmp.bin"
    torch.save(model_recovered, model_recovered_path)
    model_original_folder = "/mnt/samsng_smrt/elf_revision/models/"+model_name+"/"
    for model_org_item in os.listdir(model_original_folder):
        model_original_path = model_original_folder+model_org_item
    model_comparison_exponential(model_original_path, model_recovered_path, model_structure_file)
    print("Decompressed Model Saved.\n")

# python version
def exponent_decompression(model_name, model_decmp_folder, model_decmp_path):
    #print("\n", model_name," model_recovery_from_exponential...")
    model_recovered_folder = model_decmp_folder
    model_recovered_path = model_decmp_path
    model_original_path = model_original_folder+model_name+"/pytorch_model.bin"
    model_structure_file_path = model_elves_compression+model_name+"/model_structure.pkl"

    model_weights_flatten_f16 = list()
    model_weights_flatten_f32 = list()
    model_weights_flatten_f64 = list()
    model_structure = OrderedDict()

    model_cmp_folder = os.path.join(model_elves_compression, model_name)
    for model_item in os.listdir(model_cmp_folder):
        model_item_path = os.path.join(model_cmp_folder, model_item)
        if model_item == "exponential_dedup":

            # for c++ version
            if model_name == "albert-base-v2":
                para_name = 11685122
            elif model_name == "sentence-transformers_all-MiniLM-L6-v2":
                para_name = 22713216
            else:
                # bug to be fixed.
                sys.exit()
            elf_pthread = ["./elf_pthread", "-d", "-i", model_item_path+"/", "-p", "f32", "-n", str(para_name)]
            #print("elf_pthread:", elf_pthread)
            
            try:
                result = subprocess.run(elf_pthread, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
                print("ELF:", result.stdout, end="")
            except subprocess.CalledProcessError as e:
                print("Error occurred:", e.stderr)
                
            #print("model_cmp_folder:", model_cmp_folder) 
            model_weights_flatten_f32_file = model_item_path+"/decmp.bin"
            model_weights_flatten_f32 = np.fromfile(model_weights_flatten_f32_file, dtype=np.float32)
            os.remove(model_weights_flatten_f32_file)
            model_weights_flatten_f64 = []
            model_weights_flatten_f16 = []

            # for python version
            '''
            for model_float_weights_folder in os.listdir(model_item_path):
                model_float_weights_folder_path = os.path.join(model_item_path, model_float_weights_folder)
                for model_float_weights_exponential_file in os.listdir(model_float_weights_folder_path):
                    exponential_file_path = os.path.join(model_float_weights_folder_path, model_float_weights_exponential_file)
                    if model_float_weights_exponential_file == "exponential_over_para_file.pkl":
                        over_para_list = unseal_pickle(exponential_file_path)
                    elif model_float_weights_exponential_file == "exponential_over_position_file.pkl":
                        over_position_list = unseal_pickle(exponential_file_path)
                    elif model_float_weights_exponential_file == "exponential_within_para_file.pkl":
                        within_para_list = unseal_pickle(exponential_file_path)
                    elif model_float_weights_exponential_file == "exponential_within_str_left_file.pkl":
                        within_para_str = unseal_pickle(exponential_file_path)
                    else:
                        print("Error file type in model_recovery_from_exponential.")
                        sys.exit()
                if model_float_weights_folder == "f16":
                    model_weights_flatten_f16 = weights_recovery_from_exponential(np.float16, over_para_list, over_position_list, within_para_str, within_para_list)
                elif model_float_weights_folder == "f32":
                    model_weights_flatten_f32 = weights_recovery_from_exponential(np.float32, over_para_list, over_position_list, within_para_str, within_para_list)
                elif model_float_weights_folder == "f64":
                    model_weights_flatten_f64 = weights_recovery_from_exponential(np.float64, over_para_list, over_position_list, within_para_str, within_para_list)
                else:
                    print("Error folder in model_recovery_from_exponential for fxx folder.")
                    sys.exit()
                #print(model_float_weights_folder_path)
            '''

        elif model_item == "model_structure.pkl":
            model_structure = unseal_pickle(model_item_path)
    #print("len model_structure:", len(model_structure))
    #print("len model_weights_flatten_f16:", len(model_weights_flatten_f16))
    #print("len model_weights_flatten_f32:", len(model_weights_flatten_f32))
    #print("len model_weights_flatten_f64:", len(model_weights_flatten_f64), "\n")

    #model_recovered = model_recovery_structure_weights(model_item_path, model_structure, model_weights_flatten_f16, model_weights_flatten_f32, model_weights_flatten_f64)
    model_recovered = model_recovery_structure_weights(model_cmp_folder, model_structure, model_weights_flatten_f16, model_weights_flatten_f32, model_weights_flatten_f64)
    torch.save(model_recovered, model_recovered_path)
    #model_comparison_exponential(model_original_path, model_recovered_path, model_structure_file_path)
    print("Decompressed Model Saved.\n")



def weights_recovery_from_exponential(weights_dtype_np, over_para_list, over_position_list, within_para_str, within_para_list):
    #model_para_list_recover = np.array(list(), dtype=weights_dtype_np)
    model_para_list_recover = list()
    exp_decoding(weights_dtype_np, within_para_list, within_para_str, model_para_list_recover)
    model_para_list_recover = para_decoding(over_position_list, over_para_list, model_para_list_recover)
    model_para_list_recover = weights_dtype_np(model_para_list_recover)
    return model_para_list_recover

def exp_decoding(weights_dtype_np, table2_para_list, table2_para_str_save, para_list):
    #print("~~~~ Exponential Decoding Starts ~~~~")
    table2_para_str = ""
    pbar = tqdm(total=len(table2_para_list))
    for num in table2_para_list:
        #print("num in within_para_list:", num)
        # convert int64 into binary '064b'
        table2_para_str += format(num, '064b')
        table2_para_str = bits_update_para(weights_dtype_np, para_list, table2_para_str)
        pbar.update(1)
    pbar.close()
    #print("len table2_para_str:", len(table2_para_str), " len table2_para_str_save:", len(table2_para_str_save))
    table2_para_str += table2_para_str_save
    bit_str_left = bits_update_para(weights_dtype_np, para_list, table2_para_str)

    if len(bit_str_left) >= 1:
        print("~~~~~~~~ Bit Str Left!!! ~~~~~~~~")
        sys.exit()
    #print("~~~~ Exponential Decoding Completed ~~~~")


def bits_update_para(weights_dtype_np, para_list, bit_str):
    if weights_dtype_np == np.float16:
        decimal_bit_sign = 10 + 1
        sign_exponential_str = "0"+"01111"
    elif weights_dtype_np == np.float32:
        decimal_bit_sign = 23 + 1
        sign_exponential_str = "0"+"01111111"
    elif weights_dtype_np == np.float64:
        decimal_bit_sign = 52 + 1
        sign_exponential_str = "0"+"01111111111"
    else:
        print("~~~~ Non clear dtype in bits_update_para ~~~~")
        sys.exit()

    while len(bit_str) >= decimal_bit_sign:
        sign = bit_str[decimal_bit_sign-1]
        conv_str = sign_exponential_str + bit_str[:decimal_bit_sign-1]
        bit_str = bit_str[decimal_bit_sign:]
        para = binary_to_float(conv_str, weights_dtype_np)
        para = para - 1.0
        if sign == '1':
            para = weights_dtype_np(para*(-1))
        para_list.append(para)
    return bit_str

def binary_to_float(binary_representation, weights_dtype_np):
    if weights_dtype_np == np.float16:
        pack_arg = "H"
    elif weights_dtype_np == np.float32:
        pack_arg = "I"
    elif weights_dtype_np == np.float64:
        pack_arg = "L"
    else:
        print("~~~~ Non clear dtype in binary_to_float ~~~~")
        sys.exit()
    buffer_pack = struct.pack(pack_arg, int(binary_representation, 2))
    float_value = np.frombuffer(buffer_pack, dtype=weights_dtype_np)[0]
    return float_value

def para_decoding(table1_pos, table1_para, para_list_recover):
    #print("~~~~ para_decoding starts ~~~~")

    #print("melloc for para_list_recovered...")
    para_list_recover_new = [0] * (len(table1_pos)+len(para_list_recover))
    pbar = tqdm(total=len(table1_pos))
    for i in range(len(table1_pos)):
        if abs(table1_para[i]) < 0.99:
            print(i, table1_para[i], "in the uncompressed table/")
            sys.exit()
        para_list_recover_new[table1_pos[i]] = table1_para[i]
        pbar.update(1)
    pbar.close()
    cnt = 0
    pbar = tqdm(total=len(para_list_recover_new))
    for i in range(len(para_list_recover_new)):
        # all the 0s are shold be replaced by para_list_recover items
        if abs(para_list_recover_new[i]) < 0.001:
            para_list_recover_new[i] = para_list_recover[cnt]
            cnt += 1
        pbar.update(1)
    pbar.close()
    if cnt != len(para_list_recover):
        print("para_list_recover is not done.")
        sys.exit()
    #print("~~~~ para_decoding finished ~~~~")
    return para_list_recover_new

def model_recovery_structure_weights(model_cmp_folder, model_structure, model_weights_flatten_f16, model_weights_flatten_f32, model_weights_flatten_f64):
    model_recovered = OrderedDict()
    index_f16 = 0
    index_f32 = 0
    index_f64 = 0
    for layer_name in model_structure:
        layer_info_list = model_structure[layer_name]
        if layer_info_list[0] == 0:
            model_recovered[layer_name] = model_recovered[layer_info_list[1]]
            #print("0, inside sharing tensor.")
        elif layer_info_list[0] == 1:
            layer_hashed_file = dup_layer_folder+layer_info_list[1]+".pkl"
            layer_hashed = unseal_pickle(layer_hashed_file)
            model_recovered[layer_name] = torch.from_numpy(layer_hashed.reshape(layer_info_list[2]))
            #print("1, layer hashed tensor.")
        elif layer_info_list[0] == 2:
            non_fl_layer_file = model_cmp_folder+"/non_fl_layers/"+layer_name+".pkl"
            non_fl_layer = unseal_pickle(non_fl_layer_file)
            model_recovered[layer_name] = torch.from_numpy(non_fl_layer.reshape(layer_info_list[1]))
            #print("2, non float weights tensor.")
        else:
            weight_num = reshape_to_flat(layer_info_list[1])
            if layer_info_list[0] == 16:
                model_recovered[layer_name] = torch.from_numpy(np.array(model_weights_flatten_f16[index_f16:index_f16+weight_num]).reshape(layer_info_list[1]))
                index_f16 += weight_num
                #print("16, float16 weights tensor.")
            elif layer_info_list[0] == 32:
                model_recovered[layer_name] = torch.from_numpy(np.array(model_weights_flatten_f32[index_f32:index_f32+weight_num]).reshape(layer_info_list[1]))
                index_f32 += weight_num
                #print("32, float32 weights tensor.")
            elif layer_info_list[0] == 64:
                model_recovered[layer_name] = torch.from_numpy(np.array(model_weights_flatten_f64[index_f64:index_f64+weight_num]).reshape(layer_info_list[1]))
                index_f64 += weight_num
                #print("64, float64 weights tensor.")
            else:
                print("error with the structure code.")
    return model_recovered

def model_comparison_exponential(model_path, model_recovered_path, model_structure_file_path):
    print("~~~ Decompressed Model Validating... ~~~")
    print("original path:", model_path, os.path.getsize(model_path)/MB)
    print("recovery path:", model_recovered_path, os.path.getsize(model_recovered_path)/MB)
    print("model_structure_file_path:", model_structure_file_path)

    model_comparison_shape(model_path, model_recovered_path, model_structure_file_path)
    
    return 0
    
    model = model_loading_fun(model_path)
    model_recovered = model_loading_fun(model_recovered_path)

    #print("\nforward comparision")
    for layer_name in model:
        weights_np_flt = model[layer_name].numpy().flatten()
        weights_np_flt_recovered = model_recovered[layer_name].numpy().flatten()
        err_max_forward = weights_error_comparison(weights_np_flt, weights_np_flt_recovered)
    #print("\nbackward comparision")
    for layer_name in reversed(model_recovered):
        weights_np_flt = model[layer_name].numpy().flatten()
        weights_np_flt_recovered = model_recovered[layer_name].numpy().flatten()
        err_max_back = weights_error_comparison(weights_np_flt, weights_np_flt_recovered)
    return max(err_max_forward, err_max_back)

def weights_error_comparison(model_para_list, model_para_list_recover):
    #print("~~~~ encoding vs decoding ~~~~")
    #print("original length:", len(model_para_list), "decoding length:", len(model_para_list_recover))
    if len(model_para_list) != len(model_para_list_recover):
        print("Model Parameter Number Mismatch!!!!!")
        sys.exit()

    #print("~~~~ err info calculation starts ~~~~")
    total_err = 0
    same_para_num = 0
    err_max = PriorityQueue()
    err_abs_max = -1
    pbar = tqdm(total=len(model_para_list))
    for i in range(len(model_para_list)):
        pbar.update(1)
        if model_para_list[i] == model_para_list_recover[i]:
            same_para_num += 1
            continue

        if math.isnan(model_para_list[i]):
            print("model_para_list is nan with index:", i)
            sys.exit()
        if math.isnan(model_para_list_recover[i]):
            print("model_para_list_recover is nan with index:", i)
            sys.exit()

        err = abs(model_para_list[i]-model_para_list_recover[i])
        total_err += err
        err_abs_max = max(err_abs_max, err)
        err_max.put((err, i, model_para_list[i], model_para_list_recover[i]))
        if err_max.qsize() >= 5:
            err_max.get()
    pbar.close()

    while not err_max.empty():
        err_max_item = err_max.get()
        print("err_max data =", err_max_item)

    print("same ratio:", same_para_num/len(model_para_list))
    return err_abs_max

def reshape_to_flat(arr_shape):
    res = 1
    for item in arr_shape:
        res *= item
    return res


def model_comparison_shape(model_original_path, model_decompressed_path, model_strucutre_path):
    print("~~~~ model_comparison_shape ~~~~")
    model_strucutre = unseal_pickle(model_strucutre_path)
    model = model_loading_fun(model_original_path)
    model_recovered = model_loading_fun(model_decompressed_path)

    shape_flg = True

    print("\nforward comparison...")
    for layer_name in model:
        layer_org_shape = model[layer_name].shape
        layer_decmp_shape = model_recovered[layer_name].shape
        layer_structure = model_strucutre[layer_name]
        #print("layer_org_shape:", layer_org_shape, "    layer_decmp_shape:", layer_decmp_shape, "    layer_structure:", layer_structure)
        if layer_org_shape != layer_decmp_shape:
            model_recovered[layer_name] = model_recovered[layer_name].reshape(layer_org_shape)
            print("\n", layer_name, " layer shape inconsistency!")
            print("layer_org_shape:", layer_org_shape, "    layer_decmp_shape:", layer_decmp_shape, "    layer_structure:", layer_structure)
            print("After reshaping:", model_recovered[layer_name].shape)
            sys.exit()

    print("\nbackward comparison...")
    for layer_name in model_recovered:
        layer_org_shape = model[layer_name].shape
        layer_decmp_shape = model_recovered[layer_name].shape
        layer_structure = model_strucutre[layer_name]
        #print("layer_org_shape:", layer_org_shape, "    layer_decmp_shape:", layer_decmp_shape, "    layer_structure:", layer_structure)
        if layer_org_shape != layer_decmp_shape:
            print(layer_name, " layer shape inconsistency!")
            sys.exit()

    if shape_flg == True:
        print("models have same layer shape.")
        return

    print("Shape currection done.")
    '''
    model_name = model_original_path.split('/')[1]
    model_recovered_reshape_folder = model_decmp_reshape_folder+model_name+"/"
    if not os.path.exists(os.path.dirname(model_recovered_reshape_folder)):
        print("Making dir:", model_recovered_reshape_folder)
        os.makedirs(os.path.dirname(model_recovered_reshape_folder))
    model_recovered_reshape_path = model_recovered_reshape_folder+"/pytorch_model_recovered.bin"
    torch.save(model_recovered,model_recovered_reshape_path)
    '''
