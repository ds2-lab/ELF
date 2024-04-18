import pickle
import os
import numpy as np
from tqdm import tqdm
import sys
from Utils.utils import *
from Utils.config import *
import subprocess

def exponential_dedup(model_weights_path_list):
    cnt = 0
    for model_weights_file in model_weights_path_list:
        # C++ version ELF
        #print("model_weights_file:", model_weights_file)
        elf_pthread = ["../build/elf_pthread", "-c"]
        elf_pthread += ["-i", model_weights_file]
        weights_dtype = model_weights_file.split('/')[-1][:3]
        elf_pthread += ["-p", weights_dtype]
        output_folder = os.path.dirname(os.path.dirname(model_weights_file))+"/exponential_dedup/"+weights_dtype+"/"
        folder_making_fun(output_folder)
        elf_pthread += ["-o", output_folder]
        weights_num = model_weights_file.split('/')[-1].split('_')[-1].split('.')[0]
        elf_pthread += ["-n", weights_num]
        print("elf_pthread:", elf_pthread)
        #sys.exit()
        try: 
            result = subprocess.run(elf_pthread, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            print("ELF:", result.stdout, end="")
        except subprocess.CalledProcessError as e:
            print("Error occurred:", e.stderr)
        org_file_size = os.path.getsize(model_weights_file)
        elf_cmp_folder = output_folder
        total_file_size = get_folder_size(elf_cmp_folder)
        if total_file_size == 0:
            print("No compressed files in", elf_cmp_folder)
        elif org_file_size <= total_file_size:
            print("No Storage Saving from ELF.")
        else:
            print("ELF: org file size:", rounding(org_file_size/MB), "MB. ", "cmp file size:", rounding(total_file_size/MB), "MB. ",  "Compression Ratio:", rounding(org_file_size/total_file_size))

