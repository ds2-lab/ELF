import os
import numpy as np
import pickle
from tqdm import tqdm
import sys
from Utils.utils import *
from Utils.config import *
import subprocess

def distance_reference(model_weights_path_list):
    for model_weights_file in model_weights_path_list:
        # C++ version de
        de_pthread = ["../build/de_pthread", "-c"]
        de_pthread += ["-i", model_weights_file]
        weights_dtype = model_weights_file.split('/')[-1][:3]
        de_pthread += ["-p", weights_dtype]
        output_folder = os.path.dirname(os.path.dirname(model_weights_file))+"/distance_encoding/"+weights_dtype+"/"
        distinct_list_file = output_folder + "distinct_file.bin"
        distance_list_file = output_folder + "distance_file.bin"
        distance_str_file  = output_folder + "distance_str_left_file.bin"
        if not os.path.exists(output_folder):
            folder_making_fun(output_folder)
        elif os.path.exists(distinct_list_file) and os.path.exists(distance_list_file) and os.path.exists(distance_str_file):
            print("DE Already Applied.")
            org_file_size = os.path.getsize(model_weights_file)
            total_file_size = get_folder_size(output_folder)
            if org_file_size <= total_file_size:
                print("No Storage Saving from DE.")
            else:
                print("DE:  org file size:", rounding(org_file_size/MB), "MB. ", "cmp file size:", rounding(total_file_size/MB), "MB. ",  "Compression Ratio:", rounding(org_file_size/total_file_size))
            continue
        de_pthread += ["-o", output_folder]
        weights_num = model_weights_file.split('/')[-1].split('_')[-1].split('.')[0]
        de_pthread += ["-n", weights_num]
        print("de_pthread:", de_pthread)
        #sys.exit()
        # Start the subprocess and open pipes to stdout and stderr
        with subprocess.Popen(de_pthread, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True) as proc:
            # Read output line by line as it is produced
            for line in proc.stdout:
                print(line, end='')
            # Check if there were any errors
            errors = proc.stderr.read()
            if errors:
                print("Errors:", errors)
            # Wait for the subprocess to finish and get the exit code
            proc.wait()
            #print("Subprocess finished with exit code", proc.returncode)
        org_file_size = os.path.getsize(model_weights_file)
        total_file_size = get_folder_size(output_folder)
        if total_file_size == 0:
            print("No compressed files in", output_folder)
        elif org_file_size <= total_file_size:
            print("No Storage Saving from DE.")
        else:
            print("DE:  org file size:", rounding(org_file_size/MB), "MB. ", "cmp file size:", rounding(total_file_size/MB), "MB. ",  "Compression Ratio:", rounding(org_file_size/total_file_size))
        return


