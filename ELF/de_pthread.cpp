#include <iostream>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <thread>
#include <chrono>
#include <string>
#include <cstring>
#include <vector>
#include <bitset>
#include <cmath>
#include <sys/types.h>
#include <fstream>
#include <sstream>
#include <cassert>
#include <filesystem>
#include <cstdlib>
#include <eigen3/Eigen/Core>
#include <unordered_map>

using namespace std;

const int KB = 1024;
const int MB = KB * 1024;
const int GB = MB * 1024;

struct ModelInfo {
    std::string model_path;
    int para_num;
    // 0:float16, 1:float32, 2:float64
    int precision;
    // 0:compression, 1:decompression
    int flg = 0;
    // output folder of the output file(s)
    std::string output_folder;
};

void distance_update_bits(std::vector<uint64_t>& distance_list, std::string& bit_str) {
    while (bit_str.length() >= 64) {
        std::bitset<64> bits(bit_str.substr(0, 64));
        distance_list.push_back(bits.to_ullong());
        bit_str = bit_str.substr(64);
    }
}

void write_to_file(const std::string& filename, const char* data, std::size_t size) {
    std::ofstream file(filename, std::ios::binary);
    if (!file) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    file.write(data, size);
}

int de_func_f16(Eigen::half* data, size_t para_num, string distinct_list_file, string distance_list_file, string distance_str_file) {
    std::vector<Eigen::half>    distinct_list;
    std::vector<uint64_t> distance_list;
    std::unordered_map<Eigen::half, size_t> para_position_dict;
    std::string distance_str;
    size_t percent = para_num / 10;
    distinct_list.reserve((unsigned int)(para_num/2));
    distance_list.reserve((unsigned int)(para_num/2));
    for (size_t position = 0; position < para_num; position++) {

        if (position % percent == 0) {
            cout << "Progress of " << (position*10/percent) << "%." << endl;
        }
        Eigen::half para = data[position];
        if (para_position_dict.find(para) == para_position_dict.end()) {
            para_position_dict[para] = position;
            distinct_list.push_back(para);
            distance_str += '0';
        } else {
            size_t distance = position - para_position_dict[para];
            para_position_dict[para] = position;

            std::string distance_bin = std::bitset<64>(distance).to_string();
            distance_bin.erase(0, distance_bin.find('1'));
            size_t distance_bin_len = distance_bin.length();

            std::string distance_bin_len_bin = std::bitset<64>(distance_bin_len).to_string();
            distance_bin_len_bin.erase(0, distance_bin_len_bin.find('1'));

            size_t distance_bin_len_bin_len = distance_bin_len_bin.length();
            std::string para_5bit = std::string(5 - distance_bin_len_bin_len, '0') + distance_bin_len_bin;

            distance_str += ('1' + para_5bit + distance_bin);
        }
        distance_update_bits(distance_list, distance_str);
    }
    cout << "Progress of 100%. Writing compressed files..." << endl;
    write_to_file(distinct_list_file, reinterpret_cast<const char*>(distinct_list.data()), distinct_list.size() * sizeof(Eigen::half));
    write_to_file(distance_list_file, reinterpret_cast<const char*>(distance_list.data()), distance_list.size() * sizeof(uint64_t));
    write_to_file(distance_str_file, distance_str.c_str(), distance_str.size());
    return 0;
}

int de_func_f32(float* data, size_t para_num, string distinct_list_file, string distance_list_file, string distance_str_file) {
    std::vector<float>    distinct_list;
    std::vector<uint64_t> distance_list;
    std::unordered_map<float, size_t> para_position_dict;
    std::string distance_str;
    size_t percent = para_num / 10;
    distinct_list.reserve((unsigned int)(para_num/2));
    distance_list.reserve((unsigned int)(para_num/2));
    for (size_t position = 0; position < para_num; position++) {
	
	if (position % percent == 0) {
	    cout << "Progress of " << (position*10/percent) << "%." << endl;
	}
	float para = data[position];
        if (para_position_dict.find(para) == para_position_dict.end()) {
            para_position_dict[para] = position;
	    distinct_list.push_back(para);
            distance_str += '0';
        } else {
            size_t distance = position - para_position_dict[para];
            para_position_dict[para] = position;

            std::string distance_bin = std::bitset<64>(distance).to_string();
            distance_bin.erase(0, distance_bin.find('1'));
            size_t distance_bin_len = distance_bin.length();

            std::string distance_bin_len_bin = std::bitset<64>(distance_bin_len).to_string();
            distance_bin_len_bin.erase(0, distance_bin_len_bin.find('1'));

            size_t distance_bin_len_bin_len = distance_bin_len_bin.length();
            std::string para_5bit = std::string(5 - distance_bin_len_bin_len, '0') + distance_bin_len_bin;

            distance_str += ('1' + para_5bit + distance_bin);
        }
        distance_update_bits(distance_list, distance_str);
    }
    cout << "Progress of 100%. Writing compressed files..." << endl;
    write_to_file(distinct_list_file, reinterpret_cast<const char*>(distinct_list.data()), distinct_list.size() * sizeof(float));
    write_to_file(distance_list_file, reinterpret_cast<const char*>(distance_list.data()), distance_list.size() * sizeof(uint64_t));
    write_to_file(distance_str_file, distance_str.c_str(), distance_str.size());
    return 0;
}

int de_func_f64(double* data, size_t para_num, string distinct_list_file, string distance_list_file, string distance_str_file) {
    std::vector<double>   distinct_list;
    std::vector<uint64_t> distance_list;
    std::unordered_map<double, size_t> para_position_dict;
    std::string distance_str;
    size_t percent = para_num / 10;
    distinct_list.reserve((unsigned int)(para_num/2));
    distance_list.reserve((unsigned int)(para_num/2));
    for (size_t position = 0; position < para_num; position++) {
        if (position % percent == 0) {
            cout << "Progress of " << (position*10/percent) << "%." << endl;
        }
        double para = data[position];
        if (para_position_dict.find(para) == para_position_dict.end()) {
            para_position_dict[para] = position;
            distinct_list.push_back(para);
            distance_str += '0';
        } else {
            size_t distance = position - para_position_dict[para];
            para_position_dict[para] = position;

            std::string distance_bin = std::bitset<64>(distance).to_string();
            distance_bin.erase(0, distance_bin.find('1'));
            size_t distance_bin_len = distance_bin.length();

            std::string distance_bin_len_bin = std::bitset<64>(distance_bin_len).to_string();
            distance_bin_len_bin.erase(0, distance_bin_len_bin.find('1'));

            size_t distance_bin_len_bin_len = distance_bin_len_bin.length();
            std::string para_5bit = std::string(5 - distance_bin_len_bin_len, '0') + distance_bin_len_bin;

            distance_str += ('1' + para_5bit + distance_bin);
        }
        distance_update_bits(distance_list, distance_str);
    }
    cout << "Progress of 100%. Writing compressed files..." << endl;
    write_to_file(distinct_list_file, reinterpret_cast<const char*>(distinct_list.data()), distinct_list.size() * sizeof(double));
    write_to_file(distance_list_file, reinterpret_cast<const char*>(distance_list.data()), distance_list.size() * sizeof(uint64_t));
    write_to_file(distance_str_file, distance_str.c_str(), distance_str.size());
    return 0;
}

int de_cmp(ModelInfo& info) {
    auto start_read = std::chrono::high_resolution_clock::now();
    // 0:float16, 1:float32, 2:float64
    int precision = info.precision;
    //cout << "precision:" << precision << endl;

    int fd = open(info.model_path.c_str(), O_RDWR);
    if (fd == -1) {
        perror("open");
        return 1;
    }

    struct stat fileStat;
    if (fstat(fd, &fileStat) < 0) {
        perror("fstat");
        close(fd);
        return 1;
    }

    off_t expectedSize = 0;
    if (precision == 0) {
        //expectedSize = static_cast<off_t>(static_cast<size_t>(info.para_num) * 2);
        expectedSize = static_cast<off_t>(info.para_num * sizeof(Eigen::half));
    } else if (precision == 1) {
        expectedSize = static_cast<off_t>(info.para_num * sizeof(float));
    } else if (precision == 2) {
        expectedSize = static_cast<off_t>(info.para_num * sizeof(double));
    } else {
        cout << "No precision specified, or precision unknown." << endl;
        exit(1);
    }

    //std::cout << "fileStat.st_size:" << fileStat.st_size << ", expectedSize: " << expectedSize << std::endl;
    if (fileStat.st_size != expectedSize) {
        std::cerr << "File size does not match expected size." << std::endl;
        close(fd);
        return 1;
    }

    void* mapped_data = mmap(NULL, fileStat.st_size, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    if (mapped_data == MAP_FAILED) {
        perror("mmap");
        close(fd);
        return 1;
    }

    std::string file_folder = info.output_folder;

    string distinct_list_file = file_folder + "distinct_file.bin";
    string distance_list_file = file_folder + "distance_file.bin";
    string distance_str_file  = file_folder + "distance_str_left_file.bin";
    
    if (precision == 0) {
        Eigen::half* data = static_cast<Eigen::half*>(mapped_data);
        de_func_f16(data, size_t(info.para_num), distinct_list_file, distance_list_file, distance_str_file);
    } else if (precision == 1) {
        float* data = static_cast<float*>(mapped_data);
	de_func_f32(data, size_t(info.para_num), distinct_list_file, distance_list_file, distance_str_file);
    } else {
        double* data = static_cast<double*>(mapped_data);
        de_func_f64(data, size_t(info.para_num), distinct_list_file, distance_list_file, distance_str_file);
    }

    // Unmap and close the file
    if (munmap(mapped_data, fileStat.st_size) == -1) {
        perror("munmap");
    }
    close(fd);
    
    auto end_read = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed_read = end_read - start_read;
    cout << "compression time: " << elapsed_read.count() << " s." << endl;
    return 0;

}

int de_decmp(ModelInfo& info) {

    return 0;
}

ModelInfo parse_args(int argc, char *argv[]){
    ModelInfo model_info;
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "-i") {
            if (i + 1 < argc) {
                model_info.model_path = argv[++i];
            } else {
                std::cerr << "No input path specified after -i" << std::endl;
                exit(1);
            }
        } else if (arg == "-c") {
            model_info.flg = 1;
        } else if (arg == "-d") {
            model_info.flg = -1;
        } else if (arg == "-o"){
            if (i + 1 < argc) {
                model_info.output_folder = argv[++i];
            } else {
                std::cerr << "No output folder specified after -o" << std::endl;
                exit(1);
            }
        } else if (arg == "-p") {
            if (i + 1 < argc) {
                std::string precision_str = argv[++i];
                if (precision_str == "f16")
                    model_info.precision = 0;
                else if (precision_str == "f32")
                    model_info.precision = 1;
                else if (precision_str == "f64")
                    model_info.precision = 2;
                else {
                    std::cerr << "Invalid precision. Choose from f16, f32, f64" << std::endl;
                    exit(1);
                }
            } else {
                std::cerr << "No precision specified after -p" << std::endl;
                exit(1);
            }
        } else if (arg == "-n") {
            if (i + 1 < argc) {
                std::istringstream iss(argv[++i]);
                if (!(iss >> model_info.para_num)) {
                    std::cerr << "Invalid size. Size must be an integer." << std::endl;
                    exit(1);
                }
            } else {
                std::cerr << "No size specified after -size" << std::endl;
                exit(1);
            }
        } else {
            std::cerr << "Unknown argument: " << arg << std::endl;
            exit(1);
        }
    }
    return model_info;
}

int main(int argc, char *argv[]) {
    ModelInfo model_info = parse_args(argc, argv);
    if (model_info.flg == 1) {
        de_cmp(model_info);
    } else if (model_info.flg == -1) {
        de_decmp(model_info);
    } else {
        std::cerr << "please spicify the compression / decompression mode. -c for compression, -d for decompression. " <<  std::endl;
        exit(1);
    }
    return 0;
}


