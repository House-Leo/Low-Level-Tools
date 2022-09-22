import os
import shutil

source_path = './experiments/real_0725/val_images'
keyword = input("Input keywork:")
key = '_' + keyword
# print(key)
target_path = os.path.join('./experiments/real_0725/',keyword)
# target_path = '/home/yzj6850/houselee/ablation_results/mse/'
# print(target_path)
if not os.path.isdir(target_path):
    os.makedirs(target_path)

for dirs in os.listdir(source_path):
    # print(dirs)
    image_file_path = os.path.join(source_path,dirs)
    for files in os.listdir(image_file_path):
        # print(files)
        if key in files:
            # print(files)
            shutil.copy(os.path.join(os.path.join(source_path, dirs),files), target_path)

for files in os.listdir(target_path):
    old_name = files
    # new_name = files[0:4] + '.png' #div2k
    new_name = files[0:12] + '.png' #realsr
    print(old_name,'->',new_name)
    os.rename(os.path.join(target_path,old_name),os.path.join(target_path,new_name))
