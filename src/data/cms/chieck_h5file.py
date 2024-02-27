import h5py
import numpy as np

file = h5py.File('/outdir1/HHH_QurdJetTrigger_jetpair_PNet_training_random.h5', 'r')

dataset = file['CLASSIFICATIONS/EVENT/signal']  # 将 'your_dataset' 替换为您的数据集名称
print(len(dataset))
'''
result_list = [0, 0, 0, 0]

for item in dataset:
    result_list[item] += 1
EventNumber = len(dataset)
print(result_list)
print(list(map(lambda x: x/EventNumber, result_list)))
# 关闭 H5 文件
file.close()
'''