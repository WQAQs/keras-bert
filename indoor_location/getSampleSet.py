import indoor_location.dataProcess as dp
import indoor_location.globalConfig as globalConfig #不能删除,因为在 experiment_config.py 中更改了当前路径
import pandas as pd


pointTxtRootDir = globalConfig.root_txt_dir  # 原始数据存在的文件夹
pointCsvRootDir = globalConfig.root_csv_dir  # 转换的csv文件目标文件夹，也作为合并csv的源路径
ibeaconFilePath = globalConfig.ibeaconFilePath  # ibeacon统计文件目标路径及文件名
allPointCsvRootDir = globalConfig.generate_sampleset_all_labeled_csv_dir  # 总数据数据文件夹

# dp.loadAllTxt2Csv(pointTxtRootDir, pointCsvRootDir)  # 将原始数据加载为Csv文件

# 修改后若无单独保存csv数据需求，直接将生成的csv文件保存到总数据目录中，无需额外合并
# dp.mergeAllCsv(allPointCsvRootDir, pointCsvRootDir)  # 将生成的Csv文件加入总数据

# dp.updateAllIbeaconDataSet(allPointCsvRootDir, ibeaconFilePath)  # 更新ibeaconDataSet

# dp.create_sample_dataset(allPointCsvRootDir)  # 创建样本集

file1 = "./data/sampleset_data/new_3days1/train_dataset1.csv"
file2 = "./data/sampleset_data/new_3days1/valid_dataset1.csv"
merged_file = "./data/sampleset_data/new_3days1/pretrain_dataset.csv"

dp.merge_dataset(file1, file2, merged_file)



# sample_dataset_file = ".\\data\\sampleset_data\\valid_dataset1.csv"
# valid_dataset_file = ".\\data\\sampleset_data\\valid_dataset1.csv"
# test_dataset_file = ".\\data\\sampleset_data\\test_dataset1.csv"

# #划分出测试集
# def divide_sample_dataset(sample_dataset):
#     test_dataset = sample_dataset.sample(frac=0.5, random_state=0)
#     valid_dataset = sample_dataset.drop(test_dataset.index)
#     test_dataset.to_csv(test_dataset_file, index=False, encoding='utf-8')
#     valid_dataset.to_csv(valid_dataset_file, index=False, encoding='utf-8')
#
# dataset = pd.read_csv(sample_dataset_file)
# divide_sample_dataset(dataset)

