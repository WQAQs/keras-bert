# LAB （文件夹：7dasy_results）
## 数据
    7days数据来源：
        2019_10_21
        2019_11_4
        2019_11_21
        2019_12_4
        2019_12_16
        2019_12_26
        2019_1_8
    预训练：7days数据，Train on 66348 samples, validate on 8294 samples
    20个参考点：1——20

# 实验1（文件夹：7dasy_results\\lab1）

## 超参数
    valid_ibeacon_num = 26 #有效的ap数量
    seq_len = 50
    head_num = 4                            !!!!!!!!!!!!!!!!!!!!!!! 1_1
    transformer_num = 2
    embed_dim = 32
    feed_forward_dim = 100
    dropout_rate = 0.05

## 模型参数量
    Total params: 74,597

## 实验 1.1（文件夹：7dasy_results\\lab1\\1）
    【pretrain】
     - val_loss: 4.3533

    【fine_tune】
    - val_mean_absolute_error: 0.5008
    - val_mean_squared_error: 0.7165

    【fine tune model 测试结果】
    文件夹                                      测试集                                      误差CDF结果
    fine_tune_test1          预训练使用的7days数据集中分割出的testset                         83% (1m以下), 95% (2m以下)
    fine_tune_test2          全新的一天的testset（2019_12_17, 没有参与预训练和训练）           53% (1m以下), 75% (2m以下)

## 实验 1.2（文件夹：7dasy_results\\lab1\\2）
    【pretrain】 (epochs = 1000, earlystop, patient = 5, LR = 0.01)
    - val_loss: 4.1810   
    
    【fine_tune】 (epochs = 1000, earlystop, patient = 5, LR = 0.01)
    - val_mean_absolute_error: 0.2968 
    - val_mean_squared_error: 0.5170

    【fine tune model 测试结果】
        文件夹                                      测试集                                  CDF结果
    fine_tune_test01          预训练使用的7days数据集中分割出的validset                        93% (1m以下), 97%（2m以下）
    fine_tune_test02          预训练使用的7days数据集中分割出的testset                         93% (1m以下), 97%（2m以下）
    fine_tune_test3           全新的一天的set（2019_12_17sampleset, 没有参与预训练和训练）      70% (1m以下), 82%（2m以下）

    为了测试新的一天的部分数据参与预训练是否能使新的一天的数据测试结果变好？



# 实验2（文件夹：7dasy_results\\lab2）

## 超参数
    valid_ibeacon_num = 26 #有效的ap数量
    seq_len = 50
    head_num = 4
    transformer_num = 4                      !!!!!!!!!!!!!!!!!!!!!!! 1_2
    embed_dim = 32
    feed_forward_dim = 100
    dropout_rate = 0.05

## 模型参数量
    fine tune Total params: 93,746

## 实验2.1（文件夹：7dasy_results\\lab2\\1）
    

    【pretrain】 (epochs = 1000, earlystop, patient = 5, LR = 0.01)
    - val_loss: 4.1381

    【fine tune】 
    1. (epochs = 1000, earlystop, patient = 5, ！！！LR = 0.01！！！)
    - val_mean_absolute_error: 0.3654 
    - val_mean_squared_error: 0.7455

    2.【retrain】  (epochs = 1000, earlystop, patient = 5, ！！！LR = 0.005！！！)
    val_mean_absolute_error: 0.2130 
    - val_mean_squared_error: 0.4992

    【fine tune model 测试结果】
    1.
    fine_tune model: trained1_bert.h5

     文件夹                                      测试集                                         CDF结果
    fine_tune1_test01          预训练使用的7days数据集中分割出的validset                        。。。
    fine_tune1_test02          预训练使用的7days数据集中分割出的testset                         90% (1m以下), 95%（2m以下）
    fine_tune1_test3          全新的一天的testset（2019_12_17, 没有参与预训练和训练）            67% (1m以下), 79%（2m以下）

    2.
    fine_tune model: trained2_bert.h5

     文件夹                                      测试集                                         CDF结果
    fine_tune2_test01          预训练使用的7days数据集中分割出的validset                        。。。
    fine_tune2_test02          预训练使用的7days数据集中分割出的testset                         95% (1m以下), 97%（2m以下）
    fine_tune2_test3          全新的一天的testset（2019_12_17, 没有参与预训练和训练）            64% (1m以下), 78%（2m以下）




# 实验3 （文件夹：7dasy_results\\lab3）

**finetune测试结果**

验证预训练是否有效
            说明                                   文件目录
只参与预训练的【新一天】数据，最终定位结果        \\lab3\\inside
不参与预训练的【新一天】数据，最终定位结果        \\lab3\\outside\\test2
【同时间段】数据，标注率 0.1                    \\lab3\\outside\\test01

            

valid_ibeacon_num = 26 #有效的ap数量
seq_len = 40
head_num = 2                 !!!!!!!!!!!!!!!!!!!!!!! 1
transformer_num = 4          !!!!!!!!!!!!!!!!!!!!!!! 2
embed_dim = 32
feed_forward_dim = 100
dropout_rate = 0.05

【pretrain】
                            pretrain_res  
         dir         train_loss  valid loss：     transformer_num      head_num  
                                        5.17                2               2
                                        5.1661              4               2
                                        5.1747              8               2
                                        5.1642              2               4
                                        5.1449              2               8
                                        5.1808
    \\lab3\\inside                      4.0147 
    \\lab3\\outside                     3.98                                                 
【fine tune model】pretrain_model1.h5  
pretrain_model1.h5
- val_mean_absolute_error: 0.5776 - val_mean_squared_error: 1.1006
pretrain_model2.h5
- val_mean_absolute_error: 0.3968 - val_mean_squared_error: 0.8758


