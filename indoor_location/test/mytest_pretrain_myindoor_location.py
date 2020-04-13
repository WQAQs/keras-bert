import os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras_bert.backend import keras
from keras_bert.backend import backend as K
from keras_bert import (get_model, compile_model, get_base_dict, gen_batch_inputs, get_token_embedding,
                        get_custom_objects, set_custom_objects,load_trained_model_from_checkpoint,build_model_from_config)
from indoor_location.utils import get_sentence_pairs
# from keras_bert.optimizers import AdamWarmup
# from sklearn.preprocessing import LabelBinarizer
# from keras.applications.vgg16 import VGG16
# from keras.callbacks import ModelCheckpoint, TensorBoard
# from keras.optimizers import SGD
# from keras.datasets import cifar10
# from tensorflow.python.keras.callbacks import ModelCheckpoint

valid_ibeacon_num = 26 #有效的ap数量
# seqence_len = valid_ibeacon_num*2+3   # 因为tokens = [TOKEN_CLS] + first + [TOKEN_SEP] + second + [TOKEN_SEP]
seqence_len = valid_ibeacon_num + 1  # 因为tokens = [TOKEN_CLS] + first
pretrain_datafile_name = "..\\data\\sampleset_data\\train_dataset1.csv"
pretrain_valid_datafile_name = "..\\data\\sampleset_data\\valid_dataset1.csv"

# train_datafile_name = "..\\data\\sampleset_data\\trainset_day20-1-8_points20_average_interval_500ms.csv"


token_id_from_numerical_order_file_path = ".\\logs\\token_id_from_numerical_order.csv"
token_id_from_dataset_order_file_path = ".\\logs\\token_id_from_dataset__order1.csv"

# MODEL_DIR = ".\\logs\\"
experiment_index = "4"
trained_model_index = "2"
pretrained_model_index = "4"
pretrained_model_path = ".\\logs\\pretrained_bert" + pretrained_model_index + ".h5"
all_mask_real_tokens_file_path = ".\\logs\\all_mask_real_tokens" + experiment_index + ".csv"
all_mask_predict_tokens_file_path = ".\\logs\\all_mask_predict_tokens" + experiment_index + ".csv"
all_match_res_file_path = ".\\logs\\all_match_res" + experiment_index + ".csv"
all_predicts_mlm_tokens_file_path = ".\\logs\\all_predicts_mlm_tokens" + experiment_index + ".csv"
all_real_mlm_tokens_file_path = ".\\logs\\all_real_mlm_tokens" + experiment_index + ".csv"
# pretrained_path = ".\\logs"
# config_path = os.path.join(pretrained_path, 'mybert_config.json')
# checkpoint_path = os.path.join(pretrained_path, 'mybert_model.ckpt')
# checkpoint_dir = os.path.dirname(checkpoint_path)

flag_retrain = True
EPOCHS = 1000
LR = 1e-3
decay_steps = 30000
warmup_steps = 10000
weight_decay = 1e-3

TOKEN_PAD = ''  # Token for padding
TOKEN_UNK = '[UNK]'  # Token for unknown words
TOKEN_CLS = '[CLS]'  # Token for classification
TOKEN_SEP = '[SEP]'  # Token for separation
TOKEN_MASK = '[MASK]'  # Token for masking

def get_base_id2token_dict():
    return {
        0: TOKEN_PAD,
        1: TOKEN_UNK,
        2: TOKEN_CLS,
        3: TOKEN_SEP,
        4: TOKEN_MASK,
    }
def bert_indoorlocation_pretrain():
    # 准备训练集数据和验证集数据
    train_sentence_pairs = get_sentence_pairs(pretrain_datafile_name)
    valid_sentence_pairs = get_sentence_pairs(pretrain_valid_datafile_name)
    def get_data(sentence_pairs):
        if not os.path.exists(token_id_from_numerical_order_file_path):
            dataset_token_dict = get_base_dict()  # dataset_token_dict：key为token，value为token对应的id
            id_dict = get_base_id2token_dict()  # id_dict：key为id，value为id对应的token
            df_data = [[TOKEN_PAD, 0],
                       [TOKEN_UNK, 1],
                       [TOKEN_CLS, 2],
                       [TOKEN_SEP, 3],
                       [TOKEN_MASK, 4]
            ]
            base_tokens = [TOKEN_PAD, TOKEN_UNK, TOKEN_CLS, TOKEN_SEP, TOKEN_MASK]
            other_tokens = []

            for pairs in sentence_pairs:
                for token in pairs[0] + pairs[1]:
                    if token not in dataset_token_dict:
                        n = len(dataset_token_dict)
                        dataset_token_dict[token] = n
                        other_tokens.append(token)
                        id_dict[n] = token
                        df_data.append([token, n])
            other_tokens = sorted(other_tokens, key=lambda x: int(x))
            all_tokens = base_tokens + other_tokens
            all_ids = [i for i in range(len(all_tokens))]
            pd.DataFrame(data={"token": all_tokens, "id": all_ids}).to_csv(token_id_from_numerical_order_file_path)
            pd.DataFrame(df_data, columns=["token", "id"]).to_csv(token_id_from_dataset_order_file_path)
        df_data = pd.read_csv(token_id_from_numerical_order_file_path)
        tokens = df_data["token"]
        ids = df_data["id"]
        token_dict = dict(zip(tokens, ids))
        id_dict = dict(zip(ids, tokens))

        token_list = list(token_dict.keys())

        x, y = gen_batch_inputs(
            sentence_pairs,
            token_dict,
            token_list,
            seq_len=seqence_len,
            mask_rate=0.3,
            swap_sentence_rate=0.5,
        )
        return x, y,token_dict,id_dict
    x_train, y_train, token_dict, id_dict = get_data(train_sentence_pairs)
    x_valid, y_valid, token_dict, id_dict = get_data(valid_sentence_pairs)
    config = tf.ConfigProto(allow_soft_placement=True)
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9)
    config.gpu_options.allow_growth = True
    with tf.Session(config=config) as session:
        # saver = tf.train.Saver()  # 保存模型参数的saver
        print("compiling model .....")
        model = get_model(
            token_num=len(token_dict),
            head_num=2,
            transformer_num=2,
            embed_dim=12,
            feed_forward_dim=100,
            seq_len=seqence_len,
            pos_num=seqence_len,
            dropout_rate=0.05,
            attention_activation='gelu',
        )
        if flag_retrain:
            model.load_weights(pretrained_model_path)
        # 初始化模型和参数
        compile_model(
            model,
            learning_rate=LR,
            decay_steps=decay_steps,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
        )
        model.summary()
            # data = model.to_json()
            #
            # # 保存模型到json文件
            # with open('./bert_model_config.json', 'w') as file:
            #     file.write(data)

            # Create a callback that saves the model's weights
            # cp_callback = keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
            #                                               save_weights_only=True,
            #                                               verbose=1)
            # cp_callback = ModelCheckpoint(filepath=checkpoint_path,
            #                                               save_weights_only=True,
            #                                               verbose=1)
            #
            # early_stop = keras.callbacks.EarlyStopping(monitor='loss', patience=30)
            # callback_lists = [early_stop]

            # # 创建一个权重文件保存文件夹logs
            # log_dir = "logs/"
            # # 记录所有训练过程，每隔一定步数记录最大值
            # tensorboard = TensorBoard(log_dir=log_dir)
            # checkpoint = ModelCheckpoint(log_dir + "bert_model.h5",##!!!这里只能是h5格式
            #                              monitor="val_loss",
            #                              mode='min',
            #                              save_weights_only=True,
            #                              save_best_only=True,
            #                              verbose=1,
            #                              period=1)
            #
            # callback_lists = [tensorboard, checkpoint]

        print("training network...")
        # Train the model with the new callback
        from keras.callbacks import EarlyStopping
        early_stopping = EarlyStopping(monitor="val_loss", patience=5)
        H = model.fit(x_train, y_train, validation_data=(x_valid, y_valid),
                      batch_size=128, epochs=EPOCHS, callbacks=[early_stopping])

        # saver = tf.train.Saver()
        # saver.save(session, checkpoint_path)
        # current_path = os.path.dirname(os.path.abspath(__file__))
        # model_path = os.path.join(current_path, 'test_indoorlocation_pretrain_bert_model1.h5')

        # if os.path.exists(model_path):
        # if flag_retrain:
        #     steps_per_epoch = 1000
        #     model = keras.models.load_model(
        #         pretrained_model_path,
        #         custom_objects=get_custom_objects(),
        #     )
        # else:
        #     steps_per_epoch = 100
        #     model = get_model(
        #         token_num=len(token_dict),
        #         head_num=2,
        #         transformer_num=2,
        #         embed_dim=12,
        #         feed_forward_dim=100,
        #         seq_len=seqence_len,
        #         pos_num=seqence_len,
        #         dropout_rate=0.05,
        #         attention_activation='gelu',
        #     )
        #     compile_model(
        #         model,
        #         learning_rate=1e-3,
        #         decay_steps=30000,
        #         warmup_steps=10000,
        #         weight_decay=1e-3,
        #     )

        # def _generator():
        #     while True:
        #         yield gen_batch_inputs(
        #             sentence_pairs,
        #             token_dict,
        #             token_list,
        #             seq_len=seqence_len,
        #             mask_rate=0.3,
        #             swap_sentence_rate=1.0,
        #         )
        # model.fit_generator(
        #     generator=_generator(),
        #     steps_per_epoch=steps_per_epoch,
        #     epochs=1,
        #     validation_data=_generator(),
        #     validation_steps=steps_per_epoch // 10,
        # )
        model.save(pretrained_model_path)

        ## 分析预训练出来的模型对所有样本预测的token结果

        def id_list2token_list(id_list, id_dict):
            token_list = []
            for id in id_list:
                token = id_dict.get(id, TOKEN_UNK)
                token_list.append(token)
            return token_list

        def all_id_lists2all_token_lists(sentences_id_data):
            sentences_token_data = []
            for item in sentences_id_data:
                token_list = id_list2token_list(item, id_dict)
                sentences_token_data.append(token_list)
            return sentences_token_data

        def evaluate_pretrain_model(predicts_mlm_ids, real_mlm_ids):
            predicts_mlm_tokens = all_id_lists2all_token_lists(predicts_mlm_ids)
            real_mlm_tokens = all_id_lists2all_token_lists(real_mlm_ids)

            samples_num, seq_len = x_train[0].shape
            all_match_res = []
            all_mask_real_tokens = []
            all_mask_predict_tokens = []
            for i in range(samples_num):
                mask_real_tokens = []
                mask_predict_tokens = []
                match, total = 0, 0
                for j in range(seq_len):
                    if x_train[-1][i][j]:
                        total += 1
                        mask_real_tokens.append(real_mlm_tokens[i][j])
                        mask_predict_tokens.append(predicts_mlm_tokens[i][j])
                        if predicts_mlm_tokens[i][j] == real_mlm_tokens[i][j]:
                            match += 1
                all_mask_real_tokens.append(mask_real_tokens)
                all_mask_predict_tokens.append(mask_predict_tokens)
                match_ratio = match/total
                temp = [total, match, match_ratio]
                all_match_res.append(temp)
            pd.DataFrame(predicts_mlm_tokens).to_csv(all_predicts_mlm_tokens_file_path, encoding='utf-8')
            pd.DataFrame(real_mlm_tokens).to_csv(all_real_mlm_tokens_file_path, encoding='utf-8')
            pd.DataFrame(all_mask_real_tokens).to_csv(all_mask_real_tokens_file_path, encoding='utf-8')
            pd.DataFrame(all_mask_predict_tokens).to_csv(all_mask_predict_tokens_file_path, encoding='utf-8')
            pd.DataFrame(all_match_res, columns=["total_mask_num", "match_num", "match_ratio"])\
                .to_csv(all_match_res_file_path, encoding='utf-8')

        predicts = model.predict(x_train)
        predicts_mlm_ids = np.argmax(predicts[0], axis=-1)
        real_mlm_ids = list(map(lambda x: np.squeeze(x, axis=-1), y_train[0]))
        evaluate_pretrain_model(predicts_mlm_ids, real_mlm_ids)



bert_indoorlocation_pretrain()